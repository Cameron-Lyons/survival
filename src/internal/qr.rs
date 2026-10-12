use crate::internal::simd::{dot_product, sum_of_squares};

pub(crate) struct LinpackQr {
    householder: Vec<Vec<f64>>,
    rank: usize,
}

impl LinpackQr {
    pub(crate) fn new(x: Vec<Vec<f64>>, n: usize) -> Self {
        Self::decompose(x, n).0
    }

    fn decompose(mut x: Vec<Vec<f64>>, n: usize) -> (Self, Vec<Vec<f64>>, Vec<usize>) {
        const TOL: f64 = 1e-7;
        let p = x.len();
        let mut pivot: Vec<usize> = (0..p).collect();
        let mut qraux: Vec<f64> = x.iter().map(|column| norm(column)).collect();
        let mut original: Vec<f64> = qraux
            .iter()
            .map(|&value| if value == 0.0 { 1.0 } else { value })
            .collect();
        let mut k = p + 1;
        for l in 0..n.min(p) {
            let run = (l..k - 1)
                .take_while(|&j| qraux[j] < original[j] * TOL)
                .count();
            x[l..].rotate_left(run);
            pivot[l..].rotate_left(run);
            qraux[l..].rotate_left(run);
            original[l..].rotate_left(run);
            k -= run;
            if l + 1 == n {
                continue;
            }
            let (head, tail) = x.split_at_mut(l + 1);
            let xl = &mut head[l];
            let mut nrmxl = norm(&xl[l..]);
            if nrmxl == 0.0 {
                continue;
            }
            if xl[l] != 0.0 {
                nrmxl = nrmxl.copysign(xl[l]);
            }
            let scale = 1.0 / nrmxl;
            for value in &mut xl[l..] {
                *value *= scale;
            }
            xl[l] += 1.0;
            for (offset, xj) in tail.iter_mut().enumerate() {
                let j = l + 1 + offset;
                let t = -dot_product(&xl[l..], &xj[l..]) / xl[l];
                for (value, h) in xj[l..].iter_mut().zip(&xl[l..]) {
                    *value += t * h;
                }
                if qraux[j] != 0.0 {
                    let tt = (1.0 - (xj[l].abs() / qraux[j]).powi(2)).max(0.0);
                    if tt < 1e-6 {
                        qraux[j] = norm(&xj[l + 1..]);
                    } else {
                        qraux[j] *= tt.sqrt();
                    }
                }
            }
            qraux[l] = xl[l];
            xl[l] = -nrmxl;
        }
        let rank = (k - 1).min(n);
        let householder = (0..rank.min(n.saturating_sub(1)))
            .map(|j| {
                let mut vector = x[j][j..].to_vec();
                vector[0] = qraux[j];
                vector
            })
            .collect();
        (Self { householder, rank }, x, pivot)
    }

    pub(crate) fn qty(&self, y: &mut [f64]) {
        for (j, vector) in self.householder.iter().enumerate() {
            reflect(vector, &mut y[j..]);
        }
    }

    pub(crate) fn residual(&self, y: &[f64]) -> Vec<f64> {
        let mut rsd = y.to_vec();
        if self.rank == 0 {
            return rsd;
        }
        if self.householder.is_empty() {
            rsd[0] = 0.0;
            return rsd;
        }
        self.qty(&mut rsd);
        rsd[..self.rank].fill(0.0);
        for (j, vector) in self.householder.iter().enumerate().rev() {
            reflect(vector, &mut rsd[j..]);
        }
        rsd
    }
}

pub(crate) struct LinpackLeastSquares {
    qr: LinpackQr,
    upper: Vec<Vec<f64>>,
    pivot: Vec<usize>,
}

impl LinpackLeastSquares {
    pub(crate) fn rank(&self) -> usize {
        self.qr.rank
    }

    pub(crate) fn prediction_variance(&self, row: &[f64]) -> Option<f64> {
        if self.qr.rank != self.pivot.len() || row.len() != self.pivot.len() {
            return None;
        }
        let mut solved = vec![0.0; row.len()];
        for j in 0..row.len() {
            let previous = dot_product(&self.upper[j][..j], &solved[..j]);
            solved[j] = (row[self.pivot[j]] - previous) / self.upper[j][j];
        }
        Some(sum_of_squares(&solved))
    }

    pub(crate) fn new(x: Vec<Vec<f64>>, n: usize) -> Self {
        let (qr, factor, pivot) = LinpackQr::decompose(x, n);
        let upper = factor
            .iter()
            .take(qr.rank)
            .enumerate()
            .map(|(j, column)| column[..=j].to_vec())
            .collect();
        Self { qr, upper, pivot }
    }

    pub(crate) fn coefficients(&self, y: &[f64]) -> Vec<f64> {
        let mut qty = y.to_vec();
        self.qr.qty(&mut qty);
        let mut result = vec![f64::NAN; self.pivot.len()];
        for j in (0..self.qr.rank).rev() {
            let coefficient = qty[j] / self.upper[j][j];
            result[self.pivot[j]] = coefficient;
            for (i, value) in qty[..j].iter_mut().enumerate() {
                *value -= coefficient * self.upper[j][i];
            }
        }
        result
    }
}

fn reflect(vector: &[f64], y: &mut [f64]) {
    if vector[0] == 0.0 {
        return;
    }
    let t = -dot_product(vector, y) / vector[0];
    for (value, h) in y.iter_mut().zip(vector) {
        *value += t * h;
    }
}

fn norm(a: &[f64]) -> f64 {
    sum_of_squares(a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (value, want) in actual.iter().zip(expected) {
            assert!((value - want).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn least_squares_unpivots_coefficients_and_marks_aliases() {
        let fit = LinpackLeastSquares::new(
            vec![
                vec![0.0; 4],
                vec![1.0; 4],
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2.0, 4.0, 6.0, 8.0],
            ],
            4,
        );
        let coef = fit.coefficients(&[1.0, 3.0, 2.0, 5.0]);
        assert!(coef[0].is_nan() && coef[3].is_nan());
        assert_close(&coef[1..3], &[0.0, 1.1]);
        let coef = fit.coefficients(&[5.0, 7.0, 9.0, 11.0]);
        assert_close(&coef[1..3], &[3.0, 2.0]);
    }

    #[test]
    fn aliased_column_matches_r() {
        let qr = LinpackQr::new(
            vec![
                vec![1.0, 1.0, 1.0, 1.0],
                vec![1.0, 2.0, 3.0, 4.0],
                vec![2.0, 4.0, 6.0, 8.0],
            ],
            4,
        );
        assert_eq!(qr.rank, 2);
        let y = [1.0, 3.0, 2.0, 5.0];
        let mut qty = y.to_vec();
        qr.qty(&mut qty);
        assert_close(
            &qty,
            &[
                -5.5,
                -2.459_674_775_249_768_5,
                -1.639_344_662_916_631_5,
                -0.112_022_659_166_596_48,
            ],
        );
        assert_close(&qr.residual(&y), &[-0.1, 0.8, -1.3, 0.6]);
        let inside = qr.residual(&[3.0, 5.0, 7.0, 9.0]);
        assert!(inside.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn runs_of_negligible_columns_cycle_as_in_r() {
        let qr = LinpackQr::new(
            vec![
                vec![1.0; 4],
                vec![2.0; 4],
                vec![3.0; 4],
                vec![1.0, 2.0, 3.0, 4.0],
                vec![6.0, 7.0, 8.0, 9.0],
                vec![1.0, 0.0, 0.0, 2.0],
            ],
            4,
        );
        assert_eq!(qr.rank, 3);
        let y = [1.0, 3.0, 2.0, 5.0];
        let mut qty = y.to_vec();
        qr.qty(&mut qty);
        assert_close(
            &qty,
            &[
                -5.5,
                -2.459_674_775_249_768_5,
                0.725_318_520_735_365_4,
                -1.474_419_561_548_971_2,
            ],
        );
        assert_close(
            &qr.residual(&y),
            &[
                -0.434_782_608_695_652,
                1.086_956_521_739_130_2,
                -0.869_565_217_391_304_3,
                0.217_391_304_347_826_08,
            ],
        );
    }

    #[test]
    fn one_row_and_rank_zero() {
        let one_row = LinpackQr::new(vec![vec![2.0], vec![4.0]], 1);
        assert_eq!(one_row.rank, 1);
        assert_eq!(one_row.residual(&[3.0]), vec![0.0]);
        let zero = LinpackQr::new(vec![vec![0.0, 0.0]], 2);
        assert_eq!(zero.rank, 0);
        assert_eq!(zero.residual(&[1.0, 2.0]), vec![1.0, 2.0]);
    }
}
