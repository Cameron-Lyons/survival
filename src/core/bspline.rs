//! B-spline design matrices.
//!
//! `spline_design` ports `spline_basis` from R's `splines.c` (the C engine
//! of `splines::splineDesign` / `spline.des`, R 4.5): de Boor's recurrence
//! for the basis values and the derivative recursion `evaluate` for
//! `derivs > 0`.  The natural spline basis of [`crate::core::natural_spline`] and the
//! P-spline basis of [`crate::core::pspline`] are built on it.

use crate::error::{SurvivalError, SurvivalResult};
use ndarray::Array2;

/// `n x (n_knots - ord)` design matrix of the B-splines of order `ord`
/// (degree `ord - 1`) on the non-decreasing `knots`, evaluated at `x`;
/// `derivs[i % derivs.len()]` is the derivative order for `x[i]`.
///
/// Every `x` must lie in `[knots[ord - 1], knots[n_knots - ord]]`, the
/// range `splineDesign` accepts without `outer.ok`.
pub(crate) fn spline_design(
    knots: &[f64],
    x: &[f64],
    ord: usize,
    derivs: &[usize],
) -> SurvivalResult<Array2<f64>> {
    let nk = knots.len();
    if ord == 0 || nk < 2 * ord - 1 {
        return Err(SurvivalError::invalid_input(format!(
            "need at least 2 * ord - 1 = {} knots, got {nk}",
            2 * ord - 1
        )));
    }
    if knots.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(SurvivalError::invalid_input("knots must be non-decreasing"));
    }
    if derivs.is_empty() {
        return Err(SurvivalError::invalid_input("empty derivs"));
    }
    let (low, high) = (knots[ord - 1], knots[nk - ord]);
    let ncoef = nk - ord;
    let mut design = Array2::zeros((x.len(), ncoef));
    let mut spline = Spline::new(knots, ord);
    for (row, &value) in x.iter().enumerate() {
        if !(value >= low && value <= high) {
            return Err(SurvivalError::invalid_input(format!(
                "the 'x' data must be in the range {low} to {high}, got {value}"
            )));
        }
        spline.set_cursor(value);
        let offset = spline.curs - ord;
        let nder = derivs[row % derivs.len()];
        if nder > 0 {
            for ii in 0..ord {
                spline.a.fill(0.0);
                spline.a[ii] = 1.0;
                design[[row, offset + ii]] = spline.evaluate(value, nder);
            }
        } else {
            spline.basis_funcs(value);
            for ii in 0..ord {
                design[[row, offset + ii]] = spline.b[ii];
            }
        }
    }
    Ok(design)
}

/// The `splPTR` work structure of `splines.c`.
struct Spline<'a> {
    knots: &'a [f64],
    order: usize,
    ordm1: usize,
    curs: usize,
    boundary: bool,
    ldel: Vec<f64>,
    rdel: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl<'a> Spline<'a> {
    fn new(knots: &'a [f64], order: usize) -> Self {
        Self {
            knots,
            order,
            ordm1: order - 1,
            curs: 0,
            boundary: false,
            ldel: vec![0.0; order],
            rdel: vec![0.0; order],
            a: vec![0.0; order],
            b: vec![0.0; order],
        }
    }

    /// `set_cursor`: the index of the first knot strictly above `x` (pulled
    /// back onto the last legitimate interval when `x` sits on the right
    /// boundary).
    fn set_cursor(&mut self, x: f64) {
        let nknots = self.knots.len();
        let mut curs: isize = -1;
        self.boundary = false;
        for (i, &knot) in self.knots.iter().enumerate() {
            if knot >= x {
                curs = i as isize;
            }
            if knot > x {
                break;
            }
        }
        let last_legit = nknots - self.order;
        if curs > last_legit as isize && x == self.knots[last_legit] {
            self.boundary = true;
            curs = last_legit as isize;
        }
        self.curs = curs.max(0) as usize;
    }

    /// `diff_table`: distances from `x` to the `ndiff` knots on each side of
    /// the cursor.
    fn diff_table(&mut self, x: f64, ndiff: usize) {
        for i in 0..ndiff {
            self.rdel[i] = self.knots[self.curs + i] - x;
            self.ldel[i] = x - self.knots[self.curs - (i + 1)];
        }
    }

    /// `basis_funcs`: the `order` non-zero basis values at `x`, into `b`.
    fn basis_funcs(&mut self, x: f64) {
        self.diff_table(x, self.ordm1);
        self.b[0] = 1.0;
        for j in 1..=self.ordm1 {
            let mut saved = 0.0;
            for r in 0..j {
                let den = self.rdel[r] + self.ldel[j - 1 - r];
                if den != 0.0 {
                    let term = self.b[r] / den;
                    self.b[r] = saved + self.rdel[r] * term;
                    saved = self.ldel[j - 1 - r] * term;
                } else {
                    if r != 0 || self.rdel[r] != 0.0 {
                        self.b[r] = saved;
                    }
                    saved = 0.0;
                }
            }
            self.b[j] = saved;
        }
    }

    /// `evaluate`: the `nder`-th derivative at `x` of the spline whose
    /// B-spline coefficients on the current interval are in `a`.
    fn evaluate(&mut self, x: f64, nder: usize) -> f64 {
        let mut outer = self.ordm1;
        if self.boundary && nder == self.ordm1 {
            return 0.0;
        }
        for _ in 0..nder {
            for inner in 0..outer {
                let left = self.knots[self.curs - outer + inner];
                let right = self.knots[self.curs + inner];
                self.a[inner] = outer as f64 * (self.a[inner + 1] - self.a[inner]) / (right - left);
            }
            outer -= 1;
        }
        self.diff_table(x, outer);
        while outer > 0 {
            outer -= 1;
            for inner in 0..=outer {
                let ldel = self.ldel[outer - inner];
                let rdel = self.rdel[inner];
                self.a[inner] = (self.a[inner + 1] * ldel + self.a[inner] * rdel) / (rdel + ldel);
            }
        }
        self.a[0]
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn cubic_basis_sums_to_one_and_matches_r_at_interior_points() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let design = spline_design(&knots, &[0.0, 0.5, 1.0, 2.5, 3.0], 4, &[0]).unwrap();
        assert_eq!(design.dim(), (5, 6));
        for row in design.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
        // R: splineDesign(knots, c(0.5, 1), 4)
        let expected = [
            [
                0.125,
                0.59375,
                0.26041666666666663,
                0.020833333333333332,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.25,
                0.58333333333333326,
                0.16666666666666666,
                0.0,
                0.0,
            ],
        ];
        for (row, expected) in [1, 2].into_iter().zip(expected) {
            for (actual, expected) in design.row(row).iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-15, "{actual} != {expected}");
            }
        }
        assert_eq!(design[[0, 0]], 1.0);
        assert_eq!(design[[4, 5]], 1.0);
    }

    #[test]
    fn derivatives_match_finite_differences() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let x = 1.3;
        let h = 1e-5;
        let first = spline_design(&knots, &[x], 4, &[1]).unwrap();
        let second = spline_design(&knots, &[x], 4, &[2]).unwrap();
        let plus = spline_design(&knots, &[x + h], 4, &[0]).unwrap();
        let minus = spline_design(&knots, &[x - h], 4, &[0]).unwrap();
        let at = spline_design(&knots, &[x], 4, &[0]).unwrap();
        for j in 0..6 {
            let fd1 = (plus[[0, j]] - minus[[0, j]]) / (2.0 * h);
            let fd2 = (plus[[0, j]] - 2.0 * at[[0, j]] + minus[[0, j]]) / (h * h);
            assert!((first[[0, j]] - fd1).abs() < 1e-8, "d1[{j}]");
            assert!((second[[0, j]] - fd2).abs() < 1e-4, "d2[{j}]");
        }
    }

    #[test]
    fn second_derivative_at_boundary_knots_is_finite() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let constraint = spline_design(&knots, &[0.0, 3.0], 4, &[2, 2]).unwrap();
        // R: splineDesign(knots, c(0, 3), 4, derivs = c(2, 2))
        let expected = [
            [6.0, -9.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0, -9.0, 6.0],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (actual, expected) in constraint.row(row).iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
            }
        }
        // R: splineDesign(knots, 1.3, 4, derivs = 1)
        let first = spline_design(&knots, &[1.3], 4, &[1]).unwrap();
        let expected = [0.0, -0.36749999999999994, -0.3425, 0.6425, 0.0675, 0.0];
        for (actual, expected) in first.row(0).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14, "{actual} != {expected}");
        }
    }

    #[test]
    fn rejects_out_of_range_and_bad_knots() {
        let knots = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        assert!(spline_design(&knots, &[3.5], 4, &[0]).is_err());
        assert!(spline_design(&[1.0, 0.0, 2.0], &[1.0], 1, &[0]).is_err());
        assert!(spline_design(&[0.0, 1.0], &[0.5], 4, &[0]).is_err());
    }
}
