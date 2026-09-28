//! The matrix exponential of a multi-state transition matrix: R's
//! `survexpm` (`R/survexpm.R`) without its `setup` (Cholesky) branch, and
//! the Pade approximation it falls back on (`R/pade.R`, a trimmed copy of
//! the `expm` package's `expm2`, without derivatives).

use ndarray::Array2;

use crate::error::SurvivalResult;
use crate::internal::matrix::LuDecomposition;

/// `survexpm(rmat)`: `exp(A)` for a transition-rate matrix `A` (non-negative
/// off the diagonal, rows summing to 0).
///
/// Two shapes have a closed form: a single state with departures (the only
/// non-zero diagonal element), and a single state receiving every
/// transition.  Anything else goes to [`pade`].  A zero matrix gives the
/// identity (R's line for it is not returned, and `pade` of 0 is `I` too).
pub(crate) fn survexpm(a: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    let n = a.nrows();
    let departing: Vec<usize> = (0..n).filter(|&j| a[(j, j)] != 0.0).collect();
    if departing.is_empty() {
        return Ok(Array2::eye(n));
    }
    if let [j] = departing[..] {
        let mut emat = Array2::eye(n);
        let e = a[(j, j)].exp();
        emat[(j, j)] = e;
        let total: f64 = (0..n).filter(|&k| k != j).map(|k| a[(j, k)]).sum();
        for k in (0..n).filter(|&k| k != j) {
            emat[(j, k)] = (1.0 - e) * a[(j, k)] / total;
        }
        return Ok(emat);
    }
    let receiving: Vec<usize> = (0..n)
        .filter(|&k| a.column(k).iter().any(|&value| value > 0.0))
        .collect();
    if let [k] = receiving[..] {
        let mut emat = Array2::zeros((n, n));
        for i in 0..n {
            emat[(i, i)] = (-a[(i, k)]).exp();
        }
        for i in (0..n).filter(|&i| i != k) {
            emat[(i, k)] = 1.0 - emat[(i, i)];
        }
        return Ok(emat);
    }
    pade(a)
}

/// The numerator coefficients of the degree 3, 5, 7 and 9 Pade
/// approximants (`C` of pade.R).
const PADE_LOW: [&[f64]; 4] = [
    &[120.0, 60.0, 12.0, 1.0],
    &[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0],
    &[
        17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
    ],
    &[
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    ],
];

/// The 1-norms up to which each low-degree approximant is used.
const PADE_LOW_NORM: [f64; 4] = [0.015, 0.25, 0.95, 2.1];

/// The degree 13 coefficients (`c.` of pade.R).
const PADE_13: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// R's `pade(A)`: the smallest-degree Pade approximant of `exp(A)` its
/// 1-norm allows, and above a norm of 2.1 the degree 13 one on `A / 2^s`,
/// squared `s = max(0, ceiling(log2(norm / 5.4)))` times.
pub(crate) fn pade(a: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    let n = a.nrows();
    let identity = Array2::<f64>::eye(n);
    let norm = a
        .columns()
        .into_iter()
        .map(|column| column.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    if norm <= PADE_LOW_NORM[3] {
        let l = PADE_LOW_NORM
            .iter()
            .position(|&limit| norm <= limit)
            .expect("the norm is at most the last limit");
        let c = PADE_LOW[l];
        let a2 = a.dot(a);
        let mut power = identity.clone();
        let mut u = &identity * c[1];
        let mut v = &identity * c[0];
        for k in 1..=l + 1 {
            power = power.dot(&a2);
            u = u + &power * c[2 * k + 1];
            v = v + &power * c[2 * k];
        }
        let u = a.dot(&u);
        return solve(&(&v - &u), &(&v + &u));
    }
    let s = (norm / 5.4).log2();
    let squarings = if s > 0.0 { s.ceil() as i32 } else { 0 };
    let b = a / 2f64.powi(squarings);
    let c = &PADE_13;
    let b2 = b.dot(&b);
    let b4 = b2.dot(&b2);
    let b6 = b2.dot(&b4);
    let inner_u = &b6 * c[13] + &b4 * c[11] + &b2 * c[9];
    let u = b.dot(&(b6.dot(&inner_u) + &b6 * c[7] + &b4 * c[5] + &b2 * c[3] + &identity * c[1]));
    let inner_v = &b6 * c[12] + &b4 * c[10] + &b2 * c[8];
    let v = b6.dot(&inner_v) + &b6 * c[6] + &b4 * c[4] + &b2 * c[2] + &identity * c[0];
    let mut x = solve(&(&v - &u), &(&v + &u))?;
    for _ in 0..squarings {
        x = x.dot(&x);
    }
    Ok(x)
}

/// `solve(a, b)` for a square right-hand side.
fn solve(a: &Array2<f64>, b: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    let lu = LuDecomposition::decompose(a)?;
    let mut x = Array2::zeros(b.raw_dim());
    for (j, column) in b.columns().into_iter().enumerate() {
        let solution = lu.solve(&column.to_vec())?;
        for (i, value) in solution.into_iter().enumerate() {
            x[(i, j)] = value;
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tol * e.abs().max(1e-300) || (a - e).abs() < 1e-15,
                "{actual} != {expected}"
            );
        }
    }

    /// `A0 * target / 1.6`: `A0` has 1-norm 1.6.
    fn scaled(target: f64) -> Array2<f64> {
        let a0 = array![[-1.0, 0.6, 0.4], [0.5, -1.0, 0.5], [0.0, 0.0, 0.0]];
        a0 * (target / 1.6)
    }

    #[test]
    fn pade_matches_r_on_every_branch() {
        // R 4.5.3 / survival 3.8-12: survival:::pade(A0 * target / 1.6)
        let cases: [(f64, [[f64; 3]; 2]); 8] = [
            (
                0.01,
                [
                    [0.99377531349719, 0.00372664286842713, 0.00249804363438262],
                    [0.00310553572368928, 0.993775313497191, 0.00311915077912047],
                ],
            ),
            (
                0.2,
                [
                    [0.884566062776326, 0.0662389886173338, 0.0491949486063403],
                    [0.0551991571811115, 0.884566062776326, 0.0602347800425626],
                ],
            ),
            (
                0.9,
                [
                    [0.597039839559762, 0.195358447976627, 0.207601712463611],
                    [0.162798706647189, 0.597039839559762, 0.240161453793049],
                ],
            ),
            (
                2.0,
                [
                    [0.356318718078877, 0.232063862493432, 0.411617419427692],
                    [0.19338655207786, 0.356318718078877, 0.450294729843264],
                ],
            ),
            (
                2.5,
                [
                    [0.291174441890035, 0.221392742392065, 0.487432815717899],
                    [0.184493951993388, 0.291174441890035, 0.524331606116577],
                ],
            ),
            (
                4.0,
                [
                    [0.171841059082849, 0.165377395992185, 0.662781544924966],
                    [0.137814496660154, 0.171841059082849, 0.690344444256997],
                ],
            ),
            (
                8.0,
                [
                    [0.0523207521743452, 0.0568372537513215, 0.890841994074333],
                    [0.0473643781261013, 0.0523207521743452, 0.900314869699553],
                ],
            ),
            (
                40.0,
                [
                    [
                        6.14370091237957e-06,
                        6.73008715253353e-06,
                        0.999987126211934,
                    ],
                    [
                        5.60840596044461e-06,
                        6.14370091237957e-06,
                        0.999988247893126,
                    ],
                ],
            ),
        ];
        for (target, rows) in cases {
            let expected = array![
                [rows[0][0], rows[0][1], rows[0][2]],
                [rows[1][0], rows[1][1], rows[1][2]],
                [0.0, 0.0, 1.0]
            ];
            assert_close(&pade(&scaled(target)).unwrap(), &expected, 1e-12);
        }
    }

    #[test]
    fn survexpm_closed_forms() {
        // one departing state: exp of its rate, the rest split by the rates
        let a = array![[-0.3, 0.1, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let e = (-0.3f64).exp();
        let expected = array![
            [e, (1.0 - e) / 3.0, 2.0 * (1.0 - e) / 3.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        assert_close(&survexpm(&a).unwrap(), &expected, 1e-15);
        // one receiving state: competing entries into state 3
        let a = array![[-0.3, 0.0, 0.3], [0.0, -0.5, 0.5], [0.0, 0.0, 0.0]];
        let (e1, e2) = ((-0.3f64).exp(), (-0.5f64).exp());
        let expected = array![[e1, 0.0, 1.0 - e1], [0.0, e2, 1.0 - e2], [0.0, 0.0, 1.0]];
        assert_close(&survexpm(&a).unwrap(), &expected, 1e-15);
        // both agree with the Pade approximation
        assert_close(&survexpm(&a).unwrap(), &pade(&a).unwrap(), 1e-13);
        // no departures at all
        assert_eq!(
            survexpm(&Array2::zeros((3, 3))).unwrap(),
            Array2::<f64>::eye(3)
        );
    }
}
