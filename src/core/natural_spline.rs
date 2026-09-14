//! Natural cubic splines parameterised by the spline's values at the knots.
//!
//! [`nsk_basis`] ports R's `nsk()` (survival 3.8-12, `nsk.R`): it builds the
//! `splines::ns` natural spline basis (a B-spline basis with the second
//! derivative constrained to zero at the boundary knots, `ns` below ports
//! `ns.R`) and re-expresses it so that the coefficients are the fitted
//! values at the knots.  R's `ns` applies the constraint through a
//! LINPACK Householder QR (`dqrdc2`/`dqrsl`), reproduced here so the
//! intermediate basis matches R to rounding.

use crate::core::bspline::spline_design;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::lu_inverse;
use crate::internal::validation::validate_finite;
use ndarray::{Array2, Axis, s};
use pyo3::prelude::*;

/// A natural spline basis matrix with the attributes R attaches to it.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SplineBasisResult {
    /// Row-major `n_rows x n_cols` basis values (`NaN` rows for missing x).
    #[pyo3(get)]
    pub basis: Vec<f64>,
    #[pyo3(get)]
    pub n_rows: usize,
    #[pyo3(get)]
    pub n_cols: usize,
    /// Interior knots.
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
}

/// The knot specification of an `nsk` term: interior knots (or a target
/// `df`), boundary knots (`None` = the 5%/95% quantiles of the data, R's
/// `b = 0.05` default) and whether the basis carries an intercept column.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NaturalSplineKnot {
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub boundary_knots: Option<(f64, f64)>,
    #[pyo3(get)]
    pub intercept: bool,
    #[pyo3(get)]
    pub df: Option<usize>,
}

#[pymethods]
impl NaturalSplineKnot {
    #[new]
    #[pyo3(signature = (knots=None, boundary_knots=None, df=None, intercept=None))]
    pub fn new(
        knots: Option<Vec<f64>>,
        boundary_knots: Option<(f64, f64)>,
        df: Option<usize>,
        intercept: Option<bool>,
    ) -> PyResult<Self> {
        Ok(Self::try_new(
            knots,
            boundary_knots,
            df,
            intercept.unwrap_or(false),
        )?)
    }

    /// The basis evaluated at `x`; `NaN` entries of `x` give `NaN` rows.
    pub fn basis(&self, x: Vec<f64>) -> PyResult<SplineBasisResult> {
        Ok(self.evaluate(&x)?)
    }

    /// `basis(x) %*% coef`.
    pub fn predict(&self, x: Vec<f64>, coef: Vec<f64>) -> PyResult<Vec<f64>> {
        validate_finite(&coef, "coef")?;
        let basis = self.evaluate(&x)?;
        if coef.len() != basis.n_cols {
            return Err(SurvivalError::invalid_input(format!(
                "coef length ({}) must match number of basis functions ({})",
                coef.len(),
                basis.n_cols
            ))
            .into());
        }
        Ok(basis
            .basis
            .chunks(basis.n_cols)
            .map(|row| row.iter().zip(&coef).map(|(b, c)| b * c).sum())
            .collect())
    }
}

impl NaturalSplineKnot {
    pub fn try_new(
        knots: Option<Vec<f64>>,
        boundary_knots: Option<(f64, f64)>,
        df: Option<usize>,
        intercept: bool,
    ) -> SurvivalResult<Self> {
        if let Some((low, high)) = boundary_knots
            && !(low.is_finite() && high.is_finite() && low < high)
        {
            return Err(SurvivalError::invalid_input(
                "boundary_knots must be finite and strictly increasing",
            ));
        }
        let knots = knots.unwrap_or_default();
        validate_finite(&knots, "knots")?;
        let minimum = 1 + usize::from(intercept);
        if let Some(df) = df
            && df < minimum
        {
            return Err(SurvivalError::invalid_input(format!(
                "df must be at least {minimum} when intercept is {intercept}"
            )));
        }
        Ok(Self {
            knots,
            boundary_knots,
            intercept,
            df,
        })
    }

    fn evaluate(&self, x: &[f64]) -> SurvivalResult<SplineBasisResult> {
        nsk_basis(
            x,
            self.df,
            (!self.knots.is_empty()).then_some(self.knots.as_slice()),
            self.intercept,
            self.boundary_knots,
        )
    }
}

/// R's `nsk(x, df = NULL, knots = NULL, Boundary.knots = <5%/95%>)` without
/// an intercept column.
#[pyfunction]
#[pyo3(signature = (x, df=None, knots=None, boundary_knots=None))]
pub fn nsk(
    x: Vec<f64>,
    df: Option<usize>,
    knots: Option<Vec<f64>>,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<SplineBasisResult> {
    Ok(nsk_basis(&x, df, knots.as_deref(), false, boundary_knots)?)
}

/// R's `nsk(x, df, knots, intercept, Boundary.knots)`.  `boundary_knots =
/// None` uses the 5% and 95% quantiles of the non-missing `x` (R's default
/// `b = 0.05`); explicit interior knots outside the boundary widen it as in
/// R.  Missing (`NaN`) `x` values give `NaN` rows.
pub fn nsk_basis(
    x: &[f64],
    df: Option<usize>,
    knots: Option<&[f64]>,
    intercept: bool,
    boundary_knots: Option<(f64, f64)>,
) -> SurvivalResult<SplineBasisResult> {
    if let Some(value) = x.iter().find(|value| value.is_infinite()) {
        return Err(SurvivalError::invalid_input(format!(
            "x contains non-finite value {value}"
        )));
    }
    let observed: Vec<f64> = x.iter().copied().filter(|value| !value.is_nan()).collect();
    if observed.is_empty() {
        return Err(SurvivalError::invalid_input(
            "x must contain at least one non-missing value",
        ));
    }
    let boundary = match boundary_knots {
        Some(pair) => pair,
        None => {
            let mut sorted = observed.clone();
            sorted.sort_by(f64::total_cmp);
            (quantile_type7(&sorted, 0.05), quantile_type7(&sorted, 0.95))
        }
    };
    // Widen the boundary to enclose explicit interior knots (nsk.R).
    let mut kx: Vec<f64> = knots.map(<[f64]>::to_vec).unwrap_or_default();
    if kx.is_empty() {
        kx = vec![boundary.0, boundary.1];
    } else {
        let max_knot = kx.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_knot = kx.iter().copied().fold(f64::INFINITY, f64::min);
        if boundary.1 > max_knot {
            kx.push(boundary.1);
        }
        if boundary.0 < min_knot {
            kx.push(boundary.0);
        }
        kx.sort_by(f64::total_cmp);
    }
    if kx.len() < 2 || kx[0] >= kx[kx.len() - 1] {
        return Err(SurvivalError::invalid_input(
            "nsk needs two distinct boundary knots",
        ));
    }
    let bknot = (kx[0], kx[kx.len() - 1]);
    let mut iknot: Vec<f64> = kx[1..kx.len() - 1].to_vec();

    let mut basis = if iknot.is_empty() {
        ns(&observed, df, None, intercept, bknot)?
    } else {
        ns(&observed, df, Some(&iknot), intercept, bknot)?
    };
    iknot = basis.knots.clone();
    // Duplicate knots are not allowed but can come out of `ns` when many
    // tied values sit near an end; collapse them and refit.
    let mut all: Vec<f64> = vec![bknot.0, bknot.1];
    all.extend(&iknot);
    let mut unique = Vec::with_capacity(all.len());
    for value in all {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    if unique.len() < iknot.len() + 2 {
        basis = if unique.len() == 2 {
            ns(&observed, None, None, intercept, bknot)?
        } else {
            ns(&observed, None, Some(&unique[2..]), intercept, bknot)?
        };
        iknot = basis.knots.clone();
    }

    // Reparameterise so the coefficients are the spline values at the
    // knots: gamma = kbasis beta, hence basis beta = (basis kbasis^-1) gamma.
    let mut kx = vec![bknot.0];
    kx.extend(&iknot);
    kx.push(bknot.1);
    let kbasis = ns(&kx, df, Some(&iknot), intercept, bknot)?;
    let n_obs = observed.len();
    let ibasis = if intercept {
        let inverse = lu_inverse(&kbasis.values).map_err(|_| singular_knot_basis())?;
        basis.values.dot(&inverse)
    } else {
        let with_one = |values: &Array2<f64>| {
            let mut out = Array2::ones((values.nrows(), values.ncols() + 1));
            out.slice_mut(s![.., 1..]).assign(values);
            out
        };
        let inverse = lu_inverse(&with_one(&kbasis.values)).map_err(|_| singular_knot_basis())?;
        let full = with_one(&basis.values).dot(&inverse);
        full.slice(s![.., 1..]).to_owned()
    };
    if let Some(value) = ibasis.iter().find(|value| !value.is_finite()) {
        return Err(SurvivalError::computation(format!(
            "knot-height transform produced non-finite value {value}"
        )));
    }

    let n_cols = ibasis.ncols();
    let mut values = Vec::with_capacity(x.len() * n_cols);
    let mut observed_row = 0;
    for value in x {
        if value.is_nan() {
            values.extend(std::iter::repeat_n(f64::NAN, n_cols));
        } else {
            values.extend(ibasis.row(observed_row).iter().copied());
            observed_row += 1;
        }
    }
    debug_assert_eq!(observed_row, n_obs);
    Ok(SplineBasisResult {
        basis: values,
        n_rows: x.len(),
        n_cols,
        knots: iknot,
        boundary_knots: bknot,
    })
}

fn singular_knot_basis() -> SurvivalError {
    SurvivalError::computation(
        "knot-height transform is singular; knots must be distinct and well-spaced",
    )
}

/// The `splines::ns` basis and the interior knots it settled on.
pub(crate) struct NaturalSplineBasis {
    pub(crate) values: Array2<f64>,
    pub(crate) knots: Vec<f64>,
}

/// R's `ns(x, df, knots, intercept, Boundary.knots)` for finite `x`
/// (`ns.R`, R 4.5).  Interior knots come from `knots`, else from `df` as
/// the equally spaced quantiles of the `x` inside the boundary, else none.
pub(crate) fn ns(
    x: &[f64],
    df: Option<usize>,
    knots: Option<&[f64]>,
    intercept: bool,
    boundary_knots: (f64, f64),
) -> SurvivalResult<NaturalSplineBasis> {
    let (low, high) = if boundary_knots.0 <= boundary_knots.1 {
        boundary_knots
    } else {
        (boundary_knots.1, boundary_knots.0)
    };
    let outside: Vec<bool> = x.iter().map(|&v| v < low || v > high).collect();
    let computed_from_df = knots.is_none() && df.is_some();
    let mut knots: Vec<f64> = match (knots, df) {
        (Some(knots), _) => knots.to_vec(),
        (None, Some(df)) => {
            let n_interior = df.saturating_sub(1 + usize::from(intercept));
            let mut inside: Vec<f64> = x
                .iter()
                .zip(&outside)
                .filter(|(_, out)| !**out)
                .map(|(&v, _)| v)
                .collect();
            inside.sort_by(f64::total_cmp);
            if inside.is_empty() && n_interior > 0 {
                return Err(SurvivalError::invalid_input(
                    "no x values inside the boundary knots to place interior knots",
                ));
            }
            (1..=n_interior)
                .map(|i| quantile_type7(&inside, i as f64 / (n_interior + 1) as f64))
                .collect()
        }
        (None, None) => Vec::new(),
    };
    // Computed knots that fall on a boundary knot are shoved inside.
    if computed_from_df {
        shove_boundary_knots(&mut knots, low, high)?;
    }
    let n_interior = knots.len();
    let mut aknots: Vec<f64> = vec![low, low, low, low, high, high, high, high];
    aknots.extend(&knots);
    aknots.sort_by(f64::total_cmp);

    let n = x.len();
    let n_basis = n_interior + 4;
    let mut basis = Array2::zeros((n, n_basis));
    let inside: Vec<usize> = (0..n).filter(|&i| !outside[i]).collect();
    if inside.len() < n {
        // Linear extrapolation beyond the boundary knots.
        for (pivot, rows) in [
            (low, (0..n).filter(|&i| x[i] < low).collect::<Vec<_>>()),
            (high, (0..n).filter(|&i| x[i] > high).collect::<Vec<_>>()),
        ] {
            if rows.is_empty() {
                continue;
            }
            let tt = spline_design(&aknots, &[pivot, pivot], 4, &[0, 1])?;
            for &row in &rows {
                let delta = x[row] - pivot;
                for col in 0..n_basis {
                    basis[[row, col]] = tt[[0, col]] + delta * tt[[1, col]];
                }
            }
        }
    }
    if !inside.is_empty() {
        let inside_x: Vec<f64> = inside.iter().map(|&i| x[i]).collect();
        let values = spline_design(&aknots, &inside_x, 4, &[0])?;
        for (k, &row) in inside.iter().enumerate() {
            basis.row_mut(row).assign(&values.row(k));
        }
    }
    let mut constraint = spline_design(&aknots, &[low, high], 4, &[2, 2])?;
    if !intercept {
        constraint = constraint.slice(s![.., 1..]).to_owned();
        basis = basis.slice(s![.., 1..]).to_owned();
    }
    // basis <- t(qr.qty(qr(t(const)), t(basis)))[, -(1:2)]
    let qr = HouseholderQr::decompose(constraint.t().to_owned());
    let mut projected = basis.t().to_owned();
    for mut column in projected.axis_iter_mut(Axis(1)) {
        let mut values = column.to_vec();
        qr.qty(&mut values);
        column.assign(&ndarray::Array1::from(values));
    }
    let values = projected.t().slice(s![.., 2..]).to_owned();
    Ok(NaturalSplineBasis { values, knots })
}

/// `ns.R`: interior knots computed from `df` that coincide with a boundary
/// knot are moved 1/8 of the way to the nearest other knot.
fn shove_boundary_knots(knots: &mut [f64], low: f64, high: f64) -> SurvivalResult<()> {
    if knots.is_empty() {
        return Ok(());
    }
    for (pivot, sign) in [(low, 1.0), (high, -1.0)] {
        if !knots.contains(&pivot) {
            continue;
        }
        let nearest = knots
            .iter()
            .copied()
            .filter(|&k| if sign > 0.0 { k > pivot } else { k < pivot })
            .fold(None, |best: Option<f64>, k| {
                Some(best.map_or(k, |b| {
                    if (k - pivot).abs() < (b - pivot).abs() {
                        k
                    } else {
                        b
                    }
                }))
            })
            .ok_or_else(|| {
                SurvivalError::invalid_input(format!(
                    "all interior knots match the {} boundary knot",
                    if sign > 0.0 { "left" } else { "right" }
                ))
            })?;
        for knot in knots.iter_mut().filter(|k| **k == pivot) {
            *knot += sign * (nearest - pivot).abs() / 8.0;
        }
    }
    Ok(())
}

/// R's `quantile(x, p, type = 7)` on sorted data, with R's rounding fuzz.
fn quantile_type7(sorted: &[f64], probability: f64) -> f64 {
    let n = sorted.len();
    let fuzz = 4.0 * f64::EPSILON;
    let nppm = 1.0 + probability * (n as f64 - 1.0);
    let j = (nppm + fuzz).floor();
    let mut h = nppm - j;
    if h.abs() < fuzz {
        h = 0.0;
    }
    let at = |index: f64| sorted[(index as usize).clamp(1, n) - 1];
    let lower = at(j);
    let upper = at(j + 1.0);
    if h == 1.0 {
        upper
    } else if h > 0.0 && h < 1.0 && lower != upper {
        (1.0 - h) * lower + h * upper
    } else {
        lower
    }
}

/// LINPACK `dqrdc2` (no pivoting needed: the constraint matrix has full
/// column rank) and the `qty` branch of `dqrsl`, as used by R's `qr()` and
/// `qr.qty()`.
struct HouseholderQr {
    qr: Array2<f64>,
    qraux: Vec<f64>,
}

impl HouseholderQr {
    fn decompose(mut x: Array2<f64>) -> Self {
        let (n, p) = x.dim();
        let mut qraux = vec![0.0; p];
        for l in 0..n.min(p) {
            let nrmxl_raw: f64 = (l..n).map(|i| x[[i, l]] * x[[i, l]]).sum::<f64>().sqrt();
            if nrmxl_raw == 0.0 {
                continue;
            }
            let nrmxl = if x[[l, l]] != 0.0 {
                nrmxl_raw.copysign(x[[l, l]])
            } else {
                nrmxl_raw
            };
            for i in l..n {
                x[[i, l]] /= nrmxl;
            }
            x[[l, l]] += 1.0;
            for j in l + 1..p {
                let dot: f64 = (l..n).map(|i| x[[i, l]] * x[[i, j]]).sum();
                let t = -dot / x[[l, l]];
                for i in l..n {
                    x[[i, j]] += t * x[[i, l]];
                }
            }
            qraux[l] = x[[l, l]];
            x[[l, l]] = -nrmxl;
        }
        Self { qr: x, qraux }
    }

    /// Overwrites `y` with `t(Q) %*% y`.
    fn qty(&self, y: &mut [f64]) {
        let (n, p) = self.qr.dim();
        let ju = p.min(n.saturating_sub(1));
        for j in 0..ju {
            if self.qraux[j] == 0.0 {
                continue;
            }
            let head = self.qraux[j];
            let column = self.qr.column(j);
            let dot: f64 = head * y[j]
                + column
                    .iter()
                    .zip(y.iter())
                    .skip(j + 1)
                    .map(|(q, v)| q * v)
                    .sum::<f64>();
            let t = -dot / head;
            y[j] += t * head;
            for (value, q) in y.iter_mut().zip(column.iter()).skip(j + 1) {
                *value += t * q;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    fn rows(result: &SplineBasisResult) -> Vec<Vec<f64>> {
        result
            .basis
            .chunks(result.n_cols)
            .map(<[f64]>::to_vec)
            .collect()
    }

    #[test]
    fn ns_matches_r_for_explicit_knots() {
        // R: ns(c(1, 2.5, 4, 7, 10), knots = c(3, 6), Boundary.knots = c(1, 10))
        let basis = ns(
            &[1.0, 2.5, 4.0, 7.0, 10.0],
            None,
            Some(&[3.0, 6.0]),
            false,
            (1.0, 10.0),
        )
        .unwrap();
        let expected = [
            [0.0, 0.0, 0.0],
            [
                -0.122845343733707452,
                0.44094969526769551,
                -0.280604351533987995,
            ],
            [
                0.065189184912002263,
                0.56834878911104147,
                -0.357347497832567473,
            ],
            [
                0.498970572264373358,
                0.34211664055868773,
                0.051769930034081833,
            ],
            [
                -0.150537634408602128,
                0.41397849462365582,
                0.736559139784946248,
            ],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (col, expected) in expected.iter().enumerate() {
                let actual = basis.values[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "[{row}, {col}]: {actual} != {expected}"
                );
            }
        }
        // With an intercept and points beyond the boundary knots (linear
        // extrapolation): ns(c(0.5, 2.5, 4, 7, 12), knots = c(3, 6),
        // Boundary.knots = c(1, 10), intercept = TRUE)
        let basis = ns(
            &[0.5, 2.5, 4.0, 7.0, 12.0],
            None,
            Some(&[3.0, 6.0]),
            true,
            (1.0, 10.0),
        )
        .unwrap();
        let expected = [
            [
                -0.48225065196458722,
                -0.275073995262384130,
                0.75645348697155645,
                -0.481379491709172214,
            ],
            [
                0.44710430827990233,
                -0.078683656574832644,
                0.31950505558078979,
                -0.203321399005957110,
            ],
            [
                0.60915949400686498,
                0.208343982651436943,
                0.17467309532759609,
                -0.106826601788556796,
            ],
            [
                0.10363454326473556,
                0.525584479133833993,
                0.26892839666767082,
                0.098344267055638035,
            ],
            [
                0.0,
                -0.731182795698924526,
                0.51075268817204256,
                1.220430107526882191,
            ],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (col, expected) in expected.iter().enumerate() {
                let actual = basis.values[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "[{row}, {col}]: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn nsk_is_knot_height_parameterised() {
        let result = nsk_basis(
            &[1.0, 3.0, 5.0, 7.0, 10.0],
            None,
            Some(&[3.0, 5.0, 7.0]),
            false,
            Some((1.0, 10.0)),
        )
        .unwrap();
        assert_eq!(result.n_rows, 5);
        assert_eq!(result.n_cols, 4);
        assert_eq!(result.knots, vec![3.0, 5.0, 7.0]);
        for (row, values) in rows(&result).iter().enumerate() {
            for (col, actual) in values.iter().enumerate() {
                let expected = if row > 0 && row - 1 == col { 1.0 } else { 0.0 };
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "[{row}, {col}] = {actual}"
                );
            }
        }
    }

    #[test]
    fn nsk_df_places_quantile_knots_and_default_boundary() {
        let result = nsk_basis(&[1.0, 2.0, 3.0, 4.0, 5.0], Some(3), None, false, None).unwrap();
        assert_eq!(result.n_cols, 3);
        assert!((result.boundary_knots.0 - 1.2).abs() < 1e-12);
        assert!((result.boundary_knots.1 - 4.8).abs() < 1e-12);
        assert!((result.knots[0] - 2.6666666666666665).abs() < 1e-12);
        assert!((result.knots[1] - 3.333333333333333).abs() < 1e-12);
        // R: nsk(1:5, df = 3)
        let expected = [
            [
                -0.30663390663390655,
                0.12972972972972971,
                -0.0075075075075074962,
            ],
            [
                1.02389993299084203,
                -0.36452981907527365,
                0.0210954756409301933,
            ],
            [
                0.52303439803439766,
                0.52303439803439866,
                -0.0230343980343980653,
            ],
            [
                -0.36452981907527360,
                1.02389993299084203,
                0.3195344104435014487,
            ],
            [
                0.12972972972972999,
                -0.30663390663390683,
                1.1844116844116845400,
            ],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (actual, expected) in rows(&result)[row].iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn nsk_keeps_missing_rows_and_widens_the_boundary_to_the_knots() {
        let result =
            nsk_basis(&[1.0, f64::NAN, 2.0, 3.0, 4.0], Some(3), None, false, None).unwrap();
        let observed = nsk_basis(&[1.0, 2.0, 3.0, 4.0], Some(3), None, false, None).unwrap();
        assert_eq!(result.n_rows, 5);
        assert!(rows(&result)[1].iter().all(|v| v.is_nan()));
        assert_eq!(rows(&result)[2], rows(&observed)[1]);

        let widened = nsk_basis(
            &[2.0, 3.0, 4.0],
            None,
            Some(&[1.0, 6.0]),
            false,
            Some((2.0, 4.0)),
        )
        .unwrap();
        assert_eq!(widened.boundary_knots, (1.0, 6.0));
        assert!(widened.knots.is_empty());
    }

    #[test]
    fn nsk_collapses_duplicate_quantile_knots() {
        let tied = nsk_basis(&[0.0, 1.0, 1.0, 1.0, 2.0], Some(4), None, false, None)
            .expect("duplicate computed knots collapse like R's nsk");
        assert_eq!(tied.n_cols, 2);
        assert_eq!(tied.knots, vec![1.0]);
    }

    #[test]
    fn predict_and_constructor_validate() {
        let spline =
            NaturalSplineKnot::try_new(Some(vec![3.0, 5.0, 7.0]), Some((1.0, 10.0)), None, false)
                .unwrap();
        let x = vec![1.0, 3.0, 5.0, 7.0, 10.0];
        let predictions = spline.predict(x, vec![30.0, 50.0, 70.0, 100.0]).unwrap();
        for (actual, expected) in predictions.iter().zip([0.0, 30.0, 50.0, 70.0, 100.0]) {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(NaturalSplineKnot::try_new(None, Some((2.0, 2.0)), Some(3), false).is_err());
        assert!(NaturalSplineKnot::try_new(None, Some((0.0, 10.0)), Some(1), true).is_err());
        assert!(NaturalSplineKnot::try_new(Some(vec![f64::NAN]), None, None, false).is_err());
        assert!(nsk_basis(&[1.0, f64::INFINITY], Some(3), None, false, None).is_err());
        assert!(nsk_basis(&[f64::NAN], Some(3), None, false, None).is_err());
        assert!(nsk_basis(&[1.0, 1.0], Some(3), None, false, None).is_err());
        assert_eq!(
            nsk(vec![1.0, 2.0, 3.0], Some(2), None, None)
                .unwrap()
                .n_cols,
            2
        );
    }

    #[test]
    fn quantile_type7_matches_r() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_type7(&x, 0.05), 1.2);
        assert_eq!(quantile_type7(&x, 0.5), 3.0);
        assert!((quantile_type7(&x, 1.0 / 3.0) - 2.3333333333333335).abs() < 1e-15);
        assert_eq!(quantile_type7(&x, 1.0), 5.0);
        assert_eq!(quantile_type7(&[7.0], 0.3), 7.0);
    }
}
