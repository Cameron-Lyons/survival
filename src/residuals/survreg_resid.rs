//! Residuals of a parametric survival fit: a port of
//! `R/residuals.survreg.R` from the CRAN `survival` package.
//!
//! Every residual type derives from the same per-observation derivative
//! matrix ([`survreg_deriv`], R's `deriv`): the log-likelihood `g` and its
//! first and second derivatives with respect to the linear predictor and
//! `log(scale)`, evaluated with the distribution's `density` function.
//! `survreg.fit` uses the same derivatives (its `derfun`) for its starting
//! values.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use crate::regression::parametric_survival::SurvregFit;
use crate::regression::survreg_distributions::SurvregDistribution;
use pyo3::prelude::*;

/// The `type` argument of `residuals.survreg`.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregResidType {
    /// `itrans(center of the saturated model) - itrans(eta)`.
    Response,
    Deviance,
    /// Per-coefficient influence, one column per coefficient (plus one per
    /// `Log(scale)` when `rsigma`).
    Dfbeta,
    /// `Dfbeta` scaled by the coefficient standard errors.
    Dfbetas,
    Working,
    /// Likelihood displacement from case deletion.
    Ldcase,
    /// Likelihood displacement from a response perturbation.
    Ldresp,
    /// Likelihood displacement from a shape (scale) perturbation.
    Ldshape,
    /// The raw derivative matrix `(g, dg, ddg, ds, dds, dsg)`.
    Matrix,
}

impl SurvregResidType {
    const CHOICES: [(&'static str, Self); 9] = [
        ("response", Self::Response),
        ("deviance", Self::Deviance),
        ("dfbeta", Self::Dfbeta),
        ("dfbetas", Self::Dfbetas),
        ("working", Self::Working),
        ("ldcase", Self::Ldcase),
        ("ldresp", Self::Ldresp),
        ("ldshape", Self::Ldshape),
        ("matrix", Self::Matrix),
    ];

    /// `match.arg(type)`: an exact name or a unique prefix.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        let key = name.trim().to_lowercase();
        if let Some((_, kind)) = Self::CHOICES.iter().find(|(choice, _)| *choice == key) {
            return Ok(*kind);
        }
        let matches: Vec<Self> = Self::CHOICES
            .iter()
            .filter(|(choice, _)| !key.is_empty() && choice.starts_with(key.as_str()))
            .map(|(_, kind)| *kind)
            .collect();
        match matches.as_slice() {
            [kind] => Ok(*kind),
            _ => Err(SurvivalError::invalid_input(format!(
                "residual type '{name}' should be one of {}",
                Self::CHOICES
                    .iter()
                    .map(|(choice, _)| format!("\"{choice}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    /// Whether R returns a matrix (one row per observation) for this type.
    pub fn is_matrix(self) -> bool {
        matches!(self, Self::Dfbeta | Self::Dfbetas | Self::Matrix)
    }
}

/// Residuals of a `survreg` fit.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregResiduals {
    #[pyo3(get)]
    pub residual_type: SurvregResidType,
    /// One row per observation (per group after `collapse`).  Vector-valued
    /// types have a single column; `Dfbeta`/`Dfbetas` have one column per
    /// coefficient followed by one per `Log(scale)` when `rsigma`; `Matrix`
    /// has the columns `g, dg, ddg, ds, dds, dsg`.
    #[pyo3(get)]
    pub values: Vec<Vec<f64>>,
}

#[pymethods]
impl SurvregResiduals {
    fn __repr__(&self) -> String {
        format!(
            "SurvregResiduals(type={:?}, n={}, columns={})",
            self.residual_type,
            self.values.len(),
            self.values.first().map_or(0, Vec::len)
        )
    }
}

/// One row of R's `deriv` matrix, `cbind(g, dg, ddg, ds, dds, dsg)`: the
/// log-likelihood of one observation and its derivatives with respect to
/// `eta` (`dg`, `ddg`), `log(sigma)` (`ds`, `dds`) and the cross term
/// (`dsg`).  `y1`, `y2` are on the transformed scale; `y2` is read only for
/// an interval-censored (`status == 3`) row.
///
/// For interval-censored rows R's `residuals.survreg` gets the `log(sigma)`
/// derivatives wrong (`ds` has the opposite sign and `dsg` lacks a `1/sigma`
/// factor, which also corrupts `dds`); the formulas here are those of
/// `survregc1.c`, which the fit itself uses, and agree with finite
/// differences.  The other rows are exactly R's.
pub(crate) fn survreg_deriv(
    distribution: &SurvregDistribution,
    y1: f64,
    y2: f64,
    status: i32,
    eta: f64,
    sigma: f64,
) -> [f64; 6] {
    let z = (y1 - eta) / sigma;
    let dmat = distribution.density(z);
    let dtemp = dmat.pdf * dmat.score; // f'
    let (z2, dmat2) = if status == 3 {
        let z2 = (y2 - eta) / sigma;
        (z2, distribution.density(z2))
    } else {
        (0.0, dmat)
    };
    let (tdenom, numerator_dg, numerator_ddg) = match status {
        0 => (dmat.survival, -dmat.pdf, -dtemp),
        1 => (1.0, dmat.score, dmat.curvature),
        2 => (dmat.cdf, dmat.pdf, dtemp),
        _ => (
            if z > 0.0 {
                dmat.survival - dmat2.survival
            } else {
                dmat2.cdf - dmat.cdf
            },
            dmat2.pdf - dmat.pdf,
            dmat2.pdf * dmat2.score - dtemp,
        ),
    };
    let g = if status == 1 {
        (dmat.pdf / sigma).ln()
    } else {
        tdenom.ln()
    };
    let tdenom = 1.0 / tdenom;
    let dg = -(tdenom / sigma) * numerator_dg;
    let ddg = (tdenom / (sigma * sigma)) * numerator_ddg;
    let (ds, dds, dsg) = if status < 3 {
        (dg * sigma * z, ddg * (sigma * z).powi(2), ddg * sigma * z)
    } else {
        (
            tdenom * (z * dmat.pdf - z2 * dmat2.pdf),
            tdenom * (z2 * z2 * dmat2.pdf * dmat2.score - z * z * dtemp),
            tdenom * (z2 * dmat2.pdf * dmat2.score - z * dtemp) / sigma,
        )
    };
    [
        g,
        dg,
        ddg - dg * dg,
        if status == 1 { ds - 1.0 } else { ds },
        dds - ds * (1.0 + ds),
        dsg - dg * (1.0 + ds),
    ]
}

/// R's `sign`: zero for zero (unlike `f64::signum`), NaN for NaN.
fn r_sign(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value.signum() }
}

/// `score %*% vv` for one row.
fn row_times_matrix(score: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    (0..score.len())
        .map(|col| {
            score
                .iter()
                .zip(matrix)
                .map(|(s, row)| s * row[col])
                .sum::<f64>()
        })
        .collect()
}

/// `rowSums(score * (score %*% vv))`.
fn quadratic_form(score: &[f64], matrix: &[Vec<f64>]) -> f64 {
    row_times_matrix(score, matrix)
        .iter()
        .zip(score)
        .map(|(a, b)| a * b)
        .sum()
}

/// A per-observation score row with the `Log(scale)` block appended in the
/// observation's stratum column when `rsigma`.
fn score_row(
    x: &[f64],
    eta_part: f64,
    scale_part: f64,
    stratum: usize,
    nstrata: usize,
    rsigma: bool,
) -> Vec<f64> {
    let mut row: Vec<f64> = x.iter().map(|value| eta_part * value).collect();
    if rsigma {
        let start = row.len();
        row.resize(start + nstrata, 0.0);
        row[start + stratum] = scale_part;
    }
    row
}

/// `rowsum(rr, collapse)`: rows summed within each group, groups in
/// increasing order of their code.
fn collapse_rows(rows: Vec<Vec<f64>>, collapse: &[usize]) -> SurvivalResult<Vec<Vec<f64>>> {
    validate_length(rows.len(), collapse.len(), "collapse")?;
    let mut groups: Vec<usize> = collapse.to_vec();
    groups.sort_unstable();
    groups.dedup();
    let width = rows.first().map_or(0, Vec::len);
    let mut out = vec![vec![0.0; width]; groups.len()];
    for (row, &group) in rows.iter().zip(collapse) {
        let target = groups.binary_search(&group).expect("group was collected");
        for (sum, value) in out[target].iter_mut().zip(row) {
            *sum += value;
        }
    }
    Ok(out)
}

/// `residuals.survreg(object, type, rsigma, collapse, weighted)`.
///
/// `rsigma` includes the `Log(scale)` derivatives in the influence types;
/// it is ignored, as in R, when the scale was fixed.  `collapse` sums the
/// rows within groups (R's `rowsum`) and `weighted` multiplies each row by
/// its case weight first (a no-op when the fit had no weights).
pub fn residuals_survreg(
    fit: &SurvregFit,
    residual_type: SurvregResidType,
    rsigma: bool,
    collapse: Option<&[usize]>,
    weighted: bool,
) -> SurvivalResult<SurvregResiduals> {
    let n = fit.n;
    let nvar = fit.nvar();
    let nstrata = fit.nstrata();
    let distribution = &fit.distribution;
    let transform = distribution.transform;
    // If the variance wasn't estimated then it has no error.
    let rsigma = rsigma && fit.variance_matrix.len() != nvar;
    let vv = fit
        .naive_variance_matrix
        .as_deref()
        .unwrap_or(&fit.variance_matrix);
    let width = nvar + if rsigma { nstrata } else { 0 };
    if vv.len() < width || vv.iter().any(|row| row.len() < width) {
        return Err(SurvivalError::invalid_input(format!(
            "variance matrix must be at least {width} x {width} for these residuals"
        )));
    }

    let y1: Vec<f64> = fit.time.iter().map(|&t| transform.apply(t)).collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| match &fit.time2 {
            Some(time2) if fit.status[i] == 3 => transform.apply(time2[i]),
            _ => y1[i],
        })
        .collect();
    let sigma = |i: usize| fit.scale[fit.strata[i]];
    let eta = &fit.linear_predictors;

    let rows: Vec<Vec<f64>> = match residual_type {
        SurvregResidType::Response => (0..n)
            .map(|i| {
                let (center, _) = distribution.deviance(y1[i], y2[i], fit.status[i], sigma(i));
                vec![transform.inverse(center) - transform.inverse(eta[i])]
            })
            .collect(),
        _ => {
            let deriv: Vec<[f64; 6]> = (0..n)
                .map(|i| survreg_deriv(distribution, y1[i], y2[i], fit.status[i], eta[i], sigma(i)))
                .collect();
            let working = |i: usize| -deriv[i][1] / deriv[i][2];
            match residual_type {
                SurvregResidType::Deviance => (0..n)
                    .map(|i| {
                        let (_, saturated) =
                            distribution.deviance(y1[i], y2[i], fit.status[i], sigma(i));
                        let rr = working(i);
                        // The saturated log-likelihood bounds `g`, so a
                        // negative difference is rounding noise (R would
                        // return NaN for it).
                        vec![r_sign(rr) * (2.0 * (saturated - deriv[i][0])).max(0.0).sqrt()]
                    })
                    .collect(),
                SurvregResidType::Working => (0..n).map(|i| vec![working(i)]).collect(),
                SurvregResidType::Dfbeta | SurvregResidType::Dfbetas | SurvregResidType::Ldcase => {
                    let standardize: Vec<f64> = if residual_type == SurvregResidType::Dfbetas {
                        (0..width).map(|j| 1.0 / vv[j][j].sqrt()).collect()
                    } else {
                        vec![1.0; width]
                    };
                    (0..n)
                        .map(|i| {
                            let score = score_row(
                                &fit.covariates[i],
                                deriv[i][1],
                                deriv[i][3],
                                fit.strata[i],
                                nstrata,
                                rsigma,
                            );
                            let rr = row_times_matrix(&score, vv);
                            if residual_type == SurvregResidType::Ldcase {
                                vec![rr.iter().zip(&score).map(|(a, b)| a * b).sum()]
                            } else {
                                rr.iter().zip(&standardize).map(|(a, b)| a * b).collect()
                            }
                        })
                        .collect()
                }
                SurvregResidType::Ldresp => (0..n)
                    .map(|i| {
                        let score = score_row(
                            &fit.covariates[i],
                            deriv[i][2] * sigma(i),
                            deriv[i][5] * sigma(i),
                            fit.strata[i],
                            nstrata,
                            rsigma,
                        );
                        vec![quadratic_form(&score, vv)]
                    })
                    .collect(),
                SurvregResidType::Ldshape => (0..n)
                    .map(|i| {
                        let score = score_row(
                            &fit.covariates[i],
                            deriv[i][5],
                            deriv[i][4],
                            fit.strata[i],
                            nstrata,
                            rsigma,
                        );
                        vec![quadratic_form(&score, vv)]
                    })
                    .collect(),
                SurvregResidType::Matrix => deriv.iter().map(|row| row.to_vec()).collect(),
                SurvregResidType::Response => unreachable!("handled above"),
            }
        }
    };

    let mut rows = rows;
    if weighted && let Some(weights) = &fit.weights {
        for (row, &w) in rows.iter_mut().zip(weights) {
            row.iter_mut().for_each(|value| *value *= w);
        }
    }
    if let Some(groups) = collapse {
        rows = collapse_rows(rows, groups)?;
    }
    Ok(SurvregResiduals {
        residual_type,
        values: rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::survreg_distributions::SurvregDistribution;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn residual_types_parse_like_match_arg() {
        assert_eq!(
            SurvregResidType::parse("dfbetas").unwrap(),
            SurvregResidType::Dfbetas
        );
        assert_eq!(
            SurvregResidType::parse("dfbeta").unwrap(),
            SurvregResidType::Dfbeta
        );
        assert_eq!(
            SurvregResidType::parse("dev").unwrap(),
            SurvregResidType::Deviance
        );
        assert_eq!(
            SurvregResidType::parse("Matrix").unwrap(),
            SurvregResidType::Matrix
        );
        assert!(SurvregResidType::parse("ld").is_err());
        assert!(SurvregResidType::parse("").is_err());
        assert!(SurvregResidType::parse("dfb").is_err());
    }

    #[test]
    fn deriv_matches_the_gaussian_closed_form() {
        let gaussian = SurvregDistribution::from_name("gaussian", None).unwrap();
        let z = 0.5;
        let sigma = 2.0;
        let d = survreg_deriv(&gaussian, 1.0 + z * sigma, 0.0, 1, 1.0, sigma);
        assert_close(
            d[0],
            (crate::internal::dist::dnorm(z, false) / sigma).ln(),
            1e-14,
        );
        assert_close(d[1], z / sigma, 1e-14);
        assert_close(d[2], -1.0 / (sigma * sigma), 1e-14);
        assert_close(d[3], z * z - 1.0, 1e-14);
        assert_close(d[4], -2.0 * z * z, 1e-14);
        assert_close(d[5], -2.0 * z / sigma, 1e-14);
    }

    #[test]
    fn deriv_matches_finite_differences_for_censored_rows() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let h = 1e-5;
        for status in [0, 2, 3] {
            let (y1, y2) = (0.4, 1.1);
            let g = |eta: f64, log_sigma: f64| {
                survreg_deriv(&weibull, y1, y2, status, eta, log_sigma.exp())[0]
            };
            let (eta, log_sigma): (f64, f64) = (-0.2, 0.15);
            let d = survreg_deriv(&weibull, y1, y2, status, eta, log_sigma.exp());
            let dg = (g(eta + h, log_sigma) - g(eta - h, log_sigma)) / (2.0 * h);
            let ddg = (g(eta + h, log_sigma) - 2.0 * d[0] + g(eta - h, log_sigma)) / (h * h);
            let ds = (g(eta, log_sigma + h) - g(eta, log_sigma - h)) / (2.0 * h);
            let dds = (g(eta, log_sigma + h) - 2.0 * d[0] + g(eta, log_sigma - h)) / (h * h);
            let h2 = 1e-4;
            let dsg = (g(eta + h2, log_sigma + h2)
                - g(eta + h2, log_sigma - h2)
                - g(eta - h2, log_sigma + h2)
                + g(eta - h2, log_sigma - h2))
                / (4.0 * h2 * h2);
            assert_close(d[1], dg, 1e-6);
            assert_close(d[2], ddg, 1e-4);
            assert_close(d[3], ds, 1e-6);
            assert_close(d[4], dds, 1e-4);
            assert_close(d[5], dsg, 1e-4);
        }
    }

    #[test]
    fn r_sign_is_zero_at_zero() {
        assert_eq!(r_sign(0.0), 0.0);
        assert_eq!(r_sign(-0.0), 0.0);
        assert_eq!(r_sign(-3.0), -1.0);
        assert_eq!(r_sign(2.0), 1.0);
        assert!(r_sign(f64::NAN).is_nan());
    }

    #[test]
    fn collapse_sums_rows_by_sorted_group() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let out = collapse_rows(rows, &[7, 2, 7]).unwrap();
        assert_eq!(out, vec![vec![3.0, 4.0], vec![6.0, 8.0]]);
        assert!(collapse_rows(vec![vec![1.0]], &[1, 2]).is_err());
    }

    #[test]
    fn score_rows_place_the_scale_term_in_the_stratum_column() {
        let row = score_row(&[1.0, 4.0], 2.0, 5.0, 1, 3, true);
        assert_eq!(row, vec![2.0, 8.0, 0.0, 5.0, 0.0]);
        let row = score_row(&[1.0, 4.0], 2.0, 5.0, 1, 3, false);
        assert_eq!(row, vec![2.0, 8.0]);
    }

    #[test]
    fn quadratic_form_uses_the_full_matrix() {
        let vv = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.3, 3.0],
        ];
        let score = [2.0, 8.0, 5.0];
        // score %*% vv = (2 + .8 + 1, .2 + 16 + 1.5, .4 + 2.4 + 15)
        assert_eq!(row_times_matrix(&score, &vv), vec![3.8, 17.7, 17.8]);
        assert_close(
            quadratic_form(&score, &vv),
            2.0 * 3.8 + 8.0 * 17.7 + 5.0 * 17.8,
            1e-12,
        );
    }
}
