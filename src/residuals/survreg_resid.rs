use crate::core::strata_order::DenseGroups;
use crate::error::SurvivalResult;
use crate::internal::match_arg::match_arg;
use crate::internal::validation::validate_length;
use crate::regression::parametric_survival::SurvregFit;
use crate::regression::survreg_distributions::{
    SurvregDensity, SurvregDistribution, SurvregFamily,
};
use crate::regression::survreg_kernels::{BuiltinKernel, with_builtin_kernel};
use ndarray::{Array2, ArrayView1};
use pyo3::prelude::*;

#[doc = " The `type` argument of `residuals.survreg`."]
#[pyclass(module = "survival._survival", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregResidType {
    #[doc = " `itrans(center of the saturated model) - itrans(eta)`."]
    Response,
    Deviance,
    #[doc = " Per-coefficient influence, one column per coefficient (plus one per"]
    #[doc = " `Log(scale)` when `rsigma`)."]
    Dfbeta,
    #[doc = " `Dfbeta` scaled by the coefficient standard errors."]
    Dfbetas,
    Working,
    #[doc = " Likelihood displacement from case deletion."]
    Ldcase,
    #[doc = " Likelihood displacement from a response perturbation."]
    Ldresp,
    #[doc = " Likelihood displacement from a shape (scale) perturbation."]
    Ldshape,
    #[doc = " The raw derivative matrix `(g, dg, ddg, ds, dds, dsg)`."]
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

    pub fn parse(name: &str) -> SurvivalResult<Self> {
        let index = match_arg(name, &Self::CHOICES.map(|(choice, _)| choice))?;
        Ok(Self::CHOICES[index].1)
    }

    pub fn is_matrix(self) -> bool {
        matches!(self, Self::Dfbeta | Self::Dfbetas | Self::Matrix)
    }
}

#[doc = " Residuals of a `survreg` fit."]
#[pyclass(module = "survival._survival", from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregResiduals {
    #[pyo3(get)]
    pub residual_type: SurvregResidType,
    #[doc = " One row per observation (per group after `collapse`).  Vector-valued"]
    #[doc = " types have a single column; `Dfbeta`/`Dfbetas` have one column per"]
    #[doc = " coefficient followed by one per `Log(scale)` when `rsigma`; `Matrix`"]
    #[doc = " has the columns `g, dg, ddg, ds, dds, dsg`."]
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

/// R's `survreg` derivative matrix for one observation under a builtin family:
/// `g, dg, ddg, ds, dds, dsg`.  Generic over the family so that loops over the
/// observations compile the family's density inline.
#[inline(always)]
pub(crate) fn survreg_deriv<K: BuiltinKernel>(
    kernel: K,
    y1: f64,
    y2: f64,
    status: i32,
    eta: f64,
    sigma: f64,
) -> [f64; 6] {
    let z = (y1 - eta) / sigma;
    let dmat = kernel.density(z);
    let (z2, dmat2) = if status == 3 {
        let z2 = (y2 - eta) / sigma;
        (z2, kernel.density(z2))
    } else {
        (0.0, dmat)
    };
    deriv_from_density(dmat, dmat2, z, z2, status, sigma)
}

/// The `dg` and `ddg` columns of [`survreg_deriv`] alone, which is all that
/// `survreg`'s starting values read: the logarithm and the scale derivatives
/// are not computed.
#[inline(always)]
pub(crate) fn location_deriv<K: BuiltinKernel>(
    kernel: K,
    y1: f64,
    y2: f64,
    status: i32,
    eta: f64,
    sigma: f64,
) -> (f64, f64) {
    let z = (y1 - eta) / sigma;
    let dmat = kernel.density(z);
    let dmat2 = if status == 3 {
        kernel.density((y2 - eta) / sigma)
    } else {
        dmat
    };
    let (_, dg, ddg) = LocationTerms::new(dmat, dmat2, z, status).derivatives(sigma);
    (dg, ddg - dg * dg)
}

/// What every column of the derivative matrix is built from: the probability
/// `tdenom` of a censored observation (1 for an exact one), the numerators of
/// `dg` and `ddg`, and the density derivative `f'` at each end of the interval.
struct LocationTerms {
    tdenom: f64,
    numerator_dg: f64,
    numerator_ddg: f64,
    dtemp: f64,
    dtemp2: f64,
}

impl LocationTerms {
    #[inline(always)]
    fn new(dmat: SurvregDensity, dmat2: SurvregDensity, z: f64, status: i32) -> Self {
        let dtemp = dmat.distribution_kernel()[3];
        let dtemp2 = dmat2.distribution_kernel()[3];
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
                dtemp2 - dtemp,
            ),
        };
        Self {
            tdenom,
            numerator_dg,
            numerator_ddg,
            dtemp,
            dtemp2,
        }
    }

    /// `1 / tdenom`, `dg` and the uncentred `ddg` at scale `sigma`.
    #[inline(always)]
    fn derivatives(&self, sigma: f64) -> (f64, f64, f64) {
        let tdenom = 1.0 / self.tdenom;
        let dg = -(tdenom / sigma) * self.numerator_dg;
        let ddg = (tdenom / (sigma * sigma)) * self.numerator_ddg;
        (tdenom, dg, ddg)
    }
}

fn deriv_from_density(
    dmat: SurvregDensity,
    dmat2: SurvregDensity,
    z: f64,
    z2: f64,
    status: i32,
    sigma: f64,
) -> [f64; 6] {
    let terms = LocationTerms::new(dmat, dmat2, z, status);
    let g = if status == 1 {
        (dmat.pdf / sigma).ln()
    } else {
        terms.tdenom.ln()
    };
    let (tdenom, dg, ddg) = terms.derivatives(sigma);
    let (dtemp, dtemp2) = (terms.dtemp, terms.dtemp2);
    let (ds, dds, dsg) = if status < 3 {
        (dg * sigma * z, ddg * (sigma * z).powi(2), ddg * sigma * z)
    } else {
        (
            tdenom * (z * dmat.pdf - z2 * dmat2.pdf),
            tdenom * (z2 * z2 * dtemp2 - z * z * dtemp),
            tdenom * (z2 * dtemp2 - z * dtemp) / sigma,
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

pub(crate) fn survreg_derivatives(
    distribution: &SurvregDistribution,
    y1: &[f64],
    y2: &[f64],
    status: &[i32],
    eta: &[f64],
    scale: &[f64],
) -> SurvivalResult<Vec<[f64; 6]>> {
    if distribution.family != SurvregFamily::Custom && distribution.callbacks.is_none() {
        return Ok(with_builtin_kernel!(distribution, kernel => {
            (0..y1.len())
                .map(|i| survreg_deriv(kernel, y1[i], y2[i], status[i], eta[i], scale[i]))
                .collect()
        }));
    }
    let n = y1.len();
    let mut z = Vec::with_capacity(n + status.iter().filter(|&&s| s == 3).count());
    z.extend((0..n).map(|i| (y1[i] - eta[i]) / scale[i]));
    z.extend(
        (0..n)
            .filter(|&i| status[i] == 3)
            .map(|i| (y2[i] - eta[i]) / scale[i]),
    );
    let values = distribution.density_batch(&z)?;
    let mut upper = n;
    Ok((0..n)
        .map(|i| {
            let (z2, d2) = if status[i] == 3 {
                let j = upper;
                upper += 1;
                (z[j], values[j])
            } else {
                (0.0, values[i])
            };
            deriv_from_density(values[i], d2, z[i], z2, status[i], scale[i])
        })
        .collect())
}

fn r_sign(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value.signum() }
}

/// `out = score %*% matrix`, for the leading `out.len()` columns.
fn row_times_matrix_into(out: &mut [f64], score: &[f64], matrix: &[Vec<f64>]) {
    // Each output column accumulates in score order while each covariance
    // row is read contiguously. Match the initial value of f64::sum.
    out.fill(-0.0);
    for (&s, row) in score.iter().zip(matrix) {
        for (value, &v) in out.iter_mut().zip(row) {
            *value += s * v;
        }
    }
}

/// Writes one observation's score row into `row`: `eta_part` times its
/// covariates, then, when the row has `Log(scale)` columns, `scale_part` in its
/// stratum's column and zero in the others.
fn fill_score_row(
    row: &mut [f64],
    x: ArrayView1<'_, f64>,
    eta_part: f64,
    scale_part: f64,
    stratum: usize,
) {
    let (location, scales) = row.split_at_mut(x.len());
    for (out, value) in location.iter_mut().zip(x) {
        *out = eta_part * value;
    }
    if !scales.is_empty() {
        scales.fill(0.0);
        scales[stratum] = scale_part;
    }
}

/// Rows summed by `collapse` group as R's `rowsum` sums them: one row per
/// distinct code in ascending order, each adding its rows in observation order.
///
/// A row is added straight into its group's sum, found by the code's rank, so
/// no sorted copy of the rows is made.
struct CollapsedRows {
    ranks: Vec<usize>,
    sums: Array2<f64>,
}

impl CollapsedRows {
    fn new(collapse: &[usize], rows: usize, width: usize) -> SurvivalResult<Self> {
        validate_length(rows, collapse.len(), "collapse")?;
        let groups = DenseGroups::from_usize(collapse);
        Ok(Self {
            ranks: groups.ids,
            sums: Array2::zeros((groups.count, width)),
        })
    }

    fn add(&mut self, row: usize, values: &[f64]) {
        for (total, value) in self.sums.row_mut(self.ranks[row]).iter_mut().zip(values) {
            *total += value;
        }
    }
}

/// The variance the influence residuals are scaled by (the model-based one of a
/// robust fit), and the number of location and requested scale columns.
fn influence_variance(fit: &SurvregFit, rsigma: bool) -> SurvivalResult<(&[Vec<f64>], usize)> {
    let nvar = fit.nvar();
    let rsigma = rsigma && fit.variance_matrix.len() != nvar;
    let vv = fit
        .naive_variance_matrix
        .as_deref()
        .unwrap_or(&fit.variance_matrix);
    let width = nvar + if rsigma { fit.nstrata() } else { 0 };
    fit.check_covariance(vv)?;
    Ok((vv, width))
}

/// The robust covariance consumes weighted dfbeta rows from the same calculation
/// as the public residuals. Collapsed rows arrive in sorted group order.
pub(crate) fn visit_weighted_dfbeta(
    fit: &SurvregFit,
    cluster: Option<&[usize]>,
    visit: impl FnMut(&[f64]),
) -> SurvivalResult<()> {
    visit_residuals(fit, SurvregResidType::Dfbeta, true, cluster, true, visit)
}

pub fn residuals_survreg(
    fit: &SurvregFit,
    residual_type: SurvregResidType,
    rsigma: bool,
    collapse: Option<&[usize]>,
    weighted: bool,
) -> SurvivalResult<SurvregResiduals> {
    let mut values = Vec::new();
    visit_residuals(fit, residual_type, rsigma, collapse, weighted, |row| {
        values.push(row.to_vec());
    })?;
    Ok(SurvregResiduals {
        residual_type,
        values,
    })
}

fn visit_residuals(
    fit: &SurvregFit,
    residual_type: SurvregResidType,
    rsigma: bool,
    collapse: Option<&[usize]>,
    weighted: bool,
    mut visit: impl FnMut(&[f64]),
) -> SurvivalResult<()> {
    use SurvregResidType::*;
    fit.check_residual_inputs()?;
    let (variance, width) = influence_variance(fit, rsigma)?;
    let output_width = match residual_type {
        Dfbeta | Dfbetas => width,
        Matrix => 6,
        _ => 1,
    };
    let mut collapsed = collapse
        .map(|groups| CollapsedRows::new(groups, fit.n, output_width))
        .transpose()?;
    let weights = if weighted {
        fit.weights.as_deref()
    } else {
        None
    };
    let mut emit = |i: usize, row: &mut [f64]| {
        if let Some(weights) = weights {
            row.iter_mut().for_each(|v| *v *= weights[i]);
        }
        match &mut collapsed {
            Some(groups) => groups.add(i, row),
            None => visit(row),
        }
    };

    let distribution = &fit.distribution;
    let (y1, y2) =
        distribution.transformed_endpoints(&fit.time, fit.time2.as_deref(), &fit.status)?;
    let sigma: Vec<_> = fit.strata.iter().map(|&s| fit.scale[s]).collect();
    let eta = &fit.linear_predictors;
    if residual_type == Response {
        let (centers, _) = distribution.deviance_batch(&y1, &y2, &fit.status, &sigma)?;
        let centers = distribution.inverse_values(&centers)?;
        let predictions = distribution.inverse_values(eta)?;
        for (i, (center, fitted)) in centers.iter().zip(predictions).enumerate() {
            emit(i, &mut [center - fitted]);
        }
    } else {
        let deriv = survreg_derivatives(distribution, &y1, &y2, &fit.status, eta, &sigma)?;
        match residual_type {
            Deviance => {
                let (_, saturated) = distribution.deviance_batch(&y1, &y2, &fit.status, &sigma)?;
                for (i, (d, sat)) in deriv.iter().zip(saturated).enumerate() {
                    let working = -d[1] / d[2];
                    emit(
                        i,
                        &mut [r_sign(working) * (2.0 * (sat - d[0])).max(0.0).sqrt()],
                    );
                }
            }
            Working => {
                for (i, d) in deriv.iter().enumerate() {
                    emit(i, &mut [-d[1] / d[2]]);
                }
            }
            Matrix => {
                for (i, mut d) in deriv.into_iter().enumerate() {
                    emit(i, &mut d);
                }
            }
            Dfbeta | Dfbetas | Ldcase | Ldresp | Ldshape => {
                let standardize = (residual_type == Dfbetas).then(|| {
                    (0..width)
                        .map(|j| 1.0 / variance[j][j].sqrt())
                        .collect::<Vec<_>>()
                });
                let mut score = vec![0.0; width];
                let mut row = vec![0.0; width];
                for (i, d) in deriv.iter().enumerate() {
                    let (location, scale) = match residual_type {
                        Ldresp => (d[2] * sigma[i], d[5] * sigma[i]),
                        Ldshape => (d[5], d[4]),
                        _ => (d[1], d[3]),
                    };
                    fill_score_row(
                        &mut score,
                        fit.covariates.row(i),
                        location,
                        scale,
                        fit.strata[i],
                    );
                    row_times_matrix_into(&mut row, &score, variance);
                    if let Some(factors) = &standardize {
                        for (entry, factor) in row.iter_mut().zip(factors) {
                            *entry *= factor;
                        }
                    }
                    if matches!(residual_type, Dfbeta | Dfbetas) {
                        emit(i, &mut row);
                    } else {
                        let displacement = row.iter().zip(&score).map(|(v, s)| v * s).sum();
                        emit(i, &mut [displacement]);
                    }
                }
            }
            Response => unreachable!("handled above"),
        }
    }
    if let Some(groups) = collapsed {
        for row in groups.sums.outer_iter() {
            visit(row.as_slice().expect("standard layout"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod reference_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::survreg_kernels::{ExtremeValue, Gaussian};

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
        assert!(SurvregResidType::parse("Matrix").is_err());
        assert!(SurvregResidType::parse("ld").is_err());
        assert!(SurvregResidType::parse("").is_err());
        assert!(SurvregResidType::parse("dfb").is_err());
    }

    #[test]
    fn deriv_matches_the_gaussian_closed_form() {
        let z = 0.5;
        let sigma = 2.0;
        let d = survreg_deriv(Gaussian, 1.0 + z * sigma, 0.0, 1, 1.0, sigma);
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
        let h = 1e-5;
        for status in [0, 2, 3] {
            let (y1, y2) = (0.4, 1.1);
            let g = |eta: f64, log_sigma: f64| {
                survreg_deriv(ExtremeValue, y1, y2, status, eta, log_sigma.exp())[0]
            };
            let (eta, log_sigma): (f64, f64) = (-0.2, 0.15);
            let d = survreg_deriv(ExtremeValue, y1, y2, status, eta, log_sigma.exp());
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
        let mut groups = CollapsedRows::new(&[7, 2, 7], 3, 2).unwrap();
        groups.add(0, &[1.0, 2.0]);
        groups.add(1, &[3.0, 4.0]);
        groups.add(2, &[5.0, 6.0]);
        assert_eq!(groups.sums, ndarray::array![[3.0, 4.0], [6.0, 8.0]]);
        assert!(CollapsedRows::new(&[1, 2], 1, 1).is_err());
    }

    #[test]
    fn score_rows_place_the_scale_term_in_the_stratum_column() {
        let mut row = [0.0; 5];
        let x = ArrayView1::from(&[1.0, 4.0]);
        fill_score_row(&mut row, x, 2.0, 5.0, 1);
        assert_eq!(row, [2.0, 8.0, 0.0, 5.0, 0.0]);
        fill_score_row(&mut row[..2], x, 2.0, 5.0, 1);
        assert_eq!(row[..2], [2.0, 8.0]);
    }

    #[test]
    fn score_product_uses_the_full_matrix_in_score_order() {
        let vv = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.3, 3.0],
        ];
        let mut row = [f64::NAN; 3];
        row_times_matrix_into(&mut row, &[2.0, 8.0, 5.0], &vv);
        assert_eq!(row, [3.8, 17.7, 17.8]);
    }
}
