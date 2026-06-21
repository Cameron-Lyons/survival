use crate::constants::{
    CHOLESKY_TOL, CONVERGENCE_EPSILON, DEFAULT_MAX_ITER, MAX_HALVING_ITERATIONS, NEAR_ZERO_MATRIX,
    STEP_HALVE_FACTOR,
};
use crate::internal::matrix::regularized_lu_solve;
use crate::regression::survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, compute_linear_predictor,
    compute_quantile_prediction, compute_response_prediction, compute_se_linear_predictor,
};
use crate::regression::survregc1::{SurvivalDist, survregc1};
use crate::residuals::survreg_resid::{
    SurvregResidType, SurvregResiduals, compute_deviance_residuals_survreg, compute_dfbeta_survreg,
    compute_ldcase, compute_response_residuals, compute_response_residuals_censored,
    compute_survreg_dfbeta_residuals, compute_survreg_residual_matrix, compute_working_residuals,
    compute_working_residuals_from_derivative_matrix,
};
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;

type PredictionRows = (Vec<f64>, Option<Vec<Vec<f64>>>);

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvregConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,

    #[pyo3(get, set)]
    pub eps: f64,

    #[pyo3(get, set)]
    pub tol_chol: f64,

    #[pyo3(get, set)]
    pub distribution: DistributionType,
}

#[pymethods]
impl SurvregConfig {
    #[new]
    #[pyo3(signature = (distribution=None, max_iter=None, eps=None, tol_chol=None))]
    fn new(
        distribution: Option<DistributionType>,
        max_iter: Option<usize>,
        eps: Option<f64>,
        tol_chol: Option<f64>,
    ) -> Self {
        Self {
            distribution: distribution.unwrap_or(DistributionType::ExtremeValue),
            max_iter: max_iter.unwrap_or(DEFAULT_MAX_ITER),
            eps: eps.unwrap_or(CONVERGENCE_EPSILON),
            tol_chol: tol_chol.unwrap_or(CHOLESKY_TOL),
        }
    }
}

impl Default for SurvregConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            eps: CONVERGENCE_EPSILON,
            tol_chol: CHOLESKY_TOL,
            distribution: DistributionType::ExtremeValue,
        }
    }
}

impl SurvregConfig {
    pub fn create(
        distribution: Option<DistributionType>,
        max_iter: Option<usize>,
        eps: Option<f64>,
        tol_chol: Option<f64>,
    ) -> Self {
        Self {
            distribution: distribution.unwrap_or(DistributionType::ExtremeValue),
            max_iter: max_iter.unwrap_or(DEFAULT_MAX_ITER),
            eps: eps.unwrap_or(CONVERGENCE_EPSILON),
            tol_chol: tol_chol.unwrap_or(CHOLESKY_TOL),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalFit {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub location_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub scales: Vec<f64>,
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub n_covariates: usize,
    #[pyo3(get)]
    pub n_strata: usize,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub time2: Option<Vec<f64>>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub strata: Vec<usize>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub variance_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub convergence_flag: i32,
    #[pyo3(get)]
    pub score_vector: Vec<f64>,
}

impl DistributionType {
    fn canonical_name(self) -> &'static str {
        match self {
            DistributionType::ExtremeValue => "extreme_value",
            DistributionType::Logistic => "logistic",
            DistributionType::Gaussian => "gaussian",
            DistributionType::Weibull => "weibull",
            DistributionType::LogNormal => "lognormal",
            DistributionType::LogLogistic => "loglogistic",
        }
    }

    fn uses_log_time(self) -> bool {
        matches!(
            self,
            DistributionType::Weibull | DistributionType::LogNormal | DistributionType::LogLogistic
        )
    }
}

fn requested_distribution_name(requested: Option<&str>, distribution: DistributionType) -> String {
    let Some(name) = requested else {
        return distribution.canonical_name().to_string();
    };
    match name.to_lowercase().replace('-', "_").as_str() {
        "exponential" => "exponential".to_string(),
        "normal" => "gaussian".to_string(),
        "log_logistic" => "loglogistic".to_string(),
        "log_gaussian" | "lognormal" | "log_normal" => "lognormal".to_string(),
        "extreme" | "extremevalue" => "extreme_value".to_string(),
        _ => distribution.canonical_name().to_string(),
    }
}

fn parse_distribution_type(distribution: Option<&str>) -> PyResult<DistributionType> {
    let Some(name) = distribution else {
        return Ok(DistributionType::ExtremeValue);
    };
    match name.to_lowercase().replace('-', "_").as_str() {
        "weibull" => Ok(DistributionType::Weibull),
        "exponential" => Ok(DistributionType::Weibull),
        "extreme" | "extreme_value" | "extremevalue" => Ok(DistributionType::ExtremeValue),
        "gaussian" | "normal" => Ok(DistributionType::Gaussian),
        "logistic" => Ok(DistributionType::Logistic),
        "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
            Ok(DistributionType::LogNormal)
        }
        "loglogistic" | "log_logistic" => Ok(DistributionType::LogLogistic),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distribution must be one of weibull, exponential, gaussian, logistic, lognormal, or loglogistic",
        )),
    }
}

impl SurvivalFit {
    fn validate_covariates(&self, covariates: &[Vec<f64>]) -> PyResult<()> {
        for (idx, row) in covariates.iter().enumerate() {
            if row.len() != self.n_covariates {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariate row {} has {} columns but model expects {}",
                    idx,
                    row.len(),
                    self.n_covariates
                )));
            }
            if let Some((col_idx, _)) = row.iter().enumerate().find(|(_, value)| !value.is_finite())
            {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariates[{}][{}] contains non-finite value",
                    idx, col_idx
                )));
            }
        }
        Ok(())
    }

    fn validate_offset(offset: Option<Vec<f64>>, n: usize) -> PyResult<Option<Vec<f64>>> {
        if let Some(values) = offset {
            if values.len() != n {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "offset has {} values but covariates has {} rows",
                    values.len(),
                    n
                )));
            }
            if let Some((idx, _)) = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "offset contains non-finite value at index {}",
                    idx
                )));
            }
            Ok(Some(values))
        } else {
            Ok(None)
        }
    }

    fn location_variance_matrix(&self) -> Vec<Vec<f64>> {
        self.variance_matrix
            .iter()
            .take(self.n_covariates)
            .map(|row| row.iter().take(self.n_covariates).copied().collect())
            .collect()
    }

    fn prediction_rows(
        &self,
        covariates: Option<Vec<Vec<f64>>>,
        offset: Option<Vec<f64>>,
    ) -> PyResult<PredictionRows> {
        if let Some(rows) = covariates {
            self.validate_covariates(&rows)?;
            let offset = Self::validate_offset(offset, rows.len())?;
            let linear_predictors =
                compute_linear_predictor(&rows, &self.location_coefficients, offset.as_deref());
            Ok((linear_predictors, Some(rows)))
        } else {
            if offset.is_some() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "offset can only be supplied with new covariates",
                ));
            }
            Ok((self.linear_predictors.clone(), None))
        }
    }
}

#[pymethods]
impl SurvivalFit {
    #[pyo3(signature = (covariates=None, predict_type="response".to_string(), offset=None, se_fit=false))]
    pub fn predict(
        &self,
        covariates: Option<Vec<Vec<f64>>>,
        predict_type: String,
        offset: Option<Vec<f64>>,
        se_fit: bool,
    ) -> PyResult<SurvregPrediction> {
        let (linear_predictors, rows) = self.prediction_rows(covariates, offset)?;
        let prediction_type =
            crate::regression::survreg_predict::SurvregPredictType::from_str(&predict_type)
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown prediction type: {}. Valid types: response, lp/linear, terms",
                        predict_type
                    ))
                })?;

        let predictions = match prediction_type {
            crate::regression::survreg_predict::SurvregPredictType::Lp
            | crate::regression::survreg_predict::SurvregPredictType::Terms => {
                linear_predictors.clone()
            }
            crate::regression::survreg_predict::SurvregPredictType::Response => {
                compute_response_prediction(&linear_predictors, &self.distribution)
            }
        };

        let se = if se_fit {
            rows.as_ref()
                .map(|values| compute_se_linear_predictor(values, &self.location_variance_matrix()))
        } else {
            None
        };

        Ok(SurvregPrediction {
            n: predictions.len(),
            predictions,
            se,
            prediction_type: predict_type,
        })
    }

    #[pyo3(signature = (covariates=None, quantiles=None, offset=None))]
    pub fn predict_quantile(
        &self,
        covariates: Option<Vec<Vec<f64>>>,
        quantiles: Option<Vec<f64>>,
        offset: Option<Vec<f64>>,
    ) -> PyResult<SurvregQuantilePrediction> {
        let quantiles = quantiles.unwrap_or_else(|| vec![0.5]);
        for &q in &quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Quantiles must be between 0 and 1 (exclusive)",
                ));
            }
        }

        let (linear_predictors, _rows) = self.prediction_rows(covariates, offset)?;
        let predictions = compute_quantile_prediction(
            &linear_predictors,
            self.scale,
            &quantiles,
            &self.distribution,
        );

        Ok(SurvregQuantilePrediction {
            n: predictions.len(),
            quantiles,
            predictions,
        })
    }

    #[pyo3(signature = (residual_type="deviance".to_string()))]
    pub fn residuals(&self, residual_type: String) -> PyResult<SurvregResiduals> {
        let resid_type = SurvregResidType::from_str(&residual_type).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown residual type: {}. Valid types: response, deviance, working, ldcase, ldresp, ldshape, dfbeta, dfbetas, matrix",
                residual_type
            ))
        })?;
        if matches!(
            resid_type,
            SurvregResidType::Dfbeta | SurvregResidType::Dfbetas
        ) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "survreg dfbeta residuals are matrix-valued; use SurvivalFit.dfbeta() or survival.r_api.residuals",
            ));
        }
        if matches!(resid_type, SurvregResidType::Matrix) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "survreg matrix residuals are matrix-valued; use survival.r_api.residuals or survival.survreg_residual_matrix",
            ));
        }
        let has_interval_censoring = self.status.iter().any(|&value| value == 2 || value == 3);
        if has_interval_censoring
            && !matches!(
                resid_type,
                SurvregResidType::Response
                    | SurvregResidType::Deviance
                    | SurvregResidType::Working
                    | SurvregResidType::Ldcase
                    | SurvregResidType::Ldresp
                    | SurvregResidType::Ldshape
            )
        {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                format!(
                    "survreg residual type '{}' is not implemented for left or interval-censored data; use ldcase",
                    residual_type
                ),
            ));
        }

        let residuals = match resid_type {
            SurvregResidType::Response => {
                if has_interval_censoring {
                    compute_response_residuals_censored(
                        &self.time,
                        self.time2.as_deref(),
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                    )?
                } else {
                    compute_response_residuals(
                        &self.time,
                        &self.linear_predictors,
                        &self.distribution,
                    )
                }
            }
            SurvregResidType::Deviance => compute_deviance_residuals_survreg(
                &self.time,
                self.time2.as_deref(),
                &self.status,
                &self.linear_predictors,
                self.scale,
                &self.distribution,
            )?,
            SurvregResidType::Working => {
                if has_interval_censoring {
                    let derivative_matrix = compute_survreg_residual_matrix(
                        &self.time,
                        self.time2.as_deref(),
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                    )?;
                    compute_working_residuals_from_derivative_matrix(&derivative_matrix)?
                } else {
                    compute_working_residuals(
                        &self.time,
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                    )
                }
            }
            SurvregResidType::Ldcase | SurvregResidType::Ldresp | SurvregResidType::Ldshape => {
                compute_ldcase(
                    &self.time,
                    self.time2.as_deref(),
                    &self.status,
                    &self.linear_predictors,
                    self.scale,
                    &self.distribution,
                )?
            }
            SurvregResidType::Dfbeta | SurvregResidType::Dfbetas => unreachable!(),
            SurvregResidType::Matrix => unreachable!(),
        };

        Ok(SurvregResiduals {
            n: residuals.len(),
            residuals,
            residual_type,
        })
    }

    pub fn dfbeta(&self) -> PyResult<Vec<Vec<f64>>> {
        if self.status.iter().any(|&value| value == 2 || value == 3) {
            let derivative_matrix = compute_survreg_residual_matrix(
                &self.time,
                self.time2.as_deref(),
                &self.status,
                &self.linear_predictors,
                self.scale,
                &self.distribution,
            )?;
            return compute_survreg_dfbeta_residuals(
                &derivative_matrix,
                &self.covariates,
                &self.scales,
                &self.strata,
                &self.location_variance_matrix(),
                false,
                false,
            );
        }
        Ok(compute_dfbeta_survreg(
            &self.time,
            &self.status,
            &self.covariates,
            &self.linear_predictors,
            self.scale,
            &self.location_variance_matrix(),
            &self.distribution,
        ))
    }
}
struct LikelihoodInput<'a> {
    n: usize,
    nvar: usize,
    nstrat: usize,
    beta: &'a [f64],
    distribution: &'a DistributionType,
    strata: &'a [usize],
    offsets: &'a Array1<f64>,
    time1: &'a ArrayView1<'a, f64>,
    time2: Option<&'a ArrayView1<'a, f64>>,
    status: &'a ArrayView1<'a, f64>,
    weights: &'a Array1<f64>,
    covariates: &'a Array2<f64>,
}

struct LikelihoodOutput<'a> {
    imat: &'a mut Array2<f64>,
    jj: &'a mut Array2<f64>,
    u: &'a mut Array1<f64>,
}

fn calculate_likelihood(
    input: &LikelihoodInput<'_>,
    output: &mut LikelihoodOutput<'_>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let n = input.n;
    let nvar = input.nvar;
    let nstrat = input.nstrat;
    let beta = input.beta;
    let distribution = input.distribution;
    let strata = input.strata;
    let offsets = input.offsets;
    let time1 = input.time1;
    let time2 = input.time2;
    let status = input.status;
    let weights = input.weights;
    let covariates = input.covariates;
    let imat = &mut *output.imat;
    let jj = &mut *output.jj;
    let u = &mut *output.u;
    let dist = match distribution {
        DistributionType::ExtremeValue => SurvivalDist::ExtremeValue,
        DistributionType::Logistic => SurvivalDist::Logistic,
        DistributionType::Gaussian => SurvivalDist::Gaussian,
        DistributionType::Weibull => SurvivalDist::Weibull,
        DistributionType::LogNormal => SurvivalDist::LogNormal,
        DistributionType::LogLogistic => SurvivalDist::LogLogistic,
    };
    let strat_vec: Vec<i32> = strata.iter().map(|&s| (s + 1) as i32).collect();
    let strat_arr = Array1::from_vec(strat_vec);
    let status_vec: Vec<i32> = status.iter().map(|&s| s as i32).collect();
    let status_arr = Array1::from_vec(status_vec);
    let beta_arr = Array1::from_vec(beta.to_vec());
    let frail_arr = Array1::from_vec(vec![0i32; n]);
    let nvar2 = nvar + nstrat;
    let result = survregc1(
        n,
        nvar,
        nstrat,
        false,
        &beta_arr.view(),
        dist,
        &strat_arr.view(),
        &offsets.view(),
        time1,
        time2,
        &status_arr.view(),
        &weights.view(),
        &covariates.view(),
        0,
        &frail_arr.view(),
    )?;
    let copy_len = nvar2.min(u.len()).min(result.u.len());
    u.iter_mut()
        .zip(result.u.iter())
        .take(copy_len)
        .for_each(|(dest, &src)| *dest = src);

    let copy_rows = nvar2.min(imat.nrows()).min(result.imat.nrows());
    let copy_cols = nvar2.min(imat.ncols()).min(result.imat.ncols());
    imat.slice_mut(ndarray::s![..copy_rows, ..copy_cols])
        .assign(&result.imat.slice(ndarray::s![..copy_rows, ..copy_cols]));

    let copy_rows_jj = nvar2.min(jj.nrows()).min(result.jj.nrows());
    let copy_cols_jj = nvar2.min(jj.ncols()).min(result.jj.ncols());
    jj.slice_mut(ndarray::s![..copy_rows_jj, ..copy_cols_jj])
        .assign(&result.jj.slice(ndarray::s![..copy_rows_jj, ..copy_cols_jj]));
    Ok(result.loglik)
}
fn check_convergence(old: f64, new: f64, eps: f64) -> bool {
    (1.0 - new / old).abs() <= eps || (old - new).abs() <= eps
}
fn adjust_strata(newbeta: &mut [f64], beta: &[f64], nvar: usize, nstrat: usize) {
    newbeta[nvar..nvar + nstrat]
        .iter_mut()
        .zip(&beta[nvar..nvar + nstrat])
        .for_each(|(nb, &b)| {
            if b - *nb > 1.1 {
                *nb = b - 1.1;
            }
        });
}
fn calculate_variance_matrix(
    imat: Array2<f64>,
    _nvar2: usize,
    _tol_chol: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    use crate::internal::matrix::matrix_inverse;
    if imat.nrows() == 0 || imat.ncols() == 0 {
        return Ok(imat);
    }
    let max_val = imat.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_val < NEAR_ZERO_MATRIX {
        return Ok(imat);
    }
    match matrix_inverse(&imat) {
        Some(inv) => Ok(inv),
        None => Ok(imat),
    }
}

fn validate_time_values(time: &[f64]) -> PyResult<()> {
    if time.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time must not be empty",
        ));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time contains non-finite value at index {}",
                idx
            )));
        }
        if value <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time[{}] must be positive",
                idx
            )));
        }
    }
    Ok(())
}

fn validate_status_values(status: &[f64]) -> PyResult<()> {
    for (idx, &value) in status.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "status contains non-finite value at index {}",
                idx
            )));
        }
        if value != 0.0 && value != 1.0 && value != 2.0 && value != 3.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "status must contain only 0/1/2/3 values",
            ));
        }
    }
    Ok(())
}

fn validate_time2_values(
    time: &[f64],
    status: &[f64],
    time2: Option<Vec<f64>>,
) -> PyResult<Option<Vec<f64>>> {
    let has_interval_rows = status.contains(&3.0);
    if !has_interval_rows && time2.is_none() {
        return Ok(None);
    }
    let Some(values) = time2 else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time2 is required for interval-censored rows",
        ));
    };
    if values.len() != time.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} elements but time2 has {}. Both must have the same length.",
            time.len(),
            values.len()
        )));
    }

    let mut sanitized = Vec::with_capacity(values.len());
    for (idx, ((&start, &end), &event)) in time
        .iter()
        .zip(values.iter())
        .zip(status.iter())
        .enumerate()
    {
        if event == 3.0 {
            if !end.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "time2 contains non-finite interval endpoint at index {}",
                    idx
                )));
            }
            if end <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "time2[{}] must be positive",
                    idx
                )));
            }
            if end <= start {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "time2[{}] must be greater than time[{}] for interval-censored rows",
                    idx, idx
                )));
            }
            sanitized.push(end);
        } else {
            // survregc1 ignores time2 unless status == 3; keep the logged array finite.
            sanitized.push(start);
        }
    }
    Ok(Some(sanitized))
}

fn validate_case_weights(weights: &[f64]) -> PyResult<()> {
    let mut has_positive = false;
    for (idx, &value) in weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "weights contains non-finite value at index {}",
                idx
            )));
        }
        if value < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must be non-negative",
            ));
        }
        has_positive |= value > 0.0;
    }
    if !has_positive {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must contain at least one positive value",
        ));
    }
    Ok(())
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{} contains non-finite value at index {}",
                name, idx
            )));
        }
    }
    Ok(())
}

fn validate_covariate_values(covariates: &[Vec<f64>], nvar: usize) -> PyResult<()> {
    for (idx, row) in covariates.iter().enumerate() {
        if row.len() != nvar {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "covariate row {} has {} columns but expected {}",
                idx,
                row.len(),
                nvar
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariates[{}][{}] contains non-finite value",
                    idx, col_idx
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum DistributionType {
    #[pyo3(name = "extreme_value")]
    ExtremeValue,
    #[pyo3(name = "logistic")]
    Logistic,
    #[pyo3(name = "gaussian")]
    Gaussian,
    #[pyo3(name = "weibull")]
    Weibull,
    #[pyo3(name = "lognormal")]
    LogNormal,
    #[pyo3(name = "loglogistic")]
    LogLogistic,
}

/// Fit a parametric survival regression model.
///
/// Parameters
/// ----------
/// time : array-like
///     Exact, right-censoring, left-censoring, or interval lower-bound times.
/// status : array-like
///     Censoring status (0=right censored, 1=exact, 2=left censored, 3=interval censored).
/// covariates : list of lists
///     Covariate matrix (n_obs x n_vars).
/// weights : array-like, optional
///     Case weights.
/// offsets : array-like, optional
///     Offset terms for the linear predictor.
/// initial_beta : array-like, optional
///     Starting values for coefficients.
/// strata : array-like, optional
///     Stratum indicators for stratified analysis.
/// distribution : str, optional
///     Error distribution: "weibull" (default), "lognormal", "loglogistic", "gaussian", "exponential".
/// max_iter : int, optional
///     Maximum iterations (default 30).
/// eps : float, optional
///     Convergence tolerance (default 1e-6).
/// tol_chol : float, optional
///     Cholesky tolerance (default crate::constants::DIVISION_FLOOR).
/// time2 : array-like, optional
///     Interval upper-bound times. Required for rows with status=3.
/// fixed_scale : float, optional
///     Fixed scale parameter. When supplied, the scale is not estimated.
///
/// Returns
/// -------
/// SurvivalFit
///     Object with: coefficients, std_errors, variance_matrix, log_likelihood, convergence info.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, offsets=None, initial_beta=None, strata=None, distribution=None, max_iter=None, eps=None, tol_chol=None, time2=None, fixed_scale=None))]
#[allow(clippy::too_many_arguments)]
pub fn survreg(
    time: Vec<f64>,
    status: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    offsets: Option<Vec<f64>>,
    initial_beta: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    distribution: Option<&str>,
    max_iter: Option<usize>,
    eps: Option<f64>,
    tol_chol: Option<f64>,
    time2: Option<Vec<f64>>,
    fixed_scale: Option<f64>,
) -> PyResult<SurvivalFit> {
    let requested_distribution_key = distribution.map(|name| name.to_lowercase().replace('-', "_"));
    let dist_type = parse_distribution_type(distribution)?;
    let fixed_scale =
        if requested_distribution_key.as_deref() == Some("exponential") && fixed_scale.is_none() {
            Some(1.0)
        } else {
            fixed_scale
        };
    let config = SurvregConfig::create(Some(dist_type), max_iter, eps, tol_chol);
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} elements but status has {}. Both must have the same length.",
            n,
            status.len()
        )));
    }
    validate_time_values(&time)?;
    validate_status_values(&status)?;
    let time2_values = validate_time2_values(&time, &status, time2)?;
    if !config.eps.is_finite() || config.eps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "eps must be a finite positive value",
        ));
    }
    if !config.tol_chol.is_finite() || config.tol_chol <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tol_chol must be a finite positive value",
        ));
    }
    let covariate_rows = covariates;
    let nvar = if !covariate_rows.is_empty() {
        covariate_rows[0].len()
    } else {
        0
    };
    if !covariate_rows.is_empty() && covariate_rows.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} observations but covariates has {} rows. \
             Covariates should be a list of {} rows, each with {} covariate values.",
            n,
            covariate_rows.len(),
            n,
            nvar
        )));
    }
    validate_covariate_values(&covariate_rows, nvar)?;
    let weights_vec = weights.unwrap_or_else(|| vec![1.0; n]);
    let offsets_vec = offsets.unwrap_or_else(|| vec![0.0; n]);
    let has_strata = strata.is_some();
    let strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    if weights_vec.len() != n || offsets_vec.len() != n || strata_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights, offsets, and strata must have the same length as time",
        ));
    }
    validate_case_weights(&weights_vec)?;
    validate_finite_values("offsets", &offsets_vec)?;
    let nstrat = if has_strata {
        strata_vec.iter().max().copied().unwrap_or(0) + 1
    } else {
        1
    };
    if let Some(scale) = fixed_scale {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fixed_scale must be a finite positive value",
            ));
        }
        if nstrat > 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cannot have both a fixed scale and strata",
            ));
        }
    }
    let estimated_scale_count = if fixed_scale.is_some() { 0 } else { nstrat };
    let expected_initial_len = nvar + estimated_scale_count;
    if let Some(values) = initial_beta.as_ref()
        && values.len() != expected_initial_len
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "initial_beta has {} values but model expects {}",
            values.len(),
            expected_initial_len
        )));
    }
    if let Some(values) = initial_beta.as_ref() {
        validate_finite_values("initial_beta", values)?;
    }
    let initial_beta = initial_beta.unwrap_or_else(|| vec![0.0; expected_initial_len]);
    let y = {
        if let Some(time2) = time2_values.as_ref() {
            let mut y_data = Vec::with_capacity(n * 3);
            for i in 0..n {
                y_data.push(time[i]);
                y_data.push(time2[i]);
                y_data.push(status[i]);
            }
            Array2::from_shape_vec((n, 3), y_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            let mut y_data = Vec::with_capacity(n * 2);
            for i in 0..n {
                y_data.push(time[i]);
                y_data.push(status[i]);
            }
            Array2::from_shape_vec((n, 2), y_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        }
    };
    let cov_array = if nvar > 0 {
        let mut flat = Vec::with_capacity(n * nvar);
        for col_idx in 0..nvar {
            flat.extend(covariate_rows.iter().map(|row| row[col_idx]));
        }
        Array2::from_shape_vec((nvar, n), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
    } else {
        Array2::zeros((0, n))
    };
    let weights_arr = Array1::from_vec(weights_vec);
    let offsets_arr = Array1::from_vec(offsets_vec.clone());
    let distribution_type = config.distribution;
    let distribution_name = requested_distribution_name(distribution, distribution_type);
    let result = compute_survreg(ComputeSurvregInput {
        max_iter: config.max_iter,
        nvar,
        y: &y,
        covariates: &cov_array,
        weights: &weights_arr,
        offsets: &offsets_arr,
        beta: initial_beta,
        nstrat,
        strata: &strata_vec,
        eps: config.eps,
        tol_chol: config.tol_chol,
        distribution: distribution_type,
        fixed_scale,
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    let variance_matrix = result
        .variance_matrix
        .outer_iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    let location_coefficients = result.coefficients[..nvar].to_vec();
    let scales: Vec<f64> = if let Some(scale) = fixed_scale {
        vec![scale]
    } else {
        result.coefficients[nvar..nvar + nstrat]
            .iter()
            .map(|value| value.exp())
            .collect()
    };
    let linear_predictors =
        compute_linear_predictor(&covariate_rows, &location_coefficients, Some(&offsets_vec));
    let status_values: Vec<i32> = status.iter().map(|&value| value as i32).collect();
    let fitted_covariates = if nvar == 0 {
        vec![vec![]; n]
    } else {
        covariate_rows
    };
    Ok(SurvivalFit {
        coefficients: if fixed_scale.is_some() {
            location_coefficients.clone()
        } else {
            result.coefficients
        },
        location_coefficients,
        scale: scales.first().copied().unwrap_or(1.0),
        scales,
        distribution: distribution_name,
        n_covariates: nvar,
        n_strata: nstrat,
        linear_predictors,
        time,
        time2: time2_values,
        status: status_values,
        covariates: fitted_covariates,
        strata: strata_vec.clone(),
        weights: weights_arr.to_vec(),
        iterations: result.iterations,
        variance_matrix,
        log_likelihood: result.log_likelihood,
        convergence_flag: result.convergence_flag,
        score_vector: result.score_vector,
    })
}
fn compute_survreg(
    input: ComputeSurvregInput<'_>,
) -> Result<SurvivalFitComputed, Box<dyn std::error::Error>> {
    let ComputeSurvregInput {
        max_iter,
        nvar,
        y,
        covariates,
        weights,
        offsets,
        beta,
        nstrat,
        strata,
        eps,
        tol_chol,
        distribution,
        fixed_scale,
    } = input;
    let n = y.nrows();
    let ny = y.ncols();
    let estimated_scale_count = if fixed_scale.is_some() { 0 } else { nstrat };
    let nvar2 = nvar + estimated_scale_count;
    let mut imat = Array2::zeros((nvar2, nvar2));
    let mut jj = Array2::zeros((nvar2, nvar2));
    let mut u = Array1::zeros(nvar2);
    let mut beta = if let Some(scale) = fixed_scale {
        let mut values = beta;
        values.push(scale.ln());
        values
    } else {
        beta
    };
    let mut usave = Array1::zeros(nvar2);
    let uses_log_time = distribution.uses_log_time();
    let transform_time = |t: f64| if uses_log_time { t.ln() } else { t };
    let time1_vec: Vec<f64> = y.column(0).iter().map(|&t| transform_time(t)).collect();
    let status_vec: Vec<f64> = if ny == 2 {
        y.column(1).iter().copied().collect()
    } else {
        y.column(2).iter().copied().collect()
    };
    let time2_vec: Option<Vec<f64>> = if ny == 3 {
        Some(y.column(1).iter().map(|&t| transform_time(t)).collect())
    } else {
        None
    };
    let time1_arr = Array1::from_vec(time1_vec);
    let status_arr = Array1::from_vec(status_vec);
    let time2_arr = time2_vec.map(Array1::from_vec);
    let time1 = time1_arr.view();
    let status = status_arr.view();
    let time2_view: Option<ArrayView1<f64>> = time2_arr.as_ref().map(|v| v.view());
    let input = LikelihoodInput {
        n,
        nvar,
        nstrat: estimated_scale_count,
        beta: &beta,
        distribution: &distribution,
        strata,
        offsets,
        time1: &time1,
        time2: time2_view.as_ref(),
        status: &status,
        weights,
        covariates,
    };
    let mut output = LikelihoodOutput {
        imat: &mut imat,
        jj: &mut jj,
        u: &mut u,
    };
    let mut loglik = calculate_likelihood(&input, &mut output)?;
    usave.assign(&u);
    let mut iter = 0;
    let mut converged = false;
    while iter < max_iter {
        let old_loglik = loglik;
        let solve_result = regularized_lu_solve(&jj, &u);
        let delta = match solve_result {
            Ok(d) => d,
            Err(_) => regularized_lu_solve(&imat, &u)?,
        };

        let mut accepted = None;
        let mut step_factor = 1.0;
        for _ in 0..=MAX_HALVING_ITERATIONS {
            let mut candidate_beta = beta.clone();
            candidate_beta
                .iter_mut()
                .zip(beta.iter().zip(delta.iter()))
                .for_each(|(nb, (b, d))| *nb = b + d * step_factor);
            adjust_strata(&mut candidate_beta, &beta, nvar, estimated_scale_count);

            let mut candidate_imat = Array2::zeros((nvar2, nvar2));
            let mut candidate_jj = Array2::zeros((nvar2, nvar2));
            let mut candidate_u = Array1::zeros(nvar2);
            let candidate_input = LikelihoodInput {
                n,
                nvar,
                nstrat: estimated_scale_count,
                beta: &candidate_beta,
                distribution: &distribution,
                strata,
                offsets,
                time1: &time1,
                time2: time2_view.as_ref(),
                status: &status,
                weights,
                covariates,
            };
            let mut candidate_output = LikelihoodOutput {
                imat: &mut candidate_imat,
                jj: &mut candidate_jj,
                u: &mut candidate_u,
            };
            let candidate_loglik = calculate_likelihood(&candidate_input, &mut candidate_output)?;
            if candidate_loglik.is_finite() && candidate_loglik >= old_loglik {
                accepted = Some((
                    candidate_beta,
                    candidate_loglik,
                    candidate_imat,
                    candidate_jj,
                    candidate_u,
                ));
                break;
            }
            step_factor *= STEP_HALVE_FACTOR;
        }

        if let Some((candidate_beta, candidate_loglik, candidate_imat, candidate_jj, candidate_u)) =
            accepted
        {
            beta = candidate_beta;
            loglik = candidate_loglik;
            imat = candidate_imat;
            jj = candidate_jj;
            u = candidate_u;
            usave.assign(&u);
            iter += 1;

            if check_convergence(old_loglik, loglik, eps) {
                converged = true;
                break;
            }
        } else {
            break;
        }
    }
    let convergence_flag = if converged { 0 } else { -1 };
    let variance = calculate_variance_matrix(imat, nvar2, tol_chol)?;
    Ok(SurvivalFitComputed {
        coefficients: beta,
        iterations: iter,
        variance_matrix: variance,
        log_likelihood: loglik,
        convergence_flag,
        score_vector: usave.to_vec(),
    })
}
pub(crate) struct SurvivalFitComputed {
    coefficients: Vec<f64>,
    iterations: usize,
    variance_matrix: Array2<f64>,
    log_likelihood: f64,
    convergence_flag: i32,
    score_vector: Vec<f64>,
}

struct ComputeSurvregInput<'a> {
    max_iter: usize,
    nvar: usize,
    y: &'a Array2<f64>,
    covariates: &'a Array2<f64>,
    weights: &'a Array1<f64>,
    offsets: &'a Array1<f64>,
    beta: Vec<f64>,
    nstrat: usize,
    strata: &'a [usize],
    eps: f64,
    tol_chol: f64,
    distribution: DistributionType,
    fixed_scale: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survreg_config_default() {
        let config = SurvregConfig::default();
        assert_eq!(config.max_iter, 30);
        assert!((config.eps - 1e-6).abs() < 1e-10);
        assert!((config.tol_chol - 1e-10).abs() < 1e-15);
        assert_eq!(config.distribution, DistributionType::ExtremeValue);
    }

    #[test]
    fn test_survreg_config_create() {
        let config = SurvregConfig::create(
            Some(DistributionType::Gaussian),
            Some(50),
            Some(1e-8),
            Some(1e-12),
        );
        assert_eq!(config.max_iter, 50);
        assert!((config.eps - 1e-8).abs() < 1e-15);
        assert_eq!(config.distribution, DistributionType::Gaussian);
    }

    #[test]
    fn test_distribution_type_variants() {
        let variants = [
            DistributionType::ExtremeValue,
            DistributionType::Weibull,
            DistributionType::Gaussian,
            DistributionType::Logistic,
            DistributionType::LogNormal,
            DistributionType::LogLogistic,
        ];
        assert_eq!(variants.len(), 6);
    }

    #[test]
    fn test_requested_distribution_name_preserves_response_transform() {
        assert_eq!(
            requested_distribution_name(Some("exponential"), DistributionType::ExtremeValue),
            "exponential"
        );
        assert_eq!(
            requested_distribution_name(Some("normal"), DistributionType::Gaussian),
            "gaussian"
        );
        assert_eq!(
            requested_distribution_name(Some("log-logistic"), DistributionType::LogLogistic),
            "loglogistic"
        );
        assert_eq!(
            requested_distribution_name(Some("extreme"), DistributionType::ExtremeValue),
            "extreme_value"
        );
    }

    #[test]
    fn test_check_convergence() {
        assert!(check_convergence(-100.0, -100.0, 1e-6));
        assert!(check_convergence(-100.0, -100.00001, 1e-4));
        assert!(!check_convergence(-100.0, -99.0, 1e-6));
        assert!(check_convergence(-1e-10, -1e-10, 1e-6));
        assert!(check_convergence(-100.0, -100.0 + 1e-7, 1e-6));
    }

    #[test]
    fn test_adjust_strata() {
        let mut newbeta = vec![1.0, 2.0, 5.0];
        let beta = vec![1.0, 2.0, 3.0];
        adjust_strata(&mut newbeta, &beta, 2, 1);
        assert!(newbeta[2] <= beta[2] - 1.1 + 0.01 || (newbeta[2] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_survreg_simple() {
        let n = 10;
        let nvar = 1;
        let y = Array2::from_shape_vec(
            (n, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0, 1.0, 7.0, 1.0, 8.0, 1.0,
                9.0, 1.0, 10.0, 1.0,
            ],
        )
        .unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0, 0.0];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::Weibull,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2);
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn test_compute_survreg_convergence() {
        let n = 20;
        let nvar = 1;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5).collect();
        let y_data: Vec<f64> = times.iter().flat_map(|&t| vec![t, 1.0]).collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0, 0.0];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::Weibull,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert!(fit.log_likelihood.is_finite());
        assert!(fit.iterations <= 100);
    }

    #[test]
    fn test_compute_survreg_lognormal() {
        let n = 20;
        let nvar = 1;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5).collect();
        let y_data: Vec<f64> = times.iter().flat_map(|&t| vec![t, 1.0]).collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0, 0.0];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::LogNormal,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2);
    }

    #[test]
    fn test_compute_survreg_loglogistic() {
        let n = 20;
        let nvar = 1;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5).collect();
        let y_data: Vec<f64> = times.iter().flat_map(|&t| vec![t, 1.0]).collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0, 0.0];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::LogLogistic,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2);
    }

    #[test]
    fn test_compute_survreg_with_censoring() {
        let n = 20;
        let nvar = 1;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.5).collect();
        let statuses: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
        let y_data: Vec<f64> = times
            .iter()
            .zip(statuses.iter())
            .flat_map(|(&t, &s)| vec![t, s])
            .collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0, 0.0];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::Weibull,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2);
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn test_compute_survreg_multiple_covariates() {
        let n = 30;
        let nvar = 3;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.3).collect();
        let y_data: Vec<f64> = times.iter().flat_map(|&t| vec![t, 1.0]).collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let cov_data: Vec<f64> = (0..nvar * n)
            .map(|i| ((i % 7) as f64 - 3.0) / 3.0)
            .collect();
        let covariates = Array2::from_shape_vec((nvar, n), cov_data).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0; nvar + 1];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::Weibull,
            fixed_scale: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), nvar + 1);
    }

    #[test]
    fn test_compute_survreg_fixed_scale_uses_location_variance() {
        let n = 20;
        let nvar = 1;
        let times: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.4 + 0.5).collect();
        let y_data: Vec<f64> = times.iter().flat_map(|&t| vec![t, 1.0]).collect();
        let y = Array2::from_shape_vec((n, 2), y_data).unwrap();
        let covariates = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let weights = Array1::from_vec(vec![1.0; n]);
        let offsets = Array1::from_vec(vec![0.0; n]);
        let beta = vec![0.0; nvar];
        let strata = vec![0; n];

        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 20,
            nvar,
            y: &y,
            covariates: &covariates,
            weights: &weights,
            offsets: &offsets,
            beta,
            nstrat: 1,
            strata: &strata,
            eps: 1e-6,
            tol_chol: 1e-10,
            distribution: DistributionType::Weibull,
            fixed_scale: Some(1.25),
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), nvar + 1);
        assert_eq!(fit.score_vector.len(), nvar);
        assert_eq!(fit.variance_matrix.shape(), &[nvar, nvar]);
        assert_eq!(fit.coefficients[nvar], 1.25f64.ln());
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn test_survival_fit_fields() {
        let fit = SurvivalFitComputed {
            coefficients: vec![1.0, 0.5],
            iterations: 10,
            variance_matrix: Array2::zeros((2, 2)),
            log_likelihood: -50.0,
            convergence_flag: 0,
            score_vector: vec![0.001, 0.002],
        };

        assert_eq!(fit.coefficients.len(), 2);
        assert_eq!(fit.iterations, 10);
        assert_eq!(fit.convergence_flag, 0);
        assert!((fit.log_likelihood - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_variance_matrix_empty() {
        let imat = Array2::zeros((0, 0));
        let result = calculate_variance_matrix(imat, 0, crate::constants::DIVISION_FLOOR);
        assert!(result.is_ok());
        let var = result.unwrap();
        assert_eq!(var.nrows(), 0);
        assert_eq!(var.ncols(), 0);
    }

    #[test]
    fn test_calculate_variance_matrix_small() {
        let mut imat = Array2::zeros((2, 2));
        imat[[0, 0]] = 2.0;
        imat[[1, 1]] = 2.0;
        imat[[0, 1]] = 0.5;
        imat[[1, 0]] = 0.5;
        let result = calculate_variance_matrix(imat, 2, crate::constants::DIVISION_FLOOR);
        assert!(result.is_ok());
        let var = result.unwrap();
        assert_eq!(var.nrows(), 2);
        assert_eq!(var.ncols(), 2);
    }
}
