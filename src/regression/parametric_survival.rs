use crate::constants::{
    CHOLESKY_TOL, CONVERGENCE_EPSILON, DEFAULT_MAX_ITER, MAX_HALVING_ITERATIONS, NEAR_ZERO_MATRIX,
    STEP_HALVE_FACTOR,
};
use crate::internal::matrix::regularized_lu_solve;
use crate::regression::survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, compute_linear_predictor,
    compute_quantile_prediction, compute_response_prediction, compute_se_linear_predictor,
};
use crate::regression::survregc1::{SurvivalDist, SurvivalLikelihood, survregc1};
use crate::residuals::survreg_resid::{
    SurvregResidType, SurvregResiduals, compute_deviance_residuals_survreg_with_parameter,
    compute_dfbeta_survreg_with_parameter, compute_ldcase_with_parameter,
    compute_response_residuals, compute_response_residuals_censored_with_parameter,
    compute_survreg_dfbeta_residuals, compute_survreg_residual_matrix_with_parameter,
    compute_working_residuals_from_derivative_matrix, compute_working_residuals_with_parameter,
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
            distribution: distribution.unwrap_or(DEFAULT_SURVREG_DISTRIBUTION),
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
            distribution: DEFAULT_SURVREG_DISTRIBUTION,
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
            distribution: distribution.unwrap_or(DEFAULT_SURVREG_DISTRIBUTION),
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
    pub distribution_parameters: Vec<f64>,
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
    #[pyo3(get)]
    pub penalty_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub penalty: f64,
    #[pyo3(get)]
    pub penalized_log_likelihood: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: Option<f64>,
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
            DistributionType::StudentT => "t",
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
        "rayleigh" => "rayleigh".to_string(),
        "normal" => "gaussian".to_string(),
        "log_logistic" => "loglogistic".to_string(),
        "loggaussian" | "log_gaussian" | "lognormal" | "log_normal" => "lognormal".to_string(),
        "extreme" | "extremevalue" => "extreme_value".to_string(),
        "student" | "student_t" | "studentt" => "t".to_string(),
        _ => distribution.canonical_name().to_string(),
    }
}

fn is_student_t_distribution_name(distribution: &str) -> bool {
    matches!(
        distribution.to_lowercase().replace('-', "_").as_str(),
        "t" | "student" | "student_t" | "studentt"
    )
}

fn parse_distribution_type(distribution: Option<&str>) -> PyResult<DistributionType> {
    let Some(name) = distribution else {
        return Ok(DEFAULT_SURVREG_DISTRIBUTION);
    };
    match name.to_lowercase().replace('-', "_").as_str() {
        "weibull" => Ok(DistributionType::Weibull),
        "exponential" => Ok(DistributionType::Weibull),
        "rayleigh" => Ok(DistributionType::Weibull),
        "extreme" | "extreme_value" | "extremevalue" => Ok(DistributionType::ExtremeValue),
        "gaussian" | "normal" => Ok(DistributionType::Gaussian),
        "logistic" => Ok(DistributionType::Logistic),
        "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
            Ok(DistributionType::LogNormal)
        }
        "loglogistic" | "log_logistic" => Ok(DistributionType::LogLogistic),
        "t" | "student" | "student_t" | "studentt" => Ok(DistributionType::StudentT),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distribution must be one of weibull, exponential, rayleigh, extreme, gaussian, logistic, loggaussian, lognormal, loglogistic, or t",
        )),
    }
}

fn validate_distribution_parameter(
    distribution: DistributionType,
    distribution_parameter: Option<f64>,
) -> PyResult<Option<f64>> {
    match distribution {
        DistributionType::StudentT => {
            let df = distribution_parameter.unwrap_or(4.0);
            if !df.is_finite() || df <= 2.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Degrees of freedom must be >=3",
                ));
            }
            Ok(Some(df))
        }
        _ => {
            if distribution_parameter.is_some() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "distribution_parameter is only supported for distribution='t'",
                ));
            }
            Ok(None)
        }
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

    fn distribution_parameter(&self) -> Option<f64> {
        self.distribution_parameters.first().copied()
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
            if !q.is_finite() || q <= 0.0 || q >= 1.0 {
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
                    compute_response_residuals_censored_with_parameter(
                        &self.time,
                        self.time2.as_deref(),
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                        self.distribution_parameter(),
                    )?
                } else {
                    compute_response_residuals(
                        &self.time,
                        &self.linear_predictors,
                        &self.distribution,
                    )
                }
            }
            SurvregResidType::Deviance => compute_deviance_residuals_survreg_with_parameter(
                &self.time,
                self.time2.as_deref(),
                &self.status,
                &self.linear_predictors,
                self.scale,
                &self.distribution,
                self.distribution_parameter(),
            )?,
            SurvregResidType::Working => {
                if has_interval_censoring || is_student_t_distribution_name(&self.distribution) {
                    let derivative_matrix = compute_survreg_residual_matrix_with_parameter(
                        &self.time,
                        self.time2.as_deref(),
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                        self.distribution_parameter(),
                    )?;
                    compute_working_residuals_from_derivative_matrix(&derivative_matrix)?
                } else {
                    compute_working_residuals_with_parameter(
                        &self.time,
                        &self.status,
                        &self.linear_predictors,
                        self.scale,
                        &self.distribution,
                        self.distribution_parameter(),
                    )
                }
            }
            SurvregResidType::Ldcase | SurvregResidType::Ldresp | SurvregResidType::Ldshape => {
                compute_ldcase_with_parameter(
                    &self.time,
                    self.time2.as_deref(),
                    &self.status,
                    &self.linear_predictors,
                    self.scale,
                    &self.distribution,
                    self.distribution_parameter(),
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
            let derivative_matrix = compute_survreg_residual_matrix_with_parameter(
                &self.time,
                self.time2.as_deref(),
                &self.status,
                &self.linear_predictors,
                self.scale,
                &self.distribution,
                self.distribution_parameter(),
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
        Ok(compute_dfbeta_survreg_with_parameter(
            &self.time,
            &self.status,
            &self.covariates,
            &self.linear_predictors,
            self.scale,
            &self.location_variance_matrix(),
            &self.distribution,
            self.distribution_parameter(),
        ))
    }
}
struct LikelihoodInput<'a> {
    n: usize,
    nvar: usize,
    nstrat: usize,
    beta: &'a [f64],
    distribution: &'a DistributionType,
    distribution_parameter: Option<f64>,
    strata: &'a ArrayView1<'a, i32>,
    offsets: &'a Array1<f64>,
    time1: &'a ArrayView1<'a, f64>,
    time2: Option<&'a ArrayView1<'a, f64>>,
    status: &'a ArrayView1<'a, i32>,
    weights: &'a Array1<f64>,
    covariates: &'a Array2<f64>,
    frailty: &'a ArrayView1<'a, i32>,
}

fn calculate_likelihood(
    input: &LikelihoodInput<'_>,
) -> Result<SurvivalLikelihood, Box<dyn std::error::Error>> {
    let dist = match input.distribution {
        DistributionType::ExtremeValue => SurvivalDist::ExtremeValue,
        DistributionType::Logistic => SurvivalDist::Logistic,
        DistributionType::Gaussian => SurvivalDist::Gaussian,
        DistributionType::Weibull => SurvivalDist::Weibull,
        DistributionType::LogNormal => SurvivalDist::LogNormal,
        DistributionType::LogLogistic => SurvivalDist::LogLogistic,
        DistributionType::StudentT => SurvivalDist::StudentT(
            input
                .distribution_parameter
                .ok_or_else(|| "Student-t degrees of freedom are missing".to_string())?,
        ),
    };
    let beta = ArrayView1::from(input.beta);
    survregc1(
        input.n,
        input.nvar,
        input.nstrat,
        false,
        &beta,
        dist,
        input.strata,
        &input.offsets.view(),
        input.time1,
        input.time2,
        input.status,
        &input.weights.view(),
        &input.covariates.view(),
        0,
        input.frailty,
    )
}
fn check_convergence(old: f64, new: f64, eps: f64) -> bool {
    (1.0 - new / old).abs() <= eps || (old - new).abs() <= eps
}

fn is_positive_definite(matrix: &Array2<f64>, tolerance: f64) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    let size = matrix.nrows();
    let scale = matrix
        .diag()
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let threshold = tolerance * scale;
    let mut lower = Array2::<f64>::zeros((size, size));
    for row in 0..size {
        for column in 0..=row {
            let product_sum = (0..column)
                .map(|index| lower[[row, index]] * lower[[column, index]])
                .sum::<f64>();
            if row == column {
                let pivot = matrix[[row, row]] - product_sum;
                if !pivot.is_finite() || pivot <= threshold {
                    return false;
                }
                lower[[row, column]] = pivot.sqrt();
            } else {
                lower[[row, column]] =
                    (matrix[[row, column]] - product_sum) / lower[[column, column]];
            }
        }
    }
    true
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

fn validate_time_values(time: &[f64], require_positive: bool) -> PyResult<()> {
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
        if require_positive && value <= 0.0 {
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
    require_positive: bool,
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
            if require_positive && end <= 0.0 {
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

fn validate_penalty_matrix(values: &[Vec<f64>], nvar: usize) -> PyResult<Array2<f64>> {
    if values.len() != nvar || values.iter().any(|row| row.len() != nvar) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "penalty_matrix must have shape ({nvar}, {nvar})"
        )));
    }
    for (row_index, row) in values.iter().enumerate() {
        validate_finite_values(&format!("penalty_matrix[{row_index}]"), row)?;
    }

    let scale = values
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 1e-10 * scale.max(f64::MIN_POSITIVE);
    let mut matrix = Array2::zeros((nvar, nvar));
    for row in 0..nvar {
        for column in 0..nvar {
            let left = values[row][column];
            let right = values[column][row];
            if (left - right).abs() > tolerance {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "penalty_matrix must be symmetric",
                ));
            }
            matrix[(row, column)] = 0.5 * (left + right);
        }
    }

    // A diagonally pivoted LDL decomposition admits rank-deficient difference
    // penalties while still rejecting indefinite quadratic forms.
    let mut factor = matrix.clone();
    for pivot_index in 0..nvar {
        let largest_diagonal = (pivot_index..nvar)
            .max_by(|&left, &right| factor[(left, left)].total_cmp(&factor[(right, right)]))
            .unwrap_or(pivot_index);
        if largest_diagonal != pivot_index {
            for column in 0..nvar {
                factor.swap((pivot_index, column), (largest_diagonal, column));
            }
            for row in 0..nvar {
                factor.swap((row, pivot_index), (row, largest_diagonal));
            }
        }
        let pivot = factor[(pivot_index, pivot_index)];
        if pivot < -tolerance {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "penalty_matrix must be positive semidefinite",
            ));
        }
        if pivot <= tolerance {
            if (pivot_index..nvar).any(|row| {
                (pivot_index..nvar).any(|column| factor[(row, column)].abs() > tolerance)
            }) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "penalty_matrix must be positive semidefinite",
                ));
            }
            continue;
        }
        for row in (pivot_index + 1)..nvar {
            for column in row..nvar {
                let updated = factor[(row, column)]
                    - factor[(row, pivot_index)] * factor[(column, pivot_index)] / pivot;
                factor[(row, column)] = updated;
                factor[(column, row)] = updated;
            }
        }
    }
    Ok(matrix)
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
    #[pyo3(name = "t")]
    StudentT,
}

const DEFAULT_SURVREG_DISTRIBUTION: DistributionType = DistributionType::Weibull;

#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, offsets=None, initial_beta=None, strata=None, distribution=None, max_iter=None, eps=None, tol_chol=None, time2=None, fixed_scale=None, distribution_parameter=None, penalty_matrix=None))]
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
    distribution_parameter: Option<f64>,
    penalty_matrix: Option<Vec<Vec<f64>>>,
) -> PyResult<SurvivalFit> {
    let requested_distribution_key = distribution.map(|name| name.to_lowercase().replace('-', "_"));
    let dist_type = parse_distribution_type(distribution)?;
    let fixed_scale = match (requested_distribution_key.as_deref(), fixed_scale) {
        (Some("exponential"), None) => Some(1.0),
        (Some("rayleigh"), None) => Some(0.5),
        (_, value) => value,
    };
    let config = SurvregConfig::create(Some(dist_type), max_iter, eps, tol_chol);
    let distribution_parameter =
        validate_distribution_parameter(config.distribution, distribution_parameter)?;
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Length mismatch: time has {} elements but status has {}. Both must have the same length.",
            n,
            status.len()
        )));
    }
    let require_positive_time = dist_type.uses_log_time();
    validate_time_values(&time, require_positive_time)?;
    validate_status_values(&status)?;
    let time2_values = validate_time2_values(&time, &status, time2, require_positive_time)?;
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
    let penalty_matrix = penalty_matrix
        .as_deref()
        .map(|values| validate_penalty_matrix(values, nvar))
        .transpose()?;
    let active_columns: Vec<usize> = (0..nvar)
        .filter(|&column| {
            covariate_rows.iter().any(|row| row[column] != 0.0)
                || penalty_matrix.as_ref().is_some_and(|penalty| {
                    (0..nvar).any(|other| {
                        penalty[(column, other)] != 0.0 || penalty[(other, column)] != 0.0
                    })
                })
        })
        .collect();
    let active_nvar = active_columns.len();
    let has_aliased_columns = active_nvar != nvar;
    let reduced_covariate_rows = has_aliased_columns.then(|| {
        covariate_rows
            .iter()
            .map(|row| active_columns.iter().map(|&column| row[column]).collect())
            .collect::<Vec<Vec<f64>>>()
    });
    let fit_covariate_rows = reduced_covariate_rows.as_deref().unwrap_or(&covariate_rows);
    let reduced_penalty_matrix = if has_aliased_columns {
        penalty_matrix.as_ref().map(|penalty| {
            Array2::from_shape_fn((active_nvar, active_nvar), |(row, column)| {
                penalty[(active_columns[row], active_columns[column])]
            })
        })
    } else {
        None
    };
    let fit_penalty_matrix = reduced_penalty_matrix.as_ref().or(penalty_matrix.as_ref());
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
    let initial_beta = initial_beta.unwrap_or_else(|| vec![0.0; expected_initial_len]);
    let fit_initial_beta = if has_aliased_columns {
        let mut values = Vec::with_capacity(active_nvar + estimated_scale_count);
        values.extend(active_columns.iter().map(|&column| initial_beta[column]));
        values.extend_from_slice(&initial_beta[nvar..]);
        values
    } else {
        initial_beta
    };
    validate_finite_values("initial_beta", &fit_initial_beta)?;
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
    let cov_array = if active_nvar > 0 {
        let mut flat = Vec::with_capacity(n * active_nvar);
        for col_idx in 0..active_nvar {
            flat.extend(fit_covariate_rows.iter().map(|row| row[col_idx]));
        }
        Array2::from_shape_vec((active_nvar, n), flat)
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
        nvar: active_nvar,
        y: &y,
        covariates: &cov_array,
        weights: &weights_arr,
        offsets: &offsets_arr,
        beta: fit_initial_beta,
        nstrat,
        strata: &strata_vec,
        eps: config.eps,
        tol_chol: config.tol_chol,
        distribution: distribution_type,
        distribution_parameter,
        fixed_scale,
        penalty_matrix: fit_penalty_matrix,
    })
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    let fitted_location_coefficients = &result.coefficients[..active_nvar];
    let mut location_coefficients = vec![f64::NAN; nvar];
    for (&column, &coefficient) in active_columns
        .iter()
        .zip(fitted_location_coefficients.iter())
    {
        location_coefficients[column] = coefficient;
    }
    let scales: Vec<f64> = if let Some(scale) = fixed_scale {
        vec![scale]
    } else {
        result.coefficients[active_nvar..active_nvar + nstrat]
            .iter()
            .map(|value| value.exp())
            .collect()
    };
    let linear_predictors = compute_linear_predictor(
        fit_covariate_rows,
        fitted_location_coefficients,
        Some(&offsets_vec),
    );
    let full_parameter_count = nvar + estimated_scale_count;
    let active_parameter_indices: Vec<usize> = active_columns
        .iter()
        .copied()
        .chain(nvar..full_parameter_count)
        .collect();
    let mut variance_matrix = vec![vec![0.0; full_parameter_count]; full_parameter_count];
    for (fit_row, &full_row) in active_parameter_indices.iter().enumerate() {
        for (fit_column, &full_column) in active_parameter_indices.iter().enumerate() {
            variance_matrix[full_row][full_column] = result.variance_matrix[(fit_row, fit_column)];
        }
    }
    let mut score_vector = vec![0.0; full_parameter_count];
    for (fit_index, &full_index) in active_parameter_indices.iter().enumerate() {
        score_vector[full_index] = result.score_vector[fit_index];
    }
    let coefficients = if fixed_scale.is_some() {
        location_coefficients.clone()
    } else {
        let mut values = location_coefficients.clone();
        values.extend_from_slice(&result.coefficients[active_nvar..active_nvar + nstrat]);
        values
    };
    let status_values: Vec<i32> = status.iter().map(|&value| value as i32).collect();
    let fitted_covariates = if nvar == 0 {
        vec![vec![]; n]
    } else {
        covariate_rows
    };
    Ok(SurvivalFit {
        coefficients,
        location_coefficients,
        scale: scales.first().copied().unwrap_or(1.0),
        scales,
        distribution: distribution_name,
        distribution_parameters: distribution_parameter.into_iter().collect(),
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
        score_vector,
        penalty_matrix: penalty_matrix
            .as_ref()
            .map(|matrix| {
                matrix
                    .outer_iter()
                    .map(|row| row.iter().copied().collect())
                    .collect()
            })
            .unwrap_or_default(),
        penalty: result.penalty,
        penalized_log_likelihood: result.penalized_log_likelihood,
        degrees_of_freedom: result.degrees_of_freedom,
    })
}

fn apply_quadratic_penalty(
    likelihood: &mut SurvivalLikelihood,
    beta: &[f64],
    penalty: Option<&Array2<f64>>,
) -> f64 {
    let Some(penalty) = penalty else {
        return 0.0;
    };
    let width = penalty.nrows();
    let mut quadratic = 0.0;
    for row in 0..width {
        let penalty_score = (0..width)
            .map(|column| penalty[(row, column)] * beta[column])
            .sum::<f64>();
        likelihood.u[row] -= penalty_score;
        quadratic += beta[row] * penalty_score;
        for column in 0..width {
            likelihood.imat[(row, column)] += penalty[(row, column)];
            likelihood.jj[(row, column)] += penalty[(row, column)];
        }
    }
    0.5 * quadratic
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
        distribution_parameter,
        fixed_scale,
        penalty_matrix,
    } = input;
    let n = y.nrows();
    let ny = y.ncols();
    let estimated_scale_count = if fixed_scale.is_some() { 0 } else { nstrat };
    let nvar2 = nvar + estimated_scale_count;
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
    let status_vec: Vec<i32> = if ny == 2 {
        y.column(1).iter().map(|&status| status as i32).collect()
    } else {
        y.column(2).iter().map(|&status| status as i32).collect()
    };
    let time2_vec: Option<Vec<f64>> = if ny == 3 {
        Some(y.column(1).iter().map(|&t| transform_time(t)).collect())
    } else {
        None
    };
    let time1_arr = Array1::from_vec(time1_vec);
    let status_arr = Array1::from_vec(status_vec);
    let time2_arr = time2_vec.map(Array1::from_vec);
    let strata_arr = Array1::from_iter(strata.iter().map(|&value| (value + 1) as i32));
    let frailty_arr = Array1::<i32>::zeros(n);
    let time1 = time1_arr.view();
    let status = status_arr.view();
    let strata = strata_arr.view();
    let frailty = frailty_arr.view();
    let time2_view: Option<ArrayView1<f64>> = time2_arr.as_ref().map(|v| v.view());
    let input = LikelihoodInput {
        n,
        nvar,
        nstrat: estimated_scale_count,
        beta: &beta,
        distribution: &distribution,
        distribution_parameter,
        strata: &strata,
        offsets,
        time1: &time1,
        time2: time2_view.as_ref(),
        status: &status,
        weights,
        covariates,
        frailty: &frailty,
    };
    let mut initial_likelihood = calculate_likelihood(&input)?;
    let mut loglik = initial_likelihood.loglik;
    let mut penalty_value = apply_quadratic_penalty(&mut initial_likelihood, &beta, penalty_matrix);
    let mut penalized_loglik = loglik - penalty_value;
    let mut imat = initial_likelihood.imat;
    let mut jj = initial_likelihood.jj;
    let mut u = initial_likelihood.u;
    usave.assign(&u);
    let mut iter = 0;
    let mut converged = false;
    while iter < max_iter {
        let old_penalized_loglik = penalized_loglik;
        let mut accepted = None;
        let observed_delta = is_positive_definite(&imat, tol_chol)
            .then(|| regularized_lu_solve(&imat, &u).ok())
            .flatten();
        let delta_candidates = [
            (true, observed_delta),
            (false, regularized_lu_solve(&jj, &u).ok()),
        ];
        for (uses_observed_information, delta) in delta_candidates
            .iter()
            .filter_map(|(observed, delta)| delta.as_ref().map(|delta| (*observed, delta)))
        {
            let mut step_factor = 1.0;
            for _ in 0..=MAX_HALVING_ITERATIONS {
                let mut candidate_beta = beta.clone();
                candidate_beta
                    .iter_mut()
                    .zip(beta.iter().zip(delta.iter()))
                    .for_each(|(nb, (b, d))| *nb = b + d * step_factor);
                adjust_strata(&mut candidate_beta, &beta, nvar, estimated_scale_count);

                let candidate_input = LikelihoodInput {
                    n,
                    nvar,
                    nstrat: estimated_scale_count,
                    beta: &candidate_beta,
                    distribution: &distribution,
                    distribution_parameter,
                    strata: &strata,
                    offsets,
                    time1: &time1,
                    time2: time2_view.as_ref(),
                    status: &status,
                    weights,
                    covariates,
                    frailty: &frailty,
                };
                let mut candidate = calculate_likelihood(&candidate_input)?;
                let candidate_loglik = candidate.loglik;
                let candidate_penalty =
                    apply_quadratic_penalty(&mut candidate, &candidate_beta, penalty_matrix);
                let candidate_penalized_loglik = candidate_loglik - candidate_penalty;
                if candidate_penalized_loglik.is_finite()
                    && candidate_penalized_loglik >= old_penalized_loglik
                    && (!uses_observed_information
                        || is_positive_definite(&candidate.imat, tol_chol))
                {
                    accepted = Some((
                        candidate_beta,
                        candidate_loglik,
                        candidate_penalty,
                        candidate_penalized_loglik,
                        candidate.imat,
                        candidate.jj,
                        candidate.u,
                    ));
                    break;
                }
                step_factor *= STEP_HALVE_FACTOR;
            }
            if accepted.is_some() {
                break;
            }
        }

        if let Some((
            candidate_beta,
            candidate_loglik,
            candidate_penalty,
            candidate_penalized_loglik,
            candidate_imat,
            candidate_jj,
            candidate_u,
        )) = accepted
        {
            beta = candidate_beta;
            loglik = candidate_loglik;
            penalty_value = candidate_penalty;
            penalized_loglik = candidate_penalized_loglik;
            imat = candidate_imat;
            jj = candidate_jj;
            u = candidate_u;
            usave.assign(&u);
            iter += 1;

            if check_convergence(old_penalized_loglik, penalized_loglik, eps) {
                converged = true;
                break;
            }
        } else {
            break;
        }
    }
    let convergence_flag = if converged { 0 } else { -1 };
    let variance = calculate_variance_matrix(imat, nvar2, tol_chol)?;
    let degrees_of_freedom = penalty_matrix.map(|penalty| {
        let penalty_trace = (0..nvar)
            .map(|row| {
                (0..nvar)
                    .map(|column| penalty[(row, column)] * variance[(column, row)])
                    .sum::<f64>()
            })
            .sum::<f64>();
        ((nvar2 as f64) - penalty_trace).clamp(0.0, nvar2 as f64)
    });
    Ok(SurvivalFitComputed {
        coefficients: beta,
        iterations: iter,
        variance_matrix: variance,
        log_likelihood: loglik,
        penalty: penalty_value,
        penalized_log_likelihood: penalized_loglik,
        degrees_of_freedom,
        convergence_flag,
        score_vector: usave.to_vec(),
    })
}
pub(crate) struct SurvivalFitComputed {
    coefficients: Vec<f64>,
    iterations: usize,
    variance_matrix: Array2<f64>,
    log_likelihood: f64,
    penalty: f64,
    penalized_log_likelihood: f64,
    degrees_of_freedom: Option<f64>,
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
    distribution_parameter: Option<f64>,
    fixed_scale: Option<f64>,
    penalty_matrix: Option<&'a Array2<f64>>,
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
        assert_eq!(config.distribution, DistributionType::Weibull);
    }

    #[test]
    fn test_survreg_distribution_fallbacks_are_weibull() {
        assert_eq!(
            SurvregConfig::new(None, None, None, None).distribution,
            DistributionType::Weibull
        );
        assert_eq!(
            SurvregConfig::create(None, None, None, None).distribution,
            DistributionType::Weibull
        );
        assert_eq!(
            parse_distribution_type(None).unwrap(),
            DistributionType::Weibull
        );
        assert_eq!(
            parse_distribution_type(Some("extreme_value")).unwrap(),
            DistributionType::ExtremeValue
        );
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
            DistributionType::StudentT,
        ];
        assert_eq!(variants.len(), 7);
    }

    #[test]
    fn test_requested_distribution_name_preserves_response_transform() {
        assert_eq!(
            requested_distribution_name(Some("exponential"), DistributionType::ExtremeValue),
            "exponential"
        );
        assert_eq!(
            parse_distribution_type(Some("rayleigh")).unwrap(),
            DistributionType::Weibull
        );
        assert_eq!(
            requested_distribution_name(Some("rayleigh"), DistributionType::Weibull),
            "rayleigh"
        );
        assert_eq!(
            requested_distribution_name(Some("normal"), DistributionType::Gaussian),
            "gaussian"
        );
        assert_eq!(
            requested_distribution_name(Some("loggaussian"), DistributionType::LogNormal),
            "lognormal"
        );
        assert_eq!(
            requested_distribution_name(Some("log-logistic"), DistributionType::LogLogistic),
            "loglogistic"
        );
        assert_eq!(
            requested_distribution_name(Some("extreme"), DistributionType::ExtremeValue),
            "extreme_value"
        );
        assert_eq!(
            requested_distribution_name(Some("student-t"), DistributionType::StudentT),
            "t"
        );
    }

    #[test]
    fn base_distributions_accept_transformed_nonpositive_responses() {
        let arguments = || {
            (
                vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                vec![1.0; 6],
                vec![vec![1.0]; 6],
            )
        };
        let (time, status, covariates) = arguments();
        let gaussian = survreg(
            time,
            status,
            covariates,
            None,
            None,
            Some(vec![0.5, 0.0]),
            None,
            Some("gaussian"),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(gaussian.is_ok());

        let (time, status, covariates) = arguments();
        let weibull = survreg(
            time,
            status,
            covariates,
            None,
            None,
            None,
            None,
            Some("weibull"),
            Some(1),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(weibull.is_err());
    }

    #[test]
    fn survreg_marks_all_zero_covariates_as_aliased() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let x = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 0.25];
        let reduced_covariates: Vec<Vec<f64>> = x.iter().map(|&value| vec![1.0, value]).collect();
        let aliased_covariates: Vec<Vec<f64>> =
            x.iter().map(|&value| vec![1.0, 0.0, value]).collect();
        let fit = survreg(
            time.clone(),
            status.clone(),
            aliased_covariates,
            None,
            None,
            None,
            None,
            Some("weibull"),
            Some(100),
            Some(1e-10),
            Some(1e-10),
            None,
            None,
            None,
            None,
        )
        .expect("aliased fit should succeed");
        let reduced = survreg(
            time,
            status,
            reduced_covariates,
            None,
            None,
            None,
            None,
            Some("weibull"),
            Some(100),
            Some(1e-10),
            Some(1e-10),
            None,
            None,
            None,
            None,
        )
        .expect("reduced fit should succeed");

        assert_eq!(fit.n_covariates, 3);
        assert_eq!(fit.covariates[0].len(), 3);
        assert!(fit.location_coefficients[1].is_nan());
        assert_eq!(fit.variance_matrix.len(), 4);
        assert!(fit.variance_matrix[1].iter().all(|&value| value == 0.0));
        assert!(fit.variance_matrix.iter().all(|row| row[1] == 0.0));
        assert_eq!(fit.score_vector[1], 0.0);
        for (&actual, &expected) in [fit.location_coefficients[0], fit.location_coefficients[2]]
            .iter()
            .zip(reduced.location_coefficients.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((fit.coefficients[3] - reduced.coefficients[2]).abs() < 1e-12);
        assert_eq!(fit.linear_predictors, reduced.linear_predictors);
        assert!((fit.log_likelihood - reduced.log_likelihood).abs() < 1e-12);
    }

    #[test]
    fn quadratic_penalty_matches_reference_pspline_survreg_fit() {
        let time = vec![
            4.0, 7.0, 9.0, 12.0, 15.0, 18.0, 22.0, 25.0, 30.0, 36.0, 43.0, 51.0,
        ];
        let status = vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let x = vec![
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
        ];
        let (basis, _) = crate::core::pspline::pspline_basis_core(&x, 4, 3, (-2.0, 3.5))
            .expect("P-spline basis should be valid");
        let covariates = basis
            .into_iter()
            .map(|row| {
                std::iter::once(1.0)
                    .chain(row.into_iter().skip(1))
                    .collect()
            })
            .collect();
        let difference_penalty = [
            [5.0, -4.0, 1.0, 0.0, 0.0, 0.0],
            [-4.0, 6.0, -4.0, 1.0, 0.0, 0.0],
            [1.0, -4.0, 6.0, -4.0, 1.0, 0.0],
            [0.0, 1.0, -4.0, 6.0, -4.0, 1.0],
            [0.0, 0.0, 1.0, -4.0, 5.0, -2.0],
            [0.0, 0.0, 0.0, 1.0, -2.0, 1.0],
        ];
        let penalty = (0..7)
            .map(|row| {
                (0..7)
                    .map(|column| {
                        if row == 0 || column == 0 {
                            0.0
                        } else {
                            difference_penalty[row - 1][column - 1]
                        }
                    })
                    .collect()
            })
            .collect();

        let fit = survreg(
            time,
            status,
            covariates,
            None,
            None,
            None,
            None,
            Some("weibull"),
            Some(100),
            Some(1e-9),
            Some(1e-10),
            None,
            None,
            None,
            Some(penalty),
        )
        .expect("quadratic-penalized survreg fit should succeed");

        let expected = [
            -1.531_895_335_731_6,
            3.388_173_860_008_563_6,
            3.961_988_359_190_023,
            4.574_463_494_434_461,
            4.950_961_917_375_609,
            5.505_539_175_918_376,
            5.867_096_525_333_904,
        ];
        for (actual, expected) in fit.location_coefficients.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-8);
        }
        assert!((fit.scale - 0.004_673_727_601_541_054).abs() < 1e-10);
        assert!((fit.penalty - 4.023_383_511_207_064).abs() < 1e-8);
        assert!((fit.log_likelihood - fit.penalized_log_likelihood - fit.penalty).abs() < 1e-12);
        let degrees_of_freedom = fit
            .degrees_of_freedom
            .expect("penalized fit should report effective df");
        assert!(
            (degrees_of_freedom - 7.928_923_322_702_46).abs() < 1e-8,
            "effective df was {degrees_of_freedom}"
        );
    }

    #[test]
    fn quadratic_penalty_validation_rejects_invalid_matrices() {
        assert!(validate_penalty_matrix(&[vec![1.0, -1.0], vec![-1.0, 1.0]], 2).is_ok());
        assert!(validate_penalty_matrix(&[vec![1.0], vec![0.0]], 2).is_err());
        assert!(validate_penalty_matrix(&[vec![1.0, 0.0], vec![1.0, 1.0]], 2).is_err());
        assert!(validate_penalty_matrix(&[vec![1.0, 2.0], vec![2.0, 1.0]], 2).is_err());
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
    fn positive_definite_check_rejects_saddle_information() {
        let positive = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 1.0]).unwrap();
        let indefinite = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();

        assert!(is_positive_definite(&positive, 1e-10));
        assert!(!is_positive_definite(&indefinite, 1e-10));
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
        });

        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.coefficients.len(), 2);
        assert!(fit.log_likelihood.is_finite());
    }

    #[test]
    fn observed_information_accelerates_well_conditioned_fit() {
        let y = Array2::from_shape_vec(
            (6, 2),
            vec![1.2, 1.0, 2.1, 1.0, 3.0, 0.0, 4.5, 1.0, 6.2, 1.0, 8.1, 0.0],
        )
        .unwrap();
        let covariates = Array2::from_shape_vec(
            (2, 6),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let result = compute_survreg(ComputeSurvregInput {
            max_iter: 100,
            nvar: 2,
            y: &y,
            covariates: &covariates,
            weights: &Array1::from_vec(vec![1.0; 6]),
            offsets: &Array1::from_vec(vec![0.0; 6]),
            beta: vec![2.113333, 1.38, 1.103973],
            nstrat: 1,
            strata: &[0; 6],
            eps: 1e-10,
            tol_chol: 1e-10,
            distribution: DistributionType::Gaussian,
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
        })
        .unwrap();

        assert!(result.iterations <= 10);
        assert!((result.coefficients[0] - 2.244776).abs() < 1e-5);
        assert!((result.coefficients[1] - 1.392510).abs() < 1e-5);
        assert!(result.score_vector.iter().all(|score| score.abs() < 1e-5));
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
            distribution_parameter: None,
            fixed_scale: None,
            penalty_matrix: None,
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
            distribution_parameter: None,
            fixed_scale: Some(1.25),
            penalty_matrix: None,
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
            penalty: 0.25,
            penalized_log_likelihood: -50.25,
            degrees_of_freedom: Some(1.75),
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
