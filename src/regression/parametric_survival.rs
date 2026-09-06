use crate::constants::{CHOLESKY_TOL, DEFAULT_MAX_ITER, MAX_HALVING_ITERATIONS, STEP_HALVE_FACTOR};
use crate::regression::survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, compute_linear_predictor,
    compute_quantile_prediction, compute_response_prediction, compute_se_linear_predictor,
};
use crate::regression::survregc1::{SurvivalDist, SurvivalLikelihood, survreg_loglik, survregc1};
use crate::residuals::survreg_resid::{
    SurvregResidType, SurvregResiduals, compute_deviance_residuals_survreg_with_parameter,
    compute_dfbeta_survreg_with_parameter, compute_ldcase_with_parameter,
    compute_response_residuals, compute_response_residuals_censored_with_parameter,
    compute_survreg_dfbeta_residuals, compute_survreg_residual_matrix_with_parameter,
    compute_working_residuals_from_derivative_matrix, compute_working_residuals_with_parameter,
};
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;

mod initial;
use initial::{SurvregRescaling, initialize_survreg, is_mean_only};

type PredictionRows = (Vec<f64>, Option<Vec<Vec<f64>>>);

// Matches survival::survreg.control() without changing other model families.
const SURVREG_CONVERGENCE_TOLERANCE: f64 = 1e-9;

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
            eps: eps.unwrap_or(SURVREG_CONVERGENCE_TOLERANCE),
            tol_chol: tol_chol.unwrap_or(CHOLESKY_TOL),
        }
    }
}

impl Default for SurvregConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            eps: SURVREG_CONVERGENCE_TOLERANCE,
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
            eps: eps.unwrap_or(SURVREG_CONVERGENCE_TOLERANCE),
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
}

impl DistributionType {
    fn likelihood_distribution(
        self,
        distribution_parameter: Option<f64>,
    ) -> Result<SurvivalDist, Box<dyn std::error::Error>> {
        Ok(match self {
            DistributionType::ExtremeValue => SurvivalDist::ExtremeValue,
            DistributionType::Logistic => SurvivalDist::Logistic,
            DistributionType::Gaussian => SurvivalDist::Gaussian,
            DistributionType::Weibull => SurvivalDist::Weibull,
            DistributionType::LogNormal => SurvivalDist::LogNormal,
            DistributionType::LogLogistic => SurvivalDist::LogLogistic,
            DistributionType::StudentT => SurvivalDist::StudentT(
                distribution_parameter.ok_or("Student-t degrees of freedom are missing")?,
            ),
        })
    }

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

impl LikelihoodInput<'_> {
    fn evaluate(&self) -> Result<SurvivalLikelihood, Box<dyn std::error::Error>> {
        survregc1(
            self.n,
            self.nvar,
            self.nstrat,
            false,
            &ArrayView1::from(self.beta),
            self.distribution
                .likelihood_distribution(self.distribution_parameter)?,
            self.strata,
            &self.offsets.view(),
            self.time1,
            self.time2,
            self.status,
            &self.weights.view(),
            &self.covariates.view(),
            0,
            self.frailty,
        )
    }

    fn loglik(&self) -> Result<f64, Box<dyn std::error::Error>> {
        survreg_loglik(
            self.n,
            self.nvar,
            self.nstrat,
            &ArrayView1::from(self.beta),
            self.distribution
                .likelihood_distribution(self.distribution_parameter)?,
            self.strata,
            &self.offsets.view(),
            self.time1,
            self.time2,
            self.status,
            &self.weights.view(),
            &self.covariates.view(),
            0,
            self.frailty,
        )
    }
}
fn check_convergence(old: f64, new: f64, eps: f64) -> bool {
    (1.0 - new / old).abs() <= eps || (old - new).abs() <= eps
}

/// Ordered LDL factorization used by the AFT information and score-product
/// systems. Discarded pivots represent aliased parameters, not a request to
/// add a ridge penalty. The tolerance and signed rank follow survreg's
/// cholesky3 convention, which differs from the Cox factorization.
struct SurvregInformation {
    factors: Vec<f64>,
    size: usize,
    signed_rank: i32,
}

impl SurvregInformation {
    fn factor(matrix: &Array2<f64>, tolerance: f64) -> Self {
        let size = matrix.nrows();
        debug_assert_eq!(size, matrix.ncols());
        let mut factors: Vec<f64> = matrix.iter().copied().collect();
        // R's dense cholesky3 uses an absolute threshold when the original
        // diagonal is nonnegative, and scales by its minimum otherwise.
        let minimum = matrix.diag().iter().copied().fold(0.0_f64, f64::min);
        let threshold = if minimum == 0.0 {
            tolerance
        } else {
            minimum * tolerance
        };
        let mut rank = 0;
        let mut sign = 1;
        for column in 0..size {
            let pivot = factors[column * size + column];
            if !pivot.is_finite() || pivot < threshold {
                for row in column..size {
                    factors[row * size + column] = 0.0;
                }
                if pivot < -8.0 * threshold {
                    sign = -1;
                }
                continue;
            }
            rank += 1;
            for row in column + 1..size {
                let multiplier = factors[row * size + column] / pivot;
                factors[row * size + column] = multiplier;
                factors[row * size + row] -= multiplier * multiplier * pivot;
                for following in row + 1..size {
                    factors[following * size + row] -=
                        multiplier * factors[following * size + column];
                }
            }
        }
        Self {
            factors,
            size,
            signed_rank: rank * sign,
        }
    }

    fn solve(&self, score: &Array1<f64>) -> Option<Array1<f64>> {
        debug_assert_eq!(score.len(), self.size);
        let mut step = score.to_vec();
        for row in 0..self.size {
            let mut value = step[row];
            for (column, previous) in step.iter().enumerate().take(row) {
                value -= previous * self.factors[row * self.size + column];
            }
            step[row] = value;
        }
        for row in (0..self.size).rev() {
            let diagonal = self.factors[row * self.size + row];
            if diagonal == 0.0 {
                step[row] = 0.0;
                continue;
            }
            let mut value = step[row] / diagonal;
            for (column, following) in step.iter().enumerate().skip(row + 1) {
                value -= following * self.factors[column * self.size + row];
            }
            step[row] = value;
        }
        step.iter()
            .all(|value| value.is_finite())
            .then(|| Array1::from_vec(step))
    }

    fn covariance(mut self) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let n = self.size;
        // Invert the unit-lower factor and its positive diagonal weights.
        // As in R's chinv2, an accepted nonpositive pivot is not reciprocated.
        // This also preserves R's prescribed-parameter indefinite cases.
        for column in 0..n {
            if self.factors[column * n + column] > 0.0 {
                self.factors[column * n + column] = 1.0 / self.factors[column * n + column];
                for row in column + 1..n {
                    self.factors[row * n + column] = -self.factors[row * n + column];
                    for previous in 0..column {
                        self.factors[row * n + previous] +=
                            self.factors[row * n + column] * self.factors[column * n + previous];
                    }
                }
            }
        }
        // Accumulate the symmetric covariance in the upper triangle. Zero
        // rows and columns explicitly retain the discarded-pivot mask.
        for row in 0..n {
            if self.factors[row * n + row] == 0.0 {
                for column in 0..row {
                    self.factors[column * n + row] = 0.0;
                }
                for column in row..n {
                    self.factors[row * n + column] = 0.0;
                }
            } else {
                for following in row + 1..n {
                    let weighted =
                        self.factors[following * n + row] * self.factors[following * n + following];
                    self.factors[row * n + following] = weighted;
                    for column in row..following {
                        self.factors[row * n + column] +=
                            weighted * self.factors[following * n + column];
                    }
                }
            }
        }
        for row in 1..n {
            for column in 0..row {
                self.factors[row * n + column] = self.factors[column * n + row];
            }
        }
        if self.factors.iter().any(|value| !value.is_finite()) {
            return Err("AFT covariance contains non-finite values".into());
        }
        Ok(Array2::from_shape_vec((n, n), self.factors)?)
    }
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
    #[pyo3(name = "t")]
    StudentT,
}

const DEFAULT_SURVREG_DISTRIBUTION: DistributionType = DistributionType::Weibull;

#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, offsets=None, initial_beta=None, strata=None, distribution=None, max_iter=None, eps=None, tol_chol=None, time2=None, fixed_scale=None, distribution_parameter=None))]
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
    let initialize_scales = estimated_scale_count > 0
        && initial_beta
            .as_ref()
            .is_some_and(|values| values.len() == nvar);
    if let Some(values) = initial_beta.as_ref()
        && values.len() != expected_initial_len
        && !initialize_scales
    {
        let expected = if estimated_scale_count > 0 {
            format!("{nvar} location coefficients or {expected_initial_len} full parameters")
        } else {
            expected_initial_len.to_string()
        };
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "initial_beta has {} values but model expects {expected}",
            values.len()
        )));
    }
    if let Some(values) = initial_beta.as_ref() {
        validate_finite_values("initial_beta", values)?;
    }
    let initialize = initial_beta.is_none();
    let initial_beta = initial_beta.unwrap_or_default();
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
    let mut cov_array = if nvar > 0 {
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
    if initialize_scales && is_mean_only(&cov_array, &weights_arr) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "an intercept-only model requires a full initial vector including log-scale parameters",
        ));
    }
    let distribution_type = config.distribution;
    let distribution_name = requested_distribution_name(distribution, distribution_type);
    let rescaling = if initialize {
        SurvregRescaling::apply(&mut cov_array, &weights_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
    } else {
        None
    };
    let mut input = ComputeSurvregInput {
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
        distribution_parameter,
        fixed_scale,
    };
    if initialize || initialize_scales {
        let initial_location = (!initialize).then_some(input.beta.as_slice());
        input.beta = initialize_survreg(&input, initial_location)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    }
    let mut result = compute_survreg(input)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let rescaled_predictors = rescaling.as_ref().map(|_| {
        (0..n)
            .map(|person| {
                (0..nvar)
                    .map(|column| cov_array[[column, person]] * result.coefficients[column])
                    .sum::<f64>()
                    + offsets_vec[person]
            })
            .collect()
    });
    if let Some(rescaling) = rescaling {
        rescaling.restore(&mut result);
    }
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
    let linear_predictors = rescaled_predictors.unwrap_or_else(|| {
        compute_linear_predictor(&covariate_rows, &location_coefficients, Some(&offsets_vec))
    });
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
        distribution_parameter,
        fixed_scale,
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
    let initial_likelihood = input.evaluate()?;
    let mut loglik = initial_likelihood.loglik;
    let mut information = SurvregInformation::factor(&initial_likelihood.imat, tol_chol);
    let mut jj = initial_likelihood.jj;
    let mut u = initial_likelihood.u;
    usave.assign(&u);
    let mut iter = 0;
    let mut converged = false;
    while iter < max_iter {
        let old_loglik = loglik;
        let mut accepted = None;
        for uses_observed_information in [true, false] {
            let delta = if uses_observed_information {
                (information.signed_rank >= 0)
                    .then(|| information.solve(&u))
                    .flatten()
            } else {
                // Compute the score-product fallback only if an observed
                // information step could not be accepted.
                SurvregInformation::factor(&jj, tol_chol).solve(&u)
            };
            let Some(delta) = delta else { continue };
            let mut step_factor = 1.0;
            let mut retry = false;
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
                // The first trial usually succeeds. Once it fails, screen
                // shorter steps without allocating or accumulating derivative
                // matrices, then fully evaluate any improving candidate.
                if retry {
                    let trial_loglik = candidate_input.loglik()?;
                    if !trial_loglik.is_finite() || trial_loglik < old_loglik {
                        step_factor *= STEP_HALVE_FACTOR;
                        continue;
                    }
                }
                let candidate = candidate_input.evaluate()?;
                if candidate.loglik.is_finite()
                    && candidate.loglik >= old_loglik
                    && candidate.u.iter().all(|value| value.is_finite())
                    && candidate.imat.diag().iter().all(|value| value.is_finite())
                {
                    let candidate_information =
                        SurvregInformation::factor(&candidate.imat, tol_chol);
                    accepted = Some((
                        candidate_beta,
                        candidate.loglik,
                        candidate_information,
                        candidate.jj,
                        candidate.u,
                    ));
                    break;
                }
                retry = true;
                step_factor *= STEP_HALVE_FACTOR;
            }
            if accepted.is_some() {
                break;
            }
        }

        if let Some((
            candidate_beta,
            candidate_loglik,
            candidate_information,
            candidate_jj,
            candidate_u,
        )) = accepted
        {
            beta = candidate_beta;
            loglik = candidate_loglik;
            information = candidate_information;
            jj = candidate_jj;
            u = candidate_u;
            usave.assign(&u);
            iter += 1;

            if check_convergence(old_loglik, loglik, eps) {
                converged = true;
                break;
            }
        } else {
            // At an optimum, summation roundoff can make every trial appear
            // worse. Only declare convergence when the observed-information
            // Newton decrement is below both the requested absolute tolerance
            // and floating-point resolution of the accepted likelihood.
            if information.signed_rank >= 0 && loglik.is_finite() {
                converged = information.solve(&u).is_some_and(|delta| {
                    let decrement = u.dot(&delta);
                    decrement >= 0.0 && decrement <= eps.min(f64::EPSILON * loglik.abs().max(1.0))
                });
            }
            break;
        }
    }
    let convergence_flag = if converged { 0 } else { -1 };
    let variance = information.covariance()?;
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
    distribution_parameter: Option<f64>,
    fixed_scale: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_initial_values_use_fitted_null_scales_without_solving_locations() {
        // For these exact Gaussian observations, the intercept-only MLE has
        // mean 13/4 and variance 59/16. A zero location column would make the
        // unused scalar GLIM update undefined (0/0).
        let fit = |initial| {
            survreg(
                vec![1.0, 2.0, 4.0, 6.0],
                vec![1.0; 4],
                vec![vec![0.0]; 4],
                None,
                None,
                Some(initial),
                None,
                Some("gaussian"),
                Some(0),
                Some(1e-12),
                None,
                None,
                None,
                None,
            )
            .unwrap()
        };
        let partial = fit(vec![0.0]);
        assert_eq!(partial.coefficients[0], 0.0);
        assert!((partial.coefficients[1] - 0.5 * (59.0_f64 / 16.0).ln()).abs() < 1e-10);
        let completed = fit(partial.coefficients.clone());
        assert_eq!(partial.coefficients, completed.coefficients);
        assert_eq!(partial.variance_matrix, completed.variance_matrix);
        assert_eq!(partial.log_likelihood, completed.log_likelihood);
        assert_eq!(partial.score_vector, completed.score_vector);
    }

    #[test]
    fn partial_initial_values_require_full_vector_for_mean_only_estimated_scales() {
        let fit = |initial, scale| {
            survreg(
                vec![1.0, 2.0, 4.0, 6.0],
                vec![1.0; 4],
                vec![vec![1.0]; 4],
                None,
                None,
                Some(initial),
                None,
                Some("gaussian"),
                Some(0),
                None,
                None,
                None,
                scale,
                None,
            )
        };
        let error = fit(vec![1.2], None).unwrap_err();
        assert!(error.to_string().contains("full initial vector"));
        assert_eq!(
            fit(vec![1.2, 0.0], None).unwrap().coefficients,
            vec![1.2, 0.0]
        );
        assert_eq!(fit(vec![1.2], Some(1.0)).unwrap().coefficients, vec![1.2]);
    }

    #[test]
    fn empty_initial_locations_are_valid_only_for_a_zero_column_design() {
        let fit = |row: Vec<f64>| {
            survreg(
                vec![1.0, 2.0, 4.0, 6.0],
                vec![1.0; 4],
                vec![row; 4],
                None,
                None,
                Some(vec![]),
                None,
                Some("gaussian"),
                Some(0),
                Some(1e-12),
                None,
                None,
                None,
                None,
            )
        };
        let empty = fit(vec![]).unwrap();
        assert_eq!(empty.n_covariates, 0);
        assert_eq!(empty.coefficients.len(), 1);
        assert!((empty.scales[0] - (59.0_f64 / 16.0).sqrt()).abs() < 1e-10);
        assert_eq!(empty.linear_predictors, vec![0.0; 4]);
        assert!(fit(vec![0.0]).is_err());
    }

    #[test]
    fn test_survreg_config_default() {
        for config in [
            SurvregConfig::default(),
            SurvregConfig::new(None, None, None, None),
            SurvregConfig::create(None, None, None, None),
        ] {
            assert_eq!(config.max_iter, 30);
            assert_eq!(config.eps, 1e-9);
            assert_eq!(config.tol_chol, 1e-10);
            assert_eq!(config.distribution, DistributionType::Weibull);
        }
    }

    #[test]
    fn default_survreg_tolerance_matches_r_control_on_censored_data() {
        let fit_with_tolerance = |eps| {
            survreg(
                vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                vec![
                    vec![1.0, 0.0],
                    vec![1.0, 1.0],
                    vec![1.0, 0.0],
                    vec![1.0, 1.0],
                    vec![1.0, 0.0],
                    vec![1.0, 1.0],
                ],
                None,
                None,
                // Isolate the tolerance comparison from automatic initialization.
                Some(vec![0.0; 3]),
                None,
                Some("weibull"),
                None,
                eps,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        };
        let default = fit_with_tolerance(None);
        let explicit = fit_with_tolerance(Some(1e-9));
        assert_eq!(default.coefficients, explicit.coefficients);
        assert_eq!(default.iterations, explicit.iterations);
        assert!(default.score_vector.iter().all(|score| score.abs() < 1e-10));
        // R survival 3.8.11 default Weibull fit: intercept, x and log(scale).
        let reference = [
            2.201_004_230_665_683,
            -0.412_866_281_489_401_24,
            -0.280_610_838_292_473_6,
        ];
        for (actual, expected) in default.coefficients.iter().zip(reference) {
            assert!((actual - expected).abs() < 2e-9);
        }
        let loose = fit_with_tolerance(Some(1e-6));
        assert!(loose.iterations < default.iterations);
        assert!(loose.score_vector.iter().any(|score| score.abs() > 1e-8));
    }

    #[test]
    fn automatic_initialization_ignores_zero_weight_unresolved_intervals() {
        let fit = |time: Vec<f64>, status, weights, time2| {
            let n = time.len();
            survreg(
                time,
                status,
                vec![vec![1.0]; n],
                Some(weights),
                None,
                None,
                None,
                Some("logistic"),
                Some(0),
                None,
                None,
                time2,
                Some(1.0),
                None,
            )
            .unwrap()
        };
        let reference = fit(vec![1.0, 3.0], vec![1.0; 2], vec![1.0; 2], None);
        let with_ignored = fit(
            vec![1.0, 3.0, 1e-100],
            vec![1.0, 1.0, 3.0],
            vec![1.0, 1.0, 0.0],
            Some(vec![1.0, 3.0, 2e-100]),
        );
        assert_eq!(with_ignored.coefficients, reference.coefficients);
        assert_eq!(with_ignored.variance_matrix, reference.variance_matrix);
        assert_eq!(with_ignored.score_vector, reference.score_vector);
        assert_eq!(with_ignored.log_likelihood, reference.log_likelihood);
    }

    #[test]
    fn automatic_rescaling_ignores_zero_weight_extreme_covariates() {
        for fixed_scale in [None, Some(1.0)] {
            let fit = |ignored| {
                let mut time = vec![1.0, 2.0, 3.0, 4.0];
                let mut covariates: Vec<Vec<f64>> = time.iter().map(|&x| vec![1.0, x]).collect();
                let mut weights = vec![1.0; 4];
                if ignored {
                    time.push(2.0);
                    covariates.push(vec![1.0, 1e20]);
                    weights.push(0.0);
                }
                let status = vec![1.0; time.len()];
                survreg(
                    time,
                    status,
                    covariates,
                    Some(weights),
                    None,
                    None,
                    None,
                    Some("gaussian"),
                    Some(0),
                    None,
                    None,
                    None,
                    fixed_scale,
                    None,
                )
                .unwrap()
            };
            let reference = fit(false);
            let with_ignored = fit(true);
            assert_eq!(with_ignored.coefficients, reference.coefficients);
            assert_eq!(with_ignored.variance_matrix, reference.variance_matrix);
            assert_eq!(with_ignored.score_vector, reference.score_vector);
            assert_eq!(with_ignored.log_likelihood, reference.log_likelihood);
            assert_eq!(
                &with_ignored.linear_predictors[..4],
                reference.linear_predictors
            );
        }
    }

    #[test]
    fn automatic_gaussian_starts_match_r_without_optimizer_iterations() {
        let time = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        let all_event = survreg(
            time.clone(),
            vec![1.0; 6],
            (0..6).map(|i| vec![1.0, f64::from(i)]).collect(),
            None,
            None,
            None,
            None,
            Some("gaussian"),
            Some(0),
            None,
            None,
            None,
            Some(1.0),
            None,
        )
        .unwrap();
        for (actual, expected) in all_event
            .location_coefficients
            .iter()
            .zip([0.23809523809523903, 1.771_428_571_428_571_4])
        {
            assert!((actual - expected).abs() < 2e-14, "{actual} != {expected}");
        }
        assert_eq!(all_event.iterations, 0);
        assert_eq!(all_event.scale, 1.0);
        let censored = survreg(
            time,
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![vec![1.0]; 6],
            None,
            None,
            None,
            None,
            Some("gaussian"),
            Some(0),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!((censored.location_coefficients[0] - 6.491_626_281_538_652).abs() < 2e-8);
        assert!((censored.scale - 6.182412330330469).abs() < 2e-14);
        assert_eq!(censored.iterations, 0);
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
    fn test_check_convergence() {
        assert!(check_convergence(-100.0, -100.0, 1e-6));
        assert!(check_convergence(-100.0, -100.00001, 1e-4));
        assert!(!check_convergence(-100.0, -99.0, 1e-6));
        assert!(check_convergence(-1e-10, -1e-10, 1e-6));
        assert!(check_convergence(-100.0, -100.0 + 1e-7, 1e-6));
    }

    #[test]
    fn information_rank_distinguishes_singular_and_indefinite_systems() {
        let positive = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 1.0]).unwrap();
        let indefinite = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 1.0]).unwrap();
        let singular = Array2::from_shape_vec((2, 2), vec![1.0; 4]).unwrap();

        assert_eq!(SurvregInformation::factor(&positive, 1e-10).signed_rank, 2);
        assert_eq!(
            SurvregInformation::factor(&indefinite, 1e-10).signed_rank,
            -1
        );
        assert_eq!(SurvregInformation::factor(&singular, 1e-10).signed_rank, 1);
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
        let result = SurvregInformation::factor(&imat, 1e-10).covariance();
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
        let result = SurvregInformation::factor(&imat, 1e-10).covariance();
        assert!(result.is_ok());
        let var = result.unwrap();
        assert_eq!(var.nrows(), 2);
        assert_eq!(var.ncols(), 2);
        let identity = imat.dot(&var);
        for row in 0..2 {
            for column in 0..2 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((identity[[row, column]] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn singular_information_returns_a_generalized_covariance_and_step() {
        let imat = Array2::from_shape_vec(
            (3, 3),
            vec![6.0, 3.0, 3.0, 3.0, 19.0, 19.0, 3.0, 19.0, 19.0],
        )
        .unwrap();
        let information = SurvregInformation::factor(&imat, 1e-10);
        let step = information
            .solve(&Array1::from_vec(vec![21.0, 28.0, 28.0]))
            .unwrap();
        assert!((step[0] - 3.0).abs() < 1e-14);
        assert!((step[1] - 1.0).abs() < 1e-14);
        assert_eq!(step[2], 0.0);
        let variance = information.covariance().unwrap();
        for index in 0..3 {
            assert_eq!(variance[[2, index]], 0.0);
            assert_eq!(variance[[index, 2]], 0.0);
        }
        // A V A = A characterizes the generalized inverse on the estimable
        // column space; the selected coefficients preserve original order.
        let reconstructed = imat.dot(&variance).dot(&imat);
        for (actual, expected) in reconstructed.iter().zip(imat.iter()) {
            assert!((actual - expected).abs() < 1e-13);
        }
        assert!((variance[[0, 0]] - 19.0 / 105.0).abs() < 1e-15);
        assert!((variance[[1, 1]] - 6.0 / 105.0).abs() < 1e-15);
    }

    #[test]
    fn information_uses_absolute_tolerance_for_positive_diagonals() {
        // R Gaussian intercept fit, fixed scale 1e6 and three observations.
        let information = Array2::from_shape_vec((1, 1), vec![3e-12]).unwrap();
        let discarded = SurvregInformation::factor(&information, 1e-10)
            .covariance()
            .unwrap();
        assert_eq!(discarded[[0, 0]], 0.0);
        let retained = SurvregInformation::factor(&information, 1e-14)
            .covariance()
            .unwrap();
        assert!((retained[[0, 0]] * 3e-12 - 1.0).abs() < 1e-15);
        // cholesky3 discards pivots strictly below the threshold.
        assert_eq!(
            SurvregInformation::factor(&information, 3e-12).signed_rank,
            1
        );
    }

    #[test]
    fn indefinite_prescribed_information_matches_r_generalized_covariance() {
        // R survreg6 at Gaussian location/log(scale)=0 for y=1:4.
        let information = Array2::from_shape_vec((2, 2), vec![4.0, 20.0, 20.0, 60.0]).unwrap();
        let factor = SurvregInformation::factor(&information, 1e-10);
        assert_eq!(factor.signed_rank, -1);
        assert_eq!(
            factor.covariance().unwrap().into_raw_vec_and_offset().0,
            vec![0.25, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn information_retains_r_negative_cutoff_and_roundoff_rank_rules() {
        // R cholesky3/chinv2 can retain a tiny negative diagonal when a
        // larger original negative diagonal sets a negative cutoff.
        let diagonal = Array2::from_diag(&Array1::from_vec(vec![-1.0, -1e-12, 2.0]));
        let factor = SurvregInformation::factor(&diagonal, 1e-10);
        assert_eq!(factor.signed_rank, -2);
        assert_eq!(
            factor.covariance().unwrap().diag().to_vec(),
            vec![0.0, -1e-12, 0.5]
        );
        for (difference, rank) in [(1e-12, 1), (1e-8, -1)] {
            let information =
                Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0 - difference]).unwrap();
            assert_eq!(
                SurvregInformation::factor(&information, 1e-10).signed_rank,
                rank
            );
        }
    }

    #[test]
    fn covariance_keeps_independent_columns_after_an_alias() {
        let information =
            Array2::from_shape_vec((3, 3), vec![2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 3.0])
                .unwrap();
        let factor = SurvregInformation::factor(&information, 1e-10);
        assert_eq!(factor.signed_rank, 2);
        let variance = factor.covariance().unwrap();
        assert_eq!(variance.row(1).to_vec(), vec![0.0; 3]);
        assert_eq!(variance.column(1).to_vec(), vec![0.0; 3]);
        for (actual, expected) in information
            .dot(&variance)
            .dot(&information)
            .iter()
            .zip(information.iter())
        {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
}
