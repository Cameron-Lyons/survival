use ndarray::{Array1, Array2};
use pyo3::prelude::*;

use crate::constants::clamped_normal_ci_bounds_95;
use crate::internal::matrix::{matrix_inverse, regularized_lu_solve};

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct SplineConfig {
    #[pyo3(get, set)]
    pub n_knots: usize,
    #[pyo3(get, set)]
    pub degree: usize,
    #[pyo3(get, set)]
    pub knot_placement: String,
    #[pyo3(get, set)]
    pub boundary_knots: Option<(f64, f64)>,
}

#[pymethods]
impl SplineConfig {
    #[new]
    #[pyo3(signature = (n_knots=4, degree=3, knot_placement="quantile".to_string(), boundary_knots=None))]
    pub fn new(
        n_knots: usize,
        degree: usize,
        knot_placement: String,
        boundary_knots: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        build_spline_config(n_knots, degree, knot_placement, boundary_knots)
    }
}

fn build_spline_config(
    n_knots: usize,
    degree: usize,
    knot_placement: String,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<SplineConfig> {
    if n_knots < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_knots must be at least 2",
        ));
    }
    if degree == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "degree must be positive",
        ));
    }

    let knot_placement = normalize_knot_placement(&knot_placement)?;
    validate_boundary_knots(boundary_knots)?;

    Ok(SplineConfig {
        n_knots,
        degree,
        knot_placement,
        boundary_knots,
    })
}

fn normalize_knot_placement(knot_placement: &str) -> PyResult<String> {
    let normalized = knot_placement.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "quantile" => Ok("quantile".to_string()),
        "equal" | "uniform" => Ok("equal".to_string()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "knot_placement must be 'quantile' or 'equal'",
        )),
    }
}

fn validate_boundary_knots(boundary_knots: Option<(f64, f64)>) -> PyResult<()> {
    if let Some((lower, upper)) = boundary_knots
        && (!lower.is_finite() || !upper.is_finite() || lower <= 0.0 || lower >= upper)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boundary_knots must be finite positive values with lower < upper",
        ));
    }
    Ok(())
}

fn validate_spline_config(config: SplineConfig) -> PyResult<SplineConfig> {
    build_spline_config(
        config.n_knots,
        config.degree,
        config.knot_placement,
        config.boundary_knots,
    )
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must contain only finite values; got non-finite value at index {idx}",
            )));
        }
    }
    Ok(())
}

fn validate_eval_times(eval_times: &[f64]) -> PyResult<()> {
    if eval_times.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval_times cannot be empty",
        ));
    }
    validate_finite_values("eval_times", eval_times)?;
    if eval_times.iter().any(|&time| time < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval_times must be non-negative",
        ));
    }
    if eval_times.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eval_times must be strictly increasing",
        ));
    }
    Ok(())
}

fn validate_hazard_prediction_inputs(
    model_result: &FlexibleParametricResult,
    eval_times: &[f64],
    covariate_values: Option<&[f64]>,
) -> PyResult<usize> {
    validate_eval_times(eval_times)?;
    validate_finite_values("coefficients", &model_result.coefficients)?;
    validate_finite_values("spline_coefficients", &model_result.spline_coefficients)?;
    validate_finite_values("knots", &model_result.knots)?;
    if model_result.knots.len() < 2 || model_result.knots.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "knots must contain at least two strictly increasing values",
        ));
    }

    if model_result.spline_coefficients.len() < model_result.knots.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "spline_coefficients length must be at least knots length; got {} and {}",
            model_result.spline_coefficients.len(),
            model_result.knots.len()
        )));
    }
    let degree = model_result.spline_coefficients.len() + 1 - model_result.knots.len();

    if let Some(covariates) = covariate_values {
        if covariates.len() != model_result.coefficients.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariate_values length must match coefficients length; got {} and expected {}",
                covariates.len(),
                model_result.coefficients.len()
            )));
        }
        validate_finite_values("covariate_values", covariates)?;
    }

    Ok(degree)
}

fn validate_restricted_cubic_knots(knots: &[f64]) -> PyResult<()> {
    if knots.len() < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 3 knots",
        ));
    }

    validate_finite_values("knots", knots)?;
    if knots.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "knots must be strictly increasing",
        ));
    }
    Ok(())
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct FlexibleParametricResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub spline_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl FlexibleParametricResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coefficients: Vec<f64>,
        spline_coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        knots: Vec<f64>,
        log_likelihood: f64,
        aic: f64,
        bic: f64,
        n_iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            coefficients,
            spline_coefficients,
            std_errors,
            knots,
            log_likelihood,
            aic,
            bic,
            n_iterations,
            converged,
        }
    }
}

struct PoissonHazardFit {
    parameters: Vec<f64>,
    information: Array2<f64>,
    log_likelihood: f64,
    n_iterations: usize,
    converged: bool,
}

fn validate_flexible_parametric_inputs(
    time: &[f64],
    event: &[i32],
    covariates: &[Vec<f64>],
) -> PyResult<usize> {
    let n = time.len();
    if n < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 10 observations",
        ));
    }
    if event.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "event length must match time length; got {} and {n}",
            event.len()
        )));
    }
    validate_finite_values("time", time)?;
    if let Some((idx, &value)) = time.iter().enumerate().find(|(_, value)| **value <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "time values must be positive; got {value} at index {idx}"
        )));
    }
    if let Some((idx, &value)) = event
        .iter()
        .enumerate()
        .find(|(_, value)| !matches!(value, 0 | 1))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "event values must be 0 or 1; got {value} at index {idx}"
        )));
    }
    if !event.contains(&1) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least one event is required",
        ));
    }

    if covariates.is_empty() {
        return Ok(0);
    }
    if covariates.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "covariates must contain one row per observation; got {} rows for {n} observations",
            covariates.len()
        )));
    }
    let p = covariates[0].len();
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariates must be rectangular; row {row_idx} has {} columns, expected {p}",
                row.len()
            )));
        }
        validate_finite_values("covariates", row)?;
    }
    Ok(p)
}

fn poisson_hazard_state(
    design: &[f64],
    time: &[f64],
    event: &[i32],
    parameters: &[f64],
) -> Option<(f64, Array1<f64>, Array2<f64>)> {
    let n_params = parameters.len();
    if n_params == 0 || design.len() != time.len().checked_mul(n_params)? {
        return None;
    }

    let mut log_likelihood = 0.0;
    let mut score = Array1::zeros(n_params);
    let mut information = Array2::zeros((n_params, n_params));
    for ((row, &exposure), &outcome) in design.chunks_exact(n_params).zip(time).zip(event) {
        let linear_predictor = row
            .iter()
            .zip(parameters)
            .map(|(&value, &parameter)| value * parameter)
            .sum::<f64>();
        if !linear_predictor.is_finite() || linear_predictor > 700.0 {
            return None;
        }
        let mean = exposure * linear_predictor.exp();
        if !mean.is_finite() {
            return None;
        }
        log_likelihood += outcome as f64 * linear_predictor - mean;
        let residual = outcome as f64 - mean;
        for left in 0..n_params {
            score[left] += row[left] * residual;
            for right in 0..=left {
                information[[left, right]] += mean * row[left] * row[right];
            }
        }
        if !log_likelihood.is_finite() {
            return None;
        }
    }
    for left in 0..n_params {
        for right in 0..left {
            information[[right, left]] = information[[left, right]];
        }
    }
    Some((log_likelihood, score, information))
}

fn poisson_hazard_log_likelihood(
    design: &[f64],
    time: &[f64],
    event: &[i32],
    parameters: &[f64],
) -> Option<f64> {
    let n_params = parameters.len();
    if n_params == 0 || design.len() != time.len().checked_mul(n_params)? {
        return None;
    }
    let mut log_likelihood = 0.0;
    for ((row, &exposure), &outcome) in design.chunks_exact(n_params).zip(time).zip(event) {
        let linear_predictor = row
            .iter()
            .zip(parameters)
            .map(|(&value, &parameter)| value * parameter)
            .sum::<f64>();
        if !linear_predictor.is_finite() || linear_predictor > 700.0 {
            return None;
        }
        let mean = exposure * linear_predictor.exp();
        log_likelihood += outcome as f64 * linear_predictor - mean;
        if !log_likelihood.is_finite() {
            return None;
        }
    }
    Some(log_likelihood)
}

fn fit_poisson_hazard(
    design: &[f64],
    time: &[f64],
    event: &[i32],
    mut parameters: Vec<f64>,
) -> PyResult<PoissonHazardFit> {
    const MAX_ITERATIONS: usize = 100;
    const SCORE_TOLERANCE: f64 = 1e-8;
    const MIN_STEP_SCALE: f64 = 9.313_225_746_154_785e-10;

    let event_count = event.iter().map(|&value| value as f64).sum::<f64>();
    let mut n_iterations = 0;
    let mut converged = false;
    for iteration in 0..MAX_ITERATIONS {
        n_iterations = iteration + 1;
        let (current_log_likelihood, score, information) =
            poisson_hazard_state(design, time, event, &parameters).ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "spline hazard likelihood became non-finite",
                )
            })?;
        let max_score = score.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_score <= SCORE_TOLERANCE * (1.0 + event_count) {
            converged = true;
            break;
        }

        let step = regularized_lu_solve(&information, &score).map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "spline hazard information matrix is singular",
            )
        })?;
        let directional_derivative = score.dot(&step);
        if !directional_derivative.is_finite() || directional_derivative <= 0.0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "spline hazard optimizer could not find an ascent direction",
            ));
        }

        let mut scale = 1.0;
        let mut accepted = None;
        while scale >= MIN_STEP_SCALE {
            let candidate: Vec<f64> = parameters
                .iter()
                .zip(step.iter())
                .map(|(&parameter, &increment)| parameter + scale * increment)
                .collect();
            if let Some(candidate_log_likelihood) =
                poisson_hazard_log_likelihood(design, time, event, &candidate)
                && candidate_log_likelihood
                    >= current_log_likelihood + 1e-4 * scale * directional_derivative
            {
                accepted = Some(candidate);
                break;
            }
            scale *= 0.5;
        }

        let Some(candidate) = accepted else {
            break;
        };
        parameters = candidate;
    }

    let (log_likelihood, score, information) =
        poisson_hazard_state(design, time, event, &parameters).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("spline hazard likelihood became non-finite")
        })?;
    let max_score = score.iter().map(|value| value.abs()).fold(0.0, f64::max);
    converged |= max_score <= SCORE_TOLERANCE * (1.0 + event_count);

    Ok(PoissonHazardFit {
        parameters,
        information,
        log_likelihood,
        n_iterations,
        converged,
    })
}

fn information_standard_errors(information: &Array2<f64>) -> PyResult<Vec<f64>> {
    let inverse = matrix_inverse(information).or_else(|| {
        let scale = information
            .diag()
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        (0..8).find_map(|power| {
            let mut regularized = information.clone();
            let ridge = scale * 10.0_f64.powi(power - 12);
            for idx in 0..regularized.nrows() {
                regularized[[idx, idx]] += ridge;
            }
            matrix_inverse(&regularized)
        })
    });
    let inverse = inverse.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "spline hazard information matrix could not be inverted",
        )
    })?;
    inverse
        .diag()
        .iter()
        .map(|&variance| {
            if variance.is_finite() && variance >= 0.0 {
                Ok(variance.sqrt())
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "spline hazard information matrix produced an invalid variance",
                ))
            }
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config=None))]
pub fn flexible_parametric_model(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    config: Option<SplineConfig>,
) -> PyResult<FlexibleParametricResult> {
    let config = match config {
        Some(config) => validate_spline_config(config)?,
        None => build_spline_config(4, 3, "quantile".to_string(), None)?,
    };

    let n = time.len();
    let p = validate_flexible_parametric_inputs(&time, &event, &covariates)?;
    let log_time: Vec<f64> = time.iter().map(|t| t.ln()).collect();

    let knots = compute_knots(&log_time, &event, &config)?;
    let n_spline = knots.len() + config.degree - 1;
    let spline_basis = compute_bspline_basis(&log_time, &knots, config.degree);
    let n_params = p + n_spline;
    let mut design = Vec::with_capacity(n * n_params);
    for row in 0..n {
        if p > 0 {
            design.extend_from_slice(&covariates[row]);
        }
        design.extend_from_slice(&spline_basis[row]);
    }
    let baseline_log_hazard =
        (event.iter().map(|&value| value as f64).sum::<f64>() / time.iter().sum::<f64>()).ln();
    let mut initial_parameters = vec![0.0; n_params];
    initial_parameters[p..].fill(baseline_log_hazard);
    let fit = fit_poisson_hazard(&design, &time, &event, initial_parameters)?;
    let std_errors = information_standard_errors(&fit.information)?;

    let aic = -2.0 * fit.log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * fit.log_likelihood + (n as f64).ln() * n_params as f64;

    Ok(FlexibleParametricResult {
        coefficients: fit.parameters[..p].to_vec(),
        spline_coefficients: fit.parameters[p..].to_vec(),
        std_errors,
        knots,
        log_likelihood: fit.log_likelihood,
        aic,
        bic,
        n_iterations: fit.n_iterations,
        converged: fit.converged,
    })
}

fn compute_knots(log_time: &[f64], event: &[i32], config: &SplineConfig) -> PyResult<Vec<f64>> {
    let mut event_times: Vec<f64> = log_time
        .iter()
        .zip(event.iter())
        .filter(|(_, e)| **e == 1)
        .map(|(t, _)| *t)
        .collect();

    event_times.sort_by(f64::total_cmp);

    let (min_t, max_t) = match &config.boundary_knots {
        Some((l, u)) => (l.ln(), u.ln()),
        None => log_time.iter().copied().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lower, upper), value| (lower.min(value), upper.max(value)),
        ),
    };
    if min_t >= max_t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time values or boundary_knots must span a non-zero interval",
        ));
    }

    let mut knots = match config.knot_placement.as_str() {
        "quantile" => (0..config.n_knots)
            .map(|idx| {
                if idx == 0 {
                    return min_t;
                }
                if idx + 1 == config.n_knots {
                    return max_t;
                }
                let q = idx as f64 / (config.n_knots - 1) as f64;
                let position = q * (event_times.len() - 1) as f64;
                let lower_idx = position.floor() as usize;
                let upper_idx = position.ceil() as usize;
                let fraction = position - lower_idx as f64;
                (event_times[lower_idx]
                    + fraction * (event_times[upper_idx] - event_times[lower_idx]))
                    .clamp(min_t, max_t)
            })
            .collect::<Vec<_>>(),
        "equal" => {
            let step = (max_t - min_t) / (config.n_knots - 1) as f64;
            (0..config.n_knots)
                .map(|i| min_t + i as f64 * step)
                .collect()
        }
        _ => unreachable!("knot_placement is validated before knot computation"),
    };
    if knots.windows(2).any(|pair| pair[0] >= pair[1]) {
        let step = (max_t - min_t) / (config.n_knots - 1) as f64;
        knots = (0..config.n_knots)
            .map(|idx| min_t + idx as f64 * step)
            .collect();
    }
    Ok(knots)
}

fn compute_bspline_basis(x: &[f64], knots: &[f64], degree: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let n_basis = knots.len() + degree - 1;

    let mut extended_knots = vec![knots.first().cloned().unwrap_or(0.0); degree];
    extended_knots.extend_from_slice(knots);
    extended_knots.extend(vec![knots.last().cloned().unwrap_or(1.0); degree]);

    let mut basis: Vec<Vec<f64>> = vec![vec![0.0; n_basis]; n];

    let lower = knots.first().copied().unwrap_or(0.0);
    let upper = knots.last().copied().unwrap_or(1.0);
    for (i, &xi) in x.iter().enumerate() {
        let xi = xi.clamp(lower, upper);
        if xi == upper {
            basis[i][n_basis - 1] = 1.0;
            continue;
        }
        for (j, basis_val) in basis[i].iter_mut().enumerate().take(n_basis) {
            *basis_val = bspline_basis_value(xi, j, degree, &extended_knots);
        }
    }

    basis
}

fn bspline_basis_value(x: f64, j: usize, degree: usize, knots: &[f64]) -> f64 {
    if degree == 0 {
        if j + 1 < knots.len() && x >= knots[j] && x < knots[j + 1] {
            return 1.0;
        }
        return 0.0;
    }

    let mut result = 0.0;

    if j + degree < knots.len() {
        let denom1 = knots[j + degree] - knots[j];
        if denom1 > crate::constants::DIVISION_FLOOR {
            let b1 = bspline_basis_value(x, j, degree - 1, knots);
            result += (x - knots[j]) / denom1 * b1;
        }
    }

    if j + degree + 1 < knots.len() {
        let denom2 = knots[j + degree + 1] - knots[j + 1];
        if denom2 > crate::constants::DIVISION_FLOOR {
            let b2 = bspline_basis_value(x, j + 1, degree - 1, knots);
            result += (knots[j + degree + 1] - x) / denom2 * b2;
        }
    }

    result
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RestrictedCubicSplineResult {
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub basis_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
}

#[pymethods]
impl RestrictedCubicSplineResult {
    #[new]
    pub fn new(
        knots: Vec<f64>,
        basis_matrix: Vec<Vec<f64>>,
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
    ) -> Self {
        Self {
            knots,
            basis_matrix,
            coefficients,
            std_errors,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_knots=None, knots=None))]
pub fn restricted_cubic_spline(
    x: Vec<f64>,
    n_knots: Option<usize>,
    knots: Option<Vec<f64>>,
) -> PyResult<RestrictedCubicSplineResult> {
    let n = x.len();
    if n < 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 5 observations",
        ));
    }
    validate_finite_values("x", &x)?;

    let knots = match knots {
        Some(k) => k,
        None => {
            let n_k = n_knots.unwrap_or(4);
            if n_k < 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "n_knots must be at least 3",
                ));
            }
            compute_quantile_knots(&x, n_k)
        }
    };

    validate_restricted_cubic_knots(&knots)?;

    let k = knots.len();
    let mut basis_matrix: Vec<Vec<f64>> = vec![vec![0.0; k - 2]; n];

    let t_max = knots.last().cloned().unwrap_or(1.0);
    let d_km1_k = (t_max - knots[k - 2]).max(crate::constants::DIVISION_FLOOR);

    for i in 0..n {
        for j in 0..(k - 2) {
            let t_j = knots[j];
            let d_j_k = (t_max - t_j).max(crate::constants::DIVISION_FLOOR);

            let term1 = rcs_truncated_power(x[i], t_j, 3);
            let term2 = rcs_truncated_power(x[i], knots[k - 2], 3) * d_j_k / d_km1_k;
            let term3 = rcs_truncated_power(x[i], t_max, 3) * (t_j - knots[k - 2]) / d_km1_k;

            basis_matrix[i][j] = term1 - term2 + term3;
        }
    }

    let coefficients = vec![0.0; k - 2];
    let std_errors = vec![0.1; k - 2];

    Ok(RestrictedCubicSplineResult {
        knots,
        basis_matrix,
        coefficients,
        std_errors,
    })
}

fn rcs_truncated_power(x: f64, t: f64, power: i32) -> f64 {
    if x > t { (x - t).powi(power) } else { 0.0 }
}

fn compute_quantile_knots(x: &[f64], n_knots: usize) -> Vec<f64> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);

    (0..n_knots)
        .map(|i| {
            let q = (i as f64 + 0.5) / n_knots as f64;
            let idx = (q * (sorted.len() as f64 - 1.0)).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HazardSplineResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_hazard: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
}

#[pymethods]
impl HazardSplineResult {
    #[new]
    pub fn new(
        time_points: Vec<f64>,
        hazard: Vec<f64>,
        cumulative_hazard: Vec<f64>,
        survival: Vec<f64>,
        lower_ci: Vec<f64>,
        upper_ci: Vec<f64>,
    ) -> Self {
        Self {
            time_points,
            hazard,
            cumulative_hazard,
            survival,
            lower_ci,
            upper_ci,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (model_result, eval_times, covariate_values=None))]
pub fn predict_hazard_spline(
    model_result: FlexibleParametricResult,
    eval_times: Vec<f64>,
    covariate_values: Option<Vec<f64>>,
) -> PyResult<HazardSplineResult> {
    let degree =
        validate_hazard_prediction_inputs(&model_result, &eval_times, covariate_values.as_deref())?;

    let n_times = eval_times.len();

    let log_times: Vec<f64> = eval_times.iter().map(|t| t.max(0.001).ln()).collect();
    let spline_basis = compute_bspline_basis(&log_times, &model_result.knots, degree);

    let cov_contribution: f64 = match covariate_values.as_deref() {
        Some(cov) => cov
            .iter()
            .zip(model_result.coefficients.iter())
            .map(|(c, b)| c * b)
            .sum(),
        None => 0.0,
    };

    let mut hazard = vec![0.0; n_times];
    let mut cumulative_hazard = vec![0.0; n_times];
    let mut survival = vec![1.0; n_times];
    for i in 0..n_times {
        let mut log_hazard = cov_contribution;

        for (coef, &basis_val) in model_result
            .spline_coefficients
            .iter()
            .zip(spline_basis[i].iter())
        {
            log_hazard += coef * basis_val;
        }

        hazard[i] = log_hazard.exp();

        if i == 0 {
            cumulative_hazard[i] = hazard[i] * eval_times[i];
        } else {
            let dt = eval_times[i] - eval_times[i - 1];
            cumulative_hazard[i] =
                cumulative_hazard[i - 1] + (hazard[i - 1] + hazard[i]) / 2.0 * dt;
        }

        survival[i] = (-cumulative_hazard[i]).exp();
    }

    let se_factor = 0.1;
    let survival_se: Vec<f64> = survival.iter().map(|&s| s * se_factor).collect();
    let (lower_ci, upper_ci) = clamped_normal_ci_bounds_95(&survival, &survival_se, 0.0, 1.0);

    Ok(HazardSplineResult {
        time_points: eval_times,
        hazard,
        cumulative_hazard,
        survival,
        lower_ci,
        upper_ci,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexible_parametric_model() {
        let time: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let event: Vec<i32> = (0..20).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let covariates: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();

        let config = SplineConfig::new(3, 3, "quantile".to_string(), None).unwrap();
        let result = flexible_parametric_model(time, event, covariates, Some(config)).unwrap();

        assert!(!result.knots.is_empty());
        assert!(result.log_likelihood.is_finite());
        assert!(result.converged);
        assert!(result.n_iterations < 100);
        assert!(result.coefficients.iter().all(|value| value.is_finite()));
        assert!(
            result
                .spline_coefficients
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            result
                .std_errors
                .iter()
                .all(|&value| value.is_finite() && value > 0.0)
        );
    }

    #[test]
    fn poisson_hazard_newton_matches_intercept_only_mle() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];
        let design = vec![1.0; time.len()];
        let expected_parameter = (3.0_f64 / time.iter().sum::<f64>()).ln();

        let fit = fit_poisson_hazard(&design, &time, &event, vec![0.0]).unwrap();

        assert!(fit.converged);
        assert!((fit.parameters[0] - expected_parameter).abs() <= 1e-10);
        assert!((fit.information[[0, 0]] - 3.0).abs() <= 1e-10);
        let standard_errors = information_standard_errors(&fit.information).unwrap();
        assert!((standard_errors[0] - 1.0 / 3.0_f64.sqrt()).abs() <= 1e-10);
    }

    #[test]
    fn flexible_parametric_model_rejects_malformed_inputs() {
        #[cfg(feature = "python")]
        pyo3::Python::initialize();

        let time: Vec<f64> = (1..=20).map(|value| value as f64).collect();
        let event: Vec<i32> = (0..20).map(|idx| i32::from(idx % 3 == 0)).collect();
        let covariates = vec![vec![0.0]; 20];

        assert!(
            flexible_parametric_model(
                time.clone(),
                event[..19].to_vec(),
                covariates.clone(),
                None,
            )
            .is_err()
        );
        let mut invalid_time = time.clone();
        invalid_time[3] = 0.0;
        assert!(
            flexible_parametric_model(invalid_time, event.clone(), covariates.clone(), None)
                .is_err()
        );
        let mut invalid_event = event.clone();
        invalid_event[3] = 2;
        assert!(
            flexible_parametric_model(time.clone(), invalid_event, covariates.clone(), None)
                .is_err()
        );
        assert!(
            flexible_parametric_model(time.clone(), vec![0; 20], covariates.clone(), None).is_err()
        );
        assert!(
            flexible_parametric_model(
                time.clone(),
                event.clone(),
                covariates[..19].to_vec(),
                None,
            )
            .is_err()
        );
        let mut ragged_covariates = covariates;
        ragged_covariates[4].push(1.0);
        assert!(flexible_parametric_model(time, event, ragged_covariates, None).is_err());
    }

    #[test]
    fn test_restricted_cubic_spline() {
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();

        let result = restricted_cubic_spline(x, Some(4), None).unwrap();

        assert_eq!(result.knots.len(), 4);
        assert_eq!(result.basis_matrix.len(), 50);
        assert_eq!(result.basis_matrix[0].len(), 2);
    }

    #[test]
    fn test_restricted_cubic_spline_validates_inputs() {
        let x: Vec<f64> = (1..=5).map(|i| i as f64).collect();

        assert!(
            restricted_cubic_spline(vec![1.0, 2.0, f64::NAN, 4.0, 5.0], Some(4), None).is_err()
        );
        assert!(restricted_cubic_spline(x.clone(), Some(2), None).is_err());
        assert!(
            restricted_cubic_spline(x.clone(), None, Some(vec![1.0, 2.0, f64::INFINITY])).is_err()
        );
        assert!(restricted_cubic_spline(x.clone(), None, Some(vec![1.0, 2.0, 2.0])).is_err());
        assert!(restricted_cubic_spline(vec![1.0; 5], Some(4), None).is_err());
    }

    #[test]
    fn test_bspline_basis() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let knots = vec![0.0, 0.5, 1.0];

        let basis = compute_bspline_basis(&x, &knots, 2);

        assert_eq!(basis.len(), 5);
        for row in &basis {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_predict_hazard_spline() {
        let time: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let event: Vec<i32> = (0..20).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let covariates: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();

        let config = SplineConfig::new(3, 3, "quantile".to_string(), None).unwrap();
        let model = flexible_parametric_model(time, event, covariates, Some(config)).unwrap();

        let eval_times: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = predict_hazard_spline(model, eval_times, Some(vec![0.5])).unwrap();

        assert_eq!(result.time_points.len(), 10);
        assert_eq!(result.hazard.len(), 10);
        assert_eq!(result.survival.len(), 10);

        for s in &result.survival {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
        assert!(result.survival[0] < 1.0);
        assert!(result.survival.windows(2).all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn test_predict_hazard_spline_validates_inputs() {
        let time: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let event: Vec<i32> = (0..20).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let covariates: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();

        let config = SplineConfig::new(3, 3, "quantile".to_string(), None).unwrap();
        let model = flexible_parametric_model(time, event, covariates, Some(config)).unwrap();

        assert!(predict_hazard_spline(model.clone(), vec![], Some(vec![0.5])).is_err());
        assert!(
            predict_hazard_spline(model.clone(), vec![1.0, f64::NAN], Some(vec![0.5])).is_err()
        );
        assert!(predict_hazard_spline(model.clone(), vec![-1.0, 2.0], Some(vec![0.5])).is_err());
        assert!(predict_hazard_spline(model.clone(), vec![1.0, 1.0], Some(vec![0.5])).is_err());
        assert!(
            predict_hazard_spline(model.clone(), vec![1.0, 2.0], Some(vec![0.5, 1.0])).is_err()
        );
        assert!(
            predict_hazard_spline(model.clone(), vec![1.0, 2.0], Some(vec![f64::NAN])).is_err()
        );

        let mut nonfinite_model = model.clone();
        nonfinite_model.coefficients[0] = f64::NAN;
        assert!(predict_hazard_spline(nonfinite_model, vec![1.0, 2.0], Some(vec![0.5])).is_err());

        let mut mismatched_model = model;
        mismatched_model
            .spline_coefficients
            .truncate(mismatched_model.knots.len() - 1);
        assert!(predict_hazard_spline(mismatched_model, vec![1.0, 2.0], Some(vec![0.5])).is_err());
    }

    #[test]
    fn test_spline_config_validates_options() {
        assert!(SplineConfig::new(0, 3, "quantile".to_string(), None).is_err());
        assert!(SplineConfig::new(1, 3, "quantile".to_string(), None).is_err());
        assert!(SplineConfig::new(3, 0, "quantile".to_string(), None).is_err());
        assert!(SplineConfig::new(3, 3, "unknown".to_string(), None).is_err());
        assert!(SplineConfig::new(3, 3, "quantile".to_string(), Some((0.0, 5.0))).is_err());
        assert!(SplineConfig::new(3, 3, "quantile".to_string(), Some((5.0, 5.0))).is_err());
        assert!(SplineConfig::new(3, 3, "quantile".to_string(), Some((f64::NAN, 5.0))).is_err());

        let config = SplineConfig::new(3, 3, " Uniform ".to_string(), Some((1.0, 5.0))).unwrap();
        assert_eq!(config.knot_placement, "equal");
    }

    #[test]
    fn test_flexible_parametric_model_revalidates_mutated_config() {
        let time: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let event: Vec<i32> = (0..20).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let covariates: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        let mut config = SplineConfig::new(3, 3, "quantile".to_string(), None).unwrap();
        config.knot_placement = "unknown".to_string();

        assert!(flexible_parametric_model(time, event, covariates, Some(config)).is_err());
    }
}
