use pyo3::prelude::*;

use crate::constants::clamped_normal_ci_bounds_95;

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
    if n_knots == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_knots must be positive",
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
) -> PyResult<()> {
    validate_eval_times(eval_times)?;
    validate_finite_values("coefficients", &model_result.coefficients)?;
    validate_finite_values("spline_coefficients", &model_result.spline_coefficients)?;
    validate_finite_values("knots", &model_result.knots)?;

    let expected_spline_coefficients = model_result.knots.len() + 2;
    if model_result.spline_coefficients.len() != expected_spline_coefficients {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "spline_coefficients length must be knots length + 2 for cubic prediction; got {} and expected {}",
            model_result.spline_coefficients.len(),
            expected_spline_coefficients
        )));
    }

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

    Ok(())
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
    if n < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 10 observations",
        ));
    }

    let p = if !covariates.is_empty() && !covariates[0].is_empty() {
        covariates[0].len()
    } else {
        0
    };

    let log_time: Vec<f64> = time.iter().map(|t| t.max(0.001).ln()).collect();

    let knots = compute_knots(&log_time, &event, &config);
    let n_spline = knots.len() + config.degree - 1;

    let spline_basis = compute_bspline_basis(&log_time, &knots, config.degree);

    let mut beta: Vec<f64> = vec![0.0; p];
    let mut gamma: Vec<f64> = vec![0.0; n_spline];
    let mut converged = false;
    let mut n_iterations = 0;

    let learning_rate = 0.01;
    let max_iter = 500;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        let mut eta: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..n_spline.min(spline_basis[i].len()) {
                eta[i] += gamma[j] * spline_basis[i][j];
            }
            for j in 0..p {
                eta[i] += beta[j] * covariates[i][j];
            }
        }

        let hazard: Vec<f64> = eta.iter().map(|e| e.exp()).collect();

        let mut _log_lik = 0.0;
        for i in 0..n {
            if event[i] == 1 {
                _log_lik += eta[i];
            }
            _log_lik -= hazard[i] * time[i];
        }

        let mut grad_gamma: Vec<f64> = vec![0.0; n_spline];
        let mut grad_beta: Vec<f64> = vec![0.0; p];

        for i in 0..n {
            let residual = event[i] as f64 - hazard[i] * time[i];

            for j in 0..n_spline.min(spline_basis[i].len()) {
                grad_gamma[j] += spline_basis[i][j] * residual;
            }
            for j in 0..p {
                grad_beta[j] += covariates[i][j] * residual;
            }
        }

        let grad_norm: f64 = grad_gamma
            .iter()
            .chain(grad_beta.iter())
            .map(|g| g * g)
            .sum::<f64>()
            .sqrt();

        if grad_norm < 1e-6 {
            converged = true;
            break;
        }

        for j in 0..n_spline {
            gamma[j] += learning_rate * grad_gamma[j];
        }
        for j in 0..p {
            beta[j] += learning_rate * grad_beta[j];
        }
    }

    let mut eta: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for j in 0..n_spline.min(spline_basis[i].len()) {
            eta[i] += gamma[j] * spline_basis[i][j];
        }
        for j in 0..p {
            eta[i] += beta[j] * covariates[i][j];
        }
    }

    let hazard: Vec<f64> = eta.iter().map(|e| e.exp()).collect();

    let mut log_lik = 0.0;
    for i in 0..n {
        if event[i] == 1 {
            log_lik += eta[i];
        }
        log_lik -= hazard[i] * time[i];
    }

    let n_params = p + n_spline;
    let aic = -2.0 * log_lik + 2.0 * n_params as f64;
    let bic = -2.0 * log_lik + (n as f64).ln() * n_params as f64;

    let std_errors = compute_approximate_se(&beta, &gamma, n);

    Ok(FlexibleParametricResult {
        coefficients: beta,
        spline_coefficients: gamma,
        std_errors,
        knots,
        log_likelihood: log_lik,
        aic,
        bic,
        n_iterations,
        converged,
    })
}

fn compute_knots(log_time: &[f64], event: &[i32], config: &SplineConfig) -> Vec<f64> {
    let event_times: Vec<f64> = log_time
        .iter()
        .zip(event.iter())
        .filter(|(_, e)| **e == 1)
        .map(|(t, _)| *t)
        .collect();

    if event_times.is_empty() {
        return vec![0.0; config.n_knots];
    }

    let mut sorted_times = event_times.clone();
    sorted_times.sort_by(f64::total_cmp);

    let (min_t, max_t) = match &config.boundary_knots {
        Some((l, u)) => (l.ln(), u.ln()),
        None => (
            sorted_times.first().cloned().unwrap_or(0.0),
            sorted_times.last().cloned().unwrap_or(1.0),
        ),
    };

    match config.knot_placement.as_str() {
        "quantile" => (0..config.n_knots)
            .map(|i| {
                let q = (i as f64 + 1.0) / (config.n_knots as f64 + 1.0);
                let idx = (q * (sorted_times.len() as f64 - 1.0)).round() as usize;
                sorted_times[idx.min(sorted_times.len() - 1)]
            })
            .collect(),
        "equal" => {
            let step = (max_t - min_t) / (config.n_knots as f64 + 1.0);
            (0..config.n_knots)
                .map(|i| min_t + (i as f64 + 1.0) * step)
                .collect()
        }
        _ => unreachable!("knot_placement is validated before knot computation"),
    }
}

fn compute_bspline_basis(x: &[f64], knots: &[f64], degree: usize) -> Vec<Vec<f64>> {
    let n = x.len();
    let n_basis = knots.len() + degree - 1;

    let mut extended_knots = vec![knots.first().cloned().unwrap_or(0.0); degree];
    extended_knots.extend_from_slice(knots);
    extended_knots.extend(vec![knots.last().cloned().unwrap_or(1.0); degree]);

    let mut basis: Vec<Vec<f64>> = vec![vec![0.0; n_basis]; n];

    for (i, &xi) in x.iter().enumerate() {
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

fn compute_approximate_se(beta: &[f64], gamma: &[f64], n: usize) -> Vec<f64> {
    let mut se = Vec::with_capacity(beta.len() + gamma.len());

    for b in beta {
        se.push((b.abs() / (n as f64).sqrt()).max(0.01));
    }
    for g in gamma {
        se.push((g.abs() / (n as f64).sqrt()).max(0.01));
    }

    se
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
    validate_hazard_prediction_inputs(&model_result, &eval_times, covariate_values.as_deref())?;

    let n_times = eval_times.len();

    let log_times: Vec<f64> = eval_times.iter().map(|t| t.max(0.001).ln()).collect();
    let spline_basis = compute_bspline_basis(&log_times, &model_result.knots, 3);

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

        if i > 0 {
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
            assert!(sum >= 0.0);
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
        mismatched_model.spline_coefficients.pop();
        assert!(predict_hazard_spline(mismatched_model, vec![1.0, 2.0], Some(vec![0.5])).is_err());
    }

    #[test]
    fn test_spline_config_validates_options() {
        assert!(SplineConfig::new(0, 3, "quantile".to_string(), None).is_err());
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
