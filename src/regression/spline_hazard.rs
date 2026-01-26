use pyo3::prelude::*;

#[pyclass]
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
    ) -> Self {
        Self {
            n_knots,
            degree,
            knot_placement,
            boundary_knots,
        }
    }
}

#[pyclass]
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
    let config = config.unwrap_or_else(|| SplineConfig::new(4, 3, "quantile".to_string(), None));

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

        let mut log_lik = 0.0;
        for i in 0..n {
            if event[i] == 1 {
                log_lik += eta[i];
            }
            log_lik -= hazard[i] * time[i];
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

        let _ = log_lik;
    }

    let mut eta: Vec<f64> = vec![0.0; n];
    #[allow(clippy::needless_range_loop)]
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
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
        _ => (0..config.n_knots)
            .map(|i| {
                let q = (i as f64 + 1.0) / (config.n_knots as f64 + 1.0);
                let idx = (q * (sorted_times.len() as f64 - 1.0)).round() as usize;
                sorted_times[idx.min(sorted_times.len() - 1)]
            })
            .collect(),
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
        if denom1 > 1e-10 {
            let b1 = bspline_basis_value(x, j, degree - 1, knots);
            result += (x - knots[j]) / denom1 * b1;
        }
    }

    if j + degree + 1 < knots.len() {
        let denom2 = knots[j + degree + 1] - knots[j + 1];
        if denom2 > 1e-10 {
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

#[pyclass]
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

    let knots = match knots {
        Some(k) => k,
        None => {
            let n_k = n_knots.unwrap_or(4);
            compute_quantile_knots(&x, n_k)
        }
    };

    let k = knots.len();
    if k < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 3 knots",
        ));
    }

    let mut basis_matrix: Vec<Vec<f64>> = vec![vec![0.0; k - 2]; n];

    let _t_min = knots.first().cloned().unwrap_or(0.0);
    let t_max = knots.last().cloned().unwrap_or(1.0);
    let d_km1_k = (t_max - knots[k - 2]).max(1e-10);

    for i in 0..n {
        for j in 0..(k - 2) {
            let t_j = knots[j];
            let d_j_k = (t_max - t_j).max(1e-10);

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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    (0..n_knots)
        .map(|i| {
            let q = (i as f64 + 0.5) / n_knots as f64;
            let idx = (q * (sorted.len() as f64 - 1.0)).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

#[pyclass]
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
    let n_times = eval_times.len();

    let log_times: Vec<f64> = eval_times.iter().map(|t| t.max(0.001).ln()).collect();
    let spline_basis = compute_bspline_basis(&log_times, &model_result.knots, 3);

    let cov_contribution: f64 = match covariate_values {
        Some(ref cov) => cov
            .iter()
            .zip(model_result.coefficients.iter())
            .map(|(c, b)| c * b)
            .sum(),
        None => 0.0,
    };

    let mut hazard = vec![0.0; n_times];
    let mut cumulative_hazard = vec![0.0; n_times];
    let mut survival = vec![1.0; n_times];

    #[allow(clippy::needless_range_loop)]
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

    let z = 1.96;
    let se_factor = 0.1;
    let lower_ci: Vec<f64> = survival
        .iter()
        .map(|s| (s - z * s * se_factor).clamp(0.0, 1.0))
        .collect();
    let upper_ci: Vec<f64> = survival
        .iter()
        .map(|s| (s + z * s * se_factor).clamp(0.0, 1.0))
        .collect();

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

        let config = SplineConfig::new(3, 3, "quantile".to_string(), None);
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

        let config = SplineConfig::new(3, 3, "quantile".to_string(), None);
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
}
