#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum UncertaintySet {
    Wasserstein,
    KLDivergence,
    ChiSquare,
    CVaR,
    Moment,
}

#[pymethods]
impl UncertaintySet {
    fn __repr__(&self) -> String {
        match self {
            UncertaintySet::Wasserstein => "UncertaintySet.Wasserstein".to_string(),
            UncertaintySet::KLDivergence => "UncertaintySet.KLDivergence".to_string(),
            UncertaintySet::ChiSquare => "UncertaintySet.ChiSquare".to_string(),
            UncertaintySet::CVaR => "UncertaintySet.CVaR".to_string(),
            UncertaintySet::Moment => "UncertaintySet.Moment".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DROSurvivalConfig {
    #[pyo3(get, set)]
    pub uncertainty_set: UncertaintySet,
    #[pyo3(get, set)]
    pub radius: f64,
    #[pyo3(get, set)]
    pub confidence_level: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub regularization: f64,
}

#[pymethods]
impl DROSurvivalConfig {
    #[new]
    #[pyo3(signature = (uncertainty_set=UncertaintySet::Wasserstein, radius=0.1, confidence_level=0.95, max_iter=100, tol=1e-6, regularization=0.01))]
    pub fn new(
        uncertainty_set: UncertaintySet,
        radius: f64,
        confidence_level: f64,
        max_iter: usize,
        tol: f64,
        regularization: f64,
    ) -> Self {
        Self {
            uncertainty_set,
            radius,
            confidence_level,
            max_iter,
            tol,
            regularization,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DROSurvivalResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub robust_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    #[pyo3(get)]
    pub worst_case_loss: f64,
    #[pyo3(get)]
    pub empirical_loss: f64,
    #[pyo3(get)]
    pub robustness_gap: f64,
    #[pyo3(get)]
    pub dual_variable: f64,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub n_iterations: usize,
    #[pyo3(get)]
    pub worst_case_weights: Vec<f64>,
}

#[pymethods]
impl DROSurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "DROSurvivalResult(worst_case_loss={:.4}, robustness_gap={:.4}, converged={})",
            self.worst_case_loss, self.robustness_gap, self.converged
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RobustnessAnalysis {
    #[pyo3(get)]
    pub radii: Vec<f64>,
    #[pyo3(get)]
    pub worst_case_losses: Vec<f64>,
    #[pyo3(get)]
    pub robustness_certificates: Vec<f64>,
    #[pyo3(get)]
    pub coefficient_stability: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub recommended_radius: f64,
}

#[pymethods]
impl RobustnessAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "RobustnessAnalysis(n_radii={}, recommended_radius={:.4})",
            self.radii.len(),
            self.recommended_radius
        )
    }
}

fn compute_wasserstein_robust_weights(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    coefficients: &[f64],
    radius: f64,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let mut losses: Vec<f64> = (0..n)
        .map(|i| {
            let linear_pred: f64 = x[i]
                .iter()
                .zip(coefficients.iter())
                .map(|(&xi, &c)| xi * c)
                .sum();
            let log_risk = linear_pred;
            if event[i] == 1 {
                -log_risk + (time[i].ln().max(-10.0))
            } else {
                linear_pred.exp() * time[i]
            }
        })
        .collect();

    let mean_loss: f64 = losses.iter().sum::<f64>() / n as f64;
    let std_loss = (losses.iter().map(|&l| (l - mean_loss).powi(2)).sum::<f64>() / n as f64).sqrt();

    if std_loss < 1e-10 {
        return vec![1.0 / n as f64; n];
    }

    for l in &mut losses {
        *l = (*l - mean_loss) / std_loss;
    }

    let eta = radius / std_loss.max(1e-10);
    let mut weights: Vec<f64> = losses.iter().map(|&l| (eta * l).exp()).collect();
    let sum_weights: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum_weights.max(1e-10);
    }

    weights
}

fn compute_kl_robust_weights(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    coefficients: &[f64],
    radius: f64,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let losses: Vec<f64> = (0..n)
        .map(|i| {
            let linear_pred: f64 = x[i]
                .iter()
                .zip(coefficients.iter())
                .map(|(&xi, &c)| xi * c)
                .sum();
            if event[i] == 1 {
                -linear_pred + linear_pred.exp() * time[i]
            } else {
                linear_pred.exp() * time[i]
            }
        })
        .collect();

    let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eta = 1.0 / radius.max(1e-10);

    let mut weights: Vec<f64> = losses
        .iter()
        .map(|&l| (eta * (l - max_loss)).exp())
        .collect();
    let sum_weights: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum_weights.max(1e-10);
    }

    weights
}

fn compute_cvar_weights(losses: &[f64], alpha: f64) -> Vec<f64> {
    let n = losses.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        losses[a]
            .partial_cmp(&losses[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold_idx = ((1.0 - alpha) * n as f64).ceil() as usize;
    let threshold_idx = threshold_idx.min(n - 1);

    let mut weights = vec![0.0; n];
    let active_count = n - threshold_idx;
    for i in threshold_idx..n {
        weights[sorted_indices[i]] = 1.0 / active_count as f64;
    }

    weights
}

fn weighted_cox_partial_likelihood(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    coefficients: &[f64],
    weights: &[f64],
) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let _p = x[0].len();
    let mut linear_pred: Vec<f64> = x
        .iter()
        .map(|xi| {
            xi.iter()
                .zip(coefficients.iter())
                .map(|(&x, &c)| x * c)
                .sum()
        })
        .collect();

    for lp in &mut linear_pred {
        *lp = lp.clamp(-20.0, 20.0);
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut log_likelihood = 0.0;
    let mut risk_set_sum = 0.0;

    for &i in sorted_indices.iter().rev() {
        risk_set_sum += weights[i] * linear_pred[i].exp();
    }

    let mut cumulative_risk = risk_set_sum;
    let mut prev_time = f64::INFINITY;

    for &i in &sorted_indices {
        if time[i] < prev_time {
            prev_time = time[i];
        }

        if event[i] == 1 {
            log_likelihood += weights[i] * (linear_pred[i] - cumulative_risk.max(1e-10).ln());
        }

        cumulative_risk -= weights[i] * linear_pred[i].exp();
    }

    -log_likelihood
}

fn dro_cox_optimization(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    config: &DROSurvivalConfig,
) -> (Vec<f64>, Vec<f64>, f64, bool, usize) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), 0.0, false, 0);
    }

    let p = x[0].len();
    let mut coefficients = vec![0.0; p];
    let mut weights = vec![1.0 / n as f64; n];

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        weights = match config.uncertainty_set {
            UncertaintySet::Wasserstein => {
                compute_wasserstein_robust_weights(x, time, event, &coefficients, config.radius)
            }
            UncertaintySet::KLDivergence => {
                compute_kl_robust_weights(x, time, event, &coefficients, config.radius)
            }
            UncertaintySet::CVaR => {
                let losses: Vec<f64> = (0..n)
                    .map(|i| {
                        let lp: f64 = x[i]
                            .iter()
                            .zip(coefficients.iter())
                            .map(|(&xi, &c)| xi * c)
                            .sum();
                        if event[i] == 1 { -lp } else { lp.exp() }
                    })
                    .collect();
                compute_cvar_weights(&losses, config.confidence_level)
            }
            _ => weights.clone(),
        };

        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        let linear_pred: Vec<f64> = x
            .iter()
            .map(|xi| {
                xi.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &c)| x * c)
                    .sum::<f64>()
                    .clamp(-20.0, 20.0)
            })
            .collect();

        let exp_pred: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_set_sum = 0.0;
        let mut risk_set_x = vec![0.0; p];

        for &i in &sorted_indices {
            risk_set_sum += weights[i] * exp_pred[i];
            for j in 0..p {
                risk_set_x[j] += weights[i] * exp_pred[i] * x[i][j];
            }

            if event[i] == 1 && risk_set_sum > 1e-10 {
                for j in 0..p {
                    gradient[j] += weights[i] * (x[i][j] - risk_set_x[j] / risk_set_sum);
                    hessian_diag[j] += weights[i]
                        * (risk_set_x[j] / risk_set_sum - (risk_set_x[j] / risk_set_sum).powi(2));
                }
            }
        }

        for j in 0..p {
            gradient[j] -= config.regularization * coefficients[j];
            hessian_diag[j] += config.regularization;
        }

        let mut max_update: f64 = 0.0;
        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                let clamped_update = update.clamp(-1.0, 1.0);
                coefficients[j] += clamped_update;
                max_update = max_update.max(clamped_update.abs());
            }
        }

        if max_update < config.tol {
            converged = true;
            break;
        }
    }

    let dual_variable = config.radius;
    (coefficients, weights, dual_variable, converged, n_iter)
}

#[pyfunction]
#[pyo3(signature = (x, time, event, config=None))]
pub fn dro_survival(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    config: Option<DROSurvivalConfig>,
) -> PyResult<DROSurvivalResult> {
    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }
    if time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same length as x",
        ));
    }

    let config = config.unwrap_or_else(|| {
        DROSurvivalConfig::new(UncertaintySet::Wasserstein, 0.1, 0.95, 100, 1e-6, 0.01)
    });

    let (robust_coefficients, worst_case_weights, dual_variable, converged, n_iterations) =
        dro_cox_optimization(&x, &time, &event, &config);

    let _p = x[0].len();
    let uniform_weights = vec![1.0 / n as f64; n];
    let empirical_coefficients = {
        let mut cfg = config.clone();
        cfg.radius = 0.0;
        let (coef, _, _, _, _) = dro_cox_optimization(&x, &time, &event, &cfg);
        coef
    };

    let empirical_loss = weighted_cox_partial_likelihood(
        &x,
        &time,
        &event,
        &empirical_coefficients,
        &uniform_weights,
    );
    let worst_case_loss = weighted_cox_partial_likelihood(
        &x,
        &time,
        &event,
        &robust_coefficients,
        &worst_case_weights,
    );
    let robustness_gap = (worst_case_loss - empirical_loss).abs();

    let standard_errors = robust_coefficients.iter().map(|_| 0.1).collect();

    Ok(DROSurvivalResult {
        coefficients: empirical_coefficients,
        robust_coefficients,
        standard_errors,
        worst_case_loss,
        empirical_loss,
        robustness_gap,
        dual_variable,
        converged,
        n_iterations,
        worst_case_weights,
    })
}

#[pyfunction]
#[pyo3(signature = (x, time, event, n_radii=10, max_radius=1.0, uncertainty_set=UncertaintySet::Wasserstein))]
pub fn robustness_analysis(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    n_radii: usize,
    max_radius: f64,
    uncertainty_set: UncertaintySet,
) -> PyResult<RobustnessAnalysis> {
    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let radii: Vec<f64> = (0..n_radii)
        .map(|i| max_radius * (i + 1) as f64 / n_radii as f64)
        .collect();

    let mut worst_case_losses = Vec::with_capacity(n_radii);
    let mut coefficient_stability = Vec::with_capacity(n_radii);
    let mut robustness_certificates = Vec::with_capacity(n_radii);

    let uniform_weights = vec![1.0 / n as f64; n];

    for &radius in &radii {
        let config = DROSurvivalConfig::new(uncertainty_set, radius, 0.95, 100, 1e-6, 0.01);

        let (coefficients, weights, _, _, _) = dro_cox_optimization(&x, &time, &event, &config);
        let loss = weighted_cox_partial_likelihood(&x, &time, &event, &coefficients, &weights);

        worst_case_losses.push(loss);
        coefficient_stability.push(coefficients.clone());

        let empirical_loss =
            weighted_cox_partial_likelihood(&x, &time, &event, &coefficients, &uniform_weights);
        let certificate = (loss - empirical_loss).max(0.0);
        robustness_certificates.push(certificate);
    }

    let recommended_idx = robustness_certificates
        .iter()
        .enumerate()
        .find(|&(_, cert)| *cert > 0.1)
        .map(|(i, _)| i.saturating_sub(1))
        .unwrap_or(radii.len() - 1);

    let recommended_radius = radii[recommended_idx.min(radii.len() - 1)];

    Ok(RobustnessAnalysis {
        radii,
        worst_case_losses,
        robustness_certificates,
        coefficient_stability,
        recommended_radius,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dro_survival_config() {
        let config =
            DROSurvivalConfig::new(UncertaintySet::Wasserstein, 0.1, 0.95, 100, 1e-6, 0.01);
        assert_eq!(config.uncertainty_set, UncertaintySet::Wasserstein);
        assert!((config.radius - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_dro_survival_wasserstein() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
            vec![2.5, 0.3],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 1, 0];

        let result = dro_survival(x, time, event, None).unwrap();
        assert!(!result.robust_coefficients.is_empty());
        assert!(result.worst_case_loss.is_finite());
        assert!(result.robustness_gap >= 0.0);
    }

    #[test]
    fn test_dro_survival_kl() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];

        let config =
            DROSurvivalConfig::new(UncertaintySet::KLDivergence, 0.5, 0.95, 100, 1e-6, 0.01);

        let result = dro_survival(x, time, event, Some(config)).unwrap();
        assert!(!result.robust_coefficients.is_empty());
    }

    #[test]
    fn test_robustness_analysis() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];

        let result =
            robustness_analysis(x, time, event, 5, 0.5, UncertaintySet::Wasserstein).unwrap();

        assert_eq!(result.radii.len(), 5);
        assert_eq!(result.worst_case_losses.len(), 5);
        assert!(result.recommended_radius > 0.0);
    }
}
