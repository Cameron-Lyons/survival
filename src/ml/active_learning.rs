#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone)]
#[pyclass]
pub struct ActiveLearningConfig {
    #[pyo3(get, set)]
    pub strategy: String,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub uncertainty_threshold: f64,
    #[pyo3(get, set)]
    pub diversity_weight: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl ActiveLearningConfig {
    #[new]
    #[pyo3(signature = (
        strategy="uncertainty",
        batch_size=10,
        uncertainty_threshold=0.5,
        diversity_weight=0.3,
        seed=None
    ))]
    pub fn new(
        strategy: &str,
        batch_size: usize,
        uncertainty_threshold: f64,
        diversity_weight: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be positive",
            ));
        }
        Ok(Self {
            strategy: strategy.to_string(),
            batch_size,
            uncertainty_threshold,
            diversity_weight,
            seed,
        })
    }
}

fn compute_uncertainty_sampling(predictions: &[Vec<f64>]) -> Vec<f64> {
    predictions
        .par_iter()
        .map(|pred| {
            if pred.is_empty() {
                return 0.0;
            }
            let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
            let variance: f64 =
                pred.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / pred.len() as f64;
            variance.sqrt()
        })
        .collect()
}

fn compute_diversity_scores(covariates: &[Vec<f64>], selected: &[usize]) -> Vec<f64> {
    if selected.is_empty() {
        return vec![1.0; covariates.len()];
    }

    covariates
        .par_iter()
        .enumerate()
        .map(|(i, x)| {
            if selected.contains(&i) {
                return 0.0;
            }

            let min_dist: f64 = selected
                .iter()
                .map(|&j| {
                    let dist: f64 = x
                        .iter()
                        .zip(covariates[j].iter())
                        .map(|(&xi, &xj)| (xi - xj).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    dist
                })
                .fold(f64::INFINITY, f64::min);
            min_dist
        })
        .collect()
}

fn compute_qbc_disagreement(ensemble_predictions: &[Vec<Vec<f64>>]) -> Vec<f64> {
    let n_samples = if ensemble_predictions.is_empty() {
        0
    } else {
        ensemble_predictions[0].len()
    };

    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let predictions: Vec<f64> = ensemble_predictions
                .iter()
                .filter_map(|model_preds| model_preds.get(i))
                .map(|pred| pred.iter().sum::<f64>() / pred.len().max(1) as f64)
                .collect();

            if predictions.len() < 2 {
                return 0.0;
            }

            let mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
            predictions
                .iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ActiveLearningResult {
    #[pyo3(get)]
    pub selected_indices: Vec<usize>,
    #[pyo3(get)]
    pub uncertainty_scores: Vec<f64>,
    #[pyo3(get)]
    pub diversity_scores: Vec<f64>,
    #[pyo3(get)]
    pub combined_scores: Vec<f64>,
}

#[pymethods]
impl ActiveLearningResult {
    fn __repr__(&self) -> String {
        format!(
            "ActiveLearningResult(selected={}, max_uncertainty={:.4})",
            self.selected_indices.len(),
            self.uncertainty_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    labeled_indices,
    config
))]
pub fn active_learning_selection(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<Vec<f64>>,
    labeled_indices: Vec<usize>,
    config: ActiveLearningConfig,
) -> PyResult<ActiveLearningResult> {
    let n = covariates.len();
    if n == 0 || predictions.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates and predictions must have same non-zero length",
        ));
    }

    let uncertainty_scores = compute_uncertainty_sampling(&predictions);

    let mut selected_indices = Vec::new();
    let mut diversity_scores = vec![0.0; n];
    let mut combined_scores = vec![0.0; n];

    for _ in 0..config.batch_size {
        let current_diversity = compute_diversity_scores(&covariates, &selected_indices);

        for i in 0..n {
            if labeled_indices.contains(&i) || selected_indices.contains(&i) {
                combined_scores[i] = f64::NEG_INFINITY;
            } else {
                combined_scores[i] = (1.0 - config.diversity_weight) * uncertainty_scores[i]
                    + config.diversity_weight * current_diversity[i];
            }
        }

        if let Some((best_idx, _)) = combined_scores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            selected_indices.push(best_idx);
            diversity_scores = current_diversity;
        } else {
            break;
        }
    }

    Ok(ActiveLearningResult {
        selected_indices,
        uncertainty_scores,
        diversity_scores,
        combined_scores,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct QBCResult {
    #[pyo3(get)]
    pub selected_indices: Vec<usize>,
    #[pyo3(get)]
    pub disagreement_scores: Vec<f64>,
}

#[pymethods]
impl QBCResult {
    fn __repr__(&self) -> String {
        format!(
            "QBCResult(selected={}, max_disagreement={:.4})",
            self.selected_indices.len(),
            self.disagreement_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    ensemble_predictions,
    labeled_indices,
    batch_size
))]
pub fn query_by_committee(
    ensemble_predictions: Vec<Vec<Vec<f64>>>,
    labeled_indices: Vec<usize>,
    batch_size: usize,
) -> PyResult<QBCResult> {
    if ensemble_predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must have at least one model in ensemble",
        ));
    }

    let disagreement_scores = compute_qbc_disagreement(&ensemble_predictions);

    let mut scored_indices: Vec<(usize, f64)> = disagreement_scores
        .iter()
        .enumerate()
        .filter(|(i, _)| !labeled_indices.contains(i))
        .map(|(i, &s)| (i, s))
        .collect();

    scored_indices.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let selected_indices: Vec<usize> = scored_indices
        .iter()
        .take(batch_size)
        .map(|(i, _)| *i)
        .collect();

    Ok(QBCResult {
        selected_indices,
        disagreement_scores,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LogrankSampleSizeResult {
    #[pyo3(get)]
    pub required_events: usize,
    #[pyo3(get)]
    pub required_sample_size: usize,
    #[pyo3(get)]
    pub power: f64,
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub allocation_ratio: f64,
}

#[pymethods]
impl LogrankSampleSizeResult {
    fn __repr__(&self) -> String {
        format!(
            "LogrankSampleSizeResult(n={}, events={}, power={:.2}%)",
            self.required_sample_size,
            self.required_events,
            self.power * 100.0
        )
    }
}

#[allow(clippy::excessive_precision)]
fn standard_normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    let a1 = -3.969683028665376e1;
    let a2 = 2.209460984245205e2;
    let a3 = -2.759285104469687e2;
    let a4 = 1.383577518672690e2;
    let a5 = -3.066479806614716e1;
    let a6 = 2.506628277459239e0;

    let b1 = -5.447609879822406e1;
    let b2 = 1.615858368580409e2;
    let b3 = -1.556989798598866e2;
    let b4 = 6.680131188771972e1;
    let b5 = -1.328068155288572e1;

    let c1 = -7.784894002430293e-3;
    let c2 = -3.223964580411365e-1;
    let c3 = -2.400758277161838e0;
    let c4 = -2.549732539343734e0;
    let c5 = 4.374664141464968e0;
    let c6 = 2.938163982698783e0;

    let d1 = 7.784695709041462e-3;
    let d2 = 3.224671290700398e-1;
    let d3 = 2.445134137142996e0;
    let d4 = 3.754408661907416e0;

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    let result;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    result
}

#[pyfunction]
#[pyo3(signature = (
    hazard_ratio,
    power=0.8,
    alpha=0.05,
    allocation_ratio=1.0,
    event_rate=None,
    dropout_rate=0.0,
    accrual_time=None,
    follow_up_time=None
))]
pub fn sample_size_logrank(
    hazard_ratio: f64,
    power: f64,
    alpha: f64,
    allocation_ratio: f64,
    event_rate: Option<f64>,
    dropout_rate: f64,
    accrual_time: Option<f64>,
    follow_up_time: Option<f64>,
) -> PyResult<LogrankSampleSizeResult> {
    if hazard_ratio <= 0.0 || hazard_ratio == 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }
    if power <= 0.0 || power >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "power must be between 0 and 1",
        ));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "alpha must be between 0 and 1",
        ));
    }
    if allocation_ratio <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "allocation_ratio must be positive",
        ));
    }

    let z_alpha = standard_normal_quantile(1.0 - alpha / 2.0);
    let z_beta = standard_normal_quantile(power);

    let ln_hr = hazard_ratio.ln();
    let r = allocation_ratio;

    let required_events =
        ((z_alpha + z_beta).powi(2) * (1.0 + r).powi(2) / (r * ln_hr.powi(2))).ceil() as usize;

    let overall_event_rate = if let Some(rate) = event_rate {
        rate
    } else if let (Some(accrual), Some(follow_up)) = (accrual_time, follow_up_time) {
        let hazard = 0.1;
        1.0 - (-hazard * (accrual / 2.0 + follow_up)).exp()
    } else {
        0.5
    };

    let adjusted_rate = overall_event_rate * (1.0 - dropout_rate);
    let required_sample_size = (required_events as f64 / adjusted_rate).ceil() as usize;

    Ok(LogrankSampleSizeResult {
        required_events,
        required_sample_size,
        power,
        alpha,
        hazard_ratio,
        allocation_ratio,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LogrankPowerResult {
    #[pyo3(get)]
    pub power: f64,
    #[pyo3(get)]
    pub sample_size: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub alpha: f64,
}

#[pymethods]
impl LogrankPowerResult {
    fn __repr__(&self) -> String {
        format!(
            "LogrankPowerResult(power={:.2}%, n={}, HR={:.3})",
            self.power * 100.0,
            self.sample_size,
            self.hazard_ratio
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    sample_size,
    hazard_ratio,
    alpha=0.05,
    allocation_ratio=1.0,
    event_rate=0.5
))]
pub fn power_logrank(
    sample_size: usize,
    hazard_ratio: f64,
    alpha: f64,
    allocation_ratio: f64,
    event_rate: f64,
) -> PyResult<LogrankPowerResult> {
    if hazard_ratio <= 0.0 || hazard_ratio == 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }

    let n_events = (sample_size as f64 * event_rate).ceil() as usize;

    let z_alpha = standard_normal_quantile(1.0 - alpha / 2.0);

    let r = allocation_ratio;
    let ln_hr = hazard_ratio.ln();

    let se = ((1.0 + r).powi(2) / (r * n_events as f64)).sqrt();
    let z_stat = ln_hr.abs() / se;

    let power = normal_cdf(z_stat - z_alpha);

    Ok(LogrankPowerResult {
        power,
        sample_size,
        n_events,
        hazard_ratio,
        alpha,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AdaptiveDesignResult {
    #[pyo3(get)]
    pub stage: usize,
    #[pyo3(get)]
    pub continue_trial: bool,
    #[pyo3(get)]
    pub efficacy_boundary: f64,
    #[pyo3(get)]
    pub futility_boundary: f64,
    #[pyo3(get)]
    pub current_z_stat: f64,
    #[pyo3(get)]
    pub information_fraction: f64,
    #[pyo3(get)]
    pub conditional_power: f64,
}

#[pymethods]
impl AdaptiveDesignResult {
    fn __repr__(&self) -> String {
        let decision = if self.continue_trial {
            "continue"
        } else {
            "stop"
        };
        format!(
            "AdaptiveDesignResult(stage={}, decision={}, CP={:.2}%)",
            self.stage,
            decision,
            self.conditional_power * 100.0
        )
    }
}

fn obf_boundary(alpha: f64, info_fraction: f64) -> f64 {
    let z_alpha = standard_normal_quantile(1.0 - alpha / 2.0);
    z_alpha / info_fraction.sqrt()
}

fn pocock_boundary(alpha: f64, _info_fraction: f64, n_looks: usize) -> f64 {
    let adjusted_alpha = alpha / (n_looks as f64).sqrt();
    standard_normal_quantile(1.0 - adjusted_alpha / 2.0)
}

#[pyfunction]
#[pyo3(signature = (
    current_events,
    total_events,
    current_z_stat,
    stage,
    n_stages,
    alpha=0.05,
    futility_bound=0.0,
    boundary_type="obf"
))]
pub fn group_sequential_analysis(
    current_events: usize,
    total_events: usize,
    current_z_stat: f64,
    stage: usize,
    n_stages: usize,
    alpha: f64,
    futility_bound: f64,
    boundary_type: &str,
) -> PyResult<AdaptiveDesignResult> {
    if stage == 0 || stage > n_stages {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "stage must be between 1 and n_stages",
        ));
    }

    let info_fraction = current_events as f64 / total_events as f64;

    let efficacy_boundary = match boundary_type {
        "pocock" => pocock_boundary(alpha, info_fraction, n_stages),
        _ => obf_boundary(alpha, info_fraction),
    };

    let futility_boundary = futility_bound;

    let remaining_info = 1.0 - info_fraction;
    let conditional_power = if remaining_info > 0.0 && current_z_stat > 0.0 {
        let drift = current_z_stat / info_fraction.sqrt();
        let projected_z = drift * remaining_info.sqrt() + current_z_stat;
        let z_alpha = standard_normal_quantile(1.0 - alpha / 2.0);
        normal_cdf(projected_z - z_alpha)
    } else {
        0.0
    };

    let continue_trial =
        current_z_stat.abs() < efficacy_boundary && current_z_stat > futility_boundary;

    Ok(AdaptiveDesignResult {
        stage,
        continue_trial,
        efficacy_boundary,
        futility_boundary,
        current_z_stat,
        information_fraction: info_fraction,
        conditional_power,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_learning_selection() {
        let covariates = vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
            vec![0.9, 1.0],
        ];
        let predictions = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.5, 0.5, 0.5],
            vec![0.1, 0.9, 0.5],
            vec![0.2, 0.3, 0.25],
        ];
        let labeled = vec![0];
        let config = ActiveLearningConfig::new("uncertainty", 2, 0.5, 0.3, Some(42)).unwrap();

        let result = active_learning_selection(covariates, predictions, labeled, config).unwrap();
        assert_eq!(result.selected_indices.len(), 2);
        assert!(!result.selected_indices.contains(&0));
    }

    #[test]
    fn test_query_by_committee() {
        let ensemble = vec![
            vec![vec![0.1], vec![0.5], vec![0.9]],
            vec![vec![0.2], vec![0.4], vec![0.8]],
            vec![vec![0.15], vec![0.6], vec![0.85]],
        ];

        let result = query_by_committee(ensemble, vec![], 2).unwrap();
        assert_eq!(result.selected_indices.len(), 2);
    }

    #[test]
    fn test_sample_size_logrank() {
        let result = sample_size_logrank(0.7, 0.8, 0.05, 1.0, Some(0.5), 0.0, None, None).unwrap();
        assert!(result.required_events > 0);
        assert!(result.required_sample_size >= result.required_events);
    }

    #[test]
    fn test_power_logrank() {
        let result = power_logrank(200, 0.7, 0.05, 1.0, 0.5).unwrap();
        assert!(result.power >= 0.0 && result.power <= 1.0);
    }

    #[test]
    fn test_group_sequential_analysis() {
        let result = group_sequential_analysis(50, 200, 1.5, 1, 4, 0.05, 0.0, "obf").unwrap();
        assert!(result.information_fraction > 0.0);
        assert!(result.efficacy_boundary > 0.0);
    }
}
