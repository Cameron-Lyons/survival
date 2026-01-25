#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct FairnessMetrics {
    #[pyo3(get)]
    pub demographic_parity: f64,
    #[pyo3(get)]
    pub equalized_odds: f64,
    #[pyo3(get)]
    pub calibration_difference: f64,
    #[pyo3(get)]
    pub group_c_indices: Vec<f64>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
}

#[pymethods]
impl FairnessMetrics {
    fn __repr__(&self) -> String {
        format!(
            "FairnessMetrics(demographic_parity={:.3}, equalized_odds={:.3})",
            self.demographic_parity, self.equalized_odds
        )
    }

    fn is_fair(&self, threshold: f64) -> bool {
        self.demographic_parity.abs() < threshold && self.equalized_odds.abs() < threshold
    }
}

fn compute_c_index(risk: &[f64], time: &[f64], event: &[i32]) -> f64 {
    let n = risk.len();
    if n < 2 {
        return 0.5;
    }

    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if event[i] == 1 && time[i] < time[j] {
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            } else if event[j] == 1 && time[j] < time[i] {
                comparable += 1.0;
                if risk[j] > risk[i] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }
    }

    if comparable > 0.0 {
        concordant / comparable
    } else {
        0.5
    }
}

#[pyfunction]
#[pyo3(signature = (
    risk_scores,
    time,
    event,
    protected_attribute,
    threshold_time=None
))]
pub fn compute_fairness_metrics(
    risk_scores: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    protected_attribute: Vec<i32>,
    threshold_time: Option<f64>,
) -> PyResult<FairnessMetrics> {
    let n = risk_scores.len();
    if n == 0 || time.len() != n || event.len() != n || protected_attribute.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let threshold_time =
        threshold_time.unwrap_or_else(|| time.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    let unique_groups: Vec<i32> = {
        let mut groups: Vec<i32> = protected_attribute.clone();
        groups.sort();
        groups.dedup();
        groups
    };

    let group_data: Vec<(Vec<f64>, Vec<f64>, Vec<i32>)> = unique_groups
        .iter()
        .map(|g| {
            let indices: Vec<usize> = protected_attribute
                .iter()
                .enumerate()
                .filter(|(_, attr)| **attr == *g)
                .map(|(i, _)| i)
                .collect();

            let risks: Vec<f64> = indices.iter().map(|&i| risk_scores[i]).collect();
            let times: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let events: Vec<i32> = indices.iter().map(|&i| event[i]).collect();

            (risks, times, events)
        })
        .collect();

    let group_sizes: Vec<usize> = group_data.iter().map(|(r, _, _)| r.len()).collect();

    let group_c_indices: Vec<f64> = group_data
        .par_iter()
        .map(|(risks, times, events)| compute_c_index(risks, times, events))
        .collect();

    let median_risk: f64 = {
        let mut sorted: Vec<f64> = risk_scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    let group_high_risk_rates: Vec<f64> = group_data
        .iter()
        .map(|(risks, _, _)| {
            let high_risk_count = risks.iter().filter(|&&r| r > median_risk).count();
            high_risk_count as f64 / risks.len().max(1) as f64
        })
        .collect();

    let demographic_parity = if group_high_risk_rates.len() >= 2 {
        group_high_risk_rates
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - group_high_risk_rates
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
    } else {
        0.0
    };

    let group_event_rates: Vec<f64> = group_data
        .iter()
        .map(|(risks, times, events)| {
            let high_risk_events: usize = risks
                .iter()
                .zip(times.iter())
                .zip(events.iter())
                .filter(|((r, t), e)| **r > median_risk && **t <= threshold_time && **e == 1)
                .count();
            let high_risk_total: usize = risks
                .iter()
                .zip(times.iter())
                .filter(|(r, t)| **r > median_risk && **t <= threshold_time)
                .count();
            if high_risk_total > 0 {
                high_risk_events as f64 / high_risk_total as f64
            } else {
                0.0
            }
        })
        .collect();

    let equalized_odds = if group_event_rates.len() >= 2 {
        group_event_rates
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - group_event_rates
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
    } else {
        0.0
    };

    let calibration_difference = if group_c_indices.len() >= 2 {
        group_c_indices
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - group_c_indices
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
    } else {
        0.0
    };

    Ok(FairnessMetrics {
        demographic_parity,
        equalized_odds,
        calibration_difference,
        group_c_indices,
        group_sizes,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RobustnessResult {
    #[pyo3(get)]
    pub original_c_index: f64,
    #[pyo3(get)]
    pub perturbed_c_indices: Vec<f64>,
    #[pyo3(get)]
    pub mean_sensitivity: f64,
    #[pyo3(get)]
    pub max_sensitivity: f64,
    #[pyo3(get)]
    pub robustness_score: f64,
}

#[pymethods]
impl RobustnessResult {
    fn __repr__(&self) -> String {
        format!(
            "RobustnessResult(robustness_score={:.3}, mean_sensitivity={:.3})",
            self.robustness_score, self.mean_sensitivity
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    risk_scores,
    time,
    event,
    noise_levels=None,
    n_perturbations=100,
    seed=None
))]
pub fn assess_model_robustness(
    risk_scores: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    noise_levels: Option<Vec<f64>>,
    n_perturbations: usize,
    seed: Option<u64>,
) -> PyResult<RobustnessResult> {
    let n = risk_scores.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let noise_levels = noise_levels.unwrap_or_else(|| vec![0.01, 0.05, 0.1, 0.2]);

    let mut rng = fastrand::Rng::new();
    if let Some(s) = seed {
        rng.seed(s);
    }

    let original_c_index = compute_c_index(&risk_scores, &time, &event);

    let risk_std: f64 = {
        let mean: f64 = risk_scores.iter().sum::<f64>() / n as f64;
        let var: f64 = risk_scores.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt()
    };

    let perturbed_c_indices: Vec<f64> = noise_levels
        .iter()
        .flat_map(|&noise| {
            (0..n_perturbations)
                .map(|_| {
                    let perturbed: Vec<f64> = risk_scores
                        .iter()
                        .map(|&r| {
                            let u1: f64 = rng.f64();
                            let u2: f64 = rng.f64();
                            let normal =
                                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                            r + noise * risk_std * normal
                        })
                        .collect();
                    compute_c_index(&perturbed, &time, &event)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let sensitivities: Vec<f64> = perturbed_c_indices
        .iter()
        .map(|&c| (c - original_c_index).abs())
        .collect();

    let mean_sensitivity = sensitivities.iter().sum::<f64>() / sensitivities.len().max(1) as f64;
    let max_sensitivity = sensitivities.iter().cloned().fold(0.0_f64, f64::max);

    let robustness_score = 1.0 - mean_sensitivity.min(1.0);

    Ok(RobustnessResult {
        original_c_index,
        perturbed_c_indices,
        mean_sensitivity,
        max_sensitivity,
        robustness_score,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SubgroupAnalysisResult {
    #[pyo3(get)]
    pub subgroup_names: Vec<String>,
    #[pyo3(get)]
    pub subgroup_sizes: Vec<usize>,
    #[pyo3(get)]
    pub subgroup_c_indices: Vec<f64>,
    #[pyo3(get)]
    pub subgroup_event_rates: Vec<f64>,
    #[pyo3(get)]
    pub overall_c_index: f64,
    #[pyo3(get)]
    pub disparity_ratio: f64,
}

#[pymethods]
impl SubgroupAnalysisResult {
    fn __repr__(&self) -> String {
        format!(
            "SubgroupAnalysisResult(n_subgroups={}, disparity_ratio={:.3})",
            self.subgroup_names.len(),
            self.disparity_ratio
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    risk_scores,
    time,
    event,
    subgroup_labels
))]
pub fn subgroup_analysis(
    risk_scores: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    subgroup_labels: Vec<String>,
) -> PyResult<SubgroupAnalysisResult> {
    let n = risk_scores.len();
    if n == 0 || time.len() != n || event.len() != n || subgroup_labels.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let unique_subgroups: Vec<String> = {
        let mut groups: Vec<String> = subgroup_labels.clone();
        groups.sort();
        groups.dedup();
        groups
    };

    let subgroup_data: Vec<(Vec<f64>, Vec<f64>, Vec<i32>)> = unique_subgroups
        .iter()
        .map(|g| {
            let indices: Vec<usize> = subgroup_labels
                .iter()
                .enumerate()
                .filter(|(_, label)| *label == g)
                .map(|(i, _)| i)
                .collect();

            let risks: Vec<f64> = indices.iter().map(|&i| risk_scores[i]).collect();
            let times: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let events: Vec<i32> = indices.iter().map(|&i| event[i]).collect();

            (risks, times, events)
        })
        .collect();

    let subgroup_sizes: Vec<usize> = subgroup_data.iter().map(|(r, _, _)| r.len()).collect();

    let subgroup_c_indices: Vec<f64> = subgroup_data
        .par_iter()
        .map(|(risks, times, events)| compute_c_index(risks, times, events))
        .collect();

    let subgroup_event_rates: Vec<f64> = subgroup_data
        .iter()
        .map(|(_, _, events)| {
            let n_events = events.iter().filter(|&&e| e == 1).count();
            n_events as f64 / events.len().max(1) as f64
        })
        .collect();

    let overall_c_index = compute_c_index(&risk_scores, &time, &event);

    let disparity_ratio = if subgroup_c_indices.len() >= 2 {
        let max_c = subgroup_c_indices
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_c = subgroup_c_indices
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_c > 0.0 {
            max_c / min_c
        } else {
            f64::INFINITY
        }
    } else {
        1.0
    };

    Ok(SubgroupAnalysisResult {
        subgroup_names: unique_subgroups,
        subgroup_sizes,
        subgroup_c_indices,
        subgroup_event_rates,
        overall_c_index,
        disparity_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_c_index() {
        let risk = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 1, 1, 1];
        let c = compute_c_index(&risk, &time, &event);
        assert!(c > 0.9);
    }

    #[test]
    fn test_fairness_metrics() {
        let risk = vec![0.8, 0.6, 0.7, 0.5, 0.3, 0.2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];
        let protected = vec![0, 0, 0, 1, 1, 1];

        let result = compute_fairness_metrics(risk, time, event, protected, None).unwrap();
        assert!(result.group_sizes.len() == 2);
    }

    #[test]
    fn test_robustness() {
        let risk = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 1, 1, 1];

        let result = assess_model_robustness(risk, time, event, None, 10, Some(42)).unwrap();
        assert!(result.robustness_score >= 0.0 && result.robustness_score <= 1.0);
    }
}
