#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DecisionCurveResult {
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub net_benefit: Vec<f64>,
    #[pyo3(get)]
    pub net_benefit_all: Vec<f64>,
    #[pyo3(get)]
    pub net_benefit_none: Vec<f64>,
    #[pyo3(get)]
    pub interventions_avoided: Vec<f64>,
}

#[pymethods]
impl DecisionCurveResult {
    fn __repr__(&self) -> String {
        format!(
            "DecisionCurveResult(n_thresholds={})",
            self.thresholds.len()
        )
    }

    fn optimal_threshold(&self) -> f64 {
        let max_idx = self
            .net_benefit
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.thresholds[max_idx]
    }

    fn area_under_curve(&self) -> f64 {
        if self.thresholds.len() < 2 {
            return 0.0;
        }

        let mut auc = 0.0;
        for i in 1..self.thresholds.len() {
            let dt = self.thresholds[i] - self.thresholds[i - 1];
            let avg_nb = (self.net_benefit[i] + self.net_benefit[i - 1]) / 2.0;
            auc += dt * avg_nb.max(0.0);
        }
        auc
    }
}

#[pyfunction]
#[pyo3(signature = (
    predicted_risk,
    time,
    event,
    time_horizon,
    thresholds=None
))]
pub fn decision_curve_analysis(
    predicted_risk: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_horizon: f64,
    thresholds: Option<Vec<f64>>,
) -> PyResult<DecisionCurveResult> {
    let n = predicted_risk.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let thresholds = thresholds.unwrap_or_else(|| (1..100).map(|i| i as f64 / 100.0).collect());

    let outcomes: Vec<i32> = time
        .iter()
        .zip(event.iter())
        .map(|(&t, &e)| if t <= time_horizon && e == 1 { 1 } else { 0 })
        .collect();

    let n_events = outcomes.iter().filter(|&&o| o == 1).count();
    let prevalence = n_events as f64 / n as f64;

    let net_benefit_all: Vec<f64> = thresholds
        .iter()
        .map(|&pt| {
            let odds = pt / (1.0 - pt).max(1e-10);
            prevalence - (1.0 - prevalence) * odds
        })
        .collect();

    let net_benefit_none: Vec<f64> = vec![0.0; thresholds.len()];

    let net_benefit: Vec<f64> = thresholds
        .par_iter()
        .map(|&pt| {
            let mut tp = 0;
            let mut fp = 0;

            for i in 0..n {
                if predicted_risk[i] >= pt {
                    if outcomes[i] == 1 {
                        tp += 1;
                    } else {
                        fp += 1;
                    }
                }
            }

            let tpr = tp as f64 / n as f64;
            let fpr = fp as f64 / n as f64;
            let odds = pt / (1.0 - pt).max(1e-10);

            tpr - fpr * odds
        })
        .collect();

    let interventions_avoided: Vec<f64> = net_benefit
        .iter()
        .zip(net_benefit_all.iter())
        .map(|(&nb, &nb_all)| {
            if nb_all > 0.0 {
                ((nb - nb_all) / nb_all).max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    Ok(DecisionCurveResult {
        thresholds,
        net_benefit,
        net_benefit_all,
        net_benefit_none,
        interventions_avoided,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ClinicalUtilityResult {
    #[pyo3(get)]
    pub threshold: f64,
    #[pyo3(get)]
    pub sensitivity: f64,
    #[pyo3(get)]
    pub specificity: f64,
    #[pyo3(get)]
    pub ppv: f64,
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub nnt: f64,
    #[pyo3(get)]
    pub net_benefit: f64,
}

#[pymethods]
impl ClinicalUtilityResult {
    fn __repr__(&self) -> String {
        format!(
            "ClinicalUtilityResult(threshold={:.2}, NNT={:.1}, NB={:.3})",
            self.threshold, self.nnt, self.net_benefit
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    predicted_risk,
    time,
    event,
    time_horizon,
    threshold
))]
pub fn clinical_utility_at_threshold(
    predicted_risk: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_horizon: f64,
    threshold: f64,
) -> PyResult<ClinicalUtilityResult> {
    let n = predicted_risk.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let outcomes: Vec<i32> = time
        .iter()
        .zip(event.iter())
        .map(|(&t, &e)| if t <= time_horizon && e == 1 { 1 } else { 0 })
        .collect();

    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;

    for i in 0..n {
        let predicted_positive = predicted_risk[i] >= threshold;
        let actual_positive = outcomes[i] == 1;

        match (predicted_positive, actual_positive) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    let sensitivity = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };

    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };

    let ppv = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };

    let npv = if tn + fn_ > 0 {
        tn as f64 / (tn + fn_) as f64
    } else {
        0.0
    };

    let nnt = if ppv > 0.0 { 1.0 / ppv } else { f64::INFINITY };

    let odds = threshold / (1.0 - threshold).max(1e-10);
    let net_benefit = (tp as f64 / n as f64) - (fp as f64 / n as f64) * odds;

    Ok(ClinicalUtilityResult {
        threshold,
        sensitivity,
        specificity,
        ppv,
        npv,
        nnt,
        net_benefit,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ModelComparisonResult {
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub net_benefit_difference: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub best_model_per_threshold: Vec<String>,
}

#[pymethods]
impl ModelComparisonResult {
    fn __repr__(&self) -> String {
        format!(
            "ModelComparisonResult(n_models={}, n_thresholds={})",
            self.model_names.len(),
            self.thresholds.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    model_predictions,
    model_names,
    time,
    event,
    time_horizon,
    thresholds=None
))]
pub fn compare_decision_curves(
    model_predictions: Vec<Vec<f64>>,
    model_names: Vec<String>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_horizon: f64,
    thresholds: Option<Vec<f64>>,
) -> PyResult<ModelComparisonResult> {
    let n_models = model_predictions.len();
    if n_models == 0 || model_names.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "model_predictions and model_names must have the same non-zero length",
        ));
    }

    let thresholds = thresholds.unwrap_or_else(|| (1..100).map(|i| i as f64 / 100.0).collect());

    let mut model_net_benefits: Vec<Vec<f64>> = Vec::new();

    for predictions in &model_predictions {
        let result = decision_curve_analysis(
            predictions.clone(),
            time.clone(),
            event.clone(),
            time_horizon,
            Some(thresholds.clone()),
        )?;
        model_net_benefits.push(result.net_benefit);
    }

    let net_benefit_difference: Vec<Vec<f64>> = (0..n_models)
        .map(|i| {
            (0..n_models)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        let mean_diff: f64 = model_net_benefits[i]
                            .iter()
                            .zip(model_net_benefits[j].iter())
                            .map(|(&a, &b)| a - b)
                            .sum::<f64>()
                            / thresholds.len() as f64;
                        mean_diff
                    }
                })
                .collect()
        })
        .collect();

    let best_model_per_threshold: Vec<String> = thresholds
        .iter()
        .enumerate()
        .map(|(t_idx, _)| {
            let best_idx = (0..n_models)
                .max_by(|&a, &b| {
                    model_net_benefits[a][t_idx]
                        .partial_cmp(&model_net_benefits[b][t_idx])
                        .unwrap()
                })
                .unwrap_or(0);
            model_names[best_idx].clone()
        })
        .collect();

    Ok(ModelComparisonResult {
        model_names,
        net_benefit_difference,
        thresholds,
        best_model_per_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_curve() {
        let predicted = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![0, 1, 0, 1, 1];

        let result = decision_curve_analysis(predicted, time, event, 3.0, None).unwrap();
        assert!(!result.thresholds.is_empty());
        assert_eq!(result.net_benefit.len(), result.thresholds.len());
    }

    #[test]
    fn test_clinical_utility() {
        let predicted = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![0, 1, 0, 1, 1];

        let result = clinical_utility_at_threshold(predicted, time, event, 3.0, 0.5).unwrap();
        assert!(result.sensitivity >= 0.0 && result.sensitivity <= 1.0);
        assert!(result.specificity >= 0.0 && result.specificity <= 1.0);
    }
}
