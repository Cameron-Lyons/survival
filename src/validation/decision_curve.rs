use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::same_time;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_non_negative, validate_probability_slice,
};

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn validate_threshold_value(value: f64, field: &'static str, index: usize) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(value_error(format!(
            "{field} must contain finite values between 0 and 1 exclusive; got {value} at index {index}"
        )));
    }
    Ok(())
}

fn normalize_thresholds(thresholds: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
    let thresholds = thresholds.unwrap_or_else(|| (1..100).map(|i| i as f64 / 100.0).collect());
    if thresholds.is_empty() {
        return Err(value_error("thresholds cannot be empty"));
    }
    for (index, &threshold) in thresholds.iter().enumerate() {
        validate_threshold_value(threshold, "thresholds", index)?;
    }
    Ok(thresholds)
}

fn validate_time_event_inputs(time: &[f64], event: &[i32]) -> PyResult<()> {
    if time.is_empty() || event.len() != time.len() {
        return Err(value_error("All inputs must have the same non-zero length"));
    }
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(event, "event")?;
    Ok(())
}

fn validate_time_horizon(time_horizon: f64) -> PyResult<()> {
    if !time_horizon.is_finite() || time_horizon < 0.0 {
        return Err(value_error(
            "time_horizon must be a finite non-negative value",
        ));
    }
    Ok(())
}

fn validate_decision_inputs(
    predicted_risk: &[f64],
    time: &[f64],
    event: &[i32],
    time_horizon: f64,
) -> PyResult<()> {
    if predicted_risk.is_empty()
        || time.len() != predicted_risk.len()
        || event.len() != predicted_risk.len()
    {
        return Err(value_error("All inputs must have the same non-zero length"));
    }
    validate_probability_slice(predicted_risk, "predicted_risk")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(event, "event")?;
    validate_time_horizon(time_horizon)?;
    Ok(())
}

fn binary_outcomes(time: &[f64], event: &[i32], time_horizon: f64) -> Vec<i32> {
    time.iter()
        .zip(event.iter())
        .map(|(&t, &e)| {
            if e == 1 && (t <= time_horizon || same_time(t, time_horizon)) {
                1
            } else {
                0
            }
        })
        .collect()
}

fn compute_net_benefit(predicted_risk: &[f64], outcomes: &[i32], thresholds: &[f64]) -> Vec<f64> {
    let n = predicted_risk.len();
    thresholds
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
            let odds = pt / (1.0 - pt);

            tpr - fpr * odds
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
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
    validate_decision_inputs(&predicted_risk, &time, &event, time_horizon)?;
    let thresholds = normalize_thresholds(thresholds)?;
    let n = predicted_risk.len();
    let outcomes = binary_outcomes(&time, &event, time_horizon);

    let n_events = outcomes.iter().filter(|&&o| o == 1).count();
    let prevalence = n_events as f64 / n as f64;

    let net_benefit_all: Vec<f64> = thresholds
        .iter()
        .map(|&pt| {
            let odds = pt / (1.0 - pt);
            prevalence - (1.0 - prevalence) * odds
        })
        .collect();

    let net_benefit_none: Vec<f64> = vec![0.0; thresholds.len()];
    let net_benefit = compute_net_benefit(&predicted_risk, &outcomes, &thresholds);

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
#[pyclass(from_py_object)]
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
    validate_decision_inputs(&predicted_risk, &time, &event, time_horizon)?;
    validate_threshold_value(threshold, "threshold", 0)?;
    let n = predicted_risk.len();
    let outcomes = binary_outcomes(&time, &event, time_horizon);

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

    let odds = threshold / (1.0 - threshold);
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
#[pyclass(from_py_object)]
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
        return Err(value_error(
            "model_predictions and model_names must have the same non-zero length",
        ));
    }
    validate_time_event_inputs(&time, &event)?;
    validate_time_horizon(time_horizon)?;

    let thresholds = normalize_thresholds(thresholds)?;
    let n = time.len();
    let outcomes = binary_outcomes(&time, &event, time_horizon);

    for (model_index, predictions) in model_predictions.iter().enumerate() {
        if predictions.len() != n {
            return Err(value_error(format!(
                "model_predictions row {model_index} length mismatch: expected {n}, got {}",
                predictions.len()
            )));
        }
        validate_probability_slice(predictions, "model_predictions")?;
    }

    let model_net_benefits: Vec<Vec<f64>> = model_predictions
        .iter()
        .map(|predictions| compute_net_benefit(predictions, &outcomes, &thresholds))
        .collect();

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
                    model_net_benefits[a][t_idx].total_cmp(&model_net_benefits[b][t_idx])
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

    #[test]
    fn test_decision_curve_apis_group_near_tied_time_horizon_events() {
        let exact_time = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let near_time = vec![
            1.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            3.0,
            4.0,
        ];
        let event = vec![0, 1, 0, 1, 0];
        let risk = vec![0.1, 0.8, 0.6, 0.7, 0.2];
        let challenger = vec![0.2, 0.7, 0.4, 0.8, 0.3];
        let thresholds = vec![0.25, 0.5, 0.75];

        let expected = decision_curve_analysis(
            risk.clone(),
            exact_time.clone(),
            event.clone(),
            2.0,
            Some(thresholds.clone()),
        )
        .unwrap();
        let actual = decision_curve_analysis(
            risk.clone(),
            near_time.clone(),
            event.clone(),
            2.0,
            Some(thresholds.clone()),
        )
        .unwrap();
        assert_eq!(actual.thresholds, expected.thresholds);
        assert_eq!(actual.net_benefit, expected.net_benefit);
        assert_eq!(actual.net_benefit_all, expected.net_benefit_all);
        assert_eq!(actual.net_benefit_none, expected.net_benefit_none);
        assert_eq!(actual.interventions_avoided, expected.interventions_avoided);

        let expected_clinical = clinical_utility_at_threshold(
            risk.clone(),
            exact_time.clone(),
            event.clone(),
            2.0,
            0.5,
        )
        .unwrap();
        let actual_clinical =
            clinical_utility_at_threshold(risk.clone(), near_time.clone(), event.clone(), 2.0, 0.5)
                .unwrap();
        assert_eq!(actual_clinical.sensitivity, expected_clinical.sensitivity);
        assert_eq!(actual_clinical.specificity, expected_clinical.specificity);
        assert_eq!(actual_clinical.ppv, expected_clinical.ppv);
        assert_eq!(actual_clinical.npv, expected_clinical.npv);
        assert_eq!(actual_clinical.net_benefit, expected_clinical.net_benefit);

        let expected_comparison = compare_decision_curves(
            vec![risk.clone(), challenger.clone()],
            vec!["m1".to_string(), "m2".to_string()],
            exact_time,
            event.clone(),
            2.0,
            Some(thresholds.clone()),
        )
        .unwrap();
        let actual_comparison = compare_decision_curves(
            vec![risk, challenger],
            vec!["m1".to_string(), "m2".to_string()],
            near_time,
            event,
            2.0,
            Some(thresholds),
        )
        .unwrap();
        assert_eq!(
            actual_comparison.net_benefit_difference,
            expected_comparison.net_benefit_difference
        );
        assert_eq!(
            actual_comparison.best_model_per_threshold,
            expected_comparison.best_model_per_threshold
        );
    }

    #[test]
    fn public_decision_curve_apis_validate_inputs() {
        assert!(
            decision_curve_analysis(vec![1.2], vec![1.0], vec![1], 1.0, Some(vec![0.5]))
                .expect_err("invalid risk probability should fail")
                .to_string()
                .contains("predicted_risk must contain probabilities")
        );
        assert!(
            decision_curve_analysis(vec![0.2], vec![f64::NAN], vec![1], 1.0, Some(vec![0.5]))
                .expect_err("non-finite time should fail")
                .to_string()
                .contains("time contains non-finite")
        );
        assert!(
            decision_curve_analysis(vec![0.2], vec![1.0], vec![2], 1.0, Some(vec![0.5]))
                .expect_err("non-binary event should fail")
                .to_string()
                .contains("event values must be 0 or 1")
        );
        assert!(
            decision_curve_analysis(
                vec![0.2],
                vec![1.0],
                vec![1],
                f64::INFINITY,
                Some(vec![0.5])
            )
            .expect_err("invalid time horizon should fail")
            .to_string()
            .contains("time_horizon")
        );
        assert!(
            decision_curve_analysis(vec![0.2], vec![1.0], vec![1], 1.0, Some(vec![]))
                .expect_err("empty thresholds should fail")
                .to_string()
                .contains("thresholds cannot be empty")
        );
        assert!(
            clinical_utility_at_threshold(vec![0.2], vec![1.0], vec![1], 1.0, 1.0)
                .expect_err("boundary threshold should fail")
                .to_string()
                .contains("threshold must contain finite values")
        );
        assert!(
            compare_decision_curves(
                vec![vec![0.2], vec![0.3, 0.4]],
                vec!["a".to_string(), "b".to_string()],
                vec![1.0],
                vec![1],
                1.0,
                Some(vec![0.5]),
            )
            .expect_err("model row length mismatch should fail")
            .to_string()
            .contains("model_predictions row 1 length mismatch")
        );
    }
}
