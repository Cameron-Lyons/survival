#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct ModelPerformanceMetrics {
    #[pyo3(get, set)]
    pub c_index: f64,
    #[pyo3(get, set)]
    pub brier_score: f64,
    #[pyo3(get, set)]
    pub integrated_brier: f64,
    #[pyo3(get, set)]
    pub calibration_slope: f64,
    #[pyo3(get, set)]
    pub calibration_intercept: f64,
    #[pyo3(get, set)]
    pub ci_lower_c_index: f64,
    #[pyo3(get, set)]
    pub ci_upper_c_index: f64,
}

#[pymethods]
impl ModelPerformanceMetrics {
    #[new]
    #[pyo3(signature = (
        c_index,
        brier_score=None,
        integrated_brier=None,
        calibration_slope=None,
        calibration_intercept=None,
        ci_lower_c_index=None,
        ci_upper_c_index=None
    ))]
    pub fn new(
        c_index: f64,
        brier_score: Option<f64>,
        integrated_brier: Option<f64>,
        calibration_slope: Option<f64>,
        calibration_intercept: Option<f64>,
        ci_lower_c_index: Option<f64>,
        ci_upper_c_index: Option<f64>,
    ) -> Self {
        Self {
            c_index,
            brier_score: brier_score.unwrap_or(f64::NAN),
            integrated_brier: integrated_brier.unwrap_or(f64::NAN),
            calibration_slope: calibration_slope.unwrap_or(1.0),
            calibration_intercept: calibration_intercept.unwrap_or(0.0),
            ci_lower_c_index: ci_lower_c_index.unwrap_or(c_index - 0.05),
            ci_upper_c_index: ci_upper_c_index.unwrap_or(c_index + 0.05),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelPerformanceMetrics(C-index={:.3}, Brier={:.3})",
            self.c_index, self.brier_score
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SubgroupPerformance {
    #[pyo3(get)]
    pub subgroup_name: String,
    #[pyo3(get)]
    pub n_samples: usize,
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub event_rate: f64,
}

#[pymethods]
impl SubgroupPerformance {
    fn __repr__(&self) -> String {
        format!(
            "SubgroupPerformance({}, n={}, C={:.3})",
            self.subgroup_name, self.n_samples, self.c_index
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ModelCard {
    #[pyo3(get)]
    pub model_name: String,
    #[pyo3(get)]
    pub model_type: String,
    #[pyo3(get)]
    pub version: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub intended_use: String,
    #[pyo3(get)]
    pub limitations: Vec<String>,
    #[pyo3(get)]
    pub training_data_description: String,
    #[pyo3(get)]
    pub n_training_samples: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub feature_names: Vec<String>,
    #[pyo3(get)]
    pub overall_performance: ModelPerformanceMetrics,
    #[pyo3(get)]
    pub subgroup_performance: Vec<SubgroupPerformance>,
    #[pyo3(get)]
    pub ethical_considerations: Vec<String>,
    #[pyo3(get)]
    pub caveats: Vec<String>,
}

#[pymethods]
impl ModelCard {
    fn __repr__(&self) -> String {
        format!(
            "ModelCard(name={}, type={}, C-index={:.3})",
            self.model_name, self.model_type, self.overall_performance.c_index
        )
    }

    fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Model Card: {}\n\n", self.model_name));
        md.push_str(&format!("**Version:** {}\n\n", self.version));
        md.push_str(&format!("**Model Type:** {}\n\n", self.model_type));

        md.push_str("## Description\n\n");
        md.push_str(&format!("{}\n\n", self.description));

        md.push_str("## Intended Use\n\n");
        md.push_str(&format!("{}\n\n", self.intended_use));

        md.push_str("## Training Data\n\n");
        md.push_str(&format!("{}\n\n", self.training_data_description));
        md.push_str(&format!("- **Samples:** {}\n", self.n_training_samples));
        md.push_str(&format!("- **Events:** {}\n", self.n_events));
        md.push_str(&format!(
            "- **Event Rate:** {:.1}%\n\n",
            100.0 * self.n_events as f64 / self.n_training_samples as f64
        ));

        md.push_str("## Features\n\n");
        for (i, name) in self.feature_names.iter().enumerate() {
            md.push_str(&format!("{}. {}\n", i + 1, name));
        }
        md.push('\n');

        md.push_str("## Overall Performance\n\n");
        md.push_str("| Metric | Value | 95% CI |\n|--------|-------|--------|\n");
        md.push_str(&format!(
            "| C-index | {:.3} | [{:.3}, {:.3}] |\n",
            self.overall_performance.c_index,
            self.overall_performance.ci_lower_c_index,
            self.overall_performance.ci_upper_c_index
        ));
        if !self.overall_performance.brier_score.is_nan() {
            md.push_str(&format!(
                "| Brier Score | {:.3} | - |\n",
                self.overall_performance.brier_score
            ));
        }
        md.push('\n');

        if !self.subgroup_performance.is_empty() {
            md.push_str("## Subgroup Performance\n\n");
            md.push_str("| Subgroup | N | C-index | Event Rate |\n");
            md.push_str("|----------|---|---------|------------|\n");
            for sg in &self.subgroup_performance {
                md.push_str(&format!(
                    "| {} | {} | {:.3} | {:.1}% |\n",
                    sg.subgroup_name,
                    sg.n_samples,
                    sg.c_index,
                    sg.event_rate * 100.0
                ));
            }
            md.push('\n');
        }

        if !self.limitations.is_empty() {
            md.push_str("## Limitations\n\n");
            for lim in &self.limitations {
                md.push_str(&format!("- {}\n", lim));
            }
            md.push('\n');
        }

        if !self.ethical_considerations.is_empty() {
            md.push_str("## Ethical Considerations\n\n");
            for eth in &self.ethical_considerations {
                md.push_str(&format!("- {}\n", eth));
            }
            md.push('\n');
        }

        if !self.caveats.is_empty() {
            md.push_str("## Caveats and Recommendations\n\n");
            for cav in &self.caveats {
                md.push_str(&format!("- {}\n", cav));
            }
        }

        md
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"model_name":"{}","model_type":"{}","version":"{}","c_index":{:.4},"n_samples":{}}}"#,
            self.model_name,
            self.model_type,
            self.version,
            self.overall_performance.c_index,
            self.n_training_samples
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    model_name,
    model_type,
    version,
    description,
    intended_use,
    training_data_description,
    n_training_samples,
    n_events,
    feature_names,
    overall_performance,
    subgroup_performance=None,
    limitations=None,
    ethical_considerations=None,
    caveats=None
))]
pub fn create_model_card(
    model_name: String,
    model_type: String,
    version: String,
    description: String,
    intended_use: String,
    training_data_description: String,
    n_training_samples: usize,
    n_events: usize,
    feature_names: Vec<String>,
    overall_performance: ModelPerformanceMetrics,
    subgroup_performance: Option<Vec<SubgroupPerformance>>,
    limitations: Option<Vec<String>>,
    ethical_considerations: Option<Vec<String>>,
    caveats: Option<Vec<String>>,
) -> PyResult<ModelCard> {
    Ok(ModelCard {
        model_name,
        model_type,
        version,
        description,
        intended_use,
        limitations: limitations.unwrap_or_default(),
        training_data_description,
        n_training_samples,
        n_events,
        feature_names,
        overall_performance,
        subgroup_performance: subgroup_performance.unwrap_or_default(),
        ethical_considerations: ethical_considerations.unwrap_or_default(),
        caveats: caveats.unwrap_or_default(),
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FairnessAuditResult {
    #[pyo3(get)]
    pub protected_attribute: String,
    #[pyo3(get)]
    pub group_names: Vec<String>,
    #[pyo3(get)]
    pub group_c_indices: Vec<f64>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
    #[pyo3(get)]
    pub max_disparity: f64,
    #[pyo3(get)]
    pub passes_threshold: bool,
}

#[pymethods]
impl FairnessAuditResult {
    fn __repr__(&self) -> String {
        format!(
            "FairnessAuditResult(attr={}, disparity={:.3}, passes={})",
            self.protected_attribute, self.max_disparity, self.passes_threshold
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    predictions,
    time,
    event,
    protected_attribute,
    group_labels,
    disparity_threshold=0.1
))]
pub fn fairness_audit(
    predictions: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    protected_attribute: String,
    group_labels: Vec<String>,
    disparity_threshold: f64,
) -> PyResult<FairnessAuditResult> {
    let n = predictions.len();
    if n == 0 || time.len() != n || event.len() != n || group_labels.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut unique_groups: Vec<String> = group_labels.clone();
    unique_groups.sort();
    unique_groups.dedup();

    let mut group_c_indices = Vec::new();
    let mut group_sizes = Vec::new();
    let mut group_names = Vec::new();

    for group in &unique_groups {
        let indices: Vec<usize> = group_labels
            .iter()
            .enumerate()
            .filter(|(_, g)| *g == group)
            .map(|(i, _)| i)
            .collect();

        if indices.len() < 10 {
            continue;
        }

        let group_pred: Vec<f64> = indices.iter().map(|&i| predictions[i]).collect();
        let group_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let group_event: Vec<i32> = indices.iter().map(|&i| event[i]).collect();

        let c_index = compute_c_index(&group_pred, &group_time, &group_event);

        group_names.push(group.clone());
        group_c_indices.push(c_index);
        group_sizes.push(indices.len());
    }

    let max_c = group_c_indices
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_c = group_c_indices
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_disparity = max_c - min_c;

    Ok(FairnessAuditResult {
        protected_attribute,
        group_names,
        group_c_indices,
        group_sizes,
        max_disparity,
        passes_threshold: max_disparity <= disparity_threshold,
    })
}

fn compute_c_index(predictions: &[f64], time: &[f64], event: &[i32]) -> f64 {
    let n = predictions.len();
    let mut concordant = 0.0;
    let mut discordant = 0.0;

    for i in 0..n {
        if event[i] != 1 {
            continue;
        }
        for j in 0..n {
            if i == j || time[j] <= time[i] {
                continue;
            }
            if predictions[i] > predictions[j] {
                concordant += 1.0;
            } else if predictions[i] < predictions[j] {
                discordant += 1.0;
            } else {
                concordant += 0.5;
                discordant += 0.5;
            }
        }
    }

    if concordant + discordant > 0.0 {
        concordant / (concordant + discordant)
    } else {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_performance_metrics() {
        let metrics = ModelPerformanceMetrics::new(0.75, Some(0.15), None, None, None, None, None);
        assert!((metrics.c_index - 0.75).abs() < 1e-6);
        assert!((metrics.brier_score - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_model_card_markdown() {
        let metrics = ModelPerformanceMetrics::new(0.75, Some(0.15), None, None, None, None, None);
        let card = create_model_card(
            "TestModel".to_string(),
            "Cox".to_string(),
            "1.0".to_string(),
            "Test description".to_string(),
            "Research".to_string(),
            "Simulated data".to_string(),
            100,
            30,
            vec!["age".to_string(), "sex".to_string()],
            metrics,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let md = card.to_markdown();
        assert!(md.contains("TestModel"));
        assert!(md.contains("0.75"));
    }
}
