use pyo3::prelude::*;

use crate::constants::{
    DIVISION_FLOOR, Z_SCORE_95, exp_ci, exp_ci_bounds, same_time, z_score_for_confidence,
};
use crate::internal::statistical::{lower_incomplete_gamma, normal_cdf};
use crate::internal::validation::{
    validate_binary_i32, validate_confidence_level, validate_finite, validate_non_negative,
    validate_positive_finite_slice, validate_probability_slice,
};

const MEDIAN_CI_LOWER_FACTOR: f64 = 0.8;
const MEDIAN_CI_UPPER_FACTOR: f64 = 1.2;
const DEFAULT_LANDMARK_FRACTIONS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

#[derive(Debug, Clone, Copy)]
struct SurvivalTimeSummary {
    time: f64,
    at_risk: usize,
    n_events: usize,
    n_censored: usize,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn validate_survival_plot_inputs(time: &[f64], event: &[i32]) -> PyResult<()> {
    if time.is_empty() || event.len() != time.len() {
        return Err(value_error(
            "time and event must have the same non-zero length",
        ));
    }
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(event, "event")?;
    Ok(())
}

fn validate_forest_plot_inputs(variable_names: &[String]) -> PyResult<()> {
    if variable_names.is_empty() {
        return Err(value_error("variable_names must be non-empty"));
    }
    for (idx, name) in variable_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(value_error(format!(
                "variable_names must not contain empty names (invalid value at index {idx})"
            )));
        }
    }
    Ok(())
}

fn validate_forest_plot_outputs(
    hazard_ratios: &[f64],
    lower_ci: &[f64],
    upper_ci: &[f64],
) -> PyResult<()> {
    if hazard_ratios
        .iter()
        .chain(lower_ci.iter())
        .chain(upper_ci.iter())
        .any(|value| !value.is_finite())
    {
        return Err(value_error(
            "forest plot hazard ratios and confidence intervals must be finite",
        ));
    }
    Ok(())
}

fn validate_report_title(title: &str) -> PyResult<()> {
    if title.trim().is_empty() {
        return Err(value_error("title must be non-empty"));
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> PyResult<()> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(value_error("alpha must be finite and between 0 and 1"));
    }
    Ok(())
}

fn normalize_roc_threshold_method(method: &str) -> PyResult<&'static str> {
    let normalized = method.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "youden" => Ok("youden"),
        "closest" | "closest_topleft" | "closest_top_left" => Ok("closest_topleft"),
        _ => Err(value_error("method must be 'youden' or 'closest_topleft'")),
    }
}

fn survival_time_summaries(time: &[f64], event: &[i32]) -> Vec<SurvivalTimeSummary> {
    let mut observations: Vec<(f64, i32)> =
        time.iter().copied().zip(event.iter().copied()).collect();
    observations.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut summaries = Vec::new();
    let mut at_risk = observations.len();
    let mut index = 0;
    while index < observations.len() {
        let current_time = observations[index].0;
        let risk_at_time = at_risk;
        let mut n_events = 0;
        let mut n_censored = 0;

        while index < observations.len() && same_time(observations[index].0, current_time) {
            if observations[index].1 == 1 {
                n_events += 1;
            } else {
                n_censored += 1;
            }
            index += 1;
        }

        summaries.push(SurvivalTimeSummary {
            time: current_time,
            at_risk: risk_at_time,
            n_events,
            n_censored,
        });
        at_risk -= n_events + n_censored;
    }

    summaries
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct KaplanMeierPlotData {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival_prob: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
    #[pyo3(get)]
    pub at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
    #[pyo3(get)]
    pub n_censored: Vec<usize>,
    #[pyo3(get)]
    pub group_name: Option<String>,
}

#[pymethods]
impl KaplanMeierPlotData {
    fn __repr__(&self) -> String {
        format!(
            "KaplanMeierPlotData(n_points={}, group={:?})",
            self.time_points.len(),
            self.group_name
        )
    }

    fn to_step_data(&self) -> (Vec<f64>, Vec<f64>) {
        let mut step_x = Vec::new();
        let mut step_y = Vec::new();

        for i in 0..self.time_points.len() {
            if i > 0 {
                step_x.push(self.time_points[i]);
                step_y.push(self.survival_prob[i - 1]);
            }
            step_x.push(self.time_points[i]);
            step_y.push(self.survival_prob[i]);
        }

        (step_x, step_y)
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    confidence_level=0.95,
    group_name=None
))]
pub fn km_plot_data(
    time: Vec<f64>,
    event: Vec<i32>,
    confidence_level: f64,
    group_name: Option<String>,
) -> PyResult<KaplanMeierPlotData> {
    validate_survival_plot_inputs(&time, &event)?;
    validate_confidence_level(confidence_level)?;
    let n = time.len();
    let summaries = survival_time_summaries(&time, &event);

    let z = z_score_for_confidence(confidence_level);

    let mut time_points = vec![0.0];
    let mut survival_prob = vec![1.0];
    let mut lower_ci = vec![1.0];
    let mut upper_ci = vec![1.0];
    let mut at_risk = vec![n];
    let mut n_events = vec![0];
    let mut n_censored = vec![0];

    let mut surv = 1.0;
    let mut var_sum = 0.0;

    for summary in summaries {
        let d = summary.n_events as f64;
        let n_r = summary.at_risk as f64;

        if d > 0.0 {
            surv *= 1.0 - d / n_r;
            var_sum += d / (n_r * (n_r - d).max(1.0));
        }

        let se = surv * var_sum.sqrt();
        let bounded_surv = surv.max(DIVISION_FLOOR);
        let log_surv = bounded_surv.ln();
        let log_se = se / bounded_surv;

        let (lower, upper) = exp_ci(log_surv, log_se, z);
        let lower = lower.clamp(0.0, 1.0);
        let upper = upper.clamp(0.0, 1.0);

        time_points.push(summary.time);
        survival_prob.push(surv);
        lower_ci.push(lower);
        upper_ci.push(upper);
        at_risk.push(summary.at_risk);
        n_events.push(summary.n_events);
        n_censored.push(summary.n_censored);
    }

    Ok(KaplanMeierPlotData {
        time_points,
        survival_prob,
        lower_ci,
        upper_ci,
        at_risk,
        n_events,
        n_censored,
        group_name,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ForestPlotData {
    #[pyo3(get)]
    pub variable_names: Vec<String>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub weights: Option<Vec<f64>>,
}

#[pymethods]
impl ForestPlotData {
    fn __repr__(&self) -> String {
        format!("ForestPlotData(n_variables={})", self.variable_names.len())
    }

    fn significant_at(&self, alpha: f64) -> PyResult<Vec<bool>> {
        validate_alpha(alpha)?;
        Ok(self.p_values.iter().map(|&p| p < alpha).collect())
    }
}

#[pyfunction]
#[pyo3(signature = (
    variable_names,
    coefficients,
    standard_errors,
    confidence_level=0.95
))]
pub fn forest_plot_data(
    variable_names: Vec<String>,
    coefficients: Vec<f64>,
    standard_errors: Vec<f64>,
    confidence_level: f64,
) -> PyResult<ForestPlotData> {
    let n = variable_names.len();
    validate_forest_plot_inputs(&variable_names)?;
    if coefficients.len() != n || standard_errors.len() != n {
        return Err(value_error("All input vectors must have the same length"));
    }
    validate_finite(&coefficients, "coefficients")?;
    validate_positive_finite_slice(&standard_errors, "standard_errors")?;
    validate_confidence_level(confidence_level)?;

    let z = z_score_for_confidence(confidence_level);

    let hazard_ratios: Vec<f64> = coefficients.iter().map(|&c| c.exp()).collect();
    let (lower_ci, upper_ci) = exp_ci_bounds(&coefficients, &standard_errors, z);
    validate_forest_plot_outputs(&hazard_ratios, &lower_ci, &upper_ci)?;
    let p_values: Vec<f64> = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&c, &se)| {
            let z_stat = (c / se).abs();
            2.0 * (1.0 - normal_cdf(z_stat))
        })
        .collect();

    Ok(ForestPlotData {
        variable_names,
        hazard_ratios,
        lower_ci,
        upper_ci,
        p_values,
        weights: None,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CalibrationCurveData {
    #[pyo3(get)]
    pub predicted_prob: Vec<f64>,
    #[pyo3(get)]
    pub observed_prob: Vec<f64>,
    #[pyo3(get)]
    pub n_per_bin: Vec<usize>,
    #[pyo3(get)]
    pub bin_boundaries: Vec<f64>,
    #[pyo3(get)]
    pub hosmer_lemeshow_stat: f64,
    #[pyo3(get)]
    pub hosmer_lemeshow_p: f64,
}

#[pymethods]
impl CalibrationCurveData {
    fn __repr__(&self) -> String {
        format!(
            "CalibrationCurveData(n_bins={}, HL_stat={:.2})",
            self.predicted_prob.len(),
            self.hosmer_lemeshow_stat
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    predicted,
    observed,
    n_bins=10
))]
pub fn calibration_plot_data(
    predicted: Vec<f64>,
    observed: Vec<i32>,
    n_bins: usize,
) -> PyResult<CalibrationCurveData> {
    let n = predicted.len();
    if n == 0 || observed.len() != n {
        return Err(value_error(
            "predicted and observed must have the same non-zero length",
        ));
    }
    if n_bins == 0 || n_bins > n {
        return Err(value_error(
            "n_bins must be between 1 and the number of observations",
        ));
    }
    validate_probability_slice(&predicted, "predicted")?;
    validate_binary_i32(&observed, "observed")?;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| predicted[a].total_cmp(&predicted[b]));

    let mut predicted_prob = Vec::new();
    let mut observed_prob = Vec::new();
    let mut n_per_bin = Vec::new();
    let mut bin_boundaries = vec![0.0];

    let mut hl_stat = 0.0;

    for i in 0..n_bins {
        let start = i * n / n_bins;
        let end = (i + 1) * n / n_bins;
        let bin_indices = &sorted_indices[start..end];

        let bin_n = bin_indices.len();
        let pred_mean: f64 = bin_indices.iter().map(|&j| predicted[j]).sum::<f64>() / bin_n as f64;
        let obs_mean: f64 =
            bin_indices.iter().map(|&j| observed[j] as f64).sum::<f64>() / bin_n as f64;

        predicted_prob.push(pred_mean);
        observed_prob.push(obs_mean);
        n_per_bin.push(bin_n);

        if i < n_bins - 1 {
            bin_boundaries.push(predicted[sorted_indices[end]]);
        }

        let expected = bin_n as f64 * pred_mean;
        let obs_events = bin_indices.iter().map(|&j| observed[j] as f64).sum::<f64>();
        if expected > 0.0 && expected < bin_n as f64 {
            hl_stat += (obs_events - expected).powi(2)
                / (expected * (1.0 - pred_mean)).max(DIVISION_FLOOR);
        }
    }
    bin_boundaries.push(1.0);

    let df = n_bins.saturating_sub(2).max(1) as f64;
    let hl_p = 1.0 - lower_incomplete_gamma(df / 2.0, hl_stat / 2.0);

    Ok(CalibrationCurveData {
        predicted_prob,
        observed_prob,
        n_per_bin,
        bin_boundaries,
        hosmer_lemeshow_stat: hl_stat,
        hosmer_lemeshow_p: hl_p,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalReport {
    #[pyo3(get)]
    pub title: String,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub median_survival: Option<f64>,
    #[pyo3(get)]
    pub median_ci: Option<(f64, f64)>,
    #[pyo3(get)]
    pub survival_rates: Vec<(f64, f64, f64, f64)>,
    #[pyo3(get)]
    pub rmst: Option<f64>,
    #[pyo3(get)]
    pub hazard_ratio: Option<f64>,
    #[pyo3(get)]
    pub hazard_ratio_ci: Option<(f64, f64)>,
    #[pyo3(get)]
    pub logrank_p: Option<f64>,
}

#[pymethods]
impl SurvivalReport {
    fn __repr__(&self) -> String {
        format!(
            "SurvivalReport(n={}, events={}, median={:?})",
            self.n_subjects, self.n_events, self.median_survival
        )
    }

    fn to_markdown(&self) -> String {
        let mut md = format!("# {}\n\n", self.title);
        md.push_str(&format!("**Sample Size:** {}\n\n", self.n_subjects));
        md.push_str(&format!("**Number of Events:** {}\n\n", self.n_events));

        if let Some(median) = self.median_survival {
            md.push_str(&format!("**Median Survival:** {:.2}", median));
            if let Some((lower, upper)) = self.median_ci {
                md.push_str(&format!(" (95% CI: {:.2} - {:.2})", lower, upper));
            }
            md.push_str("\n\n");
        }

        if !self.survival_rates.is_empty() {
            md.push_str("## Survival Rates\n\n");
            md.push_str("| Time | Survival | 95% CI Lower | 95% CI Upper |\n");
            md.push_str("|------|----------|--------------|---------------|\n");
            for &(t, surv, lower, upper) in &self.survival_rates {
                md.push_str(&format!(
                    "| {:.1} | {:.3} | {:.3} | {:.3} |\n",
                    t, surv, lower, upper
                ));
            }
            md.push('\n');
        }

        if let Some(hr) = self.hazard_ratio {
            md.push_str(&format!("**Hazard Ratio:** {:.3}", hr));
            if let Some((lower, upper)) = self.hazard_ratio_ci {
                md.push_str(&format!(" (95% CI: {:.3} - {:.3})", lower, upper));
            }
            md.push_str("\n\n");
        }

        if let Some(p) = self.logrank_p {
            md.push_str(&format!("**Log-rank p-value:** {:.4}\n\n", p));
        }

        md
    }

    fn to_latex(&self) -> String {
        let mut latex = format!("\\section{{{}}}\n\n", self.title);
        latex.push_str(&format!(
            "Sample size: {} subjects with {} events.\n\n",
            self.n_subjects, self.n_events
        ));

        if let Some(median) = self.median_survival {
            latex.push_str(&format!("Median survival: {:.2}", median));
            if let Some((lower, upper)) = self.median_ci {
                latex.push_str(&format!(" (95\\% CI: {:.2}--{:.2})", lower, upper));
            }
            latex.push_str(".\n\n");
        }

        if !self.survival_rates.is_empty() {
            latex.push_str("\\begin{table}[h]\n");
            latex.push_str("\\centering\n");
            latex.push_str("\\begin{tabular}{cccc}\n");
            latex.push_str("\\hline\n");
            latex.push_str("Time & Survival & 95\\% CI Lower & 95\\% CI Upper \\\\\n");
            latex.push_str("\\hline\n");
            for &(t, surv, lower, upper) in &self.survival_rates {
                latex.push_str(&format!(
                    "{:.1} & {:.3} & {:.3} & {:.3} \\\\\n",
                    t, surv, lower, upper
                ));
            }
            latex.push_str("\\hline\n");
            latex.push_str("\\end{tabular}\n");
            latex.push_str("\\caption{Survival rates at landmark times}\n");
            latex.push_str("\\end{table}\n\n");
        }

        latex
    }
}

#[pyfunction]
#[pyo3(signature = (
    title,
    time,
    event,
    landmark_times=None
))]
pub fn generate_survival_report(
    title: String,
    time: Vec<f64>,
    event: Vec<i32>,
    landmark_times: Option<Vec<f64>>,
) -> PyResult<SurvivalReport> {
    validate_report_title(&title)?;
    validate_survival_plot_inputs(&time, &event)?;
    if let Some(ref times) = landmark_times {
        validate_finite(times, "landmark_times")?;
        validate_non_negative(times, "landmark_times")?;
    }
    let n = time.len();

    let n_events = event.iter().filter(|&&e| e == 1).count();
    let summaries = survival_time_summaries(&time, &event);

    let mut surv = 1.0;
    let mut median_survival = None;
    let mut var_sum = 0.0;

    let mut survival_at_times: Vec<(f64, f64, f64, f64)> = Vec::new();

    for summary in &summaries {
        let d = summary.n_events as f64;
        let n_r = summary.at_risk as f64;

        if d > 0.0 {
            surv *= 1.0 - d / n_r;
            var_sum += d / (n_r * (n_r - d).max(1.0));
        }

        let se = surv * var_sum.sqrt();
        let bounded_surv = surv.max(DIVISION_FLOOR);
        let log_surv = bounded_surv.ln();
        let log_se = se / bounded_surv;
        let (lower, upper) = exp_ci(log_surv, log_se, Z_SCORE_95);
        let lower = lower.clamp(0.0, 1.0);
        let upper = upper.clamp(0.0, 1.0);

        survival_at_times.push((summary.time, surv, lower, upper));

        if surv <= 0.5 && median_survival.is_none() {
            median_survival = Some(summary.time);
        }
    }

    let median_ci =
        median_survival.map(|m| (m * MEDIAN_CI_LOWER_FACTOR, m * MEDIAN_CI_UPPER_FACTOR));

    let landmarks = landmark_times.unwrap_or_else(|| {
        let max_time = summaries.last().map(|summary| summary.time).unwrap_or(1.0);
        DEFAULT_LANDMARK_FRACTIONS
            .iter()
            .map(|fraction| max_time * *fraction)
            .collect()
    });

    let survival_rates: Vec<(f64, f64, f64, f64)> = landmarks
        .iter()
        .map(|&t| {
            let nearest = survival_at_times
                .iter()
                .rev()
                .find(|(st, _, _, _)| *st <= t)
                .cloned()
                .unwrap_or((0.0, 1.0, 1.0, 1.0));
            (t, nearest.1, nearest.2, nearest.3)
        })
        .collect();

    let rmst = {
        let mut rmst_val = 0.0;
        let tau = summaries.last().map(|summary| summary.time).unwrap_or(1.0);
        let mut prev_t = 0.0;
        let mut prev_s = 1.0;

        for &(t, s, _, _) in &survival_at_times {
            if t <= tau {
                rmst_val += prev_s * (t - prev_t);
                prev_t = t;
                prev_s = s;
            }
        }
        rmst_val += prev_s * (tau - prev_t);
        Some(rmst_val)
    };

    Ok(SurvivalReport {
        title,
        n_subjects: n,
        n_events,
        median_survival,
        median_ci,
        survival_rates,
        rmst,
        hazard_ratio: None,
        hazard_ratio_ci: None,
        logrank_p: None,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ROCPlotData {
    #[pyo3(get)]
    pub fpr: Vec<f64>,
    #[pyo3(get)]
    pub tpr: Vec<f64>,
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub auc: f64,
}

#[pymethods]
impl ROCPlotData {
    fn __repr__(&self) -> String {
        format!("ROCPlotData(AUC={:.4})", self.auc)
    }

    fn optimal_threshold(&self, method: &str) -> PyResult<f64> {
        let finite_indices = self
            .thresholds
            .iter()
            .enumerate()
            .filter_map(|(idx, threshold)| threshold.is_finite().then_some(idx));

        match normalize_roc_threshold_method(method)? {
            "youden" => {
                let mut best: Option<(usize, f64)> = None;
                for i in finite_indices {
                    let j = self.tpr[i] - self.fpr[i];
                    if best.is_none_or(|(_, best_j)| j > best_j) {
                        best = Some((i, j));
                    }
                }
                let (best_idx, _) =
                    best.ok_or_else(|| value_error("ROC thresholds must contain a finite value"))?;
                Ok(self.thresholds[best_idx])
            }
            "closest_topleft" => {
                let mut best: Option<(usize, f64)> = None;
                for i in finite_indices {
                    let dist = self.fpr[i].powi(2) + (1.0 - self.tpr[i]).powi(2);
                    if best.is_none_or(|(_, best_dist)| dist < best_dist) {
                        best = Some((i, dist));
                    }
                }
                let (best_idx, _) =
                    best.ok_or_else(|| value_error("ROC thresholds must contain a finite value"))?;
                Ok(self.thresholds[best_idx])
            }
            _ => unreachable!("ROC threshold method was validated"),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    scores,
    labels
))]
pub fn roc_plot_data(scores: Vec<f64>, labels: Vec<i32>) -> PyResult<ROCPlotData> {
    let n = scores.len();
    if n == 0 || labels.len() != n {
        return Err(value_error(
            "scores and labels must have the same non-zero length",
        ));
    }
    validate_finite(&scores, "scores")?;
    validate_binary_i32(&labels, "labels")?;

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&l| l == 0).count() as f64;

    if n_pos == 0.0 || n_neg == 0.0 {
        return Err(value_error("Both positive and negative labels required"));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));

    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thresholds = vec![f64::INFINITY];

    let mut tp = 0.0;
    let mut fp = 0.0;

    let mut index = 0;
    while index < sorted_indices.len() {
        let threshold = scores[sorted_indices[index]];
        let mut threshold_pos = 0.0;
        let mut threshold_neg = 0.0;

        while index < sorted_indices.len()
            && scores[sorted_indices[index]].total_cmp(&threshold) == std::cmp::Ordering::Equal
        {
            if labels[sorted_indices[index]] == 1 {
                threshold_pos += 1.0;
            } else {
                threshold_neg += 1.0;
            }
            index += 1;
        }

        tp += threshold_pos;
        fp += threshold_neg;
        tpr.push(tp / n_pos);
        fpr.push(fp / n_neg);
        thresholds.push(threshold);
    }

    let auc = fpr
        .windows(2)
        .zip(tpr.windows(2))
        .map(|(f, t)| (f[1] - f[0]) * (t[0] + t[1]) / 2.0)
        .sum();

    Ok(ROCPlotData {
        fpr,
        tpr,
        thresholds,
        auc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_km_plot_data() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];

        let result = km_plot_data(time, event, 0.95, Some("Test".to_string())).unwrap();
        assert!(!result.time_points.is_empty());
        assert!(
            result
                .survival_prob
                .iter()
                .all(|&s| (0.0..=1.0).contains(&s))
        );
    }

    #[test]
    fn test_forest_plot_data() {
        Python::initialize();

        let names = vec!["Age".to_string(), "Sex".to_string()];
        let coefs = vec![0.5, -0.3];
        let ses = vec![0.1, 0.15];

        let result = forest_plot_data(names, coefs, ses, 0.95).unwrap();
        assert_eq!(result.hazard_ratios.len(), 2);
        assert_eq!(result.significant_at(0.05).unwrap(), vec![true, true]);
        assert!(
            result
                .significant_at(f64::NAN)
                .expect_err("non-finite alpha should fail")
                .to_string()
                .contains("alpha must be finite and between 0 and 1")
        );
        assert!(
            result
                .significant_at(0.0)
                .expect_err("zero alpha should fail")
                .to_string()
                .contains("alpha must be finite and between 0 and 1")
        );
        assert!(
            result
                .significant_at(1.0)
                .expect_err("unit alpha should fail")
                .to_string()
                .contains("alpha must be finite and between 0 and 1")
        );
    }

    #[test]
    fn test_calibration_plot_data() {
        let predicted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let observed = vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

        let result = calibration_plot_data(predicted, observed, 5).unwrap();
        assert_eq!(result.predicted_prob.len(), 5);
    }

    #[test]
    fn test_survival_report() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];

        let result =
            generate_survival_report("Test Report".to_string(), time, event, None).unwrap();
        assert_eq!(result.n_subjects, 5);
        assert_eq!(result.n_events, 3);
    }

    #[test]
    fn test_roc_plot_data() {
        Python::initialize();

        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let labels = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

        let result = roc_plot_data(scores, labels).unwrap();
        assert!(result.auc >= 0.0 && result.auc <= 1.0);
        assert!(result.optimal_threshold("youden").unwrap().is_finite());
        assert!(
            result
                .optimal_threshold("closest_topleft")
                .unwrap()
                .is_finite()
        );
        assert!(
            result
                .optimal_threshold("closest-top-left")
                .unwrap()
                .is_finite()
        );
        assert!(
            result
                .optimal_threshold("typo")
                .expect_err("unknown ROC threshold method should fail")
                .to_string()
                .contains("method must be 'youden' or 'closest_topleft'")
        );

        let reversed = roc_plot_data(vec![0.1, 0.9], vec![1, 0]).unwrap();
        assert_eq!(reversed.optimal_threshold("youden").unwrap(), 0.1);
        assert_eq!(reversed.optimal_threshold("closest_topleft").unwrap(), 0.1);
    }

    #[test]
    fn reporting_apis_validate_public_inputs() {
        Python::initialize();

        assert!(
            km_plot_data(vec![f64::NAN], vec![1], 0.95, None)
                .expect_err("non-finite time should fail")
                .to_string()
                .contains("time contains non-finite")
        );
        assert!(
            km_plot_data(vec![1.0], vec![2], 0.95, None)
                .expect_err("non-binary event should fail")
                .to_string()
                .contains("event values must be 0 or 1")
        );
        assert!(
            km_plot_data(vec![1.0], vec![1], 1.0, None)
                .expect_err("invalid confidence should fail")
                .to_string()
                .contains("confidence_level")
        );
        assert!(
            km_plot_data(vec![1.0], vec![1], 0.0, None)
                .expect_err("zero confidence should fail")
                .to_string()
                .contains("confidence_level")
        );
        assert!(
            forest_plot_data(vec!["x".to_string()], vec![1.0], vec![0.0], 0.95)
                .expect_err("non-positive standard error should fail")
                .to_string()
                .contains("standard_errors must contain positive values")
        );
        assert!(
            forest_plot_data(vec!["x".to_string()], vec![1000.0], vec![0.1], 0.95)
                .expect_err("overflowing hazard ratio should fail")
                .to_string()
                .contains("hazard ratios and confidence intervals must be finite")
        );
        assert!(
            forest_plot_data(vec![], vec![], vec![], 0.95)
                .expect_err("empty forest plot variables should fail")
                .to_string()
                .contains("variable_names must be non-empty")
        );
        assert!(
            forest_plot_data(vec![" ".to_string()], vec![1.0], vec![0.1], 0.95)
                .expect_err("blank forest plot variable should fail")
                .to_string()
                .contains("variable_names must not contain empty names")
        );
        assert!(
            calibration_plot_data(vec![0.2], vec![1], 0)
                .expect_err("zero bins should fail")
                .to_string()
                .contains("n_bins")
        );
        assert!(
            calibration_plot_data(vec![1.2], vec![1], 1)
                .expect_err("probabilities outside range should fail")
                .to_string()
                .contains("probabilities between 0 and 1")
        );
        assert!(
            generate_survival_report(" ".to_string(), vec![1.0], vec![1], None)
                .expect_err("blank report title should fail")
                .to_string()
                .contains("title must be non-empty")
        );
        assert!(
            generate_survival_report(
                "bad".to_string(),
                vec![1.0],
                vec![1],
                Some(vec![f64::INFINITY]),
            )
            .expect_err("non-finite landmark should fail")
            .to_string()
            .contains("landmark_times contains non-finite")
        );
        assert!(
            roc_plot_data(vec![f64::NAN], vec![1])
                .expect_err("non-finite score should fail")
                .to_string()
                .contains("scores contains non-finite")
        );
        assert!(
            roc_plot_data(vec![0.5], vec![2])
                .expect_err("non-binary label should fail")
                .to_string()
                .contains("labels values must be 0 or 1")
        );
    }

    #[test]
    fn roc_plot_data_groups_tied_scores() {
        let result = roc_plot_data(vec![0.5, 0.5, 0.2, 0.8], vec![1, 0, 0, 1]).unwrap();

        assert_eq!(result.thresholds, vec![f64::INFINITY, 0.8, 0.5, 0.2]);
        assert!((result.auc - 0.875).abs() < 1e-12);
    }
}
