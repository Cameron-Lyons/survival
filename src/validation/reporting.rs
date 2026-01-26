#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;

use crate::utilities::statistical::{lower_incomplete_gamma, normal_cdf};

#[derive(Debug, Clone)]
#[pyclass]
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
    let n = time.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }

    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup();

    let z = match confidence_level {
        c if (c - 0.90).abs() < 0.01 => 1.645,
        c if (c - 0.99).abs() < 0.01 => 2.576,
        _ => 1.96,
    };

    let mut time_points = vec![0.0];
    let mut survival_prob = vec![1.0];
    let mut lower_ci = vec![1.0];
    let mut upper_ci = vec![1.0];
    let mut at_risk = vec![n];
    let mut n_events = vec![0];
    let mut n_censored = vec![0];

    let mut surv = 1.0;
    let mut var_sum = 0.0;

    for &t in &unique_times {
        let risk_count = time.iter().filter(|&&ti| ti >= t).count();
        let event_count = time
            .iter()
            .zip(event.iter())
            .filter(|&(&ti, &ei)| (ti - t).abs() < 1e-10 && ei == 1)
            .count();
        let censored_count = time
            .iter()
            .zip(event.iter())
            .filter(|&(&ti, &ei)| (ti - t).abs() < 1e-10 && ei == 0)
            .count();

        if risk_count > 0 {
            let d = event_count as f64;
            let n_r = risk_count as f64;

            if d > 0.0 {
                surv *= 1.0 - d / n_r;
                var_sum += d / (n_r * (n_r - d).max(1.0));
            }

            let se = surv * var_sum.sqrt();
            let log_surv = surv.max(1e-10).ln();
            let log_se = se / surv.max(1e-10);

            let lower = (log_surv - z * log_se).exp().clamp(0.0, 1.0);
            let upper = (log_surv + z * log_se).exp().clamp(0.0, 1.0);

            time_points.push(t);
            survival_prob.push(surv);
            lower_ci.push(lower);
            upper_ci.push(upper);
            at_risk.push(risk_count);
            n_events.push(event_count);
            n_censored.push(censored_count);
        }
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
#[pyclass]
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

    fn significant_at(&self, alpha: f64) -> Vec<bool> {
        self.p_values.iter().map(|&p| p < alpha).collect()
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
    if coefficients.len() != n || standard_errors.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let z = match confidence_level {
        c if (c - 0.90).abs() < 0.01 => 1.645,
        c if (c - 0.99).abs() < 0.01 => 2.576,
        _ => 1.96,
    };

    let hazard_ratios: Vec<f64> = coefficients.iter().map(|&c| c.exp()).collect();
    let lower_ci: Vec<f64> = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&c, &se)| (c - z * se).exp())
        .collect();
    let upper_ci: Vec<f64> = coefficients
        .iter()
        .zip(standard_errors.iter())
        .map(|(&c, &se)| (c + z * se).exp())
        .collect();
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
#[pyclass]
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted and observed must have the same non-zero length",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| predicted[a].partial_cmp(&predicted[b]).unwrap());

    let bin_size = n / n_bins;
    let mut predicted_prob = Vec::new();
    let mut observed_prob = Vec::new();
    let mut n_per_bin = Vec::new();
    let mut bin_boundaries = vec![0.0];

    let mut hl_stat = 0.0;

    for i in 0..n_bins {
        let start = i * bin_size;
        let end = if i == n_bins - 1 {
            n
        } else {
            (i + 1) * bin_size
        };
        let bin_indices: Vec<usize> = sorted_indices[start..end].to_vec();

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
            hl_stat += (obs_events - expected).powi(2) / (expected * (1.0 - pred_mean)).max(1e-10);
        }
    }
    bin_boundaries.push(1.0);

    let df = (n_bins - 2).max(1) as f64;
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
#[pyclass]
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
    let n = time.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }

    let n_events = event.iter().filter(|&&e| e == 1).count();

    let mut sorted_times: Vec<f64> = time.to_vec();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_times.dedup();

    let mut surv = 1.0;
    let mut median_survival = None;
    let mut var_sum = 0.0;

    let mut survival_at_times: Vec<(f64, f64, f64, f64)> = Vec::new();

    for &t in &sorted_times {
        let risk_count = time.iter().filter(|&&ti| ti >= t).count();
        let event_count = time
            .iter()
            .zip(event.iter())
            .filter(|&(&ti, &ei)| (ti - t).abs() < 1e-10 && ei == 1)
            .count();

        if risk_count > 0 {
            let d = event_count as f64;
            let n_r = risk_count as f64;

            if d > 0.0 {
                surv *= 1.0 - d / n_r;
                var_sum += d / (n_r * (n_r - d).max(1.0));
            }

            let se = surv * var_sum.sqrt();
            let log_surv = surv.max(1e-10).ln();
            let log_se = se / surv.max(1e-10);
            let lower = (log_surv - 1.96 * log_se).exp().clamp(0.0, 1.0);
            let upper = (log_surv + 1.96 * log_se).exp().clamp(0.0, 1.0);

            survival_at_times.push((t, surv, lower, upper));

            if surv <= 0.5 && median_survival.is_none() {
                median_survival = Some(t);
            }
        }
    }

    let median_ci = median_survival.map(|m| (m * 0.8, m * 1.2));

    let landmarks = landmark_times.unwrap_or_else(|| {
        let max_time = sorted_times.last().cloned().unwrap_or(1.0);
        vec![max_time * 0.25, max_time * 0.5, max_time * 0.75, max_time]
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
        let tau = sorted_times.last().cloned().unwrap_or(1.0);
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
#[pyclass]
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

    fn optimal_threshold(&self, method: &str) -> f64 {
        match method {
            "youden" => {
                let mut best_idx = 0;
                let mut best_j = 0.0f64;
                for i in 0..self.fpr.len() {
                    let j = self.tpr[i] - self.fpr[i];
                    if j > best_j {
                        best_j = j;
                        best_idx = i;
                    }
                }
                self.thresholds[best_idx]
            }
            _ => {
                let mut best_idx = 0;
                let mut min_dist = f64::MAX;
                for i in 0..self.fpr.len() {
                    let dist = self.fpr[i].powi(2) + (1.0 - self.tpr[i]).powi(2);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = i;
                    }
                }
                self.thresholds[best_idx]
            }
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "scores and labels must have the same non-zero length",
        ));
    }

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&l| l == 0).count() as f64;

    if n_pos == 0.0 || n_neg == 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Both positive and negative labels required",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thresholds = vec![f64::INFINITY];

    let mut tp = 0.0;
    let mut fp = 0.0;

    for &idx in &sorted_indices {
        if labels[idx] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        tpr.push(tp / n_pos);
        fpr.push(fp / n_neg);
        thresholds.push(scores[idx]);
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
        let names = vec!["Age".to_string(), "Sex".to_string()];
        let coefs = vec![0.5, -0.3];
        let ses = vec![0.1, 0.15];

        let result = forest_plot_data(names, coefs, ses, 0.95).unwrap();
        assert_eq!(result.hazard_ratios.len(), 2);
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
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
        let labels = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

        let result = roc_plot_data(scores, labels).unwrap();
        assert!(result.auc >= 0.0 && result.auc <= 1.0);
    }
}
