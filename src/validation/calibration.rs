use crate::constants::{PARALLEL_THRESHOLD_LARGE, PARALLEL_THRESHOLD_SMALL, same_time};
use crate::internal::statistical::{chi2_sf, normal_cdf};
use crate::simd_ops::{dot_product_simd, mean_simd, subtract_scalar_simd, sum_of_squares_simd};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn calibration_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_probability_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(calibration_value_error(format!(
                "{name} must contain finite probabilities between 0 and 1 (invalid value at index {idx})"
            )));
        }
    }
    Ok(())
}

fn validate_finite_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(calibration_value_error(format!(
                "{name} must contain only finite values (invalid value at index {idx})"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0.0 {
            return Err(calibration_value_error(format!(
                "{name} must be non-negative (invalid value at index {idx})"
            )));
        }
    }
    Ok(())
}

fn validate_strictly_increasing_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, window) in values.windows(2).enumerate() {
        if window[1] <= window[0] {
            return Err(calibration_value_error(format!(
                "{name} must be strictly increasing (invalid values at indices {idx} and {})",
                idx + 1
            )));
        }
    }
    Ok(())
}

fn validate_nondecreasing_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, window) in values.windows(2).enumerate() {
        if window[1] < window[0] {
            return Err(calibration_value_error(format!(
                "{name} must be sorted in nondecreasing order (invalid values at indices {idx} and {})",
                idx + 1
            )));
        }
    }
    Ok(())
}

fn validate_binary_slice(name: &str, values: &[i32]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(calibration_value_error(format!(
                "{name} must contain only 0/1 values (invalid value at index {idx})"
            )));
        }
    }
    Ok(())
}

fn validate_public_calibration_inputs(
    predicted_risk: &[f64],
    observed_event: &[i32],
) -> PyResult<()> {
    if predicted_risk.is_empty() || observed_event.len() != predicted_risk.len() {
        return Err(calibration_value_error(
            "predicted_risk and observed_event must have the same non-zero length",
        ));
    }
    validate_probability_slice("predicted_risk", predicted_risk)?;
    validate_binary_slice("observed_event", observed_event)
}

fn validate_predict_cox_inputs(
    coef: &[f64],
    x: &[Vec<f64>],
    baseline_hazard: &[f64],
    baseline_times: &[f64],
    pred_times: &[f64],
) -> PyResult<()> {
    if coef.is_empty() {
        return Err(calibration_value_error("coef must be non-empty"));
    }
    validate_finite_slice("coef", coef)?;

    if x.is_empty() {
        return Err(calibration_value_error(
            "x must contain at least one observation",
        ));
    }
    for (row_idx, row) in x.iter().enumerate() {
        if row.len() != coef.len() {
            return Err(calibration_value_error(format!(
                "x row {row_idx} has {} columns, expected {}",
                row.len(),
                coef.len()
            )));
        }
        validate_finite_slice("x", row)?;
    }

    if baseline_hazard.is_empty() || baseline_hazard.len() != baseline_times.len() {
        return Err(calibration_value_error(
            "baseline_hazard and baseline_times must have the same non-zero length",
        ));
    }
    validate_finite_slice("baseline_hazard", baseline_hazard)?;
    validate_nonnegative_slice("baseline_hazard", baseline_hazard)?;
    validate_finite_slice("baseline_times", baseline_times)?;
    validate_strictly_increasing_slice("baseline_times", baseline_times)?;

    if pred_times.is_empty() {
        return Err(calibration_value_error("pred_times must be non-empty"));
    }
    validate_finite_slice("pred_times", pred_times)
}

fn validate_td_auc_inputs(
    time: &[f64],
    status: &[i32],
    risk_score: &[f64],
    eval_times: &[f64],
) -> PyResult<()> {
    if time.is_empty() || status.len() != time.len() || risk_score.len() != time.len() {
        return Err(calibration_value_error(
            "time, status, and risk_score must have the same non-zero length",
        ));
    }
    validate_finite_slice("time", time)?;
    validate_nonnegative_slice("time", time)?;
    validate_binary_slice("status", status)?;
    validate_finite_slice("risk_score", risk_score)?;

    if eval_times.is_empty() {
        return Err(calibration_value_error("eval_times must be non-empty"));
    }
    validate_finite_slice("eval_times", eval_times)?;
    validate_nonnegative_slice("eval_times", eval_times)?;
    validate_nondecreasing_slice("eval_times", eval_times)
}

fn validate_risk_stratification_inputs(risk_scores: &[f64], events: &[i32]) -> PyResult<()> {
    if risk_scores.is_empty() || events.len() != risk_scores.len() {
        return Err(calibration_value_error(
            "risk_scores and events must have the same non-zero length",
        ));
    }
    validate_finite_slice("risk_scores", risk_scores)?;
    validate_binary_slice("events", events)
}

fn validate_advanced_calibration_inputs(
    predicted_risk: &[f64],
    observed_outcome: &[i32],
    n_spline_knots: Option<usize>,
) -> PyResult<usize> {
    if predicted_risk.is_empty() || observed_outcome.len() != predicted_risk.len() {
        return Err(calibration_value_error(
            "predicted_risk and observed_outcome must have the same non-zero length",
        ));
    }
    validate_probability_slice("predicted_risk", predicted_risk)?;
    validate_binary_slice("observed_outcome", observed_outcome)?;

    let n_knots = n_spline_knots.unwrap_or(5);
    if n_knots == 0 {
        return Err(calibration_value_error("n_spline_knots must be positive"));
    }
    Ok(n_knots)
}

fn validate_time_dependent_calibration_inputs(
    time: &[f64],
    event: &[i32],
    predicted_survival: &[Vec<f64>],
    eval_times: &[f64],
) -> PyResult<()> {
    if time.is_empty() || event.len() != time.len() || predicted_survival.len() != time.len() {
        return Err(calibration_value_error(
            "time, event, and predicted_survival must have the same non-zero length",
        ));
    }
    validate_finite_slice("time", time)?;
    validate_nonnegative_slice("time", time)?;
    validate_binary_slice("event", event)?;

    if eval_times.is_empty() {
        return Err(calibration_value_error("eval_times must be non-empty"));
    }
    validate_finite_slice("eval_times", eval_times)?;
    validate_nonnegative_slice("eval_times", eval_times)?;
    validate_nondecreasing_slice("eval_times", eval_times)?;

    for (row_idx, row) in predicted_survival.iter().enumerate() {
        if row.len() != eval_times.len() {
            return Err(calibration_value_error(format!(
                "predicted_survival row {row_idx} has {} elements, expected {}",
                row.len(),
                eval_times.len()
            )));
        }
        let row_name = format!("predicted_survival row {row_idx}");
        validate_probability_slice(&row_name, row)?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CalibrationResult {
    #[pyo3(get)]
    pub risk_groups: Vec<f64>,
    #[pyo3(get)]
    pub predicted: Vec<f64>,
    #[pyo3(get)]
    pub observed: Vec<f64>,
    #[pyo3(get)]
    pub n_per_group: Vec<usize>,
    #[pyo3(get)]
    pub hosmer_lemeshow_stat: f64,
    #[pyo3(get)]
    pub hosmer_lemeshow_pvalue: f64,
    #[pyo3(get)]
    pub calibration_slope: f64,
    #[pyo3(get)]
    pub calibration_intercept: f64,
}
#[pymethods]
impl CalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        risk_groups: Vec<f64>,
        predicted: Vec<f64>,
        observed: Vec<f64>,
        n_per_group: Vec<usize>,
        hosmer_lemeshow_stat: f64,
        hosmer_lemeshow_pvalue: f64,
        calibration_slope: f64,
        calibration_intercept: f64,
    ) -> Self {
        Self {
            risk_groups,
            predicted,
            observed,
            n_per_group,
            hosmer_lemeshow_stat,
            hosmer_lemeshow_pvalue,
            calibration_slope,
            calibration_intercept,
        }
    }
}
pub(crate) fn calibration_curve(
    predicted_risk: &[f64],
    observed_event: &[i32],
    n_groups: usize,
) -> CalibrationResult {
    let n = predicted_risk.len();
    if n == 0 || n_groups == 0 {
        return CalibrationResult {
            risk_groups: vec![],
            predicted: vec![],
            observed: vec![],
            n_per_group: vec![],
            hosmer_lemeshow_stat: 0.0,
            hosmer_lemeshow_pvalue: 1.0,
            calibration_slope: 1.0,
            calibration_intercept: 0.0,
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| predicted_risk[a].total_cmp(&predicted_risk[b]));
    let group_size = n / n_groups;
    let remainder = n % n_groups;
    let mut risk_groups = Vec::with_capacity(n_groups);
    let mut predicted = Vec::with_capacity(n_groups);
    let mut observed = Vec::with_capacity(n_groups);
    let mut n_per_group = Vec::with_capacity(n_groups);
    let mut start = 0;
    for g in 0..n_groups {
        let extra = if g < remainder { 1 } else { 0 };
        let end = start + group_size + extra;
        if end <= start {
            continue;
        }
        let group_indices: Vec<usize> = indices[start..end].to_vec();
        let n_in_group = group_indices.len();
        let sum_pred: f64 = group_indices.iter().map(|&i| predicted_risk[i]).sum();
        let sum_obs: f64 = group_indices
            .iter()
            .map(|&i| observed_event[i] as f64)
            .sum();
        let mean_pred = sum_pred / n_in_group as f64;
        let mean_obs = sum_obs / n_in_group as f64;
        let mid_idx = group_indices[n_in_group / 2];
        risk_groups.push(predicted_risk[mid_idx]);
        predicted.push(mean_pred);
        observed.push(mean_obs);
        n_per_group.push(n_in_group);
        start = end;
    }
    let mut hl_stat = 0.0;
    for g in 0..risk_groups.len() {
        let n_g = n_per_group[g] as f64;
        let o_g = observed[g] * n_g;
        let e_g = predicted[g] * n_g;
        if e_g > 0.0 && e_g < n_g {
            hl_stat += (o_g - e_g).powi(2) / (e_g * (1.0 - predicted[g]));
        }
    }
    let df = if risk_groups.len() > 2 {
        risk_groups.len() - 2
    } else {
        1
    };
    let hl_pvalue = chi2_sf(hl_stat, df);
    let (slope, intercept) = calibration_regression(&predicted, &observed);
    CalibrationResult {
        risk_groups,
        predicted,
        observed,
        n_per_group,
        hosmer_lemeshow_stat: hl_stat,
        hosmer_lemeshow_pvalue: hl_pvalue,
        calibration_slope: slope,
        calibration_intercept: intercept,
    }
}
#[inline]
fn calibration_regression(predicted: &[f64], observed: &[f64]) -> (f64, f64) {
    let n = predicted.len();
    if n < 2 {
        return (1.0, 0.0);
    }
    let mean_x = mean_simd(predicted);
    let mean_y = mean_simd(observed);

    let centered_x = subtract_scalar_simd(predicted, mean_x);
    let centered_y = subtract_scalar_simd(observed, mean_y);

    let ss_xy = dot_product_simd(&centered_x, &centered_y);
    let ss_xx = sum_of_squares_simd(&centered_x);

    let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 1.0 };
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}
#[pyfunction]
#[pyo3(signature = (predicted_risk, observed_event, n_groups=None))]
pub fn calibration(
    predicted_risk: Vec<f64>,
    observed_event: Vec<i32>,
    n_groups: Option<usize>,
) -> PyResult<CalibrationResult> {
    let n_groups = n_groups.unwrap_or(10);
    validate_public_calibration_inputs(&predicted_risk, &observed_event)?;
    if n_groups == 0 {
        return Err(calibration_value_error("n_groups must be positive"));
    }
    let n_groups = n_groups.min(predicted_risk.len());
    Ok(calibration_curve(
        &predicted_risk,
        &observed_event,
        n_groups,
    ))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PredictionResult {
    #[pyo3(get)]
    pub linear_predictor: Vec<f64>,
    #[pyo3(get)]
    pub risk_score: Vec<f64>,
    #[pyo3(get)]
    pub survival_prob: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub times: Vec<f64>,
}
#[pymethods]
impl PredictionResult {
    #[new]
    fn new(
        linear_predictor: Vec<f64>,
        risk_score: Vec<f64>,
        survival_prob: Vec<Vec<f64>>,
        times: Vec<f64>,
    ) -> Self {
        Self {
            linear_predictor,
            risk_score,
            survival_prob,
            times,
        }
    }
}
pub(crate) fn predict_survival(
    coef: &[f64],
    x: &[Vec<f64>],
    baseline_hazard: &[f64],
    baseline_times: &[f64],
    pred_times: &[f64],
) -> PredictionResult {
    let n = x.len();

    let cumhaz: Vec<f64> = baseline_hazard
        .iter()
        .scan(0.0, |acc, &h| {
            *acc += h;
            Some(*acc)
        })
        .collect();

    let results: Vec<(f64, f64, Vec<f64>)> = if n > PARALLEL_THRESHOLD_SMALL {
        x.par_iter()
            .map(|xi| {
                let lp: f64 = coef.iter().zip(xi).map(|(&c, &xij)| c * xij).sum();
                let rs = lp.exp();
                let surv_at_times: Vec<f64> = pred_times
                    .iter()
                    .map(|&t| {
                        let ch = interpolate_cumhaz(baseline_times, &cumhaz, t);
                        (-ch * rs).exp()
                    })
                    .collect();
                (lp, rs, surv_at_times)
            })
            .collect()
    } else {
        x.iter()
            .map(|xi| {
                let lp: f64 = coef.iter().zip(xi).map(|(&c, &xij)| c * xij).sum();
                let rs = lp.exp();
                let surv_at_times: Vec<f64> = pred_times
                    .iter()
                    .map(|&t| {
                        let ch = interpolate_cumhaz(baseline_times, &cumhaz, t);
                        (-ch * rs).exp()
                    })
                    .collect();
                (lp, rs, surv_at_times)
            })
            .collect()
    };

    let mut linear_predictor = Vec::with_capacity(n);
    let mut risk_score = Vec::with_capacity(n);
    let mut survival_prob = Vec::with_capacity(n);

    for (lp, rs, surv) in results {
        linear_predictor.push(lp);
        risk_score.push(rs);
        survival_prob.push(surv);
    }

    PredictionResult {
        linear_predictor,
        risk_score,
        survival_prob,
        times: pred_times.to_vec(),
    }
}
#[inline]
fn interpolate_cumhaz(times: &[f64], cumhaz: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return 0.0;
    }
    let n = times.len();
    if t >= times[n - 1] {
        return cumhaz[n - 1];
    }
    let i = match times.binary_search_by(|probe| probe.total_cmp(&t)) {
        Ok(idx) => return cumhaz[idx],
        Err(idx) => idx,
    };
    if i == 0 {
        return 0.0;
    }
    let frac = (t - times[i - 1]) / (times[i] - times[i - 1]);
    cumhaz[i - 1] + frac * (cumhaz[i] - cumhaz[i - 1])
}
#[pyfunction]
#[pyo3(signature = (coef, x, baseline_hazard, baseline_times, pred_times))]
pub fn predict_cox(
    coef: Vec<f64>,
    x: Vec<Vec<f64>>,
    baseline_hazard: Vec<f64>,
    baseline_times: Vec<f64>,
    pred_times: Vec<f64>,
) -> PyResult<PredictionResult> {
    validate_predict_cox_inputs(&coef, &x, &baseline_hazard, &baseline_times, &pred_times)?;
    Ok(predict_survival(
        &coef,
        &x,
        &baseline_hazard,
        &baseline_times,
        &pred_times,
    ))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RiskStratificationResult {
    #[pyo3(get)]
    pub risk_groups: Vec<usize>,
    #[pyo3(get)]
    pub cutpoints: Vec<f64>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
    #[pyo3(get)]
    pub group_event_rates: Vec<f64>,
    #[pyo3(get)]
    pub group_median_risk: Vec<f64>,
}
#[pymethods]
impl RiskStratificationResult {
    #[new]
    fn new(
        risk_groups: Vec<usize>,
        cutpoints: Vec<f64>,
        group_sizes: Vec<usize>,
        group_event_rates: Vec<f64>,
        group_median_risk: Vec<f64>,
    ) -> Self {
        Self {
            risk_groups,
            cutpoints,
            group_sizes,
            group_event_rates,
            group_median_risk,
        }
    }
}
pub(crate) fn stratify_risk(
    risk_scores: &[f64],
    events: &[i32],
    n_groups: usize,
) -> RiskStratificationResult {
    let n = risk_scores.len();
    if n == 0 || n_groups == 0 {
        return RiskStratificationResult {
            risk_groups: vec![],
            cutpoints: vec![],
            group_sizes: vec![],
            group_event_rates: vec![],
            group_median_risk: vec![],
        };
    }
    let mut sorted_scores: Vec<f64> = risk_scores.to_vec();
    sorted_scores.sort_by(f64::total_cmp);
    let mut cutpoints = Vec::with_capacity(n_groups - 1);
    for g in 1..n_groups {
        let idx = (g * n / n_groups).min(n - 1);
        cutpoints.push(sorted_scores[idx]);
    }
    let mut risk_groups = Vec::with_capacity(n);
    for &score in risk_scores {
        let mut group = 0;
        for (g, &cut) in cutpoints.iter().enumerate() {
            if score >= cut {
                group = g + 1;
            }
        }
        risk_groups.push(group);
    }
    let mut group_sizes = vec![0usize; n_groups];
    let mut group_events = vec![0usize; n_groups];
    let mut group_scores: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
    for i in 0..n {
        let g = risk_groups[i];
        group_sizes[g] += 1;
        if events[i] == 1 {
            group_events[g] += 1;
        }
        group_scores[g].push(risk_scores[i]);
    }
    let group_event_rates: Vec<f64> = (0..n_groups)
        .map(|g| {
            if group_sizes[g] > 0 {
                group_events[g] as f64 / group_sizes[g] as f64
            } else {
                0.0
            }
        })
        .collect();

    let group_median_risk: Vec<f64> = if n > PARALLEL_THRESHOLD_LARGE {
        group_scores
            .par_iter()
            .map(|scores| {
                if scores.is_empty() {
                    0.0
                } else {
                    let mut s = scores.clone();
                    s.sort_by(f64::total_cmp);
                    s[s.len() / 2]
                }
            })
            .collect()
    } else {
        group_scores
            .iter()
            .map(|scores| {
                if scores.is_empty() {
                    0.0
                } else {
                    let mut s = scores.clone();
                    s.sort_by(f64::total_cmp);
                    s[s.len() / 2]
                }
            })
            .collect()
    };
    RiskStratificationResult {
        risk_groups,
        cutpoints,
        group_sizes,
        group_event_rates,
        group_median_risk,
    }
}
#[pyfunction]
#[pyo3(signature = (risk_scores, events, n_groups=None))]
pub fn risk_stratification(
    risk_scores: Vec<f64>,
    events: Vec<i32>,
    n_groups: Option<usize>,
) -> PyResult<RiskStratificationResult> {
    let n_groups = n_groups.unwrap_or(3);
    validate_risk_stratification_inputs(&risk_scores, &events)?;
    if n_groups == 0 {
        return Err(calibration_value_error("n_groups must be positive"));
    }
    let n_groups = n_groups.min(risk_scores.len());
    Ok(stratify_risk(&risk_scores, &events, n_groups))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TdAUCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub auc: Vec<f64>,
    #[pyo3(get)]
    pub integrated_auc: f64,
}
#[pymethods]
impl TdAUCResult {
    #[new]
    fn new(times: Vec<f64>, auc: Vec<f64>, integrated_auc: f64) -> Self {
        Self {
            times,
            auc,
            integrated_auc,
        }
    }
}
pub(crate) fn time_dependent_auc(
    time: &[f64],
    status: &[i32],
    risk_score: &[f64],
    eval_times: &[f64],
) -> TdAUCResult {
    let n = time.len();
    if n == 0 || eval_times.is_empty() {
        return TdAUCResult {
            times: vec![],
            auc: vec![],
            integrated_auc: 0.0,
        };
    }
    let auc_values: Vec<f64> = eval_times
        .par_iter()
        .map(|&t| {
            let (concordant, discordant) = (0..n)
                .filter(|&i| (time[i] <= t || same_time(time[i], t)) && status[i] == 1)
                .flat_map(|i| {
                    (0..n).filter_map(move |j| {
                        if time[j] > t && !same_time(time[j], t) {
                            Some(if risk_score[i] > risk_score[j] {
                                (1.0, 0.0)
                            } else if risk_score[i] < risk_score[j] {
                                (0.0, 1.0)
                            } else {
                                (0.5, 0.5)
                            })
                        } else {
                            None
                        }
                    })
                })
                .fold((0.0, 0.0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
            let total = concordant + discordant;
            if total > 0.0 { concordant / total } else { 0.5 }
        })
        .collect();
    let integrated = if auc_values.len() > 1 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        for i in 1..eval_times.len() {
            let dt = eval_times[i] - eval_times[i - 1];
            sum += dt * (auc_values[i] + auc_values[i - 1]) / 2.0;
            weight_sum += dt;
        }
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            auc_values.iter().sum::<f64>() / auc_values.len() as f64
        }
    } else if !auc_values.is_empty() {
        auc_values[0]
    } else {
        0.5
    };
    TdAUCResult {
        times: eval_times.to_vec(),
        auc: auc_values,
        integrated_auc: integrated,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, risk_score, eval_times))]
pub fn td_auc(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_score: Vec<f64>,
    eval_times: Vec<f64>,
) -> PyResult<TdAUCResult> {
    validate_td_auc_inputs(&time, &status, &risk_score, &eval_times)?;
    Ok(time_dependent_auc(&time, &status, &risk_score, &eval_times))
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct AdvancedCalibrationResult {
    #[pyo3(get)]
    pub ici: f64,
    #[pyo3(get)]
    pub e50: f64,
    #[pyo3(get)]
    pub e90: f64,
    #[pyo3(get)]
    pub emax: f64,
    #[pyo3(get)]
    pub calibration_in_the_large: f64,
    #[pyo3(get)]
    pub calibration_slope: f64,
    #[pyo3(get)]
    pub calibration_intercept: f64,
    #[pyo3(get)]
    pub spiegelhalter_z: f64,
    #[pyo3(get)]
    pub spiegelhalter_p: f64,
}

#[pymethods]
impl AdvancedCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ici: f64,
        e50: f64,
        e90: f64,
        emax: f64,
        calibration_in_the_large: f64,
        calibration_slope: f64,
        calibration_intercept: f64,
        spiegelhalter_z: f64,
        spiegelhalter_p: f64,
    ) -> Self {
        Self {
            ici,
            e50,
            e90,
            emax,
            calibration_in_the_large,
            calibration_slope,
            calibration_intercept,
            spiegelhalter_z,
            spiegelhalter_p,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (predicted_risk, observed_outcome, n_spline_knots=None))]
pub fn advanced_calibration_metrics(
    predicted_risk: Vec<f64>,
    observed_outcome: Vec<i32>,
    n_spline_knots: Option<usize>,
) -> PyResult<AdvancedCalibrationResult> {
    let n = predicted_risk.len();
    let n_knots =
        validate_advanced_calibration_inputs(&predicted_risk, &observed_outcome, n_spline_knots)?;

    let observed: Vec<f64> = observed_outcome.iter().map(|&x| x as f64).collect();

    let mean_pred: f64 = predicted_risk.iter().sum::<f64>() / n as f64;
    let mean_obs: f64 = observed.iter().sum::<f64>() / n as f64;
    let calibration_in_the_large = mean_obs - mean_pred;

    let (slope, intercept) = linear_regression(&predicted_risk, &observed);

    let smoothed = loess_smooth(&predicted_risk, &observed, n_knots);

    let mut errors: Vec<f64> = predicted_risk
        .iter()
        .zip(smoothed.iter())
        .map(|(p, s)| (p - s).abs())
        .collect();

    let ici = errors.iter().sum::<f64>() / n as f64;

    errors.sort_by(f64::total_cmp);

    let e50_idx = (n as f64 * 0.5).floor() as usize;
    let e90_idx = (n as f64 * 0.9).floor() as usize;

    let e50 = errors[e50_idx.min(n - 1)];
    let e90 = errors[e90_idx.min(n - 1)];
    let emax = errors.last().cloned().unwrap_or(0.0);

    let (spiegelhalter_z, spiegelhalter_p) = spiegelhalter_test(&predicted_risk, &observed_outcome);

    Ok(AdvancedCalibrationResult {
        ici,
        e50,
        e90,
        emax,
        calibration_in_the_large,
        calibration_slope: slope,
        calibration_intercept: intercept,
        spiegelhalter_z,
        spiegelhalter_p,
    })
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return (1.0, 0.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    (slope, intercept)
}

fn loess_smooth(x: &[f64], y: &[f64], n_points: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let span = (n_points as f64 / n as f64).clamp(0.1, 1.0);
    let window = (span * n as f64).ceil() as usize;

    let mut smoothed = vec![0.0; n];

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].total_cmp(&x[b]));

    for i in 0..n {
        let xi = x[i];

        let mut distances: Vec<(usize, f64)> = (0..n).map(|j| (j, (x[j] - xi).abs())).collect();
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));

        let max_dist = distances[window.min(n) - 1].1.max(1e-10);

        let mut sum_w = 0.0;
        let mut sum_wy = 0.0;

        for &(j, dist) in distances.iter().take(window) {
            let u = dist / max_dist;
            let w = if u < 1.0 {
                (1.0 - u.powi(3)).powi(3)
            } else {
                0.0
            };
            sum_w += w;
            sum_wy += w * y[j];
        }

        smoothed[i] = if sum_w > 0.0 { sum_wy / sum_w } else { y[i] };
    }

    smoothed
}

fn spiegelhalter_test(predicted: &[f64], observed: &[i32]) -> (f64, f64) {
    let n = predicted.len();
    if n == 0 {
        return (0.0, 1.0);
    }

    let mut sum_term = 0.0;
    let mut var_term = 0.0;

    for i in 0..n {
        let p = predicted[i].clamp(0.001, 0.999);
        let o = observed[i] as f64;

        sum_term += (o - p) * (1.0 - 2.0 * p);
        var_term += (1.0 - 2.0 * p).powi(2) * p * (1.0 - p);
    }

    let z = if var_term > 0.0 {
        sum_term / var_term.sqrt()
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    (z, p_value)
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct TimeDependentCalibrationResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub ici: Vec<f64>,
    #[pyo3(get)]
    pub e50: Vec<f64>,
    #[pyo3(get)]
    pub e90: Vec<f64>,
    #[pyo3(get)]
    pub calibration_slope: Vec<f64>,
    #[pyo3(get)]
    pub calibration_intercept: Vec<f64>,
}

#[pymethods]
impl TimeDependentCalibrationResult {
    #[new]
    pub fn new(
        time_points: Vec<f64>,
        ici: Vec<f64>,
        e50: Vec<f64>,
        e90: Vec<f64>,
        calibration_slope: Vec<f64>,
        calibration_intercept: Vec<f64>,
    ) -> Self {
        Self {
            time_points,
            ici,
            e50,
            e90,
            calibration_slope,
            calibration_intercept,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (time, event, predicted_survival, eval_times))]
pub fn time_dependent_calibration(
    time: Vec<f64>,
    event: Vec<i32>,
    predicted_survival: Vec<Vec<f64>>,
    eval_times: Vec<f64>,
) -> PyResult<TimeDependentCalibrationResult> {
    validate_time_dependent_calibration_inputs(&time, &event, &predicted_survival, &eval_times)?;

    let n = time.len();
    let n_times = eval_times.len();

    let mut ici_vec = vec![0.0; n_times];
    let mut e50_vec = vec![0.0; n_times];
    let mut e90_vec = vec![0.0; n_times];
    let mut slope_vec = vec![1.0; n_times];
    let mut intercept_vec = vec![0.0; n_times];

    for (t_idx, &eval_t) in eval_times.iter().enumerate() {
        let mut pred_at_t = Vec::new();
        let mut obs_at_t = Vec::new();

        for i in 0..n {
            let pred = predicted_survival[i][t_idx];
            let observed = if time[i] > eval_t && !same_time(time[i], eval_t) {
                1
            } else if event[i] == 1 {
                0
            } else {
                1
            };

            pred_at_t.push(1.0 - pred);
            obs_at_t.push(observed);
        }

        if !pred_at_t.is_empty() {
            let result = advanced_calibration_metrics(pred_at_t, obs_at_t, Some(5))?;
            ici_vec[t_idx] = result.ici;
            e50_vec[t_idx] = result.e50;
            e90_vec[t_idx] = result.e90;
            slope_vec[t_idx] = result.calibration_slope;
            intercept_vec[t_idx] = result.calibration_intercept;
        }
    }

    Ok(TimeDependentCalibrationResult {
        time_points: eval_times,
        ici: ici_vec,
        e50: e50_vec,
        e90: e90_vec,
        calibration_slope: slope_vec,
        calibration_intercept: intercept_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_calibration_metrics() {
        let predicted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let observed = vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

        let result = advanced_calibration_metrics(predicted, observed, None).unwrap();

        assert!(result.ici >= 0.0);
        assert!(result.e50 >= 0.0);
        assert!(result.e90 >= 0.0);
        assert!(result.emax >= 0.0);
        assert!(result.spiegelhalter_p >= 0.0 && result.spiegelhalter_p <= 1.0);
    }

    #[test]
    fn test_advanced_calibration_metrics_input_validation() {
        assert!(advanced_calibration_metrics(vec![], vec![], None).is_err());
        assert!(advanced_calibration_metrics(vec![0.2], vec![], None).is_err());
        assert!(advanced_calibration_metrics(vec![f64::NAN], vec![1], None).is_err());
        assert!(advanced_calibration_metrics(vec![1.2], vec![1], None).is_err());
        assert!(advanced_calibration_metrics(vec![0.2], vec![2], None).is_err());
        assert!(advanced_calibration_metrics(vec![0.2], vec![1], Some(0)).is_err());

        let result = advanced_calibration_metrics(vec![0.2, 0.8], vec![0, 1], Some(1)).unwrap();
        assert!(result.ici >= 0.0);
    }

    #[test]
    fn test_time_dependent_calibration() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];
        let predicted_survival = vec![
            vec![0.9, 0.8, 0.7],
            vec![0.95, 0.9, 0.85],
            vec![0.85, 0.75, 0.65],
            vec![0.92, 0.88, 0.82],
            vec![0.88, 0.78, 0.68],
        ];
        let eval_times = vec![1.5, 2.5, 3.5];

        let result =
            time_dependent_calibration(time, event, predicted_survival, eval_times).unwrap();

        assert_eq!(result.time_points.len(), 3);
        assert_eq!(result.ici.len(), 3);
    }

    #[test]
    fn test_time_dependent_calibration_input_validation() {
        let time = vec![1.0, 2.0];
        let event = vec![1, 0];
        let predicted_survival = vec![vec![0.9, 0.8], vec![0.7, 0.6]];
        let eval_times = vec![1.5, 2.5];

        assert!(time_dependent_calibration(vec![], vec![], vec![], vec![1.0]).is_err());
        assert!(time_dependent_calibration(vec![1.0], vec![], vec![vec![0.9]], vec![1.0]).is_err());
        assert!(time_dependent_calibration(vec![1.0], vec![1], vec![], vec![1.0]).is_err());
        assert!(
            time_dependent_calibration(vec![f64::NAN], vec![1], vec![vec![0.9]], vec![1.0])
                .is_err()
        );
        assert!(
            time_dependent_calibration(vec![-1.0], vec![1], vec![vec![0.9]], vec![1.0]).is_err()
        );
        assert!(
            time_dependent_calibration(vec![1.0], vec![2], vec![vec![0.9]], vec![1.0]).is_err()
        );
        assert!(time_dependent_calibration(vec![1.0], vec![1], vec![vec![0.9]], vec![]).is_err());
        assert!(
            time_dependent_calibration(vec![1.0], vec![1], vec![vec![0.9]], vec![f64::NAN])
                .is_err()
        );
        assert!(
            time_dependent_calibration(vec![1.0], vec![1], vec![vec![0.9]], vec![-1.0]).is_err()
        );
        assert!(
            time_dependent_calibration(vec![1.0], vec![1], vec![vec![0.9, 0.8]], vec![2.0, 1.0])
                .is_err()
        );
        assert!(
            time_dependent_calibration(vec![1.0], vec![1], vec![vec![0.9]], vec![1.0, 2.0])
                .is_err()
        );
        assert!(
            time_dependent_calibration(vec![1.0], vec![1], vec![vec![1.2]], vec![1.0]).is_err()
        );

        let result =
            time_dependent_calibration(time, event, predicted_survival, eval_times).unwrap();
        assert_eq!(result.time_points, vec![1.5, 2.5]);
        assert_eq!(result.ici.len(), 2);
    }

    #[test]
    fn test_calibration_result_new() {
        let result = CalibrationResult::new(
            vec![0.1, 0.5],
            vec![0.1, 0.5],
            vec![0.0, 1.0],
            vec![5, 5],
            1.5,
            0.5,
            0.9,
            0.1,
        );
        assert_eq!(result.risk_groups.len(), 2);
        assert!((result.hosmer_lemeshow_stat - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_curve_empty() {
        let result = calibration_curve(&[], &[], 5);
        assert!(result.risk_groups.is_empty());
        assert!((result.hosmer_lemeshow_pvalue - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_curve_basic() {
        let predicted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let observed = vec![0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
        let result = calibration_curve(&predicted, &observed, 2);
        assert_eq!(result.risk_groups.len(), 2);
        assert_eq!(result.n_per_group.len(), 2);
    }

    #[test]
    fn test_calibration_public_input_validation() {
        assert!(calibration(vec![], vec![], Some(1)).is_err());
        assert!(calibration(vec![0.2], vec![], Some(1)).is_err());
        assert!(calibration(vec![f64::NAN], vec![1], Some(1)).is_err());
        assert!(calibration(vec![1.2], vec![1], Some(1)).is_err());
        assert!(calibration(vec![0.2], vec![2], Some(1)).is_err());
        assert!(calibration(vec![0.2], vec![1], Some(0)).is_err());

        let result = calibration(vec![0.2, 0.8], vec![0, 1], None).unwrap();
        assert_eq!(result.n_per_group, vec![1, 1]);
    }

    #[test]
    fn test_predict_cox_public_input_validation() {
        assert!(predict_cox(vec![], vec![vec![1.0]], vec![0.1], vec![1.0], vec![1.0]).is_err());
        assert!(predict_cox(vec![0.5], vec![], vec![0.1], vec![1.0], vec![1.0]).is_err());
        assert!(
            predict_cox(
                vec![0.5, 0.2],
                vec![vec![1.0]],
                vec![0.1],
                vec![1.0],
                vec![1.0]
            )
            .is_err()
        );
        assert!(
            predict_cox(
                vec![0.5],
                vec![vec![f64::NAN]],
                vec![0.1],
                vec![1.0],
                vec![1.0]
            )
            .is_err()
        );
        assert!(predict_cox(vec![0.5], vec![vec![1.0]], vec![], vec![], vec![1.0]).is_err());
        assert!(
            predict_cox(
                vec![0.5],
                vec![vec![1.0]],
                vec![0.1],
                vec![1.0, 2.0],
                vec![1.0]
            )
            .is_err()
        );
        assert!(predict_cox(vec![0.5], vec![vec![1.0]], vec![-0.1], vec![1.0], vec![1.0]).is_err());
        assert!(
            predict_cox(
                vec![0.5],
                vec![vec![1.0]],
                vec![0.1, 0.2],
                vec![1.0, 1.0],
                vec![1.0],
            )
            .is_err()
        );
        assert!(predict_cox(vec![0.5], vec![vec![1.0]], vec![0.1], vec![1.0], vec![]).is_err());

        let result =
            predict_cox(vec![0.5], vec![vec![1.0]], vec![0.1], vec![1.0], vec![1.0]).unwrap();
        assert_eq!(result.linear_predictor, vec![0.5]);
        assert_eq!(result.survival_prob.len(), 1);
    }

    #[test]
    fn test_risk_stratification_public_input_validation() {
        assert!(risk_stratification(vec![], vec![], Some(1)).is_err());
        assert!(risk_stratification(vec![0.2], vec![], Some(1)).is_err());
        assert!(risk_stratification(vec![f64::NAN], vec![1], Some(1)).is_err());
        assert!(risk_stratification(vec![0.2], vec![2], Some(1)).is_err());
        assert!(risk_stratification(vec![0.2], vec![1], Some(0)).is_err());

        let default_result = risk_stratification(vec![0.2, 0.8], vec![0, 1], None).unwrap();
        assert_eq!(default_result.group_sizes, vec![1, 1]);

        let explicit_result = risk_stratification(vec![0.2, 0.8], vec![0, 1], Some(5)).unwrap();
        assert_eq!(explicit_result.group_sizes, vec![1, 1]);
    }

    #[test]
    fn test_td_auc_public_input_validation() {
        assert!(td_auc(vec![], vec![], vec![], vec![1.0]).is_err());
        assert!(td_auc(vec![1.0], vec![], vec![0.5], vec![1.0]).is_err());
        assert!(td_auc(vec![f64::NAN], vec![1], vec![0.5], vec![1.0]).is_err());
        assert!(td_auc(vec![-1.0], vec![1], vec![0.5], vec![1.0]).is_err());
        assert!(td_auc(vec![1.0], vec![2], vec![0.5], vec![1.0]).is_err());
        assert!(td_auc(vec![1.0], vec![1], vec![f64::INFINITY], vec![1.0]).is_err());
        assert!(td_auc(vec![1.0], vec![1], vec![0.5], vec![]).is_err());
        assert!(td_auc(vec![1.0], vec![1], vec![0.5], vec![f64::NAN]).is_err());
        assert!(td_auc(vec![1.0], vec![1], vec![0.5], vec![-1.0]).is_err());
        assert!(td_auc(vec![1.0, 2.0], vec![1, 0], vec![0.8, 0.2], vec![2.0, 1.0]).is_err());

        let result = td_auc(
            vec![1.0, 2.0, 3.0],
            vec![1, 0, 1],
            vec![0.8, 0.2, 0.7],
            vec![1.5, 2.5],
        )
        .unwrap();
        assert_eq!(result.times, vec![1.5, 2.5]);
        assert_eq!(result.auc.len(), 2);
    }

    #[test]
    fn test_time_dependent_auc_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let risk_score = vec![0.8, 0.6, 0.7, 0.3, 0.5];
        let eval_times = vec![2.5, 4.5];

        let result = time_dependent_auc(&time, &status, &risk_score, &eval_times);
        assert_eq!(result.times.len(), 2);
        assert_eq!(result.auc.len(), 2);
        for auc in &result.auc {
            assert!(*auc >= 0.0 && *auc <= 1.0);
        }
    }

    #[test]
    fn test_time_dependent_auc_groups_near_tied_eval_time() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];
        let risk_score = vec![0.7, 0.9, 0.8, 0.1];
        let eval_times = vec![1.0, 2.0];

        let expected = time_dependent_auc(&exact_time, &status, &risk_score, &eval_times);
        let actual = time_dependent_auc(&near_time, &status, &risk_score, &eval_times);

        assert_eq!(actual.times, expected.times);
        assert_eq!(actual.auc.len(), expected.auc.len());
        for (actual_auc, expected_auc) in actual.auc.iter().zip(expected.auc.iter()) {
            assert!((actual_auc - expected_auc).abs() < 1e-12);
        }
        assert!((actual.integrated_auc - expected.integrated_auc).abs() < 1e-12);
    }

    #[test]
    fn test_time_dependent_calibration_groups_near_tied_eval_time() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let event = vec![1, 1, 0, 0];
        let predicted_survival = vec![
            vec![0.7, 0.6],
            vec![0.4, 0.3],
            vec![0.8, 0.7],
            vec![0.9, 0.8],
        ];
        let eval_times = vec![1.0, 2.0];

        let expected = time_dependent_calibration(
            exact_time,
            event.clone(),
            predicted_survival.clone(),
            eval_times.clone(),
        )
        .unwrap();
        let actual =
            time_dependent_calibration(near_time, event, predicted_survival, eval_times).unwrap();

        assert_eq!(actual.time_points, expected.time_points);
        for (actual_ici, expected_ici) in actual.ici.iter().zip(expected.ici.iter()) {
            assert!((actual_ici - expected_ici).abs() < 1e-12);
        }
        for (actual_e50, expected_e50) in actual.e50.iter().zip(expected.e50.iter()) {
            assert!((actual_e50 - expected_e50).abs() < 1e-12);
        }
        for (actual_e90, expected_e90) in actual.e90.iter().zip(expected.e90.iter()) {
            assert!((actual_e90 - expected_e90).abs() < 1e-12);
        }
        for (actual_slope, expected_slope) in actual
            .calibration_slope
            .iter()
            .zip(expected.calibration_slope.iter())
        {
            assert!((actual_slope - expected_slope).abs() < 1e-12);
        }
        for (actual_intercept, expected_intercept) in actual
            .calibration_intercept
            .iter()
            .zip(expected.calibration_intercept.iter())
        {
            assert!((actual_intercept - expected_intercept).abs() < 1e-12);
        }
    }
}
