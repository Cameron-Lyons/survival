
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TimeDependentROCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub sensitivity: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub specificity: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub auc: Vec<f64>,
    #[pyo3(get)]
    pub optimal_threshold: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times, n_thresholds=100))]
pub fn time_dependent_roc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
    n_thresholds: usize,
) -> PyResult<TimeDependentROCResult> {
    let n = risk_scores.len();

    let min_risk = risk_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_risk = risk_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let thresholds: Vec<f64> = (0..n_thresholds)
        .map(|i| min_risk + (max_risk - min_risk) * i as f64 / (n_thresholds - 1) as f64)
        .collect();

    let mut all_sensitivity = Vec::new();
    let mut all_specificity = Vec::new();
    let mut all_auc = Vec::new();
    let mut optimal_thresholds = Vec::new();

    for &t in &eval_times {
        let cases: Vec<usize> = (0..n)
            .filter(|&i| event_time[i] <= t && event_status[i] == 1)
            .collect();

        let controls: Vec<usize> = (0..n).filter(|&i| event_time[i] > t).collect();

        let n_cases = cases.len();
        let n_controls = controls.len();

        if n_cases == 0 || n_controls == 0 {
            all_sensitivity.push(vec![0.0; n_thresholds]);
            all_specificity.push(vec![1.0; n_thresholds]);
            all_auc.push(0.5);
            optimal_thresholds.push(thresholds[n_thresholds / 2]);
            continue;
        }

        let mut sens = Vec::new();
        let mut spec = Vec::new();
        let mut max_youden = f64::NEG_INFINITY;
        let mut opt_thresh = thresholds[0];

        for &thresh in &thresholds {
            let tp = cases.iter().filter(|&&i| risk_scores[i] >= thresh).count();
            let tn = controls
                .iter()
                .filter(|&&i| risk_scores[i] < thresh)
                .count();

            let sensitivity_val = tp as f64 / n_cases as f64;
            let specificity_val = tn as f64 / n_controls as f64;

            sens.push(sensitivity_val);
            spec.push(specificity_val);

            let youden = sensitivity_val + specificity_val - 1.0;
            if youden > max_youden {
                max_youden = youden;
                opt_thresh = thresh;
            }
        }

        let mut auc = 0.0;
        for i in 1..n_thresholds {
            let dx = (1.0 - spec[i - 1]) - (1.0 - spec[i]);
            let dy = (sens[i - 1] + sens[i]) / 2.0;
            auc += dx * dy;
        }

        all_sensitivity.push(sens);
        all_specificity.push(spec);
        all_auc.push(auc.abs());
        optimal_thresholds.push(opt_thresh);
    }

    Ok(TimeDependentROCResult {
        times: eval_times,
        sensitivity: all_sensitivity,
        specificity: all_specificity,
        thresholds,
        auc: all_auc,
        optimal_threshold: optimal_thresholds,
    })
}

