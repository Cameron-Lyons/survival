use super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TwoSidedConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub is_two_sided: Vec<bool>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_two_sided: usize,
    #[pyo3(get)]
    pub n_one_sided: usize,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TwoSidedCalibrationResult {
    #[pyo3(get)]
    pub lower_quantile: f64,
    #[pyo3(get)]
    pub upper_quantile: f64,
    #[pyo3(get)]
    pub censoring_score_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_uncensored: usize,
    #[pyo3(get)]
    pub n_censored: usize,
}

fn compute_censoring_scores(status: &[i32], predicted: &[f64], time: &[f64]) -> Vec<f64> {
    let n = time.len();
    let mut scores = Vec::with_capacity(n);

    let mean_time: f64 = time.iter().sum::<f64>() / n as f64;
    let mean_pred: f64 = predicted.iter().sum::<f64>() / n as f64;

    for i in 0..n {
        let time_ratio = time[i] / mean_time;
        let pred_ratio = predicted[i] / mean_pred;
        let score = if status[i] == 0 {
            (time_ratio - pred_ratio).abs() + 0.5
        } else {
            (time_ratio - pred_ratio).abs()
        };
        scores.push(score);
    }

    scores
}

fn classify_uncensored_like(
    censoring_score: f64,
    threshold: f64,
    alpha_half: f64,
    n_censored: usize,
) -> bool {
    let p_value = (1.0 + (n_censored as f64 * (1.0 - censoring_score / threshold).max(0.0)))
        / (1.0 + n_censored as f64);
    p_value >= alpha_half
}

pub(super) fn compute_two_sided_scores(
    time: &[f64],
    status: &[i32],
    predicted: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut lower_scores = Vec::new();
    let mut upper_scores = Vec::new();

    for i in 0..time.len() {
        if status[i] == 1 {
            lower_scores.push(predicted[i] - time[i]);
            upper_scores.push(time[i] - predicted[i]);
        }
    }

    (lower_scores, upper_scores)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, coverage_level=None))]
pub fn two_sided_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<TwoSidedCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let alpha = 1.0 - coverage;
    let alpha_half = alpha / 2.0;

    let n_uncensored = status.iter().filter(|&&s| s == 1).count();
    let n_censored = n - n_uncensored;

    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let (lower_scores, upper_scores) = compute_two_sided_scores(&time, &status, &predicted);

    let uniform_weights: Vec<f64> = vec![1.0; n_uncensored];

    let lower_quantile_level =
        (1.0 - alpha_half) * (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let lower_quantile_level = lower_quantile_level.min(1.0);
    let lower_quantile = weighted_quantile(&lower_scores, &uniform_weights, lower_quantile_level);

    let upper_quantile_level =
        (1.0 - alpha_half) * (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let upper_quantile_level = upper_quantile_level.min(1.0);
    let upper_quantile = weighted_quantile(&upper_scores, &uniform_weights, upper_quantile_level);

    let censoring_scores = compute_censoring_scores(&status, &predicted, &time);
    let censored_scores: Vec<f64> = censoring_scores
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 0)
        .map(|(score, _)| *score)
        .collect();

    let censoring_score_threshold = if censored_scores.is_empty() {
        f64::INFINITY
    } else {
        let mut sorted_scores = censored_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((1.0 - alpha_half) * sorted_scores.len() as f64) as usize;
        sorted_scores[idx.min(sorted_scores.len() - 1)]
    };

    Ok(TwoSidedCalibrationResult {
        lower_quantile,
        upper_quantile,
        censoring_score_threshold,
        coverage_level: coverage,
        n_uncensored,
        n_censored,
    })
}

#[pyfunction]
#[pyo3(signature = (calibration, predicted_new, censoring_scores_new=None))]
pub fn two_sided_conformal_predict(
    calibration: &TwoSidedCalibrationResult,
    predicted_new: Vec<f64>,
    censoring_scores_new: Option<Vec<f64>>,
) -> PyResult<TwoSidedConformalResult> {
    let n_new = predicted_new.len();
    if n_new == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }

    let alpha = 1.0 - calibration.coverage_level;
    let alpha_half = alpha / 2.0;

    let censor_scores = censoring_scores_new.unwrap_or_else(|| vec![0.0; n_new]);

    if censor_scores.len() != n_new {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "censoring_scores_new must have the same length as predicted_new",
        ));
    }

    let mut lower_bound = Vec::with_capacity(n_new);
    let mut upper_bound = Vec::with_capacity(n_new);
    let mut is_two_sided = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let is_uncensored_like = classify_uncensored_like(
            censor_scores[i],
            calibration.censoring_score_threshold,
            alpha_half,
            calibration.n_censored,
        );

        let lb = (predicted_new[i] - calibration.lower_quantile).max(0.0);
        lower_bound.push(lb);

        if is_uncensored_like {
            let ub = predicted_new[i] + calibration.upper_quantile;
            upper_bound.push(ub);
            is_two_sided.push(true);
        } else {
            upper_bound.push(f64::INFINITY);
            is_two_sided.push(false);
        }
    }

    let n_two_sided = is_two_sided.iter().filter(|&&x| x).count();
    let n_one_sided = n_new - n_two_sided;

    Ok(TwoSidedConformalResult {
        lower_bound,
        upper_bound,
        predicted_time: predicted_new,
        is_two_sided,
        coverage_level: calibration.coverage_level,
        n_two_sided,
        n_one_sided,
    })
}

#[pyfunction]
#[pyo3(signature = (time_calib, status_calib, predicted_calib, predicted_new, coverage_level=None))]
pub fn two_sided_conformal_survival(
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    predicted_calib: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<TwoSidedConformalResult> {
    let calibration = two_sided_conformal_calibrate(
        time_calib.clone(),
        status_calib.clone(),
        predicted_calib.clone(),
        coverage_level,
    )?;

    let mean_pred: f64 = predicted_calib.iter().sum::<f64>() / predicted_calib.len() as f64;
    let censoring_scores_new: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p / mean_pred - 1.0).abs())
        .collect();

    two_sided_conformal_predict(&calibration, predicted_new, Some(censoring_scores_new))
}
