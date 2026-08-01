use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const PARALLEL_WORK_THRESHOLD: usize = 8_192;

#[derive(Debug, PartialEq)]
struct BrierComponents {
    brier: Vec<f64>,
    rsquared: Vec<f64>,
    effective_n: Vec<f64>,
}

struct BrierInputs<'a> {
    observed_time: &'a [f64],
    status: &'a [i32],
    case_weights: &'a [f64],
    evaluation_times: &'a [f64],
    null_predictions: &'a [f64],
    model_predictions: &'a [Vec<f64>],
    censor_times: &'a [f64],
    censor_survival: &'a [f64],
}

fn step_value_at(times: &[f64], values: &[f64], requested_time: f64) -> f64 {
    let index = times.partition_point(|&time| time <= requested_time);
    if index == 0 { 1.0 } else { values[index - 1] }
}

fn brier_at_time(
    inputs: &BrierInputs<'_>,
    time_index: usize,
    total_case_weight: f64,
) -> (f64, f64, f64) {
    let evaluation_time = inputs.evaluation_times[time_index];
    let null_prediction = inputs.null_predictions[time_index];
    let mut denominator = 0.0;
    let mut weight_square_sum = 0.0;
    let mut null_numerator = 0.0;
    let mut model_numerator = 0.0;

    for row_index in 0..inputs.observed_time.len() {
        let observed_time = inputs.observed_time[row_index];
        let status = f64::from(inputs.status[row_index]);
        let censor_survival = step_value_at(
            inputs.censor_times,
            inputs.censor_survival,
            observed_time.min(evaluation_time),
        );
        let weight = if observed_time < evaluation_time && status == 0.0 {
            0.0
        } else if censor_survival > 0.0 {
            inputs.case_weights[row_index] / total_case_weight / censor_survival
        } else {
            f64::INFINITY
        };
        let model_prediction = inputs.model_predictions[time_index][row_index];
        let (null_loss, model_loss) = if observed_time > evaluation_time {
            (
                null_prediction * null_prediction,
                model_prediction * model_prediction,
            )
        } else {
            (
                (status - null_prediction).powi(2),
                (status - model_prediction).powi(2),
            )
        };

        denominator += weight;
        weight_square_sum += weight * weight;
        null_numerator += weight * null_loss;
        model_numerator += weight * model_loss;
    }

    let null_brier = null_numerator / denominator;
    let model_brier = model_numerator / denominator;
    let rsquared = if null_brier == 0.0 {
        if model_brier == 0.0 {
            f64::NAN
        } else if model_brier < 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        1.0 - model_brier / null_brier
    };
    (model_brier, rsquared, 1.0 / weight_square_sum)
}

fn validate_inputs(inputs: &BrierInputs<'_>) -> Result<f64, String> {
    let n = inputs.observed_time.len();
    if n == 0 {
        return Err("observed_time must not be empty".to_string());
    }
    if inputs.status.len() != n || inputs.case_weights.len() != n {
        return Err(
            "observed_time, status, and case_weights must have the same length".to_string(),
        );
    }
    if inputs.null_predictions.len() != inputs.evaluation_times.len() {
        return Err("null_predictions and evaluation_times must have the same length".to_string());
    }
    if inputs.model_predictions.len() != inputs.evaluation_times.len() {
        return Err("model_predictions must have one row per evaluation time".to_string());
    }
    if let Some((row_index, _)) = inputs
        .model_predictions
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != n)
    {
        return Err(format!(
            "model_predictions row {row_index} must have length {n}"
        ));
    }
    if inputs.censor_times.len() != inputs.censor_survival.len() {
        return Err("censor_times and censor_survival must have the same length".to_string());
    }
    if inputs.censor_times.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("censor_times must be sorted ascending".to_string());
    }
    if inputs.status.iter().any(|&value| value != 0 && value != 1) {
        return Err("status must contain only 0/1 values".to_string());
    }
    for (name, values) in [
        ("observed_time", inputs.observed_time),
        ("evaluation_times", inputs.evaluation_times),
        ("null_predictions", inputs.null_predictions),
        ("censor_times", inputs.censor_times),
        ("censor_survival", inputs.censor_survival),
        ("case_weights", inputs.case_weights),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} must contain only finite values"));
        }
    }
    if inputs
        .model_predictions
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("model_predictions must contain only finite values".to_string());
    }
    if inputs.case_weights.iter().any(|&value| value < 0.0) {
        return Err("case_weights must be non-negative".to_string());
    }
    for (name, values) in [
        ("null_predictions", inputs.null_predictions),
        ("censor_survival", inputs.censor_survival),
    ] {
        if values.iter().any(|&value| !(0.0..=1.0).contains(&value)) {
            return Err(format!("{name} must contain values between 0 and 1"));
        }
    }
    if inputs
        .model_predictions
        .iter()
        .flatten()
        .any(|&value| !(0.0..=1.0).contains(&value))
    {
        return Err("model_predictions must contain values between 0 and 1".to_string());
    }
    let total_case_weight: f64 = inputs.case_weights.iter().sum();
    if !total_case_weight.is_finite() || total_case_weight <= 0.0 {
        return Err("case_weights must have positive sum".to_string());
    }
    Ok(total_case_weight)
}

fn compute_brier_components(inputs: &BrierInputs<'_>) -> Result<BrierComponents, String> {
    let total_case_weight = validate_inputs(inputs)?;
    let n_times = inputs.evaluation_times.len();
    let work = inputs.observed_time.len().saturating_mul(n_times);
    let compute = |time_index| brier_at_time(inputs, time_index, total_case_weight);
    let rows: Vec<(f64, f64, f64)> = if work >= PARALLEL_WORK_THRESHOLD {
        (0..n_times).into_par_iter().map(compute).collect()
    } else {
        (0..n_times).map(compute).collect()
    };

    let mut brier = Vec::with_capacity(n_times);
    let mut rsquared = Vec::with_capacity(n_times);
    let mut effective_n = Vec::with_capacity(n_times);
    for (score, r_squared, n_effective) in rows {
        brier.push(score);
        rsquared.push(r_squared);
        effective_n.push(n_effective);
    }
    Ok(BrierComponents {
        brier,
        rsquared,
        effective_n,
    })
}

#[pyfunction]
#[pyo3(signature = (
    observed_time,
    status,
    case_weights,
    evaluation_times,
    null_predictions,
    model_predictions,
    censor_times,
    censor_survival,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn perform_brier_calculation(
    py: Python<'_>,
    observed_time: Vec<f64>,
    status: Vec<i32>,
    case_weights: Vec<f64>,
    evaluation_times: Vec<f64>,
    null_predictions: Vec<f64>,
    model_predictions: Vec<Vec<f64>>,
    censor_times: Vec<f64>,
    censor_survival: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let result = py
        .detach(move || {
            compute_brier_components(&BrierInputs {
                observed_time: &observed_time,
                status: &status,
                case_weights: &case_weights,
                evaluation_times: &evaluation_times,
                null_predictions: &null_predictions,
                model_predictions: &model_predictions,
                censor_times: &censor_times,
                censor_survival: &censor_survival,
            })
        })
        .map_err(PyValueError::new_err)?;
    let output = PyDict::new(py);
    output.set_item("brier", result.brier)?;
    output.set_item("rsquared", result.rsquared)?;
    output.set_item("eff_n", result.effective_n)?;
    Ok(output.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_components_match_direct_ipcw_calculation() {
        let inputs = BrierInputs {
            observed_time: &[1.0, 2.5, 3.0, 4.5],
            status: &[1, 0, 1, 0],
            case_weights: &[1.0, 2.0, 1.0, 1.0],
            evaluation_times: &[2.0, 4.0],
            null_predictions: &[0.2, 0.6],
            model_predictions: &[vec![0.1, 0.2, 0.3, 0.4], vec![0.4, 0.5, 0.6, 0.7]],
            censor_times: &[0.0, 2.5, 4.5],
            censor_survival: &[1.0, 0.6, 0.0],
        };

        let result = compute_brier_components(&inputs).expect("valid scores should compute");

        assert_eq!(result.brier.len(), 2);
        assert!((result.brier[0] - 0.228).abs() < 1e-12);
        assert!((result.rsquared[0] + 0.425).abs() < 1e-12);
        assert!((result.effective_n[0] - 1.0 / 0.28).abs() < 1e-12);
        assert!((result.brier[1] - 0.333_076_923_076_923_1).abs() < 1e-12);
        assert!((result.rsquared[1] + 0.405_844_155_844_155_7).abs() < 1e-12);
        assert!((result.effective_n[1] - 3.813_559_322_033_898).abs() < 1e-12);
    }

    #[test]
    fn batched_components_reject_malformed_prediction_shapes() {
        let inputs = BrierInputs {
            observed_time: &[1.0, 2.0],
            status: &[1, 0],
            case_weights: &[1.0, 1.0],
            evaluation_times: &[1.0, 2.0],
            null_predictions: &[0.2, 0.5],
            model_predictions: &[vec![0.1, 0.2], vec![0.3]],
            censor_times: &[0.0],
            censor_survival: &[1.0],
        };

        assert_eq!(
            compute_brier_components(&inputs).expect_err("shape mismatch should fail"),
            "model_predictions row 1 must have length 2"
        );
    }

    #[test]
    fn batched_components_return_r_values_after_all_rows_are_censored() {
        let inputs = BrierInputs {
            observed_time: &[1.0, 2.0],
            status: &[0, 0],
            case_weights: &[1.0, 1.0],
            evaluation_times: &[3.0],
            null_predictions: &[0.5],
            model_predictions: &[vec![0.4, 0.6]],
            censor_times: &[0.0, 1.0, 2.0],
            censor_survival: &[1.0, 0.5, 0.0],
        };

        let result = compute_brier_components(&inputs).expect("valid scores should compute");

        assert!(result.brier[0].is_nan());
        assert!(result.rsquared[0].is_nan());
        assert_eq!(result.effective_n[0], f64::INFINITY);
    }
}
