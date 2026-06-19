use crate::constants::{CONCORDANCE_COUNT_SIZE, PARALLEL_THRESHOLD_LARGE, TIME_EPSILON};
use crate::internal::statistical::{
    ConcordanceSummary, ConcordanceTimeWeight, concordance_index_with_horizon,
    concordance_summary_with_horizon, concordance_summary_with_horizon_and_weights,
    concordance_summary_with_horizon_weights_and_time_weight, counting_process_concordance_index,
    counting_process_concordance_summary, counting_process_concordance_summary_with_weights,
    counting_process_concordance_summary_with_weights_and_time_weight,
};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

fn validate_right_concordance_inputs(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if time.len() != status.len() || time.len() != risk_scores.len() {
        return Err(PyValueError::new_err(
            "time, status, and risk_scores must have the same length",
        ));
    }
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_no_nan(risk_scores, "risk_scores")?;
    validate_finite(risk_scores, "risk_scores")?;
    validate_binary_i32(status, "status")?;
    if let Some(values) = weights {
        if values.len() != time.len() {
            return Err(PyValueError::new_err(
                "weights must have the same length as time",
            ));
        }
        validate_no_nan(values, "weights")?;
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
    }
    Ok(())
}

fn validate_counting_concordance_inputs(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if start.len() != stop.len() || start.len() != status.len() || start.len() != risk_scores.len()
    {
        return Err(PyValueError::new_err(
            "start, stop, status, and risk_scores must have the same length",
        ));
    }
    validate_no_nan(start, "start")?;
    validate_finite(start, "start")?;
    validate_non_negative(start, "start")?;
    validate_no_nan(stop, "stop")?;
    validate_finite(stop, "stop")?;
    validate_non_negative(stop, "stop")?;
    validate_no_nan(risk_scores, "risk_scores")?;
    validate_finite(risk_scores, "risk_scores")?;
    validate_binary_i32(status, "status")?;
    if let Some(values) = weights {
        if values.len() != start.len() {
            return Err(PyValueError::new_err(
                "weights must have the same length as start",
            ));
        }
        validate_no_nan(values, "weights")?;
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
    }

    for (idx, (&entry, &exit)) in start.iter().zip(stop.iter()).enumerate() {
        if entry >= exit - TIME_EPSILON {
            return Err(PyValueError::new_err(format!(
                "start must be less than stop for observation {}",
                idx
            )));
        }
    }
    Ok(())
}

fn parse_concordance_time_weight(timewt: &str) -> PyResult<ConcordanceTimeWeight> {
    match timewt {
        "n" => Ok(ConcordanceTimeWeight::N),
        "S" => Ok(ConcordanceTimeWeight::S),
        "S/G" => Ok(ConcordanceTimeWeight::SOverG),
        "n/G2" => Ok(ConcordanceTimeWeight::NOverG2),
        "I" => Ok(ConcordanceTimeWeight::I),
        _ => Err(PyValueError::new_err(
            "timewt must be one of 'n', 'S', 'S/G', 'n/G2', 'I'",
        )),
    }
}

fn parse_counting_concordance_time_weight(timewt: &str) -> PyResult<ConcordanceTimeWeight> {
    match parse_concordance_time_weight(timewt)? {
        ConcordanceTimeWeight::SOverG | ConcordanceTimeWeight::NOverG2 => {
            Err(PyValueError::new_err(
                "S/G and n/G2 timewt options are not supported for counting-process data",
            ))
        }
        value => Ok(value),
    }
}

/// Compute Harrell's concordance index for survival predictions.
#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn concordance_index(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<f64> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    Ok(
        if time_weight == ConcordanceTimeWeight::N && weights.is_none() {
            concordance_index_with_horizon(&risk_scores, &time, &status, None)
        } else {
            concordance_summary_with_horizon_weights_and_time_weight(
                &risk_scores,
                &time,
                &status,
                weights.as_deref(),
                None,
                time_weight,
            )
            .c_index()
        },
    )
}

fn build_concordance_summary_dict(summary: ConcordanceSummary) -> PyResult<Py<PyDict>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("concordance", summary.c_index())?;
        dict.set_item("concordant", summary.concordant)?;
        dict.set_item("comparable", summary.comparable)?;
        Ok(dict.into())
    })
}

/// Compute Harrell's concordance index and pair counts for survival predictions.
#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn concordance_summary(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<Py<PyDict>> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    build_concordance_summary_dict(if time_weight == ConcordanceTimeWeight::N {
        match weights.as_deref() {
            Some(values) => concordance_summary_with_horizon_and_weights(
                &risk_scores,
                &time,
                &status,
                Some(values),
                None,
            ),
            None => concordance_summary_with_horizon(&risk_scores, &time, &status, None),
        }
    } else {
        concordance_summary_with_horizon_weights_and_time_weight(
            &risk_scores,
            &time,
            &status,
            weights.as_deref(),
            None,
            time_weight,
        )
    })
}

/// Compute Harrell's concordance index for counting-process survival data.
#[pyfunction]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn counting_concordance_index(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<f64> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    Ok(
        if time_weight == ConcordanceTimeWeight::N && weights.is_none() {
            counting_process_concordance_index(&risk_scores, &start, &stop, &status)
        } else {
            counting_process_concordance_summary_with_weights_and_time_weight(
                &risk_scores,
                &start,
                &stop,
                &status,
                weights.as_deref(),
                time_weight,
            )
            .c_index()
        },
    )
}

/// Compute Harrell's concordance index and pair counts for counting-process data.
#[pyfunction]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn counting_concordance_summary(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<Py<PyDict>> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    build_concordance_summary_dict(if time_weight == ConcordanceTimeWeight::N {
        match weights.as_deref() {
            Some(values) => counting_process_concordance_summary_with_weights(
                &risk_scores,
                &start,
                &stop,
                &status,
                Some(values),
            ),
            None => counting_process_concordance_summary(&risk_scores, &start, &stop, &status),
        }
    } else {
        counting_process_concordance_summary_with_weights_and_time_weight(
            &risk_scores,
            &start,
            &stop,
            &status,
            weights.as_deref(),
            time_weight,
        )
    })
}

/// Compute concordance statistics for survival predictions.
///
/// Parameters
/// ----------
/// y : array-like
///     Survival times.
/// x : array-like
///     Predicted risk scores (integer-coded).
/// wt : array-like
///     Observation weights.
/// timewt : array-like
///     Time-dependent weights.
/// sortstart : array-like, optional
///     Start-time sort indices for left-truncated data.
/// sortstop : array-like
///     Stop-time sort indices.
///
/// Returns
/// -------
/// dict
///     Dictionary with concordance counts: concordant, discordant, tied_risk, tied_time, pairs.
#[pyfunction]
pub fn concordance(
    y: Vec<f64>,
    x: Vec<i32>,
    wt: Vec<f64>,
    timewt: Vec<f64>,
    sortstart: Option<Vec<usize>>,
    sortstop: Vec<usize>,
) -> PyResult<Py<PyDict>> {
    let n = y.len();
    let mut ntree = 0;
    let mut nwt = vec![0.0; n];
    let mut twt = vec![0.0; n];
    let mut count = vec![0.0; CONCORDANCE_COUNT_SIZE];
    for val in &x {
        ntree = ntree.max(*val as usize + 1);
    }
    let mut utime = 0;
    let i2 = 0;
    let mut i = 0;
    while i < n {
        let ii = sortstop[i];
        let current_time = y[ii];
        let should_skip = match sortstart.as_ref() {
            Some(ss) if i2 < n => y[ss[i2]] >= current_time,
            _ => false,
        };
        if should_skip || y[ii] == 0.0 {
            addin(&mut nwt, &mut twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut _dwt = 0.0;
            let mut _dwt2 = 0.0;
            let xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;
            while i + ndeath < n && y[sortstop[i + ndeath]] == current_time {
                let jj = sortstop[i + ndeath];
                if x[jj] == xsave {
                    count[2] += 1.0;
                } else if i > PARALLEL_THRESHOLD_LARGE {
                    let (concordant, discordant): (f64, f64) = (0..i)
                        .into_par_iter()
                        .map(|k| {
                            let kk = sortstop[k];
                            if x[kk] != x[jj] {
                                if (x[kk] < x[jj] && y[kk] > current_time)
                                    || (x[kk] > x[jj] && y[kk] < current_time)
                                {
                                    (1.0, 0.0)
                                } else {
                                    (0.0, 1.0)
                                }
                            } else {
                                (0.0, 0.0)
                            }
                        })
                        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                    count[0] += concordant;
                    count[1] += discordant;
                } else {
                    for &kk in &sortstop[..i] {
                        if x[kk] != x[jj] {
                            if (x[kk] < x[jj] && y[kk] > current_time)
                                || (x[kk] > x[jj] && y[kk] < current_time)
                            {
                                count[0] += 1.0;
                            } else {
                                count[1] += 1.0;
                            }
                        }
                    }
                }
                _dwt += wt[jj];
                _dwt2 += wt[jj] * adjtimewt;
                ndeath += 1;
            }
            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;
            for &jj in &sortstop[i..i + ndeath] {
                addin(&mut nwt, &mut twt, x[jj] as usize, wt[jj]);
            }
            i += ndeath;
        }
    }
    count[3] -= count[4];
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        Ok(dict.into())
    })
}
#[inline]
fn addin(nwt: &mut [f64], twt: &mut [f64], x: usize, weight: f64) {
    nwt[x] += weight;
    let mut node_index = x;
    while node_index != 0 {
        let parent_index = (node_index - 1) / 2;
        twt[parent_index] += weight;
        node_index = parent_index;
    }
    twt[x] += weight;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn validate_right_concordance_rejects_malformed_inputs() {
        initialize_python();

        let status_err = validate_right_concordance_inputs(&[1.0, 2.0], &[1, 2], &[0.4, 0.1], None)
            .expect_err("non-binary status should be rejected");
        assert!(
            status_err
                .to_string()
                .contains("status must contain only 0/1")
        );

        let time_err =
            validate_right_concordance_inputs(&[1.0, f64::INFINITY], &[1, 0], &[0.4, 0.1], None)
                .expect_err("non-finite time should be rejected");
        assert!(time_err.to_string().contains("time contains non-finite"));

        let risk_err =
            validate_right_concordance_inputs(&[1.0, 2.0], &[1, 0], &[0.4, f64::NAN], None)
                .expect_err("NaN risk score should be rejected");
        assert!(risk_err.to_string().contains("risk_scores contains NaN"));

        let weight_err = validate_right_concordance_inputs(
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            Some(&[1.0, -1.0]),
        )
        .expect_err("negative weights should be rejected");
        assert!(weight_err.to_string().contains("weights contains negative"));
    }

    #[test]
    fn validate_counting_concordance_rejects_malformed_inputs() {
        initialize_python();

        let interval_err = validate_counting_concordance_inputs(
            &[0.0, 2.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            None,
        )
        .expect_err("zero-width counting interval should be rejected");
        assert!(
            interval_err
                .to_string()
                .contains("start must be less than stop")
        );

        let start_err = validate_counting_concordance_inputs(
            &[-0.1, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            None,
        )
        .expect_err("negative start time should be rejected");
        assert!(start_err.to_string().contains("start contains negative"));

        let weight_err = validate_counting_concordance_inputs(
            &[0.0, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            Some(&[1.0, f64::NAN]),
        )
        .expect_err("NaN weights should be rejected");
        assert!(weight_err.to_string().contains("weights contains NaN"));
    }
}
