use std::collections::HashMap;

use pyo3::prelude::*;

pub(crate) fn recurrent_value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

pub(crate) fn validate_solver_controls(max_iter: usize, tol: f64) -> PyResult<()> {
    if max_iter == 0 {
        return Err(recurrent_value_error("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(recurrent_value_error("tol must be a positive finite value"));
    }
    Ok(())
}

pub(crate) fn validate_counting_process_design(
    subject_id: &[usize],
    start_time: &[f64],
    stop_time: &[f64],
    event_status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> PyResult<()> {
    validate_lengths(
        subject_id,
        &[
            ("start_time", start_time.len()),
            ("stop_time", stop_time.len()),
            ("event_status", event_status.len()),
        ],
        x,
        n_obs,
        n_vars,
    )?;
    validate_finite_nonnegative("start_time", start_time)?;
    validate_finite_nonnegative("stop_time", stop_time)?;
    validate_binary_events(event_status)?;

    for (idx, (&start, &stop)) in start_time.iter().zip(stop_time.iter()).enumerate() {
        if stop <= start {
            return Err(recurrent_value_error(format!(
                "stop_time must be greater than start_time at row {idx}"
            )));
        }
    }

    Ok(())
}

pub(crate) fn validate_event_time_design(
    subject_id: &[usize],
    event_time: &[f64],
    event_status: &[i32],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> PyResult<()> {
    validate_lengths(
        subject_id,
        &[
            ("event_time", event_time.len()),
            ("event_status", event_status.len()),
        ],
        x,
        n_obs,
        n_vars,
    )?;
    validate_finite_nonnegative("event_time", event_time)?;
    validate_binary_events(event_status)
}

pub(crate) fn validate_terminal_design(
    term_time: &[f64],
    term_status: &[i32],
    x_terminal: &[f64],
    n_subjects: usize,
    n_term_vars: usize,
) -> PyResult<()> {
    if n_subjects == 0 {
        return Err(recurrent_value_error("n_subjects must be positive"));
    }
    if term_time.len() != n_subjects {
        return Err(recurrent_value_error(
            "term_time length must equal n_subjects",
        ));
    }
    if term_status.len() != n_subjects {
        return Err(recurrent_value_error(
            "term_status length must equal n_subjects",
        ));
    }

    let expected_len = n_subjects
        .checked_mul(n_term_vars)
        .ok_or_else(|| recurrent_value_error("n_subjects * n_term_vars overflows usize"))?;
    if x_terminal.len() != expected_len {
        return Err(recurrent_value_error(
            "x_terminal length must equal n_subjects * n_term_vars",
        ));
    }
    if x_terminal.iter().any(|value| !value.is_finite()) {
        return Err(recurrent_value_error(
            "x_terminal must contain only finite values",
        ));
    }

    validate_finite_nonnegative("term_time", term_time)?;
    validate_binary_events_with_name("term_status", term_status)
}

pub(crate) fn validate_subject_ids_within_count(
    subject_id: &[usize],
    n_subjects: usize,
) -> PyResult<()> {
    for (idx, &subject) in subject_id.iter().enumerate() {
        if subject >= n_subjects {
            return Err(recurrent_value_error(format!(
                "subject_id values must be less than n_subjects; found {subject} at row {idx}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn unique_subject_count(subject_id: &[usize]) -> usize {
    compact_subject_index(subject_id).len()
}

pub(crate) fn compact_subject_index(subject_id: &[usize]) -> HashMap<usize, usize> {
    let mut unique_ids = subject_id.to_vec();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    unique_ids
        .into_iter()
        .enumerate()
        .map(|(idx, subject)| (subject, idx))
        .collect()
}

fn validate_lengths(
    subject_id: &[usize],
    named_lengths: &[(&str, usize)],
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> PyResult<()> {
    if n_obs == 0 {
        return Err(recurrent_value_error("n_obs must be positive"));
    }
    if subject_id.len() != n_obs {
        return Err(recurrent_value_error("subject_id length must equal n_obs"));
    }
    for &(name, len) in named_lengths {
        if len != n_obs {
            return Err(recurrent_value_error(format!(
                "{name} length must equal n_obs"
            )));
        }
    }

    let expected_len = n_obs
        .checked_mul(n_vars)
        .ok_or_else(|| recurrent_value_error("n_obs * n_vars overflows usize"))?;
    if x.len() != expected_len {
        return Err(recurrent_value_error("x length must equal n_obs * n_vars"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(recurrent_value_error("x must contain only finite values"));
    }

    Ok(())
}

fn validate_finite_nonnegative(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(recurrent_value_error(format!(
                "{name} must contain only finite values; found non-finite value at row {idx}"
            )));
        }
        if value < 0.0 {
            return Err(recurrent_value_error(format!(
                "{name} values must be non-negative; found negative value at row {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_binary_events(event_status: &[i32]) -> PyResult<()> {
    validate_binary_events_with_name("event_status", event_status)
}

fn validate_binary_events_with_name(name: &str, values: &[i32]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(recurrent_value_error(format!(
                "{name} must contain only 0/1 values; found {value} at row {idx}"
            )));
        }
    }
    Ok(())
}
