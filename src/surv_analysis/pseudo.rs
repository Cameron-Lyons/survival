use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::{DIVISION_FLOOR, TIME_EPSILON, normal_ci_95, same_time};
use crate::internal::statistical::normal_sf;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_negative,
};
use pyo3::exceptions::PyValueError;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PseudoResult {
    #[pyo3(get)]
    pub pseudo: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub type_: String,
    #[pyo3(get)]
    pub n: usize,
}

#[pyfunction]
#[pyo3(signature = (time, status, eval_times=None, type_=None))]
pub fn pseudo(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Option<Vec<f64>>,
    type_: Option<&str>,
) -> PyResult<PseudoResult> {
    let n = time.len();
    let pseudo_type = validate_pseudo_type(type_)?;
    validate_pseudo_inputs(&time, &status, eval_times.as_deref())?;

    if n == 0 {
        return Ok(PseudoResult {
            pseudo: vec![],
            time: vec![],
            type_: pseudo_type.to_string(),
            n: 0,
        });
    }

    let times = match eval_times {
        Some(t) => t,
        None => default_event_times(&time, &status),
    };

    if times.is_empty() {
        return Ok(PseudoResult {
            pseudo: vec![vec![]; n],
            time: vec![],
            type_: pseudo_type.to_string(),
            n,
        });
    }

    let pseudo_matrix = if matches!(pseudo_type, "survival" | "cumhaz") {
        compute_ij_pseudo(&time, &status, &times, pseudo_type)
    } else {
        compute_rmst_jackknife_pseudo(&time, &status, &times)
    };

    Ok(PseudoResult {
        pseudo: pseudo_matrix,
        time: times,
        type_: pseudo_type.to_string(),
        n,
    })
}

#[derive(Debug, Clone)]
struct EventBlock {
    time: f64,
    risk: f64,
    events: f64,
    survival: f64,
    cumhaz: f64,
}

fn event_blocks(time: &[f64], status: &[i32]) -> Vec<EventBlock> {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));

    let mut blocks = Vec::new();
    let mut n_at_risk = n as f64;
    let mut surv = 1.0;
    let mut cumhaz = 0.0;
    let mut start = 0;

    while start < n {
        let current_time = time[indices[start]];
        let mut end = start + 1;
        while end < n && same_time(time[indices[end]], current_time) {
            end += 1;
        }

        let n_events = indices[start..end]
            .iter()
            .filter(|&&idx| status[idx] == 1)
            .count() as f64;
        let n_removed = (end - start) as f64;
        if n_events > 0.0 && n_at_risk > 0.0 {
            let hazard = n_events / n_at_risk;
            surv *= 1.0 - hazard;
            cumhaz += hazard;
            blocks.push(EventBlock {
                time: current_time,
                risk: n_at_risk,
                events: n_events,
                survival: surv,
                cumhaz,
            });
        }

        n_at_risk -= n_removed;
        start = end;
    }

    blocks
}

fn sorted_time_indices(times: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..times.len()).collect();
    indices.sort_by(|&left, &right| {
        times[left]
            .total_cmp(&times[right])
            .then_with(|| left.cmp(&right))
    });
    indices
}

fn subject_at_risk(subject_time: f64, event_time: f64) -> bool {
    subject_time + TIME_EPSILON >= event_time
}

fn subject_event_at_time(subject_time: f64, subject_status: i32, event_time: f64) -> bool {
    subject_status == 1 && same_time(subject_time, event_time)
}

fn compute_ij_pseudo(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
    type_: &str,
) -> Vec<Vec<f64>> {
    let n = time.len();
    let n_f64 = n as f64;
    let blocks = event_blocks(time, status);
    let eval_order = sorted_time_indices(eval_times);
    let is_survival = type_ == "survival";

    (0..n)
        .into_par_iter()
        .map(|row| {
            let mut values = vec![0.0; eval_times.len()];
            let mut influence = 0.0;
            let mut block_idx = 0;
            for &eval_idx in &eval_order {
                let eval_time = eval_times[eval_idx];
                while block_idx < blocks.len() && blocks[block_idx].time <= eval_time + TIME_EPSILON
                {
                    let block = &blocks[block_idx];
                    if subject_at_risk(time[row], block.time) {
                        if is_survival {
                            if subject_event_at_time(time[row], status[row], block.time) {
                                influence -= 1.0 / block.risk;
                            } else if block.risk > block.events + DIVISION_FLOOR {
                                influence +=
                                    block.events / (block.risk * (block.risk - block.events));
                            }
                        } else if subject_event_at_time(time[row], status[row], block.time) {
                            influence += (block.risk - block.events) / (block.risk * block.risk);
                        } else {
                            influence -= block.events / (block.risk * block.risk);
                        }
                    }
                    block_idx += 1;
                }

                values[eval_idx] = if block_idx == 0 {
                    if is_survival { 1.0 } else { 0.0 }
                } else {
                    let block = &blocks[block_idx - 1];
                    if is_survival {
                        block.survival + n_f64 * block.survival * influence
                    } else {
                        block.cumhaz + n_f64 * influence
                    }
                };
            }
            values
        })
        .collect()
}

fn rmst_leave_one_out_values(
    blocks: &[EventBlock],
    subject_time: f64,
    subject_status: i32,
    eval_times: &[f64],
    eval_order: &[usize],
) -> Vec<f64> {
    let mut values = vec![0.0; eval_times.len()];
    let mut block_idx = 0;
    let mut previous_time = 0.0;
    let mut survival = 1.0;
    let mut area = 0.0;

    for &eval_idx in eval_order {
        let eval_time = eval_times[eval_idx];
        while block_idx < blocks.len() && blocks[block_idx].time <= eval_time {
            let block = &blocks[block_idx];
            area += survival * (block.time - previous_time);
            let risk = block.risk - usize::from(subject_at_risk(subject_time, block.time)) as f64;
            let events = block.events
                - usize::from(subject_event_at_time(
                    subject_time,
                    subject_status,
                    block.time,
                )) as f64;
            if events > 0.0 && risk > 0.0 {
                survival *= 1.0 - events / risk;
            }
            previous_time = block.time;
            block_idx += 1;
        }
        values[eval_idx] = area + survival * (eval_time - previous_time);
    }
    values
}

fn compute_rmst_block_jackknife_pseudo(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
) -> Vec<Vec<f64>> {
    let n_f64 = time.len() as f64;
    let full_rmst = compute_km(time, status, eval_times, "rmst");
    let blocks = event_blocks(time, status);
    let eval_order = sorted_time_indices(eval_times);

    (0..time.len())
        .into_par_iter()
        .map(|row| {
            let leave_one_out =
                rmst_leave_one_out_values(&blocks, time[row], status[row], eval_times, &eval_order);
            full_rmst
                .iter()
                .zip(leave_one_out)
                .map(|(&full, leave_one_out)| n_f64 * full - (n_f64 - 1.0) * leave_one_out)
                .collect()
        })
        .collect()
}

fn compute_rmst_repeated_jackknife_pseudo(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
) -> Vec<Vec<f64>> {
    let n_f64 = time.len() as f64;
    let full_rmst = compute_km(time, status, eval_times, "rmst");

    (0..time.len())
        .into_par_iter()
        .map(|i| {
            let loo_time: Vec<f64> = time
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &t)| t)
                .collect();
            let loo_status: Vec<i32> = status
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &s)| s)
                .collect();

            let leave_one_out = compute_km(&loo_time, &loo_status, eval_times, "rmst");

            full_rmst
                .iter()
                .zip(leave_one_out)
                .map(|(&full, leave_one_out)| n_f64 * full - (n_f64 - 1.0) * leave_one_out)
                .collect()
        })
        .collect()
}

fn has_non_exact_near_ties(time: &[f64]) -> bool {
    let mut sorted = time.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
        .windows(2)
        .any(|pair| pair[0] != pair[1] && same_time(pair[0], pair[1]))
}

fn compute_rmst_jackknife_pseudo(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
) -> Vec<Vec<f64>> {
    if has_non_exact_near_ties(time) {
        compute_rmst_repeated_jackknife_pseudo(time, status, eval_times)
    } else {
        compute_rmst_block_jackknife_pseudo(time, status, eval_times)
    }
}

fn validate_pseudo_type(type_: Option<&str>) -> PyResult<&'static str> {
    match type_.unwrap_or("survival") {
        "survival" => Ok("survival"),
        "cumhaz" => Ok("cumhaz"),
        "rmst" => Ok("rmst"),
        _ => Err(PyValueError::new_err(
            "type must be 'survival', 'cumhaz', or 'rmst'",
        )),
    }
}

fn validate_pseudo_inputs(
    time: &[f64],
    status: &[i32],
    eval_times: Option<&[f64]>,
) -> PyResult<()> {
    if status.len() != time.len() {
        return Err(PyValueError::new_err(
            "time and status must have same length",
        ));
    }

    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;

    if let Some(eval_times) = eval_times {
        validate_no_nan(eval_times, "eval_times")?;
        validate_finite(eval_times, "eval_times")?;
        validate_non_negative(eval_times, "eval_times")?;
    }

    Ok(())
}

fn default_event_times(time: &[f64], status: &[i32]) -> Vec<f64> {
    let mut event_times: Vec<f64> = time
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 1)
        .map(|(t, _)| *t)
        .collect();
    event_times.sort_by(|a, b| a.total_cmp(b));
    event_times.dedup_by(|a, b| same_time(*a, *b));
    event_times
}

fn rmst_values_at(km_times: &[f64], km_surv: &[f64], eval_times: &[f64]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(km_times.len());
    prefix.push(0.0);
    for idx in 1..km_times.len() {
        prefix.push(prefix[idx - 1] + km_surv[idx - 1] * (km_times[idx] - km_times[idx - 1]));
    }

    eval_times
        .iter()
        .map(|&eval_time| {
            let idx = km_times.partition_point(|&time| time <= eval_time);
            let idx = idx.saturating_sub(1);
            prefix[idx] + km_surv[idx] * (eval_time - km_times[idx])
        })
        .collect()
}

fn compute_km(time: &[f64], status: &[i32], eval_times: &[f64], type_: &str) -> Vec<f64> {
    let n = time.len();
    if n == 0 {
        return vec![1.0; eval_times.len()];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));

    let mut km_times = Vec::new();
    let mut km_surv = Vec::new();
    let mut km_cumhaz = Vec::new();

    let mut n_at_risk = n as f64;
    let mut surv = 1.0;
    let mut cumhaz = 0.0;
    let mut prev_time = f64::NEG_INFINITY;

    km_times.push(0.0);
    km_surv.push(1.0);
    km_cumhaz.push(0.0);

    let mut start = 0;
    while start < n {
        let current_time = time[indices[start]];
        let mut end = start + 1;
        while end < n && same_time(time[indices[end]], current_time) {
            end += 1;
        }

        let n_events = indices[start..end]
            .iter()
            .filter(|&&idx| status[idx] == 1)
            .count() as f64;
        let n_removed = (end - start) as f64;

        if n_events > 0.0 && n_at_risk > 0.0 {
            let hazard = n_events / n_at_risk;
            surv *= 1.0 - hazard;
            cumhaz += hazard;
        }

        n_at_risk -= n_removed;

        if current_time > prev_time + TIME_EPSILON {
            if current_time > *km_times.last().unwrap_or(&0.0) + TIME_EPSILON {
                km_times.push(current_time);
                km_surv.push(surv);
                km_cumhaz.push(cumhaz);
            } else {
                let last = km_times.len() - 1;
                km_surv[last] = surv;
                km_cumhaz[last] = cumhaz;
            }
            prev_time = current_time;
        }

        start = end;
    }

    if type_ == "rmst" {
        return rmst_values_at(&km_times, &km_surv, eval_times);
    }

    let mut result = Vec::with_capacity(eval_times.len());
    for &eval_t in eval_times {
        let val = match type_ {
            "survival" => {
                let idx = km_times.partition_point(|&time| time <= eval_t + TIME_EPSILON);
                km_surv[idx.saturating_sub(1)]
            }
            "cumhaz" => {
                let idx = km_times.partition_point(|&time| time <= eval_t + TIME_EPSILON);
                km_cumhaz[idx.saturating_sub(1)]
            }
            _ => unreachable!("pseudo type is validated before Kaplan-Meier evaluation"),
        };
        result.push(val);
    }

    result
}

#[pyfunction]
#[pyo3(signature = (time, status, eval_times=None, type_=None))]
pub fn pseudo_fast(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Option<Vec<f64>>,
    type_: Option<&str>,
) -> PyResult<PseudoResult> {
    pseudo(time, status, eval_times, type_)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    fn reference_compute_ij_pseudo(
        time: &[f64],
        status: &[i32],
        eval_times: &[f64],
        type_: &str,
    ) -> Vec<Vec<f64>> {
        let n_f64 = time.len() as f64;
        let blocks = event_blocks(time, status);
        (0..time.len())
            .map(|row| {
                eval_times
                    .iter()
                    .map(|&eval_time| {
                        let last_idx = blocks
                            .iter()
                            .position(|block| block.time > eval_time + TIME_EPSILON)
                            .map_or_else(
                                || blocks.len().checked_sub(1),
                                |idx| if idx == 0 { None } else { Some(idx - 1) },
                            );
                        let Some(last_idx) = last_idx else {
                            return if type_ == "survival" { 1.0 } else { 0.0 };
                        };
                        let mut influence = 0.0;
                        for block in &blocks[..=last_idx] {
                            if !subject_at_risk(time[row], block.time) {
                                continue;
                            }
                            if type_ == "survival" {
                                if subject_event_at_time(time[row], status[row], block.time) {
                                    influence -= 1.0 / block.risk;
                                } else if block.risk > block.events + DIVISION_FLOOR {
                                    influence +=
                                        block.events / (block.risk * (block.risk - block.events));
                                }
                            } else if subject_event_at_time(time[row], status[row], block.time) {
                                influence +=
                                    (block.risk - block.events) / (block.risk * block.risk);
                            } else {
                                influence -= block.events / (block.risk * block.risk);
                            }
                        }
                        let block = &blocks[last_idx];
                        if type_ == "survival" {
                            block.survival + n_f64 * block.survival * influence
                        } else {
                            block.cumhaz + n_f64 * influence
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn reference_rmst_values(km_times: &[f64], km_surv: &[f64], eval_times: &[f64]) -> Vec<f64> {
        eval_times
            .iter()
            .map(|&eval_time| {
                let mut rmst = 0.0;
                let mut previous_time = 0.0;
                let mut previous_survival = 1.0;
                for idx in 0..km_times.len() {
                    if km_times[idx] >= eval_time {
                        rmst += previous_survival * (eval_time - previous_time);
                        break;
                    }
                    rmst += previous_survival * (km_times[idx] - previous_time);
                    previous_time = km_times[idx];
                    previous_survival = km_surv[idx];
                    if idx == km_times.len() - 1 {
                        rmst += previous_survival * (eval_time - previous_time);
                    }
                }
                rmst
            })
            .collect()
    }

    fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
                assert_close(actual_value, expected_value);
            }
        }
    }

    #[test]
    fn test_pseudo_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let result = pseudo(time, status, None, Some("survival")).unwrap();

        assert_eq!(result.n, 5);
        assert!(!result.time.is_empty());
        assert_eq!(result.pseudo.len(), 5);

        for t_idx in 0..result.time.len() {
            let avg: f64 = result.pseudo.iter().map(|p| p[t_idx]).sum::<f64>() / 5.0;
            assert!(avg.is_finite());
        }
    }

    #[test]
    fn ij_pseudo_time_sweep_matches_repeated_block_scans() {
        for case_idx in 0..200 {
            let n = 3 + case_idx % 19;
            let time: Vec<f64> = (0..n)
                .map(|idx| {
                    let base = 1 + (idx * 7 + case_idx * 3) % 11;
                    let jitter = if (idx + case_idx) % 5 == 0 {
                        TIME_EPSILON / 2.0
                    } else {
                        0.0
                    };
                    base as f64 + jitter
                })
                .collect();
            let status: Vec<i32> = (0..n)
                .map(|idx| i32::from((idx * 5 + case_idx) % 4 != 0))
                .collect();
            let eval_times: Vec<f64> = (0..17)
                .map(|idx| {
                    let base = (idx * 5 + case_idx * 2) % 14;
                    let jitter = if (idx + case_idx) % 3 == 0 {
                        TIME_EPSILON / 2.0
                    } else {
                        0.0
                    };
                    base as f64 + jitter
                })
                .collect();

            for type_ in ["survival", "cumhaz"] {
                assert_matrix_close(
                    &compute_ij_pseudo(&time, &status, &eval_times, type_),
                    &reference_compute_ij_pseudo(&time, &status, &eval_times, type_),
                );
            }
        }
    }

    #[test]
    fn rmst_jackknife_dispatch_matches_repeated_leave_one_out_fits() {
        for case_idx in 0..160 {
            let n = 3 + case_idx % 23;
            let time: Vec<f64> = (0..n)
                .map(|idx| {
                    let base = 1 + (idx * 11 + case_idx * 5) % 17;
                    let jitter = if (idx + 2 * case_idx) % 7 == 0 {
                        TIME_EPSILON / 2.0
                    } else {
                        0.0
                    };
                    base as f64 + jitter
                })
                .collect();
            let status: Vec<i32> = (0..n)
                .map(|idx| i32::from((idx * 3 + case_idx) % 5 != 0))
                .collect();
            let eval_times: Vec<f64> = (0..21)
                .map(|idx| ((idx * 13 + case_idx * 7) % 23) as f64 * 0.75)
                .collect();

            assert_matrix_close(
                &compute_rmst_jackknife_pseudo(&time, &status, &eval_times),
                &compute_rmst_repeated_jackknife_pseudo(&time, &status, &eval_times),
            );
        }
    }

    #[test]
    fn rmst_block_jackknife_matches_exact_time_leave_one_out_fits() {
        for case_idx in 0..160 {
            let n = 3 + case_idx % 23;
            let time: Vec<f64> = (0..n)
                .map(|idx| (1 + (idx * 11 + case_idx * 5) % 17) as f64)
                .collect();
            let status: Vec<i32> = (0..n)
                .map(|idx| i32::from((idx * 3 + case_idx) % 5 != 0))
                .collect();
            let eval_times: Vec<f64> = (0..21)
                .map(|idx| ((idx * 13 + case_idx * 7) % 23) as f64 * 0.75)
                .collect();

            assert_matrix_close(
                &compute_rmst_block_jackknife_pseudo(&time, &status, &eval_times),
                &compute_rmst_repeated_jackknife_pseudo(&time, &status, &eval_times),
            );
        }
    }

    #[test]
    fn rmst_prefix_areas_match_repeated_step_scans() {
        for size in 1..80 {
            let km_times: Vec<f64> = (0..size)
                .map(|idx| idx as f64 * 0.75 + (idx % 3) as f64 * 0.125)
                .collect();
            let km_surv: Vec<f64> = (0..size)
                .map(|idx| 1.0 - 0.8 * idx as f64 / size as f64)
                .collect();
            let eval_times: Vec<f64> = (0..97)
                .map(|idx| ((idx * 31 + size * 7) % 101) as f64 * 0.625)
                .collect();

            let actual = rmst_values_at(&km_times, &km_surv, &eval_times);
            let expected = reference_rmst_values(&km_times, &km_surv, &eval_times);
            for (actual_value, expected_value) in actual.into_iter().zip(expected) {
                assert_close(actual_value, expected_value);
            }
        }
    }

    #[test]
    fn test_compute_km_groups_event_and_censor_ties() {
        let time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0];
        let status = vec![0, 1, 0];
        let eval_times = vec![1.0];

        let survival = compute_km(&time, &status, &eval_times, "survival");
        let cumhaz = compute_km(&time, &status, &eval_times, "cumhaz");

        assert_close(survival[0], 2.0 / 3.0);
        assert_close(cumhaz[0], 1.0 / 3.0);
    }

    #[test]
    fn test_default_event_times_deduplicate_near_ties() {
        let time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0];
        let status = vec![1, 1, 0];

        let event_times = default_event_times(&time, &status);

        assert_eq!(event_times.len(), 1);
        assert_close(event_times[0], 1.0);
    }

    #[test]
    fn test_pseudo_rejects_malformed_inputs() {
        let err = pseudo(vec![1.0, 2.0], vec![1, 2], None, Some("survival")).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = pseudo(vec![1.0, f64::INFINITY], vec![1, 0], None, Some("survival")).unwrap_err();
        assert!(err.to_string().contains("time contains non-finite"));

        let err = pseudo(
            vec![1.0, 2.0],
            vec![1, 0],
            Some(vec![-1.0]),
            Some("survival"),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("eval_times contains negative value")
        );

        let err = pseudo(vec![], vec![], None, Some("weird")).unwrap_err();
        assert!(
            err.to_string()
                .contains("type must be 'survival', 'cumhaz', or 'rmst'")
        );
    }

    #[test]
    fn test_pseudo_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = pseudo(time, status, None, None).unwrap();
        assert_eq!(result.n, 0);
    }

    #[test]
    fn test_pseudo_rmst() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let eval_times = vec![3.0];

        let result = pseudo(time, status, Some(eval_times), Some("rmst")).unwrap();

        assert_eq!(result.type_, "rmst");
        assert_eq!(result.pseudo.len(), 5);
    }

    #[test]
    fn test_pseudo_cumhaz() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];

        let result = pseudo(time, status, None, Some("cumhaz")).unwrap();

        assert_eq!(result.type_, "cumhaz");
        for p in &result.pseudo {
            for &val in p {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_pseudo_gee_regression() {
        let pseudo_values = vec![vec![0.8], vec![0.7], vec![0.6], vec![0.5], vec![0.4]];
        let covariates = vec![
            vec![1.0, 0.5],
            vec![1.0, 1.0],
            vec![1.0, 1.5],
            vec![1.0, 2.0],
            vec![1.0, 2.5],
        ];

        let config = GEEConfig::new(
            "independence".to_string(),
            "identity".to_string(),
            100,
            1e-6,
        )
        .unwrap();
        let result = pseudo_gee_regression(pseudo_values, covariates, None, Some(config)).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_errors.len(), 2);
    }

    #[test]
    fn test_pseudo_gee_rejects_malformed_inputs() {
        let err = GEEConfig::new("weird".to_string(), "identity".to_string(), 100, 1e-6)
            .expect_err("invalid correlation structure should fail");
        assert!(err.to_string().contains("correlation_structure"));

        let err = GEEConfig::new("independence".to_string(), "identity".to_string(), 0, 1e-6)
            .expect_err("zero max_iter should fail");
        assert!(err.to_string().contains("max_iter"));

        let err = pseudo_gee_regression(
            vec![vec![0.8], vec![0.7, 0.6]],
            vec![vec![1.0], vec![1.0]],
            None,
            None,
        )
        .expect_err("ragged pseudo_values should fail");
        assert!(err.to_string().contains("pseudo_values row 1"));

        let err = pseudo_gee_regression(
            vec![vec![0.8], vec![0.7]],
            vec![vec![1.0], vec![1.0]],
            Some(vec![0]),
            None,
        )
        .expect_err("short cluster_id should fail");
        assert!(err.to_string().contains("cluster_id length"));
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GEEConfig {
    #[pyo3(get, set)]
    pub correlation_structure: String,
    #[pyo3(get, set)]
    pub link_function: String,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl GEEConfig {
    #[new]
    #[pyo3(signature = (correlation_structure="independence".to_string(), link_function="identity".to_string(), max_iter=100, tol=1e-6))]
    pub fn new(
        correlation_structure: String,
        link_function: String,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let config = Self {
            correlation_structure,
            link_function,
            max_iter,
            tol,
        };
        validate_gee_config(&config)?;
        Ok(config)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GEEResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_values: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub confidence_intervals: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub qic: f64,
    #[pyo3(get)]
    pub n_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl GEEResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        z_values: Vec<f64>,
        p_values: Vec<f64>,
        confidence_intervals: Vec<(f64, f64)>,
        qic: f64,
        n_iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            coefficients,
            std_errors,
            z_values,
            p_values,
            confidence_intervals,
            qic,
            n_iterations,
            converged,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (pseudo_values, covariates, cluster_id=None, config=None))]
pub fn pseudo_gee_regression(
    pseudo_values: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    cluster_id: Option<Vec<usize>>,
    config: Option<GEEConfig>,
) -> PyResult<GEEResult> {
    let config = match config {
        Some(config) => {
            validate_gee_config(&config)?;
            config
        }
        None => GEEConfig::new(
            "independence".to_string(),
            "identity".to_string(),
            100,
            1e-6,
        )?,
    };

    validate_pseudo_gee_inputs(&pseudo_values, &covariates, cluster_id.as_deref())?;

    let n = pseudo_values.len();
    let n_times = pseudo_values[0].len();
    let p = covariates[0].len();
    let cluster_id = cluster_id.unwrap_or_else(|| (0..n).collect());

    let y: Vec<f64> = pseudo_values
        .iter()
        .flat_map(|pv| pv.iter().cloned())
        .collect();
    let n_obs = y.len();

    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for cov in covariates.iter() {
        for _ in 0..n_times {
            x.push(cov.clone());
        }
    }

    let mut beta: Vec<f64> = vec![0.0; p];
    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..config.max_iter {
        n_iterations = iter + 1;

        let eta: Vec<f64> = x
            .iter()
            .map(|xi| xi.iter().zip(beta.iter()).map(|(x, b)| x * b).sum())
            .collect();

        let mu: Vec<f64> = apply_link_inverse(&eta, &config.link_function);

        let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(y, m)| y - m).collect();

        let link_deriv: Vec<f64> = compute_link_derivative(&mu, &config.link_function);

        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];

        for i in 0..n_obs {
            let w = link_deriv[i].powi(2);
            for j in 0..p {
                xty[j] += x[i][j] * residuals[i] * w;
                for k in 0..p {
                    xtx[j][k] += x[i][j] * x[i][k] * w;
                }
            }
        }

        let xtx_inv = invert_matrix(&xtx);
        let delta: Vec<f64> = (0..p)
            .map(|j| xtx_inv[j].iter().zip(xty.iter()).map(|(a, b)| a * b).sum())
            .collect();

        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if delta_norm < config.tol {
            converged = true;
            break;
        }

        for k in 0..p {
            beta[k] += delta[k];
        }
    }

    let eta: Vec<f64> = x
        .iter()
        .map(|xi| xi.iter().zip(beta.iter()).map(|(x, b)| x * b).sum())
        .collect();
    let mu: Vec<f64> = apply_link_inverse(&eta, &config.link_function);
    let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(y, m)| y - m).collect();

    let sandwich_variance =
        compute_sandwich_variance(&x, &residuals, &cluster_id, n_times, p, &config);

    let std_errors: Vec<f64> = (0..p).map(|k| sandwich_variance[k][k].sqrt()).collect();

    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(b, se)| if *se > 0.0 { b / se } else { f64::NAN })
        .collect();

    let p_values: Vec<f64> = z_values
        .iter()
        .map(|z| {
            if z.is_finite() {
                2.0 * normal_sf(z.abs())
            } else {
                f64::NAN
            }
        })
        .collect();

    let confidence_intervals: Vec<(f64, f64)> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&beta, &std_error)| normal_ci_95(beta, std_error))
        .collect();

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let qic = n_obs as f64 * (rss / n_obs as f64).ln() + 2.0 * p as f64;

    Ok(GEEResult {
        coefficients: beta,
        std_errors,
        z_values,
        p_values,
        confidence_intervals,
        qic,
        n_iterations,
        converged,
    })
}

fn validate_gee_config(config: &GEEConfig) -> PyResult<()> {
    match config.correlation_structure.as_str() {
        "independence" | "exchangeable" | "ar1" => {}
        _ => {
            return Err(PyValueError::new_err(
                "correlation_structure must be 'independence', 'exchangeable', or 'ar1'",
            ));
        }
    }

    match config.link_function.as_str() {
        "identity" | "log" | "logit" | "cloglog" => {}
        _ => {
            return Err(PyValueError::new_err(
                "link_function must be 'identity', 'log', 'logit', or 'cloglog'",
            ));
        }
    }

    if config.max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(PyValueError::new_err(
            "tol must be finite and strictly positive",
        ));
    }

    Ok(())
}

fn validate_matrix_values(
    matrix: &[Vec<f64>],
    name: &'static str,
    require_non_empty_rows: bool,
) -> PyResult<usize> {
    let n_cols = matrix
        .first()
        .ok_or_else(|| PyValueError::new_err("Input data must be non-empty"))?
        .len();
    if require_non_empty_rows && n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} rows must not be empty"
        )));
    }

    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != n_cols {
            return Err(PyValueError::new_err(format!(
                "{name} row {row_idx} has {} columns, expected {n_cols}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if value.is_nan() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains NaN at row {row_idx}, column {col_idx}"
                )));
            }
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }

    Ok(n_cols)
}

fn validate_pseudo_gee_inputs(
    pseudo_values: &[Vec<f64>],
    covariates: &[Vec<f64>],
    cluster_id: Option<&[usize]>,
) -> PyResult<()> {
    if pseudo_values.is_empty() || covariates.is_empty() {
        return Err(PyValueError::new_err("Input data must be non-empty"));
    }
    if covariates.len() != pseudo_values.len() {
        return Err(PyValueError::new_err(format!(
            "covariates length must equal pseudo_values length; got {} and {}",
            covariates.len(),
            pseudo_values.len()
        )));
    }

    validate_matrix_values(pseudo_values, "pseudo_values", true)?;
    validate_matrix_values(covariates, "covariates", true)?;

    if let Some(cluster_id) = cluster_id
        && cluster_id.len() != pseudo_values.len()
    {
        return Err(PyValueError::new_err(format!(
            "cluster_id length must equal pseudo_values length; got {} and {}",
            cluster_id.len(),
            pseudo_values.len()
        )));
    }

    Ok(())
}

fn apply_link_inverse(eta: &[f64], link: &str) -> Vec<f64> {
    match link {
        "identity" => eta.to_vec(),
        "log" => eta.iter().map(|e| e.exp()).collect(),
        "logit" => eta.iter().map(|e| 1.0 / (1.0 + (-e).exp())).collect(),
        "cloglog" => eta.iter().map(|e| 1.0 - (-e.exp()).exp()).collect(),
        _ => eta.to_vec(),
    }
}

fn compute_link_derivative(mu: &[f64], link: &str) -> Vec<f64> {
    match link {
        "identity" => vec![1.0; mu.len()],
        "log" => mu.iter().map(|m| 1.0 / m.max(DIVISION_FLOOR)).collect(),
        "logit" => mu
            .iter()
            .map(|m| 1.0 / (m.max(DIVISION_FLOOR) * (1.0 - m).max(DIVISION_FLOOR)))
            .collect(),
        "cloglog" => mu
            .iter()
            .map(|m| {
                let m = m.clamp(DIVISION_FLOOR, 1.0 - DIVISION_FLOOR);
                1.0 / ((1.0 - m) * (-(1.0 - m).ln()))
            })
            .collect(),
        _ => vec![1.0; mu.len()],
    }
}

fn compute_sandwich_variance(
    x: &[Vec<f64>],
    residuals: &[f64],
    cluster_id: &[usize],
    n_times: usize,
    p: usize,
    _config: &GEEConfig,
) -> Vec<Vec<f64>> {
    let n_obs = x.len();

    let mut xtx = vec![vec![0.0; p]; p];
    for xi in x.iter() {
        for j in 0..p {
            for k in 0..p {
                xtx[j][k] += xi[j] * xi[k];
            }
        }
    }
    let xtx_inv = invert_matrix(&xtx);

    let mut meat = vec![vec![0.0; p]; p];
    let max_cluster = *cluster_id.iter().max().unwrap_or(&0);

    for c in 0..=max_cluster {
        let mut score = vec![0.0; p];
        for (i, &cluster) in cluster_id.iter().enumerate().take(n_obs / n_times) {
            if cluster == c {
                for t in 0..n_times {
                    let idx = i * n_times + t;
                    for j in 0..p {
                        score[j] += x[idx][j] * residuals[idx];
                    }
                }
            }
        }

        for j in 0..p {
            for k in 0..p {
                meat[j][k] += score[j] * score[k];
            }
        }
    }

    let mut result = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            for k in 0..p {
                for l in 0..p {
                    result[i][j] += xtx_inv[i][k] * meat[k][l] * xtx_inv[l][j];
                }
            }
        }
    }

    result
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    if n == 0 {
        return vec![];
    }

    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        let pivot = aug[i][i];
        if pivot.abs() < DIVISION_FLOOR {
            continue;
        }

        for val in aug[i].iter_mut() {
            *val /= pivot;
        }

        let row_i = aug[i].clone();
        for (k, row_k) in aug.iter_mut().enumerate() {
            if k != i {
                let factor = row_k[i];
                for (val, &ri) in row_k.iter_mut().zip(row_i.iter()) {
                    *val -= factor * ri;
                }
            }
        }
    }

    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = aug[i][n + j];
        }
    }

    result
}
