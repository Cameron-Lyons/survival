use pyo3::prelude::*;

use crate::internal::validation::{
    PermutationIndexError, validate_binary_f64, validate_zero_based_usize_permutation,
};

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn validate_same_length(n: usize, actual: usize, name: &str) -> PyResult<()> {
    if actual != n {
        return Err(value_error(format!(
            "{name} length must match time length ({actual} != {n})"
        )));
    }
    Ok(())
}

fn validate_finite(values: &[f64], name: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} must contain only finite values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_strata_markers(values: &[i32], n: usize) -> PyResult<()> {
    if n > i32::MAX as usize {
        return Err(value_error(
            "input length exceeds i32 output index capacity",
        ));
    }
    for (idx, &value) in values.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(value_error(format!(
                "strata values must be 0 or 1; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_sort_indices(values: &[usize], n: usize, name: &str) -> PyResult<()> {
    match validate_zero_based_usize_permutation(values, n) {
        Ok(()) => Ok(()),
        Err(PermutationIndexError::OutOfBounds { position, value }) => Err(value_error(format!(
            "{name} index out of bounds at position {position}: {value} >= {n}"
        ))),
        Err(PermutationIndexError::Duplicate { position, value }) => Err(value_error(format!(
            "{name} must be a permutation of 0..{n}; duplicate index {value} at position {position}"
        ))),
        Err(PermutationIndexError::Negative { .. }) => {
            unreachable!("usize indices are never negative")
        }
    }
}

fn validate_coxcount1_inputs(time: &[f64], status: &[f64], strata: &[i32]) -> PyResult<()> {
    let n = time.len();
    validate_same_length(n, status.len(), "status")?;
    validate_same_length(n, strata.len(), "strata")?;
    validate_finite(time, "time")?;
    validate_binary_f64(status, "status")?;
    validate_strata_markers(strata, n)
}

fn validate_coxcount2_inputs(
    time1: &[f64],
    time2: &[f64],
    status: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    strata: &[i32],
) -> PyResult<()> {
    let n = time1.len();
    validate_same_length(n, time2.len(), "time2")?;
    validate_same_length(n, status.len(), "status")?;
    validate_same_length(n, sort1.len(), "sort1")?;
    validate_same_length(n, sort2.len(), "sort2")?;
    validate_same_length(n, strata.len(), "strata")?;
    validate_finite(time1, "time1")?;
    validate_finite(time2, "time2")?;
    validate_binary_f64(status, "status")?;
    validate_strata_markers(strata, n)?;
    validate_sort_indices(sort1, n, "sort1")?;
    validate_sort_indices(sort2, n, "sort2")
}

#[pyclass]
pub struct CoxCountOutput {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub nrisk: Vec<i32>,
    #[pyo3(get)]
    pub index: Vec<i32>,
    #[pyo3(get)]
    pub status: Vec<i32>,
}
fn coxcount1_impl(time: &[f64], status: &[f64], strata: &[i32]) -> PyResult<CoxCountOutput> {
    validate_coxcount1_inputs(time, status, strata)?;
    let n = time.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut nrisk = 0;
    let mut i = 0;
    while i < n {
        if strata[i] == 1 {
            nrisk = 0;
        }
        nrisk += 1;
        if status[i] == 1.0 {
            let dtime = time[i];
            let mut j = i + 1;
            while j < n && time[j] == dtime && status[j] == 1.0 && strata[j] == 0 {
                nrisk += 1;
                j += 1;
            }
            ntime += 1;
            nrow += nrisk;
            i = j - 1;
        }
        i += 1;
    }
    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);
    let mut stratum_start = 0;
    let mut i = 0;
    while i < n {
        if strata[i] == 1 {
            stratum_start = i;
        }
        if status[i] == 1.0 {
            let dtime = time[i];
            let mut j = i + 1;
            while j < n && time[j] == dtime && status[j] == 1.0 && strata[j] == 0 {
                j += 1;
            }
            for k in stratum_start..i {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }
            for k in i..j {
                status_vec.push(1);
                index_vec.push((k + 1) as i32);
            }
            time_vec.push(dtime);
            nrisk_vec.push((j - stratum_start) as i32);
            i = j - 1;
        }
        i += 1;
    }
    Ok(CoxCountOutput {
        time: time_vec,
        nrisk: nrisk_vec,
        index: index_vec,
        status: status_vec,
    })
}

#[pyfunction]
pub fn coxcount1(
    time: Vec<f64>,
    status: Vec<f64>,
    strata: Vec<i32>,
) -> PyResult<Py<CoxCountOutput>> {
    let output = coxcount1_impl(&time, &status, &strata)?;
    Python::attach(|py| Py::new(py, output))
}

fn coxcount2_impl(
    time1: &[f64],
    time2: &[f64],
    status: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    strata: &[i32],
) -> PyResult<CoxCountOutput> {
    validate_coxcount2_inputs(time1, time2, status, sort1, sort2, strata)?;
    let n = time1.len();
    let mut ntime = 0;
    let mut nrow = 0;
    let mut j = 0;
    let mut i = 0;
    let mut nrisk = 0;
    while i < n {
        let iptr = sort2[i];
        if strata[i] == 1 {
            nrisk = 0;
            j = i;
        }
        if status[iptr] == 1.0 {
            let dtime = time2[iptr];
            while j < i && time1[sort1[j]] >= dtime {
                if nrisk == 0 {
                    return Err(value_error(
                        "coxcount2 sort order is inconsistent with the risk set",
                    ));
                }
                nrisk -= 1;
                j += 1;
            }
            nrisk += 1;
            i += 1;
            // The native routine indexes boundary markers through the original
            // row number while extending a tied counting-process event set.
            while i < n && strata[sort2[i]] == 0 && time2[sort2[i]] == dtime {
                nrisk += 1;
                i += 1;
            }
            nrow += nrisk;
            ntime += 1;
        } else {
            nrisk += 1;
            i += 1;
        }
    }
    let mut time_vec = Vec::with_capacity(ntime);
    let mut nrisk_vec = Vec::with_capacity(ntime);
    let mut index_vec = Vec::with_capacity(nrow);
    let mut status_vec = Vec::with_capacity(nrow);
    let mut atrisk = vec![0; n];
    let mut who = Vec::with_capacity(n);
    let mut j = 0;
    let mut i = 0;
    while i < n {
        let iptr = sort2[i];
        if strata[i] == 1 {
            who.clear();
            j = i;
        }
        if status[iptr] == 0.0 {
            atrisk[iptr] = who.len();
            who.push(iptr);
            i += 1;
        } else {
            let dtime = time2[iptr];
            while j < i {
                let jptr = sort1[j];
                if time1[jptr] >= dtime {
                    let pos = atrisk[jptr];
                    if pos >= who.len() || who[pos] != jptr {
                        return Err(value_error(
                            "coxcount2 sort order is inconsistent with the risk set",
                        ));
                    }
                    who.swap_remove(pos);
                    if pos < who.len() {
                        atrisk[who[pos]] = pos;
                    }
                    j += 1;
                } else {
                    break;
                }
            }
            for &k in &who {
                status_vec.push(0);
                index_vec.push((k + 1) as i32);
            }
            status_vec.push(1);
            index_vec.push((iptr + 1) as i32);
            atrisk[iptr] = who.len();
            who.push(iptr);
            i += 1;
            // Preserve the same original-row boundary lookup used above.
            while i < n && strata[sort2[i]] == 0 && time2[sort2[i]] == dtime {
                let k = sort2[i];
                status_vec.push(1);
                index_vec.push((k + 1) as i32);
                atrisk[k] = who.len();
                who.push(k);
                i += 1;
            }
            time_vec.push(dtime);
            nrisk_vec.push(who.len() as i32);
        }
    }
    Ok(CoxCountOutput {
        time: time_vec,
        nrisk: nrisk_vec,
        index: index_vec,
        status: status_vec,
    })
}

#[pyfunction]
pub fn coxcount2(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    strata: Vec<i32>,
) -> PyResult<Py<CoxCountOutput>> {
    let output = coxcount2_impl(&time1, &time2, &status, &sort1, &sort2, &strata)?;
    Python::attach(|py| Py::new(py, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn coxcount1_rejects_mismatched_lengths() {
        initialize_python();

        let err = match coxcount1(vec![1.0], vec![1.0, 0.0], vec![1]) {
            Ok(_) => panic!("mismatched status length should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("status length"));
    }

    #[test]
    fn coxcount1_rejects_non_binary_status() {
        initialize_python();

        let err = match coxcount1(vec![1.0], vec![2.0], vec![1]) {
            Ok(_) => panic!("non-binary status should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn coxcount2_rejects_out_of_bounds_sort_index() {
        initialize_python();

        let err = match coxcount2(vec![0.0], vec![1.0], vec![1.0], vec![1], vec![0], vec![1]) {
            Ok(_) => panic!("out-of-bounds sort1 index should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("sort1 index out of bounds"));
    }

    #[test]
    fn coxcount2_rejects_duplicate_sort_index() {
        initialize_python();

        let err = match coxcount2(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 0.0],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
        ) {
            Ok(_) => panic!("duplicate sort1 index should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("sort1 must be a permutation"));
    }

    #[test]
    fn coxcount1_keeps_adjacent_floating_point_times_separate() {
        initialize_python();
        let previous = f64::from_bits(1.0f64.to_bits() - 1);
        let output = coxcount1_impl(&[1.0, previous], &[1.0, 1.0], &[1, 0]).unwrap();

        assert_eq!(output.time, vec![1.0, previous]);
        assert_eq!(output.nrisk, vec![1, 2]);
        assert_eq!(output.index, vec![1, 1, 2]);
        assert_eq!(output.status, vec![1, 0, 1]);
    }

    #[test]
    fn coxcount2_matches_stratified_tied_event_reference() {
        initialize_python();
        let output = coxcount2_impl(
            &[
                0.0, 4.0, 5.0, 1.0, 6.0, 1.0, 3.0, 5.0, 1.0, 2.0, 2.0, 5.0, 5.0,
            ],
            &[
                5.0, 5.0, 6.0, 4.0, 9.0, 4.0, 4.0, 6.0, 4.0, 5.0, 3.0, 9.0, 8.0,
            ],
            &[
                1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            &[4, 12, 6, 10, 0, 7, 11, 1, 9, 3, 5, 2, 8],
            &[4, 12, 0, 6, 10, 11, 7, 1, 9, 3, 5, 2, 8],
            &[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        )
        .unwrap();

        assert_eq!(output.time, vec![9.0, 5.0, 9.0, 4.0, 4.0, 6.0]);
        assert_eq!(output.nrisk, vec![1, 1, 1, 2, 3, 1]);
        assert_eq!(output.index, vec![5, 1, 12, 10, 4, 10, 4, 6, 3]);
        assert_eq!(output.status, vec![1, 1, 1, 0, 1, 0, 0, 1, 1]);
    }
}
