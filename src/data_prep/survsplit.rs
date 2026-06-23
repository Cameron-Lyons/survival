use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct SplitResult {
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub interval: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    #[pyo3(get)]
    pub censor: Vec<bool>,
}

fn interior_cut_bounds(cutpoints: &[f64], start: f64, stop: f64) -> (usize, usize) {
    if start >= stop {
        return (0, 0);
    }

    let first = cutpoints.partition_point(|&cutpoint| cutpoint <= start);
    let last = cutpoints.partition_point(|&cutpoint| cutpoint < stop);
    (first, last)
}

#[pyfunction]
pub fn survsplit(tstart: Vec<f64>, tstop: Vec<f64>, cut: Vec<f64>) -> PyResult<SplitResult> {
    let n = tstart.len();
    if tstop.len() != n {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "tstart and tstop must have same length, got {} and {}",
            n,
            tstop.len()
        )));
    }

    for i in 0..n {
        if tstart[i].is_infinite() || tstop[i].is_infinite() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "tstart and tstop may be finite or NaN, but not infinite; got infinite value at row {}",
                i + 1
            )));
        }
    }

    for (idx, value) in cut.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "cut must be a vector of finite numbers; got non-finite value at index {}",
                idx
            )));
        }
    }

    let mut cutpoints = cut;
    cutpoints.sort_by(|a, b| a.total_cmp(b));
    cutpoints.dedup_by(|a, b| a == b);

    let mut extra = 0;
    for i in 0..n {
        if tstart[i].is_nan() || tstop[i].is_nan() {
            continue;
        }
        let (first, last) = interior_cut_bounds(&cutpoints, tstart[i], tstop[i]);
        extra += last - first;
    }
    let n2 = n + extra;
    let mut result = SplitResult {
        row: Vec::with_capacity(n2),
        interval: Vec::with_capacity(n2),
        start: Vec::with_capacity(n2),
        end: Vec::with_capacity(n2),
        censor: Vec::with_capacity(n2),
    };
    for i in 0..n {
        let current_start = tstart[i];
        let current_stop = tstop[i];
        if current_start.is_nan() || current_stop.is_nan() {
            result.row.push(i + 1);
            result.interval.push(1);
            result.start.push(current_start);
            result.end.push(current_stop);
            result.censor.push(false);
            continue;
        }
        let mut current = current_start;
        let mut interval_num = 1;
        let (first, last) = interior_cut_bounds(&cutpoints, current_start, current_stop);
        for &cutpoint in &cutpoints[first..last] {
            result.row.push(i + 1);
            result.interval.push(interval_num);
            result.start.push(current);
            result.end.push(cutpoint);
            result.censor.push(true);
            current = cutpoint;
            interval_num += 1;
        }
        result.row.push(i + 1);
        result.interval.push(interval_num);
        result.start.push(current);
        result.end.push(current_stop);
        result.censor.push(false);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn no_cuts() {
        let result = survsplit(vec![0.0, 5.0], vec![10.0, 15.0], vec![]).unwrap();
        assert_eq!(result.row.len(), 2);
        assert_eq!(result.start, vec![0.0, 5.0]);
        assert_eq!(result.end, vec![10.0, 15.0]);
    }

    #[test]
    fn single_cut_splits_interval() {
        let result = survsplit(vec![0.0], vec![10.0], vec![5.0]).unwrap();
        assert_eq!(result.row.len(), 2);
        assert_eq!(result.start, vec![0.0, 5.0]);
        assert_eq!(result.end, vec![5.0, 10.0]);
        assert_eq!(result.censor, vec![true, false]);
    }

    #[test]
    fn multiple_cuts() {
        let result = survsplit(vec![0.0], vec![10.0], vec![3.0, 7.0]).unwrap();
        assert_eq!(result.row.len(), 3);
        assert_eq!(result.start, vec![0.0, 3.0, 7.0]);
        assert_eq!(result.end, vec![3.0, 7.0, 10.0]);
    }

    #[test]
    fn unsorted_duplicate_cuts_are_normalized() {
        let result = survsplit(vec![0.0], vec![10.0], vec![7.0, 3.0, 3.0]).unwrap();
        assert_eq!(result.start, vec![0.0, 3.0, 7.0]);
        assert_eq!(result.end, vec![3.0, 7.0, 10.0]);
    }

    #[test]
    fn cut_outside_interval() {
        let result = survsplit(vec![0.0], vec![5.0], vec![10.0]).unwrap();
        assert_eq!(result.row.len(), 1);
        assert_eq!(result.start, vec![0.0]);
        assert_eq!(result.end, vec![5.0]);
    }

    #[test]
    fn cuts_equal_to_interval_boundaries_do_not_split() {
        let result = survsplit(vec![0.0], vec![10.0], vec![0.0, 3.0, 10.0]).unwrap();

        assert_eq!(result.row, vec![1, 1]);
        assert_eq!(result.interval, vec![1, 2]);
        assert_eq!(result.start, vec![0.0, 3.0]);
        assert_eq!(result.end, vec![3.0, 10.0]);
        assert_eq!(result.censor, vec![true, false]);
    }

    #[test]
    fn reversed_or_zero_length_intervals_are_not_split() {
        let result = survsplit(vec![5.0, 2.0], vec![5.0, 1.0], vec![1.5, 3.0, 4.0]).unwrap();

        assert_eq!(result.row, vec![1, 2]);
        assert_eq!(result.interval, vec![1, 1]);
        assert_eq!(result.start, vec![5.0, 2.0]);
        assert_eq!(result.end, vec![5.0, 1.0]);
        assert_eq!(result.censor, vec![false, false]);
    }

    #[test]
    fn nan_handling() {
        let result = survsplit(vec![f64::NAN], vec![f64::NAN], vec![5.0]).unwrap();
        assert_eq!(result.row.len(), 1);
        assert!(result.start[0].is_nan());
        assert!(result.end[0].is_nan());
    }

    #[test]
    fn infinite_endpoints_are_value_error() {
        initialize_python();

        let err = survsplit(vec![0.0], vec![f64::INFINITY], vec![5.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("tstart and tstop may be finite or NaN, but not infinite")
        );

        let err = survsplit(vec![f64::NEG_INFINITY], vec![10.0], vec![5.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("tstart and tstop may be finite or NaN, but not infinite")
        );
    }

    #[test]
    fn multiple_observations() {
        let result = survsplit(vec![0.0, 0.0, 0.0], vec![10.0, 5.0, 8.0], vec![3.0, 7.0]).unwrap();
        assert_eq!(result.row.iter().filter(|&&r| r == 1).count(), 3);
        assert_eq!(result.row.iter().filter(|&&r| r == 2).count(), 2);
        assert_eq!(result.row.iter().filter(|&&r| r == 3).count(), 3);
    }

    #[test]
    fn length_mismatch_is_value_error() {
        initialize_python();

        let err = survsplit(vec![0.0], vec![], vec![1.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("tstart and tstop must have same length")
        );
    }

    #[test]
    fn nonfinite_cuts_are_value_error() {
        initialize_python();

        let err = survsplit(vec![0.0], vec![10.0], vec![3.0, f64::NAN]).unwrap_err();
        assert!(
            err.to_string()
                .contains("cut must be a vector of finite numbers")
        );

        let err = survsplit(vec![0.0], vec![10.0], vec![f64::INFINITY]).unwrap_err();
        assert!(
            err.to_string()
                .contains("cut must be a vector of finite numbers")
        );
    }
}
