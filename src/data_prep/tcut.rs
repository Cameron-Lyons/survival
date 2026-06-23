use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::internal::numpy_utils::extract_vec_f64;

type ExpandedIntervals = (Vec<f64>, Vec<f64>, Vec<i32>, Vec<usize>);

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{} values must be finite, got non-finite value at index {}",
                name, idx
            )));
        }
    }
    Ok(())
}

fn sorted_unique_points(name: &str, mut values: Vec<f64>, min_len: usize) -> PyResult<Vec<f64>> {
    if values.len() < min_len {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "{} must have at least {} elements",
            name, min_len
        )));
    }
    validate_finite_values(name, &values)?;
    values.sort_by(|a, b| a.total_cmp(b));
    for window in values.windows(2) {
        if window[0] == window[1] {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{} must contain unique values",
                name
            )));
        }
    }
    Ok(values)
}

fn sequence(from: f64, to: f64, len: usize) -> Vec<f64> {
    if len <= 1 {
        return vec![from];
    }
    let step = (to - from) / ((len - 1) as f64);
    (0..len).map(|idx| from + step * (idx as f64)).collect()
}

fn interval_count_from_scalar_break(value: f64) -> PyResult<usize> {
    if !value.is_finite() {
        return Err(PyErr::new::<PyValueError, _>(
            "breaks values must be finite, got non-finite value at index 0",
        ));
    }
    if value < 1.0 {
        return Err(PyErr::new::<PyValueError, _>(
            "Must specify at least one interval",
        ));
    }
    let count = value.ceil();
    if count > (usize::MAX - 1) as f64 {
        return Err(PyErr::new::<PyValueError, _>(
            "breaks interval count is too large",
        ));
    }
    Ok(count as usize)
}

fn tcut_breaks_and_default_labels(
    value: &[f64],
    breaks: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<String>)> {
    if breaks.len() == 1 {
        let n_intervals = interval_count_from_scalar_break(breaks[0])?;
        if value.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "value must be non-empty when breaks is a scalar interval count",
            ));
        }

        let mut min_value = value[0];
        let mut max_value = value[0];
        for &current in &value[1..] {
            min_value = min_value.min(current);
            max_value = max_value.max(current);
        }

        let mut width = max_value - min_value;
        if width == 0.0 {
            max_value = min_value + 1.0;
            width = 1.0;
        }

        let generated_breaks = sequence(
            min_value - 0.01 * width,
            max_value + 0.01 * width,
            n_intervals + 1,
        );
        let labels = (1..=n_intervals)
            .map(|idx| format!("Range {idx}"))
            .collect();
        return Ok((generated_breaks, labels));
    }

    if breaks.len() < 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "breaks must have at least 2 elements",
        ));
    }
    validate_finite_values("breaks", &breaks)?;
    for window in breaks.windows(2) {
        if window[0] > window[1] {
            return Err(PyErr::new::<PyValueError, _>(
                "breaks must be given in ascending order and contain no NA's",
            ));
        }
    }

    let n_intervals = breaks.len() - 1;
    let labels = (0..n_intervals)
        .map(|i| {
            if i == n_intervals - 1 {
                format!("[{}, {}]", breaks[i], breaks[i + 1])
            } else {
                format!("[{}, {})", breaks[i], breaks[i + 1])
            }
        })
        .collect();

    Ok((breaks, labels))
}

/// Result of time cutting for person-years calculations
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TcutResult {
    /// Factor codes for each observation (0-indexed interval)
    #[pyo3(get)]
    pub codes: Vec<i32>,
    /// Labels for each interval
    #[pyo3(get)]
    pub levels: Vec<String>,
    /// The break points used
    #[pyo3(get)]
    pub breaks: Vec<f64>,
    /// Count of observations in each interval
    #[pyo3(get)]
    pub counts: Vec<usize>,
}

/// Create factor for person-years calculations with time-dependent cutpoints.
///
/// This function assigns observations to intervals based on break points,
/// creating a factor suitable for use with person-years calculations.
/// Unlike regular cut, this is designed for time-varying data where
/// subjects can contribute to multiple intervals.
///
/// # Arguments
/// * `value` - Vector of time values to categorize
/// * `breaks` - Vector of break points defining intervals
/// * `labels` - Optional labels for each interval (length should be len(breaks) - 1)
///
/// # Returns
/// * `TcutResult` with interval codes and level information
#[pyfunction]
#[pyo3(signature = (value, breaks, labels=None))]
pub fn tcut(
    value: &Bound<'_, PyAny>,
    breaks: &Bound<'_, PyAny>,
    labels: Option<Vec<String>>,
) -> PyResult<TcutResult> {
    let value = extract_vec_f64(value)?;
    let breaks = match breaks.extract::<f64>() {
        Ok(scalar) => vec![scalar],
        Err(_) => extract_vec_f64(breaks)?,
    };
    tcut_from_vecs(value, breaks, labels)
}

fn tcut_from_vecs(
    value: Vec<f64>,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
) -> PyResult<TcutResult> {
    validate_finite_values("value", &value)?;
    let (cut_breaks, default_labels) = tcut_breaks_and_default_labels(&value, breaks)?;

    let n_intervals = cut_breaks.len() - 1;

    let interval_labels = match labels {
        Some(l) => {
            if l.len() != n_intervals {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "labels length ({}) must equal number of intervals ({})",
                    l.len(),
                    n_intervals
                )));
            }
            l
        }
        None => default_labels,
    };

    let mut codes = Vec::with_capacity(value.len());
    let mut counts = vec![0usize; n_intervals];

    for &v in &value {
        let code = find_interval(&cut_breaks, v);
        codes.push(code);
        if code >= 0 {
            counts[code as usize] += 1;
        }
    }

    Ok(TcutResult {
        codes,
        levels: interval_labels,
        breaks: cut_breaks,
        counts,
    })
}

/// Find which interval a value belongs to.
/// Returns -1 if outside all intervals.
fn find_interval(breaks: &[f64], value: f64) -> i32 {
    let n = breaks.len();
    if n < 2 {
        return -1;
    }

    if value < breaks[0] || value > breaks[n - 1] {
        return -1;
    }

    if value == breaks[n - 1] {
        return (n - 2) as i32;
    }

    let pos = breaks.partition_point(|&breakpoint| breakpoint <= value);
    (pos.saturating_sub(1)) as i32
}

fn find_expanded_interval_code(cuts: &[f64], midpoint: f64) -> i32 {
    let pos = cuts.partition_point(|&cut| cut <= midpoint);
    if pos == 0 { -1 } else { (pos - 1) as i32 }
}

/// Split time intervals for person-years analysis.
///
/// This function takes start/stop times and splits them at the specified
/// cut points, returning expanded data suitable for pyears calculations.
///
/// # Arguments
/// * `start` - Start times for each interval
/// * `stop` - Stop times for each interval
/// * `cuts` - Cut points to split at
///
/// # Returns
/// * Tuple of (new_start, new_stop, interval_codes, original_indices)
#[pyfunction]
pub fn tcut_expand(start: Vec<f64>, stop: Vec<f64>, cuts: Vec<f64>) -> PyResult<ExpandedIntervals> {
    let n = start.len();
    if stop.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start and stop must have same length",
        ));
    }
    validate_finite_values("start", &start)?;
    validate_finite_values("stop", &stop)?;

    let sorted_cuts = sorted_unique_points("cuts", cuts, 1)?;

    let mut new_start = Vec::new();
    let mut new_stop = Vec::new();
    let mut interval_codes = Vec::new();
    let mut original_indices = Vec::new();

    for i in 0..n {
        let t1 = start[i];
        let t2 = stop[i];

        if t1 >= t2 {
            continue;
        }

        let first_cut = sorted_cuts.partition_point(|&c| c <= t1);
        let last_cut = sorted_cuts.partition_point(|&c| c < t2);

        let mut split_points = Vec::with_capacity(last_cut.saturating_sub(first_cut) + 2);
        split_points.push(t1);
        split_points.extend_from_slice(&sorted_cuts[first_cut..last_cut]);
        split_points.push(t2);

        for j in 0..(split_points.len() - 1) {
            let s = split_points[j];
            let e = split_points[j + 1];

            new_start.push(s);
            new_stop.push(e);

            let midpoint = s + (e - s) / 2.0;
            interval_codes.push(find_expanded_interval_code(&sorted_cuts, midpoint));
            original_indices.push(i);
        }
    }

    Ok((new_start, new_stop, interval_codes, original_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcut_basic() {
        let values = vec![5.0, 15.0, 25.0, 35.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0, 40.0];

        let result = tcut_from_vecs(values, breaks, None).unwrap();
        assert_eq!(result.codes, vec![0, 1, 2, 3]);
        assert_eq!(result.levels.len(), 4);
    }

    #[test]
    fn test_tcut_with_labels() {
        let values = vec![5.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0];
        let labels = vec!["young".to_string(), "old".to_string()];

        let result = tcut_from_vecs(values, breaks, Some(labels)).unwrap();
        assert_eq!(result.levels, vec!["young", "old"]);
    }

    #[test]
    fn test_tcut_scalar_break_count_generates_range_intervals() {
        let result = tcut_from_vecs(vec![5.0, 15.0, 25.0], vec![3.0], None).unwrap();

        assert_eq!(result.codes, vec![0, 1, 2]);
        assert_eq!(result.levels, vec!["Range 1", "Range 2", "Range 3"]);
        assert_eq!(result.breaks, vec![4.8, 11.6, 18.4, 25.2]);
        assert_eq!(result.counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_tcut_scalar_break_count_pads_constant_values() {
        let result = tcut_from_vecs(vec![5.0, 5.0, 5.0], vec![2.0], None).unwrap();

        assert_eq!(result.codes, vec![0, 0, 0]);
        assert_eq!(result.levels, vec!["Range 1", "Range 2"]);
        assert_eq!(result.breaks, vec![4.99, 5.5, 6.01]);
        assert_eq!(result.counts, vec![3, 0]);
    }

    #[test]
    fn test_tcut_outside_range() {
        let values = vec![-5.0, 50.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        let result = tcut_from_vecs(values, breaks, None).unwrap();
        assert_eq!(result.codes[0], -1);
        assert_eq!(result.codes[1], -1);
        assert_eq!(result.codes[2], 1);
    }

    #[test]
    fn test_tcut_breakpoint_boundaries_are_left_closed() {
        let values = vec![0.0, 10.0, 20.0, 30.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        let result = tcut_from_vecs(values, breaks, None).unwrap();
        assert_eq!(result.codes, vec![0, 1, 2, 2]);
        assert_eq!(result.counts, vec![1, 1, 2]);
    }

    #[test]
    fn test_tcut_duplicate_ordered_breaks_are_allowed() {
        let result =
            tcut_from_vecs(vec![5.0, 15.0, 25.0], vec![0.0, 10.0, 10.0, 30.0], None).unwrap();

        assert_eq!(result.codes, vec![0, 2, 2]);
        assert_eq!(result.counts, vec![1, 0, 2]);
        assert_eq!(result.breaks, vec![0.0, 10.0, 10.0, 30.0]);
    }

    #[test]
    fn test_tcut_expand_basic() {
        let start = vec![0.0, 5.0];
        let stop = vec![25.0, 15.0];
        let cuts = vec![0.0, 10.0, 20.0, 30.0];

        let (new_start, new_stop, codes, indices) = tcut_expand(start, stop, cuts).unwrap();

        assert!(new_start.len() > 2);
        assert_eq!(new_start.len(), new_stop.len());
        assert_eq!(new_start.len(), codes.len());
        assert_eq!(new_start.len(), indices.len());
    }

    #[test]
    fn test_tcut_expand_codes_before_between_and_after_cuts() {
        let (new_start, new_stop, codes, indices) =
            tcut_expand(vec![-5.0, 35.0], vec![25.0, 40.0], vec![0.0, 10.0, 20.0]).unwrap();

        assert_eq!(new_start, vec![-5.0, 0.0, 10.0, 20.0, 35.0]);
        assert_eq!(new_stop, vec![0.0, 10.0, 20.0, 25.0, 40.0]);
        assert_eq!(codes, vec![-1, 0, 1, 2, 2]);
        assert_eq!(indices, vec![0, 0, 0, 0, 1]);
    }

    #[test]
    fn test_tcut_rejects_malformed_breaks_and_values() {
        assert!(tcut_from_vecs(vec![f64::NAN], vec![0.0, 1.0], None).is_err());
        assert!(tcut_from_vecs(vec![0.5], vec![0.0, f64::INFINITY], None).is_err());
        assert!(tcut_from_vecs(vec![0.5], vec![2.0, 1.0], None).is_err());
        assert!(tcut_from_vecs(vec![0.5], vec![0.0], None).is_err());
        assert!(tcut_from_vecs(Vec::new(), vec![2.0], None).is_err());
    }

    #[test]
    fn test_tcut_expand_rejects_malformed_inputs() {
        assert!(tcut_expand(vec![f64::NAN], vec![1.0], vec![0.0]).is_err());
        assert!(tcut_expand(vec![0.0], vec![f64::INFINITY], vec![0.0]).is_err());
        assert!(tcut_expand(vec![0.0], vec![1.0], vec![]).is_err());
        assert!(tcut_expand(vec![0.0], vec![1.0], vec![0.0, 0.0]).is_err());
    }
}
