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
    if breaks.is_empty() {
        return Ok((breaks, Vec::new()));
    }
    if breaks.len() == 1 {
        let n_intervals = interval_count_from_scalar_break(breaks[0])?;
        if value.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "value must be non-empty when breaks is a scalar interval count",
            ));
        }
        if value.iter().any(|current| current.is_infinite()) {
            return Err(PyErr::new::<PyValueError, _>(
                "value must not contain infinite values when breaks is a scalar interval count",
            ));
        }

        let mut finite_values = value.iter().copied().filter(|current| current.is_finite());
        let Some(mut min_value) = finite_values.next() else {
            return Err(PyErr::new::<PyValueError, _>(
                "value must contain a finite value when breaks is a scalar interval count",
            ));
        };
        let mut max_value = min_value;
        for current in finite_values {
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
    if breaks.iter().any(|value| value.is_nan()) {
        return Err(PyErr::new::<PyValueError, _>(
            "breaks must be given in ascending order and contain no NA's",
        ));
    }
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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TcutResult {
    #[pyo3(get)]
    pub values: Vec<f64>,
    #[pyo3(get)]
    pub codes: Vec<i32>,
    #[pyo3(get)]
    pub levels: Vec<String>,
    #[pyo3(get)]
    pub breaks: Vec<f64>,
    #[pyo3(get)]
    pub counts: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (value, breaks, labels=None, scale=1.0))]
pub fn tcut(
    value: &Bound<'_, PyAny>,
    breaks: &Bound<'_, PyAny>,
    labels: Option<Vec<String>>,
    scale: f64,
) -> PyResult<TcutResult> {
    let value = extract_vec_f64(value)?;
    let breaks = match breaks.extract::<f64>() {
        Ok(scalar) => vec![scalar],
        Err(_) => extract_vec_f64(breaks)?,
    };
    tcut_from_vecs(value, breaks, labels, scale)
}

fn tcut_from_vecs(
    value: Vec<f64>,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    scale: f64,
) -> PyResult<TcutResult> {
    if let Some(labels) = labels.as_ref() {
        let valid_length = if breaks.len() == 1 {
            labels.len() as f64 == breaks[0]
        } else {
            !breaks.is_empty() && labels.len() == breaks.len() - 1
        };
        if !valid_length {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "labels length does not match the number of intervals",
            ));
        }
    }
    let (cut_breaks, default_labels) = tcut_breaks_and_default_labels(&value, breaks)?;

    let n_intervals = cut_breaks.len().saturating_sub(1);

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
        values: value.into_iter().map(|item| item * scale).collect(),
        codes,
        levels: interval_labels,
        breaks: cut_breaks.into_iter().map(|item| item * scale).collect(),
        counts,
    })
}

fn find_interval(breaks: &[f64], value: f64) -> i32 {
    let n = breaks.len();
    if n < 2 || value.is_nan() {
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

        let result = tcut_from_vecs(values, breaks, None, 1.0).unwrap();
        assert_eq!(result.codes, vec![0, 1, 2, 3]);
        assert_eq!(result.levels.len(), 4);
    }

    #[test]
    fn test_tcut_with_labels() {
        let values = vec![5.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0];
        let labels = vec!["young".to_string(), "old".to_string()];

        let result = tcut_from_vecs(values, breaks, Some(labels), 1.0).unwrap();
        assert_eq!(result.levels, vec!["young", "old"]);
    }

    #[test]
    fn test_tcut_scalar_break_count_generates_range_intervals() {
        let result = tcut_from_vecs(vec![5.0, 15.0, 25.0], vec![3.0], None, 1.0).unwrap();

        assert_eq!(result.codes, vec![0, 1, 2]);
        assert_eq!(result.levels, vec!["Range 1", "Range 2", "Range 3"]);
        assert_eq!(result.breaks, vec![4.8, 11.6, 18.4, 25.2]);
        assert_eq!(result.counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_tcut_scalar_break_count_pads_constant_values() {
        let result = tcut_from_vecs(vec![5.0, 5.0, 5.0], vec![2.0], None, 1.0).unwrap();

        assert_eq!(result.codes, vec![0, 0, 0]);
        assert_eq!(result.levels, vec!["Range 1", "Range 2"]);
        assert_eq!(result.breaks, vec![4.99, 5.5, 6.01]);
        assert_eq!(result.counts, vec![3, 0]);
    }

    #[test]
    fn test_tcut_scalar_break_count_preserves_missing_values() {
        let result = tcut_from_vecs(vec![1.0, f64::NAN, 3.0], vec![2.0], None, 1.0).unwrap();

        assert_eq!(result.codes, vec![0, -1, 1]);
        assert_eq!(result.levels, vec!["Range 1", "Range 2"]);
        assert_eq!(result.breaks, vec![0.98, 2.0, 3.02]);
        assert_eq!(result.counts, vec![1, 1]);
    }

    #[test]
    fn test_tcut_outside_range() {
        let values = vec![-5.0, 50.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        let result = tcut_from_vecs(values, breaks, None, 1.0).unwrap();
        assert_eq!(result.codes[0], -1);
        assert_eq!(result.codes[1], -1);
        assert_eq!(result.codes[2], 1);
    }

    #[test]
    fn test_tcut_breakpoint_boundaries_are_left_closed() {
        let values = vec![0.0, 10.0, 20.0, 30.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        let result = tcut_from_vecs(values, breaks, None, 1.0).unwrap();
        assert_eq!(result.codes, vec![0, 1, 2, 2]);
        assert_eq!(result.counts, vec![1, 1, 2]);
    }

    #[test]
    fn test_tcut_duplicate_ordered_breaks_are_allowed() {
        let result = tcut_from_vecs(
            vec![5.0, 15.0, 25.0],
            vec![0.0, 10.0, 10.0, 30.0],
            None,
            1.0,
        )
        .unwrap();

        assert_eq!(result.codes, vec![0, 2, 2]);
        assert_eq!(result.counts, vec![1, 0, 2]);
        assert_eq!(result.breaks, vec![0.0, 10.0, 10.0, 30.0]);
    }

    #[test]
    fn test_tcut_preserves_special_values_and_infinite_breaks() {
        let result = tcut_from_vecs(
            vec![5.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            vec![f64::NEG_INFINITY, 10.0, 20.0, f64::INFINITY],
            None,
            1.0,
        )
        .unwrap();

        assert_eq!(result.codes, vec![0, -1, 2, 0]);
        assert_eq!(result.counts, vec![2, 0, 1]);
    }

    #[test]
    fn test_tcut_scales_values_and_cutpoints_after_classification() {
        let negative = tcut_from_vecs(vec![1.0, 2.0], vec![2.0], None, -1.0).unwrap();
        assert_eq!(negative.values, vec![-1.0, -2.0]);
        assert_eq!(negative.breaks, vec![-0.99, -1.5, -2.01]);
        assert_eq!(negative.codes, vec![0, 1]);
        assert_eq!(negative.counts, vec![1, 1]);

        let zero = tcut_from_vecs(vec![1.0, 2.0], vec![2.0], None, 0.0).unwrap();
        assert_eq!(zero.values, vec![0.0, 0.0]);
        assert_eq!(zero.breaks, vec![0.0, 0.0, 0.0]);
        assert_eq!(zero.codes, vec![0, 1]);
    }

    #[test]
    fn test_tcut_accepts_empty_explicit_breaks_without_labels() {
        let result = tcut_from_vecs(vec![1.0, 2.0], Vec::new(), None, 1.0).unwrap();
        assert_eq!(result.values, vec![1.0, 2.0]);
        assert_eq!(result.codes, vec![-1, -1]);
        assert!(result.levels.is_empty());
        assert!(result.breaks.is_empty());
        assert!(result.counts.is_empty());

        assert!(tcut_from_vecs(vec![1.0], Vec::new(), Some(Vec::new()), 1.0).is_err());
    }

    #[test]
    fn test_tcut_fractional_interval_count_rejects_explicit_labels() {
        assert!(
            tcut_from_vecs(
                vec![1.0, 2.0],
                vec![1.5],
                Some(vec!["a".to_string(), "b".to_string()]),
                1.0,
            )
            .is_err()
        );
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
        assert!(tcut_from_vecs(vec![0.5], vec![0.0, f64::NAN], None, 1.0).is_err());
        assert!(tcut_from_vecs(vec![0.5], vec![2.0, 1.0], None, 1.0).is_err());
        assert!(tcut_from_vecs(vec![f64::NAN], vec![2.0], None, 1.0).is_err());
        assert!(tcut_from_vecs(vec![0.5], vec![0.0], None, 1.0).is_err());
        assert!(tcut_from_vecs(Vec::new(), vec![2.0], None, 1.0).is_err());
    }

    #[test]
    fn test_tcut_expand_rejects_malformed_inputs() {
        assert!(tcut_expand(vec![f64::NAN], vec![1.0], vec![0.0]).is_err());
        assert!(tcut_expand(vec![0.0], vec![f64::INFINITY], vec![0.0]).is_err());
        assert!(tcut_expand(vec![0.0], vec![1.0], vec![]).is_err());
        assert!(tcut_expand(vec![0.0], vec![1.0], vec![0.0, 0.0]).is_err());
    }
}
