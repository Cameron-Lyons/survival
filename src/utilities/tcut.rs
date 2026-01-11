use pyo3::prelude::*;

/// Result of time cutting for person-years calculations
#[derive(Debug, Clone)]
#[pyclass]
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
    value: Vec<f64>,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
) -> PyResult<TcutResult> {
    if breaks.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "breaks must have at least 2 elements",
        ));
    }

    let mut sorted_breaks = breaks.clone();
    sorted_breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_intervals = sorted_breaks.len() - 1;

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
        None => (0..n_intervals)
            .map(|i| {
                if i == n_intervals - 1 {
                    format!("[{}, {}]", sorted_breaks[i], sorted_breaks[i + 1])
                } else {
                    format!("[{}, {})", sorted_breaks[i], sorted_breaks[i + 1])
                }
            })
            .collect(),
    };

    let mut codes = Vec::with_capacity(value.len());
    let mut counts = vec![0usize; n_intervals];

    for &v in &value {
        let code = find_interval(&sorted_breaks, v);
        codes.push(code);
        if code >= 0 {
            counts[code as usize] += 1;
        }
    }

    Ok(TcutResult {
        codes,
        levels: interval_labels,
        breaks: sorted_breaks,
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

    if value < breaks[0] {
        return -1;
    }

    if value > breaks[n - 1] {
        return -1;
    }

    for i in 0..(n - 1) {
        if i == n - 2 {
            if value >= breaks[i] && value <= breaks[i + 1] {
                return i as i32;
            }
        } else if value >= breaks[i] && value < breaks[i + 1] {
            return i as i32;
        }
    }

    -1
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
#[allow(clippy::type_complexity)]
pub fn tcut_expand(
    start: Vec<f64>,
    stop: Vec<f64>,
    cuts: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>, Vec<usize>)> {
    let n = start.len();
    if stop.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start and stop must have same length",
        ));
    }

    if cuts.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "cuts cannot be empty",
        ));
    }

    let mut sorted_cuts = cuts.clone();
    sorted_cuts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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

        let mut split_points = vec![t1];
        for &c in &sorted_cuts {
            if c > t1 && c < t2 {
                split_points.push(c);
            }
        }
        split_points.push(t2);

        for j in 0..(split_points.len() - 1) {
            let s = split_points[j];
            let e = split_points[j + 1];

            new_start.push(s);
            new_stop.push(e);

            let midpoint = (s + e) / 2.0;
            let mut code = -1i32;
            for (k, window) in sorted_cuts.windows(2).enumerate() {
                if midpoint >= window[0] && midpoint < window[1] {
                    code = k as i32;
                    break;
                }
            }
            if code == -1 && !sorted_cuts.is_empty() {
                if midpoint >= *sorted_cuts.last().unwrap() {
                    code = (sorted_cuts.len() - 1) as i32;
                } else if midpoint < sorted_cuts[0] {
                    code = -1;
                }
            }

            interval_codes.push(code);
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

        let result = tcut(values, breaks, None).unwrap();
        assert_eq!(result.codes, vec![0, 1, 2, 3]);
        assert_eq!(result.levels.len(), 4);
    }

    #[test]
    fn test_tcut_with_labels() {
        let values = vec![5.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0];
        let labels = vec!["young".to_string(), "old".to_string()];

        let result = tcut(values, breaks, Some(labels)).unwrap();
        assert_eq!(result.levels, vec!["young", "old"]);
    }

    #[test]
    fn test_tcut_outside_range() {
        let values = vec![-5.0, 50.0, 15.0];
        let breaks = vec![0.0, 10.0, 20.0, 30.0];

        let result = tcut(values, breaks, None).unwrap();
        assert_eq!(result.codes[0], -1); // Below range
        assert_eq!(result.codes[1], -1); // Above range
        assert_eq!(result.codes[2], 1); // In range
    }

    #[test]
    fn test_tcut_expand_basic() {
        let start = vec![0.0, 5.0];
        let stop = vec![25.0, 15.0];
        let cuts = vec![0.0, 10.0, 20.0, 30.0];

        let (new_start, new_stop, codes, indices) = tcut_expand(start, stop, cuts).unwrap();

        // First interval [0, 25) should be split at 10 and 20
        // Second interval [5, 15) should be split at 10
        assert!(new_start.len() > 2);
        assert_eq!(new_start.len(), new_stop.len());
        assert_eq!(new_start.len(), codes.len());
        assert_eq!(new_start.len(), indices.len());
    }
}
