

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct DCalibrationResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: usize,
    #[pyo3(get)]
    pub n_bins: usize,
    #[pyo3(get)]
    pub observed_counts: Vec<usize>,
    #[pyo3(get)]
    pub expected_counts: Vec<f64>,
    #[pyo3(get)]
    pub bin_edges: Vec<f64>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub is_calibrated: bool,
}

#[pymethods]
impl DCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        statistic: f64,
        p_value: f64,
        degrees_of_freedom: usize,
        n_bins: usize,
        observed_counts: Vec<usize>,
        expected_counts: Vec<f64>,
        bin_edges: Vec<f64>,
        n_events: usize,
        is_calibrated: bool,
    ) -> Self {
        Self {
            statistic,
            p_value,
            degrees_of_freedom,
            n_bins,
            observed_counts,
            expected_counts,
            bin_edges,
            n_events,
            is_calibrated,
        }
    }
}

pub(crate) fn d_calibration_core(
    survival_probs: &[f64],
    status: &[i32],
    n_bins: usize,
) -> DCalibrationResult {
    let events: Vec<f64> = survival_probs
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 1)
        .map(|(p, _)| *p)
        .collect();

    let n_events = events.len();

    if n_events < n_bins * 2 {
        return DCalibrationResult {
            statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
            n_bins,
            observed_counts: vec![],
            expected_counts: vec![],
            bin_edges: vec![],
            n_events,
            is_calibrated: true,
        };
    }

    let mut bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
    bin_edges[0] = 0.0;
    bin_edges[n_bins] = 1.0 + 1e-10;

    let mut observed_counts = vec![0usize; n_bins];
    for &p in &events {
        for bin_idx in 0..n_bins {
            if p >= bin_edges[bin_idx] && p < bin_edges[bin_idx + 1] {
                observed_counts[bin_idx] += 1;
                break;
            }
        }
    }

    let expected_per_bin = n_events as f64 / n_bins as f64;
    let expected_counts: Vec<f64> = vec![expected_per_bin; n_bins];

    let mut chi2_stat = 0.0;
    for bin_idx in 0..n_bins {
        let observed = observed_counts[bin_idx] as f64;
        let expected = expected_counts[bin_idx];
        if expected > 0.0 {
            chi2_stat += (observed - expected).powi(2) / expected;
        }
    }

    let df = n_bins - 1;
    let p_value = chi2_sf(chi2_stat, df);

    let is_calibrated = p_value >= 0.05;

    bin_edges.pop();

    DCalibrationResult {
        statistic: chi2_stat,
        p_value,
        degrees_of_freedom: df,
        n_bins,
        observed_counts,
        expected_counts,
        bin_edges,
        n_events,
        is_calibrated,
    }
}

#[pyfunction]
#[pyo3(signature = (survival_probs, status, n_bins=None))]
pub fn d_calibration(
    survival_probs: Vec<f64>,
    status: Vec<i32>,
    n_bins: Option<usize>,
) -> PyResult<DCalibrationResult> {
    if survival_probs.len() != status.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survival_probs and status must have the same length",
        ));
    }

    let n_bins = n_bins.unwrap_or(10);
    if n_bins < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be at least 2",
        ));
    }

    Ok(d_calibration_core(&survival_probs, &status, n_bins))
}
