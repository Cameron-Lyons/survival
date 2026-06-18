use crate::constants::TIME_EPSILON;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxphDetailRow {
    #[pyo3(get)]
    pub stratum: i32,
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub n_risk: usize,
    #[pyo3(get)]
    pub n_event: usize,
    #[pyo3(get)]
    pub n_censor: usize,
    #[pyo3(get)]
    pub hazard: f64,
    #[pyo3(get)]
    pub cumhaz: f64,
    #[pyo3(get)]
    pub varhaz: f64,
    #[pyo3(get)]
    pub wtrisk: f64,
    #[pyo3(get)]
    pub n_event_weight: f64,
    #[pyo3(get)]
    pub score: Vec<f64>,
    #[pyo3(get)]
    pub schoenfeld: Option<Vec<f64>>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub imat: Vec<Vec<f64>>,
}

#[pymethods]
impl CoxphDetailRow {
    fn __repr__(&self) -> String {
        format!(
            "CoxphDetailRow(time={:.4}, n_risk={}, n_event={}, hazard={:.6})",
            self.time, self.n_risk, self.n_event, self.hazard
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxphDetail {
    #[pyo3(get)]
    pub rows: Vec<CoxphDetailRow>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_observations: usize,
    #[pyo3(get)]
    pub n_covariates: usize,
}

pub(crate) struct CoxphDetailOptions<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub covariates: &'a [Vec<f64>],
    pub coefficients: &'a [f64],
    pub weights: Option<&'a [f64]>,
    pub entry_times: Option<&'a [f64]>,
    pub strata: Option<&'a [i32]>,
    pub offset: Option<&'a [f64]>,
    pub method: &'a str,
    pub center: f64,
}

#[pymethods]
impl CoxphDetail {
    fn __repr__(&self) -> String {
        format!(
            "CoxphDetail(n_events={}, n_obs={}, n_times={})",
            self.n_events,
            self.n_observations,
            self.rows.len()
        )
    }

    pub fn times(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.time).collect()
    }

    pub fn hazards(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.hazard).collect()
    }

    pub fn cumulative_hazards(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.cumhaz).collect()
    }

    pub fn n_risk_at_times(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.n_risk).collect()
    }

    pub fn scores(&self) -> Vec<Vec<f64>> {
        self.rows.iter().map(|r| r.score.clone()).collect()
    }

    pub fn means(&self) -> Vec<Vec<f64>> {
        self.rows.iter().map(|r| r.means.clone()).collect()
    }

    pub fn information_matrices(&self) -> Vec<Vec<Vec<f64>>> {
        self.rows.iter().map(|r| r.imat.clone()).collect()
    }

    pub fn variance_hazards(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.varhaz).collect()
    }

    pub fn weighted_risk(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.wtrisk).collect()
    }

    pub fn schoenfeld_residuals(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .filter_map(|r| r.schoenfeld.clone())
            .collect()
    }
}

fn weighted_sums(
    covariates: &[Vec<f64>],
    risk_weights: &[f64],
    indices: &[usize],
    nvar: usize,
) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let mut s0 = 0.0;
    let mut s1 = vec![0.0; nvar];
    let mut s2 = vec![vec![0.0; nvar]; nvar];
    for &idx in indices {
        let weight = risk_weights[idx];
        s0 += weight;
        for col in 0..nvar {
            let weighted_value = weight * covariates[idx][col];
            s1[col] += weighted_value;
            for inner in 0..nvar {
                s2[col][inner] += weighted_value * covariates[idx][inner];
            }
        }
    }
    (s0, s1, s2)
}

fn mean_and_covariance(
    s0: f64,
    s1: &[f64],
    s2: &[Vec<f64>],
    nvar: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    if s0 <= 0.0 {
        return (vec![0.0; nvar], vec![vec![0.0; nvar]; nvar]);
    }
    let means: Vec<f64> = s1.iter().map(|value| value / s0).collect();
    let mut covariance = vec![vec![0.0; nvar]; nvar];
    for row in 0..nvar {
        for col in 0..nvar {
            covariance[row][col] = s2[row][col] / s0 - means[row] * means[col];
        }
    }
    (means, covariance)
}

fn add_matrix(target: &mut [Vec<f64>], values: &[Vec<f64>], scale: f64) {
    for (row_idx, row) in values.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate() {
            target[row_idx][col_idx] += scale * value;
        }
    }
}

fn event_groups(time: &[f64], status: &[i32], strata: &[i32]) -> Vec<(i32, f64)> {
    let mut groups: Vec<(i32, f64)> = Vec::new();
    let mut strata_values = strata.to_vec();
    strata_values.sort_unstable();
    strata_values.dedup();

    for stratum in strata_values {
        let mut times: Vec<f64> = time
            .iter()
            .zip(status.iter())
            .zip(strata.iter())
            .filter_map(|((&event_time, &event), &row_stratum)| {
                (event == 1 && row_stratum == stratum).then_some(event_time)
            })
            .collect();
        times.sort_by(|a, b| a.total_cmp(b));
        times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);
        groups.extend(times.into_iter().map(|event_time| (stratum, event_time)));
    }
    groups
}

fn at_risk_indices(
    time: &[f64],
    entry_times: Option<&[f64]>,
    strata: &[i32],
    stratum: i32,
    event_time: f64,
) -> Vec<usize> {
    (0..time.len())
        .filter(|&idx| {
            strata[idx] == stratum
                && time[idx] >= event_time
                && entry_times.is_none_or(|entry| entry[idx] < event_time)
        })
        .collect()
}

fn event_indices(
    time: &[f64],
    status: &[i32],
    strata: &[i32],
    stratum: i32,
    event_time: f64,
) -> Vec<usize> {
    (0..time.len())
        .filter(|&idx| {
            strata[idx] == stratum
                && status[idx] == 1
                && (time[idx] - event_time).abs() < TIME_EPSILON
        })
        .collect()
}

fn censor_count(
    time: &[f64],
    status: &[i32],
    strata: &[i32],
    stratum: i32,
    event_time: f64,
) -> usize {
    (0..time.len())
        .filter(|&idx| {
            strata[idx] == stratum
                && status[idx] == 0
                && (time[idx] - event_time).abs() < TIME_EPSILON
        })
        .count()
}

pub(crate) fn compute_coxph_detail_with_options(
    options: CoxphDetailOptions<'_>,
) -> PyResult<CoxphDetail> {
    let CoxphDetailOptions {
        time,
        status,
        covariates,
        coefficients,
        weights,
        entry_times,
        strata,
        offset,
        method,
        center,
    } = options;
    let n = time.len();
    let nvar = coefficients.len();

    if n == 0 {
        return Ok(CoxphDetail {
            rows: vec![],
            n_events: 0,
            n_observations: 0,
            n_covariates: nvar,
        });
    }

    let method = method.to_ascii_lowercase().replace('_', "-");
    if method != "breslow" && method != "efron" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "detailed output is not available for the {} method",
            method
        )));
    }

    let wts: Vec<f64> = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; n]);
    let strata_values: Vec<i32> = strata.map(|s| s.to_vec()).unwrap_or_else(|| vec![0; n]);
    let offsets: Vec<f64> = offset.map(|o| o.to_vec()).unwrap_or_else(|| vec![0.0; n]);

    let linear_predictors: Vec<f64> = covariates
        .iter()
        .enumerate()
        .map(|(idx, cov)| {
            let mut lp = offsets[idx];
            for j in 0..nvar {
                lp += cov[j] * coefficients[j];
            }
            lp
        })
        .collect();
    let risk_shift = linear_predictors
        .iter()
        .zip(wts.iter())
        .filter_map(|(&lp, &weight)| (weight > 0.0).then_some(lp))
        .fold(f64::NEG_INFINITY, f64::max);
    let risk_shift = if risk_shift.is_finite() {
        risk_shift
    } else {
        0.0
    };

    let risk_weights: Vec<f64> = linear_predictors
        .iter()
        .zip(wts.iter())
        .map(|(&lp, &weight)| {
            if weight == 0.0 {
                0.0
            } else {
                (lp - risk_shift).exp() * weight
            }
        })
        .collect();

    let groups = event_groups(time, status, &strata_values);
    let mut rows = Vec::with_capacity(groups.len());
    let mut total_events = 0;
    let mut current_stratum = None;
    let mut cumhaz = 0.0_f64;
    let hazard_scale = (center - risk_shift).exp();
    let hazard_scale_squared = (2.0 * (center - risk_shift)).exp();
    let risk_output_scale = risk_shift.exp();

    for (stratum, event_time) in groups {
        if current_stratum != Some(stratum) {
            current_stratum = Some(stratum);
            cumhaz = 0.0;
        }
        let at_risk = at_risk_indices(time, entry_times, &strata_values, stratum, event_time);
        let deaths = event_indices(time, status, &strata_values, stratum, event_time);
        let event_weight = deaths.iter().map(|&idx| wts[idx]).sum::<f64>();
        let mut event_covariates = vec![0.0; nvar];
        for &idx in &deaths {
            for col in 0..nvar {
                event_covariates[col] += wts[idx] * covariates[idx][col];
            }
        }

        let (s0, s1, s2) = weighted_sums(covariates, &risk_weights, &at_risk, nvar);
        let (means, covariance) = mean_and_covariance(s0, &s1, &s2, nvar);
        let mut score = event_covariates
            .iter()
            .zip(means.iter())
            .map(|(&event_sum, &mean)| event_sum - event_weight * mean)
            .collect::<Vec<_>>();
        let mut imat = covariance
            .iter()
            .map(|row| row.iter().map(|value| event_weight * value).collect())
            .collect::<Vec<Vec<f64>>>();
        let mut hazard = if s0 > 0.0 {
            event_weight * hazard_scale / s0
        } else {
            0.0
        };
        let mut varhaz = if s0 > 0.0 {
            event_weight * hazard_scale_squared / (s0 * s0)
        } else {
            0.0
        };

        if method == "efron" && deaths.len() > 1 {
            let (d0, d1, d2) = weighted_sums(covariates, &risk_weights, &deaths, nvar);
            score = event_covariates.clone();
            imat = vec![vec![0.0; nvar]; nvar];
            hazard = 0.0;
            varhaz = 0.0;
            let step_weight = event_weight / deaths.len() as f64;
            for step in 0..deaths.len() {
                let fraction = step as f64 / deaths.len() as f64;
                let step_s0 = s0 - fraction * d0;
                let step_s1 = s1
                    .iter()
                    .zip(d1.iter())
                    .map(|(&value, &death_value)| value - fraction * death_value)
                    .collect::<Vec<_>>();
                let step_s2 = s2
                    .iter()
                    .zip(d2.iter())
                    .map(|(row, death_row)| {
                        row.iter()
                            .zip(death_row.iter())
                            .map(|(&value, &death_value)| value - fraction * death_value)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let (step_means, step_covariance) =
                    mean_and_covariance(step_s0, &step_s1, &step_s2, nvar);
                for col in 0..nvar {
                    score[col] -= step_weight * step_means[col];
                }
                add_matrix(&mut imat, &step_covariance, step_weight);
                if step_s0 > 0.0 {
                    hazard += step_weight * hazard_scale / step_s0;
                    varhaz += step_weight * hazard_scale_squared / (step_s0 * step_s0);
                }
            }
        }

        cumhaz += hazard;
        total_events += deaths.len();
        rows.push(CoxphDetailRow {
            stratum,
            time: event_time,
            n_risk: at_risk.len(),
            n_event: deaths.len(),
            n_censor: censor_count(time, status, &strata_values, stratum, event_time),
            hazard,
            cumhaz,
            varhaz,
            wtrisk: if s0 > 0.0 {
                s0 * risk_output_scale
            } else {
                0.0
            },
            n_event_weight: event_weight,
            schoenfeld: Some(score.clone()),
            means,
            imat,
            score,
        });
    }

    Ok(CoxphDetail {
        rows,
        n_events: total_events,
        n_observations: n,
        n_covariates: nvar,
    })
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    covariates,
    coefficients,
    weights=None,
    entry_times=None,
    strata=None,
    offset=None,
    method="breslow".to_string(),
    center=0.0
))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_detail(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    weights: Option<Vec<f64>>,
    entry_times: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    offset: Option<Vec<f64>>,
    method: String,
    center: f64,
) -> PyResult<CoxphDetail> {
    let n = time.len();
    if status.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and covariates must have the same length",
        ));
    }
    if let Some(values) = weights.as_ref()
        && values.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must have the same length as time",
        ));
    }
    if let Some(values) = entry_times.as_ref()
        && values.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "entry_times must have the same length as time",
        ));
    }
    if let Some(values) = strata.as_ref()
        && values.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "strata must have the same length as time",
        ));
    }
    if let Some(values) = offset.as_ref()
        && values.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "offset must have the same length as time",
        ));
    }
    for (idx, value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time contains non-finite value at index {}",
                idx
            )));
        }
    }
    for (idx, value) in status.iter().enumerate() {
        if *value != 0 && *value != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "status must contain only 0/1 values; got {} at index {}",
                value, idx
            )));
        }
    }
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != coefficients.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "covariate row {} has {} columns but coefficients has {}",
                row_idx,
                row.len(),
                coefficients.len()
            )));
        }
        for (col_idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariates contains non-finite value at row {}, column {}",
                    row_idx, col_idx
                )));
            }
        }
    }
    for (idx, value) in coefficients.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "coefficients contains non-finite value at index {}",
                idx
            )));
        }
    }
    if let Some(values) = weights.as_ref() {
        for (idx, value) in values.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights must contain non-negative finite values; got {} at index {}",
                    value, idx
                )));
            }
        }
    }
    if let Some(values) = entry_times.as_ref() {
        for (idx, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "entry_times contains non-finite value at index {}",
                    idx
                )));
            }
            if *value >= time[idx] {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "entry_times[{}] must be less than time[{}]",
                    idx, idx
                )));
            }
        }
    }
    if let Some(values) = offset.as_ref() {
        for (idx, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "offset contains non-finite value at index {}",
                    idx
                )));
            }
        }
    }
    if !center.is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "center must be finite",
        ));
    }

    compute_coxph_detail_with_options(CoxphDetailOptions {
        time: &time,
        status: &status,
        covariates: &covariates,
        coefficients: &coefficients,
        weights: weights.as_deref(),
        entry_times: entry_times.as_deref(),
        strata: strata.as_deref(),
        offset: offset.as_deref(),
        method: &method,
        center,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coxph_detail() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let covariates = vec![vec![1.0], vec![2.0], vec![1.5], vec![2.5], vec![3.0]];
        let coefficients = vec![0.5];

        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: None,
            entry_times: None,
            strata: None,
            offset: None,
            method: "breslow",
            center: 0.0,
        })
        .unwrap();

        assert_eq!(detail.n_events, 3);
        assert_eq!(detail.n_observations, 5);
        assert_eq!(detail.rows.len(), 3);
        assert!(detail.rows[0].wtrisk > 0.0);
        assert!(detail.rows[0].varhaz > 0.0);
        assert_ne!(detail.rows[0].score, vec![0.0]);
        assert_ne!(detail.rows[0].imat, vec![vec![0.0]]);
    }

    #[test]
    fn test_coxph_detail_uses_shifted_risk_scores_for_large_linear_predictors() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let covariates = vec![vec![1.0], vec![709.0 / 710.0], vec![708.0 / 710.0]];
        let coefficients = vec![710.0];

        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: None,
            entry_times: None,
            strata: None,
            offset: None,
            method: "breslow",
            center: 0.0,
        })
        .unwrap();

        let expected_first = (-710.0_f64).exp() / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());
        assert_eq!(detail.rows.len(), 3);
        assert!(detail.rows[0].hazard.is_finite());
        assert!(detail.rows[0].hazard > 0.0);
        assert!((detail.rows[0].hazard - expected_first).abs() <= expected_first * 1e-12);
        assert_eq!(detail.rows[0].cumhaz, detail.rows[0].hazard);
        assert!(detail.rows[1].cumhaz > detail.rows[0].cumhaz);
        assert!(detail.rows[2].cumhaz > detail.rows[1].cumhaz);
    }
}
