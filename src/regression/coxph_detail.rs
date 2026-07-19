use crate::constants::TIME_EPSILON;
use crate::internal::cox_risk::{cox_risk_shift, shifted_weighted_exp_eta_with_shift};
use crate::internal::validation::validate_binary_i32;
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::BTreeMap;

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

fn weighted_sums_into(
    covariates: &[Vec<f64>],
    risk_weights: &[f64],
    indices: &[usize],
    s1: &mut [f64],
    s2: &mut [f64],
) -> f64 {
    s1.fill(0.0);
    s2.fill(0.0);
    let mut s0 = 0.0;
    let nvar = s1.len();
    for &idx in indices {
        let weight = risk_weights[idx];
        s0 += weight;
        for col in 0..nvar {
            let weighted_value = weight * covariates[idx][col];
            s1[col] += weighted_value;
            for inner in 0..nvar {
                s2[col * nvar + inner] += weighted_value * covariates[idx][inner];
            }
        }
    }
    s0
}

#[inline]
fn observation_weight(weights: Option<&[f64]>, idx: usize) -> f64 {
    weights.map_or(1.0, |values| values[idx])
}

fn cox_risk_shift_optional(eta: &[f64], weights: Option<&[f64]>) -> f64 {
    if let Some(weights) = weights {
        return cox_risk_shift(eta, weights);
    }

    let shift = eta
        .iter()
        .copied()
        .filter(|eta_i| eta_i.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    if shift.is_finite() { shift } else { 0.0 }
}

fn shifted_weighted_exp_eta_optional(eta: &[f64], weights: Option<&[f64]>, shift: f64) -> Vec<f64> {
    if let Some(weights) = weights {
        return shifted_weighted_exp_eta_with_shift(eta, weights, shift);
    }

    eta.iter().map(|eta_i| (*eta_i - shift).exp()).collect()
}

fn mean_and_covariance_into(
    s0: f64,
    s1: &[f64],
    s2: &[f64],
    means: &mut [f64],
    covariance: &mut [f64],
) {
    if s0 <= 0.0 {
        means.fill(0.0);
        covariance.fill(0.0);
        return;
    }
    for (mean, value) in means.iter_mut().zip(s1) {
        *mean = value / s0;
    }
    for row in 0..means.len() {
        for col in 0..means.len() {
            covariance[row * means.len() + col] =
                s2[row * means.len() + col] / s0 - means[row] * means[col];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn efron_step_mean_and_covariance_into(
    step_s0: f64,
    s1: &[f64],
    death_s1: &[f64],
    s2: &[f64],
    death_s2: &[f64],
    fraction: f64,
    means: &mut [f64],
    covariance: &mut [f64],
) {
    if step_s0 <= 0.0 {
        means.fill(0.0);
        covariance.fill(0.0);
        return;
    }
    for col in 0..means.len() {
        means[col] = (s1[col] - fraction * death_s1[col]) / step_s0;
    }
    for row in 0..means.len() {
        for col in 0..means.len() {
            let idx = row * means.len() + col;
            covariance[idx] =
                (s2[idx] - fraction * death_s2[idx]) / step_s0 - means[row] * means[col];
        }
    }
}

fn scale_matrix_in_place(target: &mut [f64], scale: f64) {
    for value in target {
        *value *= scale;
    }
}

fn add_matrix(target: &mut [f64], values: &[f64], scale: f64) {
    for (target_value, &value) in target.iter_mut().zip(values.iter()) {
        *target_value += scale * value;
    }
}

fn nested_matrix_from_flat(values: &[f64], nvar: usize) -> Vec<Vec<f64>> {
    if nvar == 0 {
        return Vec::new();
    }
    values.chunks(nvar).map(|row| row.to_vec()).collect()
}

fn event_groups(time: &[f64], status: &[i32], strata: &[i32]) -> Vec<(i32, f64)> {
    let mut groups: Vec<(i32, f64)> = Vec::new();
    let mut times_by_stratum: BTreeMap<i32, Vec<f64>> = BTreeMap::new();

    for idx in 0..time.len() {
        if status[idx] == 1 {
            times_by_stratum
                .entry(strata[idx])
                .or_default()
                .push(time[idx]);
        }
    }

    for (stratum, mut times) in times_by_stratum {
        times.sort_by(|a, b| a.total_cmp(b));
        times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);
        groups.extend(times.into_iter().map(|event_time| (stratum, event_time)));
    }
    groups
}

fn fill_at_risk_indices(
    target: &mut Vec<usize>,
    time: &[f64],
    entry_times: Option<&[f64]>,
    strata: &[i32],
    stratum: i32,
    event_time: f64,
) {
    target.clear();
    target.extend((0..time.len()).filter(|&idx| {
        strata[idx] == stratum
            && time[idx] >= event_time
            && entry_times.is_none_or(|entry| entry[idx] < event_time)
    }));
}

fn fill_event_indices(
    target: &mut Vec<usize>,
    time: &[f64],
    status: &[i32],
    strata: &[i32],
    stratum: i32,
    event_time: f64,
) {
    target.clear();
    target.extend((0..time.len()).filter(|&idx| {
        strata[idx] == stratum && status[idx] == 1 && (time[idx] - event_time).abs() < TIME_EPSILON
    }));
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

    let strata_values = strata
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![0; n]));
    let offsets = offset
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(vec![0.0; n]));

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
    let risk_shift = cox_risk_shift_optional(&linear_predictors, weights);
    let risk_weights = shifted_weighted_exp_eta_optional(&linear_predictors, weights, risk_shift);

    let groups = event_groups(time, status, strata_values.as_ref());
    let mut rows = Vec::with_capacity(groups.len());
    let mut total_events = 0;
    let mut current_stratum = None;
    let mut cumhaz = 0.0_f64;
    let hazard_scale = (center - risk_shift).exp();
    let hazard_scale_squared = (2.0 * (center - risk_shift)).exp();
    let risk_output_scale = risk_shift.exp();
    let mut at_risk = Vec::with_capacity(n);
    let mut deaths = Vec::with_capacity(status.iter().filter(|&&value| value == 1).count());
    let mut event_covariates = vec![0.0; nvar];
    let mut risk_s1 = vec![0.0; nvar];
    let mut risk_s2 = vec![0.0; nvar * nvar];
    let mut death_s1 = vec![0.0; nvar];
    let mut death_s2 = vec![0.0; nvar * nvar];
    let mut means = vec![0.0; nvar];
    let mut score = vec![0.0; nvar];
    let mut imat = vec![0.0; nvar * nvar];
    let mut step_means = vec![0.0; nvar];
    let mut step_covariance = vec![0.0; nvar * nvar];

    for (stratum, event_time) in groups {
        if current_stratum != Some(stratum) {
            current_stratum = Some(stratum);
            cumhaz = 0.0;
        }
        fill_at_risk_indices(
            &mut at_risk,
            time,
            entry_times,
            strata_values.as_ref(),
            stratum,
            event_time,
        );
        fill_event_indices(
            &mut deaths,
            time,
            status,
            strata_values.as_ref(),
            stratum,
            event_time,
        );
        let event_weight = deaths
            .iter()
            .map(|&idx| observation_weight(weights, idx))
            .sum::<f64>();
        let mean_event_weight = event_weight / deaths.len() as f64;
        event_covariates.fill(0.0);
        for &idx in &deaths {
            let weight = observation_weight(weights, idx);
            for col in 0..nvar {
                event_covariates[col] += weight * covariates[idx][col];
            }
        }

        let s0 = weighted_sums_into(
            covariates,
            &risk_weights,
            &at_risk,
            &mut risk_s1,
            &mut risk_s2,
        );
        mean_and_covariance_into(s0, &risk_s1, &risk_s2, &mut means, &mut imat);
        for col in 0..nvar {
            score[col] = event_covariates[col] - event_weight * means[col];
        }
        scale_matrix_in_place(&mut imat, event_weight);
        let mut hazard = if s0 > 0.0 {
            event_weight * hazard_scale / s0
        } else {
            0.0
        };
        let mut varhaz = if s0 > 0.0 {
            event_weight * mean_event_weight * hazard_scale_squared / (s0 * s0)
        } else {
            0.0
        };

        if method == "efron" && deaths.len() > 1 {
            let d0 = weighted_sums_into(
                covariates,
                &risk_weights,
                &deaths,
                &mut death_s1,
                &mut death_s2,
            );
            score.copy_from_slice(&event_covariates);
            imat.fill(0.0);
            means.fill(0.0);
            hazard = 0.0;
            varhaz = 0.0;
            let step_weight = mean_event_weight;
            for step in 0..deaths.len() {
                let fraction = step as f64 / deaths.len() as f64;
                let step_s0 = s0 - fraction * d0;
                efron_step_mean_and_covariance_into(
                    step_s0,
                    &risk_s1,
                    &death_s1,
                    &risk_s2,
                    &death_s2,
                    fraction,
                    &mut step_means,
                    &mut step_covariance,
                );
                for col in 0..nvar {
                    means[col] += step_means[col] / deaths.len() as f64;
                    score[col] -= step_weight * step_means[col];
                }
                add_matrix(&mut imat, &step_covariance, step_weight);
                if step_s0 > 0.0 {
                    hazard += step_weight * hazard_scale / step_s0;
                    varhaz +=
                        step_weight * step_weight * hazard_scale_squared / (step_s0 * step_s0);
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
            n_censor: censor_count(time, status, strata_values.as_ref(), stratum, event_time),
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
            means: means.clone(),
            imat: nested_matrix_from_flat(&imat, nvar),
            score: score.clone(),
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
    validate_binary_i32(&status, "status")?;
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

    fn assert_close(left: f64, right: f64) {
        if left == right {
            return;
        }

        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to equal {right}"
        );
    }

    fn assert_vec_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (&left_value, &right_value) in left.iter().zip(right) {
            assert_close(left_value, right_value);
        }
    }

    fn assert_nested_vec_close(left: &[Vec<f64>], right: &[Vec<f64>]) {
        assert_eq!(left.len(), right.len());
        for (left_row, right_row) in left.iter().zip(right) {
            assert_vec_close(left_row, right_row);
        }
    }

    fn assert_optional_vec_close(left: &Option<Vec<f64>>, right: &Option<Vec<f64>>) {
        match (left, right) {
            (Some(left), Some(right)) => assert_vec_close(left, right),
            (None, None) => {}
            _ => panic!("expected matching optional vectors"),
        }
    }

    fn assert_detail_close(left: &CoxphDetail, right: &CoxphDetail) {
        assert_eq!(left.n_events, right.n_events);
        assert_eq!(left.n_observations, right.n_observations);
        assert_eq!(left.n_covariates, right.n_covariates);
        assert_eq!(left.rows.len(), right.rows.len());

        for (left_row, right_row) in left.rows.iter().zip(&right.rows) {
            assert_eq!(left_row.stratum, right_row.stratum);
            assert_close(left_row.time, right_row.time);
            assert_eq!(left_row.n_risk, right_row.n_risk);
            assert_eq!(left_row.n_event, right_row.n_event);
            assert_eq!(left_row.n_censor, right_row.n_censor);
            assert_close(left_row.hazard, right_row.hazard);
            assert_close(left_row.cumhaz, right_row.cumhaz);
            assert_close(left_row.varhaz, right_row.varhaz);
            assert_close(left_row.wtrisk, right_row.wtrisk);
            assert_close(left_row.n_event_weight, right_row.n_event_weight);
            assert_vec_close(&left_row.score, &right_row.score);
            assert_optional_vec_close(&left_row.schoenfeld, &right_row.schoenfeld);
            assert_vec_close(&left_row.means, &right_row.means);
            assert_nested_vec_close(&left_row.imat, &right_row.imat);
        }
    }

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
    fn test_coxph_detail_unweighted_matches_unit_weights() {
        let time = vec![1.0, 1.0, 2.0, 4.0, 5.0, 5.0];
        let status = vec![1, 1, 0, 1, 0, 1];
        let covariates = vec![
            vec![0.2, 0.4],
            vec![0.8, 0.1],
            vec![0.5, 1.1],
            vec![0.4, 0.3],
            vec![1.1, 0.2],
            vec![0.7, 0.6],
        ];
        let coefficients = vec![0.3, -0.2];

        let unweighted = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: None,
            entry_times: None,
            strata: None,
            offset: None,
            method: "efron",
            center: 0.0,
        })
        .unwrap();
        let unit_weights = vec![1.0; time.len()];
        let unit_weighted = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: Some(&unit_weights),
            entry_times: None,
            strata: None,
            offset: None,
            method: "efron",
            center: 0.0,
        })
        .unwrap();

        assert_detail_close(&unweighted, &unit_weighted);
    }

    #[test]
    fn test_coxph_detail_efron_tie_matches_hand_computed_values() {
        let time = vec![1.0, 1.0, 2.0];
        let status = vec![1, 1, 0];
        let covariates = vec![vec![0.0], vec![1.0], vec![2.0]];
        let coefficients = vec![0.0];

        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: None,
            entry_times: None,
            strata: None,
            offset: None,
            method: "efron",
            center: 0.0,
        })
        .unwrap();

        assert_eq!(detail.rows.len(), 1);
        let row = &detail.rows[0];
        assert_eq!(row.n_risk, 3);
        assert_eq!(row.n_event, 2);
        assert!((row.hazard - 5.0 / 6.0).abs() < 1e-12);
        assert!((row.varhaz - 13.0 / 36.0).abs() < 1e-12);
        assert!((row.score[0] + 1.25).abs() < 1e-12);
        assert!((row.imat[0][0] - 65.0 / 48.0).abs() < 1e-12);
        assert_eq!(row.means, vec![9.0 / 8.0]);
    }

    #[test]
    fn test_coxph_detail_weighted_variance_matches_hand_computed_values() {
        let time = vec![1.0, 1.0, 2.0];
        let status = vec![1, 1, 0];
        let covariates = vec![vec![0.0], vec![1.0], vec![2.0]];
        let coefficients = vec![0.0];
        let weights = vec![1.0, 2.0, 0.5];

        let breslow = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: Some(&weights),
            entry_times: None,
            strata: None,
            offset: None,
            method: "breslow",
            center: 0.0,
        })
        .unwrap();
        let efron = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: Some(&weights),
            entry_times: None,
            strata: None,
            offset: None,
            method: "efron",
            center: 0.0,
        })
        .unwrap();

        assert_eq!(breslow.rows.len(), 1);
        assert_eq!(efron.rows.len(), 1);
        assert!((breslow.rows[0].varhaz - 18.0 / 49.0).abs() < 1e-12);
        assert!((efron.rows[0].varhaz - 585.0 / 784.0).abs() < 1e-12);
        assert_eq!(efron.rows[0].means, vec![13.0 / 14.0]);
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

    #[test]
    fn test_coxph_detail_excludes_zero_weight_rows_from_risk() {
        let time = vec![1.0, 2.0];
        let status = vec![1, 0];
        let covariates = vec![vec![0.0], vec![1000.0]];
        let coefficients = vec![1.0];
        let weights = vec![1.0, 0.0];

        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: Some(&weights),
            entry_times: None,
            strata: None,
            offset: None,
            method: "breslow",
            center: 0.0,
        })
        .unwrap();

        assert_eq!(detail.rows.len(), 1);
        assert_eq!(detail.rows[0].wtrisk, 1.0);
        assert_eq!(detail.rows[0].hazard, 1.0);
        assert!(detail.rows[0].cumhaz.is_finite());
    }

    #[test]
    fn test_coxph_detail_groups_event_times_by_sorted_strata() {
        let time = vec![2.0, 1.0, 1.0 + TIME_EPSILON / 2.0, 3.0];
        let status = vec![1, 1, 1, 0];
        let covariates = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];
        let coefficients = vec![0.0];
        let strata = vec![2, 1, 1, 2];

        let detail = compute_coxph_detail_with_options(CoxphDetailOptions {
            time: &time,
            status: &status,
            covariates: &covariates,
            coefficients: &coefficients,
            weights: None,
            entry_times: None,
            strata: Some(&strata),
            offset: None,
            method: "breslow",
            center: 0.0,
        })
        .unwrap();

        assert_eq!(detail.rows.len(), 2);
        assert_eq!(detail.rows[0].stratum, 1);
        assert!((detail.rows[0].time - 1.0).abs() < TIME_EPSILON);
        assert_eq!(detail.rows[0].n_event, 2);
        assert_eq!(detail.rows[1].stratum, 2);
        assert_eq!(detail.rows[1].time, 2.0);
    }
}
