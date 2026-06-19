use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvDiffResult {
    #[pyo3(get)]
    pub observed: Vec<f64>,
    #[pyo3(get)]
    pub expected: Vec<f64>,
    #[pyo3(get)]
    pub variance: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub chi_squared: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: usize,
}

#[pyfunction]
pub fn compute_logrank_components(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    if n > i32::MAX as usize {
        return Err(value_error("time length exceeds i32 calculation capacity"));
    }
    validate_length(n, status.len(), "status")?;
    validate_length(n, group.len(), "group")?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    validate_binary_i32(&status, "status")?;
    validate_group_codes(&group, n)?;
    let strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    validate_length(n, strata_vec.len(), "strata")?;
    validate_strata_markers(&strata_vec)?;
    let rho_val = rho.unwrap_or(0.0);
    if !rho_val.is_finite() {
        return Err(value_error("rho must be finite"));
    }
    let max_group = group.iter().max().copied().unwrap_or(0);
    let ngroup = if max_group > 0 { max_group as usize } else { 1 };
    let nstrat = strata_vec
        .iter()
        .filter(|&&marker| marker == 1)
        .count()
        .max(1);
    let mut obs = vec![0.0; ngroup * nstrat];
    let mut exp = vec![0.0; ngroup * nstrat];
    let mut var = vec![0.0; ngroup * ngroup * nstrat];
    let mut risk = vec![0.0; ngroup];
    let mut kaplan = vec![0.0; n];
    let params = SurvDiffParams {
        nn: n as i32,
        nngroup: ngroup as i32,
        _nstrat: nstrat as i32,
        rho: rho_val,
    };
    let input = SurvDiffInput {
        time: &time,
        status: &status,
        group: &group,
        strata: &strata_vec,
    };
    let mut output = SurvDiffOutput {
        obs: &mut obs,
        exp: &mut exp,
        var: &mut var,
        risk: &mut risk,
        kaplan: &mut kaplan,
    };
    compute_survdiff(params, input, &mut output);
    let mut observed_by_group = vec![0.0; ngroup];
    let mut expected_by_group = vec![0.0; ngroup];
    for stratum_idx in 0..nstrat {
        let offset = stratum_idx * ngroup;
        for group_idx in 0..ngroup {
            observed_by_group[group_idx] += obs[offset + group_idx];
            expected_by_group[group_idx] += exp[offset + group_idx];
        }
    }

    let mut chi_sq = 0.0;
    let mut df = 0usize;
    for (obs_val, exp_val) in observed_by_group.iter().zip(expected_by_group.iter()) {
        let diff = obs_val - exp_val;
        if *exp_val > 0.0 {
            chi_sq += diff * diff / exp_val;
            df += 1;
        }
    }
    df = df.saturating_sub(1);
    let mut variance_matrix = Vec::new();
    for i in 0..ngroup {
        let start = i * ngroup;
        let end = start + ngroup;
        variance_matrix.push(var[start..end].to_vec());
    }
    Ok(SurvDiffResult {
        observed: observed_by_group,
        expected: expected_by_group,
        variance: variance_matrix,
        chi_squared: chi_sq,
        degrees_of_freedom: df,
    })
}

fn validate_group_codes(values: &[i32], n: usize) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 1 {
            return Err(value_error(format!(
                "group must be >= 1; got {value} at index {idx}"
            )));
        }
        if value as usize > n {
            return Err(value_error(format!(
                "group values must be between 1 and the number of observations ({n}); got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_strata_markers(values: &[i32]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "strata must be >= 0; got {value} at index {idx}"
            )));
        }
        if value > 1 {
            return Err(value_error(format!(
                "strata values must be 0 or 1; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

/// Backward-compatible alias retained for the public crate and Python APIs.
#[pyfunction]
pub fn survdiff2(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
) -> PyResult<SurvDiffResult> {
    compute_logrank_components(time, status, group, strata, rho)
}

pub(crate) struct SurvDiffInput<'a> {
    pub(crate) time: &'a [f64],
    pub(crate) status: &'a [i32],
    pub(crate) group: &'a [i32],
    pub(crate) strata: &'a [i32],
}

pub(crate) struct SurvDiffOutput<'a> {
    pub(crate) obs: &'a mut [f64],
    pub(crate) exp: &'a mut [f64],
    pub(crate) var: &'a mut [f64],
    pub(crate) risk: &'a mut [f64],
    pub(crate) kaplan: &'a mut [f64],
}

pub(crate) struct SurvDiffParams {
    pub(crate) nn: i32,
    pub(crate) nngroup: i32,
    pub(crate) _nstrat: i32,
    pub(crate) rho: f64,
}

pub(crate) fn compute_survdiff(
    params: SurvDiffParams,
    input: SurvDiffInput,
    output: &mut SurvDiffOutput,
) {
    let ntotal = params.nn as usize;
    let ngroup = params.nngroup as usize;
    let mut istart = 0;
    let mut koff = 0;
    for v in output.var.iter_mut() {
        *v = 0.0;
    }
    for o in output.obs.iter_mut() {
        *o = 0.0;
    }
    for e in output.exp.iter_mut() {
        *e = 0.0;
    }
    while istart < ntotal {
        let mut n = istart;
        while n < ntotal && input.strata[n] != 1 {
            n += 1;
        }
        if n < ntotal {
            n += 1;
        }
        if params.rho != 0.0 {
            let mut km = 1.0;
            let mut i = istart;
            while i < n {
                let current_time = input.time[i];
                let mut deaths = 0;
                let mut j = i;
                while j < n && input.time[j] == current_time {
                    output.kaplan[j] = km;
                    deaths += input.status[j] as usize;
                    j += 1;
                }
                let nrisk = (n - i) as f64;
                if nrisk > 0.0 && deaths > 0 {
                    km *= (nrisk - deaths as f64) / nrisk;
                }
                i = j;
            }
        }
        let mut i = n.saturating_sub(1);
        loop {
            if i < istart || (istart == 0 && n == 0) {
                break;
            }
            let current_time = input.time[i];
            let mut deaths = 0;
            let mut j = i;
            let wt = if params.rho == 0.0 {
                1.0
            } else {
                output.kaplan[i].powf(params.rho)
            };
            for r in output.risk.iter_mut().take(ngroup) {
                *r = 0.0;
            }
            loop {
                let k = (input.group[j] - 1) as usize;
                output.risk[k] += 1.0;
                deaths += input.status[j] as usize;
                if j == istart {
                    break;
                }
                if input.time[j - 1] != current_time {
                    break;
                }
                j -= 1;
            }
            let nrisk = (n - j) as f64;
            if deaths > 0 {
                for (k, risk_val) in output.risk.iter().take(ngroup).enumerate() {
                    let exp_index = koff + k;
                    output.exp[exp_index] += wt * (deaths as f64) * risk_val / nrisk;
                }
                for ti in j..=i {
                    if input.status[ti] == 1 {
                        let obs_index = koff + (input.group[ti] - 1) as usize;
                        output.obs[obs_index] += wt;
                    }
                }
                if nrisk > 1.0 {
                    let wt_sq = wt * wt;
                    let factor =
                        wt_sq * (deaths as f64) * (nrisk - deaths as f64) / (nrisk * (nrisk - 1.0));
                    for (j_group, &rj) in output.risk.iter().take(ngroup).enumerate() {
                        let var_start = j_group * ngroup;
                        let tmp = factor * rj;
                        for (k_group, &rk) in output.risk.iter().take(ngroup).enumerate() {
                            output.var[var_start + k_group] += tmp
                                * (if j_group == k_group {
                                    rj - rk / nrisk
                                } else {
                                    -rk / nrisk
                                });
                        }
                    }
                }
            }
            if j == istart {
                break;
            }
            i = j - 1;
        }
        istart = n;
        koff += ngroup;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_logrank_components_all_censored_has_zero_degrees_of_freedom() {
        let result = compute_logrank_components(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 0, 0],
            vec![1, 1, 2, 2],
            None,
            None,
        )
        .expect("all-censored logrank components should not underflow df");

        assert_eq!(result.chi_squared, 0.0);
        assert_eq!(result.degrees_of_freedom, 0);
    }

    #[test]
    fn compute_logrank_components_rejects_non_binary_status() {
        let err = compute_logrank_components(vec![1.0], vec![2], vec![1], None, None)
            .expect_err("non-binary status should fail");

        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn compute_logrank_components_rejects_sparse_huge_group_codes() {
        let err = compute_logrank_components(vec![1.0, 2.0], vec![1, 0], vec![1, 3], None, None)
            .expect_err("group code beyond n should fail");

        assert!(err.to_string().contains("group values must be between 1"));
    }

    #[test]
    fn compute_logrank_components_rejects_non_marker_strata() {
        let err = compute_logrank_components(vec![1.0], vec![1], vec![1], Some(vec![2]), None)
            .expect_err("strata markers should be 0 or 1");

        assert!(err.to_string().contains("strata values must be 0 or 1"));
    }

    #[test]
    fn compute_logrank_components_aggregates_strata() {
        let stratified = compute_logrank_components(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 0, 1],
            vec![1, 2, 1, 2],
            Some(vec![0, 1, 0, 1]),
            None,
        )
        .expect("stratified components should compute");
        let first_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![1, 0], vec![1, 2], None, None)
                .expect("first stratum should compute");
        let second_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![0, 1], vec![1, 2], None, None)
                .expect("second stratum should compute");

        assert_eq!(stratified.observed.len(), 2);
        assert_eq!(stratified.expected.len(), 2);
        assert!(
            (stratified.observed[0] - first_stratum.observed[0] - second_stratum.observed[0]).abs()
                < 1e-12
        );
        assert!(
            (stratified.observed[1] - first_stratum.observed[1] - second_stratum.observed[1]).abs()
                < 1e-12
        );
        assert!(
            (stratified.expected[0] - first_stratum.expected[0] - second_stratum.expected[0]).abs()
                < 1e-12
        );
        assert!(
            (stratified.expected[1] - first_stratum.expected[1] - second_stratum.expected[1]).abs()
                < 1e-12
        );
    }
}
