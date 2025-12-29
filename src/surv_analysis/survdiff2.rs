use pyo3::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
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
pub fn survdiff2(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    if status.len() != n || group.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and group must have the same length",
        ));
    }
    let strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    if strata_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "strata must have the same length as time",
        ));
    }
    let rho_val = rho.unwrap_or(0.0);
    let max_group = group.iter().max().copied().unwrap_or(0);
    let ngroup = if max_group > 0 { max_group as usize } else { 1 };
    let nstrat = if strata_vec.is_empty() {
        1
    } else {
        strata_vec.iter().max().copied().unwrap_or(0) as usize + 1
    };
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
    let mut chi_sq = 0.0;
    let mut df = 0;
    for (obs_val, exp_val) in obs.iter().zip(exp.iter()).take(ngroup) {
        let diff = obs_val - exp_val;
        if *exp_val > 0.0 {
            chi_sq += diff * diff / exp_val;
            df += 1;
        }
    }
    df = (df - 1).max(0);
    let mut variance_matrix = Vec::new();
    for i in 0..ngroup {
        let start = i * ngroup;
        let end = start + ngroup;
        variance_matrix.push(var[start..end].to_vec());
    }
    Ok(SurvDiffResult {
        observed: obs[..ngroup].to_vec(),
        expected: exp[..ngroup].to_vec(),
        variance: variance_matrix,
        chi_squared: chi_sq,
        degrees_of_freedom: df,
    })
}
pub struct SurvDiffInput<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub group: &'a [i32],
    pub strata: &'a [i32],
}
pub struct SurvDiffOutput<'a> {
    pub obs: &'a mut [f64],
    pub exp: &'a mut [f64],
    pub var: &'a mut [f64],
    pub risk: &'a mut [f64],
    pub kaplan: &'a mut [f64],
}
pub struct SurvDiffParams {
    pub nn: i32,
    pub nngroup: i32,
    pub _nstrat: i32,
    pub rho: f64,
}
pub fn compute_survdiff(params: SurvDiffParams, input: SurvDiffInput, output: &mut SurvDiffOutput) {
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
        let _n_in_stratum = n - istart;
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
#[pymodule]
#[pyo3(name = "survdiff2")]
fn survdiff2_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survdiff2, &m)?)?;
    m.add_class::<SurvDiffResult>()?;
    Ok(())
}
