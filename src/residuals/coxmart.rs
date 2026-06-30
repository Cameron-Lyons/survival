use crate::internal::typed_inputs::{CoxMartInput, SurvivalData, Weights};
use pyo3::prelude::*;

pub(crate) struct CoxMartSurvivalData<'a> {
    pub(crate) time: &'a [f64],
    pub(crate) status: &'a [i32],
    pub(crate) strata: &'a [i32],
}

pub(crate) struct CoxMartWeights<'a> {
    pub(crate) score: &'a [f64],
    pub(crate) wt: &'a [f64],
}

pub fn coxmart(
    time: Vec<f64>,
    status: Vec<i32>,
    score: Vec<f64>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    method: Option<i32>,
) -> PyResult<Vec<f64>> {
    let input = CoxMartInput::try_new(
        SurvivalData::try_new(time, status)?,
        score,
        weights.map(Weights::try_new).transpose()?,
        strata,
    )?;
    coxmart_typed(&input, method)
}

#[pyfunction(name = "coxmart")]
#[pyo3(signature = (input, method=None))]
pub(crate) fn coxmart_typed(input: &CoxMartInput, method: Option<i32>) -> PyResult<Vec<f64>> {
    let n = input.survival.time.len();
    let weights = input.weights_or_unit_cow();
    let strata = input.strata_or_default_cow();
    let method_val = method.unwrap_or(0);
    let mut expect = vec![0.0; n];
    let surv_data = CoxMartSurvivalData {
        time: &input.survival.time,
        status: &input.survival.status,
        strata: strata.as_ref(),
    };
    let weights_data = CoxMartWeights {
        score: &input.score,
        wt: weights.as_ref(),
    };
    compute_coxmart(n, method_val, surv_data, weights_data, &mut expect);
    Ok(expect)
}

#[inline]
fn is_stratum_end(strata: &[i32], index: usize, n: usize) -> bool {
    index + 1 == n || strata[index] == 1
}

pub(crate) fn compute_coxmart(
    n: usize,
    method: i32,
    surv_data: CoxMartSurvivalData,
    weights: CoxMartWeights,
    expect: &mut [f64],
) {
    if n == 0 {
        return;
    }
    let mut denom = 0.0;
    for i in (0..n).rev() {
        if is_stratum_end(surv_data.strata, i, n) {
            denom = 0.0;
        }
        denom += weights.score[i] * weights.wt[i];
        let condition = if i == 0 {
            true
        } else {
            is_stratum_end(surv_data.strata, i - 1, n)
                || (surv_data.time[i - 1] != surv_data.time[i])
        };
        expect[i] = if condition { denom } else { 0.0 };
    }
    let mut deaths = 0;
    let mut wtsum = 0.0;
    let mut e_denom = 0.0;
    let mut hazard = 0.0;
    let mut lastone = 0;
    let mut current_denom = 0.0;
    for i in 0..n {
        if expect[i] != 0.0 {
            current_denom = expect[i];
        }
        expect[i] = surv_data.status[i] as f64;
        deaths += surv_data.status[i];
        wtsum += surv_data.status[i] as f64 * weights.wt[i];
        e_denom += weights.score[i] * surv_data.status[i] as f64 * weights.wt[i];
        let is_last = is_stratum_end(surv_data.strata, i, n)
            || (i < n - 1 && surv_data.time[i + 1] != surv_data.time[i]);
        if is_last {
            if deaths < 2 || method == 0 {
                hazard += wtsum / current_denom;
                for (j, expect_j) in expect.iter_mut().enumerate().take(i + 1).skip(lastone) {
                    *expect_j -= weights.score[j] * hazard;
                }
            } else {
                let mut temp = hazard;
                let deaths_f = deaths as f64;
                wtsum /= deaths_f;
                for j in 0..deaths {
                    let j_f = j as f64;
                    let downwt = j_f / deaths_f;
                    hazard += wtsum / (current_denom - e_denom * downwt);
                    temp += wtsum * (1.0 - downwt) / (current_denom - e_denom * downwt);
                }
                for (j, expect_j) in expect.iter_mut().enumerate().take(i + 1).skip(lastone) {
                    if surv_data.status[j] == 0 {
                        *expect_j = -weights.score[j] * hazard;
                    } else {
                        *expect_j -= weights.score[j] * temp;
                    }
                }
            }
            lastone = i + 1;
            deaths = 0;
            wtsum = 0.0;
            e_denom = 0.0;
        }
        if is_stratum_end(surv_data.strata, i, n) {
            hazard = 0.0;
        }
    }
    for (j, expect_j) in expect.iter_mut().enumerate().take(n).skip(lastone) {
        *expect_j -= weights.score[j] * hazard;
    }
}
