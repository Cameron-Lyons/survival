pub struct SurvivalData<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub strata: &'a mut [i32],
}
pub struct Weights<'a> {
    pub score: &'a [f64],
    pub wt: &'a [f64],
}
#[pymodule]
#[pyo3(name = "coxmart")]
fn coxmart_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coxmart, &m)?)?;
    Ok(())
}
use pyo3::prelude::*;
#[pyfunction]
pub fn coxmart(
    time: Vec<f64>,
    status: Vec<i32>,
    score: Vec<f64>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    method: Option<i32>,
) -> PyResult<Vec<f64>> {
    let n = time.len();
    if status.len() != n || score.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and score must have the same length",
        ));
    }
    let weights_vec = weights.unwrap_or_else(|| vec![1.0; n]);
    let mut strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    let method_val = method.unwrap_or(0);
    if weights_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must have the same length as time",
        ));
    }
    if strata_vec.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "strata must have the same length as time",
        ));
    }
    let mut expect = vec![0.0; n];
    let surv_data = SurvivalData {
        time: &time,
        status: &status,
        strata: &mut strata_vec,
    };
    let weights_data = Weights {
        score: &score,
        wt: &weights_vec,
    };
    compute_coxmart(n, method_val, surv_data, weights_data, &mut expect);
    Ok(expect)
}
pub fn compute_coxmart(
    n: usize,
    method: i32,
    surv_data: SurvivalData,
    weights: Weights,
    expect: &mut [f64],
) {
    if n == 0 {
        return;
    }
    surv_data.strata[n - 1] = 1;
    let mut denom = 0.0;
    for i in (0..n).rev() {
        if surv_data.strata[i] == 1 {
            denom = 0.0;
        }
        denom += weights.score[i] * weights.wt[i];
        let condition = if i == 0 {
            true
        } else {
            surv_data.strata[i - 1] == 1 || (surv_data.time[i - 1] != surv_data.time[i])
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
        let is_last =
            surv_data.strata[i] == 1 || (i < n - 1 && surv_data.time[i + 1] != surv_data.time[i]);
        if is_last {
            if deaths < 2 || method == 0 {
                hazard += wtsum / current_denom;
                #[allow(clippy::needless_range_loop)]
                for j in lastone..=i {
                    expect[j] -= weights.score[j] * hazard;
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
                #[allow(clippy::needless_range_loop)]
                for j in lastone..=i {
                    if surv_data.status[j] == 0 {
                        expect[j] = -weights.score[j] * hazard;
                    } else {
                        expect[j] -= weights.score[j] * temp;
                    }
                }
            }
            lastone = i + 1;
            deaths = 0;
            wtsum = 0.0;
            e_denom = 0.0;
        }
        if surv_data.strata[i] == 1 {
            hazard = 0.0;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for j in lastone..n {
        expect[j] -= weights.score[j] * hazard;
    }
}
