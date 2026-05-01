use crate::internal::typed_inputs::{AndersenGillInput, CountingProcessData, Weights};
use pyo3::prelude::*;

struct AgmartData<'a> {
    start: &'a [f64],
    stop: &'a [f64],
    event: &'a [i32],
    score: &'a [f64],
    wt: &'a [f64],
    strata: &'a [i32],
}

fn compute_agmart(method: i32, input: AgmartData) -> Vec<f64> {
    let n = input.start.len();
    let start_slice = input.start;
    let stop_slice = input.stop;
    let event_slice = input.event;
    let score_slice = input.score;
    let wt_slice = input.wt;
    let strata_slice = input.strata;
    let mut resid = vec![0.0; n];
    let nused = n;
    for i in 0..nused {
        resid[i] = event_slice[i] as f64;
    }
    let mut person = 0;
    while person < nused {
        if event_slice[person] == 0 {
            person += 1;
            continue;
        }
        let time = stop_slice[person];
        let mut denom = 0.0;
        let mut e_denom = 0.0;
        let mut deaths = 0;
        let mut wtsum = 0.0;
        let mut k = person;
        while k < nused {
            if start_slice[k] < time {
                denom += score_slice[k] * wt_slice[k];
                if stop_slice[k] == time && event_slice[k] == 1 {
                    deaths += 1;
                    wtsum += wt_slice[k];
                    e_denom += score_slice[k] * wt_slice[k];
                }
            }
            if strata_slice[k] == 1 || k == nused - 1 {
                break;
            }
            k += 1;
        }
        let (hazard, e_hazard) = if deaths == 0 {
            (0.0, 0.0)
        } else {
            let wtsum_normalized = wtsum / deaths as f64;
            let mut hazard_total = 0.0;
            let mut e_hazard_total = 0.0;
            for i in 0..deaths {
                let temp = method as f64 * (i as f64 / deaths as f64);
                let denominator = denom - temp * e_denom;
                hazard_total += wtsum_normalized / denominator;
                e_hazard_total += wtsum_normalized * (1.0 - temp) / denominator;
            }
            (hazard_total, e_hazard_total)
        };
        let initial_person = person;
        let mut k = initial_person;
        while k < nused {
            if start_slice[k] < time {
                if stop_slice[k] == time && event_slice[k] == 1 {
                    resid[k] -= score_slice[k] * e_hazard;
                } else {
                    resid[k] -= score_slice[k] * hazard;
                }
            }
            if stop_slice[k] == time {
                person += 1;
            }
            if strata_slice[k] == 1 || k == nused - 1 {
                break;
            }
            k += 1;
        }
    }
    resid
}

#[allow(clippy::too_many_arguments)]
pub fn agmart(
    n: usize,
    method: i32,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    score: Vec<f64>,
    wt: Vec<f64>,
    strata: Vec<i32>,
) -> PyResult<Vec<f64>> {
    let input = AndersenGillInput::try_new(
        CountingProcessData::try_new(start, stop, event)?,
        score,
        Some(Weights::try_new(wt)?),
        Some(strata),
    )?;
    if input.counting.start.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start length must equal n",
        ));
    }
    agmart_typed(&input, Some(method))
}

#[pyfunction(name = "agmart")]
#[pyo3(signature = (input, method=None))]
pub(crate) fn agmart_typed(input: &AndersenGillInput, method: Option<i32>) -> PyResult<Vec<f64>> {
    let weights = input.weights_or_unit();
    let strata = input.strata_or_default();
    let data = AgmartData {
        start: &input.counting.start,
        stop: &input.counting.stop,
        event: &input.counting.event,
        score: &input.score,
        wt: &weights,
        strata: &strata,
    };
    Ok(compute_agmart(method.unwrap_or(0), data))
}
