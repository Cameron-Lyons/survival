use crate::utilities::validation::{validate_length, validate_non_empty};
use pyo3::prelude::*;

// agmart computes martingale residuals using a counting process formulation.
// This is the standard version suitable for data that is already sorted by time.
//
// Key differences from agmart3:
// - agmart: Takes separate start, stop, event, score, wt, strata vectors
//   Uses an iterative approach moving forward through event times.
//   Suitable for pre-sorted data with explicit left-truncation (start times).
//
// - agmart3: Takes a combined surv vector (tstart, tstop, event flattened),
//   plus sort indices. Uses a two-pointer approach with separate
//   sort orders for entries and exits. More flexible for complex
//   sorting scenarios but requires pre-computed sort indices.

struct AgmartInput {
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    score: Vec<f64>,
    wt: Vec<f64>,
    strata: Vec<i32>,
}

fn agmart_internal(n: usize, method: i32, input: AgmartInput) -> Vec<f64> {
    let start_slice = &input.start;
    let stop_slice = &input.stop;
    let event_slice = &input.event;
    let score_slice = &input.score;
    let wt_slice = &input.wt;
    let strata_slice = &input.strata;
    let mut resid = vec![0.0; n];
    let nused = n;
    let mut local_strata = strata_slice.to_vec();
    if nused > 0 {
        local_strata[nused - 1] = 1;
    }

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
            if local_strata[k] == 1 {
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

            if local_strata[k] == 1 {
                break;
            }
            k += 1;
        }
    }

    resid
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
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
    validate_non_empty(&start, "start")?;
    validate_length(n, start.len(), "start")?;
    validate_length(n, stop.len(), "stop")?;
    validate_length(n, event.len(), "event")?;
    validate_length(n, score.len(), "score")?;
    validate_length(n, wt.len(), "wt")?;
    validate_length(n, strata.len(), "strata")?;

    let input = AgmartInput {
        start,
        stop,
        event,
        score,
        wt,
        strata,
    };
    Ok(agmart_internal(n, method, input))
}
