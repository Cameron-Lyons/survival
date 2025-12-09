use pyo3::prelude::*;

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
    let start_slice = &start;
    let stop_slice = &stop;
    let event_slice = &event;
    let score_slice = &score;
    let wt_slice = &wt;
    let strata_slice = &strata;
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

    Ok(resid)
}
