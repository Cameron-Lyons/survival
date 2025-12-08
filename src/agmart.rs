#[allow(dead_code)]
pub fn agmart(
    n: usize,
    method: i32,
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    score: &[f64],
    wt: &[f64],
    strata: &[i32],
    resid: &mut [f64],
) {
    let nused = n;
    let mut local_strata = strata.to_vec();
    if nused > 0 {
        local_strata[nused - 1] = 1;
    }

    for i in 0..nused {
        resid[i] = event[i] as f64;
    }

    let mut person = 0;
    while person < nused {
        if event[person] == 0 {
            person += 1;
            continue;
        }

        let time = stop[person];
        let mut denom = 0.0;
        let mut e_denom = 0.0;
        let mut deaths = 0;
        let mut wtsum = 0.0;

        let mut k = person;
        while k < nused {
            if start[k] < time {
                denom += score[k] * wt[k];
                if stop[k] == time && event[k] == 1 {
                    deaths += 1;
                    wtsum += wt[k];
                    e_denom += score[k] * wt[k];
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
            if start[k] < time {
                if stop[k] == time && event[k] == 1 {
                    resid[k] -= score[k] * e_hazard;
                } else {
                    resid[k] -= score[k] * hazard;
                }
            }

            if stop[k] == time {
                person += 1;
            }

            if local_strata[k] == 1 {
                break;
            }
            k += 1;
        }
    }
}
