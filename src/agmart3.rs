use extendr_api::prelude::*;
use std::cmp::Ordering;

#[extendr]
fn agmart3(
    nused: i32,
    surv: Vec<f64>,
    score: Vec<f64>,
    weight: Vec<f64>,
    strata: Vec<i32>,
    sort1: Vec<i32>,
    sort2: Vec<i32>,
    method: i32,
) -> Vec<f64> {
    let nused = nused as usize;
    let nr = surv.len() / 3;
    let method = method as i32;

    let tstart = &surv[0..nr];
    let tstop = &surv[nr..2 * nr];
    let event = &surv[2 * nr..3 * nr];

    let sort1: Vec<usize> = sort1.iter().map(|&x| (x - 1) as usize).collect();
    let sort2: Vec<usize> = sort2.iter().map(|&x| (x - 1) as usize).collect();

    let mut resid = vec![0.0; nr];
    let mut atrisk = vec![false; nr];
    let mut cumhaz = 0.0;
    let mut denom = 0.0;
    let mut current_stratum = strata[sort2[0]];

    let mut person1 = 0;
    let mut person2 = 0;

    while person2 < nused {
        let (dtime, new_stratum) = {
            let mut found = false;
            let mut dtime = 0.0;
            let mut new_stratum = current_stratum;

            for &p2 in sort2[person2..nused].iter() {
                if strata[p2] != current_stratum {
                    new_stratum = strata[p2];
                    break;
                }
                if event[p2] > 0.0 {
                    dtime = tstop[p2];
                    found = true;
                    break;
                }
            }

            if !found {
                current_stratum = new_stratum;
                continue;
            }
            (dtime, new_stratum)
        };

        if new_stratum != current_stratum {
            for &p1 in sort1[person1..].iter() {
                if strata[p1] != current_stratum || p1 >= nused {
                    break;
                }
                if atrisk[p1] {
                    resid[p1] -= cumhaz * score[p1];
                }
                person1 += 1;
            }

            current_stratum = new_stratum;
            cumhaz = 0.0;
            denom = 0.0;
            person2 = person1;
            continue;
        }

        while person1 < nused {
            let p1 = sort1[person1];
            if tstart[p1] < dtime || strata[p1] != current_stratum {
                break;
            }
            if atrisk[p1] {
                denom -= score[p1] * weight[p1];
                resid[p1] -= cumhaz * score[p1];
                atrisk[p1] = false;
            }
            person1 += 1;
        }

        let mut deaths = 0;
        let mut e_denom = 0.0;
        let mut wtsum = 0.0;
        let mut k = person2;

        while k < nused {
            let p2 = sort2[k];
            if tstop[p2] < dtime || strata[p2] != current_stratum {
                break;
            }

            if event[p2] > 0.0 {
                // Event case
                atrisk[p2] = true;
                resid[p2] = 1.0 + cumhaz * score[p2];
                deaths += 1;
                denom += score[p2] * weight[p2];
                e_denom += score[p2] * weight[p2];
                wtsum += weight[p2];
            } else if tstart[p2] < dtime {
                // At risk but no event
                atrisk[p2] = true;
                denom += score[p2] * weight[p2];
                resid[p2] = cumhaz * score[p2];
            }
            k += 1;
        }

        let hazard = if deaths == 0 {
            0.0
        } else if method == 0 || deaths == 1 {
            // Breslow method
            wtsum / denom
        } else {
            // Efron method
            let wtsum_norm = wtsum / deaths as f64;
            let mut hazard_total = 0.0;
            let mut e_hazard_total = 0.0;

            for i in 0..deaths {
                let temp = i as f64 / deaths as f64;
                let denominator = denom - temp * e_denom;
                hazard_total += wtsum_norm / denominator;
                e_hazard_total += wtsum_norm * (1.0 - temp) / denominator;
            }

            // Adjust residuals for tied deaths
            let temp = hazard_total - e_hazard_total;
            for p2 in sort2[person2..k].iter().filter(|&&p| event[p] > 0.0) {
                resid[*p] += temp * score[*p];
            }

            hazard_total
        };

        cumhaz += hazard;
        person2 = k;
    }

    // Finalize remaining subjects
    while person1 < nused {
        let p1 = sort1[person1];
        if atrisk[p1] {
            resid[p1] -= cumhaz * score[p1];
        }
        person1 += 1;
    }

    resid
}

extendr_module! {
    mod coxresiduals;
    fn agmart3;
}
