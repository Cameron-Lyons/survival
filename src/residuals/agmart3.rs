#[allow(dead_code)]
pub(crate) struct Agmart3Input {
    pub surv: Vec<f64>,
    pub score: Vec<f64>,
    pub weight: Vec<f64>,
    pub strata: Vec<i32>,
    pub sort1: Vec<i32>,
    pub sort2: Vec<i32>,
}

#[allow(dead_code)]
pub(crate) fn agmart3(nused: i32, input: Agmart3Input, method: i32) -> Vec<f64> {
    let nused = nused as usize;
    let nr = input.surv.len() / 3;

    let tstart = &input.surv[0..nr];
    let tstop = &input.surv[nr..2 * nr];
    let event = &input.surv[2 * nr..3 * nr];

    let sort1: Vec<usize> = input.sort1.iter().map(|&x| (x - 1) as usize).collect();
    let sort2: Vec<usize> = input.sort2.iter().map(|&x| (x - 1) as usize).collect();

    let mut resid = vec![0.0; nr];
    let mut atrisk = vec![false; nr];
    let mut cumhaz = 0.0;
    let mut denom = 0.0;
    let mut current_stratum = input.strata[sort2[0]];

    let mut person1 = 0;
    let mut person2 = 0;

    while person2 < nused {
        let (dtime, new_stratum) = {
            let mut found = false;
            let mut dtime = 0.0;
            let mut new_stratum = current_stratum;

            for &p2 in sort2[person2..nused].iter() {
                if input.strata[p2] != current_stratum {
                    new_stratum = input.strata[p2];
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
                if input.strata[p1] != current_stratum || p1 >= nused {
                    break;
                }
                if atrisk[p1] {
                    resid[p1] -= cumhaz * input.score[p1];
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
            if tstart[p1] < dtime || input.strata[p1] != current_stratum {
                break;
            }
            if atrisk[p1] {
                denom -= input.score[p1] * input.weight[p1];
                resid[p1] -= cumhaz * input.score[p1];
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
            if tstop[p2] < dtime || input.strata[p2] != current_stratum {
                break;
            }

            if event[p2] > 0.0 {
                atrisk[p2] = true;
                resid[p2] = 1.0 + cumhaz * input.score[p2];
                deaths += 1;
                denom += input.score[p2] * input.weight[p2];
                e_denom += input.score[p2] * input.weight[p2];
                wtsum += input.weight[p2];
            } else if tstart[p2] < dtime {
                atrisk[p2] = true;
                denom += input.score[p2] * input.weight[p2];
                resid[p2] = cumhaz * input.score[p2];
            }
            k += 1;
        }

        let hazard = if deaths == 0 {
            0.0
        } else if method == 0 || deaths == 1 {
            wtsum / denom
        } else {
            let wtsum_norm = wtsum / deaths as f64;
            let mut hazard_total = 0.0;
            let mut e_hazard_total = 0.0;

            for i in 0..deaths {
                let temp = i as f64 / deaths as f64;
                let denominator = denom - temp * e_denom;
                hazard_total += wtsum_norm / denominator;
                e_hazard_total += wtsum_norm * (1.0 - temp) / denominator;
            }

            let temp = hazard_total - e_hazard_total;
            for p2 in sort2[person2..k].iter().filter(|&&p| event[p] > 0.0) {
                resid[*p2] += temp * input.score[*p2];
            }

            hazard_total
        };

        cumhaz += hazard;
        person2 = k;
    }

    while person1 < nused {
        let p1 = sort1[person1];
        if atrisk[p1] {
            resid[p1] -= cumhaz * input.score[p1];
        }
        person1 += 1;
    }

    resid
}
