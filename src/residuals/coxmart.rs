#![allow(dead_code)]
pub struct SurvivalData<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub strata: &'a mut [i32],
}

pub struct Weights<'a> {
    pub score: &'a [f64],
    pub wt: &'a [f64],
}

pub fn coxmart(
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
