#![allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn coxmart(
    n: usize,
    method: i32,
    time: &[f64],
    status: &[i32],
    strata: &mut [i32],
    score: &[f64],
    wt: &[f64],
    expect: &mut [f64],
) {
    if n == 0 {
        return;
    }

    strata[n - 1] = 1;

    let mut denom = 0.0;
    for i in (0..n).rev() {
        if strata[i] == 1 {
            denom = 0.0;
        }
        denom += score[i] * wt[i];
        let condition = if i == 0 {
            true
        } else {
            strata[i - 1] == 1 || (time[i - 1] != time[i])
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
        expect[i] = status[i] as f64;
        deaths += status[i];
        wtsum += status[i] as f64 * wt[i];
        e_denom += score[i] * status[i] as f64 * wt[i];

        let is_last = strata[i] == 1 || (i < n - 1 && time[i + 1] != time[i]);

        if is_last {
            if deaths < 2 || method == 0 {
                hazard += wtsum / current_denom;
                for j in lastone..=i {
                    expect[j] -= score[j] * hazard;
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
                for j in lastone..=i {
                    if status[j] == 0 {
                        expect[j] = -score[j] * hazard;
                    } else {
                        expect[j] -= score[j] * temp;
                    }
                }
            }

            lastone = i + 1;
            deaths = 0;
            wtsum = 0.0;
            e_denom = 0.0;
        }

        if strata[i] == 1 {
            hazard = 0.0;
        }
    }

    for j in lastone..n {
        expect[j] -= score[j] * hazard;
    }
}
