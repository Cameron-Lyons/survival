fn cox_score_residuals(
    y: &[f64],
    strata: &[i32],
    covar: &[f64],
    score: &[f64],
    weights: &[f64],
    method: i32,
    n: usize,
    nvar: usize,
) -> Vec<f64> {
    let time = &y[0..n];
    let status = &y[n..2 * n];
    let mut resid = vec![0.0; n * nvar];
    let mut a = vec![0.0; nvar];
    let mut a2 = vec![0.0; nvar];
    let mut xhaz = vec![0.0; nvar];
    let mut denom = 0.0;
    let mut cumhaz = 0.0;
    let mut stratastart = n as i32 - 1;
    let mut currentstrata = if n > 0 { strata[n - 1] } else { 0 };
    let mut i = stratastart;

    while i >= 0 {
        let i_usize = i as usize;
        let newtime = time[i_usize];
        let mut deaths_count = 0;
        let mut e_denom = 0.0;
        let mut meanwt = 0.0;
        a2.fill(0.0);

        let mut j = i;

        while j >= 0 {
            let j_usize = j as usize;
            if time[j_usize] != newtime || strata[j_usize] != currentstrata {
                break;
            }

            let risk = score[j_usize] * weights[j_usize];
            denom += risk;

            for var in 0..nvar {
                let idx = j_usize * nvar + var;
                let covar_val = covar[idx];
                resid[idx] = score[j_usize] * (covar_val * cumhaz - xhaz[var]);
            }

            for var in 0..nvar {
                a[var] += risk * covar[j_usize * nvar + var];
            }

            if status[j_usize] == 1.0 {
                deaths_count += 1;
                e_denom += risk;
                meanwt += weights[j_usize];
                for var in 0..nvar {
                    a2[var] += risk * covar[j_usize * nvar + var];
                }
            }

            j -= 1;
        }

        let processed_start = j + 1;
        let processed_end = i;

        i = j;

        if deaths_count > 0 {
            let deaths = deaths_count as f64;
            if deaths < 2.0 || method == 0 {
                let hazard = meanwt / denom;
                cumhaz += hazard;

                for var in 0..nvar {
                    let xbar = a[var] / denom;
                    xhaz[var] += xbar * hazard;

                    for k in processed_start..=processed_end {
                        let k_usize = k as usize;
                        let idx = k_usize * nvar + var;
                        resid[idx] += covar[idx] - xbar;
                    }
                }
            } else {
                let meanwt_per_death = meanwt / deaths;
                for dd in 0..deaths_count {
                    let downwt = dd as f64 / deaths;
                    let temp = denom - downwt * e_denom;
                    let hazard = meanwt_per_death / temp;
                    cumhaz += hazard;

                    for var in 0..nvar {
                        let xbar = (a[var] - downwt * a2[var]) / temp;
                        xhaz[var] += xbar * hazard;

                        for k in processed_start..=processed_end {
                            let k_usize = k as usize;
                            let idx = k_usize * nvar + var;
                            let temp2 = covar[idx] - xbar;
                            resid[idx] += temp2 / deaths;
                            resid[idx] += temp2 * score[k_usize] * hazard * downwt;
                        }
                    }
                }
            }
        }

        if i < 0 || (i >= 0 && strata[i as usize] != currentstrata) {
            for k in (i + 1)..=stratastart {
                let k_usize = k as usize;
                for var in 0..nvar {
                    let idx = k_usize * nvar + var;
                    resid[idx] += score[k_usize] * (xhaz[var] - covar[idx] * cumhaz);
                }
            }

            denom = 0.0;
            cumhaz = 0.0;
            a.fill(0.0);
            xhaz.fill(0.0);

            stratastart = i;
            if i >= 0 {
                currentstrata = strata[i as usize];
            }
        }
    }

    resid
}
