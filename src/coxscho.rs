pub fn coxscho(
    nused: usize,
    nvar: usize,
    y: &[f64],
    covar: &mut [f64],
    score: &[f64],
    strata: &[i32],
    method: i32,
    work: &mut [f64],
) {
    assert!(y.len() >= 3 * nused, "y array too short");
    assert!(
        covar.len() >= nvar * nused,
        "covar array too short for nvar and nused"
    );
    assert!(score.len() >= nused, "score array too short");
    assert!(strata.len() >= nused, "strata array too short");
    assert!(
        work.len() >= 3 * nvar,
        "work array must be at least 3 * nvar in length"
    );

    let start = &y[0..nused];
    let stop = &y[nused..2 * nused];
    let event = &y[2 * nused..3 * nused];

    // Split covar into columns (column-major order)
    let mut covar_cols = Vec::with_capacity(nvar);
    let mut remaining = covar;
    for _ in 0..nvar {
        let (col, rest) = remaining.split_at_mut(nused);
        covar_cols.push(col);
        remaining = rest;
    }

    // Split work into a, a2, mean
    let (a, rest) = work.split_at_mut(nvar);
    let (a2, mean) = rest.split_at_mut(nvar);

    let mut person = 0;
    while person < nused {
        if event[person] != 1.0 {
            person += 1;
            continue;
        }

        let time = stop[person];
        let mut deaths = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;

        // Initialize a and a2 to zero
        for i in 0..nvar {
            a[i] = 0.0;
            a2[i] = 0.0;
        }

        // Compute denominator, a, a2, deaths, and efron_wt
        let mut k = person;
        while k < nused {
            if start[k] < time {
                let weight = score[k];
                denom += weight;
                for i in 0..nvar {
                    a[i] += weight * covar_cols[i][k];
                }

                if stop[k] == time && event[k] == 1.0 {
                    deaths += 1.0;
                    efron_wt += weight;
                    for i in 0..nvar {
                        a2[i] += weight * covar_cols[i][k];
                    }
                }
            }

            if strata[k] == 1 {
                break;
            }
            k += 1;
        }

        // Compute mean
        for i in 0..nvar {
            mean[i] = 0.0;
        }
        if deaths > 0.0 {
            for k_death in 0..(deaths as usize) {
                let temp = if method == 1 {
                    (k_death as f64) / deaths
                } else {
                    0.0
                };
                for i in 0..nvar {
                    let denominator = deaths * (denom - temp * efron_wt);
                    if denominator != 0.0 {
                        mean[i] += (a[i] - temp * a2[i]) / denominator;
                    }
                }
            }
        }

        // Update residuals for each death at this time
        let mut k = person;
        while k < nused && stop[k] == time {
            if event[k] == 1.0 {
                for i in 0..nvar {
                    covar_cols[i][k] -= mean[i];
                }
            }
            person += 1;
            if strata[k] == 1 {
                break;
            }
            k += 1;
        }
    }
}
