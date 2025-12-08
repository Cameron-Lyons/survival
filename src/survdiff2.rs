pub fn survdiff2(
    nn: i32,
    nngroup: i32,
    nstrat: i32,
    rho: f64,
    time: &[f64],
    status: &[i32],
    group: &[i32],
    strata: &[i32],
    obs: &mut [f64],
    exp: &mut [f64],
    var: &mut [f64],
    risk: &mut [f64],
    kaplan: &mut [f64],
) {
    let ntotal = nn as usize;
    let ngroup = nngroup as usize;
    let mut istart = 0;
    let mut koff = 0;

    for v in var.iter_mut() {
        *v = 0.0;
    }
    for o in obs.iter_mut() {
        *o = 0.0;
    }
    for e in exp.iter_mut() {
        *e = 0.0;
    }

    while istart < ntotal {
        let mut n = istart;
        while n < ntotal && strata[n] != 1 {
            n += 1;
        }
        n += 1;
        let n_in_stratum = n - istart;

        if rho != 0.0 {
            let mut km = 1.0;
            let mut i = istart;
            while i < n {
                let current_time = time[i];
                let mut deaths = 0;
                let mut j = i;

                while j < n && time[j] == current_time {
                    kaplan[j] = km;
                    deaths += status[j] as usize;
                    j += 1;
                }

                let nrisk = (n - i) as f64;
                if nrisk > 0.0 && deaths > 0 {
                    km *= (nrisk - deaths as f64) / nrisk;
                }
                i = j;
            }
        }

        let mut i = n.saturating_sub(1);
        while i >= istart {
            let current_time = time[i];
            let mut deaths = 0;
            let mut j = i;
            let mut wt = if rho == 0.0 { 1.0 } else { kaplan[i].powf(rho) };

            for r in risk.iter_mut().take(ngroup) {
                *r = 0.0;
            }

            while j >= istart && time[j] == current_time {
                let k = (group[j] - 1) as usize;
                risk[k] += 1.0;
                deaths += status[j] as usize;
                j -= 1;
            }
            j += 1;

            let nrisk = (n - j) as f64;
            if deaths > 0 {
                for k in 0..ngroup {
                    let exp_index = koff + k;
                    exp[exp_index] += wt * (deaths as f64) * risk[k] / nrisk;

                    let obs_index = koff + k;
                    obs[obs_index] += (status[i] as f64) * wt;
                }

                if nrisk > 1.0 {
                    let wt_sq = wt * wt;
                    let factor =
                        wt_sq * (deaths as f64) * (nrisk - deaths as f64) / (nrisk * (nrisk - 1.0));

                    for j_group in 0..ngroup {
                        let rj = risk[j_group];
                        let var_start = j_group * ngroup;
                        let tmp = factor * rj;

                        for k_group in 0..ngroup {
                            let rk = risk[k_group];
                            var[var_start + k_group] += tmp
                                * (if j_group == k_group {
                                    rj - rk / nrisk
                                } else {
                                    -rk / nrisk
                                });
                        }
                    }
                }
            }

            i = j.saturating_sub(1);
        }

        istart = n;
        koff += ngroup;
    }
}
