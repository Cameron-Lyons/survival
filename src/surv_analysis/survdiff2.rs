#![allow(clippy::needless_range_loop)]
#[allow(dead_code)]
pub(crate) struct SurvDiffInput<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub group: &'a [i32],
    pub strata: &'a [i32],
}

#[allow(dead_code)]
pub(crate) struct SurvDiffOutput<'a> {
    pub obs: &'a mut [f64],
    pub exp: &'a mut [f64],
    pub var: &'a mut [f64],
    pub risk: &'a mut [f64],
    pub kaplan: &'a mut [f64],
}

#[allow(dead_code)]
pub(crate) struct SurvDiffParams {
    pub nn: i32,
    pub nngroup: i32,
    pub _nstrat: i32,
    pub rho: f64,
}

#[allow(dead_code)]
pub(crate) fn survdiff2(params: SurvDiffParams, input: SurvDiffInput, output: SurvDiffOutput) {
    let ntotal = params.nn as usize;
    let ngroup = params.nngroup as usize;
    let mut istart = 0;
    let mut koff = 0;

    for v in output.var.iter_mut() {
        *v = 0.0;
    }
    for o in output.obs.iter_mut() {
        *o = 0.0;
    }
    for e in output.exp.iter_mut() {
        *e = 0.0;
    }

    while istart < ntotal {
        let mut n = istart;
        while n < ntotal && input.strata[n] != 1 {
            n += 1;
        }
        n += 1;
        let _n_in_stratum = n - istart;

        if params.rho != 0.0 {
            let mut km = 1.0;
            let mut i = istart;
            while i < n {
                let current_time = input.time[i];
                let mut deaths = 0;
                let mut j = i;

                while j < n && input.time[j] == current_time {
                    output.kaplan[j] = km;
                    deaths += input.status[j] as usize;
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
            let current_time = input.time[i];
            let mut deaths = 0;
            let mut j = i;
            let wt = if params.rho == 0.0 {
                1.0
            } else {
                output.kaplan[i].powf(params.rho)
            };

            for r in output.risk.iter_mut().take(ngroup) {
                *r = 0.0;
            }

            while j >= istart && input.time[j] == current_time {
                let k = (input.group[j] - 1) as usize;
                output.risk[k] += 1.0;
                deaths += input.status[j] as usize;
                j -= 1;
            }
            j += 1;

            let nrisk = (n - j) as f64;
            if deaths > 0 {
                for k in 0..ngroup {
                    let exp_index = koff + k;
                    output.exp[exp_index] += wt * (deaths as f64) * output.risk[k] / nrisk;

                    let obs_index = koff + k;
                    output.obs[obs_index] += (input.status[i] as f64) * wt;
                }

                if nrisk > 1.0 {
                    let wt_sq = wt * wt;
                    let factor =
                        wt_sq * (deaths as f64) * (nrisk - deaths as f64) / (nrisk * (nrisk - 1.0));

                    for j_group in 0..ngroup {
                        let rj = output.risk[j_group];
                        let var_start = j_group * ngroup;
                        let tmp = factor * rj;

                        for k_group in 0..ngroup {
                            let rk = output.risk[k_group];
                            output.var[var_start + k_group] += tmp
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
