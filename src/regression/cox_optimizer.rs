use crate::constants::{
    CONVERGENCE_FLAG, COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE,
    PARALLEL_THRESHOLD_MEDIUM,
};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub(crate) struct CoxFitConfig {
    pub method: Method,

    pub max_iter: usize,

    pub eps: f64,

    pub toler: f64,
}

impl Default for CoxFitConfig {
    fn default() -> Self {
        Self {
            method: Method::Breslow,
            max_iter: COX_MAX_ITER,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler: COX_RANK_TOLERANCE,
        }
    }
}

pub(crate) type CoxError = std::convert::Infallible;
#[derive(Debug, Clone, Copy)]
pub(crate) enum Method {
    Breslow,
    Efron,
    Exact,
}
pub(crate) type CoxFitResults = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Array2<f64>,
    [f64; 2],
    f64,
    i32,
    usize,
);

fn add_risk_sums(
    covar: &Array2<f64>,
    nvar: usize,
    person: usize,
    risk: f64,
    denom: &mut f64,
    a: &mut [f64],
    cmat: &mut Array2<f64>,
) {
    *denom += risk;
    for i in 0..nvar {
        let covar_i = covar[(person, i)];
        let risk_covar_i = risk * covar_i;
        a[i] += risk_covar_i;
        for j in 0..=i {
            cmat[(i, j)] += risk_covar_i * covar[(person, j)];
        }
    }
}

fn sort_entry_order(order: &mut [usize], entry_times: &Array1<f64>) {
    order.sort_by(|&lhs, &rhs| {
        entry_times[rhs]
            .total_cmp(&entry_times[lhs])
            .then_with(|| rhs.cmp(&lhs))
    });
}

pub(crate) fn exact_tied_moments(
    risk_indices: &[usize],
    deaths: usize,
    risk_vals: &[f64],
    covar: &Array2<f64>,
) -> (f64, Vec<f64>, Array2<f64>) {
    let nvar = covar.ncols();
    let cmat_len = nvar * nvar;
    let mut denom = vec![0.0; deaths + 1];
    let mut a = vec![vec![0.0; nvar]; deaths + 1];
    let mut cmat = vec![vec![0.0; cmat_len]; deaths + 1];
    denom[0] = 1.0;

    for (seen, &person) in risk_indices.iter().enumerate() {
        let risk = risk_vals[person];
        let max_size = deaths.min(seen + 1);
        for size in (1..=max_size).rev() {
            let base = denom[size - 1];
            if base == 0.0 {
                continue;
            }
            let (prev_a_rows, current_a_rows) = a.split_at_mut(size);
            let prev_a = &prev_a_rows[size - 1];
            let current_a = &mut current_a_rows[0];
            let (prev_cmat_rows, current_cmat_rows) = cmat.split_at_mut(size);
            let prev_cmat = &prev_cmat_rows[size - 1];
            let current_cmat = &mut current_cmat_rows[0];
            denom[size] += risk * base;
            for i in 0..nvar {
                let xi = covar[(person, i)];
                current_a[i] += risk * (prev_a[i] + base * xi);
                for j in 0..=i {
                    let xj = covar[(person, j)];
                    current_cmat[i * nvar + j] += risk
                        * (prev_cmat[i * nvar + j]
                            + xi * prev_a[j]
                            + xj * prev_a[i]
                            + base * xi * xj);
                }
            }
        }
    }

    let tied_a = a.pop().expect("exact tied score row must exist");
    let tied_cmat = cmat.pop().expect("exact tied information row must exist");
    let mut cmat_array = Array2::zeros((nvar, nvar));
    for i in 0..nvar {
        for j in 0..=i {
            cmat_array[(i, j)] = tied_cmat[i * nvar + j];
        }
    }
    (denom[deaths], tied_a, cmat_array)
}

pub(crate) struct CoxFit {
    time: Array1<f64>,
    status: Array1<i32>,
    entry_times: Option<Array1<f64>>,
    entry_order: Option<Vec<usize>>,
    covar: Array2<f64>,
    strata: Array1<i32>,
    offset: Array1<f64>,
    weights: Array1<f64>,
    method: Method,
    max_iter: usize,
    eps: f64,
    toler: f64,
    scale: Vec<f64>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Array2<f64>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter: usize,
}

pub(crate) struct CoxFitBuilder {
    time: Array1<f64>,
    status: Array1<i32>,
    covar: Array2<f64>,
    strata: Option<Array1<i32>>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    method: Method,
    max_iter: usize,
    eps: f64,
    toler: f64,
    doscale: Option<Vec<bool>>,
    initial_beta: Option<Vec<f64>>,
}

impl CoxFitBuilder {
    pub(crate) fn new(time: Array1<f64>, status: Array1<i32>, covar: Array2<f64>) -> Self {
        Self {
            time,
            status,
            covar,
            strata: None,
            offset: None,
            weights: None,
            method: Method::Breslow,
            max_iter: COX_MAX_ITER,
            eps: COX_CONVERGENCE_TOLERANCE,
            toler: COX_RANK_TOLERANCE,
            doscale: None,
            initial_beta: None,
        }
    }

    pub(crate) fn strata(mut self, strata: Array1<i32>) -> Self {
        self.strata = Some(strata);
        self
    }

    pub(crate) fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub(crate) fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub(crate) fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub(crate) fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub(crate) fn toler(mut self, toler: f64) -> Self {
        self.toler = toler;
        self
    }

    pub(crate) fn initial_beta(mut self, initial_beta: Vec<f64>) -> Self {
        self.initial_beta = Some(initial_beta);
        self
    }

    pub(crate) fn build(self) -> Result<CoxFit, CoxError> {
        let nused = self.covar.nrows();
        let nvar = self.covar.ncols();

        let strata = self.strata.unwrap_or_else(|| Array1::from_elem(nused, 0));
        let offset = self.offset.unwrap_or_else(|| Array1::from_elem(nused, 0.0));
        let weights = self
            .weights
            .unwrap_or_else(|| Array1::from_elem(nused, 1.0));
        let doscale = self.doscale.unwrap_or_else(|| vec![true; nvar]);
        let initial_beta = self.initial_beta.unwrap_or_else(|| vec![0.0; nvar]);
        CoxFit::new(
            self.time,
            self.status,
            self.covar,
            strata,
            offset,
            weights,
            self.method,
            self.max_iter,
            self.eps,
            self.toler,
            doscale,
            initial_beta,
        )
    }
}
impl CoxFit {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_config(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        config: CoxFitConfig,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        Self::with_config_and_entry_times(
            time,
            status,
            covar,
            None,
            strata,
            offset,
            weights,
            config,
            doscale,
            initial_beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_config_and_entry_times(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        entry_times: Option<Array1<f64>>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        config: CoxFitConfig,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        let nvar = covar.ncols();
        let entry_order = entry_times.as_ref().map(|entry_times| {
            let mut order: Vec<usize> = (0..entry_times.len()).collect();
            let mut stratum_start = 0;
            for stratum_end in 0..strata.len() {
                if strata[stratum_end] != 1 {
                    continue;
                }
                sort_entry_order(&mut order[stratum_start..=stratum_end], entry_times);
                stratum_start = stratum_end + 1;
            }
            if stratum_start < order.len() {
                sort_entry_order(&mut order[stratum_start..], entry_times);
            }
            order
        });
        let mut cox = Self {
            time,
            status,
            entry_times,
            entry_order,
            covar,
            strata,
            offset,
            weights,
            method: config.method,
            max_iter: config.max_iter,
            eps: config.eps,
            toler: config.toler,
            scale: vec![1.0; nvar],
            means: vec![0.0; nvar],
            beta: initial_beta,
            u: vec![0.0; nvar],
            imat: Array2::zeros((nvar, nvar)),
            loglik: [0.0; 2],
            sctest: 0.0,
            flag: 0,
            iter: 0,
        };
        cox.scale_center(doscale)?;
        Ok(cox)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        method: Method,
        max_iter: usize,
        eps: f64,
        toler: f64,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        let config = CoxFitConfig {
            method,
            max_iter,
            eps,
            toler,
        };
        Self::with_config(
            time,
            status,
            covar,
            strata,
            offset,
            weights,
            config,
            doscale,
            initial_beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_entry_times(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        entry_times: Option<Array1<f64>>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        method: Method,
        max_iter: usize,
        eps: f64,
        toler: f64,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        let config = CoxFitConfig {
            method,
            max_iter,
            eps,
            toler,
        };
        Self::with_config_and_entry_times(
            time,
            status,
            covar,
            entry_times,
            strata,
            offset,
            weights,
            config,
            doscale,
            initial_beta,
        )
    }
    fn scale_center(&mut self, doscale: Vec<bool>) -> Result<(), CoxError> {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let total_weight: f64 = self.weights.sum();
        let means: Vec<f64> = (0..nvar)
            .into_par_iter()
            .map(|i| {
                if !doscale[i] {
                    0.0
                } else {
                    let mut mean = 0.0;
                    for (person, &w) in self.weights.iter().enumerate() {
                        mean += w * self.covar[(person, i)];
                    }
                    mean / total_weight
                }
            })
            .collect();
        let scales: Vec<f64> = (0..nvar)
            .into_par_iter()
            .map(|i| {
                if !doscale[i] {
                    1.0
                } else {
                    let mean = means[i];
                    let abs_sum: f64 = (0..nused)
                        .map(|person| self.weights[person] * (self.covar[(person, i)] - mean).abs())
                        .sum();
                    if abs_sum > 0.0 {
                        total_weight / abs_sum
                    } else {
                        1.0
                    }
                }
            })
            .collect();
        if nused > PARALLEL_THRESHOLD_MEDIUM && nvar > 1 {
            use std::sync::atomic::{AtomicPtr, Ordering};
            let covar_ptr = AtomicPtr::new(self.covar.as_mut_ptr());
            let covar_stride = self.covar.strides();
            let row_stride = covar_stride[0];
            let col_stride = covar_stride[1];

            (0..nvar).into_par_iter().for_each(|i| {
                if doscale[i] {
                    let mean = means[i];
                    let scale_val = scales[i];
                    let base_ptr = covar_ptr.load(Ordering::Relaxed);
                    for person in 0..nused {
                        // SAFETY: `person < nused == self.covar.nrows()` and
                        // `i < nvar == self.covar.ncols()`, so the stride
                        // calculation addresses an in-bounds matrix element.
                        // Rayon partitions work by unique column index `i`,
                        // so concurrent writes target disjoint `(person, i)`
                        // elements of the owned `Array2`, and `self.covar` is
                        // not otherwise mutated while this closure runs.
                        unsafe {
                            let offset = person as isize * row_stride + i as isize * col_stride;
                            let ptr = base_ptr.offset(offset);
                            *ptr = (*ptr - mean) * scale_val;
                        }
                    }
                }
            });
        } else {
            for i in 0..nvar {
                if doscale[i] {
                    let mean = means[i];
                    let scale_val = scales[i];
                    for person in 0..nused {
                        self.covar[(person, i)] = (self.covar[(person, i)] - mean) * scale_val;
                    }
                }
            }
        }
        self.means = means;
        self.scale = scales;
        let new_beta: Vec<f64> = self
            .beta
            .par_iter()
            .zip(self.scale.par_iter())
            .map(|(&b, &s)| b / s)
            .collect();
        self.beta = new_beta;
        Ok(())
    }
    fn iterate_right_censored(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let mut a = vec![0.0; nvar];
        let mut a2 = vec![0.0; nvar];
        let mut cmat = Array2::zeros((nvar, nvar));
        let mut cmat2 = Array2::zeros((nvar, nvar));
        let mut loglik = 0.0;
        let mut denom = 0.0;
        let mut risk_indices: Vec<usize> = Vec::new();

        let (zbeta_vals, risk_vals): (Vec<f64>, Vec<f64>) = if nused > PARALLEL_THRESHOLD_MEDIUM {
            (0..nused)
                .into_par_iter()
                .map(|p| {
                    let zb = self.offset[p]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(p, i)]);
                    (zb, zb.exp() * self.weights[p])
                })
                .unzip()
        } else {
            (0..nused)
                .map(|p| {
                    let zb = self.offset[p]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(p, i)]);
                    (zb, zb.exp() * self.weights[p])
                })
                .unzip()
        };

        let mut person = nused as isize - 1;
        while person >= 0 {
            let person_idx = person as usize;
            if self.strata[person_idx] == 1 {
                a.fill(0.0);
                cmat.fill(0.0);
                denom = 0.0;
                risk_indices.clear();
            }
            let dtime = self.time[person_idx];
            let mut ndead = 0;
            let mut deadwt = 0.0;
            let mut denom2 = 0.0;
            let mut _nrisk = 0;
            while person >= 0 && self.time[person as usize] == dtime {
                let person_i = person as usize;
                _nrisk += 1;
                let zbeta = zbeta_vals[person_i];
                let risk = risk_vals[person_i];
                risk_indices.push(person_i);
                if self.status[person_i] == 0 {
                    denom += risk;
                    for i in 0..nvar {
                        let covar_i = self.covar[(person_i, i)];
                        let risk_covar_i = risk * covar_i;
                        a[i] += risk_covar_i;
                        for j in 0..=i {
                            cmat[(i, j)] += risk_covar_i * self.covar[(person_i, j)];
                        }
                    }
                } else {
                    ndead += 1;
                    deadwt += self.weights[person_i];
                    denom2 += risk;
                    loglik += self.weights[person_i] * zbeta;
                    for i in 0..nvar {
                        let covar_i = self.covar[(person_i, i)];
                        self.u[i] += self.weights[person_i] * covar_i;
                        let risk_covar_i = risk * covar_i;
                        a2[i] += risk_covar_i;
                        for j in 0..=i {
                            cmat2[(i, j)] += risk_covar_i * self.covar[(person_i, j)];
                        }
                    }
                }
                person -= 1;
                if person >= 0 && self.strata[person as usize] == 1 {
                    break;
                }
            }
            if ndead > 0 {
                if matches!(method, Method::Exact) && ndead > 1 {
                    let (exact_denom, exact_a, exact_cmat) =
                        exact_tied_moments(&risk_indices, ndead, &risk_vals, &self.covar);
                    loglik -= exact_denom.ln();
                    for i in 0..nvar {
                        let temp = exact_a[i] / exact_denom;
                        self.u[i] -= temp;
                        for j in 0..=i {
                            let val = (exact_cmat[(i, j)] - temp * exact_a[j]) / exact_denom;
                            self.imat[(j, i)] += val;
                            if i != j {
                                self.imat[(i, j)] += val;
                            }
                        }
                    }
                    denom += denom2;
                    for i in 0..nvar {
                        a[i] += a2[i];
                        for j in 0..=i {
                            cmat[(i, j)] += cmat2[(i, j)];
                        }
                    }
                } else if matches!(method, Method::Breslow) || ndead == 1 {
                    denom += denom2;
                    loglik -= deadwt * denom.ln();
                    for i in 0..nvar {
                        a[i] += a2[i];
                        let temp = a[i] / denom;
                        self.u[i] -= deadwt * temp;
                        for j in 0..=i {
                            cmat[(i, j)] += cmat2[(i, j)];
                            let val = deadwt * (cmat[(i, j)] - temp * a[j]) / denom;
                            self.imat[(j, i)] += val;
                            if i != j {
                                self.imat[(i, j)] += val;
                            }
                        }
                    }
                } else {
                    let death_count = ndead as f64;
                    let risk_fraction = denom2 / death_count;
                    let weight_average = deadwt / death_count;
                    for _ in 0..ndead {
                        denom += risk_fraction;
                        loglik -= weight_average * denom.ln();
                        for i in 0..nvar {
                            a[i] += a2[i] / death_count;
                            let temp = a[i] / denom;
                            self.u[i] -= weight_average * temp;
                            for j in 0..=i {
                                cmat[(i, j)] += cmat2[(i, j)] / death_count;
                                let val = weight_average * (cmat[(i, j)] - temp * a[j]) / denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    }
                }
                a2.fill(0.0);
                cmat2.fill(0.0);
            }
        }
        Ok(loglik)
    }

    fn iterate_counting_process(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored(beta);
        };
        let entry_order = self
            .entry_order
            .as_deref()
            .expect("entry order must accompany counting-process entry times");
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;
        self.u.fill(0.0);
        self.imat.fill(0.0);

        let (zbeta_vals, risk_vals): (Vec<f64>, Vec<f64>) = if nused > PARALLEL_THRESHOLD_MEDIUM {
            (0..nused)
                .into_par_iter()
                .map(|person| {
                    let zbeta = self.offset[person]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(person, i)]);
                    (zbeta, zbeta.exp() * self.weights[person])
                })
                .unzip()
        } else {
            (0..nused)
                .map(|person| {
                    let zbeta = self.offset[person]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(person, i)]);
                    (zbeta, zbeta.exp() * self.weights[person])
                })
                .unzip()
        };

        let mut loglik = 0.0;
        let mut stratum_start = 0usize;
        for stratum_end in 0..nused {
            if self.strata[stratum_end] != 1 {
                continue;
            }

            let start_order = &entry_order[stratum_start..=stratum_end];

            let mut stop_denom = 0.0;
            let mut stop_a = vec![0.0; nvar];
            let mut stop_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut unentered_denom = 0.0;
            let mut unentered_a = vec![0.0; nvar];
            let mut unentered_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut stop_ptr = stratum_end as isize;
            let mut start_ptr = 0usize;
            let mut time_end = stratum_end;
            let mut death_a = vec![0.0; nvar];
            let mut death_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut event_a = vec![0.0; nvar];
            let mut event_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut risk_indices: Vec<usize> = Vec::new();

            loop {
                let event_time = self.time[time_end];
                while stop_ptr >= stratum_start as isize
                    && self.time[stop_ptr as usize] >= event_time
                {
                    let person = stop_ptr as usize;
                    add_risk_sums(
                        &self.covar,
                        nvar,
                        person,
                        risk_vals[person],
                        &mut stop_denom,
                        &mut stop_a,
                        &mut stop_cmat,
                    );
                    stop_ptr -= 1;
                }
                while start_ptr < start_order.len()
                    && entry_times[start_order[start_ptr]] >= event_time
                {
                    let person = start_order[start_ptr];
                    add_risk_sums(
                        &self.covar,
                        nvar,
                        person,
                        risk_vals[person],
                        &mut unentered_denom,
                        &mut unentered_a,
                        &mut unentered_cmat,
                    );
                    start_ptr += 1;
                }

                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }

                let mut ndead = 0usize;
                let mut deadwt = 0.0;
                let mut denom2 = 0.0;
                death_a.fill(0.0);
                death_cmat.fill(0.0);

                for person in time_start..=time_end {
                    if self.status[person] == 0 {
                        continue;
                    }
                    ndead += 1;
                    deadwt += self.weights[person];
                    loglik += self.weights[person] * zbeta_vals[person];
                    add_risk_sums(
                        &self.covar,
                        nvar,
                        person,
                        risk_vals[person],
                        &mut denom2,
                        &mut death_a,
                        &mut death_cmat,
                    );
                    for i in 0..nvar {
                        self.u[i] += self.weights[person] * self.covar[(person, i)];
                    }
                }

                if ndead > 0 {
                    let denom = stop_denom - unentered_denom;
                    event_a.fill(0.0);
                    event_cmat.fill(0.0);
                    for i in 0..nvar {
                        event_a[i] = stop_a[i] - unentered_a[i];
                        for j in 0..=i {
                            event_cmat[(i, j)] = stop_cmat[(i, j)] - unentered_cmat[(i, j)];
                        }
                    }
                    if matches!(method, Method::Exact) && ndead > 1 {
                        risk_indices.clear();
                        risk_indices.extend((stratum_start..=stratum_end).filter(|&idx| {
                            entry_times[idx] < event_time && self.time[idx] >= event_time
                        }));
                        let (exact_denom, exact_a, exact_cmat) =
                            exact_tied_moments(&risk_indices, ndead, &risk_vals, &self.covar);
                        loglik -= exact_denom.ln();
                        for i in 0..nvar {
                            let temp = exact_a[i] / exact_denom;
                            self.u[i] -= temp;
                            for j in 0..=i {
                                let val = (exact_cmat[(i, j)] - temp * exact_a[j]) / exact_denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    } else if matches!(method, Method::Breslow) || ndead == 1 {
                        loglik -= deadwt * denom.ln();
                        for i in 0..nvar {
                            let temp = event_a[i] / denom;
                            self.u[i] -= deadwt * temp;
                            for j in 0..=i {
                                let val = deadwt * (event_cmat[(i, j)] - temp * event_a[j]) / denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    } else {
                        let death_count = ndead as f64;
                        let risk_fraction = denom2 / death_count;
                        let weight_average = deadwt / death_count;
                        let mut efron_denom = denom - denom2;
                        for i in 0..nvar {
                            event_a[i] -= death_a[i];
                            for j in 0..=i {
                                event_cmat[(i, j)] -= death_cmat[(i, j)];
                            }
                        }
                        for _ in 0..ndead {
                            efron_denom += risk_fraction;
                            loglik -= weight_average * efron_denom.ln();
                            for i in 0..nvar {
                                event_a[i] += death_a[i] / death_count;
                                let temp = event_a[i] / efron_denom;
                                self.u[i] -= weight_average * temp;
                                for j in 0..=i {
                                    event_cmat[(i, j)] += death_cmat[(i, j)] / death_count;
                                    let val = weight_average
                                        * (event_cmat[(i, j)] - temp * event_a[j])
                                        / efron_denom;
                                    self.imat[(j, i)] += val;
                                    if i != j {
                                        self.imat[(i, j)] += val;
                                    }
                                }
                            }
                        }
                    }
                }

                if time_start == stratum_start {
                    break;
                }
                time_end = time_start - 1;
            }
            stratum_start = stratum_end + 1;
        }

        Ok(loglik)
    }

    fn iterate(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        if self.entry_times.is_some() {
            self.iterate_counting_process(beta)
        } else {
            self.iterate_right_censored(beta)
        }
    }

    pub(crate) fn fit(&mut self) -> Result<(), CoxError> {
        let nvar = self.beta.len();
        let mut newbeta = vec![0.0; nvar];
        let mut a = vec![0.0; nvar];
        let mut halving = 0;
        let mut _notfinite;
        let beta_copy = self.beta.clone();
        self.loglik[0] = self.iterate(&beta_copy)?;
        self.loglik[1] = self.loglik[0];
        if nvar == 0 {
            self.flag = 0;
            return Ok(());
        }
        a.copy_from_slice(&self.u);
        self.flag = Self::cholesky(&mut self.imat, self.toler);
        Self::chsolve(&self.imat, &mut a);
        self.sctest = a.iter().zip(&self.u).map(|(ai, ui)| ai * ui).sum();
        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            Self::chinv(&mut self.imat);
            self.rescale_params();
            return Ok(());
        }
        newbeta.copy_from_slice(&self.beta);
        for i in 0..nvar {
            newbeta[i] += a[i];
        }
        self.loglik[1] = self.loglik[0];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            let newlk = match self.iterate(&newbeta) {
                Ok(lk) if lk.is_finite() => lk,
                _ => {
                    _notfinite = true;
                    f64::NAN
                }
            };
            self.flag = Self::cholesky(&mut self.imat, self.toler);
            _notfinite = !newlk.is_finite();
            if !_notfinite {
                for i in 0..nvar {
                    if !self.u[i].is_finite() {
                        _notfinite = true;
                        break;
                    }
                    for j in 0..nvar {
                        if !self.imat[(i, j)].is_finite() {
                            _notfinite = true;
                            break;
                        }
                    }
                }
            }
            if !_notfinite && (1.0 - self.loglik[1] / newlk).abs() <= self.eps {
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                Self::chinv(&mut self.imat);
                self.rescale_params();
                if halving > 0 {
                    self.flag = -2;
                }
                return Ok(());
            }
            if _notfinite || newlk < self.loglik[1] {
                halving += 1;
                for (newbeta_elem, beta_elem) in newbeta.iter_mut().zip(self.beta.iter()).take(nvar)
                {
                    *newbeta_elem =
                        (*newbeta_elem + (halving as f64) * beta_elem) / (halving as f64 + 1.0);
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                a.copy_from_slice(&self.u);
                Self::chsolve(&self.imat, &mut a);
                for (newbeta_elem, (beta_elem, a_elem)) in newbeta
                    .iter_mut()
                    .zip(self.beta.iter().zip(a.iter()))
                    .take(nvar)
                {
                    *newbeta_elem = beta_elem + a_elem;
                }
            }
        }
        let beta_final = self.beta.clone();
        self.loglik[1] = self.iterate(&beta_final)?;
        self.flag = Self::cholesky(&mut self.imat, self.toler);
        Self::chinv(&mut self.imat);
        self.rescale_params();
        self.flag = CONVERGENCE_FLAG;
        Ok(())
    }
    fn rescale_params(&mut self) {
        for (i, (&scale_i, (beta, u))) in self
            .scale
            .iter()
            .zip(self.beta.iter_mut().zip(self.u.iter_mut()))
            .enumerate()
        {
            *beta *= scale_i;
            *u /= scale_i;
            for (j, &scale_j) in self.scale.iter().enumerate() {
                self.imat[(i, j)] *= scale_i * scale_j;
            }
        }
    }
    fn cholesky(mat: &mut Array2<f64>, toler: f64) -> i32 {
        let n = mat.nrows();
        let mut eps = 0.0_f64;
        for i in 0..n {
            if mat[(i, i)] > eps {
                eps = mat[(i, i)];
            }
            for j in (i + 1)..n {
                mat[(j, i)] = mat[(i, j)];
            }
        }
        eps = if eps == 0.0 { toler } else { eps * toler };

        let mut rank = 0_i32;
        let mut nonnegative = 1_i32;
        for i in 0..n {
            let pivot = mat[(i, i)];
            if !pivot.is_finite() || pivot < eps {
                mat[(i, i)] = 0.0;
                if pivot < -8.0 * eps {
                    nonnegative = -1;
                }
                continue;
            }

            rank += 1;
            for j in (i + 1)..n {
                let temp = mat[(j, i)] / pivot;
                mat[(j, i)] = temp;
                mat[(j, j)] -= temp * temp * pivot;
                for k in (j + 1)..n {
                    mat[(k, j)] -= temp * mat[(k, i)];
                }
            }
        }
        rank * nonnegative
    }
    fn chsolve(chol: &Array2<f64>, a: &mut [f64]) {
        for i in 0..a.len() {
            let mut temp = a[i];
            for j in 0..i {
                temp -= a[j] * chol[(i, j)];
            }
            a[i] = temp;
        }
        for i in (0..a.len()).rev() {
            if chol[(i, i)] == 0.0 {
                a[i] = 0.0;
            } else {
                let mut temp = a[i] / chol[(i, i)];
                for j in (i + 1)..a.len() {
                    temp -= a[j] * chol[(j, i)];
                }
                a[i] = temp;
            }
        }
    }
    fn chinv(mat: &mut Array2<f64>) {
        let n = mat.nrows();
        for i in 0..n {
            if mat[(i, i)] > 0.0 {
                mat[(i, i)] = 1.0 / mat[(i, i)];
                for j in (i + 1)..n {
                    mat[(j, i)] = -mat[(j, i)];
                    for k in 0..i {
                        mat[(j, k)] += mat[(j, i)] * mat[(i, k)];
                    }
                }
            }
        }

        for i in 0..n {
            if mat[(i, i)] == 0.0 {
                for j in 0..i {
                    mat[(j, i)] = 0.0;
                }
                for j in i..n {
                    mat[(i, j)] = 0.0;
                }
            } else {
                for j in (i + 1)..n {
                    let temp = mat[(j, i)] * mat[(j, j)];
                    mat[(i, j)] = temp;
                    for k in i..j {
                        mat[(i, k)] += temp * mat[(j, k)];
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..i {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }
    pub(crate) fn results(self) -> CoxFitResults {
        (
            self.beta,
            self.means,
            self.u,
            self.imat,
            self.loglik,
            self.sctest,
            self.flag,
            self.iter,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counting_process_order_fixture() -> CoxFit {
        CoxFit::new_with_entry_times(
            Array1::from_vec(vec![2.0, 3.0, 4.0, 2.5, 4.0, 5.0]),
            Array1::from_vec(vec![1, 1, 0, 1, 0, 1]),
            Array2::from_shape_vec(
                (6, 2),
                vec![0.2, 1.0, 0.8, 0.4, 0.5, 1.2, 1.1, 0.3, 0.7, 0.9, 1.3, 0.6],
            )
            .expect("counting-process fixture covariates should have a valid shape"),
            Some(Array1::from_vec(vec![0.5, 1.5, 1.5, 2.0, 0.25, 2.0])),
            Array1::from_vec(vec![0, 0, 1, 0, 0, 1]),
            Array1::zeros(6),
            Array1::ones(6),
            Method::Efron,
            10,
            1e-9,
            1e-9,
            vec![true; 2],
            vec![0.0; 2],
        )
        .expect("counting-process fixture should initialize")
    }

    #[test]
    fn test_cox_fit_config_default() {
        let config = CoxFitConfig::default();

        assert!(matches!(config.method, Method::Breslow));
        assert_eq!(config.max_iter, COX_MAX_ITER);
        assert_eq!(config.eps, COX_CONVERGENCE_TOLERANCE);
        assert_eq!(config.toler, COX_RANK_TOLERANCE);
    }

    #[test]
    fn test_cox_fit_builder_basic() {
        let time = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let status = Array1::from_vec(vec![1, 0, 1, 0, 1]);
        let covar = Array2::from_shape_vec((5, 1), vec![0.5, 1.0, 0.3, 0.8, 0.6]).unwrap();

        let builder = CoxFitBuilder::new(time, status, covar);
        assert_eq!(builder.max_iter, COX_MAX_ITER);
        assert_eq!(builder.eps, COX_CONVERGENCE_TOLERANCE);
        assert_eq!(builder.toler, COX_RANK_TOLERANCE);
        let result = builder.build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_cox_fit_builder_with_options() {
        let time = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let status = Array1::from_vec(vec![1, 0, 1, 0, 1]);
        let covar = Array2::from_shape_vec((5, 1), vec![0.5, 1.0, 0.3, 0.8, 0.6]).unwrap();
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);

        let builder = CoxFitBuilder::new(time, status, covar)
            .weights(weights)
            .max_iter(50)
            .eps(1e-8);
        let result = builder.build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_cox_fit_and_results() {
        let time = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let status = Array1::from_vec(vec![1, 0, 1, 0, 1, 0, 1, 0]);
        let covar =
            Array2::from_shape_vec((8, 1), vec![0.5, 1.0, 0.3, 0.8, 0.6, 0.4, 0.9, 0.2]).unwrap();

        let builder = CoxFitBuilder::new(time, status, covar);
        let mut cox = builder.build().unwrap();

        let fit_result = cox.fit();
        assert!(fit_result.is_ok());

        let (beta, _means, _u, _imat, loglik, _sctest, _flag, _iter) = cox.results();

        assert_eq!(beta.len(), 1);
        assert!(loglik[0].is_finite());
        assert!(loglik[1].is_finite());
    }

    #[test]
    fn converged_coefficients_match_the_reported_log_likelihood() {
        let time = Array1::from_vec((1..=16).map(f64::from).collect());
        let status = Array1::from_vec(vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]);
        let covariates = vec![
            0.5, 1.2, 1.8, 0.3, 0.2, 2.1, 2.5, 0.8, 0.8, 1.5, 1.5, 0.5, 0.3, 1.8, 2.2, 1.1, 1.0,
            0.9, 0.7, 1.7, 2.0, 0.4, 1.2, 1.3, 0.9, 2.0, 1.6, 0.7, 0.4, 1.4, 2.1, 1.0,
        ];
        let covar = Array2::from_shape_vec((16, 2), covariates)
            .expect("fixture covariates should have a valid shape");

        let mut fit = CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
            .max_iter(20)
            .eps(1e-5)
            .build()
            .expect("fixture fit should initialize");
        fit.fit().expect("fixture fit should converge");
        let (beta, _means, _u, _variance, loglik, _sctest, _flag, _iter) = fit.results();

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(beta)
            .build()
            .expect("coefficient evaluation should initialize");
        evaluation
            .fit()
            .expect("coefficient evaluation should succeed");
        let (_beta, _means, _u, _variance, evaluated, _sctest, _flag, _iter) = evaluation.results();

        assert!((evaluated[0] - loglik[1]).abs() < 1e-12);
    }

    #[test]
    fn nonconverged_fit_refactors_information_at_the_last_accepted_beta() {
        let time = Array1::from_vec((1..=8).map(f64::from).collect());
        let status = Array1::from_vec(vec![1, 0, 1, 1, 0, 1, 0, 1]);
        let covar = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.2, 1.0, 0.7, 0.4, 1.2, 1.5, 0.1, 0.8, 1.0, 1.1, 0.3, 1.8, 0.9, 2.0, 0.5,
            ],
        )
        .unwrap();
        let mut fit = CoxFitBuilder::new(time.clone(), status.clone(), covar.clone())
            .max_iter(1)
            .eps(1e-12)
            .build()
            .expect("limited-iteration fixture should initialize");

        fit.fit().expect("limited-iteration fixture should fit");
        let (beta, _means, score, variance, loglik, _sctest, flag, _iter) = fit.results();
        assert_eq!(flag, CONVERGENCE_FLAG);

        let mut evaluation = CoxFitBuilder::new(time, status, covar)
            .max_iter(0)
            .initial_beta(beta)
            .build()
            .expect("accepted coefficient evaluation should initialize");
        evaluation
            .fit()
            .expect("accepted coefficient evaluation should succeed");
        let (_beta, _means, evaluated_score, evaluated_variance, evaluated_loglik, ..) =
            evaluation.results();

        assert!((evaluated_loglik[0] - loglik[1]).abs() < 1e-12);
        for i in 0..2 {
            assert!((evaluated_score[i] - score[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!((evaluated_variance[(i, j)] - variance[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exact_tied_moments_match_hand_computed_two_death_set() {
        let covar = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap();

        let (denom, score, information) =
            exact_tied_moments(&[0, 1, 2], 2, &[2.0, 3.0, 5.0], &covar);

        assert_eq!(denom, 31.0);
        assert_eq!(score, vec![91.0, 142.0]);
        assert_eq!(information[(0, 0)], 301.0);
        assert_eq!(information[(1, 0)], 442.0);
        assert_eq!(information[(1, 1)], 724.0);
    }

    #[test]
    fn counting_process_tied_methods_fit_with_entry_times() {
        for method in [Method::Efron, Method::Exact] {
            let mut cox = CoxFit::new_with_entry_times(
                Array1::from_vec(vec![2.0, 2.0, 3.0, 4.0, 4.0, 5.0]),
                Array1::from_vec(vec![1, 1, 0, 1, 1, 0]),
                Array2::from_shape_vec((6, 1), vec![0.0, 0.4, 0.2, 1.0, 1.4, 0.8]).unwrap(),
                Some(Array1::from_vec(vec![0.0, 0.5, 0.0, 1.0, 2.0, 0.0])),
                Array1::from_vec(vec![0, 0, 0, 0, 0, 1]),
                Array1::from_vec(vec![0.0; 6]),
                Array1::from_vec(vec![1.0; 6]),
                method,
                5,
                1e-8,
                1e-8,
                vec![true],
                vec![0.0],
            )
            .expect("counting-process Cox fit should initialize");

            let result = cox.fit();
            assert!(result.is_ok());
            let (beta, _means, _u, information, loglik, _sctest, _flag, _iter) = cox.results();
            assert_eq!(beta.len(), 1);
            assert!(beta[0].is_finite());
            assert!(information[(0, 0)].is_finite());
            assert!(loglik[0].is_finite());
            assert!(loglik[1].is_finite());
        }
    }

    #[test]
    fn counting_process_entry_order_is_precomputed_per_stratum_with_index_ties() {
        let fit = counting_process_order_fixture();

        assert_eq!(
            fit.entry_order.as_deref(),
            Some([2, 1, 0, 5, 3, 4].as_slice())
        );
    }

    #[test]
    fn counting_process_entry_order_is_reused_across_evaluations() {
        let mut fit = counting_process_order_fixture();
        let beta = [0.2, -0.15];
        let cached_order = fit.entry_order.clone();

        let first_loglik = fit
            .iterate(&beta)
            .expect("first counting-process evaluation should succeed");
        let first_score = fit.u.clone();
        let first_information = fit.imat.clone();
        let second_loglik = fit
            .iterate(&beta)
            .expect("second counting-process evaluation should succeed");

        assert_eq!(second_loglik, first_loglik);
        assert_eq!(fit.u, first_score);
        assert_eq!(fit.imat, first_information);
        assert_eq!(fit.entry_order, cached_order);
    }

    #[test]
    fn information_factorization_inverts_spd_matrix() {
        let mut information = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 0.0, 3.0]).unwrap();

        let rank = CoxFit::cholesky(&mut information, 1e-9);
        CoxFit::chinv(&mut information);

        assert_eq!(rank, 2);
        let expected = [[0.375, -0.25], [-0.25, 0.5]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((information[(i, j)] - expected[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn singular_information_solve_and_inverse_zero_the_alias() {
        let mut information = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 0.0, 2.0]).unwrap();
        let rank = CoxFit::cholesky(&mut information, 1e-9);
        let mut score = vec![2.0, 2.0];

        CoxFit::chsolve(&information, &mut score);
        CoxFit::chinv(&mut information);

        assert_eq!(rank, 1);
        assert_eq!(score, vec![1.0, 0.0]);
        assert_eq!(information[(0, 0)], 0.5);
        assert_eq!(information[(0, 1)], 0.0);
        assert_eq!(information[(1, 0)], 0.0);
        assert_eq!(information[(1, 1)], 0.0);
    }

    #[test]
    fn singular_information_preserves_active_pivots_after_an_alias() {
        let mut information =
            Array2::from_shape_vec((3, 3), vec![2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])
                .unwrap();
        let rank = CoxFit::cholesky(&mut information, 1e-9);
        let mut score = vec![2.0, 2.0, 6.0];

        CoxFit::chsolve(&information, &mut score);
        CoxFit::chinv(&mut information);

        assert_eq!(rank, 2);
        assert_eq!(score, vec![1.0, 0.0, 2.0]);
        assert_eq!(information[(0, 0)], 0.5);
        assert_eq!(information[(1, 1)], 0.0);
        assert_eq!(information[(2, 2)], 1.0 / 3.0);
        for i in 0..3 {
            assert_eq!(information[(i, 1)], 0.0);
            assert_eq!(information[(1, i)], 0.0);
        }
    }

    #[test]
    fn information_factorization_reports_indefinite_rank() {
        let mut information = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 0.0, 1.0]).unwrap();

        let rank = CoxFit::cholesky(&mut information, 1e-9);

        assert_eq!(rank, -1);
        assert_eq!(information[(1, 1)], 0.0);
    }

    #[test]
    fn information_factorization_uses_a_strict_relative_pivot_tolerance() {
        let tolerance = 1e-9;
        let threshold = 2.0 * tolerance;
        let mut at_threshold =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, threshold]).unwrap();
        let mut below_threshold =
            Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, threshold / 2.0]).unwrap();

        assert_eq!(CoxFit::cholesky(&mut at_threshold, tolerance), 2);
        assert_eq!(CoxFit::cholesky(&mut below_threshold, tolerance), 1);
    }

    #[test]
    fn collinear_tied_efron_fit_reports_rank_and_zero_alias_variance() {
        let time = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let status = Array1::from_vec(vec![1, 1, 1, 0, 1, 1, 0, 1, 0, 1]);
        let x1 = [0.0, 0.4, 0.8, 0.2, 1.0, 1.4, 0.6, 1.2, 1.6, 1.8];
        let x2 = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97];
        let mut covariates = Vec::with_capacity(30);
        for (&first, &second) in x1.iter().zip(&x2) {
            covariates.extend_from_slice(&[first, second, first + second]);
        }
        let covar = Array2::from_shape_vec((10, 3), covariates).unwrap();
        let mut strata = Array1::zeros(10);
        strata[9] = 1;
        let mut fit = CoxFit::new(
            time,
            status,
            covar,
            strata,
            Array1::zeros(10),
            Array1::ones(10),
            Method::Efron,
            50,
            1e-9,
            1e-12,
            vec![true; 3],
            vec![0.0; 3],
        )
        .expect("collinear Efron fixture should initialize");

        fit.fit().expect("collinear Efron fixture should fit");
        let (beta, _means, _u, variance, loglik, _sctest, flag, iter) = fit.results();

        assert_eq!(flag, 2);
        assert_eq!(iter, 4);
        let expected_beta = [-2.3468678070137803, 0.5775928193386433, 0.0];
        let expected_variance = [
            [3.9806704210981683, -4.116538359266848, 0.0],
            [-4.116538359266848, 6.056737323572425, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for i in 0..3 {
            assert!((beta[i] - expected_beta[i]).abs() < 1e-10);
            for j in 0..3 {
                assert!((variance[(i, j)] - expected_variance[i][j]).abs() < 1e-10);
            }
        }
        assert!((loglik[0] - -11.079060882340368).abs() < 1e-12);
        assert!((loglik[1] - -9.002136268091796).abs() < 1e-12);
    }

    #[test]
    fn test_method_variants() {
        let breslow = Method::Breslow;
        let efron = Method::Efron;
        let exact = Method::Exact;
        assert!(matches!(breslow, Method::Breslow));
        assert!(matches!(efron, Method::Efron));
        assert!(matches!(exact, Method::Exact));
    }
}
