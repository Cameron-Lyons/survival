use crate::constants::{
    CHOLESKY_TOL, CONVERGENCE_EPSILON, CONVERGENCE_FLAG, COX_MAX_ITER, DEFAULT_MAX_ITER,
    PARALLEL_THRESHOLD_MEDIUM, RIDGE_REGULARIZATION,
};
use crate::internal::matrix::{cholesky_check, lu_solve, matrix_inverse};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::fmt;

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
            max_iter: DEFAULT_MAX_ITER,
            eps: CONVERGENCE_EPSILON,
            toler: CHOLESKY_TOL,
        }
    }
}

#[derive(Debug)]
pub(crate) enum CoxError {
    CholeskyDecomposition,
    MatrixInversion,
}

impl fmt::Display for CoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoxError::CholeskyDecomposition => write!(f, "Cholesky decomposition failed"),
            CoxError::MatrixInversion => write!(f, "Matrix inversion failed"),
        }
    }
}

impl std::error::Error for CoxError {}
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
            let prev_a = a[size - 1].clone();
            let prev_cmat = cmat[size - 1].clone();
            denom[size] += risk * base;
            for i in 0..nvar {
                let xi = covar[(person, i)];
                a[size][i] += risk * (prev_a[i] + base * xi);
                for j in 0..=i {
                    let xj = covar[(person, j)];
                    cmat[size][i * nvar + j] += risk
                        * (prev_cmat[i * nvar + j]
                            + xi * prev_a[j]
                            + xj * prev_a[i]
                            + base * xi * xj);
                }
            }
        }
    }

    let mut cmat_array = Array2::zeros((nvar, nvar));
    for i in 0..nvar {
        for j in 0..=i {
            cmat_array[(i, j)] = cmat[deaths][i * nvar + j];
        }
    }
    (denom[deaths], a[deaths].clone(), cmat_array)
}

pub(crate) struct CoxFit {
    time: Array1<f64>,
    status: Array1<i32>,
    entry_times: Option<Array1<f64>>,
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
            eps: CONVERGENCE_EPSILON,
            toler: CONVERGENCE_EPSILON,
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
        let mut cox = Self {
            time,
            status,
            entry_times,
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

            let mut start_order: Vec<usize> = (stratum_start..=stratum_end).collect();
            start_order.sort_by(|&lhs, &rhs| {
                entry_times[rhs]
                    .partial_cmp(&entry_times[lhs])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| rhs.cmp(&lhs))
            });

            let mut stop_denom = 0.0;
            let mut stop_a = vec![0.0; nvar];
            let mut stop_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut unentered_denom = 0.0;
            let mut unentered_a = vec![0.0; nvar];
            let mut unentered_cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut stop_ptr = stratum_end as isize;
            let mut start_ptr = 0usize;
            let mut time_end = stratum_end;

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
                let mut a2 = vec![0.0; nvar];
                let mut cmat2: Array2<f64> = Array2::zeros((nvar, nvar));

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
                        &mut a2,
                        &mut cmat2,
                    );
                    for i in 0..nvar {
                        self.u[i] += self.weights[person] * self.covar[(person, i)];
                    }
                }

                if ndead > 0 {
                    let denom = stop_denom - unentered_denom;
                    let mut a = vec![0.0; nvar];
                    let mut cmat: Array2<f64> = Array2::zeros((nvar, nvar));
                    for i in 0..nvar {
                        a[i] = stop_a[i] - unentered_a[i];
                        for j in 0..=i {
                            cmat[(i, j)] = stop_cmat[(i, j)] - unentered_cmat[(i, j)];
                        }
                    }
                    if matches!(method, Method::Exact) && ndead > 1 {
                        let risk_indices: Vec<usize> = (stratum_start..=stratum_end)
                            .filter(|&idx| {
                                entry_times[idx] < event_time && self.time[idx] >= event_time
                            })
                            .collect();
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
                            let temp = a[i] / denom;
                            self.u[i] -= deadwt * temp;
                            for j in 0..=i {
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
                        let mut efron_denom = denom - denom2;
                        let mut efron_a: Vec<f64> = a
                            .iter()
                            .zip(a2.iter())
                            .map(|(&all, &dead)| all - dead)
                            .collect();
                        let mut efron_cmat = cmat.clone();
                        for i in 0..nvar {
                            for j in 0..=i {
                                efron_cmat[(i, j)] -= cmat2[(i, j)];
                            }
                        }
                        for _ in 0..ndead {
                            efron_denom += risk_fraction;
                            loglik -= weight_average * efron_denom.ln();
                            for i in 0..nvar {
                                efron_a[i] += a2[i] / death_count;
                                let temp = efron_a[i] / efron_denom;
                                self.u[i] -= weight_average * temp;
                                for j in 0..=i {
                                    efron_cmat[(i, j)] += cmat2[(i, j)] / death_count;
                                    let val = weight_average
                                        * (efron_cmat[(i, j)] - temp * efron_a[j])
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
        self.flag = Self::cholesky(&mut self.imat, self.toler)?;
        Self::chsolve(&self.imat, &mut a)?;
        self.sctest = a.iter().zip(&self.u).map(|(ai, ui)| ai * ui).sum();
        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            Self::chinv(&mut self.imat)?;
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
            if !_notfinite && ((self.loglik[1] - newlk).abs() / newlk.abs() <= self.eps) {
                self.loglik[1] = newlk;
                Self::chinv(&mut self.imat)?;
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
                Self::chsolve(&self.imat, &mut a)?;
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
        Self::chinv(&mut self.imat)?;
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
    fn cholesky(mat: &mut Array2<f64>, toler: f64) -> Result<i32, CoxError> {
        let n = mat.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                mat[(i, j)] = mat[(j, i)];
            }
        }
        if cholesky_check(mat) {
            Ok(n as i32)
        } else {
            for i in 0..n {
                if mat[(i, i)] < toler {
                    return Ok(i as i32);
                }
            }
            Err(CoxError::CholeskyDecomposition)
        }
    }
    fn chsolve(chol: &Array2<f64>, a: &mut [f64]) -> Result<(), CoxError> {
        let b = Array1::from_iter(a.iter().copied());
        let result = lu_solve(chol, &b).ok_or(CoxError::CholeskyDecomposition)?;
        a.copy_from_slice(result.as_slice().unwrap_or(&[]));
        Ok(())
    }
    fn chinv(mat: &mut Array2<f64>) -> Result<(), CoxError> {
        let mut mat_reg = mat.clone();
        mat_reg
            .diag_mut()
            .iter_mut()
            .for_each(|d| *d += RIDGE_REGULARIZATION);
        let inv = matrix_inverse(&mat_reg).ok_or(CoxError::MatrixInversion)?;
        *mat = inv;
        Ok(())
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

    #[test]
    fn test_cox_fit_config_default() {
        let config = CoxFitConfig::default();

        assert!(matches!(config.method, Method::Breslow));
        assert!(config.max_iter > 0);
        assert!(config.eps > 0.0);
        assert!(config.toler > 0.0);
    }

    #[test]
    fn test_cox_fit_builder_basic() {
        let time = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let status = Array1::from_vec(vec![1, 0, 1, 0, 1]);
        let covar = Array2::from_shape_vec((5, 1), vec![0.5, 1.0, 0.3, 0.8, 0.6]).unwrap();

        let builder = CoxFitBuilder::new(time, status, covar);
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
    fn test_cox_error_display() {
        let chol_err = CoxError::CholeskyDecomposition;
        let inv_err = CoxError::MatrixInversion;

        assert_eq!(format!("{}", chol_err), "Cholesky decomposition failed");
        assert_eq!(format!("{}", inv_err), "Matrix inversion failed");
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
