use crate::constants::{
    CONVERGENCE_FLAG, COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE,
    PARALLEL_THRESHOLD_MEDIUM,
};
use crate::internal::statistical::ln_gamma;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::exact_ties::{ExactRiskAccumulator, exact_tied_moments};

const EXACT_COMPATIBILITY_DIRECT_THRESHOLD: usize = 64;

enum CoxPenalty {
    Diagonal(Vec<f64>),
    Dense(Array2<f64>),
    Frailty {
        ordinary: Array2<f64>,
        columns: Vec<usize>,
        theta: f64,
        distribution: CoxFrailtyPenalty,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum CoxFrailtyPenalty {
    Gamma,
    StudentT(f64),
}

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum FitMode {
    Standard,
    AgexactCompatibility,
}

#[derive(Clone, Copy)]
pub(crate) struct ProductAccumulator {
    contracted: bool,
}

impl ProductAccumulator {
    pub(crate) const fn new(contracted: bool) -> Self {
        Self { contracted }
    }

    #[inline]
    pub(crate) fn add(self, accumulator: f64, left: f64, right: f64) -> f64 {
        if self.contracted {
            left.mul_add(right, accumulator)
        } else {
            accumulator + left * right
        }
    }

    #[inline]
    pub(crate) fn subtract(self, accumulator: f64, left: f64, right: f64) -> f64 {
        if self.contracted {
            (-left).mul_add(right, accumulator)
        } else {
            accumulator - left * right
        }
    }
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

#[allow(clippy::too_many_arguments)]
fn add_risk_sums(
    covar: &Array2<f64>,
    nvar: usize,
    person: usize,
    risk: f64,
    denom: &mut f64,
    a: &mut [f64],
    cmat: &mut Array2<f64>,
    arithmetic: ProductAccumulator,
) {
    *denom += risk;
    for i in 0..nvar {
        let covar_i = covar[(person, i)];
        let risk_covar_i = risk * covar_i;
        a[i] = arithmetic.add(a[i], risk, covar_i);
        for j in 0..=i {
            cmat[(i, j)] = arithmetic.add(cmat[(i, j)], risk_covar_i, covar[(person, j)]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn remove_risk_sums(
    covar: &Array2<f64>,
    nvar: usize,
    person: usize,
    risk: f64,
    denom: &mut f64,
    a: &mut [f64],
    cmat: &mut Array2<f64>,
    arithmetic: ProductAccumulator,
) {
    *denom -= risk;
    for i in 0..nvar {
        let covar_i = covar[(person, i)];
        let risk_covar_i = risk * covar_i;
        a[i] = arithmetic.subtract(a[i], risk, covar_i);
        for j in 0..=i {
            cmat[(i, j)] = arithmetic.subtract(cmat[(i, j)], risk_covar_i, covar[(person, j)]);
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

fn sort_entry_order_by_input(
    order: &mut [usize],
    entry_times: &Array1<f64>,
    input_order: &[usize],
) {
    order.sort_by(|&lhs, &rhs| {
        entry_times[rhs]
            .total_cmp(&entry_times[lhs])
            .then_with(|| input_order[lhs].cmp(&input_order[rhs]))
    });
}

fn counting_used_rows(
    time: &Array1<f64>,
    status: &Array1<i32>,
    entry_times: &Array1<f64>,
    strata: &Array1<i32>,
) -> Vec<bool> {
    // Intervals that span no event time are excluded from the native fit.
    let mut used = vec![false; time.len()];
    let mut stratum_start = 0;
    for stratum_end in 0..time.len() {
        if strata[stratum_end] != 1 {
            continue;
        }
        let event_times = (stratum_start..=stratum_end)
            .filter_map(|person| (status[person] != 0).then_some(time[person]))
            .collect::<Vec<_>>();
        for person in stratum_start..=stratum_end {
            let first_event =
                event_times.partition_point(|&event_time| event_time <= entry_times[person]);
            used[person] =
                first_event < event_times.len() && event_times[first_event] <= time[person];
        }
        stratum_start = stratum_end + 1;
    }
    used
}

#[allow(clippy::too_many_arguments)]
fn apply_exact_event_moments(
    covar: &Array2<f64>,
    weights: &Array1<f64>,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    death_indices: &[usize],
    linear_predictors: &[f64],
    log_denom: f64,
    mean: &[f64],
    covariance: &Array2<f64>,
) -> f64 {
    let mut contribution = -log_denom;
    for &person in death_indices {
        contribution += weights[person] * linear_predictors[person];
        for (variable, value) in u.iter_mut().enumerate() {
            *value += weights[person] * covar[(person, variable)];
        }
    }
    for (variable, value) in u.iter_mut().enumerate() {
        *value -= mean[variable];
        for other in 0..covar.ncols() {
            imat[(variable, other)] += covariance[(variable, other)];
        }
    }
    contribution
}

#[allow(clippy::too_many_arguments)]
fn add_exact_event_contribution(
    covar: &Array2<f64>,
    weights: &Array1<f64>,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    death_indices: &[usize],
    risk_indices: &[usize],
    linear_predictors: &[f64],
    log_risk: &[f64],
) -> f64 {
    if death_indices.len() == risk_indices.len() {
        return 0.0;
    }

    let moments = exact_tied_moments(risk_indices, death_indices.len(), log_risk, covar);
    apply_exact_event_moments(
        covar,
        weights,
        u,
        imat,
        death_indices,
        linear_predictors,
        moments.log_denom,
        &moments.mean,
        &moments.covariance,
    )
}

pub(crate) struct CoxFit {
    time: Array1<f64>,
    status: Array1<i32>,
    entry_times: Option<Array1<f64>>,
    all_entered_before_first_event: bool,
    counting_roundoff_compatibility: bool,
    counting_used: Option<Vec<bool>>,
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
    penalty: CoxPenalty,
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
            false,
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
        counting_roundoff_compatibility: bool,
    ) -> Result<Self, CoxError> {
        let nvar = covar.ncols();
        let mut strata = strata;
        if let Some(last) = strata.last_mut() {
            *last = 1;
        }
        let all_entered_before_first_event = entry_times.as_ref().is_some_and(|entries| {
            let first_event_time = time
                .iter()
                .zip(&status)
                .filter_map(|(&event_time, &event_status)| {
                    (event_status != 0).then_some(event_time)
                })
                .min_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY);
            entries.iter().all(|&entry| entry < first_event_time)
        });
        // The counting-process C routine centers on the first used row in
        // descending stop-time order before computing its global scale.
        let reverse_centering_order = counting_roundoff_compatibility && entry_times.is_some();
        let counting_used = entry_times
            .as_ref()
            .map(|entries| counting_used_rows(&time, &status, entries, &strata));
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
            all_entered_before_first_event,
            counting_roundoff_compatibility,
            counting_used,
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
            penalty: CoxPenalty::Diagonal(vec![0.0; nvar]),
            means: vec![0.0; nvar],
            beta: initial_beta,
            u: vec![0.0; nvar],
            imat: Array2::zeros((nvar, nvar)),
            loglik: [0.0; 2],
            sctest: 0.0,
            flag: 0,
            iter: 0,
        };
        cox.scale_center(doscale, reverse_centering_order)?;
        Ok(cox)
    }

    pub(crate) fn set_ridge_penalty(&mut self, penalty: &[f64]) {
        debug_assert_eq!(penalty.len(), self.scale.len());
        self.penalty = CoxPenalty::Diagonal(
            penalty
                .iter()
                .zip(&self.scale)
                .map(|(&value, &scale)| value * scale * scale)
                .collect(),
        );
    }

    pub(crate) fn set_quadratic_penalty(&mut self, penalty: &Array2<f64>) {
        debug_assert_eq!(penalty.dim(), (self.scale.len(), self.scale.len()));
        self.penalty = CoxPenalty::Dense(Array2::from_shape_fn(penalty.dim(), |(row, column)| {
            penalty[(row, column)] * self.scale[row] * self.scale[column]
        }));
    }

    /// Restores stable original-row ordering for equal entry times after stop-time sorting.
    pub(crate) fn preserve_entry_input_order(&mut self, input_order: &[usize]) {
        let (Some(entry_times), Some(entry_order)) =
            (self.entry_times.as_ref(), self.entry_order.as_mut())
        else {
            return;
        };
        debug_assert_eq!(input_order.len(), entry_times.len());
        let mut stratum_start = 0;
        for stratum_end in 0..self.strata.len() {
            if self.strata[stratum_end] != 1 {
                continue;
            }
            sort_entry_order_by_input(
                &mut entry_order[stratum_start..=stratum_end],
                entry_times,
                input_order,
            );
            stratum_start = stratum_end + 1;
        }
    }

    pub(crate) fn set_frailty_penalty(
        &mut self,
        ordinary: &Array2<f64>,
        columns: Vec<usize>,
        theta: f64,
        distribution: CoxFrailtyPenalty,
    ) {
        debug_assert_eq!(ordinary.dim(), (self.scale.len(), self.scale.len()));
        self.penalty = CoxPenalty::Frailty {
            ordinary: Array2::from_shape_fn(ordinary.dim(), |(row, column)| {
                ordinary[(row, column)] * self.scale[row] * self.scale[column]
            }),
            columns,
            theta,
            distribution,
        };
    }

    fn recenter_penalty(&self, beta: &mut [f64]) {
        let CoxPenalty::Frailty {
            columns,
            theta,
            distribution,
            ..
        } = &self.penalty
        else {
            return;
        };
        if columns.is_empty() {
            return;
        }
        let center = match distribution {
            CoxFrailtyPenalty::Gamma => {
                let maximum = columns
                    .iter()
                    .map(|&column| beta[column])
                    .fold(f64::NEG_INFINITY, f64::max);
                maximum
                    + (columns
                        .iter()
                        .map(|&column| (beta[column] - maximum).exp())
                        .sum::<f64>()
                        / columns.len() as f64)
                        .ln()
            }
            CoxFrailtyPenalty::StudentT(degrees_of_freedom) => {
                let denominator = *theta * (*degrees_of_freedom - 2.0);
                let (first_sum, second_sum) =
                    columns
                        .iter()
                        .fold((0.0, 0.0), |(first_sum, second_sum), &column| {
                            let value = beta[column];
                            let scaled_square = value * value / denominator;
                            let temp = 1.0 + scaled_square;
                            (
                                first_sum + value / temp,
                                second_sum + 1.0 / temp - 2.0 * scaled_square / (temp * temp),
                            )
                        });
                first_sum / second_sum
            }
        };
        for &column in columns {
            beta[column] -= center;
        }
    }

    fn penalty_value(&self, beta: &[f64]) -> f64 {
        match &self.penalty {
            CoxPenalty::Diagonal(values) => {
                0.5 * beta
                    .iter()
                    .zip(values)
                    .map(|(&coefficient, &value)| value * coefficient * coefficient)
                    .sum::<f64>()
            }
            CoxPenalty::Dense(matrix) => {
                0.5 * beta
                    .iter()
                    .enumerate()
                    .map(|(row, &coefficient)| {
                        coefficient
                            * beta
                                .iter()
                                .enumerate()
                                .map(|(column, &other)| matrix[(row, column)] * other)
                                .sum::<f64>()
                    })
                    .sum::<f64>()
            }
            CoxPenalty::Frailty {
                ordinary,
                columns,
                theta,
                distribution,
            } => {
                let ordinary_value = 0.5
                    * beta
                        .iter()
                        .enumerate()
                        .map(|(row, &coefficient)| {
                            coefficient
                                * beta
                                    .iter()
                                    .enumerate()
                                    .map(|(column, &other)| ordinary[(row, column)] * other)
                                    .sum::<f64>()
                        })
                        .sum::<f64>();
                let frailty_value = match distribution {
                    CoxFrailtyPenalty::Gamma => {
                        -columns.iter().map(|&column| beta[column]).sum::<f64>() / theta
                    }
                    CoxFrailtyPenalty::StudentT(degrees_of_freedom) => {
                        let denominator = theta * (degrees_of_freedom - 2.0);
                        let constant = 0.5 * (std::f64::consts::PI * denominator).ln()
                            + ln_gamma(degrees_of_freedom / 2.0)
                            - ln_gamma((degrees_of_freedom + 1.0) / 2.0);
                        columns
                            .iter()
                            .map(|&column| {
                                constant
                                    + 0.5
                                        * (degrees_of_freedom + 1.0)
                                        * (1.0 + beta[column] * beta[column] / denominator).ln()
                            })
                            .sum::<f64>()
                    }
                };
                ordinary_value + frailty_value
            }
        }
    }

    pub(crate) fn penalty_hessian(&self) -> Array2<f64> {
        let width = self.beta.len();
        match &self.penalty {
            CoxPenalty::Diagonal(values) => {
                Array2::from_shape_fn((width, width), |(row, column)| {
                    if row == column { values[row] } else { 0.0 }
                })
            }
            CoxPenalty::Dense(matrix) => matrix.clone(),
            CoxPenalty::Frailty {
                ordinary,
                columns,
                theta,
                distribution,
            } => {
                let mut result = ordinary.clone();
                for &column in columns {
                    let second = match distribution {
                        CoxFrailtyPenalty::Gamma => self.beta[column].exp() / theta,
                        CoxFrailtyPenalty::StudentT(degrees_of_freedom) => {
                            let denominator = theta * (degrees_of_freedom - 2.0);
                            let scaled_square = self.beta[column] * self.beta[column] / denominator;
                            let temp = 1.0 + scaled_square;
                            (degrees_of_freedom + 1.0) / denominator
                                * (1.0 / temp - 2.0 * scaled_square / (temp * temp))
                        }
                    };
                    result[(column, column)] += second;
                }
                result
            }
        }
    }

    pub(crate) fn penalized_log_likelihood(&self) -> f64 {
        self.loglik[1]
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
        counting_roundoff_compatibility: bool,
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
            counting_roundoff_compatibility,
        )
    }
    #[allow(clippy::undocumented_unsafe_blocks)]
    fn scale_center(&mut self, doscale: Vec<bool>, reverse_order: bool) -> Result<(), CoxError> {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let scale_rows = reverse_order.then(|| {
            self.counting_used
                .as_deref()
                .map(|used| {
                    used.iter()
                        .enumerate()
                        .filter_map(|(person, &included)| included.then_some(person))
                        .collect::<Vec<_>>()
                })
                .filter(|rows| !rows.is_empty())
                .unwrap_or_else(|| (0..nused).collect())
        });
        let total_weight = if let Some(rows) = scale_rows.as_deref() {
            rows.iter().rev().map(|&person| self.weights[person]).sum()
        } else {
            self.weights.sum()
        };
        let means: Vec<f64> = (0..nvar)
            .into_par_iter()
            .map(|i| {
                if !doscale[i] {
                    0.0
                } else if let Some(rows) = scale_rows.as_deref() {
                    self.covar[(*rows.last().expect("scaling rows exist"), i)]
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
                    let abs_sum = if let Some(rows) = scale_rows.as_deref() {
                        rows.iter().rev().fold(0.0, |sum, &person| {
                            sum + self.weights[person] * (self.covar[(person, i)] - mean).abs()
                        })
                    } else {
                        (0..nused).fold(0.0, |sum, person| {
                            sum + self.weights[person] * (self.covar[(person, i)] - mean).abs()
                        })
                    };
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
                    if let Some(rows) = scale_rows.as_deref() {
                        for &person in rows {
                            unsafe {
                                let offset = person as isize * row_stride + i as isize * col_stride;
                                let ptr = base_ptr.offset(offset);
                                *ptr = (*ptr - mean) * scale_val;
                            }
                        }
                    } else {
                        for person in 0..nused {
                            unsafe {
                                let offset = person as isize * row_stride + i as isize * col_stride;
                                let ptr = base_ptr.offset(offset);
                                *ptr = (*ptr - mean) * scale_val;
                            }
                        }
                    }
                }
            });
        } else {
            for i in 0..nvar {
                if doscale[i] {
                    let mean = means[i];
                    let scale_val = scales[i];
                    if let Some(rows) = scale_rows.as_deref() {
                        for &person in rows {
                            self.covar[(person, i)] = (self.covar[(person, i)] - mean) * scale_val;
                        }
                    } else {
                        for person in 0..nused {
                            self.covar[(person, i)] = (self.covar[(person, i)] - mean) * scale_val;
                        }
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

    fn exact_predictors(&self, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let evaluate = |person: usize| {
            let linear_predictor = self.offset[person]
                + beta
                    .iter()
                    .enumerate()
                    .fold(0.0, |sum, (variable, &coefficient)| {
                        sum + coefficient * self.covar[(person, variable)]
                    });
            (
                linear_predictor,
                linear_predictor + self.weights[person].ln(),
            )
        };
        if self.covar.nrows() > PARALLEL_THRESHOLD_MEDIUM {
            (0..self.covar.nrows())
                .into_par_iter()
                .map(evaluate)
                .unzip()
        } else {
            (0..self.covar.nrows()).map(evaluate).unzip()
        }
    }

    fn iterate_right_censored_exact(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let mut loglik = 0.0;
        let mut stratum_start = 0usize;

        for stratum_end in 0..self.covar.nrows() {
            if self.strata[stratum_end] != 1 {
                continue;
            }

            let mut risk_indices = Vec::with_capacity(stratum_end - stratum_start + 1);
            let mut death_indices = Vec::new();
            let mut singleton_moments = ExactRiskAccumulator::new(self.covar.ncols());
            let mut time_end = stratum_end;
            loop {
                let event_time = self.time[time_end];
                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }
                risk_indices.extend(time_start..=time_end);
                for (offset, &log_weight) in log_risk[time_start..=time_end].iter().enumerate() {
                    singleton_moments.add(time_start + offset, log_weight, &self.covar);
                }
                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&person| self.status[person] != 0));
                if !death_indices.is_empty() {
                    loglik += if death_indices.len() == 1 && risk_indices.len() > 1 {
                        apply_exact_event_moments(
                            &self.covar,
                            &self.weights,
                            &mut self.u,
                            &mut self.imat,
                            &death_indices,
                            &linear_predictors,
                            singleton_moments.log_denom,
                            &singleton_moments.mean,
                            &singleton_moments.covariance,
                        )
                    } else {
                        add_exact_event_contribution(
                            &self.covar,
                            &self.weights,
                            &mut self.u,
                            &mut self.imat,
                            &death_indices,
                            &risk_indices,
                            &linear_predictors,
                            &log_risk,
                        )
                    };
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

    fn iterate_counting_process_exact(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored_exact(beta);
        };
        let entry_order = self
            .entry_order
            .as_ref()
            .expect("entry order must accompany counting-process entry times");
        let nvar = self.covar.ncols();
        let arithmetic = ProductAccumulator::new(false);
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let raw_risk: Vec<f64> = log_risk.iter().map(|value| value.exp()).collect();
        let mut loglik = 0.0;
        let mut stratum_start = 0usize;

        for stratum_end in 0..self.covar.nrows() {
            if self.strata[stratum_end] != 1 {
                continue;
            }

            let start_order = &entry_order[stratum_start..=stratum_end];
            let mut stop_denom = 0.0;
            let mut stop_a = vec![0.0; nvar];
            let mut stop_cmat = Array2::zeros((nvar, nvar));
            let mut unentered_denom = 0.0;
            let mut unentered_a = vec![0.0; nvar];
            let mut unentered_cmat = Array2::zeros((nvar, nvar));
            let mut stop_count = 0usize;
            let mut unentered_count = 0usize;
            let mut stop_ptr = stratum_end as isize;
            let mut start_ptr = 0usize;
            let mut death_indices = Vec::new();
            let mut risk_indices = Vec::new();
            let mut mean = vec![0.0; nvar];
            let mut covariance = Array2::zeros((nvar, nvar));
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
                        raw_risk[person],
                        &mut stop_denom,
                        &mut stop_a,
                        &mut stop_cmat,
                        arithmetic,
                    );
                    stop_count += 1;
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
                        raw_risk[person],
                        &mut unentered_denom,
                        &mut unentered_a,
                        &mut unentered_cmat,
                        arithmetic,
                    );
                    unentered_count += 1;
                    start_ptr += 1;
                }

                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }
                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&person| self.status[person] != 0));
                if !death_indices.is_empty() {
                    let active_count = stop_count - unentered_count;
                    if death_indices.len() != active_count {
                        let denom = stop_denom - unentered_denom;
                        let cancellation_scale = stop_denom.abs() + unentered_denom.abs();
                        let mut reliable_singleton = death_indices.len() == 1
                            && denom.is_finite()
                            && denom > 0.0
                            && (cancellation_scale == 0.0
                                || denom > 64.0 * f64::EPSILON * cancellation_scale);

                        if reliable_singleton {
                            for variable in 0..nvar {
                                let active_sum = stop_a[variable] - unentered_a[variable];
                                let cancellation_scale =
                                    stop_a[variable].abs() + unentered_a[variable].abs();
                                if cancellation_scale != 0.0
                                    && active_sum.abs() <= 64.0 * f64::EPSILON * cancellation_scale
                                {
                                    reliable_singleton = false;
                                }
                                mean[variable] = active_sum / denom;
                            }
                            for row in 0..nvar {
                                for column in 0..=row {
                                    let active_sum =
                                        stop_cmat[(row, column)] - unentered_cmat[(row, column)];
                                    let cancellation_scale = stop_cmat[(row, column)].abs()
                                        + unentered_cmat[(row, column)].abs();
                                    if cancellation_scale != 0.0
                                        && active_sum.abs()
                                            <= 64.0 * f64::EPSILON * cancellation_scale
                                    {
                                        reliable_singleton = false;
                                    }
                                    let active_first_moment = stop_a[column] - unentered_a[column];
                                    let mut value =
                                        (active_sum - mean[row] * active_first_moment) / denom;
                                    if row == column && value < 0.0 {
                                        if value
                                            >= -64.0 * f64::EPSILON * (active_sum / denom).abs()
                                        {
                                            value = 0.0;
                                        } else {
                                            reliable_singleton = false;
                                        }
                                    }
                                    covariance[(row, column)] = value;
                                    covariance[(column, row)] = value;
                                    reliable_singleton &= value.is_finite();
                                }
                            }
                        }

                        loglik += if reliable_singleton {
                            apply_exact_event_moments(
                                &self.covar,
                                &self.weights,
                                &mut self.u,
                                &mut self.imat,
                                &death_indices,
                                &linear_predictors,
                                denom.ln(),
                                &mean,
                                &covariance,
                            )
                        } else {
                            risk_indices.clear();
                            risk_indices.extend((stratum_start..=stratum_end).filter(|&person| {
                                entry_times[person] < event_time && self.time[person] >= event_time
                            }));
                            add_exact_event_contribution(
                                &self.covar,
                                &self.weights,
                                &mut self.u,
                                &mut self.imat,
                                &death_indices,
                                &risk_indices,
                                &linear_predictors,
                                &log_risk,
                            )
                        };
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

    fn iterate_counting_process_exact_compatibility(
        &mut self,
        beta: &[f64],
    ) -> Result<f64, CoxError> {
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored_exact(beta);
        };
        self.u.fill(0.0);
        self.imat.fill(0.0);
        let (linear_predictors, log_risk) = self.exact_predictors(beta);
        let mut loglik = 0.0;
        let mut stratum_start = 0usize;

        for stratum_end in 0..self.covar.nrows() {
            if self.strata[stratum_end] != 1 {
                continue;
            }

            let mut death_indices = Vec::new();
            let mut risk_indices = Vec::new();
            let mut time_end = stratum_end;
            loop {
                let event_time = self.time[time_end];
                let mut time_start = time_end;
                while time_start > stratum_start && self.time[time_start - 1] == event_time {
                    time_start -= 1;
                }
                death_indices.clear();
                death_indices
                    .extend((time_start..=time_end).filter(|&person| self.status[person] != 0));
                if !death_indices.is_empty() {
                    risk_indices.clear();
                    risk_indices.extend((stratum_start..=stratum_end).filter(|&person| {
                        entry_times[person] < event_time && self.time[person] >= event_time
                    }));
                    loglik += add_exact_event_contribution(
                        &self.covar,
                        &self.weights,
                        &mut self.u,
                        &mut self.imat,
                        &death_indices,
                        &risk_indices,
                        &linear_predictors,
                        &log_risk,
                    );
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

    fn iterate_right_censored(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        if matches!(self.method, Method::Exact) {
            return self.iterate_right_censored_exact(beta);
        }
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
                if matches!(method, Method::Breslow) || ndead == 1 {
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
        if matches!(self.method, Method::Exact) {
            return self.iterate_counting_process_exact(beta);
        }
        let Some(entry_times) = self.entry_times.as_ref() else {
            return self.iterate_right_censored(beta);
        };
        // For one covariate with no delayed entry, the simpler recurrence also
        // preserves the scalar reference path used by the CCH calculation.
        if self.all_entered_before_first_event
            && !(self.counting_roundoff_compatibility && self.covar.ncols() > 1)
        {
            return self.iterate_right_censored(beta);
        }
        let entry_order = self
            .entry_order
            .as_deref()
            .expect("entry order must accompany counting-process entry times");
        let used = self
            .counting_used
            .as_deref()
            .expect("counting-process rows must have a usage mask");
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;
        let arithmetic = ProductAccumulator::new(self.counting_roundoff_compatibility);
        self.u.fill(0.0);
        self.imat.fill(0.0);

        // Match the contracted dot product used by the counting-process C fit.
        let zbeta_vals: Vec<f64> = if nused > PARALLEL_THRESHOLD_MEDIUM {
            (0..nused)
                .into_par_iter()
                .map(|person| {
                    beta.iter().enumerate().fold(0.0, |acc, (i, &b)| {
                        arithmetic.add(acc, b, self.covar[(person, i)])
                    }) + self.offset[person]
                })
                .collect()
        } else {
            (0..nused)
                .map(|person| {
                    beta.iter().enumerate().fold(0.0, |acc, (i, &b)| {
                        arithmetic.add(acc, b, self.covar[(person, i)])
                    }) + self.offset[person]
                })
                .collect()
        };

        let mut loglik = 0.0;
        let mut stratum_start = 0usize;
        for stratum_end in 0..nused {
            if self.strata[stratum_end] != 1 {
                continue;
            }

            let start_order = &entry_order[stratum_start..=stratum_end];
            // Maintain active sums in descending event-time order to avoid
            // subtracting two large cumulative totals at every event.
            let mut denom = 0.0;
            let mut a = vec![0.0; nvar];
            let mut cmat: Array2<f64> = Array2::zeros((nvar, nvar));
            let mut stop_ptr = stratum_end as isize;
            let mut start_ptr = 0usize;
            let mut nrisk = 0usize;
            let mut eta_sum = 0.0;
            let mut recenter = 0.0;
            let mut death_a = vec![0.0; nvar];
            let mut death_cmat: Array2<f64> = Array2::zeros((nvar, nvar));

            loop {
                let mut event_ptr = stop_ptr;
                while event_ptr >= stratum_start as isize && self.status[event_ptr as usize] == 0 {
                    event_ptr -= 1;
                }
                if event_ptr < stratum_start as isize {
                    break;
                }
                let event_time = self.time[event_ptr as usize];

                while start_ptr < start_order.len()
                    && entry_times[start_order[start_ptr]] >= event_time
                {
                    let person = start_order[start_ptr];
                    if used[person] {
                        debug_assert!(nrisk > 0);
                        nrisk -= 1;
                        if nrisk == 0 {
                            eta_sum = 0.0;
                            denom = 0.0;
                            a.fill(0.0);
                            cmat.fill(0.0);
                        } else {
                            eta_sum -= zbeta_vals[person];
                            let risk = (zbeta_vals[person] - recenter).exp() * self.weights[person];
                            remove_risk_sums(
                                &self.covar,
                                nvar,
                                person,
                                risk,
                                &mut denom,
                                &mut a,
                                &mut cmat,
                                arithmetic,
                            );
                        }
                    }
                    start_ptr += 1;
                }

                let mut ndead = 0usize;
                let mut deadwt = 0.0;
                let mut denom2 = 0.0;
                death_a.fill(0.0);
                death_cmat.fill(0.0);

                while stop_ptr >= stratum_start as isize
                    && self.time[stop_ptr as usize] >= event_time
                {
                    let person = stop_ptr as usize;
                    stop_ptr -= 1;
                    if !used[person] {
                        continue;
                    }
                    nrisk += 1;
                    eta_sum += zbeta_vals[person];
                    let eta_shift = eta_sum / nrisk as f64 - recenter;
                    if eta_shift.abs() > 200.0 {
                        recenter = eta_sum / nrisk as f64;
                        if denom > 0.0 {
                            let scale = (-eta_shift).exp();
                            denom *= scale;
                            for i in 0..nvar {
                                a[i] *= scale;
                                for j in 0..=i {
                                    cmat[(i, j)] *= scale;
                                }
                            }
                        }
                    }
                    let risk = (zbeta_vals[person] - recenter).exp() * self.weights[person];
                    if self.status[person] == 0 {
                        add_risk_sums(
                            &self.covar,
                            nvar,
                            person,
                            risk,
                            &mut denom,
                            &mut a,
                            &mut cmat,
                            arithmetic,
                        );
                    } else {
                        ndead += 1;
                        deadwt += self.weights[person];
                        loglik = arithmetic.add(
                            loglik,
                            self.weights[person],
                            zbeta_vals[person] - recenter,
                        );
                        add_risk_sums(
                            &self.covar,
                            nvar,
                            person,
                            risk,
                            &mut denom2,
                            &mut death_a,
                            &mut death_cmat,
                            arithmetic,
                        );
                        for i in 0..nvar {
                            self.u[i] = arithmetic.add(
                                self.u[i],
                                self.weights[person],
                                self.covar[(person, i)],
                            );
                        }
                    }
                }

                debug_assert!(ndead > 0);
                if matches!(method, Method::Breslow) || ndead == 1 {
                    denom += denom2;
                    loglik = arithmetic.subtract(loglik, deadwt, denom.ln());
                    for i in 0..nvar {
                        a[i] += death_a[i];
                        let temp = a[i] / denom;
                        self.u[i] = arithmetic.subtract(self.u[i], deadwt, temp);
                        for j in 0..=i {
                            cmat[(i, j)] += death_cmat[(i, j)];
                            let centered = arithmetic.subtract(cmat[(i, j)], temp, a[j]) / denom;
                            let updated = arithmetic.add(self.imat[(j, i)], deadwt, centered);
                            self.imat[(j, i)] = updated;
                            if i != j {
                                self.imat[(i, j)] = updated;
                            }
                        }
                    }
                } else {
                    let death_count = ndead as f64;
                    let weight_average = deadwt / death_count;
                    for _ in 0..ndead {
                        denom += denom2 / death_count;
                        loglik = arithmetic.subtract(loglik, weight_average, denom.ln());
                        for i in 0..nvar {
                            a[i] += death_a[i] / death_count;
                            let temp = a[i] / denom;
                            self.u[i] = arithmetic.subtract(self.u[i], weight_average, temp);
                            for j in 0..=i {
                                cmat[(i, j)] += death_cmat[(i, j)] / death_count;
                                let centered =
                                    arithmetic.subtract(cmat[(i, j)], temp, a[j]) / denom;
                                let updated =
                                    arithmetic.add(self.imat[(j, i)], weight_average, centered);
                                self.imat[(j, i)] = updated;
                                if i != j {
                                    self.imat[(i, j)] = updated;
                                }
                            }
                        }
                    }
                }
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

    fn iterate_with_mode(&mut self, beta: &[f64], mode: FitMode) -> Result<f64, CoxError> {
        let log_likelihood = if mode == FitMode::AgexactCompatibility
            && self.entry_times.is_some()
            && matches!(self.method, Method::Exact)
            && self.covar.nrows() <= EXACT_COMPATIBILITY_DIRECT_THRESHOLD
        {
            self.iterate_counting_process_exact_compatibility(beta)
        } else {
            self.iterate(beta)
        }?;
        let penalty = match &self.penalty {
            CoxPenalty::Diagonal(values) => {
                let mut penalty = 0.0;
                for (variable, (&coefficient, &diagonal)) in beta.iter().zip(values).enumerate() {
                    self.u[variable] -= diagonal * coefficient;
                    self.imat[(variable, variable)] += diagonal;
                    penalty += diagonal * coefficient * coefficient;
                }
                penalty
            }
            CoxPenalty::Dense(matrix) => {
                let mut penalty = 0.0;
                for (row, &coefficient) in beta.iter().enumerate() {
                    let penalty_score = beta
                        .iter()
                        .enumerate()
                        .map(|(column, &other)| matrix[(row, column)] * other)
                        .sum::<f64>();
                    self.u[row] -= penalty_score;
                    penalty += coefficient * penalty_score;
                    for column in 0..beta.len() {
                        self.imat[(row, column)] += matrix[(row, column)];
                    }
                }
                penalty
            }
            CoxPenalty::Frailty {
                ordinary,
                columns,
                theta,
                distribution,
            } => {
                let mut penalty = 0.0;
                for (row, &coefficient) in beta.iter().enumerate() {
                    let penalty_score = beta
                        .iter()
                        .enumerate()
                        .map(|(column, &other)| ordinary[(row, column)] * other)
                        .sum::<f64>();
                    self.u[row] -= penalty_score;
                    penalty += 0.5 * coefficient * penalty_score;
                    for column in 0..beta.len() {
                        self.imat[(row, column)] += ordinary[(row, column)];
                    }
                }
                match distribution {
                    CoxFrailtyPenalty::Gamma => {
                        for &column in columns {
                            let relative_risk = beta[column].exp();
                            self.u[column] -= (relative_risk - 1.0) / theta;
                            self.imat[(column, column)] += relative_risk / theta;
                            penalty -= beta[column] / theta;
                        }
                    }
                    CoxFrailtyPenalty::StudentT(degrees_of_freedom) => {
                        let denominator = theta * (degrees_of_freedom - 2.0);
                        let scale = (degrees_of_freedom + 1.0) / denominator;
                        let constant = 0.5 * (std::f64::consts::PI * denominator).ln()
                            + ln_gamma(degrees_of_freedom / 2.0)
                            - ln_gamma((degrees_of_freedom + 1.0) / 2.0);
                        for &column in columns {
                            let value = beta[column];
                            let scaled_square = value * value / denominator;
                            let temp = 1.0 + scaled_square;
                            self.u[column] -= scale * value / temp;
                            self.imat[(column, column)] +=
                                scale * (1.0 / temp - 2.0 * scaled_square / (temp * temp));
                            penalty += constant + 0.5 * (degrees_of_freedom + 1.0) * temp.ln();
                        }
                    }
                }
                penalty
            }
        };
        Ok(log_likelihood
            - if matches!(self.penalty, CoxPenalty::Frailty { .. }) {
                penalty
            } else {
                0.5 * penalty
            })
    }

    pub(crate) fn fit(&mut self) -> Result<(), CoxError> {
        self.fit_with_mode(FitMode::Standard)
    }

    pub(crate) fn fit_agexact_compatibility(&mut self) -> Result<(), CoxError> {
        self.fit_with_mode(FitMode::AgexactCompatibility)
    }

    fn fit_with_mode(&mut self, mode: FitMode) -> Result<(), CoxError> {
        let agexact_compatibility = mode == FitMode::AgexactCompatibility;
        let factor_arithmetic = ProductAccumulator::new(self.counting_roundoff_compatibility);
        let nvar = self.beta.len();
        let mut newbeta = vec![0.0; nvar];
        let mut a = vec![0.0; nvar];
        let mut halving = 0;
        let mut _notfinite;
        let mut beta_copy = self.beta.clone();
        self.recenter_penalty(&mut beta_copy);
        self.beta.copy_from_slice(&beta_copy);
        self.loglik[0] = self.iterate_with_mode(&beta_copy, mode)?;
        self.loglik[1] = self.loglik[0];
        if nvar == 0 {
            self.flag = 0;
            return Ok(());
        }
        a.copy_from_slice(&self.u);
        self.flag = Self::cholesky(&mut self.imat, self.toler, factor_arithmetic);
        Self::chsolve(&self.imat, &mut a, factor_arithmetic);
        self.sctest = a
            .iter()
            .zip(&self.u)
            .fold(0.0, |sum, (&ai, &ui)| factor_arithmetic.add(sum, ai, ui));
        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            Self::chinv(&mut self.imat, factor_arithmetic);
            self.rescale_params();
            if agexact_compatibility && self.max_iter == 0 {
                self.flag = 0;
            }
            return Ok(());
        }
        newbeta.copy_from_slice(&self.beta);
        for i in 0..nvar {
            newbeta[i] += a[i];
        }
        self.loglik[1] = self.loglik[0];
        let mut newlk = self.loglik[1];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            self.recenter_penalty(&mut newbeta);
            newlk = match self.iterate_with_mode(&newbeta, mode) {
                Ok(lk) if lk.is_finite() => lk,
                _ => {
                    _notfinite = true;
                    f64::NAN
                }
            };
            self.flag = Self::cholesky(&mut self.imat, self.toler, factor_arithmetic);
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
            if !_notfinite
                && (1.0 - self.loglik[1] / newlk).abs() <= self.eps
                && (!agexact_compatibility || halving == 0)
            {
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                Self::chinv(&mut self.imat, factor_arithmetic);
                self.rescale_params();
                if !agexact_compatibility && halving > 0 {
                    self.flag = -2;
                }
                return Ok(());
            }
            if agexact_compatibility && iter == self.max_iter {
                break;
            }
            if _notfinite || newlk < self.loglik[1] {
                halving += 1;
                for (newbeta_elem, beta_elem) in newbeta.iter_mut().zip(self.beta.iter()).take(nvar)
                {
                    *newbeta_elem = if agexact_compatibility {
                        (*newbeta_elem + beta_elem) / 2.0
                    } else {
                        (*newbeta_elem + (halving as f64) * beta_elem) / (halving as f64 + 1.0)
                    };
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                a.copy_from_slice(&self.u);
                Self::chsolve(&self.imat, &mut a, factor_arithmetic);
                for (newbeta_elem, (beta_elem, a_elem)) in newbeta
                    .iter_mut()
                    .zip(self.beta.iter().zip(a.iter()))
                    .take(nvar)
                {
                    *newbeta_elem = beta_elem + a_elem;
                }
            }
        }
        if agexact_compatibility {
            self.loglik[1] = newlk;
            self.beta.copy_from_slice(&newbeta);
            Self::chinv(&mut self.imat, factor_arithmetic);
            self.rescale_params();
            self.flag = CONVERGENCE_FLAG;
            return Ok(());
        }
        let mut beta_final = self.beta.clone();
        self.recenter_penalty(&mut beta_final);
        self.beta.copy_from_slice(&beta_final);
        self.loglik[1] = self.iterate_with_mode(&beta_final, mode)?;
        self.flag = Self::cholesky(&mut self.imat, self.toler, factor_arithmetic);
        Self::chinv(&mut self.imat, factor_arithmetic);
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
        match &mut self.penalty {
            CoxPenalty::Diagonal(values) => {
                for (value, &scale) in values.iter_mut().zip(&self.scale) {
                    *value /= scale * scale;
                }
            }
            CoxPenalty::Dense(matrix) => {
                for ((row, column), value) in matrix.indexed_iter_mut() {
                    *value /= self.scale[row] * self.scale[column];
                }
            }
            CoxPenalty::Frailty { ordinary, .. } => {
                for ((row, column), value) in ordinary.indexed_iter_mut() {
                    *value /= self.scale[row] * self.scale[column];
                }
            }
        }
    }
    fn cholesky(mat: &mut Array2<f64>, toler: f64, arithmetic: ProductAccumulator) -> i32 {
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
                mat[(j, j)] = arithmetic.subtract(mat[(j, j)], temp * temp, pivot);
                for k in (j + 1)..n {
                    mat[(k, j)] = arithmetic.subtract(mat[(k, j)], temp, mat[(k, i)]);
                }
            }
        }
        rank * nonnegative
    }
    fn chsolve(chol: &Array2<f64>, a: &mut [f64], arithmetic: ProductAccumulator) {
        for i in 0..a.len() {
            let mut temp = a[i];
            for j in 0..i {
                temp = arithmetic.subtract(temp, a[j], chol[(i, j)]);
            }
            a[i] = temp;
        }
        for i in (0..a.len()).rev() {
            if chol[(i, i)] == 0.0 {
                a[i] = 0.0;
            } else {
                let mut temp = a[i] / chol[(i, i)];
                for j in (i + 1)..a.len() {
                    temp = arithmetic.subtract(temp, a[j], chol[(j, i)]);
                }
                a[i] = temp;
            }
        }
    }
    fn chinv(mat: &mut Array2<f64>, arithmetic: ProductAccumulator) {
        let n = mat.nrows();
        for i in 0..n {
            if mat[(i, i)] > 0.0 {
                mat[(i, i)] = 1.0 / mat[(i, i)];
                for j in (i + 1)..n {
                    mat[(j, i)] = -mat[(j, i)];
                    for k in 0..i {
                        mat[(j, k)] = arithmetic.add(mat[(j, k)], mat[(j, i)], mat[(i, k)]);
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
                        mat[(i, k)] = arithmetic.add(mat[(i, k)], temp, mat[(j, k)]);
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
        let mut log_likelihood = self.loglik;
        log_likelihood[1] += self.penalty_value(&self.beta);
        (
            self.beta,
            self.means,
            self.u,
            self.imat,
            log_likelihood,
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
            false,
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
    fn exact_builder_treats_default_strata_as_one_complete_stratum() {
        let mut fit = CoxFitBuilder::new(
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![1, 1, 0]),
            Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
        )
        .method(Method::Exact)
        .max_iter(0)
        .build()
        .expect("default-stratum exact fit should initialize");

        fit.fit()
            .expect("default-stratum exact fit should evaluate");
        let (_beta, _means, score, variance, loglik, ..) = fit.results();

        assert!(loglik[0] < 0.0);
        assert!(score[0].is_finite());
        assert!(variance[(0, 0)].is_finite());
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
                false,
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
    fn exact_counting_process_falls_back_when_moment_subtraction_cancels() {
        let mut fit = CoxFit::new_with_entry_times(
            Array1::from_vec(vec![1.0, 2.0, 2.0]),
            Array1::from_vec(vec![1, 0, 0]),
            Array2::from_shape_vec((3, 1), vec![1.0, 1.0, 1e16]).unwrap(),
            Some(Array1::from_vec(vec![0.0, 0.0, 1.0])),
            Array1::from_vec(vec![0, 0, 1]),
            Array1::zeros(3),
            Array1::ones(3),
            Method::Exact,
            0,
            1e-9,
            1e-9,
            vec![false],
            vec![0.0],
            false,
        )
        .expect("cancellation fixture should initialize");

        let loglik = fit
            .iterate(&[0.0])
            .expect("exact cancellation fixture should evaluate");

        assert!((loglik + 2.0_f64.ln()).abs() < 1e-14);
        assert_eq!(fit.u, vec![0.0]);
        assert_eq!(fit.imat[(0, 0)], 0.0);
    }

    #[test]
    fn large_compatibility_sweep_matches_sequential_exact_evaluation() {
        let n = EXACT_COMPATIBILITY_DIRECT_THRESHOLD + 1;
        let build = || {
            CoxFit::new_with_entry_times(
                Array1::from_vec((1..=n).map(|value| value as f64).collect()),
                Array1::ones(n),
                Array2::from_shape_vec(
                    (n, 1),
                    (0..n).map(|value| (value % 11) as f64 - 5.0).collect(),
                )
                .unwrap(),
                Some(Array1::zeros(n)),
                Array1::from_vec(vec![0; n]),
                Array1::from_vec((0..n).map(|value| (value % 5) as f64 / 20.0).collect()),
                Array1::ones(n),
                Method::Exact,
                0,
                1e-9,
                1e-9,
                vec![false],
                vec![0.0],
                false,
            )
            .expect("large exact comparison fixture should initialize")
        };
        let beta = [0.15];
        let mut swept = build();
        let swept_loglik = swept
            .iterate_counting_process_exact(&beta)
            .expect("swept exact evaluation should succeed");
        let mut sequential = build();
        let sequential_loglik = sequential
            .iterate_counting_process_exact_compatibility(&beta)
            .expect("sequential exact evaluation should succeed");

        assert!((swept_loglik - sequential_loglik).abs() < 1e-12);
        assert!((swept.u[0] - sequential.u[0]).abs() < 1e-12);
        assert!((swept.imat[(0, 0)] - sequential.imat[(0, 0)]).abs() < 1e-11);
    }

    #[test]
    fn information_factorization_inverts_spd_matrix() {
        let mut information = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 0.0, 3.0]).unwrap();

        let arithmetic = ProductAccumulator::new(false);
        let rank = CoxFit::cholesky(&mut information, 1e-9, arithmetic);
        CoxFit::chinv(&mut information, arithmetic);

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
        let arithmetic = ProductAccumulator::new(false);
        let rank = CoxFit::cholesky(&mut information, 1e-9, arithmetic);
        let mut score = vec![2.0, 2.0];

        CoxFit::chsolve(&information, &mut score, arithmetic);
        CoxFit::chinv(&mut information, arithmetic);

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
        let arithmetic = ProductAccumulator::new(false);
        let rank = CoxFit::cholesky(&mut information, 1e-9, arithmetic);
        let mut score = vec![2.0, 2.0, 6.0];

        CoxFit::chsolve(&information, &mut score, arithmetic);
        CoxFit::chinv(&mut information, arithmetic);

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

        let rank = CoxFit::cholesky(&mut information, 1e-9, ProductAccumulator::new(false));

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

        let arithmetic = ProductAccumulator::new(false);
        assert_eq!(
            CoxFit::cholesky(&mut at_threshold, tolerance, arithmetic),
            2
        );
        assert_eq!(
            CoxFit::cholesky(&mut below_threshold, tolerance, arithmetic),
            1
        );
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
