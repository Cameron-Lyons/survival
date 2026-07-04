use crate::internal::typed_inputs::{CovariateMatrix, CoxRegressionInput, SurvivalData, Weights};

struct FastCoxCvFold {
    train_x: Vec<f64>,
    train_time: Vec<f64>,
    train_status: Vec<i32>,
    train_wt: Vec<f64>,
    train_offset: Vec<f64>,
    test_x: Vec<f64>,
    test_time: Vec<f64>,
    test_status: Vec<i32>,
    test_wt: Vec<f64>,
    test_offset: Vec<f64>,
}

impl FastCoxCvFold {
    fn with_capacity(train_n: usize, test_n: usize, n_vars: usize) -> Self {
        Self {
            train_x: Vec::with_capacity(train_n * n_vars),
            train_time: Vec::with_capacity(train_n),
            train_status: Vec::with_capacity(train_n),
            train_wt: Vec::with_capacity(train_n),
            train_offset: Vec::with_capacity(train_n),
            test_x: Vec::with_capacity(test_n * n_vars),
            test_time: Vec::with_capacity(test_n),
            test_status: Vec::with_capacity(test_n),
            test_wt: Vec::with_capacity(test_n),
            test_offset: Vec::with_capacity(test_n),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_train(
        &mut self,
        idx: usize,
        x: &[f64],
        n_vars: usize,
        time: &[f64],
        status: &[i32],
        wt: &[f64],
        offset: &[f64],
    ) {
        let start = idx * n_vars;
        self.train_x.extend_from_slice(&x[start..start + n_vars]);
        self.train_time.push(time[idx]);
        self.train_status.push(status[idx]);
        self.train_wt.push(wt[idx]);
        self.train_offset.push(offset[idx]);
    }

    #[allow(clippy::too_many_arguments)]
    fn push_test(
        &mut self,
        idx: usize,
        x: &[f64],
        n_vars: usize,
        time: &[f64],
        status: &[i32],
        wt: &[f64],
        offset: &[f64],
    ) {
        let start = idx * n_vars;
        self.test_x.extend_from_slice(&x[start..start + n_vars]);
        self.test_time.push(time[idx]);
        self.test_status.push(status[idx]);
        self.test_wt.push(wt[idx]);
        self.test_offset.push(offset[idx]);
    }
}

struct FastCoxCvDevianceSummary {
    sums: Vec<f64>,
    sum_squares: Vec<f64>,
    counts: Vec<usize>,
}

impl FastCoxCvDevianceSummary {
    fn with_len(n_lambda: usize) -> Self {
        Self {
            sums: vec![0.0; n_lambda],
            sum_squares: vec![0.0; n_lambda],
            counts: vec![0; n_lambda],
        }
    }

    fn record(&mut self, lambda_idx: usize, deviance: f64) {
        self.sums[lambda_idx] += deviance;
        self.sum_squares[lambda_idx] += deviance * deviance;
        self.counts[lambda_idx] += 1;
    }

    fn merge(&mut self, other: Self) {
        for lambda_idx in 0..self.sums.len() {
            self.sums[lambda_idx] += other.sums[lambda_idx];
            self.sum_squares[lambda_idx] += other.sum_squares[lambda_idx];
            self.counts[lambda_idx] += other.counts[lambda_idx];
        }
    }

    fn mean_deviances(&self) -> Vec<f64> {
        self.sums
            .iter()
            .zip(self.counts.iter())
            .map(|(&sum, &count)| {
                if count == 0 {
                    f64::INFINITY
                } else {
                    sum / count as f64
                }
            })
            .collect()
    }

    fn se_deviances(&self, mean_deviances: &[f64]) -> Vec<f64> {
        self.counts
            .iter()
            .enumerate()
            .map(|(lambda_idx, &count)| {
                if count < 2 {
                    f64::INFINITY
                } else {
                    let mean = mean_deviances[lambda_idx];
                    let centered_sum_squares = self.sum_squares[lambda_idx]
                        - 2.0 * mean * self.sums[lambda_idx]
                        + count as f64 * mean * mean;
                    let variance = centered_sum_squares.max(0.0) / (count - 1) as f64;
                    (variance / count as f64).sqrt()
                }
            })
            .collect()
    }
}

struct FastCoxCoefficientFit {
    coefficients: Vec<f64>,
    n_iter: usize,
    converged: bool,
    scale_factors: Option<Vec<f64>>,
    center_values: Option<Vec<f64>>,
    screened_out: usize,
    active_set_size: usize,
}

fn fast_cox_lambda_max(data: &FastCoxData, l1_ratio: f64) -> f64 {
    let mut gradient = vec![0.0; data.p];
    let mut eta = vec![0.0; data.n];
    let mut exp_eta = vec![0.0; data.n];
    let mut risk_data = CoxRiskSetFirstOrderData::with_capacity(data.n, data.p);
    let mut risk_scratch = CoxRiskSetFirstOrderScratch::with_capacity(data.n, data.p);
    compute_gradient_at_zero_fast_into(
        data,
        &mut gradient,
        &mut eta,
        &mut exp_eta,
        &mut risk_data,
        &mut risk_scratch,
    );

    gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / (data.n as f64 * l1_ratio.max(0.001))
}

fn fast_cox_lambda_sequence(
    data: &FastCoxData,
    l1_ratio: f64,
    n_lambda: usize,
    lambda_min_ratio: Option<f64>,
) -> Vec<f64> {
    let lambda_max = fast_cox_lambda_max(data, l1_ratio);
    if lambda_max == 0.0 {
        return vec![0.0; n_lambda];
    }

    let min_ratio = lambda_min_ratio.unwrap_or(if data.n < data.p { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    (0..n_lambda)
        .map(|i| {
            let frac = i as f64 / (n_lambda - 1).max(1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect()
}

fn unstandardize_fast_cox_coefficients(
    beta_std: Vec<f64>,
    sds: &[f64],
    standardize: bool,
) -> Vec<f64> {
    if standardize {
        unstandardize_standard_fast_cox_coefficients(&beta_std, sds)
    } else {
        beta_std
    }
}

fn unstandardize_standard_fast_cox_coefficients(beta_std: &[f64], sds: &[f64]) -> Vec<f64> {
    beta_std
        .iter()
        .zip(sds.iter())
        .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
        .collect()
}

fn unstandardize_standard_fast_cox_coefficients_into(
    beta_std: &[f64],
    sds: &[f64],
    coefficients: &mut Vec<f64>,
) {
    debug_assert_eq!(beta_std.len(), sds.len());
    coefficients.resize(beta_std.len(), 0.0);
    for ((coefficient, &beta), &sd) in coefficients.iter_mut().zip(beta_std.iter()).zip(sds.iter())
    {
        *coefficient = if sd > 0.0 { beta / sd } else { beta };
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_fast_cox_coefficients(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    config: &FastCoxConfig,
    wt: &[f64],
    off: &[f64],
) -> FastCoxCoefficientFit {
    let (x_std, means, sds) =
        standardize_or_borrow_row_major_matrix(x, n_obs, n_vars, config.standardize);

    let fit_data = FastCoxData {
        x: x_std.as_ref(),
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: wt,
        offset: off,
    };
    let lambda_max = fast_cox_lambda_max(&fit_data, 1.0);

    let (beta_std, n_iter, converged, screened_out, active_set_size) =
        cyclic_coordinate_descent_fast(
            &fit_data,
            FastCoxDescentConfig {
                lambda: config.lambda,
                l1_ratio: config.l1_ratio,
                max_iter: config.max_iter,
                tol: config.tol,
                beta_init: None,
                screening: config.screening,
                active_set_update_freq: config.active_set_update_freq,
                lambda_prev: None,
                lambda_max,
            },
        );

    let coefficients = unstandardize_fast_cox_coefficients(beta_std, &sds, config.standardize);

    FastCoxCoefficientFit {
        coefficients,
        n_iter,
        converged,
        scale_factors: if config.standardize { Some(sds) } else { None },
        center_values: if config.standardize {
            Some(means)
        } else {
            None
        },
        screened_out,
        active_set_size,
    }
}

#[allow(clippy::too_many_arguments)]
fn fast_cox_slices(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    config: &FastCoxConfig,
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
) -> SurvivalResult<FastCoxResult> {
    CoxRegressionInput::validate_slices(x, n_obs, n_vars, time, status, weights, offset)?;

    let wt_owned;
    let wt = match weights {
        Some(weights) => weights,
        None => {
            wt_owned = vec![1.0; n_obs];
            &wt_owned
        }
    };
    let off_owned;
    let off = match offset {
        Some(offset) => offset,
        None => {
            off_owned = vec![0.0; n_obs];
            &off_owned
        }
    };

    let fit = fit_fast_cox_coefficients(x, n_obs, n_vars, time, status, config, wt, off);

    let nonzero_indices: Vec<usize> = fit
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > crate::constants::DIVISION_FLOOR)
        .map(|(i, _)| i)
        .collect();

    let df = nonzero_indices.len() as f64;
    let deviance_data = FastCoxData {
        x,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: wt,
        offset: off,
    };
    let deviance = compute_cox_deviance(&deviance_data, &fit.coefficients);

    Ok(FastCoxResult {
        coefficients: fit.coefficients,
        nonzero_indices,
        lambda_used: config.lambda,
        l1_ratio: config.l1_ratio,
        n_iter: fit.n_iter,
        converged: fit.converged,
        deviance,
        df,
        scale_factors: fit.scale_factors,
        center_values: fit.center_values,
        screened_out: fit.screened_out,
        active_set_size: fit.active_set_size,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fast_cox_array_view(
    x: ArrayView2<'_, f64>,
    time: ArrayView1<'_, f64>,
    status: ArrayView1<'_, i32>,
    config: &FastCoxConfig,
    weights: Option<ArrayView1<'_, f64>>,
    offset: Option<ArrayView1<'_, f64>>,
) -> SurvivalResult<FastCoxResult> {
    let (n_obs, n_vars) = x.dim();
    let x = x
        .as_slice()
        .ok_or_else(|| SurvivalError::invalid_input("x must be C-contiguous"))?;
    let time = time
        .as_slice_memory_order()
        .ok_or_else(|| SurvivalError::invalid_input("time must be contiguous"))?;
    let status = status
        .as_slice_memory_order()
        .ok_or_else(|| SurvivalError::invalid_input("status must be contiguous"))?;
    let weights = weights
        .as_ref()
        .map(|weights| {
            weights
                .as_slice_memory_order()
                .ok_or_else(|| SurvivalError::invalid_input("weights must be contiguous"))
        })
        .transpose()?;
    let offset = offset
        .as_ref()
        .map(|offset| {
            offset
                .as_slice_memory_order()
                .ok_or_else(|| SurvivalError::invalid_input("offset must be contiguous"))
        })
        .transpose()?;

    fast_cox_slices(x, n_obs, n_vars, time, status, config, weights, offset)
}

pub(crate) fn fast_cox_typed(
    input: &CoxRegressionInput,
    config: &FastCoxConfig,
) -> SurvivalResult<FastCoxResult> {
    fast_cox_slices(
        &input.covariates.values,
        input.covariates.n_obs,
        input.covariates.n_vars,
        &input.survival.time,
        &input.survival.status,
        config,
        input
            .weights
            .as_ref()
            .map(|weights| weights.values.as_slice()),
        input.offset.as_deref(),
    )
}

pub(crate) fn fast_cox_path_typed(
    input: &CoxRegressionInput,
    config: Option<&FastCoxPathConfig>,
) -> SurvivalResult<FastCoxPath> {
    let default_config;
    let config = match config {
        Some(config) => config,
        None => {
            default_config = FastCoxPathConfig {
                l1_ratio: 1.0,
                n_lambda: 100,
                lambda_min_ratio: None,
                max_iter: 1000,
                tol: 1e-7,
                screening: ScreeningRule::Strong,
            };
            &default_config
        }
    };

    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit_cow();
    let off = input.offset_or_zero_cow();

    let (x_std, _means, sds) = standardize_row_major_matrix(x, n_obs, n_vars);

    let fit_data = FastCoxData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: wt.as_ref(),
        offset: off.as_ref(),
    };
    let lambdas = fast_cox_lambda_sequence(
        &fit_data,
        config.l1_ratio,
        config.n_lambda,
        config.lambda_min_ratio,
    );
    let lambda_max = lambdas.first().copied().unwrap_or(0.0);

    let mut all_coefficients = Vec::with_capacity(config.n_lambda);
    let mut all_deviances = Vec::with_capacity(config.n_lambda);
    let mut all_df = Vec::with_capacity(config.n_lambda);
    let mut all_n_iters = Vec::with_capacity(config.n_lambda);
    let mut all_converged = Vec::with_capacity(config.n_lambda);

    let mut beta_warm = vec![0.0; n_vars];
    let mut lambda_prev: Option<f64> = None;
    let mut deviance_eta = vec![0.0; n_obs];
    let mut deviance_exp_eta = vec![0.0; n_obs];
    let mut deviance_risk_data = CoxRiskSetData::with_capacity(n_obs, n_vars);
    let mut deviance_risk_scratch = CoxRiskSetScratch::with_capacity(n_obs, n_vars);
    let deviance_data = FastCoxData {
        x,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: wt.as_ref(),
        offset: off.as_ref(),
    };

    for &lambda in lambdas.iter() {
        let (beta_std, n_iter, conv, _screened, _active) = cyclic_coordinate_descent_fast(
            &fit_data,
            FastCoxDescentConfig {
                lambda,
                l1_ratio: config.l1_ratio,
                max_iter: config.max_iter,
                tol: config.tol,
                beta_init: Some(&beta_warm),
                screening: config.screening,
                active_set_update_freq: 10,
                lambda_prev,
                lambda_max,
            },
        );

        let coefficients: Vec<f64> = beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect();
        beta_warm = beta_std;
        lambda_prev = Some(lambda);

        let df = coefficients
            .iter()
            .filter(|&&c| c.abs() > crate::constants::DIVISION_FLOOR)
            .count() as f64;
        let deviance = compute_cox_deviance_into(
            &deviance_data,
            &coefficients,
            &mut deviance_eta,
            &mut deviance_exp_eta,
            &mut deviance_risk_data,
            &mut deviance_risk_scratch,
        );

        all_coefficients.push(coefficients);
        all_deviances.push(deviance);
        all_df.push(df);
        all_n_iters.push(n_iter);
        all_converged.push(conv);
    }

    Ok(FastCoxPath {
        lambdas,
        coefficients: all_coefficients,
        deviances: all_deviances,
        df: all_df,
        n_iters: all_n_iters,
        converged: all_converged,
    })
}

pub(crate) fn fast_cox_cv_typed(
    input: &CoxRegressionInput,
    config: Option<&FastCoxCVConfig>,
) -> SurvivalResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let default_config;
    let config = match config {
        Some(config) => config,
        None => {
            default_config = FastCoxCVConfig {
                l1_ratio: 1.0,
                n_lambda: 100,
                n_folds: 5,
                screening: ScreeningRule::Strong,
                seed: None,
            };
            &default_config
        }
    };

    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit_cow();
    let offset = input.offset_or_zero_cow();
    let (x_std, _means, _sds) = standardize_row_major_matrix(x, n_obs, n_vars);
    let lambda_data = FastCoxData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: wt.as_ref(),
        offset: offset.as_ref(),
    };
    let lambdas = fast_cox_lambda_sequence(&lambda_data, config.l1_ratio, config.n_lambda, None);

    let mut rng =
        fastrand::Rng::with_seed(config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED));
    let mut fold_assign: Vec<usize> = (0..n_obs).map(|i| i % config.n_folds).collect();
    for i in (1..n_obs).rev() {
        let j = rng.usize(0..=i);
        fold_assign.swap(i, j);
    }

    let mut fold_counts = vec![0usize; config.n_folds];
    for &fold in &fold_assign {
        fold_counts[fold] += 1;
    }

    let folds: Vec<FastCoxCvFold> = (0..config.n_folds)
        .map(|fold| {
            let test_n = fold_counts[fold];
            let train_n = n_obs - test_n;
            let mut fold_data = FastCoxCvFold::with_capacity(train_n, test_n, n_vars);
            for (idx, &assigned) in fold_assign.iter().enumerate() {
                if assigned == fold {
                    fold_data.push_test(idx, x, n_vars, time, status, wt.as_ref(), offset.as_ref());
                } else {
                    fold_data.push_train(
                        idx,
                        x,
                        n_vars,
                        time,
                        status,
                        wt.as_ref(),
                        offset.as_ref(),
                    );
                }
            }
            fold_data
        })
        .collect();

    let cv_deviance_summary = folds
        .par_iter()
        .map(|fold| {
            let mut summary = FastCoxCvDevianceSummary::with_len(lambdas.len());
            let train_n = fold.train_time.len();
            let test_n = fold.test_time.len();
            if train_n == 0 || test_n == 0 {
                return summary;
            }

            let mut deviance_eta = vec![0.0; test_n];
            let mut deviance_exp_eta = vec![0.0; test_n];
            let mut deviance_risk_data = CoxRiskSetData::with_capacity(test_n, n_vars);
            let mut deviance_risk_scratch = CoxRiskSetScratch::with_capacity(test_n, n_vars);
            let test_data = FastCoxData {
                x: &fold.test_x,
                n: test_n,
                p: n_vars,
                time: &fold.test_time,
                status: &fold.test_status,
                weights: &fold.test_wt,
                offset: &fold.test_offset,
            };
            let (train_x_std, _train_means, train_sds) =
                standardize_row_major_matrix(&fold.train_x, train_n, n_vars);
            let train_data = FastCoxData {
                x: &train_x_std,
                n: train_n,
                p: n_vars,
                time: &fold.train_time,
                status: &fold.train_status,
                weights: &fold.train_wt,
                offset: &fold.train_offset,
            };
            let train_lambda_max = fast_cox_lambda_max(&train_data, 1.0);
            let mut beta_warm = vec![0.0; n_vars];
            let mut coefficients = Vec::with_capacity(n_vars);
            let mut lambda_prev: Option<f64> = None;

            for (lambda_idx, &lambda) in lambdas.iter().enumerate() {
                let (beta_std, _n_iter, _conv, _screened, _active) = cyclic_coordinate_descent_fast(
                    &train_data,
                    FastCoxDescentConfig {
                        lambda,
                        l1_ratio: config.l1_ratio,
                        max_iter: 1000,
                        tol: 1e-7,
                        beta_init: Some(&beta_warm),
                        screening: config.screening,
                        active_set_update_freq: 10,
                        lambda_prev,
                        lambda_max: train_lambda_max,
                    },
                );
                unstandardize_standard_fast_cox_coefficients_into(
                    &beta_std,
                    &train_sds,
                    &mut coefficients,
                );
                beta_warm = beta_std;
                lambda_prev = Some(lambda);
                let deviance = compute_cox_deviance_into(
                    &test_data,
                    &coefficients,
                    &mut deviance_eta,
                    &mut deviance_exp_eta,
                    &mut deviance_risk_data,
                    &mut deviance_risk_scratch,
                );
                summary.record(lambda_idx, deviance);
            }

            summary
        })
        .reduce(
            || FastCoxCvDevianceSummary::with_len(lambdas.len()),
            |mut combined, fold_summary| {
                combined.merge(fold_summary);
                combined
            },
        );

    let mean_deviances = cv_deviance_summary.mean_deviances();
    let se_deviances = cv_deviance_summary.se_deviances(&mean_deviances);

    let (min_idx, &min_dev) = mean_deviances
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((0, &f64::INFINITY));

    let lambda_min = lambdas[min_idx];

    let threshold = min_dev + se_deviances[min_idx];
    let lambda_1se = mean_deviances
        .iter()
        .enumerate()
        .filter(|(_, d)| **d <= threshold)
        .map(|(i, _)| lambdas[i])
        .next()
        .unwrap_or(lambda_min);

    Ok((lambda_min, lambda_1se, mean_deviances, se_deviances))
}

#[allow(clippy::too_many_arguments)]
pub fn fast_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: &FastCoxConfig,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
) -> SurvivalResult<FastCoxResult> {
    let input = CoxRegressionInput::try_new(
        CovariateMatrix::try_new(x, n_obs, n_vars)?,
        SurvivalData::try_new(time, status)?,
        weights.map(Weights::try_new).transpose()?,
        offset,
    )?;
    fast_cox_typed(&input, config)
}

#[allow(clippy::too_many_arguments)]
pub fn fast_cox_path(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    l1_ratio: f64,
    n_lambda: usize,
    lambda_min_ratio: Option<f64>,
    weights: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
    screening: ScreeningRule,
) -> SurvivalResult<FastCoxPath> {
    let input = CoxRegressionInput::try_new(
        CovariateMatrix::try_new(x, n_obs, n_vars)?,
        SurvivalData::try_new(time, status)?,
        weights.map(Weights::try_new).transpose()?,
        None,
    )?;
    let config = FastCoxPathConfig::new(
        l1_ratio,
        n_lambda,
        lambda_min_ratio,
        max_iter,
        tol,
        screening,
    )?;
    fast_cox_path_typed(&input, Some(&config))
}

#[allow(clippy::too_many_arguments)]
pub fn fast_cox_cv(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    l1_ratio: f64,
    n_lambda: usize,
    n_folds: usize,
    weights: Option<Vec<f64>>,
    screening: ScreeningRule,
    seed: Option<u64>,
) -> SurvivalResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let input = CoxRegressionInput::try_new(
        CovariateMatrix::try_new(x, n_obs, n_vars)?,
        SurvivalData::try_new(time, status)?,
        weights.map(Weights::try_new).transpose()?,
        None,
    )?;
    let config = FastCoxCVConfig::new(l1_ratio, n_lambda, n_folds, screening, seed)?;
    fast_cox_cv_typed(&input, Some(&config))
}
