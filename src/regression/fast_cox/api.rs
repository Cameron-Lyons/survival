
use crate::internal::typed_inputs::{CovariateMatrix, CoxRegressionInput, SurvivalData, Weights};

#[pyfunction]
#[pyo3(signature = (input, config))]
pub fn fast_cox(input: &CoxRegressionInput, config: &FastCoxConfig) -> PyResult<FastCoxResult> {
    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit();
    let off = input.offset_or_zero();

    let (x_std, means, sds) = if config.standardize {
        standardize_matrix(x, n_obs, n_vars)
    } else {
        (x.to_vec(), vec![0.0; n_vars], vec![1.0; n_vars])
    };

    let beta_zero = vec![0.0; n_vars];
    let fit_data = FastCoxData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: &wt,
        offset: &off,
    };
    let (gradient, _) = compute_gradient_hessian_diag_fast(&fit_data, &beta_zero, None);
    let lambda_max = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / n_obs as f64;

    let (beta_std, n_iter, converged, screened_out, active_set_size) =
        cyclic_coordinate_descent_fast(&fit_data, FastCoxDescentConfig {
            lambda: config.lambda,
            l1_ratio: config.l1_ratio,
            max_iter: config.max_iter,
            tol: config.tol,
            beta_init: None,
            screening: config.screening,
            active_set_update_freq: config.active_set_update_freq,
            lambda_prev: None,
            lambda_max,
        });

    let coefficients: Vec<f64> = if config.standardize {
        beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect()
    } else {
        beta_std
    };

    let nonzero_indices: Vec<usize> = coefficients
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
        weights: &wt,
        offset: &off,
    };
    let deviance = compute_cox_deviance(&deviance_data, &coefficients);

    Ok(FastCoxResult {
        coefficients,
        nonzero_indices,
        lambda_used: config.lambda,
        l1_ratio: config.l1_ratio,
        n_iter,
        converged,
        deviance,
        df,
        scale_factors: if config.standardize { Some(sds) } else { None },
        center_values: if config.standardize {
            Some(means)
        } else {
            None
        },
        screened_out,
        active_set_size,
    })
}

#[pyfunction]
#[pyo3(signature = (input, config=None))]
pub fn fast_cox_path(
    input: &CoxRegressionInput,
    config: Option<&FastCoxPathConfig>,
) -> PyResult<FastCoxPath> {
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
    let wt = input.weights_or_unit();
    let off = input.offset_or_zero();

    let (x_std, _means, sds) = standardize_matrix(x, n_obs, n_vars);

    let beta_zero = vec![0.0; n_vars];
    let fit_data = FastCoxData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: &wt,
        offset: &off,
    };
    let (gradient, _) = compute_gradient_hessian_diag_fast(&fit_data, &beta_zero, None);

    let lambda_max = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max)
        / (n_obs as f64 * config.l1_ratio.max(0.001));

    let min_ratio = config
        .lambda_min_ratio
        .unwrap_or(if n_obs < n_vars { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    let lambdas: Vec<f64> = (0..config.n_lambda)
        .map(|i| {
            let frac = i as f64 / (config.n_lambda - 1).max(1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect();

    let mut all_coefficients = Vec::with_capacity(config.n_lambda);
    let mut all_deviances = Vec::with_capacity(config.n_lambda);
    let mut all_df = Vec::with_capacity(config.n_lambda);
    let mut all_n_iters = Vec::with_capacity(config.n_lambda);
    let mut all_converged = Vec::with_capacity(config.n_lambda);

    let mut beta_warm = vec![0.0; n_vars];
    let mut lambda_prev: Option<f64> = None;

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

        beta_warm = beta_std.clone();
        lambda_prev = Some(lambda);

        let coefficients: Vec<f64> = beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect();

        let df = coefficients
            .iter()
            .filter(|&&c| c.abs() > crate::constants::DIVISION_FLOOR)
            .count() as f64;
        let deviance_data = FastCoxData {
            x,
            n: n_obs,
            p: n_vars,
            time,
            status,
            weights: &wt,
            offset: &off,
        };
        let deviance = compute_cox_deviance(&deviance_data, &coefficients);

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

#[pyfunction]
#[pyo3(signature = (input, config=None))]
pub fn fast_cox_cv(
    input: &CoxRegressionInput,
    config: Option<&FastCoxCVConfig>,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
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

    let path_config = FastCoxPathConfig {
        l1_ratio: config.l1_ratio,
        n_lambda: config.n_lambda,
        lambda_min_ratio: None,
        max_iter: 1000,
        tol: 1e-7,
        screening: config.screening,
    };
    let path = fast_cox_path(input, Some(&path_config))?;

    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit();
    let offset = input.offset_or_zero();

    let mut rng =
        fastrand::Rng::with_seed(config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED));
    let mut fold_assign: Vec<usize> = (0..n_obs).map(|i| i % config.n_folds).collect();
    for i in (1..n_obs).rev() {
        let j = rng.usize(0..=i);
        fold_assign.swap(i, j);
    }

    let fold_indices: Vec<(Vec<usize>, Vec<usize>)> = (0..config.n_folds)
        .map(|fold| {
            let train_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] == fold).collect();
            (train_idx, test_idx)
        })
        .collect();

    let x_ref = x;
    let time_ref = time;
    let status_ref = status;
    let wt_ref = &wt;
    let cv_deviances: Vec<Vec<f64>> = path
        .lambdas
        .par_iter()
        .map(|&lambda| {
            let fold_devs_by_fold: Vec<Option<f64>> = fold_indices
                .par_iter()
                .map(|(train_idx, test_idx)| {
                    if train_idx.is_empty() || test_idx.is_empty() {
                        return None;
                    }

                    let train_x: Vec<f64> = train_idx
                        .iter()
                        .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                        .collect();
                    let train_time: Vec<f64> = train_idx.iter().map(|&i| time_ref[i]).collect();
                    let train_status: Vec<i32> = train_idx.iter().map(|&i| status_ref[i]).collect();
                    let train_wt: Vec<f64> = train_idx.iter().map(|&i| wt_ref[i]).collect();
                    let train_offset: Vec<f64> =
                        train_idx.iter().map(|&i| offset[i]).collect();

                    let Ok(solver_config) =
                        FastCoxSolverConfig::new(1000, 1e-7, config.screening, None, 10)
                    else {
                        return None;
                    };
                    let Ok(fit_config) =
                        FastCoxConfig::new(lambda, config.l1_ratio, Some(&solver_config), true, true)
                    else {
                        return None;
                    };

                    let Ok(train_input) = CoxRegressionInput::try_new(
                        match CovariateMatrix::try_new(train_x, train_idx.len(), n_vars) {
                            Ok(covariates) => covariates,
                            Err(_) => return None,
                        },
                        match SurvivalData::try_new(train_time, train_status) {
                            Ok(survival) => survival,
                            Err(_) => return None,
                        },
                        match Weights::try_new(train_wt) {
                            Ok(weights) => Some(weights),
                            Err(_) => return None,
                        },
                        Some(train_offset),
                    ) else {
                        return None;
                    };

                    if let Ok(result) = fast_cox(&train_input, &fit_config) {
                        let test_x: Vec<f64> = test_idx
                            .iter()
                            .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                            .collect();
                        let test_time: Vec<f64> = test_idx.iter().map(|&i| time_ref[i]).collect();
                        let test_status: Vec<i32> =
                            test_idx.iter().map(|&i| status_ref[i]).collect();
                        let test_wt: Vec<f64> = test_idx.iter().map(|&i| wt_ref[i]).collect();
                        let test_off: Vec<f64> = test_idx.iter().map(|&i| offset[i]).collect();

                        let test_data = FastCoxData {
                            x: &test_x,
                            n: test_idx.len(),
                            p: n_vars,
                            time: &test_time,
                            status: &test_status,
                            weights: &test_wt,
                            offset: &test_off,
                        };
                        let dev = compute_cox_deviance(&test_data, &result.coefficients);
                        Some(dev)
                    } else {
                        None
                    }
                })
                .collect();

            fold_devs_by_fold.into_iter().flatten().collect()
        })
        .collect();

    let mean_deviances: Vec<f64> = cv_deviances
        .iter()
        .map(|devs| {
            if devs.is_empty() {
                f64::INFINITY
            } else {
                devs.iter().sum::<f64>() / devs.len() as f64
            }
        })
        .collect();

    let se_deviances: Vec<f64> = cv_deviances
        .iter()
        .zip(mean_deviances.iter())
        .map(|(devs, &mean)| {
            if devs.len() < 2 {
                f64::INFINITY
            } else {
                let var =
                    devs.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (devs.len() - 1) as f64;
                (var / devs.len() as f64).sqrt()
            }
        })
        .collect();

    let (min_idx, &min_dev) = mean_deviances
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &f64::INFINITY));

    let lambda_min = path.lambdas[min_idx];

    let threshold = min_dev + se_deviances[min_idx];
    let lambda_1se = mean_deviances
        .iter()
        .enumerate()
        .filter(|(_, d)| **d <= threshold)
        .map(|(i, _)| path.lambdas[i])
        .next()
        .unwrap_or(lambda_min);

    Ok((lambda_min, lambda_1se, mean_deviances, se_deviances))
}
