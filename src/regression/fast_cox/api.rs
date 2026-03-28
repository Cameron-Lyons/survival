
#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config, weights=None, offset=None))]
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
) -> PyResult<FastCoxResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have length n_obs",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);
    let off = offset.unwrap_or_else(|| vec![0.0; n_obs]);

    let (x_std, means, sds) = if config.standardize {
        standardize_matrix(&x, n_obs, n_vars)
    } else {
        (x.clone(), vec![0.0; n_vars], vec![1.0; n_vars])
    };

    let beta_zero = vec![0.0; n_vars];
    let (gradient, _) = compute_gradient_hessian_diag_fast(
        &x_std, n_obs, n_vars, &time, &status, &wt, &beta_zero, &off, None,
    );
    let lambda_max = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / n_obs as f64;

    let (beta_std, n_iter, converged, screened_out, active_set_size) =
        cyclic_coordinate_descent_fast(
            &x_std,
            n_obs,
            n_vars,
            &time,
            &status,
            &wt,
            &off,
            config.lambda,
            config.l1_ratio,
            config.max_iter,
            config.tol,
            None,
            config.screening,
            config.active_set_update_freq,
            None,
            lambda_max,
        );

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
    let deviance =
        compute_cox_deviance(&x, n_obs, n_vars, &time, &status, &wt, &coefficients, &off);

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
#[pyo3(signature = (
    x,
    n_obs,
    n_vars,
    time,
    status,
    l1_ratio=1.0,
    n_lambda=100,
    lambda_min_ratio=None,
    weights=None,
    max_iter=1000,
    tol=1e-7,
    screening=ScreeningRule::Strong
))]
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
) -> PyResult<FastCoxPath> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);
    let off = vec![0.0; n_obs];

    let (x_std, _means, sds) = standardize_matrix(&x, n_obs, n_vars);

    let beta_zero = vec![0.0; n_vars];
    let (gradient, _) = compute_gradient_hessian_diag_fast(
        &x_std, n_obs, n_vars, &time, &status, &wt, &beta_zero, &off, None,
    );

    let lambda_max =
        gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / (n_obs as f64 * l1_ratio.max(0.001));

    let min_ratio = lambda_min_ratio.unwrap_or(if n_obs < n_vars { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    let lambdas: Vec<f64> = (0..n_lambda)
        .map(|i| {
            let frac = i as f64 / (n_lambda - 1).max(1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect();

    let mut all_coefficients = Vec::with_capacity(n_lambda);
    let mut all_deviances = Vec::with_capacity(n_lambda);
    let mut all_df = Vec::with_capacity(n_lambda);
    let mut all_n_iters = Vec::with_capacity(n_lambda);
    let mut all_converged = Vec::with_capacity(n_lambda);

    let mut beta_warm = vec![0.0; n_vars];
    let mut lambda_prev: Option<f64> = None;

    for &lambda in lambdas.iter() {
        let (beta_std, n_iter, conv, _screened, _active) = cyclic_coordinate_descent_fast(
            &x_std,
            n_obs,
            n_vars,
            &time,
            &status,
            &wt,
            &off,
            lambda,
            l1_ratio,
            max_iter,
            tol,
            Some(&beta_warm),
            screening,
            10,
            lambda_prev,
            lambda_max,
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
        let deviance =
            compute_cox_deviance(&x, n_obs, n_vars, &time, &status, &wt, &coefficients, &off);

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
#[pyo3(signature = (
    x,
    n_obs,
    n_vars,
    time,
    status,
    l1_ratio=1.0,
    n_lambda=100,
    n_folds=5,
    weights=None,
    screening=ScreeningRule::Strong,
    seed=None
))]
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
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let path = fast_cox_path(
        x.clone(),
        n_obs,
        n_vars,
        time.clone(),
        status.clone(),
        l1_ratio,
        n_lambda,
        None,
        weights.clone(),
        1000,
        1e-7,
        screening,
    )?;

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);

    let mut rng = fastrand::Rng::with_seed(seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED));
    let mut fold_assign: Vec<usize> = (0..n_obs).map(|i| i % n_folds).collect();
    for i in (1..n_obs).rev() {
        let j = rng.usize(0..=i);
        fold_assign.swap(i, j);
    }

    let fold_indices: Vec<(Vec<usize>, Vec<usize>)> = (0..n_folds)
        .map(|fold| {
            let train_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] == fold).collect();
            (train_idx, test_idx)
        })
        .collect();

    let x_ref = &x;
    let time_ref = &time;
    let status_ref = &status;
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

                    let Ok(config) = FastCoxConfig::new(
                        lambda, l1_ratio, 1000, 1e-7, screening, None, 10, true, true,
                    ) else {
                        return None;
                    };

                    if let Ok(result) = fast_cox(
                        train_x,
                        train_idx.len(),
                        n_vars,
                        train_time,
                        train_status,
                        &config,
                        Some(train_wt),
                        None,
                    ) {
                        let test_x: Vec<f64> = test_idx
                            .iter()
                            .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                            .collect();
                        let test_time: Vec<f64> = test_idx.iter().map(|&i| time_ref[i]).collect();
                        let test_status: Vec<i32> =
                            test_idx.iter().map(|&i| status_ref[i]).collect();
                        let test_wt: Vec<f64> = test_idx.iter().map(|&i| wt_ref[i]).collect();
                        let test_off = vec![0.0; test_idx.len()];

                        let dev = compute_cox_deviance(
                            &test_x,
                            test_idx.len(),
                            n_vars,
                            &test_time,
                            &test_status,
                            &test_wt,
                            &result.coefficients,
                            &test_off,
                        );
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

