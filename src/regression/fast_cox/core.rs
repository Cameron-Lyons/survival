fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

struct FastCoxData<'a> {
    x: &'a [f64],
    n: usize,
    p: usize,
    time: &'a [f64],
    status: &'a [i32],
    weights: &'a [f64],
    offset: &'a [f64],
}

struct FastCoxDescentConfig<'a> {
    lambda: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    beta_init: Option<&'a [f64]>,
    screening: ScreeningRule,
    active_set_update_freq: usize,
    lambda_prev: Option<f64>,
    lambda_max: f64,
}

#[allow(clippy::needless_range_loop)]
fn linear_predictors(data: &FastCoxData, beta: &[f64]) -> Vec<f64> {
    let mut eta = vec![0.0; data.n];
    linear_predictors_into(data, beta, &mut eta);
    eta
}

#[allow(clippy::needless_range_loop)]
fn linear_predictors_into(data: &FastCoxData, beta: &[f64], eta: &mut [f64]) {
    debug_assert_eq!(eta.len(), data.n);
    for i in 0..data.n {
        let mut value = data.offset[i];
        for j in 0..data.p {
            value += data.x[i * data.p + j] * beta[j];
        }
        eta[i] = value;
    }
}

fn shifted_exp_eta_with_shift_into(eta: &[f64], weights: &[f64], shift: f64, target: &mut [f64]) {
    debug_assert_eq!(target.len(), eta.len());
    debug_assert_eq!(weights.len(), eta.len());
    for ((target_value, &eta_i), &weight) in target.iter_mut().zip(eta.iter()).zip(weights.iter()) {
        *target_value = if weight > 0.0 {
            (eta_i - shift).exp()
        } else {
            0.0
        };
    }
}

#[allow(clippy::needless_range_loop)]
fn compute_gradient_hessian_diag_fast(
    data: &FastCoxData,
    beta: &[f64],
    active_set: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>) {
    let mut gradient = vec![0.0; data.p];
    let mut hessian_diag = vec![0.0; data.p];
    let mut eta = vec![0.0; data.n];
    let mut exp_eta = vec![0.0; data.n];
    compute_gradient_hessian_diag_fast_into(
        data,
        beta,
        active_set,
        &mut gradient,
        &mut hessian_diag,
        &mut eta,
        &mut exp_eta,
    );
    (gradient, hessian_diag)
}

#[allow(clippy::needless_range_loop)]
fn compute_gradient_hessian_diag_fast_into(
    data: &FastCoxData,
    beta: &[f64],
    active_set: Option<&[usize]>,
    gradient: &mut [f64],
    hessian_diag: &mut [f64],
    eta: &mut [f64],
    exp_eta: &mut [f64],
) {
    debug_assert_eq!(gradient.len(), data.p);
    debug_assert_eq!(hessian_diag.len(), data.p);
    debug_assert_eq!(eta.len(), data.n);
    debug_assert_eq!(exp_eta.len(), data.n);
    gradient.fill(0.0);
    hessian_diag.fill(0.0);

    linear_predictors_into(data, beta, eta);
    let shift = cox_risk_shift(eta, data.weights);
    shifted_exp_eta_with_shift_into(eta, data.weights, shift, exp_eta);

    let risk_data = precompute_cox_risk_set_cumsum(
        data.x,
        data.n,
        data.p,
        data.time,
        data.weights,
        exp_eta,
    );

    for i in 0..data.n {
        if data.status[i] != 1 {
            continue;
        }

        let pos = risk_data.risk_set_pos[i];
        let risk_sum = risk_data.cumsum_exp_eta[pos];
        if risk_sum <= 0.0 {
            continue;
        }

        let mut accumulate_feature = |j: usize| {
            let xij = data.x[i * data.p + j];
            let cumsum_idx = pos * data.p + j;
            let x_bar = risk_data.cumsum_weighted_x[cumsum_idx] / risk_sum;
            let x_sq_bar = risk_data.cumsum_weighted_x_sq[cumsum_idx] / risk_sum;

            gradient[j] += data.weights[i] * (xij - x_bar);
            hessian_diag[j] += data.weights[i] * (x_sq_bar - x_bar * x_bar);
        };

        match active_set {
            Some(features) => {
                for &j in features {
                    accumulate_feature(j);
                }
            }
            None => {
                for j in 0..data.p {
                    accumulate_feature(j);
                }
            }
        }
    }
}

fn apply_strong_screening(
    gradient: &[f64],
    lambda: f64,
    lambda_prev: Option<f64>,
    beta: &[f64],
    p: usize,
) -> Vec<usize> {
    let threshold = match lambda_prev {
        Some(lp) => 2.0 * lambda - lp,
        None => lambda,
    };

    (0..p)
        .filter(|&j| {
            beta[j].abs() > crate::constants::DIVISION_FLOOR || gradient[j].abs() >= threshold
        })
        .collect()
}

fn apply_safe_screening(gradient: &[f64], lambda: f64, beta: &[f64], p: usize) -> Vec<usize> {
    (0..p)
        .filter(|&j| {
            beta[j].abs() > crate::constants::DIVISION_FLOOR || gradient[j].abs() >= lambda
        })
        .collect()
}

fn apply_edpp_screening(
    gradient: &[f64],
    lambda: f64,
    lambda_max: f64,
    beta: &[f64],
    p: usize,
) -> Vec<usize> {
    let threshold = lambda * (1.0 - (lambda / lambda_max).min(1.0));

    (0..p)
        .filter(|&j| {
            beta[j].abs() > crate::constants::DIVISION_FLOOR || gradient[j].abs() >= threshold
        })
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn compute_cox_deviance(data: &FastCoxData, beta: &[f64]) -> f64 {
    let eta = linear_predictors(data, beta);
    let risk_shift = cox_risk_shift(&eta, data.weights);
    let exp_eta = shifted_exp_eta_with_shift(&eta, data.weights, risk_shift);
    let risk_data = precompute_cox_risk_set_cumsum(
        data.x,
        data.n,
        data.p,
        data.time,
        data.weights,
        &exp_eta,
    );

    let mut loglik = 0.0;

    for i in 0..data.n {
        let risk_sum = risk_data.cumsum_exp_eta[risk_data.risk_set_pos[i]];
        if data.status[i] == 1 && risk_sum > 0.0 {
            loglik += data.weights[i] * (eta[i] - risk_shift - risk_sum.ln());
        }
    }

    -2.0 * loglik
}

fn cyclic_coordinate_descent_fast(
    data: &FastCoxData,
    config: FastCoxDescentConfig,
) -> (Vec<f64>, usize, bool, usize, usize) {
    let mut beta = config
        .beta_init
        .map(|b| b.to_vec())
        .unwrap_or_else(|| vec![0.0; data.p]);

    let l1_penalty = config.lambda * config.l1_ratio;
    let l2_penalty = config.lambda * (1.0 - config.l1_ratio);

    let mut gradient = vec![0.0; data.p];
    let mut hessian_diag = vec![0.0; data.p];
    let mut eta = vec![0.0; data.n];
    let mut exp_eta = vec![0.0; data.n];
    compute_gradient_hessian_diag_fast_into(
        data,
        &beta,
        None,
        &mut gradient,
        &mut hessian_diag,
        &mut eta,
        &mut exp_eta,
    );

    let mut active_set: Vec<usize> = match config.screening {
        ScreeningRule::None => (0..data.p).collect(),
        ScreeningRule::Safe => apply_safe_screening(&gradient, config.lambda, &beta, data.p),
        ScreeningRule::Strong => apply_strong_screening(
            &gradient,
            config.lambda,
            config.lambda_prev,
            &beta,
            data.p,
        ),
        ScreeningRule::EDPP => apply_edpp_screening(
            &gradient,
            config.lambda,
            config.lambda_max,
            &beta,
            data.p,
        ),
    };

    let screened_out = data.p - active_set.len();

    let mut converged = false;
    let mut n_iter = 0;
    let mut kkt_violations = Vec::new();
    let mut new_active = Vec::new();

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        compute_gradient_hessian_diag_fast_into(
            data,
            &beta,
            Some(&active_set),
            &mut gradient,
            &mut hessian_diag,
            &mut eta,
            &mut exp_eta,
        );

        let mut max_change: f64 = 0.0;
        for &j in &active_set {
            let h_jj = hessian_diag[j] + l2_penalty;
            if h_jj.abs() < crate::constants::DIVISION_FLOOR {
                continue;
            }

            let z = gradient[j] + hessian_diag[j] * beta[j];
            let old_beta = beta[j];
            let new_beta = soft_threshold(z, l1_penalty) / h_jj;
            beta[j] = new_beta;
            max_change = max_change.max((new_beta - old_beta).abs());
        }

        if max_change < config.tol {
            compute_gradient_hessian_diag_fast_into(
                data,
                &beta,
                None,
                &mut gradient,
                &mut hessian_diag,
                &mut eta,
                &mut exp_eta,
            );

            kkt_violations.clear();
            kkt_violations.extend((0..data.p).filter(|&j| {
                if beta[j].abs() > crate::constants::DIVISION_FLOOR {
                    false
                } else {
                    gradient[j].abs() > l1_penalty * 1.01
                }
            }));

            if kkt_violations.is_empty() {
                converged = true;
                break;
            } else {
                active_set.extend(kkt_violations.iter().copied());
                active_set.sort();
                active_set.dedup();
            }
        }

        if iter % config.active_set_update_freq == 0 && iter > 0 {
            compute_gradient_hessian_diag_fast_into(
                data,
                &beta,
                None,
                &mut gradient,
                &mut hessian_diag,
                &mut eta,
                &mut exp_eta,
            );

            new_active.clear();
            new_active.extend((0..data.p).filter(|&j| {
                beta[j].abs() > crate::constants::DIVISION_FLOOR
                    || gradient[j].abs() >= l1_penalty * 0.5
            }));

            if !new_active.is_empty() {
                std::mem::swap(&mut active_set, &mut new_active);
            }
        }
    }

    let active_set_size = active_set.len();
    (beta, n_iter, converged, screened_out, active_set_size)
}
