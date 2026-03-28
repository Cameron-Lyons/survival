fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

fn standardize_matrix(x: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; p];
    let mut sds = vec![1.0; p];
    let mut x_std = x.to_vec();

    for j in 0..p {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let val = x[i * p + j];
            sum += val;
            sum_sq += val * val;
        }
        means[j] = sum / n as f64;
        let var = sum_sq / n as f64 - means[j] * means[j];
        sds[j] = var.sqrt().max(crate::constants::DIVISION_FLOOR);

        for i in 0..n {
            x_std[i * p + j] = (x[i * p + j] - means[j]) / sds[j];
        }
    }

    (x_std, means, sds)
}

struct RiskSetData {
    sorted_indices: Vec<usize>,
    cumsum_exp_eta: Vec<f64>,
    cumsum_weighted_x: Vec<Vec<f64>>,
    cumsum_weighted_x_sq: Vec<Vec<f64>>,
}

fn precompute_risk_set_cumsum(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
) -> RiskSetData {
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumsum_exp_eta = vec![0.0; n];
    let mut cumsum_weighted_x = vec![vec![0.0; p]; n];
    let mut cumsum_weighted_x_sq = vec![vec![0.0; p]; n];

    let mut running_exp = 0.0;
    let mut running_wx = vec![0.0; p];
    let mut running_wxsq = vec![0.0; p];

    for (pos, &idx) in sorted_indices.iter().enumerate() {
        let w = weights[idx] * exp_eta[idx];
        running_exp += w;

        for j in 0..p {
            let xij = x[idx * p + j];
            running_wx[j] += w * xij;
            running_wxsq[j] += w * xij * xij;
        }

        cumsum_exp_eta[pos] = running_exp;
        cumsum_weighted_x[pos] = running_wx.clone();
        cumsum_weighted_x_sq[pos] = running_wxsq.clone();
    }

    RiskSetData {
        sorted_indices,
        cumsum_exp_eta,
        cumsum_weighted_x,
        cumsum_weighted_x_sq,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_gradient_hessian_diag_fast(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    beta: &[f64],
    offset: &[f64],
    active_set: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>) {
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = offset[i];
            for j in 0..p {
                e += x[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let risk_data = precompute_risk_set_cumsum(x, n, p, time, weights, &exp_eta);

    let features_to_process: Vec<usize> = active_set
        .map(|a| a.to_vec())
        .unwrap_or_else(|| (0..p).collect());

    let mut gradient = vec![0.0; p];
    let mut hessian_diag = vec![0.0; p];

    let mut index_to_pos = vec![0usize; n];
    for (pos, &idx) in risk_data.sorted_indices.iter().enumerate() {
        index_to_pos[idx] = pos;
    }

    for i in 0..n {
        if status[i] != 1 {
            continue;
        }

        let pos = index_to_pos[i];
        let risk_sum = risk_data.cumsum_exp_eta[pos];
        if risk_sum <= 0.0 {
            continue;
        }

        for &j in &features_to_process {
            let xij = x[i * p + j];
            let x_bar = risk_data.cumsum_weighted_x[pos][j] / risk_sum;
            let x_sq_bar = risk_data.cumsum_weighted_x_sq[pos][j] / risk_sum;

            gradient[j] += weights[i] * (xij - x_bar);
            hessian_diag[j] += weights[i] * (x_sq_bar - x_bar * x_bar);
        }
    }

    (gradient, hessian_diag)
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

#[allow(clippy::too_many_arguments)]
fn compute_cox_deviance(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    beta: &[f64],
    offset: &[f64],
) -> f64 {
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = offset[i];
            for j in 0..p {
                e += x[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut loglik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += weights[i] * exp_eta[i];

        if status[i] == 1 && risk_sum > 0.0 {
            loglik += weights[i] * (eta[i] - risk_sum.ln());
        }
    }

    -2.0 * loglik
}

#[allow(clippy::too_many_arguments)]
fn cyclic_coordinate_descent_fast(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    offset: &[f64],
    lambda: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    beta_init: Option<&[f64]>,
    screening: ScreeningRule,
    active_set_update_freq: usize,
    lambda_prev: Option<f64>,
    lambda_max: f64,
) -> (Vec<f64>, usize, bool, usize, usize) {
    let mut beta = beta_init
        .map(|b| b.to_vec())
        .unwrap_or_else(|| vec![0.0; p]);

    let l1_penalty = lambda * l1_ratio;
    let l2_penalty = lambda * (1.0 - l1_ratio);

    let (gradient, _) =
        compute_gradient_hessian_diag_fast(x, n, p, time, status, weights, &beta, offset, None);

    let mut active_set: Vec<usize> = match screening {
        ScreeningRule::None => (0..p).collect(),
        ScreeningRule::Safe => apply_safe_screening(&gradient, lambda, &beta, p),
        ScreeningRule::Strong => apply_strong_screening(&gradient, lambda, lambda_prev, &beta, p),
        ScreeningRule::EDPP => apply_edpp_screening(&gradient, lambda, lambda_max, &beta, p),
    };

    let screened_out = p - active_set.len();

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let beta_old = beta.clone();

        let (gradient, hessian_diag) = compute_gradient_hessian_diag_fast(
            x,
            n,
            p,
            time,
            status,
            weights,
            &beta,
            offset,
            Some(&active_set),
        );

        for &j in &active_set {
            let h_jj = hessian_diag[j] + l2_penalty;
            if h_jj.abs() < crate::constants::DIVISION_FLOOR {
                continue;
            }

            let z = gradient[j] + hessian_diag[j] * beta[j];
            beta[j] = soft_threshold(z, l1_penalty) / h_jj;
        }

        let max_change: f64 = active_set
            .iter()
            .map(|&j| (beta[j] - beta_old[j]).abs())
            .fold(0.0, f64::max);

        if max_change < tol {
            let (full_gradient, _) = compute_gradient_hessian_diag_fast(
                x, n, p, time, status, weights, &beta, offset, None,
            );

            let kkt_violations: Vec<usize> = (0..p)
                .filter(|&j| {
                    if beta[j].abs() > crate::constants::DIVISION_FLOOR {
                        false
                    } else {
                        full_gradient[j].abs() > l1_penalty * 1.01
                    }
                })
                .collect();

            if kkt_violations.is_empty() {
                converged = true;
                break;
            } else {
                active_set.extend(kkt_violations);
                active_set.sort();
                active_set.dedup();
            }
        }

        if iter % active_set_update_freq == 0 && iter > 0 {
            let (full_gradient, _) = compute_gradient_hessian_diag_fast(
                x, n, p, time, status, weights, &beta, offset, None,
            );

            let new_active: Vec<usize> = (0..p)
                .filter(|&j| {
                    beta[j].abs() > crate::constants::DIVISION_FLOOR
                        || full_gradient[j].abs() >= l1_penalty * 0.5
                })
                .collect();

            if !new_active.is_empty() {
                active_set = new_active;
            }
        }
    }

    let active_set_size = active_set.len();
    (beta, n_iter, converged, screened_out, active_set_size)
}
