fn compute_shapley_kernel_weights(n_features: usize, coalition_sizes: &[usize]) -> Vec<f64> {
    let n = n_features as f64;
    coalition_sizes
        .iter()
        .map(|&k| {
            if k == 0 || k == n_features {
                f64::INFINITY
            } else {
                let k_f = k as f64;
                let binom = binomial(n_features, k) as f64;
                (n - 1.0) / (binom * k_f * (n - k_f))
            }
        })
        .collect()
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / ((i + 1) as u64);
    }
    result
}

fn sample_coalitions(
    n_features: usize,
    n_coalitions: usize,
    seed: u64,
) -> (Vec<Vec<bool>>, Vec<usize>) {
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut coalitions = Vec::with_capacity(n_coalitions);
    let mut coalition_sizes = Vec::with_capacity(n_coalitions);

    coalitions.push(vec![false; n_features]);
    coalition_sizes.push(0);

    coalitions.push(vec![true; n_features]);
    coalition_sizes.push(n_features);

    let n_remaining = n_coalitions.saturating_sub(2);

    for _ in 0..n_remaining {
        let target_size = if rng.bool() {
            let half = n_features / 2;
            let offset = rng.usize(0..=(n_features / 4).max(1));
            if rng.bool() {
                (half + offset).min(n_features - 1)
            } else {
                half.saturating_sub(offset).max(1)
            }
        } else {
            rng.usize(1..n_features)
        };

        let mut coalition = vec![false; n_features];
        let mut indices: Vec<usize> = (0..n_features).collect();
        for i in (1..n_features).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }
        for &idx in indices.iter().take(target_size) {
            coalition[idx] = true;
        }

        coalitions.push(coalition);
        coalition_sizes.push(target_size);
    }

    (coalitions, coalition_sizes)
}

fn weighted_least_squares(
    x_matrix: &[f64],
    y: &[f64],
    weights: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f64> {
    let mut xtwx = vec![0.0; n_cols * n_cols];
    let mut xtwy = vec![0.0; n_cols];

    for i in 0..n_rows {
        let w = weights[i];
        if !w.is_finite() || w <= 0.0 {
            continue;
        }

        for j in 0..n_cols {
            let xij = x_matrix[i * n_cols + j];
            xtwy[j] += w * xij * y[i];
            for k in 0..n_cols {
                let xik = x_matrix[i * n_cols + k];
                xtwx[j * n_cols + k] += w * xij * xik;
            }
        }
    }

    let reg = 1e-8;
    for j in 0..n_cols {
        xtwx[j * n_cols + j] += reg;
    }

    solve_positive_definite(&mut xtwx, &xtwy, n_cols)
}

fn solve_positive_definite(a: &mut [f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    l[i * n + j] = 1e-10;
                } else {
                    l[i * n + j] = sum.sqrt();
                }
            } else {
                l[i * n + j] = sum / l[j * n + j].max(1e-10);
            }
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * n + j] * y[j];
        }
        y[i] = sum / l[i * n + i].max(1e-10);
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j * n + i] * x[j];
        }
        x[i] = sum / l[i * n + i].max(1e-10);
    }

    x
}

#[allow(clippy::too_many_arguments)]
fn evaluate_coalition_predictions(
    _x_explain: &[f64],
    _x_background: &[f64],
    predictions_explain: &[f64],
    predictions_background: &[f64],
    coalitions: &[Vec<bool>],
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_times: usize,
    parallel: bool,
) -> Vec<Vec<f64>> {
    let process_coalition = |coalition: &Vec<bool>| -> Vec<f64> {
        let mut coalition_preds = vec![0.0; n_explain * n_times];

        for i in 0..n_explain {
            for bg_idx in 0..n_background {
                for t in 0..n_times {
                    let mut uses_explain_fully = true;
                    let mut uses_background_fully = true;

                    for &included in coalition.iter().take(n_features) {
                        if included {
                            uses_background_fully = false;
                        } else {
                            uses_explain_fully = false;
                        }
                    }

                    let pred = if uses_explain_fully {
                        predictions_explain[i * n_times + t]
                    } else if uses_background_fully {
                        predictions_background[bg_idx * n_times + t]
                    } else {
                        let weight_explain: f64 =
                            coalition.iter().map(|&c| if c { 1.0 } else { 0.0 }).sum();
                        let weight_bg = n_features as f64 - weight_explain;
                        let total = n_features as f64;

                        (weight_explain / total) * predictions_explain[i * n_times + t]
                            + (weight_bg / total) * predictions_background[bg_idx * n_times + t]
                    };

                    coalition_preds[i * n_times + t] += pred / n_background as f64;
                }
            }
        }

        coalition_preds
    };

    if parallel {
        coalitions.par_iter().map(process_coalition).collect()
    } else {
        coalitions.iter().map(process_coalition).collect()
    }
}
