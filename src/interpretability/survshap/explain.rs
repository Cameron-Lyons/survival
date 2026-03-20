
#[allow(clippy::too_many_arguments)]
fn compute_shap_inner(
    x_explain: &[f64],
    x_background: &[f64],
    predictions_explain: &[f64],
    predictions_background: &[f64],
    time_points: &[f64],
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_coalitions: usize,
    seed: u64,
    parallel: bool,
) -> ShapComputation {
    let n_times = time_points.len();

    let (coalitions, coalition_sizes) = sample_coalitions(n_features, n_coalitions, seed);
    let kernel_weights = compute_shapley_kernel_weights(n_features, &coalition_sizes);

    let coalition_preds = evaluate_coalition_predictions(
        x_explain,
        x_background,
        predictions_explain,
        predictions_background,
        &coalitions,
        n_explain,
        n_background,
        n_features,
        n_times,
        parallel,
    );

    let base_value: Vec<f64> = (0..n_times)
        .map(|t| {
            predictions_background
                .iter()
                .skip(t)
                .step_by(n_times)
                .sum::<f64>()
                / n_background as f64
        })
        .collect();

    let mut shap_values = vec![vec![vec![0.0; n_times]; n_features]; n_explain];

    for sample_idx in 0..n_explain {
        for t in 0..n_times {
            let n_coal = coalitions.len();
            let mut x_matrix = vec![0.0; n_coal * n_features];
            let mut y = vec![0.0; n_coal];

            for (c_idx, coalition) in coalitions.iter().enumerate() {
                for (f_idx, &included) in coalition.iter().enumerate() {
                    x_matrix[c_idx * n_features + f_idx] = if included { 1.0 } else { 0.0 };
                }
                y[c_idx] = coalition_preds[c_idx][sample_idx * n_times + t] - base_value[t];
            }

            let shap_t = weighted_least_squares(&x_matrix, &y, &kernel_weights, n_coal, n_features);

            for (f_idx, &val) in shap_t.iter().enumerate() {
                shap_values[sample_idx][f_idx][t] = val;
            }
        }
    }

    (shap_values, base_value)
}

#[pyfunction]
#[pyo3(signature = (
    x_explain,
    x_background,
    predictions_explain,
    predictions_background,
    time_points,
    n_explain,
    n_background,
    n_features,
    config=None,
    aggregation_method=None
))]
#[allow(clippy::too_many_arguments)]
pub fn survshap(
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    predictions_explain: Vec<f64>,
    predictions_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    config: Option<&SurvShapConfig>,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<SurvShapResult> {
    let n_times = time_points.len();

    if x_explain.len() != n_explain * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_explain length must equal n_explain * n_features",
        ));
    }
    if x_background.len() != n_background * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_background length must equal n_background * n_features",
        ));
    }
    if predictions_explain.len() != n_explain * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_explain length must equal n_explain * n_times",
        ));
    }
    if predictions_background.len() != n_background * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_background length must equal n_background * n_times",
        ));
    }

    let default_config = SurvShapConfig::new(2048, 100, None, true)?;
    let cfg = config.unwrap_or(&default_config);

    let seed = cfg.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let (shap_values, base_value) = compute_shap_inner(
        &x_explain,
        &x_background,
        &predictions_explain,
        &predictions_background,
        &time_points,
        n_explain,
        n_background,
        n_features,
        cfg.n_coalitions,
        seed,
        cfg.parallel,
    );

    let aggregated_importance = aggregation_method.map(|method| {
        aggregate_shap_values(&shap_values, &time_points, method, n_features, n_times)
    });

    Ok(SurvShapResult {
        shap_values,
        base_value,
        time_points,
        aggregated_importance,
    })
}

#[pyfunction]
#[pyo3(signature = (
    x_explain,
    x_background,
    predictions_explain,
    predictions_background,
    time_points,
    n_explain,
    n_background,
    n_features,
    n_bootstrap=100,
    confidence_level=0.95,
    config=None
))]
#[allow(clippy::too_many_arguments)]
pub fn survshap_bootstrap(
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    predictions_explain: Vec<f64>,
    predictions_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    n_bootstrap: usize,
    confidence_level: f64,
    config: Option<&SurvShapConfig>,
) -> PyResult<BootstrapSurvShapResult> {
    let n_times = time_points.len();

    if x_explain.len() != n_explain * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_explain length must equal n_explain * n_features",
        ));
    }
    if x_background.len() != n_background * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_background length must equal n_background * n_features",
        ));
    }
    if predictions_explain.len() != n_explain * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_explain length must equal n_explain * n_times",
        ));
    }
    if predictions_background.len() != n_background * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions_background length must equal n_background * n_times",
        ));
    }
    if !(0.0..1.0).contains(&confidence_level) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "confidence_level must be between 0 and 1",
        ));
    }

    let default_config = SurvShapConfig::new(2048, 100, None, true)?;
    let cfg = config.unwrap_or(&default_config);
    let base_seed = cfg.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let x_bg_ref = &x_background;
    let preds_bg_ref = &predictions_background;
    let x_exp_ref = &x_explain;
    let preds_exp_ref = &predictions_explain;
    let time_pts_ref = &time_points;

    let bootstrap_results: Vec<ShapComputation> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let seed = base_seed.wrapping_add(b as u64);
            let mut rng = fastrand::Rng::with_seed(seed);

            let bg_indices: Vec<usize> = (0..n_background)
                .map(|_| rng.usize(0..n_background))
                .collect();

            let sampled_x_bg: Vec<f64> = bg_indices
                .iter()
                .flat_map(|&idx| (0..n_features).map(move |f| x_bg_ref[idx * n_features + f]))
                .collect();

            let sampled_preds_bg: Vec<f64> = bg_indices
                .iter()
                .flat_map(|&idx| (0..n_times).map(move |t| preds_bg_ref[idx * n_times + t]))
                .collect();

            compute_shap_inner(
                x_exp_ref,
                &sampled_x_bg,
                preds_exp_ref,
                &sampled_preds_bg,
                time_pts_ref,
                n_explain,
                n_background,
                n_features,
                cfg.n_coalitions,
                seed,
                false,
            )
        })
        .collect();

    let mut shap_values_mean = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_std = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_lower = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
    let mut shap_values_upper = vec![vec![vec![0.0; n_times]; n_features]; n_explain];

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    let mut values_buffer = vec![0.0; n_bootstrap];
    for i in 0..n_explain {
        for f in 0..n_features {
            for t in 0..n_times {
                for (b, (shap, _)) in bootstrap_results.iter().enumerate() {
                    values_buffer[b] = shap[i][f][t];
                }

                let mean = values_buffer.iter().sum::<f64>() / n_bootstrap as f64;
                let variance = values_buffer
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / n_bootstrap as f64;

                values_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                shap_values_mean[i][f][t] = mean;
                shap_values_std[i][f][t] = variance.sqrt();
                shap_values_lower[i][f][t] = values_buffer[lower_idx.min(n_bootstrap - 1)];
                shap_values_upper[i][f][t] = values_buffer[upper_idx.min(n_bootstrap - 1)];
            }
        }
    }

    let base_value: Vec<f64> = (0..n_times)
        .map(|t| preds_bg_ref.iter().skip(t).step_by(n_times).sum::<f64>() / n_background as f64)
        .collect();

    Ok(BootstrapSurvShapResult {
        shap_values_mean,
        shap_values_std,
        shap_values_lower,
        shap_values_upper,
        base_value,
        time_points,
        n_bootstrap,
        confidence_level,
    })
}

#[pyfunction]
#[pyo3(signature = (
    predictions,
    time_points,
    times,
    events,
    n_samples,
    n_features,
    n_repeats=10,
    seed=None,
    parallel=true
))]
#[allow(clippy::too_many_arguments)]
pub fn permutation_importance(
    predictions: Vec<f64>,
    time_points: Vec<f64>,
    times: Vec<f64>,
    events: Vec<i32>,
    n_samples: usize,
    n_features: usize,
    n_repeats: usize,
    seed: Option<u64>,
    parallel: bool,
) -> PyResult<PermutationImportanceResult> {
    let n_times = time_points.len();

    if predictions.len() != n_samples * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions length must equal n_samples * n_times",
        ));
    }
    if times.len() != n_samples || events.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "times and events must have length n_samples",
        ));
    }

    let baseline_score =
        compute_concordance_index(&predictions, &times, &events, n_samples, n_times);

    let base_seed = seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let compute_feature_importance = |feature_idx: usize| -> (f64, f64) {
        let mut scores = Vec::with_capacity(n_repeats);

        for r in 0..n_repeats {
            let mut rng =
                fastrand::Rng::with_seed(base_seed + feature_idx as u64 * 1000 + r as u64);

            let mut perm_indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = rng.usize(0..=i);
                perm_indices.swap(i, j);
            }

            let mut permuted_preds = predictions.clone();
            for (new_idx, &orig_idx) in perm_indices.iter().enumerate() {
                for t in 0..n_times {
                    permuted_preds[new_idx * n_times + t] = predictions[orig_idx * n_times + t];
                }
            }

            let score =
                compute_concordance_index(&permuted_preds, &times, &events, n_samples, n_times);
            scores.push(baseline_score - score);
        }

        let mean = scores.iter().sum::<f64>() / n_repeats as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n_repeats as f64;
        (mean, variance.sqrt())
    };

    let results: Vec<(f64, f64)> = if parallel {
        (0..n_features)
            .into_par_iter()
            .map(compute_feature_importance)
            .collect()
    } else {
        (0..n_features).map(compute_feature_importance).collect()
    };

    let importance: Vec<f64> = results.iter().map(|(m, _)| *m).collect();
    let importance_std: Vec<f64> = results.iter().map(|(_, s)| *s).collect();

    Ok(PermutationImportanceResult {
        importance,
        importance_std,
        baseline_score,
        n_repeats,
    })
}

fn compute_concordance_index(
    predictions: &[f64],
    times: &[f64],
    events: &[i32],
    n_samples: usize,
    n_times: usize,
) -> f64 {
    let risk_scores: Vec<f64> = (0..n_samples)
        .map(|i| {
            predictions[i * n_times..(i + 1) * n_times]
                .iter()
                .sum::<f64>()
                / n_times as f64
        })
        .collect();

    let mut concordant = 0.0;
    let mut discordant = 0.0;

    for i in 0..n_samples {
        if events[i] == 0 {
            continue;
        }
        for j in 0..n_samples {
            if i == j || times[j] < times[i] {
                continue;
            }
            if risk_scores[i] < risk_scores[j] {
                concordant += 1.0;
            } else if risk_scores[i] > risk_scores[j] {
                discordant += 1.0;
            } else {
                concordant += 0.5;
                discordant += 0.5;
            }
        }
    }

    let total = concordant + discordant;
    if total > 0.0 { concordant / total } else { 0.5 }
}

#[pyfunction]
#[pyo3(signature = (
    shap_values,
    time_points,
    n_features,
    aggregation_method=None
))]
pub fn compute_shap_interactions(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    n_features: usize,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<ShapInteractionResult> {
    let n_samples = shap_values.len();
    let n_times = time_points.len();

    if n_samples == 0 || n_features == 0 || n_times == 0 {
        return Ok(ShapInteractionResult {
            interaction_values: vec![vec![vec![0.0; n_times]; n_features]; n_features],
            time_points,
            aggregated_interactions: None,
        });
    }

    let mut interaction_values = vec![vec![vec![0.0; n_times]; n_features]; n_features];

    let mut means = vec![vec![0.0; n_times]; n_features];
    for i in 0..n_features {
        for t in 0..n_times {
            means[i][t] = shap_values.iter().map(|s| s[i][t]).sum::<f64>() / n_samples as f64;
        }
    }

    for t in 0..n_times {
        for i in 0..n_features {
            for j in i..n_features {
                let mut covariance = 0.0;
                let mean_i = means[i][t];
                let mean_j = means[j][t];

                for sample in &shap_values {
                    covariance += (sample[i][t] - mean_i) * (sample[j][t] - mean_j);
                }
                covariance /= n_samples as f64;

                interaction_values[i][j][t] = covariance;
                if i != j {
                    interaction_values[j][i][t] = covariance;
                }
            }
        }
    }

    let aggregated_interactions = aggregation_method.map(|method| {
        let time_diffs: Vec<f64> = if n_times >= 2 {
            time_points.windows(2).map(|w| w[1] - w[0]).collect()
        } else {
            Vec::new()
        };
        let total_time = time_points.last().unwrap_or(&1.0) - time_points.first().unwrap_or(&0.0);
        let time_weights: Vec<f64> = if total_time > 0.0 {
            time_points
                .iter()
                .map(|&t| 1.0 - (t - time_points[0]) / total_time)
                .collect()
        } else {
            Vec::new()
        };

        let mut agg = vec![vec![0.0; n_features]; n_features];
        for i in 0..n_features {
            for j in 0..n_features {
                let values = &interaction_values[i][j];
                agg[i][j] = match method {
                    AggregationMethod::Mean => {
                        values.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                    }
                    AggregationMethod::MaxAbsolute => {
                        values.iter().map(|v| v.abs()).fold(0.0, f64::max)
                    }
                    AggregationMethod::Integral => {
                        if n_times < 2 {
                            values.first().copied().unwrap_or(0.0).abs()
                        } else {
                            let mut integral = 0.0;
                            for k in 0..time_diffs.len() {
                                let avg = (values[k + 1].abs() + values[k].abs()) / 2.0;
                                integral += avg * time_diffs[k];
                            }
                            integral
                        }
                    }
                    AggregationMethod::TimeWeighted => {
                        if total_time <= 0.0 {
                            values.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                        } else {
                            let mut weighted_sum = 0.0;
                            for (k, &weight) in time_weights.iter().enumerate() {
                                weighted_sum += values[k].abs() * weight;
                            }
                            weighted_sum / n_times as f64
                        }
                    }
                };
            }
        }
        agg
    });

    Ok(ShapInteractionResult {
        interaction_values,
        time_points,
        aggregated_interactions,
    })
}

