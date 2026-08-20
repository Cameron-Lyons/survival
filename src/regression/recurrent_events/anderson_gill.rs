
#[pyfunction]
#[pyo3(signature = (id, start, stop, event, covariates, max_iter=100, tol=1e-6))]
pub fn anderson_gill_model(
    id: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<AndersonGillResult> {
    let n = id.len();
    validate_recurrent_lengths(n, &[start.len(), stop.len(), event.len()])?;
    validate_counting_process_inputs(&start, &stop, &event)?;
    validate_recurrent_solver_controls(max_iter, tol)?;

    let (p, x_mat) = covariate_matrix_or_intercept(covariates, n)?;

    let unique_ids = sorted_unique_i32(&id);
    let n_subjects = unique_ids.len();
    let n_events_total = event.iter().filter(|&&value| value == 1).count();
    let total_time: f64 = stop
        .iter()
        .zip(&start)
        .map(|(&end, &begin)| end - begin)
        .sum();
    let mean_event_rate = n_events_total as f64 / total_time;
    let covariate_rows = x_mat
        .chunks_exact(p)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    let fit = coxph_fit(
        stop,
        event,
        covariate_rows,
        None,
        None,
        None,
        None,
        Some(max_iter),
        Some(tol),
        None,
        Some("efron"),
        Some(start),
        None,
        None,
        None,
    )?;
    let beta = fit.coefficients.first().cloned().unwrap_or_default();
    let covariance = fit.information_matrix.clone();

    let id_to_idx = index_by_i32(&unique_ids);
    let clusters = id
        .iter()
        .map(|value| id_to_idx[value])
        .collect::<Vec<_>>();
    let score_residuals = fit.score_residuals()?;

    let std_errors = (0..p)
        .map(|idx| {
            let variance = covariance
                .get(idx)
                .and_then(|row| row.get(idx))
                .copied()
                .unwrap_or(0.0);
            if variance > 0.0 {
                variance.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect::<Vec<_>>();
    let robust_covariance = clustered_sandwich_variance(
        score_residuals,
        vec![1.0; n],
        clusters,
        covariance,
    )?;
    let robust_std_errors = (0..p)
        .map(|idx| {
            robust_covariance
                .get(idx)
                .and_then(|row| row.get(idx))
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
                .sqrt()
        })
        .collect::<Vec<_>>();

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(robust_std_errors.iter())
        .map(|(&b, &se)| {
            if se > DIVISION_FLOOR {
                b / se
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let (hr_lower, hr_upper) = exp_ci_bounds_95(&beta, &robust_std_errors);

    Ok(AndersonGillResult {
        coef: beta,
        std_errors,
        robust_std_errors,
        z_scores,
        p_values,
        hazard_ratios,
        hr_lower,
        hr_upper,
        log_likelihood: fit.log_likelihood.last().copied().unwrap_or(0.0),
        n_events: n_events_total,
        n_subjects,
        n_iter: fit.iterations,
        converged: fit.convergence_flag != CONVERGENCE_FLAG,
        mean_event_rate,
    })
}
