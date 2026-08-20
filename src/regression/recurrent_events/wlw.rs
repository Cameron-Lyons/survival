
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct WLWConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub robust_variance: bool,
    #[pyo3(get, set)]
    pub common_baseline: bool,
}

#[pymethods]
impl WLWConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, robust_variance=true, common_baseline=false))]
    pub fn new(max_iter: usize, tol: f64, robust_variance: bool, common_baseline: bool) -> Self {
        WLWConfig {
            max_iter,
            tol,
            robust_variance,
            common_baseline,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct WLWResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub robust_std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_strata: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub stratum_coef: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub global_test_stat: f64,
    #[pyo3(get)]
    pub global_test_pvalue: f64,
}

#[pyfunction]
#[pyo3(signature = (id, time, event, stratum, covariates, config))]
pub fn wlw_model(
    id: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<i32>,
    covariates: Vec<f64>,
    config: &WLWConfig,
) -> PyResult<WLWResult> {
    let n = id.len();
    validate_recurrent_lengths(n, &[time.len(), event.len(), stratum.len()])?;
    validate_time_event_inputs(&time, &event)?;
    validate_recurrent_solver_controls(config.max_iter, config.tol)?;

    let (p, x_mat) = covariate_matrix_or_intercept(covariates, n)?;

    let unique_ids = sorted_unique_i32(&id);
    let n_subjects = unique_ids.len();

    let unique_strata = sorted_unique_i32(&stratum);
    let n_strata = unique_strata.len();

    let n_events_total = event.iter().filter(|&&e| e == 1).count();

    let covariate_rows = x_mat
        .chunks_exact(p)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    let fit = coxph_fit(
        time,
        event,
        covariate_rows,
        (!config.common_baseline).then_some(stratum),
        None,
        None,
        None,
        Some(config.max_iter),
        Some(config.tol),
        None,
        Some("efron"),
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
    let robust_covariance = clustered_sandwich_variance(
        fit.score_residuals()?,
        vec![1.0; n],
        clusters,
        covariance.clone(),
    )?;

    let std_errors = (0..p)
        .map(|idx| {
            covariance
                .get(idx)
                .and_then(|row| row.get(idx))
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
                .sqrt()
        })
        .collect::<Vec<_>>();
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

    let se_to_use = if config.robust_variance {
        &robust_std_errors
    } else {
        &std_errors
    };

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(se_to_use.iter())
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

    let (hr_lower, hr_upper) = exp_ci_bounds_95(&beta, se_to_use);

    let stratum_coef: Vec<Vec<f64>> = unique_strata.iter().map(|_| beta.clone()).collect();

    let test_covariance = if config.robust_variance {
        &robust_covariance
    } else {
        &covariance
    };
    let global_test_stat = invert_matrix(test_covariance)
        .map(|inverse| {
            beta.iter()
                .enumerate()
                .map(|(row, &left)| {
                    left * beta
                        .iter()
                        .enumerate()
                        .map(|(column, &right)| inverse[row][column] * right)
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .unwrap_or_else(|| z_scores.iter().map(|&z| z * z).sum())
        .max(0.0);
    let global_test_pvalue = 1.0 - chi2_cdf(global_test_stat, p as f64);

    Ok(WLWResult {
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
        n_strata,
        n_iter: fit.iterations,
        converged: fit.convergence_flag != CONVERGENCE_FLAG,
        stratum_coef,
        global_test_stat,
        global_test_pvalue,
    })
}
