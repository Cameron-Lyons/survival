
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct OutlierDetectionResult {
    #[pyo3(get)]
    pub martingale_residuals: Vec<f64>,
    #[pyo3(get)]
    pub deviance_residuals: Vec<f64>,
    #[pyo3(get)]
    pub standardized_deviance: Vec<f64>,
    #[pyo3(get)]
    pub outlier_indices: Vec<usize>,
    #[pyo3(get)]
    pub extreme_survivor_indices: Vec<usize>,
    #[pyo3(get)]
    pub outlier_scores: Vec<f64>,
    #[pyo3(get)]
    pub threshold: f64,
    #[pyo3(get)]
    pub n_outliers: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients, outlier_threshold=3.0))]
pub fn outlier_detection_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    outlier_threshold: f64,
) -> PyResult<OutlierDetectionResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * coefficients[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut cumulative_hazard = vec![0.0; n];
    let mut risk_sum = 0.0;

    for &i in &sorted_indices {
        risk_sum += exp_eta[i];

        if event[i] == 1 && risk_sum > 0.0 {
            let h0_increment = 1.0 / risk_sum;

            for k in 0..n {
                if time[k] >= time[i] {
                    cumulative_hazard[k] += exp_eta[k] * h0_increment;
                }
            }
        }
    }

    let martingale_residuals: Vec<f64> = (0..n)
        .map(|i| event[i] as f64 - cumulative_hazard[i])
        .collect();

    let deviance_residuals: Vec<f64> = (0..n)
        .map(|i| {
            let m = martingale_residuals[i];
            let d = event[i] as f64;
            let sign = if m >= 0.0 { 1.0 } else { -1.0 };

            let dev_sq = -2.0 * (m + d * (d - m).max(1e-10).ln());
            sign * dev_sq.max(0.0).sqrt()
        })
        .collect();

    let mean_dev: f64 = deviance_residuals.iter().sum::<f64>() / n as f64;
    let std_dev: f64 = (deviance_residuals
        .iter()
        .map(|&d| (d - mean_dev).powi(2))
        .sum::<f64>()
        / (n - 1) as f64)
        .sqrt()
        .max(1e-10);

    let standardized_deviance: Vec<f64> = deviance_residuals
        .iter()
        .map(|&d| (d - mean_dev) / std_dev)
        .collect();

    let outlier_indices: Vec<usize> = (0..n)
        .filter(|&i| standardized_deviance[i].abs() > outlier_threshold)
        .collect();

    let extreme_survivor_indices: Vec<usize> = (0..n)
        .filter(|&i| event[i] == 0 && martingale_residuals[i] < -2.0)
        .collect();

    let outlier_scores: Vec<f64> = standardized_deviance.iter().map(|&d| d.abs()).collect();

    Ok(OutlierDetectionResult {
        martingale_residuals,
        deviance_residuals,
        standardized_deviance,
        outlier_indices: outlier_indices.clone(),
        extreme_survivor_indices,
        outlier_scores,
        threshold: outlier_threshold,
        n_outliers: outlier_indices.len(),
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ModelInfluenceResult {
    #[pyo3(get)]
    pub cooks_distance: Vec<f64>,
    #[pyo3(get)]
    pub covratio: Vec<f64>,
    #[pyo3(get)]
    pub dffits: Vec<f64>,
    #[pyo3(get)]
    pub likelihood_displacement: Vec<f64>,
    #[pyo3(get)]
    pub influential_by_cooks: Vec<usize>,
    #[pyo3(get)]
    pub influential_by_covratio: Vec<usize>,
    #[pyo3(get)]
    pub influential_by_dffits: Vec<usize>,
    #[pyo3(get)]
    pub overall_influential: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients))]
pub fn model_influence_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
) -> PyResult<ModelInfluenceResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let dfbeta_result = dfbeta_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        None,
    )?;

    let leverage_result = leverage_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        2.0,
    )?;

    let coef_var = estimate_coefficient_variance(&dfbeta_result.dfbeta, n_covariates);

    let mut cooks_distance = vec![0.0; n];
    for (i, cook_dist) in cooks_distance.iter_mut().enumerate().take(n) {
        let mut cook = 0.0;
        for (j, &coef_var_j) in coef_var.iter().enumerate().take(n_covariates) {
            if coef_var_j > 1e-10 {
                cook += dfbeta_result.dfbeta[i][j].powi(2) / coef_var_j;
            }
        }
        *cook_dist = cook / n_covariates as f64;
    }

    let mut covratio = vec![1.0; n];
    for (i, covratio_i) in covratio.iter_mut().enumerate().take(n) {
        let h_i = leverage_result.leverage[i];
        if h_i < 1.0 {
            *covratio_i = 1.0 / (1.0 - h_i).powf(n_covariates as f64);
        }
    }

    let mut dffits = vec![0.0; n];
    for (i, dffits_i) in dffits.iter_mut().enumerate().take(n) {
        let h_i = leverage_result.leverage[i];
        if h_i > 0.0 && h_i < 1.0 {
            let sum_dfbeta_sq: f64 = dfbeta_result.dfbeta[i].iter().map(|&d| d * d).sum();
            *dffits_i = sum_dfbeta_sq.sqrt() * (h_i / (1.0 - h_i)).sqrt();
        }
    }

    let mut likelihood_displacement = vec![0.0; n];
    for i in 0..n {
        likelihood_displacement[i] = cooks_distance[i] * n_covariates as f64;
    }

    let cooks_threshold = 4.0 / n as f64;
    let covratio_threshold = 1.0 + 3.0 * n_covariates as f64 / n as f64;
    let dffits_threshold = 2.0 * ((n_covariates as f64 + 1.0) / n as f64).sqrt();

    let influential_by_cooks: Vec<usize> = (0..n)
        .filter(|&i| cooks_distance[i] > cooks_threshold)
        .collect();

    let influential_by_covratio: Vec<usize> = (0..n)
        .filter(|&i| covratio[i] > covratio_threshold || covratio[i] < 1.0 / covratio_threshold)
        .collect();

    let influential_by_dffits: Vec<usize> = (0..n)
        .filter(|&i| dffits[i].abs() > dffits_threshold)
        .collect();

    let mut overall_influential: Vec<usize> = influential_by_cooks
        .iter()
        .chain(influential_by_covratio.iter())
        .chain(influential_by_dffits.iter())
        .cloned()
        .collect();
    overall_influential.sort_unstable();
    overall_influential.dedup();

    Ok(ModelInfluenceResult {
        cooks_distance,
        covratio,
        dffits,
        likelihood_displacement,
        influential_by_cooks,
        influential_by_covratio,
        influential_by_dffits,
        overall_influential,
        n_obs: n,
    })
}

fn estimate_coefficient_variance(dfbeta: &[Vec<f64>], n_covariates: usize) -> Vec<f64> {
    let n = dfbeta.len();
    let mut var = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let mean: f64 = dfbeta.iter().map(|row| row[j]).sum::<f64>() / n as f64;
        var[j] = dfbeta
            .iter()
            .map(|row| (row[j] - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
    }

    var
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GofTestResult {
    #[pyo3(get)]
    pub global_test_stat: f64,
    #[pyo3(get)]
    pub global_p_value: f64,
    #[pyo3(get)]
    pub variable_test_stats: Vec<f64>,
    #[pyo3(get)]
    pub variable_p_values: Vec<f64>,
    #[pyo3(get)]
    pub linear_test_stat: f64,
    #[pyo3(get)]
    pub linear_p_value: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, coefficients))]
pub fn goodness_of_fit_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
) -> PyResult<GofTestResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let outlier_result = outlier_detection_cox(
        time.clone(),
        event.clone(),
        covariates.clone(),
        n_covariates,
        coefficients.clone(),
        3.0,
    )?;

    let n_events = event.iter().filter(|&&e| e == 1).count();

    let chi_sq: f64 = outlier_result
        .deviance_residuals
        .iter()
        .map(|&d| d * d)
        .sum();

    let df = n_events - n_covariates;
    let global_p_value = chi_sq_p_value(chi_sq, df);

    let mut variable_test_stats = vec![0.0; n_covariates];
    let mut variable_p_values = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let mut corr_sum = 0.0;
        for i in 0..n {
            corr_sum += covariates[i * n_covariates + j] * outlier_result.martingale_residuals[i];
        }
        let test_stat = corr_sum.powi(2);
        variable_test_stats[j] = test_stat;
        variable_p_values[j] = chi_sq_p_value(test_stat, 1);
    }

    let mut linear_corr = 0.0;
    for i in 0..n {
        let eta: f64 = (0..n_covariates)
            .map(|j| covariates[i * n_covariates + j] * coefficients[j])
            .sum();
        linear_corr += eta * outlier_result.martingale_residuals[i];
    }
    let linear_test_stat = linear_corr.powi(2);
    let linear_p_value = chi_sq_p_value(linear_test_stat, 1);

    Ok(GofTestResult {
        global_test_stat: chi_sq,
        global_p_value,
        variable_test_stats,
        variable_p_values,
        linear_test_stat,
        linear_p_value,
        df,
        n_obs: n,
    })
}

fn chi_sq_p_value(chi_sq: f64, df: usize) -> f64 {
    if df == 0 {
        return 0.0;
    }
    1.0 - lower_incomplete_gamma(df as f64 / 2.0, chi_sq / 2.0)
}

