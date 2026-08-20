

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum PWPTimescale {
    Gap,
    Total,
}

#[pymethods]
impl PWPTimescale {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "gap" => Ok(PWPTimescale::Gap),
            "total" => Ok(PWPTimescale::Total),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown timescale: must be 'gap' or 'total'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PWPConfig {
    #[pyo3(get, set)]
    pub timescale: PWPTimescale,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub stratify_by_event: bool,
    #[pyo3(get, set)]
    pub robust_variance: bool,
}

#[pymethods]
impl PWPConfig {
    #[new]
    #[pyo3(signature = (timescale=PWPTimescale::Gap, max_iter=100, tol=1e-6, stratify_by_event=true, robust_variance=true))]
    pub fn new(
        timescale: PWPTimescale,
        max_iter: usize,
        tol: f64,
        stratify_by_event: bool,
        robust_variance: bool,
    ) -> Self {
        PWPConfig {
            timescale,
            max_iter,
            tol,
            stratify_by_event,
            robust_variance,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PWPResult {
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
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub event_specific_coef: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub baseline_cumhaz: Vec<f64>,
    #[pyo3(get)]
    pub baseline_times: Vec<f64>,
    #[pyo3(get)]
    pub baseline_strata: Vec<i32>,
}

#[pyfunction]
#[pyo3(signature = (id, start, stop, event, event_number, covariates, config))]
pub fn pwp_model(
    id: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    event_number: Vec<i32>,
    covariates: Vec<f64>,
    config: &PWPConfig,
) -> PyResult<PWPResult> {
    let n = id.len();
    validate_recurrent_lengths(n, &[start.len(), stop.len(), event.len(), event_number.len()])?;
    validate_counting_process_inputs(&start, &stop, &event)?;
    validate_event_numbers(&event_number)?;
    validate_recurrent_solver_controls(config.max_iter, config.tol)?;

    let (p, x_mat) = covariate_matrix_or_intercept(covariates, n)?;

    let unique_ids = sorted_unique_i32(&id);
    let n_subjects = unique_ids.len();
    let n_events_total = event.iter().filter(|&&e| e == 1).count();

    let max_event_num = *event_number.iter().max().unwrap_or(&1) as usize;

    let covariate_rows = x_mat
        .chunks_exact(p)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    let (time_var, entry_times) = match config.timescale {
        PWPTimescale::Gap => (
            stop.iter()
                .zip(&start)
                .map(|(&end, &begin)| end - begin)
                .collect(),
            None,
        ),
        PWPTimescale::Total => (stop, Some(start)),
    };
    let fit = coxph_fit(
        time_var,
        event,
        covariate_rows,
        config.stratify_by_event.then_some(event_number),
        None,
        None,
        None,
        Some(config.max_iter),
        Some(config.tol),
        None,
        Some("efron"),
        entry_times,
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

    let event_specific_coef: Vec<Vec<f64>> = (1..=max_event_num).map(|_| beta.clone()).collect();
    let (baseline_times, baseline_cumhaz, baseline_strata) =
        fit.basehaz_with_strata(false)?;

    Ok(PWPResult {
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
        event_specific_coef,
        baseline_cumhaz,
        baseline_times,
        baseline_strata,
    })
}
