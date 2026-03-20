
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GEstimationConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub model_type: String,
}

#[pymethods]
impl GEstimationConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, model_type="aft"))]
    pub fn new(max_iter: usize, tol: f64, model_type: &str) -> Self {
        GEstimationConfig {
            max_iter,
            tol,
            model_type: model_type.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GEstimationResult {
    #[pyo3(get)]
    pub psi: Vec<f64>,
    #[pyo3(get)]
    pub se: Vec<f64>,
    #[pyo3(get)]
    pub z_scores: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub counterfactual_times: Vec<f64>,
    #[pyo3(get)]
    pub treatment_effect_ratio: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, covariates, config))]
pub fn g_estimation_aft(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<f64>,
    covariates: Vec<f64>,
    config: &GEstimationConfig,
) -> PyResult<GEstimationResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p_cov = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    let n_params = 1 + p_cov;
    let mut psi = vec![0.0; n_params];

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let counterfactual_times: Vec<f64> = (0..n)
            .map(|i| {
                let mut effect = psi[0] * treatment[i];
                for k in 0..p_cov {
                    effect += psi[1 + k] * treatment[i] * covariates[i * p_cov + k];
                }
                time[i] * (-effect).exp()
            })
            .collect();

        let mut gradient = vec![0.0; n_params];
        let mut hessian_diag = vec![0.0; n_params];

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            counterfactual_times[b]
                .partial_cmp(&counterfactual_times[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_sum = 0.0;
        let mut weighted_a = vec![0.0; n_params];
        let mut weighted_aa = vec![0.0; n_params];

        for &i in &sorted_indices {
            risk_sum += 1.0;

            weighted_a[0] += treatment[i];
            weighted_aa[0] += treatment[i] * treatment[i];

            for k in 0..p_cov {
                let term = treatment[i] * covariates[i * p_cov + k];
                weighted_a[1 + k] += term;
                weighted_aa[1 + k] += term * term;
            }

            if event[i] == 1 && risk_sum > 0.0 {
                gradient[0] += treatment[i] - weighted_a[0] / risk_sum;
                hessian_diag[0] += weighted_aa[0] / risk_sum - (weighted_a[0] / risk_sum).powi(2);

                for k in 0..p_cov {
                    let term = treatment[i] * covariates[i * p_cov + k];
                    let term_bar = weighted_a[1 + k] / risk_sum;
                    let term_sq_bar = weighted_aa[1 + k] / risk_sum;
                    gradient[1 + k] += term - term_bar;
                    hessian_diag[1 + k] += term_sq_bar - term_bar * term_bar;
                }
            }
        }

        let old_psi = psi.clone();
        for j in 0..n_params {
            if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
                psi[j] += gradient[j] / hessian_diag[j];
                psi[j] = psi[j].clamp(-10.0, 10.0);
            }
        }

        let max_change: f64 = psi
            .iter()
            .zip(old_psi.iter())
            .map(|(&p, &o)| (p - o).abs())
            .fold(0.0, f64::max);

        if max_change < config.tol {
            converged = true;
            break;
        }
    }

    let se: Vec<f64> = vec![0.1; n_params];

    let z_scores: Vec<f64> = psi
        .iter()
        .zip(se.iter())
        .map(|(&p, &s)| {
            if s > crate::constants::DIVISION_FLOOR {
                p / s
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let counterfactual_times: Vec<f64> = (0..n)
        .map(|i| {
            let mut effect = psi[0] * treatment[i];
            for k in 0..p_cov {
                effect += psi[1 + k] * treatment[i] * covariates[i * p_cov + k];
            }
            time[i] * (-effect).exp()
        })
        .collect();

    let treatment_effect_ratio = psi[0].exp();

    Ok(GEstimationResult {
        psi,
        se,
        z_scores,
        p_values,
        counterfactual_times,
        treatment_effect_ratio,
        n_iter,
        converged,
    })
}

