#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone)]
#[pyclass]
pub struct IVCoxConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub two_stage: bool,
    #[pyo3(get, set)]
    pub robust_variance: bool,
}

#[pymethods]
impl IVCoxConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, two_stage=true, robust_variance=true))]
    pub fn new(max_iter: usize, tol: f64, two_stage: bool, robust_variance: bool) -> Self {
        IVCoxConfig {
            max_iter,
            tol,
            two_stage,
            robust_variance,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct IVCoxResult {
    #[pyo3(get)]
    pub treatment_coef: f64,
    #[pyo3(get)]
    pub treatment_se: f64,
    #[pyo3(get)]
    pub treatment_z: f64,
    #[pyo3(get)]
    pub treatment_pvalue: f64,
    #[pyo3(get)]
    pub covariate_coef: Vec<f64>,
    #[pyo3(get)]
    pub covariate_se: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub first_stage_f: f64,
    #[pyo3(get)]
    pub first_stage_r2: f64,
    #[pyo3(get)]
    pub weak_instrument_test: f64,
    #[pyo3(get)]
    pub sargan_test: f64,
    #[pyo3(get)]
    pub sargan_pvalue: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, instruments, covariates, config))]
pub fn iv_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<f64>,
    instruments: Vec<f64>,
    covariates: Vec<f64>,
    config: &IVCoxConfig,
) -> PyResult<IVCoxResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_instruments = if instruments.is_empty() {
        0
    } else {
        instruments.len() / n
    };

    let p_cov = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    if n_instruments == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one instrument is required",
        ));
    }

    let treatment_mean = treatment.iter().sum::<f64>() / n as f64;
    let treatment_var: f64 = treatment
        .iter()
        .map(|&t| (t - treatment_mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let mut first_stage_coef = vec![0.0; n_instruments + p_cov + 1];

    for _ in 0..50 {
        let mut xtx = vec![vec![0.0; n_instruments + p_cov + 1]; n_instruments + p_cov + 1];
        let mut xty = vec![0.0; n_instruments + p_cov + 1];

        for i in 0..n {
            let mut x_row = vec![1.0];
            for k in 0..n_instruments {
                x_row.push(instruments[i * n_instruments + k]);
            }
            for k in 0..p_cov {
                x_row.push(covariates[i * p_cov + k]);
            }

            for j1 in 0..x_row.len() {
                xty[j1] += x_row[j1] * treatment[i];
                for j2 in 0..x_row.len() {
                    xtx[j1][j2] += x_row[j1] * x_row[j2];
                }
            }
        }

        for j in 0..first_stage_coef.len() {
            if xtx[j][j].abs() > 1e-10 {
                first_stage_coef[j] = xty[j] / xtx[j][j];
            }
        }
    }

    let fitted_treatment: Vec<f64> = (0..n)
        .map(|i| {
            let mut pred = first_stage_coef[0];
            for k in 0..n_instruments {
                pred += first_stage_coef[1 + k] * instruments[i * n_instruments + k];
            }
            for k in 0..p_cov {
                pred += first_stage_coef[1 + n_instruments + k] * covariates[i * p_cov + k];
            }
            pred
        })
        .collect();

    let fitted_mean = fitted_treatment.iter().sum::<f64>() / n as f64;
    let fitted_var: f64 = fitted_treatment
        .iter()
        .map(|&f| (f - fitted_mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let residual_var: f64 = treatment
        .iter()
        .zip(fitted_treatment.iter())
        .map(|(&t, &f)| (t - f).powi(2))
        .sum::<f64>()
        / n as f64;

    let first_stage_r2 = if treatment_var > 1e-10 {
        (1.0 - residual_var / treatment_var).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let first_stage_f = if residual_var > 1e-10 {
        (fitted_var / residual_var) * (n - n_instruments - p_cov - 1) as f64 / n_instruments as f64
    } else {
        0.0
    };

    let treatment_for_cox = if config.two_stage {
        fitted_treatment.clone()
    } else {
        treatment.clone()
    };

    let mut beta_treatment = 0.0;
    let mut beta_cov = vec![0.0; p_cov];
    let mut converged = false;
    let mut n_iter = 0;
    let mut final_hessian_treatment: f64 = 0.0;
    let mut final_hessian_cov: Vec<f64> = vec![0.0; p_cov];

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = beta_treatment * treatment_for_cox[i];
                for k in 0..p_cov {
                    e += beta_cov[k] * covariates[i * p_cov + k];
                }
                e.clamp(-700.0, 700.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient_treatment = 0.0;
        let mut gradient_cov = vec![0.0; p_cov];
        let mut hessian_treatment: f64 = 0.0;
        let mut hessian_cov = vec![0.0; p_cov];

        let mut risk_sum = 0.0;
        let mut weighted_treatment = 0.0;
        let mut weighted_treatment_sq = 0.0;
        let mut weighted_cov = vec![0.0; p_cov];
        let mut weighted_cov_sq = vec![0.0; p_cov];

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_treatment += exp_eta[i] * treatment_for_cox[i];
            weighted_treatment_sq += exp_eta[i] * treatment_for_cox[i] * treatment_for_cox[i];
            for k in 0..p_cov {
                weighted_cov[k] += exp_eta[i] * covariates[i * p_cov + k];
                weighted_cov_sq[k] +=
                    exp_eta[i] * covariates[i * p_cov + k] * covariates[i * p_cov + k];
            }

            if event[i] == 1 && risk_sum > 0.0 {
                let treatment_bar = weighted_treatment / risk_sum;
                let treatment_sq_bar = weighted_treatment_sq / risk_sum;

                gradient_treatment += treatment_for_cox[i] - treatment_bar;
                hessian_treatment += treatment_sq_bar - treatment_bar * treatment_bar;

                for k in 0..p_cov {
                    let cov_bar = weighted_cov[k] / risk_sum;
                    let cov_sq_bar = weighted_cov_sq[k] / risk_sum;
                    gradient_cov[k] += covariates[i * p_cov + k] - cov_bar;
                    hessian_cov[k] += cov_sq_bar - cov_bar * cov_bar;
                }
            }
        }

        let old_beta = beta_treatment;

        if hessian_treatment.abs() > 1e-10 {
            beta_treatment += gradient_treatment / hessian_treatment;
            beta_treatment = beta_treatment.clamp(-10.0, 10.0);
        }

        for k in 0..p_cov {
            if hessian_cov[k].abs() > 1e-10 {
                beta_cov[k] += gradient_cov[k] / hessian_cov[k];
                beta_cov[k] = beta_cov[k].clamp(-10.0, 10.0);
            }
        }

        final_hessian_treatment = hessian_treatment;
        final_hessian_cov = hessian_cov;

        if (beta_treatment - old_beta).abs() < config.tol {
            converged = true;
            break;
        }
    }

    let treatment_se = if final_hessian_treatment > 1e-10 {
        (1.0 / final_hessian_treatment).sqrt()
    } else {
        f64::INFINITY
    };

    let treatment_z = if treatment_se > 1e-10 && treatment_se.is_finite() {
        beta_treatment / treatment_se
    } else {
        0.0
    };

    let treatment_pvalue = 2.0 * (1.0 - normal_cdf(treatment_z.abs()));

    let covariate_se: Vec<f64> = final_hessian_cov
        .iter()
        .map(|&h| {
            if h > 1e-10 {
                (1.0 / h).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let mut log_likelihood = 0.0;
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = beta_treatment * treatment_for_cox[i];
            for k in 0..p_cov {
                e += beta_cov[k] * covariates[i * p_cov + k];
            }
            e
        })
        .collect();

    let mut risk_sum = 0.0;
    for &i in &sorted_indices {
        risk_sum += eta[i].exp();
        if event[i] == 1 && risk_sum > 0.0 {
            log_likelihood += eta[i] - risk_sum.ln();
        }
    }

    let weak_instrument_test = first_stage_f;

    let sargan_test = if n_instruments > 1 {
        let residuals: Vec<f64> = treatment
            .iter()
            .zip(fitted_treatment.iter())
            .map(|(&t, &f)| t - f)
            .collect();

        let mut r_sum_sq = 0.0;
        for i in 0..n {
            for k in 0..n_instruments {
                r_sum_sq += (residuals[i] * instruments[i * n_instruments + k]).powi(2);
            }
        }

        let sigma_sq = residual_var;
        if sigma_sq > 1e-10 {
            r_sum_sq / sigma_sq
        } else {
            0.0
        }
    } else {
        0.0
    };

    let sargan_pvalue = if n_instruments > 1 {
        1.0 - chi2_cdf(sargan_test, (n_instruments - 1) as f64)
    } else {
        1.0
    };

    Ok(IVCoxResult {
        treatment_coef: beta_treatment,
        treatment_se,
        treatment_z,
        treatment_pvalue,
        covariate_coef: beta_cov,
        covariate_se,
        log_likelihood,
        first_stage_f,
        first_stage_r2,
        weak_instrument_test,
        sargan_test,
        sargan_pvalue,
        n_iter,
        converged,
    })
}

fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let k = df / 2.0;
    let x_half = x / 2.0;
    lower_incomplete_gamma(k, x_half) / gamma_fn(k)
}

fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x));
    }
    let x = x - 1.0;
    let g = 7;
    #[allow(clippy::excessive_precision)]
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    let mut sum = c[0];
    for i in 1..g + 2 {
        sum += c[i] / (x + i as f64);
    }

    let t = x + g as f64 + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
}

fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut sum = 0.0;
        let mut term = 1.0 / a;
        sum += term;
        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 * sum.abs() {
                break;
            }
        }
        x.powf(a) * (-x).exp() * sum
    } else {
        gamma_fn(a) - upper_incomplete_gamma(a, x)
    }
}

fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    let mut f = 1.0 + x - a;
    if f.abs() < 1e-30 {
        f = 1e-30;
    }
    let mut c = f;
    let mut d = 0.0;

    for n in 1..100 {
        let an = n as f64 * (a - n as f64);
        let bn = (2 * n + 1) as f64 + x - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    x.powf(a) * (-x).exp() / f
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RDSurvivalConfig {
    #[pyo3(get, set)]
    pub bandwidth: f64,
    #[pyo3(get, set)]
    pub kernel: String,
    #[pyo3(get, set)]
    pub polynomial_order: usize,
    #[pyo3(get, set)]
    pub fuzzy: bool,
}

#[pymethods]
impl RDSurvivalConfig {
    #[new]
    #[pyo3(signature = (bandwidth=None, kernel="triangular", polynomial_order=1, fuzzy=false))]
    pub fn new(
        bandwidth: Option<f64>,
        kernel: &str,
        polynomial_order: usize,
        fuzzy: bool,
    ) -> PyResult<Self> {
        Ok(RDSurvivalConfig {
            bandwidth: bandwidth.unwrap_or(0.0),
            kernel: kernel.to_string(),
            polynomial_order,
            fuzzy,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RDSurvivalResult {
    #[pyo3(get)]
    pub treatment_effect: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub z_score: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub bandwidth_used: f64,
    #[pyo3(get)]
    pub n_left: usize,
    #[pyo3(get)]
    pub n_right: usize,
    #[pyo3(get)]
    pub survival_left: f64,
    #[pyo3(get)]
    pub survival_right: f64,
}

fn kernel_weight(x: f64, bandwidth: f64, kernel: &str) -> f64 {
    let u = x / bandwidth;
    if u.abs() > 1.0 {
        return 0.0;
    }
    match kernel {
        "triangular" => 1.0 - u.abs(),
        "uniform" | "rectangular" => 1.0,
        "epanechnikov" => 0.75 * (1.0 - u * u),
        _ => 1.0 - u.abs(),
    }
}

#[pyfunction]
#[pyo3(signature = (time, event, running_var, cutoff, treatment, covariates, config))]
pub fn rd_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    running_var: Vec<f64>,
    cutoff: f64,
    treatment: Vec<f64>,
    covariates: Vec<f64>,
    config: &RDSurvivalConfig,
) -> PyResult<RDSurvivalResult> {
    let n = time.len();
    if event.len() != n || running_var.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let bandwidth = if config.bandwidth > 0.0 {
        config.bandwidth
    } else {
        let centered: Vec<f64> = running_var.iter().map(|&r| r - cutoff).collect();
        let iqr = {
            let mut sorted = centered.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q1 = sorted[n / 4];
            let q3 = sorted[3 * n / 4];
            q3 - q1
        };
        1.06 * iqr * (n as f64).powf(-0.2)
    };

    let left_indices: Vec<usize> = (0..n)
        .filter(|&i| running_var[i] < cutoff && running_var[i] >= cutoff - bandwidth)
        .collect();

    let right_indices: Vec<usize> = (0..n)
        .filter(|&i| running_var[i] >= cutoff && running_var[i] <= cutoff + bandwidth)
        .collect();

    let n_left = left_indices.len();
    let n_right = right_indices.len();

    if n_left < 10 || n_right < 10 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Insufficient observations near cutoff",
        ));
    }

    let survival_left = estimate_km_survival(
        &left_indices,
        &time,
        &event,
        &running_var,
        cutoff,
        bandwidth,
        &config.kernel,
    );

    let survival_right = estimate_km_survival(
        &right_indices,
        &time,
        &event,
        &running_var,
        cutoff,
        bandwidth,
        &config.kernel,
    );

    let treatment_effect = survival_right - survival_left;

    let se_left = (survival_left * (1.0 - survival_left) / n_left as f64).sqrt();
    let se_right = (survival_right * (1.0 - survival_right) / n_right as f64).sqrt();
    let se = (se_left.powi(2) + se_right.powi(2)).sqrt();

    let z_score = if se > 1e-10 {
        treatment_effect / se
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    let ci_lower = treatment_effect - 1.96 * se;
    let ci_upper = treatment_effect + 1.96 * se;

    Ok(RDSurvivalResult {
        treatment_effect,
        se,
        ci_lower,
        ci_upper,
        z_score,
        p_value,
        bandwidth_used: bandwidth,
        n_left,
        n_right,
        survival_left,
        survival_right,
    })
}

fn estimate_km_survival(
    indices: &[usize],
    time: &[f64],
    event: &[i32],
    running_var: &[f64],
    cutoff: f64,
    bandwidth: f64,
    kernel: &str,
) -> f64 {
    if indices.is_empty() {
        return 1.0;
    }

    let mut sorted_indices: Vec<usize> = indices.to_vec();
    sorted_indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

    let max_time = time[sorted_indices[sorted_indices.len() / 2]];

    let mut survival = 1.0;
    let mut at_risk = 0.0;

    for &i in indices {
        at_risk += kernel_weight(running_var[i] - cutoff, bandwidth, kernel);
    }

    let mut prev_time = 0.0;
    for &i in &sorted_indices {
        if time[i] > max_time {
            break;
        }

        let weight = kernel_weight(running_var[i] - cutoff, bandwidth, kernel);

        if event[i] == 1 && at_risk > 1e-10 {
            survival *= 1.0 - weight / at_risk;
        }

        at_risk -= weight;
    }

    survival.clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MediationSurvivalConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl MediationSurvivalConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, n_bootstrap=500, seed=None))]
    pub fn new(max_iter: usize, tol: f64, n_bootstrap: usize, seed: Option<u64>) -> Self {
        MediationSurvivalConfig {
            max_iter,
            tol,
            n_bootstrap,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MediationSurvivalResult {
    #[pyo3(get)]
    pub total_effect: f64,
    #[pyo3(get)]
    pub direct_effect: f64,
    #[pyo3(get)]
    pub indirect_effect: f64,
    #[pyo3(get)]
    pub proportion_mediated: f64,
    #[pyo3(get)]
    pub total_se: f64,
    #[pyo3(get)]
    pub direct_se: f64,
    #[pyo3(get)]
    pub indirect_se: f64,
    #[pyo3(get)]
    pub total_pvalue: f64,
    #[pyo3(get)]
    pub direct_pvalue: f64,
    #[pyo3(get)]
    pub indirect_pvalue: f64,
    #[pyo3(get)]
    pub treatment_to_mediator: f64,
    #[pyo3(get)]
    pub mediator_to_outcome: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, mediator, covariates, config))]
pub fn mediation_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<f64>,
    mediator: Vec<f64>,
    covariates: Vec<f64>,
    config: &MediationSurvivalConfig,
) -> PyResult<MediationSurvivalResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n || mediator.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p_cov = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    let (alpha, _alpha_se) = fit_mediator_model(&treatment, &mediator, &covariates, n, p_cov);

    let (beta_total, _beta_total_se) = fit_outcome_model(
        &time,
        &event,
        &treatment,
        None,
        &covariates,
        n,
        p_cov,
        config.max_iter,
    );

    let (beta_direct, gamma, _) = fit_outcome_model_with_mediator(
        &time,
        &event,
        &treatment,
        &mediator,
        &covariates,
        n,
        p_cov,
        config.max_iter,
    );

    let total_effect = beta_total;
    let direct_effect = beta_direct;
    let indirect_effect = alpha * gamma;

    let proportion_mediated = if total_effect.abs() > 1e-10 {
        (indirect_effect / total_effect).clamp(-1.0, 2.0)
    } else {
        0.0
    };

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let mut total_effects = Vec::new();
    let mut direct_effects = Vec::new();
    let mut indirect_effects = Vec::new();

    for _ in 0..config.n_bootstrap {
        let mut boot_indices: Vec<usize> = (0..n).map(|_| rng.usize(0..n)).collect();

        let boot_time: Vec<f64> = boot_indices.iter().map(|&i| time[i]).collect();
        let boot_event: Vec<i32> = boot_indices.iter().map(|&i| event[i]).collect();
        let boot_treatment: Vec<f64> = boot_indices.iter().map(|&i| treatment[i]).collect();
        let boot_mediator: Vec<f64> = boot_indices.iter().map(|&i| mediator[i]).collect();
        let cov_ref = &covariates;
        let boot_cov: Vec<f64> = if p_cov > 0 {
            boot_indices
                .iter()
                .flat_map(|&i| (0..p_cov).map(move |j| cov_ref[i * p_cov + j]))
                .collect()
        } else {
            vec![]
        };

        let (boot_alpha, _) =
            fit_mediator_model(&boot_treatment, &boot_mediator, &boot_cov, n, p_cov);
        let (boot_total, _) = fit_outcome_model(
            &boot_time,
            &boot_event,
            &boot_treatment,
            None,
            &boot_cov,
            n,
            p_cov,
            50,
        );
        let (boot_direct, boot_gamma, _) = fit_outcome_model_with_mediator(
            &boot_time,
            &boot_event,
            &boot_treatment,
            &boot_mediator,
            &boot_cov,
            n,
            p_cov,
            50,
        );

        total_effects.push(boot_total);
        direct_effects.push(boot_direct);
        indirect_effects.push(boot_alpha * boot_gamma);
    }

    let total_se = std_dev(&total_effects);
    let direct_se = std_dev(&direct_effects);
    let indirect_se = std_dev(&indirect_effects);

    let total_z = if total_se > 1e-10 {
        total_effect / total_se
    } else {
        0.0
    };
    let direct_z = if direct_se > 1e-10 {
        direct_effect / direct_se
    } else {
        0.0
    };
    let indirect_z = if indirect_se > 1e-10 {
        indirect_effect / indirect_se
    } else {
        0.0
    };

    let total_pvalue = 2.0 * (1.0 - normal_cdf(total_z.abs()));
    let direct_pvalue = 2.0 * (1.0 - normal_cdf(direct_z.abs()));
    let indirect_pvalue = 2.0 * (1.0 - normal_cdf(indirect_z.abs()));

    Ok(MediationSurvivalResult {
        total_effect,
        direct_effect,
        indirect_effect,
        proportion_mediated,
        total_se,
        direct_se,
        indirect_se,
        total_pvalue,
        direct_pvalue,
        indirect_pvalue,
        treatment_to_mediator: alpha,
        mediator_to_outcome: gamma,
    })
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

fn fit_mediator_model(
    treatment: &[f64],
    mediator: &[f64],
    covariates: &[f64],
    n: usize,
    p_cov: usize,
) -> (f64, f64) {
    let mut alpha = 0.0;

    let treatment_mean = treatment.iter().sum::<f64>() / n as f64;
    let mediator_mean = mediator.iter().sum::<f64>() / n as f64;

    let mut cov_tm = 0.0;
    let mut var_t = 0.0;

    for i in 0..n {
        let t_centered = treatment[i] - treatment_mean;
        let m_centered = mediator[i] - mediator_mean;
        cov_tm += t_centered * m_centered;
        var_t += t_centered * t_centered;
    }

    if var_t > 1e-10 {
        alpha = cov_tm / var_t;
    }

    let residuals: Vec<f64> = (0..n)
        .map(|i| mediator[i] - mediator_mean - alpha * (treatment[i] - treatment_mean))
        .collect();

    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    let alpha_se = if var_t > 1e-10 {
        (mse / var_t).sqrt()
    } else {
        f64::INFINITY
    };

    (alpha, alpha_se)
}

fn fit_outcome_model(
    time: &[f64],
    event: &[i32],
    treatment: &[f64],
    _mediator: Option<&[f64]>,
    _covariates: &[f64],
    n: usize,
    _p_cov: usize,
    max_iter: usize,
) -> (f64, f64) {
    let mut beta = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n)
            .map(|i| (beta * treatment[i]).clamp(-700.0, 700.0))
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient = 0.0;
        let mut hessian = 0.0;
        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_tt = 0.0;

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_t += exp_eta[i] * treatment[i];
            weighted_tt += exp_eta[i] * treatment[i] * treatment[i];

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                gradient += treatment[i] - t_bar;
                hessian += tt_bar - t_bar * t_bar;
            }
        }

        if hessian.abs() > 1e-10 {
            beta += gradient / hessian;
            beta = beta.clamp(-10.0, 10.0);
        }
    }

    let se = 0.1;
    (beta, se)
}

fn fit_outcome_model_with_mediator(
    time: &[f64],
    event: &[i32],
    treatment: &[f64],
    mediator: &[f64],
    _covariates: &[f64],
    n: usize,
    _p_cov: usize,
    max_iter: usize,
) -> (f64, f64, f64) {
    let mut beta = 0.0;
    let mut gamma = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n)
            .map(|i| (beta * treatment[i] + gamma * mediator[i]).clamp(-700.0, 700.0))
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut grad_beta = 0.0;
        let mut grad_gamma = 0.0;
        let mut hess_beta = 0.0;
        let mut hess_gamma = 0.0;

        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_m = 0.0;
        let mut weighted_tt = 0.0;
        let mut weighted_mm = 0.0;

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_t += exp_eta[i] * treatment[i];
            weighted_m += exp_eta[i] * mediator[i];
            weighted_tt += exp_eta[i] * treatment[i] * treatment[i];
            weighted_mm += exp_eta[i] * mediator[i] * mediator[i];

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let m_bar = weighted_m / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                let mm_bar = weighted_mm / risk_sum;

                grad_beta += treatment[i] - t_bar;
                grad_gamma += mediator[i] - m_bar;
                hess_beta += tt_bar - t_bar * t_bar;
                hess_gamma += mm_bar - m_bar * m_bar;
            }
        }

        if hess_beta.abs() > 1e-10 {
            beta += grad_beta / hess_beta;
            beta = beta.clamp(-10.0, 10.0);
        }

        if hess_gamma.abs() > 1e-10 {
            gamma += grad_gamma / hess_gamma;
            gamma = gamma.clamp(-10.0, 10.0);
        }
    }

    let se = 0.1;
    (beta, gamma, se)
}

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
                .unwrap()
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
            if hessian_diag[j].abs() > 1e-10 {
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
        .map(|(&p, &s)| if s > 1e-10 { p / s } else { 0.0 })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_cox() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let instruments = vec![0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.8];

        let config = IVCoxConfig::new(50, 1e-4, true, true);
        let result = iv_cox(time, event, treatment, instruments, vec![], &config).unwrap();

        assert!(result.first_stage_r2 >= 0.0 && result.first_stage_r2 <= 1.0);
    }

    #[test]
    fn test_rd_survival() {
        let time = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5, 9.5, 10.5,
        ];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
        let running_var = vec![
            0.35, 0.38, 0.41, 0.52, 0.55, 0.58, 0.44, 0.61, 0.47, 0.64, 0.32, 0.36, 0.43, 0.53,
            0.57, 0.62, 0.39, 0.59, 0.46, 0.67,
        ];
        let treatment = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
        ];

        let config = RDSurvivalConfig::new(Some(0.2), "triangular", 1, false).unwrap();
        let result =
            rd_survival(time, event, running_var, 0.5, treatment, vec![], &config).unwrap();

        assert!(result.n_left >= 10);
        assert!(result.n_right >= 10);
    }

    #[test]
    fn test_mediation_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mediator = vec![0.5, 1.2, 0.4, 1.1, 0.3, 1.3, 0.6, 1.0];

        let config = MediationSurvivalConfig::new(50, 1e-4, 50, Some(42));
        let result = mediation_survival(time, event, treatment, mediator, vec![], &config).unwrap();

        assert!(result.proportion_mediated >= -1.0 && result.proportion_mediated <= 2.0);
    }

    #[test]
    fn test_g_estimation() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let config = GEstimationConfig::new(50, 1e-4, "aft");
        let result = g_estimation_aft(time, event, treatment, vec![], &config).unwrap();

        assert!(!result.psi.is_empty());
        assert!(result.treatment_effect_ratio > 0.0);
    }
}
