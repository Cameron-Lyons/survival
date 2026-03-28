
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DiffusionSurvivalConfig {
    #[pyo3(get, set)]
    pub diffusion_rate: f64,
    #[pyo3(get, set)]
    pub recovery_rate: f64,
    #[pyo3(get, set)]
    pub susceptibility_covariate: bool,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl DiffusionSurvivalConfig {
    #[new]
    #[pyo3(signature = (diffusion_rate=0.1, recovery_rate=0.05, susceptibility_covariate=true, max_iter=100, tol=1e-6))]
    pub fn new(
        diffusion_rate: f64,
        recovery_rate: f64,
        susceptibility_covariate: bool,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        DiffusionSurvivalConfig {
            diffusion_rate,
            recovery_rate,
            susceptibility_covariate,
            max_iter,
            tol,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DiffusionSurvivalResult {
    #[pyo3(get)]
    pub diffusion_rate: f64,
    #[pyo3(get)]
    pub diffusion_rate_se: f64,
    #[pyo3(get)]
    pub recovery_rate: f64,
    #[pyo3(get)]
    pub recovery_rate_se: f64,
    #[pyo3(get)]
    pub susceptibility_coef: f64,
    #[pyo3(get)]
    pub susceptibility_se: f64,
    #[pyo3(get)]
    pub infection_probabilities: Vec<f64>,
    #[pyo3(get)]
    pub expected_infection_times: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub r0: f64,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (infection_time, infected, covariates, n_covariates, adjacency_matrix, n_nodes, config))]
pub fn diffusion_survival_model(
    infection_time: Vec<f64>,
    infected: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    adjacency_matrix: Vec<f64>,
    n_nodes: usize,
    config: &DiffusionSurvivalConfig,
) -> PyResult<DiffusionSurvivalResult> {
    let n = infection_time.len();
    if infected.len() != n || n != n_nodes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "infection_time and infected must have length n_nodes",
        ));
    }

    let mut beta = config.diffusion_rate;
    let gamma = config.recovery_rate;
    let mut susceptibility = 0.0;
    let mut converged = false;
    let mut log_lik = f64::NEG_INFINITY;

    for _iter in 0..config.max_iter {
        let (hazards, cumulative_hazards) = compute_infection_hazards(
            &infection_time,
            &infected,
            &covariates,
            n_covariates,
            &adjacency_matrix,
            n_nodes,
            beta,
            susceptibility,
            config,
        );

        let new_log_lik = if n > PARALLEL_THRESHOLD_MEDIUM {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let event_term = if infected[i] == 1 {
                        hazards[i].max(1e-10).ln()
                    } else {
                        0.0
                    };
                    event_term - cumulative_hazards[i]
                })
                .sum()
        } else {
            let mut acc = 0.0;
            for i in 0..n {
                if infected[i] == 1 {
                    acc += hazards[i].max(1e-10).ln();
                }
                acc -= cumulative_hazards[i];
            }
            acc
        };

        let (grad_beta, hess_beta) = compute_beta_derivatives(
            &infection_time,
            &infected,
            &adjacency_matrix,
            n_nodes,
            beta,
            susceptibility,
            &covariates,
            n_covariates,
            config,
        );

        if hess_beta.abs() > 1e-10 {
            let update = grad_beta / (-hess_beta).max(1e-10);
            beta += 0.1 * update;
            beta = beta.clamp(0.001, 10.0);
        }

        if config.susceptibility_covariate && n_covariates > 0 {
            let (grad_s, hess_s) = compute_susceptibility_derivatives(
                &infection_time,
                &infected,
                &adjacency_matrix,
                n_nodes,
                beta,
                susceptibility,
                &covariates,
                n_covariates,
            );
            if hess_s.abs() > 1e-10 {
                let update = grad_s / (-hess_s).max(1e-10);
                susceptibility += 0.1 * update;
                susceptibility = susceptibility.clamp(-5.0, 5.0);
            }
        }

        if (new_log_lik - log_lik).abs() < config.tol {
            converged = true;
            log_lik = new_log_lik;
            break;
        }
        log_lik = new_log_lik;
    }

    let degree_sum: f64 = if n_nodes > PARALLEL_THRESHOLD_MEDIUM {
        (0..n_nodes)
            .into_par_iter()
            .map(|i| {
                (0..n_nodes)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum::<f64>()
            })
            .sum()
    } else {
        (0..n_nodes)
            .map(|i| {
                (0..n_nodes)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum::<f64>()
            })
            .sum()
    };
    let avg_degree = degree_sum / n_nodes as f64;
    let r0 = beta * avg_degree / gamma.max(0.01);

    let infection_probabilities: Vec<f64> = if n_nodes > PARALLEL_THRESHOLD_MEDIUM {
        (0..n_nodes)
            .into_par_iter()
            .map(|i| {
                let neighbors_infected: f64 = (0..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0 && infected[j] == 1)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum();
                (1.0 - (-beta * neighbors_infected).exp()).clamp(0.0, 1.0)
            })
            .collect()
    } else {
        (0..n_nodes)
            .map(|i| {
                let neighbors_infected: f64 = (0..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0 && infected[j] == 1)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum();
                (1.0 - (-beta * neighbors_infected).exp()).clamp(0.0, 1.0)
            })
            .collect()
    };

    let expected_infection_times: Vec<f64> = if n_nodes > PARALLEL_THRESHOLD_MEDIUM {
        (0..n_nodes)
            .into_par_iter()
            .map(|i| {
                let neighbors_infected: f64 = (0..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0 && infected[j] == 1)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum();
                if neighbors_infected > 0.0 {
                    1.0 / (beta * neighbors_infected)
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    } else {
        (0..n_nodes)
            .map(|i| {
                let neighbors_infected: f64 = (0..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0 && infected[j] == 1)
                    .map(|j| adjacency_matrix[i * n_nodes + j])
                    .sum();
                if neighbors_infected > 0.0 {
                    1.0 / (beta * neighbors_infected)
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    };

    let beta_se = 0.1 * beta;
    let gamma_se = 0.1 * gamma;
    let susceptibility_se = 0.1;

    Ok(DiffusionSurvivalResult {
        diffusion_rate: beta,
        diffusion_rate_se: beta_se,
        recovery_rate: gamma,
        recovery_rate_se: gamma_se,
        susceptibility_coef: susceptibility,
        susceptibility_se,
        infection_probabilities,
        expected_infection_times,
        log_likelihood: log_lik,
        r0,
        converged,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_infection_hazards(
    infection_time: &[f64],
    infected: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    adjacency: &[f64],
    n: usize,
    beta: f64,
    susceptibility: f64,
    config: &DiffusionSurvivalConfig,
) -> (Vec<f64>, Vec<f64>) {
    let max_time = infection_time.iter().cloned().fold(0.0_f64, f64::max);
    if n > PARALLEL_THRESHOLD_MEDIUM {
        let pairs: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let t_i = infection_time[i];
                let suscept_mult = if config.susceptibility_covariate && n_covariates > 0 {
                    let x_i = covariates[i * n_covariates];
                    (susceptibility * x_i).exp()
                } else {
                    1.0
                };

                let mut n_infected_neighbors = 0.0;
                for j in 0..n {
                    if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                        n_infected_neighbors += adjacency[i * n + j];
                    }
                }

                let h = beta * n_infected_neighbors * suscept_mult;
                (h, h * t_i.min(max_time))
            })
            .collect();
        pairs.into_iter().unzip()
    } else {
        let mut hazards = vec![0.0; n];
        let mut cumulative_hazards = vec![0.0; n];
        for i in 0..n {
            let t_i = infection_time[i];
            let suscept_mult = if config.susceptibility_covariate && n_covariates > 0 {
                let x_i = covariates[i * n_covariates];
                (susceptibility * x_i).exp()
            } else {
                1.0
            };

            let mut n_infected_neighbors = 0.0;
            for j in 0..n {
                if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                    n_infected_neighbors += adjacency[i * n + j];
                }
            }

            let h = beta * n_infected_neighbors * suscept_mult;
            hazards[i] = h;
            cumulative_hazards[i] = h * t_i.min(max_time);
        }
        (hazards, cumulative_hazards)
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_beta_derivatives(
    infection_time: &[f64],
    infected: &[i32],
    adjacency: &[f64],
    n: usize,
    beta: f64,
    susceptibility: f64,
    covariates: &[f64],
    n_covariates: usize,
    config: &DiffusionSurvivalConfig,
) -> (f64, f64) {
    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let t_i = infection_time[i];
                let suscept_mult = if config.susceptibility_covariate && n_covariates > 0 {
                    let x_i = covariates[i * n_covariates];
                    (susceptibility * x_i).exp()
                } else {
                    1.0
                };

                let mut n_infected_neighbors = 0.0;
                for j in 0..n {
                    if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                        n_infected_neighbors += adjacency[i * n + j];
                    }
                }

                let h = beta * n_infected_neighbors * suscept_mult;
                let mut grad = 0.0;
                if infected[i] == 1 && h > 1e-10 {
                    grad += n_infected_neighbors * suscept_mult / h;
                }
                grad -= n_infected_neighbors * suscept_mult * t_i;
                let hess = -(n_infected_neighbors * suscept_mult).powi(2) * t_i;
                (grad, hess)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut gradient = 0.0;
        let mut hessian = 0.0;
        for i in 0..n {
            let t_i = infection_time[i];
            let suscept_mult = if config.susceptibility_covariate && n_covariates > 0 {
                let x_i = covariates[i * n_covariates];
                (susceptibility * x_i).exp()
            } else {
                1.0
            };

            let mut n_infected_neighbors = 0.0;
            for j in 0..n {
                if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                    n_infected_neighbors += adjacency[i * n + j];
                }
            }

            let h = beta * n_infected_neighbors * suscept_mult;
            if infected[i] == 1 && h > 1e-10 {
                gradient += n_infected_neighbors * suscept_mult / h;
            }
            gradient -= n_infected_neighbors * suscept_mult * t_i;
            hessian -= (n_infected_neighbors * suscept_mult).powi(2) * t_i;
        }
        (gradient, hessian)
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_susceptibility_derivatives(
    infection_time: &[f64],
    infected: &[i32],
    adjacency: &[f64],
    n: usize,
    beta: f64,
    susceptibility: f64,
    covariates: &[f64],
    n_covariates: usize,
) -> (f64, f64) {
    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let t_i = infection_time[i];
                let x_i = if n_covariates > 0 {
                    covariates[i * n_covariates]
                } else {
                    0.0
                };
                let suscept_mult = (susceptibility * x_i).exp();

                let mut n_infected_neighbors = 0.0;
                for j in 0..n {
                    if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                        n_infected_neighbors += adjacency[i * n + j];
                    }
                }

                let mut grad = 0.0;
                if infected[i] == 1 {
                    grad += x_i;
                }
                grad -= beta * n_infected_neighbors * suscept_mult * x_i * t_i;
                let hess = -beta * n_infected_neighbors * suscept_mult * x_i.powi(2) * t_i;
                (grad, hess)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut gradient = 0.0;
        let mut hessian = 0.0;
        for i in 0..n {
            let t_i = infection_time[i];
            let x_i = if n_covariates > 0 {
                covariates[i * n_covariates]
            } else {
                0.0
            };
            let suscept_mult = (susceptibility * x_i).exp();

            let mut n_infected_neighbors = 0.0;
            for j in 0..n {
                if adjacency[i * n + j] > 0.0 && infected[j] == 1 && infection_time[j] < t_i {
                    n_infected_neighbors += adjacency[i * n + j];
                }
            }

            if infected[i] == 1 {
                gradient += x_i;
            }
            gradient -= beta * n_infected_neighbors * suscept_mult * x_i * t_i;
            hessian -= beta * n_infected_neighbors * suscept_mult * x_i.powi(2) * t_i;
        }
        (gradient, hessian)
    }
}
