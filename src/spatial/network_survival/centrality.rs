

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum CentralityType {
    Degree,
    Betweenness,
    Closeness,
    Eigenvector,
    PageRank,
}

#[pymethods]
impl CentralityType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "degree" => Ok(CentralityType::Degree),
            "betweenness" => Ok(CentralityType::Betweenness),
            "closeness" => Ok(CentralityType::Closeness),
            "eigenvector" | "eigen" => Ok(CentralityType::Eigenvector),
            "pagerank" | "page_rank" => Ok(CentralityType::PageRank),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown centrality type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NetworkSurvivalConfig {
    #[pyo3(get, set)]
    pub include_peer_effects: bool,
    #[pyo3(get, set)]
    pub include_centrality: bool,
    #[pyo3(get, set)]
    pub centrality_type: CentralityType,
    #[pyo3(get, set)]
    pub peer_lag: usize,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl NetworkSurvivalConfig {
    #[new]
    #[pyo3(signature = (include_peer_effects=true, include_centrality=true, centrality_type=CentralityType::Degree, peer_lag=1, max_iter=100, tol=1e-6))]
    pub fn new(
        include_peer_effects: bool,
        include_centrality: bool,
        centrality_type: CentralityType,
        peer_lag: usize,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        NetworkSurvivalConfig {
            include_peer_effects,
            include_centrality,
            centrality_type,
            peer_lag,
            max_iter,
            tol,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NetworkSurvivalResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub peer_effect: f64,
    #[pyo3(get)]
    pub peer_effect_se: f64,
    #[pyo3(get)]
    pub centrality_effect: f64,
    #[pyo3(get)]
    pub centrality_effect_se: f64,
    #[pyo3(get)]
    pub centrality_values: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_nodes: usize,
    #[pyo3(get)]
    pub n_edges: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, adjacency_matrix, n_nodes, config))]
pub fn network_survival_model(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    adjacency_matrix: Vec<f64>,
    n_nodes: usize,
    config: &NetworkSurvivalConfig,
) -> PyResult<NetworkSurvivalResult> {
    let n = time.len();
    if event.len() != n || n != n_nodes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, event must have length n_nodes",
        ));
    }
    if adjacency_matrix.len() != n_nodes * n_nodes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "adjacency_matrix must have length n_nodes * n_nodes",
        ));
    }

    let centrality = compute_centrality(&adjacency_matrix, n_nodes, &config.centrality_type);

    let n_edges: usize = if n_nodes > PARALLEL_THRESHOLD_MEDIUM {
        (0..n_nodes)
            .into_par_iter()
            .map(|i| {
                ((i + 1)..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0)
                    .count()
            })
            .sum()
    } else {
        (0..n_nodes)
            .map(|i| {
                ((i + 1)..n_nodes)
                    .filter(|&j| adjacency_matrix[i * n_nodes + j] > 0.0)
                    .count()
            })
            .sum()
    };

    let n_extra = (if config.include_peer_effects { 1 } else { 0 })
        + (if config.include_centrality { 1 } else { 0 });
    let total_vars = n_covariates + n_extra;

    let mut beta = vec![0.0; total_vars];
    let mut converged = false;
    let mut log_lik = f64::NEG_INFINITY;

    let peer_hazard: Vec<f64> = if config.include_peer_effects {
        compute_peer_hazard(&time, &event, &adjacency_matrix, n_nodes)
    } else {
        vec![0.0; n_nodes]
    };

    for _iter in 0..config.max_iter {
        let eta: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..n_covariates {
                        e += covariates[i * n_covariates + j] * beta[j];
                    }
                    let mut idx = n_covariates;
                    if config.include_peer_effects {
                        e += peer_hazard[i] * beta[idx];
                        idx += 1;
                    }
                    if config.include_centrality {
                        e += centrality[i] * beta[idx];
                    }
                    e.clamp(-500.0, 500.0)
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..n_covariates {
                        e += covariates[i * n_covariates + j] * beta[j];
                    }
                    let mut idx = n_covariates;
                    if config.include_peer_effects {
                        e += peer_hazard[i] * beta[idx];
                        idx += 1;
                    }
                    if config.include_centrality {
                        e += centrality[i] * beta[idx];
                    }
                    e.clamp(-500.0, 500.0)
                })
                .collect()
        };

        let exp_eta: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
            eta.par_iter().map(|&e| e.exp()).collect()
        } else {
            eta.iter().map(|&e| e.exp()).collect()
        };

        let indices = descending_time_indices(&time);

        let mut gradient = vec![0.0; total_vars];
        let mut hessian_diag = vec![0.0; total_vars];
        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; total_vars];
        let mut weighted_x_sq = vec![0.0; total_vars];

        for &i in &indices {
            let x_i = build_x_row(
                &covariates,
                &peer_hazard,
                &centrality,
                i,
                n_covariates,
                config,
            );

            risk_sum += exp_eta[i];
            for j in 0..total_vars {
                weighted_x[j] += exp_eta[i] * x_i[j];
                weighted_x_sq[j] += exp_eta[i] * x_i[j].powi(2);
            }

            if event[i] == 1 && risk_sum > 0.0 {
                for j in 0..total_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    gradient[j] += x_i[j] - x_bar;
                    hessian_diag[j] += x_sq_bar - x_bar.powi(2);
                }
            }
        }

        let mut max_change = 0.0_f64;
        for j in 0..total_vars {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        let new_log_lik = compute_partial_loglik(&time, &event, &eta, &exp_eta);

        if (new_log_lik - log_lik).abs() < config.tol && max_change < config.tol {
            converged = true;
            log_lik = new_log_lik;
            break;
        }
        log_lik = new_log_lik;
    }

    let se = compute_se(
        &time,
        &event,
        &covariates,
        &peer_hazard,
        &centrality,
        &beta,
        n,
        n_covariates,
        config,
    );

    let hazard_ratios: Vec<f64> = beta[..n_covariates].iter().map(|&b| b.exp()).collect();
    let z = 1.96;
    let hr_ci_lower: Vec<f64> = beta[..n_covariates]
        .iter()
        .zip(se[..n_covariates].iter())
        .map(|(&b, &s)| (b - z * s).exp())
        .collect();
    let hr_ci_upper: Vec<f64> = beta[..n_covariates]
        .iter()
        .zip(se[..n_covariates].iter())
        .map(|(&b, &s)| (b + z * s).exp())
        .collect();

    let mut idx = n_covariates;
    let (peer_effect, peer_effect_se) = if config.include_peer_effects {
        let pe = beta[idx];
        let pe_se = se[idx];
        idx += 1;
        (pe, pe_se)
    } else {
        (0.0, 0.0)
    };

    let (centrality_effect, centrality_effect_se) = if config.include_centrality {
        (beta[idx], se[idx])
    } else {
        (0.0, 0.0)
    };

    let n_params = total_vars as f64;
    let aic = -2.0 * log_lik + 2.0 * n_params;
    let bic = -2.0 * log_lik + n_params * (n as f64).ln();

    Ok(NetworkSurvivalResult {
        coefficients: beta[..n_covariates].to_vec(),
        std_errors: se[..n_covariates].to_vec(),
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        peer_effect,
        peer_effect_se,
        centrality_effect,
        centrality_effect_se,
        centrality_values: centrality,
        log_likelihood: log_lik,
        aic,
        bic,
        n_nodes,
        n_edges,
        converged,
    })
}

fn build_x_row(
    covariates: &[f64],
    peer_hazard: &[f64],
    centrality: &[f64],
    i: usize,
    n_covariates: usize,
    config: &NetworkSurvivalConfig,
) -> Vec<f64> {
    let n_extra = (if config.include_peer_effects { 1 } else { 0 })
        + (if config.include_centrality { 1 } else { 0 });
    let mut x = Vec::with_capacity(n_covariates + n_extra);

    for j in 0..n_covariates {
        x.push(covariates[i * n_covariates + j]);
    }
    if config.include_peer_effects {
        x.push(peer_hazard[i]);
    }
    if config.include_centrality {
        x.push(centrality[i]);
    }
    x
}

fn compute_centrality(adjacency: &[f64], n: usize, centrality_type: &CentralityType) -> Vec<f64> {
    match centrality_type {
        CentralityType::Degree => {
            if n > PARALLEL_THRESHOLD_MEDIUM {
                (0..n)
                    .into_par_iter()
                    .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum::<f64>())
                    .collect()
            } else {
                (0..n)
                    .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum::<f64>())
                    .collect()
            }
        }
        CentralityType::Closeness => compute_closeness_centrality(adjacency, n),
        CentralityType::Betweenness => compute_betweenness_centrality(adjacency, n),
        CentralityType::Eigenvector => compute_eigenvector_centrality(adjacency, n),
        CentralityType::PageRank => compute_pagerank(adjacency, n, 0.85),
    }
}

fn compute_closeness_centrality(adjacency: &[f64], n: usize) -> Vec<f64> {
    let distances = floyd_warshall(adjacency, n);
    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let sum_dist: f64 = (0..n)
                    .filter(|&j| j != i && distances[i * n + j].is_finite())
                    .map(|j| distances[i * n + j])
                    .sum();

                if sum_dist > 0.0 {
                    let reachable = (0..n)
                        .filter(|&j| j != i && distances[i * n + j].is_finite())
                        .count();
                    reachable as f64 / sum_dist
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let mut closeness = vec![0.0; n];
        for i in 0..n {
            let sum_dist: f64 = (0..n)
                .filter(|&j| j != i && distances[i * n + j].is_finite())
                .map(|j| distances[i * n + j])
                .sum();

            if sum_dist > 0.0 {
                let reachable = (0..n)
                    .filter(|&j| j != i && distances[i * n + j].is_finite())
                    .count();
                closeness[i] = reachable as f64 / sum_dist;
            }
        }
        closeness
    }
}

fn compute_betweenness_centrality(adjacency: &[f64], n: usize) -> Vec<f64> {
    let source_contrib = |s: usize| {
        let (dist, paths, predecessors) = bfs_shortest_paths(adjacency, n, s);

        let mut local = vec![0.0; n];
        let mut dependency = vec![0.0; n];
        let mut sorted_by_dist: Vec<usize> = (0..n).collect();
        sorted_by_dist.sort_by(|&a, &b| {
            dist[b]
                .partial_cmp(&dist[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &w in &sorted_by_dist {
            for &v in &predecessors[w] {
                let ratio = paths[v] / paths[w].max(1.0);
                dependency[v] += ratio * (1.0 + dependency[w]);
            }
            if w != s {
                local[w] += dependency[w];
            }
        }
        local
    };

    let mut betweenness = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n).into_par_iter().map(source_contrib).reduce(
            || vec![0.0; n],
            |mut a, b| {
                for idx in 0..n {
                    a[idx] += b[idx];
                }
                a
            },
        )
    } else {
        let mut acc = vec![0.0; n];
        for s in 0..n {
            let local = source_contrib(s);
            for idx in 0..n {
                acc[idx] += local[idx];
            }
        }
        acc
    };

    for b in betweenness.iter_mut() {
        *b /= 2.0;
    }

    betweenness
}

fn bfs_shortest_paths(
    adjacency: &[f64],
    n: usize,
    source: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<usize>>) {
    let mut dist = vec![f64::INFINITY; n];
    let mut paths = vec![0.0; n];
    let mut predecessors: Vec<Vec<usize>> = vec![vec![]; n];

    dist[source] = 0.0;
    paths[source] = 1.0;

    let mut queue = vec![source];
    let mut head = 0;

    while head < queue.len() {
        let v = queue[head];
        head += 1;

        for w in 0..n {
            if adjacency[v * n + w] > 0.0 {
                if dist[w] == f64::INFINITY {
                    dist[w] = dist[v] + 1.0;
                    queue.push(w);
                }
                if dist[w] == dist[v] + 1.0 {
                    paths[w] += paths[v];
                    predecessors[w].push(v);
                }
            }
        }
    }

    (dist, paths, predecessors)
}

fn compute_eigenvector_centrality(adjacency: &[f64], n: usize) -> Vec<f64> {
    let mut centrality = vec![1.0 / n as f64; n];
    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        let mut new_centrality: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += adjacency[i * n + j] * centrality[j];
                    }
                    acc
                })
                .collect()
        } else {
            let mut v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    v[i] += adjacency[i * n + j] * centrality[j];
                }
            }
            v
        };

        let norm_sq: f64 = if n > PARALLEL_THRESHOLD_MEDIUM {
            new_centrality.par_iter().map(|&x| x * x).sum()
        } else {
            new_centrality.iter().map(|&x| x * x).sum()
        };
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for c in new_centrality.iter_mut() {
                *c /= norm;
            }
        }

        let max_diff: f64 = if n > PARALLEL_THRESHOLD_MEDIUM {
            centrality
                .par_iter()
                .zip(new_centrality.par_iter())
                .map(|(&a, &b)| (a - b).abs())
                .reduce(|| 0.0, f64::max)
        } else {
            centrality
                .iter()
                .zip(new_centrality.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max)
        };

        centrality = new_centrality;

        if max_diff < tol {
            break;
        }
    }

    centrality
}

fn compute_pagerank(adjacency: &[f64], n: usize, damping: f64) -> Vec<f64> {
    let mut pagerank = vec![1.0 / n as f64; n];
    let max_iter = 100;
    let tol = 1e-6;

    let out_degree: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum::<f64>().max(1.0))
            .collect()
    } else {
        (0..n)
            .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum::<f64>().max(1.0))
            .collect()
    };

    for _ in 0..max_iter {
        let new_pagerank: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut val = (1.0 - damping) / n as f64;
                    for j in 0..n {
                        if adjacency[j * n + i] > 0.0 {
                            val += damping * pagerank[j] / out_degree[j];
                        }
                    }
                    val
                })
                .collect()
        } else {
            let mut v = vec![(1.0 - damping) / n as f64; n];
            for i in 0..n {
                for j in 0..n {
                    if adjacency[j * n + i] > 0.0 {
                        v[i] += damping * pagerank[j] / out_degree[j];
                    }
                }
            }
            v
        };

        let max_diff: f64 = if n > PARALLEL_THRESHOLD_MEDIUM {
            pagerank
                .par_iter()
                .zip(new_pagerank.par_iter())
                .map(|(&a, &b)| (a - b).abs())
                .reduce(|| 0.0, f64::max)
        } else {
            pagerank
                .iter()
                .zip(new_pagerank.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max)
        };

        pagerank = new_pagerank;

        if max_diff < tol {
            break;
        }
    }

    pagerank
}

fn floyd_warshall(adjacency: &[f64], n: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; n * n];

    for i in 0..n {
        dist[i * n + i] = 0.0;
        for j in 0..n {
            if adjacency[i * n + j] > 0.0 {
                dist[i * n + j] = 1.0;
            }
        }
    }

    for k in 0..n {
        let row_k = dist[k * n..(k + 1) * n].to_vec();
        let col_k: Vec<f64> = (0..n).map(|i| dist[i * n + k]).collect();
        if n > PARALLEL_THRESHOLD_MEDIUM {
            dist.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                let dik = col_k[i];
                if !dik.is_finite() {
                    return;
                }
                for j in 0..n {
                    let dkj = row_k[j];
                    if dkj.is_finite() {
                        let alt = dik + dkj;
                        if alt < row[j] {
                            row[j] = alt;
                        }
                    }
                }
            });
        } else {
            for i in 0..n {
                let dik = col_k[i];
                if !dik.is_finite() {
                    continue;
                }
                for j in 0..n {
                    let dkj = row_k[j];
                    if dkj.is_finite() {
                        let alt = dik + dkj;
                        if alt < dist[i * n + j] {
                            dist[i * n + j] = alt;
                        }
                    }
                }
            }
        }
    }

    dist
}

fn compute_peer_hazard(time: &[f64], event: &[i32], adjacency: &[f64], n: usize) -> Vec<f64> {
    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut n_neighbors = 0.0;
                let mut neighbor_events = 0.0;

                for j in 0..n {
                    let w = adjacency[i * n + j];
                    if w > 0.0 {
                        n_neighbors += w;
                        if event[j] == 1 && time[j] <= time[i] {
                            neighbor_events += w;
                        }
                    }
                }

                if n_neighbors > 0.0 {
                    neighbor_events / n_neighbors
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let mut peer_hazard = vec![0.0; n];
        for i in 0..n {
            let mut n_neighbors = 0.0;
            let mut neighbor_events = 0.0;

            for j in 0..n {
                let w = adjacency[i * n + j];
                if w > 0.0 {
                    n_neighbors += w;
                    if event[j] == 1 && time[j] <= time[i] {
                        neighbor_events += w;
                    }
                }
            }

            if n_neighbors > 0.0 {
                peer_hazard[i] = neighbor_events / n_neighbors;
            }
        }
        peer_hazard
    }
}

fn compute_partial_loglik(time: &[f64], event: &[i32], eta: &[f64], exp_eta: &[f64]) -> f64 {
    let indices = descending_time_indices(time);

    let mut log_lik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += exp_eta[i];
        if event[i] == 1 && risk_sum > 0.0 {
            log_lik += eta[i] - risk_sum.ln();
        }
    }

    log_lik
}

#[allow(clippy::too_many_arguments)]
fn compute_se(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    peer_hazard: &[f64],
    centrality: &[f64],
    beta: &[f64],
    n: usize,
    n_covariates: usize,
    config: &NetworkSurvivalConfig,
) -> Vec<f64> {
    let total_vars = beta.len();

    let eta: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let x = build_x_row(covariates, peer_hazard, centrality, i, n_covariates, config);
                x.iter()
                    .zip(beta.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum::<f64>()
                    .clamp(-500.0, 500.0)
            })
            .collect()
    } else {
        (0..n)
            .map(|i| {
                let x = build_x_row(covariates, peer_hazard, centrality, i, n_covariates, config);
                x.iter()
                    .zip(beta.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum::<f64>()
                    .clamp(-500.0, 500.0)
            })
            .collect()
    };

    let exp_eta: Vec<f64> = if n > PARALLEL_THRESHOLD_MEDIUM {
        eta.par_iter().map(|&e| e.exp()).collect()
    } else {
        eta.iter().map(|&e| e.exp()).collect()
    };

    let indices = descending_time_indices(time);

    let mut info_diag = vec![0.0; total_vars];
    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; total_vars];
    let mut weighted_x_sq = vec![0.0; total_vars];

    for &i in &indices {
        let x_i = build_x_row(covariates, peer_hazard, centrality, i, n_covariates, config);

        risk_sum += exp_eta[i];
        for j in 0..total_vars {
            weighted_x[j] += exp_eta[i] * x_i[j];
            weighted_x_sq[j] += exp_eta[i] * x_i[j].powi(2);
        }

        if event[i] == 1 && risk_sum > 0.0 {
            for j in 0..total_vars {
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;
                info_diag[j] += x_sq_bar - x_bar.powi(2);
            }
        }
    }

    info_diag
        .iter()
        .map(|&info| {
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}
