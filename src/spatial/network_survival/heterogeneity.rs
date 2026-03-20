
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NetworkHeterogeneityResult {
    #[pyo3(get)]
    pub community_hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub within_community_correlation: f64,
    #[pyo3(get)]
    pub between_community_effect: f64,
    #[pyo3(get)]
    pub modularity: f64,
    #[pyo3(get)]
    pub community_assignments: Vec<usize>,
    #[pyo3(get)]
    pub log_likelihood: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, adjacency_matrix, n_nodes, n_communities=None))]
pub fn network_heterogeneity_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    adjacency_matrix: Vec<f64>,
    n_nodes: usize,
    n_communities: Option<usize>,
) -> PyResult<NetworkHeterogeneityResult> {
    let n = time.len();
    if event.len() != n || n != n_nodes {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, event must have length n_nodes",
        ));
    }

    let k = n_communities.unwrap_or(3);
    let (communities, modularity) = detect_communities(&adjacency_matrix, n_nodes, k);

    let mut community_hazard_ratios: Vec<f64> = if k > PARALLEL_THRESHOLD_MEDIUM {
        (0..k)
            .into_par_iter()
            .map(|c| {
                let mut n_events = 0.0;
                let mut total_time = 0.0;
                let mut n_members = 0usize;
                for i in 0..n {
                    if communities[i] == c {
                        n_members += 1;
                        n_events += event[i] as f64;
                        total_time += time[i];
                    }
                }
                if n_members == 0 {
                    1.0
                } else if total_time > 0.0 {
                    n_events / total_time
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let mut vals = vec![0.0; k];
        for (c, value) in vals.iter_mut().enumerate().take(k) {
            let mut n_events = 0.0;
            let mut total_time = 0.0;
            let mut n_members = 0usize;
            for i in 0..n {
                if communities[i] == c {
                    n_members += 1;
                    n_events += event[i] as f64;
                    total_time += time[i];
                }
            }
            *value = if n_members == 0 {
                1.0
            } else if total_time > 0.0 {
                n_events / total_time
            } else {
                0.0
            };
        }
        vals
    };

    let overall_rate: f64 = event.iter().map(|&e| e as f64).sum::<f64>() / time.iter().sum::<f64>();
    for hr in community_hazard_ratios.iter_mut() {
        if overall_rate > 0.0 {
            *hr /= overall_rate;
        }
    }

    let within_correlation = compute_within_community_correlation(
        &time,
        &event,
        &communities,
        &adjacency_matrix,
        n_nodes,
    );
    let between_effect =
        compute_between_community_effect(&time, &event, &communities, &adjacency_matrix, n_nodes);

    let log_lik = compute_community_loglik(&time, &event, &communities, &community_hazard_ratios);

    Ok(NetworkHeterogeneityResult {
        community_hazard_ratios,
        within_community_correlation: within_correlation,
        between_community_effect: between_effect,
        modularity,
        community_assignments: communities,
        log_likelihood: log_lik,
    })
}

fn detect_communities(adjacency: &[f64], n: usize, k: usize) -> (Vec<usize>, f64) {
    let mut communities = vec![0; n];
    for (i, community) in communities.iter_mut().enumerate().take(n) {
        *community = i % k;
    }

    let degree: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum())
        .collect();
    let total_edges: f64 = degree.iter().sum::<f64>() / 2.0;

    for _ in 0..50 {
        let mut changed = false;
        for i in 0..n {
            let current_community = communities[i];
            let mut best_community = current_community;
            let mut best_delta = 0.0;

            for c in 0..k {
                if c == current_community {
                    continue;
                }

                let delta = compute_modularity_delta(
                    adjacency,
                    &degree,
                    &communities,
                    i,
                    current_community,
                    c,
                    n,
                    total_edges,
                );

                if delta > best_delta {
                    best_delta = delta;
                    best_community = c;
                }
            }

            if best_community != current_community {
                communities[i] = best_community;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    let modularity = compute_modularity(adjacency, &degree, &communities, n, total_edges, k);

    (communities, modularity)
}

#[allow(clippy::too_many_arguments)]
fn compute_modularity_delta(
    adjacency: &[f64],
    degree: &[f64],
    communities: &[usize],
    node: usize,
    _from_community: usize,
    to_community: usize,
    n: usize,
    total_edges: f64,
) -> f64 {
    if total_edges < 1e-10 {
        return 0.0;
    }

    let k_i = degree[node];

    let mut sum_in = 0.0;
    let mut sum_tot = 0.0;
    for j in 0..n {
        if communities[j] == to_community {
            sum_in += adjacency[node * n + j];
            sum_tot += degree[j];
        }
    }

    (sum_in - k_i * sum_tot / (2.0 * total_edges)) / (2.0 * total_edges)
}

fn compute_modularity(
    adjacency: &[f64],
    degree: &[f64],
    communities: &[usize],
    n: usize,
    total_edges: f64,
    _k: usize,
) -> f64 {
    if total_edges < 1e-10 {
        return 0.0;
    }

    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            if communities[i] == communities[j] {
                q += adjacency[i * n + j] - degree[i] * degree[j] / (2.0 * total_edges);
            }
        }
    }

    q / (2.0 * total_edges)
}

fn compute_within_community_correlation(
    time: &[f64],
    event: &[i32],
    communities: &[usize],
    adjacency: &[f64],
    n: usize,
) -> f64 {
    let (within_pairs, within_concordant) = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut local_pairs = 0.0;
                let mut local_concordant = 0.0;
                for j in (i + 1)..n {
                    if communities[i] == communities[j] && adjacency[i * n + j] > 0.0 {
                        local_pairs += 1.0;
                        if (time[i] < time[j] && event[i] == 1)
                            || (time[j] < time[i] && event[j] == 1)
                        {
                            local_concordant += 1.0;
                        }
                    }
                }
                (local_pairs, local_concordant)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut pairs = 0.0;
        let mut concordant = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if communities[i] == communities[j] && adjacency[i * n + j] > 0.0 {
                    pairs += 1.0;
                    if (time[i] < time[j] && event[i] == 1) || (time[j] < time[i] && event[j] == 1)
                    {
                        concordant += 1.0;
                    }
                }
            }
        }
        (pairs, concordant)
    };

    if within_pairs > 0.0 {
        within_concordant / within_pairs
    } else {
        0.0
    }
}

fn compute_between_community_effect(
    time: &[f64],
    event: &[i32],
    communities: &[usize],
    adjacency: &[f64],
    n: usize,
) -> f64 {
    let (between_effect, n_between) = if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut n_between_neighbors_events = 0.0;
                let mut n_between_neighbors = 0.0;

                for j in 0..n {
                    if adjacency[i * n + j] > 0.0 && communities[i] != communities[j] {
                        n_between_neighbors += 1.0;
                        if event[j] == 1 && time[j] <= time[i] {
                            n_between_neighbors_events += 1.0;
                        }
                    }
                }

                if n_between_neighbors > 0.0 {
                    (n_between_neighbors_events / n_between_neighbors, 1.0)
                } else {
                    (0.0, 0.0)
                }
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        let mut effect = 0.0;
        let mut count = 0.0;
        for i in 0..n {
            let mut n_between_neighbors_events = 0.0;
            let mut n_between_neighbors = 0.0;

            for j in 0..n {
                if adjacency[i * n + j] > 0.0 && communities[i] != communities[j] {
                    n_between_neighbors += 1.0;
                    if event[j] == 1 && time[j] <= time[i] {
                        n_between_neighbors_events += 1.0;
                    }
                }
            }

            if n_between_neighbors > 0.0 {
                effect += n_between_neighbors_events / n_between_neighbors;
                count += 1.0;
            }
        }
        (effect, count)
    };

    if n_between > 0.0 {
        between_effect / n_between
    } else {
        0.0
    }
}

fn compute_community_loglik(
    time: &[f64],
    event: &[i32],
    communities: &[usize],
    community_rates: &[f64],
) -> f64 {
    let n = time.len();
    if n > PARALLEL_THRESHOLD_MEDIUM {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let c = communities[i];
                let lambda = community_rates[c].max(1e-10);
                let event_term = if event[i] == 1 { lambda.ln() } else { 0.0 };
                event_term - lambda * time[i]
            })
            .sum()
    } else {
        let mut log_lik = 0.0;
        for i in 0..n {
            let c = communities[i];
            let lambda = community_rates[c].max(1e-10);
            if event[i] == 1 {
                log_lik += lambda.ln();
            }
            log_lik -= lambda * time[i];
        }
        log_lik
    }
}

