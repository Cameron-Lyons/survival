use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiStateConfig {
    #[pyo3(get, set)]
    pub n_states: usize,
    #[pyo3(get, set)]
    pub state_names: Vec<String>,
    #[pyo3(get, set)]
    pub transition_matrix: Vec<Vec<bool>>,
    #[pyo3(get, set)]
    pub absorbing_states: Vec<usize>,
}

#[pymethods]
impl MultiStateConfig {
    #[new]
    #[pyo3(signature = (n_states, state_names=None, transition_matrix=None, absorbing_states=None))]
    pub fn new(
        n_states: usize,
        state_names: Option<Vec<String>>,
        transition_matrix: Option<Vec<Vec<bool>>>,
        absorbing_states: Option<Vec<usize>>,
    ) -> Self {
        let state_names =
            state_names.unwrap_or_else(|| (0..n_states).map(|i| format!("State_{}", i)).collect());

        let transition_matrix = transition_matrix.unwrap_or_else(|| {
            let mut tm = vec![vec![false; n_states]; n_states];
            for i in 0..n_states - 1 {
                tm[i][i + 1] = true;
            }
            tm
        });

        let absorbing_states = absorbing_states.unwrap_or_else(|| vec![n_states - 1]);

        Self {
            n_states,
            state_names,
            transition_matrix,
            absorbing_states,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct TransitionIntensityResult {
    #[pyo3(get)]
    pub intensities: HashMap<String, Vec<f64>>,
    #[pyo3(get)]
    pub cumulative_intensities: HashMap<String, Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub variance: HashMap<String, Vec<f64>>,
    #[pyo3(get)]
    pub n_at_risk: HashMap<String, Vec<f64>>,
    #[pyo3(get)]
    pub n_transitions: HashMap<String, Vec<i32>>,
}

#[pymethods]
impl TransitionIntensityResult {
    #[new]
    pub fn new(
        intensities: HashMap<String, Vec<f64>>,
        cumulative_intensities: HashMap<String, Vec<f64>>,
        time_points: Vec<f64>,
        variance: HashMap<String, Vec<f64>>,
        n_at_risk: HashMap<String, Vec<f64>>,
        n_transitions: HashMap<String, Vec<i32>>,
    ) -> Self {
        Self {
            intensities,
            cumulative_intensities,
            time_points,
            variance,
            n_at_risk,
            n_transitions,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultiStateResult {
    #[pyo3(get)]
    pub state_probabilities: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub transition_intensities: TransitionIntensityResult,
    #[pyo3(get)]
    pub restricted_mean_times: Vec<f64>,
    #[pyo3(get)]
    pub sojourn_times: Vec<f64>,
    #[pyo3(get)]
    pub state_occupancy: Vec<Vec<f64>>,
}

#[pymethods]
impl MultiStateResult {
    #[new]
    pub fn new(
        state_probabilities: Vec<Vec<f64>>,
        time_points: Vec<f64>,
        transition_intensities: TransitionIntensityResult,
        restricted_mean_times: Vec<f64>,
        sojourn_times: Vec<f64>,
        state_occupancy: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            state_probabilities,
            time_points,
            transition_intensities,
            restricted_mean_times,
            sojourn_times,
            state_occupancy,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    entry_time,
    exit_time,
    from_state,
    to_state,
    event,
    config
))]
pub fn estimate_transition_intensities(
    entry_time: Vec<f64>,
    exit_time: Vec<f64>,
    from_state: Vec<usize>,
    to_state: Vec<usize>,
    event: Vec<i32>,
    config: MultiStateConfig,
) -> PyResult<TransitionIntensityResult> {
    let n = entry_time.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input vectors must be non-empty",
        ));
    }

    let mut all_times: Vec<f64> = exit_time.to_vec();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_times.dedup();

    let mut intensities: HashMap<String, Vec<f64>> = HashMap::new();
    let mut cumulative_intensities: HashMap<String, Vec<f64>> = HashMap::new();
    let mut variance: HashMap<String, Vec<f64>> = HashMap::new();
    let mut n_at_risk: HashMap<String, Vec<f64>> = HashMap::new();
    let mut n_transitions_map: HashMap<String, Vec<i32>> = HashMap::new();

    for i in 0..config.n_states {
        for j in 0..config.n_states {
            if i != j && config.transition_matrix[i][j] {
                let key = format!("{}->{}", i, j);
                intensities.insert(key.clone(), vec![0.0; all_times.len()]);
                cumulative_intensities.insert(key.clone(), vec![0.0; all_times.len()]);
                variance.insert(key.clone(), vec![0.0; all_times.len()]);
                n_at_risk.insert(key.clone(), vec![0.0; all_times.len()]);
                n_transitions_map.insert(key.clone(), vec![0; all_times.len()]);
            }
        }
    }

    for (t_idx, &t) in all_times.iter().enumerate() {
        for i in 0..config.n_states {
            for j in 0..config.n_states {
                if i != j && config.transition_matrix[i][j] {
                    let key = format!("{}->{}", i, j);

                    let at_risk: f64 = (0..n)
                        .filter(|&k| entry_time[k] < t && exit_time[k] >= t && from_state[k] == i)
                        .count() as f64;

                    let transitions: i32 = (0..n)
                        .filter(|&k| {
                            (exit_time[k] - t).abs() < 1e-10
                                && event[k] == 1
                                && from_state[k] == i
                                && to_state[k] == j
                        })
                        .count() as i32;

                    n_at_risk.get_mut(&key).unwrap()[t_idx] = at_risk;
                    n_transitions_map.get_mut(&key).unwrap()[t_idx] = transitions;

                    if at_risk > 0.0 {
                        let intensity = transitions as f64 / at_risk;
                        intensities.get_mut(&key).unwrap()[t_idx] = intensity;

                        let var = if at_risk > 0.0 && transitions > 0 {
                            transitions as f64 / (at_risk * at_risk)
                        } else {
                            0.0
                        };
                        variance.get_mut(&key).unwrap()[t_idx] = var;
                    }
                }
            }
        }
    }

    for key in intensities.keys() {
        let int_vec = intensities.get(key).unwrap();
        let cum_int: Vec<f64> = int_vec
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        cumulative_intensities.insert(key.clone(), cum_int);
    }

    Ok(TransitionIntensityResult {
        intensities,
        cumulative_intensities,
        time_points: all_times,
        variance,
        n_at_risk,
        n_transitions: n_transitions_map,
    })
}

#[pyfunction]
#[pyo3(signature = (
    entry_time,
    exit_time,
    from_state,
    to_state,
    event,
    eval_times,
    config
))]
pub fn fit_multi_state_model(
    entry_time: Vec<f64>,
    exit_time: Vec<f64>,
    from_state: Vec<usize>,
    to_state: Vec<usize>,
    event: Vec<i32>,
    eval_times: Vec<f64>,
    config: MultiStateConfig,
) -> PyResult<MultiStateResult> {
    let transition_result = estimate_transition_intensities(
        entry_time.clone(),
        exit_time.clone(),
        from_state.clone(),
        to_state.clone(),
        event.clone(),
        config.clone(),
    )?;

    let n_states = config.n_states;
    let n_times = eval_times.len();

    let mut state_probs = vec![vec![0.0; n_states]; n_times];
    state_probs[0][0] = 1.0;

    for t_idx in 1..n_times {
        let dt = eval_times[t_idx] - eval_times[t_idx - 1];

        let mut transition_probs = vec![vec![0.0; n_states]; n_states];
        for (i, row) in transition_probs.iter_mut().enumerate().take(n_states) {
            row[i] = 1.0;
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_states {
            for j in 0..n_states {
                if i != j && config.transition_matrix[i][j] {
                    let key = format!("{}->{}", i, j);
                    if let Some(int_vec) = transition_result.intensities.get(&key) {
                        let intensity = interpolate_intensity(
                            &transition_result.time_points,
                            int_vec,
                            eval_times[t_idx],
                        );
                        transition_probs[i][j] = intensity * dt;
                        transition_probs[i][i] -= intensity * dt;
                    }
                }
            }
        }

        for j in 0..n_states {
            let mut prob = 0.0;
            for i in 0..n_states {
                prob += state_probs[t_idx - 1][i] * transition_probs[i][j];
            }
            state_probs[t_idx][j] = prob.clamp(0.0, 1.0);
        }

        let sum: f64 = state_probs[t_idx].iter().sum();
        if sum > 0.0 {
            for prob in state_probs[t_idx].iter_mut() {
                *prob /= sum;
            }
        }
    }

    let mut restricted_mean_times = vec![0.0; n_states];
    for j in 0..n_states {
        for t_idx in 1..n_times {
            let dt = eval_times[t_idx] - eval_times[t_idx - 1];
            restricted_mean_times[j] +=
                (state_probs[t_idx - 1][j] + state_probs[t_idx][j]) / 2.0 * dt;
        }
    }

    let sojourn_times = compute_sojourn_times(&entry_time, &exit_time, &from_state, n_states);

    let state_occupancy = state_probs.clone();

    Ok(MultiStateResult {
        state_probabilities: state_probs,
        time_points: eval_times,
        transition_intensities: transition_result,
        restricted_mean_times,
        sojourn_times,
        state_occupancy,
    })
}

fn interpolate_intensity(times: &[f64], values: &[f64], target_time: f64) -> f64 {
    if times.is_empty() || values.is_empty() {
        return 0.0;
    }

    if target_time <= times[0] {
        return values[0];
    }
    if target_time >= times[times.len() - 1] {
        return values[values.len() - 1];
    }

    for i in 1..times.len() {
        if times[i] >= target_time {
            let t0 = times[i - 1];
            let t1 = times[i];
            let v0 = values[i - 1];
            let v1 = values[i];
            let alpha = (target_time - t0) / (t1 - t0);
            return v0 + alpha * (v1 - v0);
        }
    }

    values[values.len() - 1]
}

fn compute_sojourn_times(
    entry_time: &[f64],
    exit_time: &[f64],
    from_state: &[usize],
    n_states: usize,
) -> Vec<f64> {
    let mut total_time = vec![0.0; n_states];
    let mut count = vec![0usize; n_states];

    for i in 0..entry_time.len() {
        let state = from_state[i];
        let duration = exit_time[i] - entry_time[i];
        if state < n_states {
            total_time[state] += duration;
            count[state] += 1;
        }
    }

    (0..n_states)
        .map(|s| {
            if count[s] > 0 {
                total_time[s] / count[s] as f64
            } else {
                0.0
            }
        })
        .collect()
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MarkovMSMResult {
    #[pyo3(get)]
    pub transition_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub generator_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub stationary_distribution: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub state_probabilities: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: f64,
}

#[pymethods]
impl MarkovMSMResult {
    #[new]
    pub fn new(
        transition_matrix: Vec<Vec<f64>>,
        generator_matrix: Vec<Vec<f64>>,
        stationary_distribution: Vec<f64>,
        time_points: Vec<f64>,
        state_probabilities: Vec<Vec<f64>>,
        log_likelihood: f64,
    ) -> Self {
        Self {
            transition_matrix,
            generator_matrix,
            stationary_distribution,
            time_points,
            state_probabilities,
            log_likelihood,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    entry_time,
    exit_time,
    from_state,
    to_state,
    event,
    eval_times,
    config
))]
pub fn fit_markov_msm(
    entry_time: Vec<f64>,
    exit_time: Vec<f64>,
    from_state: Vec<usize>,
    to_state: Vec<usize>,
    event: Vec<i32>,
    eval_times: Vec<f64>,
    config: MultiStateConfig,
) -> PyResult<MarkovMSMResult> {
    let n_states = config.n_states;
    let n = entry_time.len();

    let mut transition_counts = vec![vec![0.0; n_states]; n_states];
    let mut total_time = vec![0.0; n_states];

    for i in 0..n {
        let duration = exit_time[i] - entry_time[i];
        let from = from_state[i];

        if from < n_states {
            total_time[from] += duration;

            if event[i] == 1 {
                let to = to_state[i];
                if to < n_states {
                    transition_counts[from][to] += 1.0;
                }
            }
        }
    }

    let mut generator_matrix = vec![vec![0.0; n_states]; n_states];
    for i in 0..n_states {
        if total_time[i] > 0.0 {
            for j in 0..n_states {
                if i != j {
                    generator_matrix[i][j] = transition_counts[i][j] / total_time[i];
                }
            }
            generator_matrix[i][i] -= generator_matrix[i].iter().sum::<f64>();
        }
    }

    let transition_matrix = matrix_exponential(&generator_matrix, 1.0);

    let stationary_distribution = compute_stationary_distribution(&transition_matrix);

    let n_times = eval_times.len();
    let mut state_probs = vec![vec![0.0; n_states]; n_times];
    state_probs[0][0] = 1.0;

    for t_idx in 1..n_times {
        let dt = eval_times[t_idx] - eval_times[t_idx - 1];
        let p_t = matrix_exponential(&generator_matrix, dt);

        for j in 0..n_states {
            for i in 0..n_states {
                state_probs[t_idx][j] += state_probs[t_idx - 1][i] * p_t[i][j];
            }
        }
    }

    let mut log_lik = 0.0;
    for i in 0..n {
        let from = from_state[i];
        let duration = exit_time[i] - entry_time[i];

        if from < n_states && duration > 0.0 {
            if generator_matrix[from][from] < 0.0 {
                log_lik += duration * generator_matrix[from][from];
            }

            if event[i] == 1 {
                let to = to_state[i];
                if to < n_states && generator_matrix[from][to] > 0.0 {
                    log_lik += generator_matrix[from][to].ln();
                }
            }
        }
    }

    Ok(MarkovMSMResult {
        transition_matrix,
        generator_matrix,
        stationary_distribution,
        time_points: eval_times,
        state_probabilities: state_probs,
        log_likelihood: log_lik,
    })
}

#[allow(clippy::needless_range_loop)]
fn matrix_exponential(q: &[Vec<f64>], t: f64) -> Vec<Vec<f64>> {
    let n = q.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        result[i][i] = 1.0;
    }

    let mut qt = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            qt[i][j] = q[i][j] * t;
        }
    }

    let mut term = vec![vec![0.0; n]; n];
    for i in 0..n {
        term[i][i] = 1.0;
    }

    let max_iter = 50;
    for k in 1..=max_iter {
        let new_term = matrix_multiply(&term, &qt);
        for i in 0..n {
            for j in 0..n {
                term[i][j] = new_term[i][j] / k as f64;
                result[i][j] += term[i][j];
            }
        }

        let norm: f64 = term
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| x.abs())
            .sum();
        if norm < 1e-16 {
            break;
        }
    }

    result
}

fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn compute_stationary_distribution(p: &[Vec<f64>]) -> Vec<f64> {
    let n = p.len();
    let mut pi = vec![1.0 / n as f64; n];

    let max_iter = 1000;
    let tol = 1e-10;

    for _ in 0..max_iter {
        let mut new_pi = vec![0.0; n];

        for j in 0..n {
            for i in 0..n {
                new_pi[j] += pi[i] * p[i][j];
            }
        }

        let diff: f64 = pi
            .iter()
            .zip(new_pi.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        pi = new_pi;

        if diff < tol {
            break;
        }
    }

    let sum: f64 = pi.iter().sum();
    if sum > 0.0 {
        for p in &mut pi {
            *p /= sum;
        }
    }

    pi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_intensities() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let exit_time = vec![1.0, 2.0, 3.0, 2.5, 1.5];
        let from_state = vec![0, 0, 0, 0, 0];
        let to_state = vec![1, 1, 1, 1, 1];
        let event = vec![1, 1, 1, 0, 1];

        let config = MultiStateConfig::new(2, None, None, None);
        let result = estimate_transition_intensities(
            entry_time, exit_time, from_state, to_state, event, config,
        )
        .unwrap();

        assert!(!result.time_points.is_empty());
        assert!(result.intensities.contains_key("0->1"));
    }

    #[test]
    fn test_multi_state_model() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0];
        let exit_time = vec![1.0, 2.0, 1.5, 2.5];
        let from_state = vec![0, 0, 0, 0];
        let to_state = vec![1, 1, 1, 1];
        let event = vec![1, 1, 1, 0];
        let eval_times = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

        let config = MultiStateConfig::new(2, None, None, None);
        let result = fit_multi_state_model(
            entry_time, exit_time, from_state, to_state, event, eval_times, config,
        )
        .unwrap();

        assert_eq!(result.state_probabilities.len(), 7);
        assert_eq!(result.state_probabilities[0][0], 1.0);
    }

    #[test]
    fn test_markov_msm() {
        let entry_time = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let exit_time = vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0];
        let from_state = vec![0, 0, 0, 0, 1, 1];
        let to_state = vec![1, 1, 1, 1, 2, 2];
        let event = vec![1, 1, 1, 0, 1, 0];
        let eval_times = vec![0.0, 1.0, 2.0, 3.0];

        let mut tm = vec![vec![false; 3]; 3];
        tm[0][1] = true;
        tm[1][2] = true;

        let config = MultiStateConfig::new(
            3,
            Some(vec![
                "Healthy".to_string(),
                "Disease".to_string(),
                "Death".to_string(),
            ]),
            Some(tm),
            Some(vec![2]),
        );

        let result = fit_markov_msm(
            entry_time, exit_time, from_state, to_state, event, eval_times, config,
        )
        .unwrap();

        assert_eq!(result.generator_matrix.len(), 3);
        assert_eq!(result.stationary_distribution.len(), 3);
    }

    #[test]
    fn test_matrix_exponential() {
        let q = vec![vec![-1.0, 1.0], vec![0.0, 0.0]];
        let p = matrix_exponential(&q, 1.0);

        assert!(p[0][0] > 0.0 && p[0][0] < 1.0);
        assert!(p[0][1] > 0.0 && p[0][1] < 1.0);
        assert!((p[0][0] + p[0][1] - 1.0).abs() < 1e-6);
    }
}
