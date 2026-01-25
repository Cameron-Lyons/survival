#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum VarianceEstimator {
    Greenwood,
    Aalen,
    Bootstrap,
}

#[pymethods]
impl VarianceEstimator {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "greenwood" => Ok(VarianceEstimator::Greenwood),
            "aalen" => Ok(VarianceEstimator::Aalen),
            "bootstrap" => Ok(VarianceEstimator::Bootstrap),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown variance estimator. Use 'greenwood', 'aalen', or 'bootstrap'",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum TransitionType {
    Standard,
    MarkovIllnessDeath,
    Progressive,
    Custom,
}

#[pymethods]
impl TransitionType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "standard" => Ok(TransitionType::Standard),
            "markov" | "illness_death" | "illnessdeath" => Ok(TransitionType::MarkovIllnessDeath),
            "progressive" => Ok(TransitionType::Progressive),
            "custom" => Ok(TransitionType::Custom),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown transition type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AalenJohansenExtendedConfig {
    #[pyo3(get, set)]
    pub variance_estimator: VarianceEstimator,
    #[pyo3(get, set)]
    pub transition_type: TransitionType,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub confidence_level: f64,
    #[pyo3(get, set)]
    pub compute_sojourn: bool,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl AalenJohansenExtendedConfig {
    #[new]
    #[pyo3(signature = (
        variance_estimator=VarianceEstimator::Greenwood,
        transition_type=TransitionType::Standard,
        n_bootstrap=200,
        confidence_level=0.95,
        compute_sojourn=true,
        seed=None
    ))]
    pub fn new(
        variance_estimator: VarianceEstimator,
        transition_type: TransitionType,
        n_bootstrap: usize,
        confidence_level: f64,
        compute_sojourn: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&confidence_level) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "confidence_level must be between 0 and 1",
            ));
        }
        if variance_estimator == VarianceEstimator::Bootstrap && n_bootstrap == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_bootstrap must be positive when using bootstrap variance",
            ));
        }

        Ok(AalenJohansenExtendedConfig {
            variance_estimator,
            transition_type,
            n_bootstrap,
            confidence_level,
            compute_sojourn,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TransitionMatrix {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_transitions: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AalenJohansenExtendedResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub state_probs: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub variance: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub ci_lower: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub ci_upper: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub transition_matrices: Vec<TransitionMatrix>,
    #[pyo3(get)]
    pub cumulative_incidence: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub expected_sojourn: Option<Vec<f64>>,
    #[pyo3(get)]
    pub n_states: usize,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_events: usize,
}

#[pymethods]
impl AalenJohansenExtendedResult {
    fn __repr__(&self) -> String {
        format!(
            "AalenJohansenExtendedResult(n_states={}, n_times={}, n_obs={})",
            self.n_states,
            self.time.len(),
            self.n_obs
        )
    }

    fn get_cif(&self, to_state: usize) -> PyResult<Vec<f64>> {
        if to_state >= self.n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "to_state out of range",
            ));
        }
        Ok(self.cumulative_incidence[to_state].clone())
    }

    fn get_state_prob(&self, from_state: usize, to_state: usize) -> PyResult<Vec<f64>> {
        if from_state >= self.n_states || to_state >= self.n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "state index out of range",
            ));
        }
        Ok(self
            .state_probs
            .iter()
            .map(|mat| mat[from_state][to_state])
            .collect())
    }

    fn interpolate_at(&self, query_time: f64) -> Vec<Vec<f64>> {
        if self.time.is_empty() {
            return vec![vec![0.0; self.n_states]; self.n_states];
        }

        let idx = self
            .time
            .iter()
            .position(|&t| t > query_time)
            .unwrap_or(self.time.len())
            .saturating_sub(1);

        self.state_probs[idx].clone()
    }
}

fn compute_transition_matrix(
    from_state: &[usize],
    to_state: &[usize],
    time: &[f64],
    weights: &[f64],
    n_states: usize,
    event_time: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<usize>>) {
    let mut transitions = vec![vec![0usize; n_states]; n_states];
    let mut n_at_risk = vec![0usize; n_states];

    for (i, &t) in time.iter().enumerate() {
        if (t - event_time).abs() < 1e-9 {
            let from = from_state[i];
            let to = to_state[i];
            if from < n_states && to < n_states {
                transitions[from][to] += 1;
            }
        }
    }

    for &fs in from_state {
        if fs < n_states {
            n_at_risk[fs] += 1;
        }
    }

    let mut matrix = vec![vec![0.0; n_states]; n_states];

    for from in 0..n_states {
        if n_at_risk[from] > 0 {
            for to in 0..n_states {
                if from == to {
                    let out_transitions: usize = (0..n_states)
                        .filter(|&k| k != from)
                        .map(|k| transitions[from][k])
                        .sum();
                    matrix[from][to] = 1.0 - (out_transitions as f64 / n_at_risk[from] as f64);
                } else {
                    matrix[from][to] = transitions[from][to] as f64 / n_at_risk[from] as f64;
                }
            }
        } else {
            matrix[from][from] = 1.0;
        }
    }

    (matrix, n_at_risk, transitions)
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

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        mat[i][i] = 1.0;
    }
    mat
}

fn compute_greenwood_variance(
    state_probs: &[Vec<Vec<f64>>],
    transition_matrices: &[TransitionMatrix],
    n_states: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n_times = state_probs.len();
    let mut variance = vec![vec![vec![0.0; n_states]; n_states]; n_times];

    for t in 0..n_times {
        let p = &state_probs[t];

        for i in 0..n_states {
            for j in 0..n_states {
                let mut var_ij = 0.0;

                for s in 0..=t {
                    let tm = &transition_matrices[s];
                    for k in 0..n_states {
                        if tm.n_at_risk[k] > 1 {
                            let n_k = tm.n_at_risk[k] as f64;
                            let p_kj = tm.matrix[k][j];
                            var_ij += p[i][k] * p[i][k] * p_kj * (1.0 - p_kj) / (n_k - 1.0);
                        }
                    }
                }

                variance[t][i][j] = var_ij.max(0.0);
            }
        }
    }

    variance
}

fn compute_aalen_variance(
    state_probs: &[Vec<Vec<f64>>],
    transition_matrices: &[TransitionMatrix],
    n_states: usize,
) -> Vec<Vec<Vec<f64>>> {
    let n_times = state_probs.len();
    let mut variance = vec![vec![vec![0.0; n_states]; n_states]; n_times];

    for t in 0..n_times {
        let p = &state_probs[t];

        for i in 0..n_states {
            for j in 0..n_states {
                let mut var_ij = 0.0;

                for s in 0..=t {
                    let tm = &transition_matrices[s];
                    for k in 0..n_states {
                        if tm.n_at_risk[k] > 0 {
                            let n_k = tm.n_at_risk[k] as f64;
                            let n_kj = tm.n_transitions[k][j] as f64;
                            if n_k > 0.0 {
                                var_ij +=
                                    p[i][k] * p[i][k] * n_kj * (n_k - n_kj) / (n_k * n_k * n_k);
                            }
                        }
                    }
                }

                variance[t][i][j] = var_ij.max(0.0);
            }
        }
    }

    variance
}

fn compute_bootstrap_variance(
    from_state: &[usize],
    to_state: &[usize],
    time: &[f64],
    weights: &[f64],
    n_states: usize,
    unique_times: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> Vec<Vec<Vec<f64>>> {
    let n = from_state.len();
    let n_times = unique_times.len();

    let bootstrap_results: Vec<Vec<Vec<Vec<f64>>>> = (0..n_bootstrap)
        .into_par_iter()
        .map(|b| {
            let mut rng = fastrand::Rng::with_seed(seed + b as u64);

            let indices: Vec<usize> = (0..n).map(|_| rng.usize(0..n)).collect();

            let boot_from: Vec<usize> = indices.iter().map(|&i| from_state[i]).collect();
            let boot_to: Vec<usize> = indices.iter().map(|&i| to_state[i]).collect();
            let boot_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let boot_weights: Vec<f64> = indices.iter().map(|&i| weights[i]).collect();

            let mut p_current = identity_matrix(n_states);
            let mut probs = Vec::with_capacity(n_times);

            for &ut in unique_times {
                let (trans_mat, _, _) = compute_transition_matrix(
                    &boot_from,
                    &boot_to,
                    &boot_time,
                    &boot_weights,
                    n_states,
                    ut,
                );
                p_current = matrix_multiply(&p_current, &trans_mat);
                probs.push(p_current.clone());
            }

            probs
        })
        .collect();

    let mut variance = vec![vec![vec![0.0; n_states]; n_states]; n_times];

    for t in 0..n_times {
        for i in 0..n_states {
            for j in 0..n_states {
                let values: Vec<f64> = bootstrap_results.iter().map(|r| r[t][i][j]).collect();
                let mean = values.iter().sum::<f64>() / n_bootstrap as f64;
                let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                    / (n_bootstrap - 1).max(1) as f64;
                variance[t][i][j] = var;
            }
        }
    }

    variance
}

fn compute_expected_sojourn(
    time: &[f64],
    state_probs: &[Vec<Vec<f64>>],
    n_states: usize,
) -> Vec<f64> {
    let n_times = time.len();
    let mut sojourn = vec![0.0; n_states];

    if n_times < 2 {
        return sojourn;
    }

    for state in 0..n_states {
        let mut total = 0.0;

        for t in 1..n_times {
            let dt = time[t] - time[t - 1];
            let p_state = state_probs[t - 1][0][state];
            total += dt * p_state;
        }

        sojourn[state] = total;
    }

    sojourn
}

#[pyfunction]
#[pyo3(signature = (from_state, to_state, time, config, weights=None))]
pub fn survfitaj_extended(
    from_state: Vec<usize>,
    to_state: Vec<usize>,
    time: Vec<f64>,
    config: &AalenJohansenExtendedConfig,
    weights: Option<Vec<f64>>,
) -> PyResult<AalenJohansenExtendedResult> {
    let n = from_state.len();
    if to_state.len() != n || time.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "from_state, to_state, and time must have equal length",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n]);

    let n_states = from_state
        .iter()
        .chain(to_state.iter())
        .max()
        .map(|&m| m + 1)
        .unwrap_or(2);

    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let mut transition_matrices = Vec::with_capacity(unique_times.len());
    let mut p_current = identity_matrix(n_states);
    let mut state_probs = Vec::with_capacity(unique_times.len());

    for &ut in &unique_times {
        let (trans_mat, n_at_risk, n_transitions) =
            compute_transition_matrix(&from_state, &to_state, &time, &wt, n_states, ut);

        transition_matrices.push(TransitionMatrix {
            time: ut,
            matrix: trans_mat.clone(),
            n_at_risk,
            n_transitions,
        });

        p_current = matrix_multiply(&p_current, &trans_mat);
        state_probs.push(p_current.clone());
    }

    let variance = match config.variance_estimator {
        VarianceEstimator::Greenwood => {
            compute_greenwood_variance(&state_probs, &transition_matrices, n_states)
        }
        VarianceEstimator::Aalen => {
            compute_aalen_variance(&state_probs, &transition_matrices, n_states)
        }
        VarianceEstimator::Bootstrap => {
            let seed = config.seed.unwrap_or(42);
            compute_bootstrap_variance(
                &from_state,
                &to_state,
                &time,
                &wt,
                n_states,
                &unique_times,
                config.n_bootstrap,
                seed,
            )
        }
    };

    let alpha = 1.0 - config.confidence_level;
    let z = match config.confidence_level {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        _ => 1.96,
    };

    let ci_lower: Vec<Vec<Vec<f64>>> = variance
        .iter()
        .zip(state_probs.iter())
        .map(|(var, probs)| {
            var.iter()
                .zip(probs.iter())
                .map(|(var_row, prob_row)| {
                    var_row
                        .iter()
                        .zip(prob_row.iter())
                        .map(|(&v, &p)| (p - z * v.sqrt()).clamp(0.0, 1.0))
                        .collect()
                })
                .collect()
        })
        .collect();

    let ci_upper: Vec<Vec<Vec<f64>>> = variance
        .iter()
        .zip(state_probs.iter())
        .map(|(var, probs)| {
            var.iter()
                .zip(probs.iter())
                .map(|(var_row, prob_row)| {
                    var_row
                        .iter()
                        .zip(prob_row.iter())
                        .map(|(&v, &p)| (p + z * v.sqrt()).clamp(0.0, 1.0))
                        .collect()
                })
                .collect()
        })
        .collect();

    let cumulative_incidence: Vec<Vec<f64>> = (0..n_states)
        .map(|j| state_probs.iter().map(|p| p[0][j]).collect())
        .collect();

    let expected_sojourn = if config.compute_sojourn {
        Some(compute_expected_sojourn(
            &unique_times,
            &state_probs,
            n_states,
        ))
    } else {
        None
    };

    let n_events = to_state
        .iter()
        .zip(from_state.iter())
        .filter(|(to, from)| **to != **from)
        .count();

    Ok(AalenJohansenExtendedResult {
        time: unique_times,
        state_probs,
        variance,
        ci_lower,
        ci_upper,
        transition_matrices,
        cumulative_incidence,
        expected_sojourn,
        n_states,
        n_obs: n,
        n_events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = AalenJohansenExtendedConfig::new(
            VarianceEstimator::Greenwood,
            TransitionType::Standard,
            200,
            0.95,
            true,
            None,
        )
        .unwrap();
        assert_eq!(config.n_bootstrap, 200);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            AalenJohansenExtendedConfig::new(
                VarianceEstimator::Greenwood,
                TransitionType::Standard,
                200,
                1.5,
                true,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn test_identity_matrix() {
        let id = identity_matrix(3);
        assert_eq!(id[0][0], 1.0);
        assert_eq!(id[0][1], 0.0);
        assert_eq!(id[1][1], 1.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = matrix_multiply(&a, &b);

        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[0][1] - 22.0).abs() < 1e-10);
        assert!((c[1][0] - 43.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_survfitaj_basic() {
        let from_state = vec![0, 0, 0, 1, 1];
        let to_state = vec![1, 2, 0, 2, 1];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = AalenJohansenExtendedConfig::new(
            VarianceEstimator::Greenwood,
            TransitionType::Standard,
            10,
            0.95,
            true,
            None,
        )
        .unwrap();

        let result = survfitaj_extended(from_state, to_state, time, &config, None).unwrap();

        assert!(result.n_states >= 3);
        assert_eq!(result.time.len(), 5);
        assert!(!result.state_probs.is_empty());
    }

    #[test]
    fn test_cif() {
        let from_state = vec![0, 0, 0, 0, 0];
        let to_state = vec![1, 1, 2, 0, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = AalenJohansenExtendedConfig::new(
            VarianceEstimator::Aalen,
            TransitionType::Standard,
            10,
            0.95,
            false,
            None,
        )
        .unwrap();

        let result = survfitaj_extended(from_state, to_state, time, &config, None).unwrap();

        let cif_1 = result.get_cif(1).unwrap();
        assert_eq!(cif_1.len(), result.time.len());
    }
}
