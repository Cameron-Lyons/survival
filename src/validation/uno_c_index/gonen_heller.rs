
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct GonenHellerResult {
    pub cpe: f64,
    pub n_pairs: usize,
    pub n_ties: usize,
    pub variance: f64,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

impl fmt::Display for GonenHellerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GonenHellerResult(cpe={:.4}, se={:.4}, ci=[{:.4}, {:.4}])",
            self.cpe, self.std_error, self.ci_lower, self.ci_upper
        )
    }
}

#[pymethods]
impl GonenHellerResult {
    #[new]
    fn new(
        cpe: f64,
        n_pairs: usize,
        n_ties: usize,
        variance: f64,
        std_error: f64,
        ci_lower: f64,
        ci_upper: f64,
    ) -> Self {
        Self {
            cpe,
            n_pairs,
            n_ties,
            variance,
            std_error,
            ci_lower,
            ci_upper,
        }
    }
}

pub(crate) fn gonen_heller_core(linear_predictor: &[f64]) -> GonenHellerResult {
    let n = linear_predictor.len();

    if n < 2 {
        return GonenHellerResult {
            cpe: 0.5,
            n_pairs: 0,
            n_ties: 0,
            variance: 0.0,
            std_error: 0.0,
            ci_lower: 0.5,
            ci_upper: 0.5,
        };
    }

    let compute_contributions = |i: usize| -> (f64, usize, usize, Vec<f64>) {
        let mut sum = 0.0;
        let mut pairs = 0usize;
        let mut ties = 0usize;
        let mut influence = vec![0.0; n];

        for j in (i + 1)..n {
            let diff = linear_predictor[i] - linear_predictor[j];

            if diff.abs() < 1e-10 {
                ties += 1;
                continue;
            }

            pairs += 1;
            let contribution = 1.0 / (1.0 + (-diff.abs()).exp());
            sum += contribution;

            let deriv = contribution * (1.0 - contribution);
            if diff > 0.0 {
                influence[i] += deriv;
                influence[j] -= deriv;
            } else {
                influence[i] -= deriv;
                influence[j] += deriv;
            }
        }

        (sum, pairs, ties, influence)
    };

    let results: Vec<(f64, usize, usize, Vec<f64>)> = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n).into_par_iter().map(compute_contributions).collect()
    } else {
        (0..n).map(compute_contributions).collect()
    };

    let mut total_sum = 0.0;
    let mut total_pairs = 0usize;
    let mut total_ties = 0usize;
    let mut influence_sums = vec![0.0; n];

    for (sum, pairs, ties, influence) in results {
        total_sum += sum;
        total_pairs += pairs;
        total_ties += ties;
        for (k, &inf) in influence.iter().enumerate() {
            influence_sums[k] += inf;
        }
    }

    let cpe = if total_pairs > 0 {
        total_sum / total_pairs as f64
    } else {
        0.5
    };

    let variance = if total_pairs > 0 {
        let n_f = n as f64;
        let pairs_f = total_pairs as f64;
        let mut var_sum = 0.0;

        for &inf in &influence_sums {
            let normalized = inf / pairs_f;
            var_sum += normalized * normalized;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error = variance.sqrt();
    let z = 1.96;
    let ci_lower = (cpe - z * std_error).clamp(0.0, 1.0);
    let ci_upper = (cpe + z * std_error).clamp(0.0, 1.0);

    GonenHellerResult {
        cpe,
        n_pairs: total_pairs,
        n_ties: total_ties,
        variance,
        std_error,
        ci_lower,
        ci_upper,
    }
}

#[pyfunction]
pub fn gonen_heller_concordance(linear_predictor: Vec<f64>) -> PyResult<GonenHellerResult> {
    if linear_predictor.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "linear_predictor must not be empty",
        ));
    }

    Ok(gonen_heller_core(&linear_predictor))
}

