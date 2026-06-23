
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

struct GonenHellerAccumulator {
    total_sum: f64,
    total_pairs: usize,
    total_ties: usize,
    influence_sums: Vec<f64>,
}

impl GonenHellerAccumulator {
    fn new(n: usize) -> Self {
        Self {
            total_sum: 0.0,
            total_pairs: 0,
            total_ties: 0,
            influence_sums: vec![0.0; n],
        }
    }

    fn merge(&mut self, other: Self) {
        self.total_sum += other.total_sum;
        self.total_pairs += other.total_pairs;
        self.total_ties += other.total_ties;
        for (left, right) in self
            .influence_sums
            .iter_mut()
            .zip(other.influence_sums.iter())
        {
            *left += right;
        }
    }
}

fn accumulate_gonen_heller_row(
    accumulator: &mut GonenHellerAccumulator,
    linear_predictor: &[f64],
    i: usize,
) {
    for j in (i + 1)..linear_predictor.len() {
        let diff = linear_predictor[i] - linear_predictor[j];

        if diff.abs() < 1e-10 {
            accumulator.total_ties += 1;
            continue;
        }

        accumulator.total_pairs += 1;
        let contribution = 1.0 / (1.0 + (-diff.abs()).exp());
        accumulator.total_sum += contribution;

        let deriv = contribution * (1.0 - contribution);
        if diff > 0.0 {
            accumulator.influence_sums[i] += deriv;
            accumulator.influence_sums[j] -= deriv;
        } else {
            accumulator.influence_sums[i] -= deriv;
            accumulator.influence_sums[j] += deriv;
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

    let accumulator = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n)
            .into_par_iter()
            .fold(
                || GonenHellerAccumulator::new(n),
                |mut accumulator, i| {
                    accumulate_gonen_heller_row(&mut accumulator, linear_predictor, i);
                    accumulator
                },
            )
            .reduce(
                || GonenHellerAccumulator::new(n),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            )
    } else {
        let mut accumulator = GonenHellerAccumulator::new(n);
        for i in 0..n {
            accumulate_gonen_heller_row(&mut accumulator, linear_predictor, i);
        }
        accumulator
    };

    let cpe = if accumulator.total_pairs > 0 {
        accumulator.total_sum / accumulator.total_pairs as f64
    } else {
        0.5
    };

    let variance = if accumulator.total_pairs > 0 {
        let n_f = n as f64;
        let pairs_f = accumulator.total_pairs as f64;
        let mut var_sum = 0.0;

        for &inf in &accumulator.influence_sums {
            let normalized = inf / pairs_f;
            var_sum += normalized * normalized;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error = variance.sqrt();
    let (ci_lower, ci_upper) = clamped_normal_ci_95(cpe, std_error, 0.0, 1.0);

    GonenHellerResult {
        cpe,
        n_pairs: accumulator.total_pairs,
        n_ties: accumulator.total_ties,
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
    validate_no_nan(&linear_predictor, "linear_predictor")?;
    validate_finite(&linear_predictor, "linear_predictor")?;

    Ok(gonen_heller_core(&linear_predictor))
}
