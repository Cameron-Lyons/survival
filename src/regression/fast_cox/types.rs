

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum ScreeningRule {
    None,
    Safe,
    Strong,
    EDPP,
}

#[pymethods]
impl ScreeningRule {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "none" => Ok(ScreeningRule::None),
            "safe" => Ok(ScreeningRule::Safe),
            "strong" => Ok(ScreeningRule::Strong),
            "edpp" => Ok(ScreeningRule::EDPP),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown screening rule. Use 'none', 'safe', 'strong', or 'edpp'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FastCoxConfig {
    #[pyo3(get, set)]
    pub lambda: f64,
    #[pyo3(get, set)]
    pub l1_ratio: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub screening: ScreeningRule,
    #[pyo3(get, set)]
    pub working_set_size: Option<usize>,
    #[pyo3(get, set)]
    pub active_set_update_freq: usize,
    #[pyo3(get, set)]
    pub standardize: bool,
    #[pyo3(get, set)]
    pub use_simd: bool,
}

#[pymethods]
impl FastCoxConfig {
    #[new]
    #[pyo3(signature = (
        lambda=0.1,
        l1_ratio=1.0,
        max_iter=1000,
        tol=1e-7,
        screening=ScreeningRule::Strong,
        working_set_size=None,
        active_set_update_freq=10,
        standardize=true,
        use_simd=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lambda: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
        screening: ScreeningRule,
        working_set_size: Option<usize>,
        active_set_update_freq: usize,
        standardize: bool,
        use_simd: bool,
    ) -> PyResult<Self> {
        if lambda < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lambda must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "l1_ratio must be between 0 and 1",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        if active_set_update_freq == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "active_set_update_freq must be positive",
            ));
        }

        Ok(FastCoxConfig {
            lambda,
            l1_ratio,
            max_iter,
            tol,
            screening,
            working_set_size,
            active_set_update_freq,
            standardize,
            use_simd,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FastCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub nonzero_indices: Vec<usize>,
    #[pyo3(get)]
    pub lambda_used: f64,
    #[pyo3(get)]
    pub l1_ratio: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub deviance: f64,
    #[pyo3(get)]
    pub df: f64,
    #[pyo3(get)]
    pub scale_factors: Option<Vec<f64>>,
    #[pyo3(get)]
    pub center_values: Option<Vec<f64>>,
    #[pyo3(get)]
    pub screened_out: usize,
    #[pyo3(get)]
    pub active_set_size: usize,
}

#[pymethods]
impl FastCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "FastCoxResult(nonzero={}, iter={}, converged={}, screened_out={})",
            self.nonzero_indices.len(),
            self.n_iter,
            self.converged,
            self.screened_out
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FastCoxPath {
    #[pyo3(get)]
    pub lambdas: Vec<f64>,
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub deviances: Vec<f64>,
    #[pyo3(get)]
    pub df: Vec<f64>,
    #[pyo3(get)]
    pub n_iters: Vec<usize>,
    #[pyo3(get)]
    pub converged: Vec<bool>,
}

#[pymethods]
impl FastCoxPath {
    fn __repr__(&self) -> String {
        format!(
            "FastCoxPath(n_lambda={}, converged={})",
            self.lambdas.len(),
            self.converged.iter().filter(|&&c| c).count()
        )
    }
}

