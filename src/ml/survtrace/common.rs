

fn gelu<B: burn::prelude::Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sqrt_2 = (2.0_f32).sqrt();
    let cdf = (x.clone() / sqrt_2).erf().add_scalar(1.0) * 0.5;
    x * cdf
}

fn layer_norm<B: burn::prelude::Backend>(
    x: Tensor<B, 2>,
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 2> {
    let [_batch, hidden] = x.dims();
    let mean = x.clone().mean_dim(1);
    let var = x.clone().var(1);
    let x_norm = (x - mean) / (var + eps).sqrt();
    let gamma_expanded: Tensor<B, 2> = gamma.reshape([1, hidden]);
    let beta_expanded: Tensor<B, 2> = beta.reshape([1, hidden]);
    x_norm * gamma_expanded + beta_expanded
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum SurvTraceActivation {
    GELU,
    ReLU,
}

#[pymethods]
impl SurvTraceActivation {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "gelu" => Ok(SurvTraceActivation::GELU),
            "relu" => Ok(SurvTraceActivation::ReLU),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown activation. Use 'gelu' or 'relu'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvTraceConfig {
    #[pyo3(get, set)]
    pub hidden_size: usize,
    #[pyo3(get, set)]
    pub num_hidden_layers: usize,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub intermediate_size: usize,
    #[pyo3(get, set)]
    pub hidden_dropout_prob: f64,
    #[pyo3(get, set)]
    pub attention_dropout_prob: f64,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_events: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub layer_norm_eps: f32,
}

#[pymethods]
impl SurvTraceConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.1,
        num_durations=5,
        num_events=1,
        vocab_size=8,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        weight_decay=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1,
        layer_norm_eps=1e-12
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        hidden_dropout_prob: f64,
        attention_dropout_prob: f64,
        num_durations: usize,
        num_events: usize,
        vocab_size: usize,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        weight_decay: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        layer_norm_eps: f32,
    ) -> PyResult<Self> {
        ensure_positive_usize("hidden_size", hidden_size)?;
        ensure_positive_usize("num_hidden_layers", num_hidden_layers)?;
        ensure_positive_usize("num_attention_heads", num_attention_heads)?;
        if !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        ensure_positive_usize("num_durations", num_durations)?;
        ensure_positive_usize("batch_size", batch_size)?;
        ensure_positive_usize("n_epochs", n_epochs)?;
        ensure_open_unit_interval("validation_fraction", validation_fraction)?;

        Ok(SurvTraceConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_dropout_prob,
            attention_dropout_prob,
            num_durations,
            num_events,
            vocab_size,
            learning_rate,
            batch_size,
            n_epochs,
            weight_decay,
            seed,
            early_stopping_patience,
            validation_fraction,
            layer_norm_eps,
        })
    }
}
