

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

pub enum TemporalType {
    LSTM,
    GRU,
    Attention,
    LSTMAttention,
}

#[pymethods]
impl TemporalType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "lstm" => Ok(TemporalType::LSTM),
            "gru" => Ok(TemporalType::GRU),
            "attention" => Ok(TemporalType::Attention),
            "lstm_attention" | "lstmattention" => Ok(TemporalType::LSTMAttention),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown temporal type. Use 'lstm', 'gru', 'attention', or 'lstm_attention'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DynamicDeepHitConfig {
    #[pyo3(get, set)]
    pub temporal_type: TemporalType,
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub num_temporal_layers: usize,
    #[pyo3(get, set)]
    pub bidirectional: bool,
    #[pyo3(get, set)]
    pub shared_hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub cause_hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_causes: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl DynamicDeepHitConfig {
    #[new]
    #[pyo3(signature = (
        temporal_type=TemporalType::LSTM,
        embedding_dim=64,
        num_temporal_layers=2,
        bidirectional=false,
        shared_hidden_sizes=vec![64, 64],
        cause_hidden_sizes=vec![32],
        num_durations=10,
        num_causes=1,
        dropout_rate=0.1,
        alpha=0.5,
        sigma=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        temporal_type: TemporalType,
        embedding_dim: usize,
        num_temporal_layers: usize,
        bidirectional: bool,
        shared_hidden_sizes: Vec<usize>,
        cause_hidden_sizes: Vec<usize>,
        num_durations: usize,
        num_causes: usize,
        dropout_rate: f64,
        alpha: f64,
        sigma: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if embedding_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "embedding_dim must be positive",
            ));
        }
        if num_temporal_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_temporal_layers must be positive",
            ));
        }
        if num_durations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_durations must be positive",
            ));
        }
        if num_causes == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_causes must be positive",
            ));
        }
        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be positive",
            ));
        }
        if !(0.0..1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be in [0, 1)",
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be in [0, 1]",
            ));
        }
        if !(0.0..1.0).contains(&validation_fraction) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "validation_fraction must be in [0, 1)",
            ));
        }

        Ok(DynamicDeepHitConfig {
            temporal_type,
            embedding_dim,
            num_temporal_layers,
            bidirectional,
            shared_hidden_sizes,
            cause_hidden_sizes,
            num_durations,
            num_causes,
            dropout_rate,
            alpha,
            sigma,
            learning_rate,
            batch_size,
            n_epochs,
            early_stopping_patience,
            validation_fraction,
            seed,
        })
    }
}
