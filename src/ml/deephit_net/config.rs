

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct DeepHitConfig {
    #[pyo3(get, set)]
    pub shared_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub cause_specific_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_risks: usize,
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
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub use_batch_norm: bool,
}

#[pymethods]
impl DeepHitConfig {
    #[new]
    #[pyo3(signature = (
        shared_layers=None,
        cause_specific_layers=None,
        num_durations=10,
        num_risks=1,
        dropout_rate=0.1,
        alpha=0.2,
        sigma=0.1,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=100,
        weight_decay=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1,
        use_batch_norm=true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shared_layers: Option<Vec<usize>>,
        cause_specific_layers: Option<Vec<usize>>,
        num_durations: usize,
        num_risks: usize,
        dropout_rate: f64,
        alpha: f64,
        sigma: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        weight_decay: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        use_batch_norm: bool,
    ) -> PyResult<Self> {
        ensure_positive_usize("num_durations", num_durations)?;
        ensure_positive_usize("num_risks", num_risks)?;
        ensure_open_unit_interval("dropout_rate", dropout_rate)?;
        ensure_closed_unit_interval("alpha", alpha)?;
        ensure_positive_f64("sigma", sigma)?;
        ensure_positive_usize("batch_size", batch_size)?;
        ensure_positive_usize("n_epochs", n_epochs)?;
        ensure_open_unit_interval("validation_fraction", validation_fraction)?;

        Ok(DeepHitConfig {
            shared_layers: shared_layers.unwrap_or_else(|| vec![64, 64]),
            cause_specific_layers: cause_specific_layers.unwrap_or_else(|| vec![32]),
            num_durations,
            num_risks,
            dropout_rate,
            alpha,
            sigma,
            learning_rate,
            batch_size,
            n_epochs,
            weight_decay,
            seed,
            early_stopping_patience,
            validation_fraction,
            use_batch_norm,
        })
    }
}
