

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

fn layer_norm_3d<B: burn::prelude::Backend>(
    x: Tensor<B, 3>,
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let [batch, seq, hidden] = x.dims();
    let x_2d: Tensor<B, 2> = x.reshape([batch * seq, hidden]);
    let normed = layer_norm(x_2d, gamma, beta, eps);
    normed.reshape([batch, seq, hidden])
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TracerConfig {
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub num_factorized_layers: usize,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_events: usize,
    #[pyo3(get, set)]
    pub mlp_hidden_size: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub layer_norm_eps: f32,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl TracerConfig {
    #[new]
    #[pyo3(signature = (
        embedding_dim=32,
        num_factorized_layers=2,
        num_attention_heads=4,
        num_durations=10,
        num_events=1,
        mlp_hidden_size=64,
        dropout_rate=0.1,
        learning_rate=0.0001,
        weight_decay=0.00001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        layer_norm_eps=1e-12,
        seed=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        embedding_dim: usize,
        num_factorized_layers: usize,
        num_attention_heads: usize,
        num_durations: usize,
        num_events: usize,
        mlp_hidden_size: usize,
        dropout_rate: f64,
        learning_rate: f64,
        weight_decay: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        layer_norm_eps: f32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        ensure_positive_usize("embedding_dim", embedding_dim)?;
        ensure_positive_usize("num_factorized_layers", num_factorized_layers)?;
        ensure_positive_usize("num_attention_heads", num_attention_heads)?;
        if !embedding_dim.is_multiple_of(num_attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "embedding_dim must be divisible by num_attention_heads",
            ));
        }
        ensure_positive_usize("num_durations", num_durations)?;
        ensure_positive_usize("batch_size", batch_size)?;
        ensure_positive_usize("n_epochs", n_epochs)?;
        ensure_open_unit_interval("validation_fraction", validation_fraction)?;
        ensure_open_unit_interval("dropout_rate", dropout_rate)?;

        Ok(TracerConfig {
            embedding_dim,
            num_factorized_layers,
            num_attention_heads,
            num_durations,
            num_events,
            mlp_hidden_size,
            dropout_rate,
            learning_rate,
            weight_decay,
            batch_size,
            n_epochs,
            early_stopping_patience,
            validation_fraction,
            layer_norm_eps,
            seed,
        })
    }
}
