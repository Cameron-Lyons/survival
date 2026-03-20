

fn selu_activation<B: burn::prelude::Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let alpha: f32 = 1.673_263_2;
    let scale: f32 = 1.050_701;
    let positive = x.clone().clamp_min(0.0);
    let negative = (x.clamp_max(0.0).exp() - 1.0) * alpha;
    (positive + negative) * scale
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum Activation {
    ReLU,
    SELU,
    Tanh,
}

#[pymethods]
impl Activation {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "relu" => Ok(Activation::ReLU),
            "selu" => Ok(Activation::SELU),
            "tanh" => Ok(Activation::Tanh),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown activation function. Use 'relu', 'selu', or 'tanh'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DeepSurvConfig {
    #[pyo3(get, set)]
    pub hidden_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub activation: Activation,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub l2_reg: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
}

#[pymethods]
impl DeepSurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_layers=None,
        activation=Activation::SELU,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=100,
        l2_reg=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_layers: Option<Vec<usize>>,
        activation: Activation,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        l2_reg: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
    ) -> PyResult<Self> {
        ensure_open_unit_interval("dropout_rate", dropout_rate)?;
        ensure_positive_f64("learning_rate", learning_rate)?;
        ensure_positive_usize("batch_size", batch_size)?;
        ensure_positive_usize("n_epochs", n_epochs)?;
        ensure_open_unit_interval("validation_fraction", validation_fraction)?;

        Ok(DeepSurvConfig {
            hidden_layers: hidden_layers.unwrap_or_else(|| vec![64, 32]),
            activation,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            l2_reg,
            seed,
            early_stopping_patience,
            validation_fraction,
        })
    }
}
