use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::activation::{relu, tanh},
};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::config_validation::{
    ensure_open_unit_interval, ensure_positive_f64, ensure_positive_usize,
};
use super::utils::{
    EarlyStopping, shuffled_epoch_indices, tensor_to_vec_f32, train_validation_split_indices,
};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

include!("common.rs");
include!("network.rs");
include!("training.rs");
include!("api.rs");
include!("tests.rs");
