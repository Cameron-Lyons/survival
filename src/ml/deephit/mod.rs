use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::activation::relu,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::config_validation::{
    ensure_closed_unit_interval, ensure_open_unit_interval, ensure_positive_f64,
    ensure_positive_usize,
};
use super::utils::{
    EarlyStopping, compute_duration_bins, linear_forward, relu_vec, shuffled_epoch_indices,
    tensor_to_vec_f32, train_validation_split_indices,
};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

include!("config.rs");
include!("network.rs");
include!("weights.rs");
include!("api.rs");
include!("tests.rs");
