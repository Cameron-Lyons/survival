use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::activation::relu,
};
use pyo3::prelude::*;

use super::utils::{
    EarlyStopping, compute_duration_bins, shuffled_epoch_indices, tensor_to_vec_f32,
    train_validation_split_indices,
};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

include!("config.rs");
include!("network.rs");
include!("api.rs");
include!("tests.rs");
