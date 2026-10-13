use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::{Device, activation::relu},
};
use pyo3::prelude::*;

use super::utils::{
    EarlyStopping, compute_duration_bins, shuffled_epoch_indices, tensor_to_vec_f32,
    train_validation_split_indices, training_device,
};

include!("config.rs");
include!("network.rs");
include!("api.rs");
include!("tests.rs");
