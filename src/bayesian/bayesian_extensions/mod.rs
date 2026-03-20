use crate::internal::statistical::sample_normal;
use pyo3::prelude::*;

include!("dirichlet_process.rs");
include!("model_averaging.rs");
include!("shrinkage.rs");
include!("tests.rs");
