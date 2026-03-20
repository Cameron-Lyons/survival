use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

type PredictionMatrix = Vec<Vec<f64>>;
type QuantilePredictionBands = (PredictionMatrix, PredictionMatrix, PredictionMatrix);

include!("dropout.rs");
include!("quantiles.rs");
include!("conformal.rs");
include!("bootstrap.rs");
include!("jackknife.rs");
include!("tests.rs");
