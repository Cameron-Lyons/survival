use pyo3::Py;
use pyo3::prelude::*;
use rayon::prelude::*;

type ShapTensor = Vec<Vec<Vec<f64>>>;
type ShapComputation = (ShapTensor, Vec<f64>);

include!("types.rs");
include!("kernel.rs");
include!("explain.rs");
include!("aggregate.rs");
include!("tests.rs");
