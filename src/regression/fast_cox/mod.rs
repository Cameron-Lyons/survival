use ndarray::{ArrayView1, ArrayView2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{SurvivalError, SurvivalResult};

include!("types.rs");
include!("core.rs");
include!("api.rs");
include!("tests.rs");
