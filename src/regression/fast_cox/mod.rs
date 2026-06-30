use ndarray::{ArrayView1, ArrayView2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::internal::matrix::standardize_row_major_matrix;
use crate::{SurvivalError, SurvivalResult};

include!("types.rs");
include!("core.rs");
include!("api.rs");
include!("tests.rs");
