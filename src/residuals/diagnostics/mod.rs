use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::PARALLEL_THRESHOLD_MEDIUM;
use crate::internal::matrix::invert_flat_square_matrix_with_fallback;
use crate::internal::statistical::{lower_incomplete_gamma, normal_cdf};

include!("dfbeta.rs");
include!("leverage.rs");
include!("schoenfeld.rs");
include!("influence.rs");
include!("tests.rs");
