use crate::internal::statistical::chi2_sf;
use crate::simd_ops::{dot_product_simd, mean_simd, subtract_scalar_simd, sum_of_squares_simd};
use pyo3::prelude::*;

include!("dcal.rs");
include!("one_calibration.rs");
include!("plot.rs");
include!("brier.rs");
include!("multi_time.rs");
include!("smoothed.rs");
include!("tests.rs");
