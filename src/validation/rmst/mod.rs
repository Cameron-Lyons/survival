use crate::constants::{
    DEFAULT_CONFIDENCE_LEVEL, PARALLEL_THRESHOLD_XLARGE, z_score_for_confidence,
};
use crate::internal::statistical::{chi2_cdf, normal_cdf as norm_cdf};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

include!("core.rs");
include!("quantiles.rs");
include!("threshold.rs");
include!("tests.rs");
