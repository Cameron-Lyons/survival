use pyo3::prelude::*;
use rayon::prelude::*;

use crate::internal::statistical::{chi2_cdf, ln_gamma, normal_cdf};

include!("pwp.rs");
include!("wlw.rs");
include!("negative_binomial.rs");
include!("anderson_gill.rs");
include!("tests.rs");
