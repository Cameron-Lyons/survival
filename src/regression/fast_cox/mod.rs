use ndarray::{ArrayView1, ArrayView2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::internal::cox_risk::{
    cox_risk_shift, precompute_cox_risk_set_cumsum, shifted_exp_eta_with_shift,
};
use crate::internal::matrix::{
    standardize_or_borrow_row_major_matrix, standardize_row_major_matrix,
};
use crate::{SurvivalError, SurvivalResult};

include!("types.rs");
include!("core.rs");
include!("api.rs");
include!("tests.rs");
