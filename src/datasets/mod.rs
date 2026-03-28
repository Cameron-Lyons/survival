//! Embedded survival analysis datasets from the R survival package
//!
//! Each dataset is accessible via a `load_*` function that returns a Python dict
//! with column names as keys and lists/arrays as values.

mod classic;
mod cohort;
mod common;
mod parser;

pub(crate) use classic::*;
pub(crate) use cohort::*;
