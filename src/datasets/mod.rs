//! Bundled copies of the datasets shipped with R's `survival` package.
//!
//! See [`catalog`] for the list of tables and [`common`] for how R column
//! types are represented.

mod catalog;
mod common;
mod parser;

pub(crate) use catalog::*;
