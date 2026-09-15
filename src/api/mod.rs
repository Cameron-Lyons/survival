//! Language bindings. `python` registers every `#[pyclass]`/`#[pyfunction]`
//! with the `_survival` extension module; `registration_audit` is the
//! test-time scan that keeps that registration complete.

#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod registration_audit;
