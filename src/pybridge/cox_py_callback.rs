//! Port of R's `cox_Rcallback.c`: evaluate a penalty closure and read back the
//! `coxlist` it returns.
//!
//! `coxpenal.fit` (R `coxfit5.c`/`agfit5.c`) calls `cox_callback` once per
//! Newton iteration for the sparse frailty term (`which == 1`, R's `coxlist1`)
//! and once for the remaining penalised terms (`which == 2`, `coxlist2`). The
//! Rust port (`regression::coxpenal`) evaluates its built-in penalties in
//! Rust and reaches [`evaluate_penalty`] for a user-defined penalty
//! (`PenaltyTerm::Callback`), adding the returned derivatives to the score
//! and information matrix; the `#[pyfunction]` wrapper exists so the Python
//! side can exercise the same contract directly.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::internal::numpy_utils::{BoolVec, FloatVec};
pub use crate::regression::coxpenal::CoxPenaltyTerms;

fn item<'py>(coxlist: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    coxlist.get_item(key).map_err(|err| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
            "penalty callback result has no '{key}' entry: {err}"
        ))
    })
}

fn float_item(coxlist: &Bound<'_, PyAny>, key: &str) -> PyResult<Vec<f64>> {
    item(coxlist, key)?
        .extract::<FloatVec>()
        .map(FloatVec::into_inner)
        .map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("{key}: invalid type ({err})"))
        })
}

/// Calls `fexpr(coef, which=which)` and reads the returned mapping.
///
/// The callback receives the coefficients as a NumPy array and may return
/// lists, NumPy arrays or scalars for each entry; `penalty` is a number and
/// `flag` a boolean or a sequence of booleans.
pub(crate) fn evaluate_penalty(
    fexpr: &Bound<'_, PyAny>,
    which: i32,
    coef: &[f64],
) -> PyResult<CoxPenaltyTerms> {
    let py = fexpr.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("which", which)?;
    let coxlist = fexpr.call((FloatVec(coef.to_vec()),), Some(&kwargs))?;

    let penalty_item = item(&coxlist, "penalty")?;
    let penalty = match penalty_item.extract::<f64>() {
        Ok(value) => value,
        Err(_) => match penalty_item.extract::<FloatVec>()?.as_ref() {
            [value] => *value,
            values => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "penalty must be a single number, got {} values",
                    values.len()
                )));
            }
        },
    };
    let flag_item = item(&coxlist, "flag")?;
    let flag = match flag_item.extract::<bool>() {
        Ok(value) => vec![value],
        Err(_) => flag_item.extract::<BoolVec>()?.into_inner(),
    };

    Ok(CoxPenaltyTerms {
        coef: float_item(&coxlist, "coef")?,
        first: float_item(&coxlist, "first")?,
        second: float_item(&coxlist, "second")?,
        penalty,
        flag,
    })
}

#[pyfunction]
#[pyo3(signature = (which, coef, fexpr))]
pub(crate) fn cox_callback(
    which: i32,
    coef: FloatVec,
    fexpr: &Bound<'_, PyAny>,
) -> PyResult<CoxPenaltyTerms> {
    evaluate_penalty(fexpr, which, &coef)
}

#[cfg(test)]
mod tests {
    //! Needs an interpreter with NumPy: `PYO3_PYTHON` must point at one.

    use super::*;
    use std::ffi::CString;

    fn callback<'py>(py: Python<'py>, body: &str) -> Bound<'py, PyAny> {
        let globals = PyDict::new(py);
        globals.set_item("np", py.import("numpy").unwrap()).unwrap();
        let code = CString::new(format!("lambda coef, *, which: {body}")).unwrap();
        py.eval(&code, Some(&globals), None).unwrap()
    }

    #[test]
    fn reads_back_lists_arrays_and_scalars() {
        Python::initialize();
        Python::attach(|py| {
            let fexpr = callback(
                py,
                "{'coef': coef + which, 'first': [1.0, 2.0], 'second': np.eye(2).ravel(), \
                 'penalty': 5.0, 'flag': [True, False]}",
            );
            let terms = evaluate_penalty(&fexpr, 2, &[1.0, 2.0]).unwrap();
            assert_eq!(
                terms,
                CoxPenaltyTerms {
                    coef: vec![3.0, 4.0],
                    first: vec![1.0, 2.0],
                    second: vec![1.0, 0.0, 0.0, 1.0],
                    penalty: 5.0,
                    flag: vec![true, false],
                }
            );

            let fexpr = callback(
                py,
                "{'coef': coef, 'first': [0.5], 'second': [0.25], 'penalty': [-1.0], 'flag': False}",
            );
            let terms = evaluate_penalty(&fexpr, 1, &[0.0]).unwrap();
            assert_eq!(terms.penalty, -1.0);
            assert_eq!(terms.flag, vec![false]);
        });
    }

    #[test]
    fn missing_or_mistyped_entries_are_reported_by_name() {
        Python::initialize();
        Python::attach(|py| {
            let fexpr = callback(py, "{'coef': coef, 'first': [0.0]}");
            let err = evaluate_penalty(&fexpr, 1, &[0.0]).unwrap_err();
            assert!(err.to_string().contains("'penalty'"), "{err}");

            let fexpr = callback(
                py,
                "{'coef': coef, 'first': 'x', 'second': [0.0], 'penalty': 0.0, 'flag': True}",
            );
            let err = evaluate_penalty(&fexpr, 1, &[0.0]).unwrap_err();
            assert!(err.to_string().starts_with("TypeError: first:"), "{err}");

            let fexpr = callback(
                py,
                "{'coef': coef, 'first': [0.0], 'second': [0.0], 'penalty': [1.0, 2.0], 'flag': True}",
            );
            let err = evaluate_penalty(&fexpr, 1, &[0.0]).unwrap_err();
            assert!(err.to_string().contains("single number"), "{err}");
        });
    }
}
