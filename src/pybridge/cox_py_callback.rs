use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

type CoxCallbackResult = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i32>);

#[pyfunction]
pub fn cox_callback(
    which: i32,
    mut coef: Vec<f64>,
    mut first: Vec<f64>,
    mut second: Vec<f64>,
    mut penalty: Vec<f64>,
    mut flag: Vec<i32>,
    fexpr: &Bound<PyAny>,
) -> PyResult<CoxCallbackResult> {
    let py = fexpr.py();
    let coef_vec: Vec<f64> = coef.to_vec();
    let coef_list = PyList::new(py, &coef_vec)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("which", which)?;
    let result = fexpr.call((coef_list.as_any(),), Some(&kwargs))?;
    let dict = result.cast::<PyDict>()?;
    macro_rules! extract_values {
        ($key:expr, $rust_slice:expr, $pytype:ty) => {
            let item = dict.get_item($key)?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Missing key: {}", $key))
            })?;
            let py_values = item.cast::<PyList>()?;
            if py_values.len() != $rust_slice.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid length for {}",
                    $key
                )));
            }
            for (i, item) in py_values.iter().enumerate() {
                $rust_slice[i] = item.extract::<$pytype>()?;
            }
        };
    }
    extract_values!("coef", coef, f64);
    extract_values!("first", first, f64);
    extract_values!("second", second, f64);
    extract_values!("penalty", penalty, f64);
    let flag_item = dict
        .get_item("flag")?
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key: flag"))?;
    let py_flags = flag_item.cast::<PyList>()?;
    if py_flags.len() != flag.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid length for flag",
        ));
    }
    for (i, item) in py_flags.iter().enumerate() {
        flag[i] = match item.extract::<bool>() {
            Ok(b) => b as i32,
            Err(_) => item.extract::<i32>()?,
        };
    }
    Ok((coef, first, second, penalty, flag))
}
