use pyo3::prelude::*;
use pyo3::types::PyDict;
#[pyfunction]
pub fn agsurv5(
    n: usize,
    nvar: usize,
    dd: Vec<i32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    xsum: Vec<f64>,
    xsum2: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    let dd_slice = &dd;
    let x1_slice = &x1;
    let x2_slice = &x2;
    let xsum_slice = &xsum;
    let xsum2_slice = &xsum2;
    let mut sum1 = vec![0.0; n];
    let mut sum2 = vec![0.0; n];
    let mut xbar = vec![0.0; n * nvar];
    for i in 0..n {
        let d = dd_slice[i] as f64;
        if d == 1.0 {
            let temp = 1.0 / x1_slice[i];
            sum1[i] = temp;
            sum2[i] = temp.powi(2);
            for k in 0..nvar {
                let idx = i + n * k;
                xbar[idx] = xsum_slice[idx] * temp.powi(2);
            }
        } else {
            let d_int = dd_slice[i];
            let mut temp;
            for j in 0..d_int {
                let j_f64 = j as f64;
                temp = 1.0 / (x1_slice[i] - x2_slice[i] * j_f64 / d);
                sum1[i] += temp / d;
                sum2[i] += temp.powi(2) / d;
                for k in 0..nvar {
                    let idx = i + n * k;
                    let weighted_x = xsum_slice[idx] - xsum2_slice[idx] * j_f64 / d;
                    xbar[idx] += (weighted_x * temp.powi(2)) / d;
                }
            }
        }
    }
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("sum1", sum1)?;
        dict.set_item("sum2", sum2)?;
        dict.set_item("xbar", xbar)?;
        Ok(dict.into())
    })
}
