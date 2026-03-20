use pyo3::prelude::*;
use pyo3::types::PyDict;
#[pyfunction]
pub fn collapse(
    y: Vec<f64>,
    x: Vec<i32>,
    istate: Vec<i32>,
    id: Vec<i32>,
    wt: Vec<f64>,
    order: Vec<i32>,
) -> PyResult<Py<PyAny>> {
    let y_slice = &y;
    let x_slice = &x;
    let istate_slice = &istate;
    let id_slice = &id;
    let wt_slice = &wt;
    let order_slice = &order;
    let n = id_slice.len();
    assert_eq!(y_slice.len(), 3 * n, "y must have 3 columns");
    assert_eq!(x_slice.len(), n, "x length mismatch");
    assert_eq!(istate_slice.len(), n, "istate length mismatch");
    assert_eq!(wt_slice.len(), n, "wt length mismatch");
    assert_eq!(order_slice.len(), n, "order length mismatch");
    let time1 = &y_slice[0..n];
    let time2 = &y_slice[n..2 * n];
    let status = &y_slice[2 * n..3 * n];
    let mut i1 = Vec::new();
    let mut i2 = Vec::new();
    let mut i = 0;
    while i < n {
        let start_pos = i;
        let mut k1 = order_slice[start_pos] as usize;
        let mut k = i + 1;
        while k < n {
            let k2 = order_slice[k] as usize;
            if status[k1] != 0.0
                || id_slice[k1] != id_slice[k2]
                || x_slice[k1] != x_slice[k2]
                || (time1[k1] - time2[k2]).abs() > 1e-9
                || istate_slice[k1] != istate_slice[k2]
                || (wt_slice[k1] - wt_slice[k2]).abs() > 1e-9
            {
                break;
            }
            k1 = k2;
            i += 1;
            k += 1;
        }
        i1.push((k1 + 1) as i32);
        i2.push((order_slice[start_pos] as usize + 1) as i32);
        i += 1;
    }
    let mut matrix = Vec::new();
    for (start, end) in i2.iter().zip(i1.iter()) {
        matrix.push(vec![*start, *end]);
    }
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("matrix", matrix)?;
        dict.set_item("dimnames", vec!["start", "end"])?;
        Ok(dict.into())
    })
}
