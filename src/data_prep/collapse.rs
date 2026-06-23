use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn length_error(field: &str, actual: usize, expected: usize) -> PyErr {
    PyValueError::new_err(format!(
        "{field} length mismatch: got {actual}, expected {expected}"
    ))
}

fn validate_finite(values: &[f64], field: &str) -> PyResult<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{field} values must be finite, got non-finite value at index {index}"
            )));
        }
    }
    Ok(())
}

fn validate_order(order: &[i32], n: usize) -> PyResult<()> {
    let mut seen = vec![false; n];
    for (position, &raw_index) in order.iter().enumerate() {
        if raw_index < 0 {
            return Err(PyValueError::new_err(format!(
                "order values must be non-negative, got {raw_index} at position {position}"
            )));
        }

        let index = raw_index as usize;
        if index >= n {
            return Err(PyValueError::new_err(format!(
                "order values must be less than {n}, got {raw_index} at position {position}"
            )));
        }
        if seen[index] {
            return Err(PyValueError::new_err(format!(
                "order must be a permutation; duplicate index {raw_index} at position {position}"
            )));
        }
        seen[index] = true;
    }
    Ok(())
}

struct CollapseSlices<'a> {
    time1: &'a [f64],
    time2: &'a [f64],
    status: &'a [f64],
    x: &'a [i32],
    istate: &'a [i32],
    id: &'a [i32],
    wt: &'a [f64],
    order: &'a [i32],
}

fn collapse_matrix(input: CollapseSlices<'_>) -> Vec<Vec<i32>> {
    let n = input.id.len();
    let mut matrix = Vec::new();
    let mut i = 0;
    while i < n {
        let start_pos = i;
        let mut k1 = input.order[start_pos] as usize;
        let mut k = i + 1;
        while k < n {
            let k2 = input.order[k] as usize;
            if input.status[k1] != 0.0
                || input.id[k1] != input.id[k2]
                || input.x[k1] != input.x[k2]
                || (input.time2[k1] - input.time1[k2]).abs() > 1e-9
                || input.istate[k1] != input.istate[k2]
                || (input.wt[k1] - input.wt[k2]).abs() > 1e-9
            {
                break;
            }
            k1 = k2;
            i += 1;
            k += 1;
        }
        matrix.push(vec![
            (input.order[start_pos] as usize + 1) as i32,
            (k1 + 1) as i32,
        ]);
        i += 1;
    }
    matrix
}

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
    if y_slice.len() != 3 * n {
        return Err(PyValueError::new_err(format!(
            "y must have 3 columns and length 3 * len(id), got y length {} and id length {}",
            y_slice.len(),
            n
        )));
    }
    if x_slice.len() != n {
        return Err(length_error("x", x_slice.len(), n));
    }
    if istate_slice.len() != n {
        return Err(length_error("istate", istate_slice.len(), n));
    }
    if wt_slice.len() != n {
        return Err(length_error("wt", wt_slice.len(), n));
    }
    if order_slice.len() != n {
        return Err(length_error("order", order_slice.len(), n));
    }
    validate_finite(y_slice, "y")?;
    validate_finite(wt_slice, "wt")?;
    validate_order(order_slice, n)?;

    let time1 = &y_slice[0..n];
    let time2 = &y_slice[n..2 * n];
    let status = &y_slice[2 * n..3 * n];
    let matrix = collapse_matrix(CollapseSlices {
        time1,
        time2,
        status,
        x: x_slice,
        istate: istate_slice,
        id: id_slice,
        wt: wt_slice,
        order: order_slice,
    });
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("matrix", matrix)?;
        dict.set_item("dimnames", vec!["start", "end"])?;
        Ok(dict.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    type CollapseInputs = (Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>);

    fn valid_inputs() -> CollapseInputs {
        (
            vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 0.0, 1.0, 0.0],
            vec![1, 1, 1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1, 2, 2],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![0, 1, 2, 3],
        )
    }

    #[test]
    fn collapse_rejects_length_mismatches_without_panicking() {
        initialize_python();
        let (y, x, istate, id, wt, order) = valid_inputs();

        let err = collapse(
            y[..11].to_vec(),
            x.clone(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            order.clone(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("y must have 3 columns"));

        let err = collapse(
            y.clone(),
            x[..3].to_vec(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            order.clone(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("x length mismatch"));

        let err = collapse(
            y.clone(),
            x.clone(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            order[..3].to_vec(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("order length mismatch"));
    }

    #[test]
    fn collapse_rejects_malformed_values_and_order() {
        initialize_python();
        let (y, x, istate, id, wt, order) = valid_inputs();

        let mut bad_y = y.clone();
        bad_y[0] = f64::NAN;
        let err = collapse(
            bad_y,
            x.clone(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            order.clone(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("y values must be finite"));

        let mut bad_wt = wt.clone();
        bad_wt[0] = f64::INFINITY;
        let err = collapse(
            y.clone(),
            x.clone(),
            istate.clone(),
            id.clone(),
            bad_wt,
            order.clone(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("wt values must be finite"));

        let err = collapse(
            y.clone(),
            x.clone(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            vec![-1, 1, 2, 3],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("order values must be non-negative")
        );

        let err = collapse(
            y.clone(),
            x.clone(),
            istate.clone(),
            id.clone(),
            wt.clone(),
            vec![0, 1, 2, 4],
        )
        .unwrap_err();
        assert!(err.to_string().contains("order values must be less than"));

        let err = collapse(y, x, istate, id, wt, vec![0, 1, 1, 3]).unwrap_err();
        assert!(err.to_string().contains("order must be a permutation"));
    }

    #[test]
    fn collapse_merges_adjacent_compatible_rows() {
        let matrix = collapse_matrix(CollapseSlices {
            time1: &[1.0, 2.0, 3.0, 4.0],
            time2: &[2.0, 3.0, 4.0, 5.0],
            status: &[0.0, 0.0, 1.0, 0.0],
            x: &[1, 1, 1, 1],
            istate: &[0, 0, 0, 0],
            id: &[1, 1, 2, 2],
            wt: &[1.0, 1.0, 1.0, 1.0],
            order: &[0, 1, 2, 3],
        });

        assert_eq!(matrix, vec![vec![1, 2], vec![3, 3], vec![4, 4]]);
    }
}
