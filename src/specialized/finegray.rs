use pyo3::prelude::*;
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct FineGrayOutput {
    #[pyo3(get)]
    pub row: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub end: Vec<f64>,
    #[pyo3(get)]
    pub wt: Vec<f64>,
    #[pyo3(get)]
    pub add: Vec<usize>,
}
#[pyfunction]
pub fn finegray(
    tstart: Vec<f64>,
    tstop: Vec<f64>,
    ctime: Vec<f64>,
    cprob: Vec<f64>,
    extend: Vec<bool>,
    keep: Vec<bool>,
) -> FineGrayOutput {
    compute_finegray(&tstart, &tstop, &ctime, &cprob, &extend, &keep)
}
pub(crate) fn compute_finegray(
    tstart: &[f64],
    tstop: &[f64],
    ctime: &[f64],
    cprob: &[f64],
    extend: &[bool],
    keep: &[bool],
) -> FineGrayOutput {
    let n = tstart.len();
    assert_eq!(tstop.len(), n);
    assert_eq!(extend.len(), n);
    let ncut = ctime.len();
    assert_eq!(cprob.len(), ncut);
    assert_eq!(keep.len(), ncut);
    let mut extra = 0;
    for (&ext, (&ts, &te)) in extend.iter().zip(tstart.iter().zip(tstop.iter())) {
        if ext && !ts.is_nan() && !te.is_nan() {
            let j_initial = {
                let mut j = 0;
                while j < ncut && ctime[j] < te {
                    j += 1;
                }
                j
            };
            let j_start = j_initial + 1;
            for &k in keep.iter().take(ncut).skip(j_start) {
                if k {
                    extra += 1;
                }
            }
        }
    }
    let total = n + extra;
    let mut row = Vec::with_capacity(total);
    let mut start = Vec::with_capacity(total);
    let mut end = Vec::with_capacity(total);
    let mut wt = Vec::with_capacity(total);
    let mut add = Vec::with_capacity(total);
    for (i, ((&original_start, &original_end), &ext)) in tstart
        .iter()
        .zip(tstop.iter())
        .zip(extend.iter())
        .enumerate()
    {
        let is_valid = !original_start.is_nan() && !original_end.is_nan();
        let is_extended = ext && is_valid;
        let (current_end, temp_wt, j_initial) = if is_extended {
            let mut j = 0;
            while j < ncut && ctime[j] < original_end {
                j += 1;
            }
            if j < ncut {
                (ctime[j], cprob[j], j)
            } else {
                (original_end, 1.0, ncut)
            }
        } else {
            (original_end, 1.0, ncut)
        };
        row.push(i + 1);
        start.push(original_start);
        end.push(current_end);
        wt.push(1.0);
        add.push(0);
        if is_extended && j_initial < ncut {
            let mut iadd = 0;
            for j in (j_initial + 1)..ncut {
                if keep[j] {
                    iadd += 1;
                    row.push(i + 1);
                    start.push(ctime[j - 1]);
                    end.push(ctime[j]);
                    wt.push(cprob[j] / temp_wt);
                    add.push(iadd);
                }
            }
        }
    }
    FineGrayOutput {
        row,
        start,
        end,
        wt,
        add,
    }
}
#[pymodule]
#[pyo3(name = "finegray")]
fn finegray_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(finegray, &m)?)?;
    m.add_class::<FineGrayOutput>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_finegray_matches_legacy_extension_weights() {
        let ctime = vec![3.0, 4.0, 6.0, 8.0, 9.0];
        let cprob = vec![
            11.0 / 12.0,
            (11.0 / 12.0) * (8.0 / 10.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0),
            (11.0 / 12.0) * (8.0 / 10.0) * (5.0 / 6.0) * (3.0 / 4.0) * (2.0 / 3.0),
        ];

        let result = compute_finegray(
            &[0.0, 0.0],
            &[4.0, 2.0],
            &ctime,
            &cprob,
            &[true, false],
            &[true, true, true, true, true],
        );

        assert_eq!(result.row, vec![1, 1, 1, 1, 2]);
        assert_eq!(result.start, vec![0.0, 4.0, 6.0, 8.0, 0.0]);
        assert_eq!(result.end, vec![4.0, 6.0, 8.0, 9.0, 2.0]);
        assert_eq!(result.add, vec![0, 1, 2, 3, 0]);

        let expected_wt = [1.0, 5.0 / 6.0, 5.0 / 8.0, 5.0 / 12.0, 1.0];
        assert_eq!(result.wt.len(), expected_wt.len());
        for (&actual, &expected) in result.wt.iter().zip(expected_wt.iter()) {
            assert_close(actual, expected, 1e-12);
        }
    }
}
