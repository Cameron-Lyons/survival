use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
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
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if extend[i] && !tstart[i].is_nan() && !tstop[i].is_nan() {
            let j_initial = {
                let mut j = 0;
                while j < ncut && ctime[j] < tstop[i] {
                    j += 1;
                }
                j
            };
            let j_start = j_initial + 1;
            for j in j_start..ncut {
                if keep[j] {
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

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let original_start = tstart[i];
        let original_end = tstop[i];
        let is_valid = !original_start.is_nan() && !original_end.is_nan();
        let is_extended = extend[i] && is_valid;

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
