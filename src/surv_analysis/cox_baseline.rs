use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn compute_baseline_survival_steps(
    ndeath: Vec<i32>,
    risk: Vec<f64>,
    wt: Vec<f64>,
    sn: usize,
    denom: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let ndeath_slice = &ndeath;
    let risk_slice = &risk;
    let wt_slice = &wt;
    let denom_slice = &denom;
    let mut km = vec![0.0; sn];
    let n = sn;
    let mut j = 0;
    for i in 0..n {
        match ndeath_slice[i] {
            0 => km[i] = 1.0,
            1 => {
                let numerator = wt_slice[j] * risk_slice[j];
                km[i] = (1.0 - numerator / denom_slice[i]).powf(1.0 / risk_slice[j]);
                j += 1;
            }
            _ => {
                let mut guess: f64 = 0.5;
                let mut inc = 0.25;
                let death_count = ndeath_slice[i] as usize;
                for _ in 0..35 {
                    let mut sumt = 0.0;
                    for k in j..(j + death_count) {
                        let term = wt_slice[k] * risk_slice[k] / (1.0 - guess.powf(risk_slice[k]));
                        sumt += term;
                    }
                    if sumt < denom_slice[i] {
                        guess += inc;
                    } else {
                        guess -= inc;
                    }
                    inc /= 2.0;
                }
                km[i] = guess;
                j += death_count;
            }
        }
    }
    Ok(km)
}

#[pyfunction]
pub fn compute_tied_baseline_summaries(
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
