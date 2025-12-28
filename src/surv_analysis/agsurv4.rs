use pyo3::prelude::*;
#[pyfunction]
pub fn agsurv4(
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
                let range = j..(j + death_count);
                for _ in 0..35 {
                    let mut sumt = 0.0;
                    for k in range.clone() {
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
