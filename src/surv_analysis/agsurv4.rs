//! `agsurv4`: the R-named alias of `compute_baseline_survival_steps`
//! (`src/agsurv4.c`, the Kalbfleisch-Prentice increments).

use super::cox_baseline::compute_baseline_survival_steps;
use pyo3::prelude::*;

#[pyfunction]
pub fn agsurv4(
    ndeath: Vec<i32>,
    risk: Vec<f64>,
    wt: Vec<f64>,
    sn: usize,
    denom: Vec<f64>,
) -> PyResult<Vec<f64>> {
    compute_baseline_survival_steps(ndeath, risk, wt, sn, denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agsurv4_matches_the_kalbfleisch_prentice_increments() {
        let ndeath = vec![1, 2, 0];
        let risk = vec![1.0, 1.0, 1.0];
        let wt = vec![0.2, 0.3, 0.4];
        let denom = vec![5.0, 4.0, 3.0];

        let alias = agsurv4(ndeath.clone(), risk.clone(), wt.clone(), 3, denom.clone()).unwrap();
        let descriptive = compute_baseline_survival_steps(ndeath, risk, wt, 3, denom).unwrap();

        assert_eq!(alias, descriptive);
        assert!((alias[0] - (1.0 - 0.2 / 5.0)).abs() < 1e-12);
        assert_eq!(alias[2], 1.0);
        // Two tied deaths solve sum w r / (1 - s^r) = denom by bisection.
        let s = alias[1];
        assert!(((0.3 + 0.4) / (1.0 - s) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn agsurv4_rejects_malformed_inputs_without_panicking() {
        let err = agsurv4(vec![1], vec![], vec![1.0], 1, vec![2.0])
            .expect_err("short risk vector should fail");

        assert!(err.to_string().contains("risk length must be at least 1"));
    }
}
