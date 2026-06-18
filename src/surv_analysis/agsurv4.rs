use super::cox_baseline::compute_baseline_survival_steps;
use pyo3::prelude::*;

/// Backward-compatible aggregate survival helper kept for the public crate API.
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
    use crate::surv_analysis::compute_baseline_survival_steps;

    #[test]
    fn agsurv4_matches_validated_baseline_steps() {
        let ndeath = vec![1, 2, 0];
        let risk = vec![1.0, 1.0, 1.0];
        let wt = vec![0.2, 0.3, 0.4];
        let denom = vec![5.0, 4.0, 3.0];

        let alias = agsurv4(ndeath.clone(), risk.clone(), wt.clone(), 3, denom.clone()).unwrap();
        let descriptive = compute_baseline_survival_steps(ndeath, risk, wt, 3, denom).unwrap();

        assert_eq!(alias, descriptive);
    }

    #[test]
    fn agsurv4_rejects_malformed_inputs_without_panicking() {
        let err = agsurv4(vec![1], vec![], vec![1.0], 1, vec![2.0])
            .expect_err("short risk vector should fail");

        assert!(err.to_string().contains("risk length must be at least 1"));
    }
}
