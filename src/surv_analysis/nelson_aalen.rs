//! The Nelson-Aalen cumulative hazard as a stand-alone estimate: the
//! `cumhaz` / `std.chaz` components of `survfit(..., ctype = 1)` restricted
//! to the event times, with a plain normal band.  Everything is computed by
//! [`survfitkm`](super::survfitkm::survfitkm).

use super::survfit_confint::{ConfType, survfit_confint};
use super::survfitkm::{HazardType, SurvfitKMData, SurvfitKMOptions, survfitkm};
use crate::error::SurvivalResult;
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// Nelson-Aalen estimate at the distinct event times.
///
/// `n_risk` and `n_events` are unweighted counts; `cumulative_hazard` and
/// `variance` (`std.chaz^2`, the Poisson variance `sum d / n^2`) use the
/// case weights when given.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct NelsonAalenResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_hazard: Vec<f64>,
    #[pyo3(get)]
    pub variance: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
}

#[pymethods]
impl NelsonAalenResult {
    /// `exp(-cumulative_hazard)`, the Fleming-Harrington survival estimate.
    pub fn survival(&self) -> Vec<f64> {
        self.cumulative_hazard.iter().map(|&h| (-h).exp()).collect()
    }
}

/// Nelson-Aalen cumulative hazard with a `confidence_level` plain band
/// (`H +- z * se`, lower limit floored at 0).  An empty input gives empty
/// vectors.
pub fn nelson_aalen(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    confidence_level: f64,
) -> SurvivalResult<NelsonAalenResult> {
    validate_length(time.len(), status.len(), "status")?;
    if time.is_empty() {
        return Ok(NelsonAalenResult {
            time: Vec::new(),
            cumulative_hazard: Vec::new(),
            variance: Vec::new(),
            ci_lower: Vec::new(),
            ci_upper: Vec::new(),
            n_risk: Vec::new(),
            n_events: Vec::new(),
        });
    }
    let data = SurvfitKMData::try_new(
        None,
        time.to_vec(),
        status.to_vec(),
        weights.map(<[f64]>::to_vec),
        None,
        None,
        None,
    )?;
    let options = SurvfitKMOptions {
        ctype: HazardType::NelsonAalen,
        conf_int: confidence_level,
        conf_type: ConfType::None,
        // the Poisson variance whatever the weights, as a plain estimate
        robust: Some(false),
        ..Default::default()
    };
    let fit = survfitkm(&data, &options)?;
    let std_chaz = fit.std_chaz.as_deref().expect("se.fit is on");
    let unweighted = |weighted: &[f64], counts: Option<&Vec<f64>>| -> Vec<usize> {
        counts
            .map_or(weighted, Vec::as_slice)
            .iter()
            .map(|&value| value as usize)
            .collect()
    };
    let n_risk_all = unweighted(&fit.n_risk, fit.counts.as_ref().map(|c| &c.n_risk));
    let n_event_all = unweighted(&fit.n_event, fit.counts.as_ref().map(|c| &c.n_event));
    let rows: Vec<usize> = (0..fit.time.len())
        .filter(|&i| n_event_all[i] > 0)
        .collect();
    let cumulative_hazard: Vec<f64> = rows.iter().map(|&i| fit.cumhaz[i]).collect();
    let std_err: Vec<f64> = rows.iter().map(|&i| std_chaz[i]).collect();
    let bands = survfit_confint(
        &cumulative_hazard,
        &std_err,
        false,
        ConfType::Plain,
        confidence_level,
        None,
        false,
    )?;
    Ok(NelsonAalenResult {
        time: rows.iter().map(|&i| fit.time[i]).collect(),
        cumulative_hazard,
        variance: std_err.iter().map(|se| se * se).collect(),
        ci_lower: bands.lower,
        ci_upper: bands.upper,
        n_risk: rows.iter().map(|&i| n_risk_all[i]).collect(),
        n_events: rows.iter().map(|&i| n_event_all[i]).collect(),
    })
}

/// Python binding of [`nelson_aalen`].
#[pyfunction(name = "nelson_aalen")]
#[pyo3(signature = (time, status, weights=None, confidence_level=0.95))]
pub fn nelson_aalen_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    weights: Option<FloatVec>,
    confidence_level: f64,
) -> PyResult<NelsonAalenResult> {
    Ok(py.detach(|| nelson_aalen(&time, &status, weights.as_deref(), confidence_level))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn empty_and_all_censored_inputs_give_empty_curves() {
        let result = nelson_aalen(&[], &[], None, 0.95).unwrap();
        assert!(result.time.is_empty());
        let result = nelson_aalen(&[1.0, 2.0, 3.0], &[0, 0, 0], None, 0.95).unwrap();
        assert!(result.time.is_empty());
        assert!(result.cumulative_hazard.is_empty());
    }

    #[test]
    fn matches_the_hand_computed_estimate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0];
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.n_risk, vec![6, 4, 2]);
        assert_eq!(result.n_events, vec![1, 1, 1]);
        assert_vec_close(
            &result.cumulative_hazard,
            &[
                1.0 / 6.0,
                1.0 / 6.0 + 1.0 / 4.0,
                1.0 / 6.0 + 1.0 / 4.0 + 1.0 / 2.0,
            ],
        );
        assert_vec_close(
            &result.variance,
            &[
                1.0 / 36.0,
                1.0 / 36.0 + 1.0 / 16.0,
                1.0 / 36.0 + 1.0 / 16.0 + 1.0 / 4.0,
            ],
        );
        for i in 0..3 {
            assert!(result.ci_lower[i] <= result.cumulative_hazard[i]);
            assert!(result.cumulative_hazard[i] <= result.ci_upper[i]);
            assert!(result.ci_lower[i] >= 0.0);
        }
        assert_vec_close(
            &result.survival(),
            &[
                (-1.0_f64 / 6.0).exp(),
                (-5.0_f64 / 12.0).exp(),
                (-11.0_f64 / 12.0).exp(),
            ],
        );
    }

    #[test]
    fn weights_scale_the_hazard_but_not_the_counts() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let weighted = nelson_aalen(&time, &status, Some(&[2.0, 1.0, 1.0]), 0.95).unwrap();
        assert_eq!(weighted.n_risk, vec![3, 2, 1]);
        assert!((weighted.cumulative_hazard[0] - 0.5).abs() < 1e-12);
        let unit = nelson_aalen(&time, &status, Some(&[1.0; 3]), 0.95).unwrap();
        let plain = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(unit, plain);
    }

    #[test]
    fn near_tied_event_times_are_grouped() {
        let time = vec![1.0 + 5e-10, 2.0, 1.0];
        let status = vec![1, 0, 1];
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time, vec![1.0]);
        assert_eq!(result.n_risk, vec![3]);
        assert_eq!(result.n_events, vec![2]);
        assert!((result.cumulative_hazard[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((result.variance[0] - 2.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 2], None, 0.95).is_err());
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 0], Some(&[1.0, f64::INFINITY]), 0.95).is_err());
        assert!(nelson_aalen(&[1.0, 2.0], &[1, 0], None, 1.0).is_err());
        assert!(nelson_aalen(&[1.0], &[1, 0], None, 0.95).is_err());
    }
}
