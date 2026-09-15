//! `quantile.survfit` (`R/quantile.survfit.R`) from stacked curve vectors.
//! The port itself is `surv_analysis::quantile_survfit`.

use super::{StackedCurves, stacked_curves};
use crate::surv_analysis::{SurvfitQuantiles, quantile_survfit_from};
use pyo3::prelude::*;

/// `quantile(fit, probs, conf.int)`: one row per curve, `NaN` for a
/// quantile the curve does not reach (R's `NA`).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvfitCurveQuantiles {
    pub probs: Vec<f64>,
    pub quantile: Vec<Vec<f64>>,
    pub lower: Option<Vec<Vec<f64>>>,
    pub upper: Option<Vec<Vec<f64>>>,
}

impl From<SurvfitQuantiles> for SurvfitCurveQuantiles {
    fn from(quantiles: SurvfitQuantiles) -> Self {
        Self {
            probs: quantiles.probs,
            quantile: quantiles.quantile,
            lower: quantiles.lower,
            upper: quantiles.upper,
        }
    }
}

/// Python entry point of `quantile.survfit` on the stacked vectors of one
/// `survfit` object (`strata` gives the rows of each curve, `None` for a
/// single curve).  `start_time` is R's `x$start.time` (0 when absent), the
/// time reported for a probability of 0; `probs` defaults to the
/// quartiles and `tolerance` to `sqrt(.Machine$double.eps)`.
#[pyfunction(name = "quantile_survfit_curves")]
#[pyo3(signature = (time, surv, lower=None, upper=None, strata=None, probs=None, conf_int=true, start_time=0.0, scale=1.0, tolerance=None))]
#[allow(clippy::too_many_arguments)]
pub fn quantile_survfit_curves_py(
    time: Vec<f64>,
    surv: Vec<f64>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    probs: Option<Vec<f64>>,
    conf_int: bool,
    start_time: f64,
    scale: f64,
    tolerance: Option<f64>,
) -> PyResult<SurvfitCurveQuantiles> {
    let probs = probs.unwrap_or_else(|| vec![0.25, 0.5, 0.75]);
    let zeros = vec![0.0; time.len()];
    let n_curves = strata.as_ref().map_or(1, Vec::len);
    let fit = stacked_curves(&StackedCurves {
        time: &time,
        surv: &surv,
        n_risk: &zeros,
        n_event: &zeros,
        lower: lower.as_deref(),
        upper: upper.as_deref(),
        strata: strata.as_deref(),
        n: &vec![0.0; n_curves],
        n_id: None,
        t0: start_time,
    })?;
    Ok(quantile_survfit_from(&fit, &probs, conf_int, start_time, scale, tolerance)?.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_curves_report_the_origin_for_a_zero_probability() {
        let time = [1.0, 2.0, 3.0];
        let surv = [0.6, 0.4, 0.2];
        let fit = stacked_curves(&StackedCurves {
            time: &time,
            surv: &surv,
            n_risk: &[0.0; 3],
            n_event: &[0.0; 3],
            lower: Some(&[0.3, 0.1, 0.05]),
            upper: Some(&[0.9, 0.7, 0.5]),
            strata: None,
            n: &[0.0],
            n_id: None,
            t0: 0.5,
        })
        .unwrap();
        let result: SurvfitCurveQuantiles =
            quantile_survfit_from(&fit, &[0.0, 0.5], true, 0.5, 1.0, None)
                .unwrap()
                .into();
        assert_eq!(result.quantile, vec![vec![0.5, 2.0]]);
        // the lower bound crosses 0.5 at t=1, the upper bound at t=3
        assert_eq!(result.lower.unwrap()[0][1], 1.0);
        assert_eq!(result.upper.unwrap()[0][1], 3.0);
    }
}
