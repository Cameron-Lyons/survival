//! `quantile.survfit` (`R/quantile.survfit.R`) from stacked curve vectors.
//! The port itself is `surv_analysis::quantile_survfit`.

use crate::data_types::FloatVec;
use crate::surv_analysis::{
    StackedCurves, SurvfitKMResult, SurvfitQuantiles, quantile_survfit_from,
};
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
    py: Python<'_>,
    time: FloatVec,
    surv: FloatVec,
    lower: Option<FloatVec>,
    upper: Option<FloatVec>,
    strata: Option<Vec<usize>>,
    probs: Option<FloatVec>,
    conf_int: bool,
    start_time: f64,
    scale: f64,
    tolerance: Option<f64>,
) -> PyResult<SurvfitCurveQuantiles> {
    let time = time.into_inner();
    let surv = surv.into_inner();
    let lower = lower.map(FloatVec::into_inner);
    let upper = upper.map(FloatVec::into_inner);
    let probs = probs.map_or_else(|| vec![0.25, 0.5, 0.75], FloatVec::into_inner);
    Ok(py.detach(|| {
        let zeros = vec![0.0; time.len()];
        let n_curves = strata.as_ref().map_or(1, Vec::len);
        let fit = SurvfitKMResult::from_stacked(StackedCurves {
            lower,
            upper,
            t0: start_time,
            ..StackedCurves::new(time, zeros.clone(), zeros, surv, strata, vec![0; n_curves])
        })?;
        quantile_survfit_from(&fit, &probs, conf_int, start_time, scale, tolerance)
            .map(SurvfitCurveQuantiles::from)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_quantiles_report_the_origin_for_a_zero_probability() {
        let time = [1.0, 2.0, 3.0];
        let surv = [0.6, 0.4, 0.2];
        let fit = SurvfitKMResult::from_stacked(StackedCurves {
            lower: Some(vec![0.3, 0.1, 0.05]),
            upper: Some(vec![0.9, 0.7, 0.5]),
            t0: 0.5,
            ..StackedCurves::new(
                time.to_vec(),
                vec![0.0; 3],
                vec![0.0; 3],
                surv.to_vec(),
                None,
                vec![0],
            )
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
