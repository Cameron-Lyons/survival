use crate::internal::dist;
use pyo3::prelude::*;

#[doc = " `pchisq(q, df, lower.tail, log.p)` for one quantile.  As in R, a missing"]
#[doc = " argument or a negative `df` gives `NaN`, `df = 0` is the point mass at 0"]
#[doc = " and `q <= 0` has lower tail 0."]
#[pyfunction(name = "pchisq")]
#[pyo3(signature = (q, df, lower_tail=true, log_p=false))]
pub(crate) fn pchisq_py(q: f64, df: f64, lower_tail: bool, log_p: bool) -> f64 {
    dist::pchisq(q, df, lower_tail, log_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-14 * expected.abs(),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn fractional_df_matches_r() {
        assert_close(
            pchisq_py(3.6432229393649322, 3.08864312775353, false, false),
            0.31611266250077846,
        );
        assert_close(
            pchisq_py(2.9406959209324954, 3.092186074265781, false, true),
            -0.8750484229716111,
        );
        assert_close(pchisq_py(0.5, 0.25, true, false), 0.8696646545502863);
    }

    #[test]
    fn keeps_r_edge_cases() {
        assert!(pchisq_py(f64::NAN, 1.0, false, false).is_nan());
        assert!(pchisq_py(1.0, f64::NAN, false, false).is_nan());
        assert!(pchisq_py(4.05, -6.9, false, false).is_nan());
        assert_eq!(pchisq_py(1.0, 0.0, false, false), 0.0);
        assert_eq!(pchisq_py(0.0, 0.0, false, false), 1.0);
        assert_eq!(pchisq_py(-1.0, 2.0, true, false), 0.0);
        assert_eq!(pchisq_py(0.0, 2.0, false, true), 0.0);
        assert_eq!(pchisq_py(f64::INFINITY, 2.0, false, false), 0.0);
    }
}
