//! Confidence bands for survival-type curves: the port of `survfit_confint`
//! in R's `survfit.R`, used by every Kaplan-Meier, Nelson-Aalen and
//! Aalen-Johansen curve in this crate.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::qnorm;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// The `conf.type` argument of `survfit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfType {
    #[default]
    Log,
    LogLog,
    Plain,
    None,
    Logit,
    Arcsin,
}

impl ConfType {
    /// Parse R's spelling (`"log-log"`); `"loglog"` and `"log_log"` are
    /// accepted as well since Python callers cannot type the hyphen in a
    /// keyword.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "log" => Ok(Self::Log),
            "log-log" | "loglog" | "log_log" => Ok(Self::LogLog),
            "plain" => Ok(Self::Plain),
            "none" => Ok(Self::None),
            "logit" => Ok(Self::Logit),
            "arcsin" => Ok(Self::Arcsin),
            other => Err(SurvivalError::invalid_input(format!(
                "conf.type must be one of 'log', 'log-log', 'plain', 'none', 'logit', 'arcsin'; got {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::LogLog => "log-log",
            Self::Plain => "plain",
            Self::None => "none",
            Self::Logit => "logit",
            Self::Arcsin => "arcsin",
        }
    }
}

/// The `conf.lower` argument of `survfit`: how the lower band is widened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfLower {
    #[default]
    Usual,
    Peto,
    Modified,
}

impl ConfLower {
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "usual" => Ok(Self::Usual),
            "peto" => Ok(Self::Peto),
            "modified" => Ok(Self::Modified),
            other => Err(SurvivalError::invalid_input(format!(
                "conf.lower must be one of 'usual', 'peto', 'modified'; got {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Usual => "usual",
            Self::Peto => "peto",
            Self::Modified => "modified",
        }
    }
}

/// Lower and upper confidence limits, element for element with the curve
/// they were computed for.  `NaN` marks a limit R reports as `NA`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct ConfidenceBands {
    #[pyo3(get)]
    pub lower: Vec<f64>,
    #[pyo3(get)]
    pub upper: Vec<f64>,
}

/// Validate a confidence level the way `survfit_confint` does.
pub(crate) fn validate_conf_int(conf_int: f64) -> SurvivalResult<()> {
    if !(conf_int > 0.0 && conf_int < 1.0) {
        return Err(SurvivalError::invalid_input(
            "confidence intervals must be between 0 and 1",
        ));
    }
    Ok(())
}

/// R's `log()` on a value that may be `NA`: non-positive arguments give NaN.
fn r_log(value: f64) -> f64 {
    if value > 0.0 { value.ln() } else { f64::NAN }
}

/// `pmin(value, limit)` with R's NA propagation.
fn r_pmin(value: f64, limit: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else {
        value.min(limit)
    }
}

/// `pmax(value, limit)` with R's NA propagation.
fn r_pmax(value: f64, limit: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else {
        value.max(limit)
    }
}

/// Port of `survfit_confint` (R `survfit.R`).
///
/// `p` is the estimate, `se` its standard error: on the log scale when
/// `logse` is true (the simple Greenwood variance of a Kaplan-Meier curve),
/// otherwise on the scale of `p` (robust variances, multi-state curves).
/// `selow`, when given, widens only the lower limit (`conf.lower = "peto"` /
/// `"modified"`); `ulimit` caps the upper limit at 1 for the `plain` and
/// `log` transforms, which R turns off for cumulative hazards.
///
/// Edge rule, as in R: when `se == 0` both limits equal `p`; otherwise a
/// transform that cannot be evaluated at `p` (`log(0)`, `log(-log(1))`) gives
/// `NA`, so a curve that reaches 0 has `NA` limits there rather than limits
/// that dive to zero.
pub fn survfit_confint(
    p: &[f64],
    se: &[f64],
    logse: bool,
    conf_type: ConfType,
    conf_int: f64,
    selow: Option<&[f64]>,
    ulimit: bool,
) -> SurvivalResult<ConfidenceBands> {
    validate_conf_int(conf_int)?;
    validate_length(p.len(), se.len(), "se")?;
    if let Some(selow) = selow {
        validate_length(p.len(), selow.len(), "selow")?;
    }
    let zval = qnorm(1.0 - (1.0 - conf_int) / 2.0, true, false);
    let n = p.len();
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for i in 0..n {
        let p_i = p[i];
        // scale = ifelse(selow == 0, 1, selow / se); avoids 0/0 at the origin
        let scale = match selow {
            Some(selow) if selow[i] != 0.0 => selow[i] / se[i],
            _ => 1.0,
        };
        // se of log(survival) when the caller supplied se(S)
        let se_i = if logse {
            se[i]
        } else if se[i] == 0.0 {
            0.0
        } else {
            se[i] / p_i
        };
        let (lo, hi) = match conf_type {
            ConfType::Plain => {
                // equation 4.3.1 in Klein & Moeschberger
                let se2 = se_i * p_i * zval;
                let hi = p_i + se2;
                (
                    r_pmax(p_i - se2 * scale, 0.0),
                    if ulimit { r_pmin(hi, 1.0) } else { hi },
                )
            }
            ConfType::Log => {
                let xx = if p_i == 0.0 { f64::NAN } else { p_i };
                let se2 = zval * se_i;
                let lo = if se_i == 0.0 {
                    p_i
                } else {
                    (r_log(xx) - se2 * scale).exp()
                };
                let hi = if se_i == 0.0 {
                    p_i
                } else {
                    (r_log(xx) + se2).exp()
                };
                (lo, if ulimit { r_pmin(hi, 1.0) } else { hi })
            }
            ConfType::LogLog => {
                let xx = if p_i == 0.0 || p_i == 1.0 {
                    f64::NAN
                } else {
                    p_i
                };
                let se2 = zval * se_i / r_log(xx);
                let base = r_log(-r_log(xx));
                let lo = if se_i == 0.0 {
                    p_i
                } else {
                    (-(base - se2 * scale).exp()).exp()
                };
                let hi = if se_i == 0.0 {
                    p_i
                } else {
                    (-(base + se2).exp()).exp()
                };
                (lo, hi)
            }
            ConfType::Logit => {
                let xx = if p_i == 0.0 { f64::NAN } else { p_i };
                let se2 = zval * se_i * (1.0 + xx / (1.0 - xx));
                let logit = r_log(p_i / (1.0 - p_i));
                let lo = if se_i == 0.0 {
                    p_i
                } else {
                    1.0 - 1.0 / (1.0 + (logit - se2 * scale).exp())
                };
                let hi = if se_i == 0.0 {
                    p_i
                } else {
                    1.0 - 1.0 / (1.0 + (logit + se2).exp())
                };
                (lo, hi)
            }
            ConfType::Arcsin => {
                let xx = if p_i == 0.0 { f64::NAN } else { p_i };
                let se2 = 0.5 * zval * se_i * (xx / (1.0 - xx)).sqrt();
                let angle = xx.sqrt().asin();
                (
                    r_pmax(angle - se2 * scale, 0.0).sin().powi(2),
                    r_pmin(angle + se2, std::f64::consts::FRAC_PI_2)
                        .sin()
                        .powi(2),
                )
            }
            ConfType::None => {
                return Err(SurvivalError::invalid_input("invalid conf.int type"));
            }
        };
        lower.push(lo);
        upper.push(hi);
    }
    Ok(ConfidenceBands { lower, upper })
}

/// Python binding of [`survfit_confint`].
#[pyfunction(name = "survfit_confint")]
#[pyo3(signature = (p, se, logse=true, conf_type="log", conf_int=0.95, selow=None, ulimit=true))]
pub fn survfit_confint_py(
    p: Vec<f64>,
    se: Vec<f64>,
    logse: bool,
    conf_type: &str,
    conf_int: f64,
    selow: Option<Vec<f64>>,
    ulimit: bool,
) -> PyResult<ConfidenceBands> {
    let conf_type = ConfType::parse(conf_type)?;
    Ok(survfit_confint(
        &p,
        &se,
        logse,
        conf_type,
        conf_int,
        selow.as_deref(),
        ulimit,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&left, &right) in actual.iter().zip(expected) {
            if right.is_nan() {
                assert!(left.is_nan(), "{left} should be NaN");
            } else {
                assert!((left - right).abs() < 1e-7, "{left} != {right}");
            }
        }
    }

    fn bands(p: &[f64], se: &[f64], conf_type: ConfType) -> ConfidenceBands {
        survfit_confint(p, se, true, conf_type, 0.95, None, true).unwrap()
    }

    #[test]
    fn matches_r_confidence_transforms() {
        // survfit_confint(c(.2, .5, .9), .1, conf.type=...) in R
        let p = [0.2, 0.5, 0.9];
        let se = [0.1; 3];
        let plain = bands(&p, &se, ConfType::Plain);
        assert_close(&plain.lower, &[0.16080072, 0.4020018, 0.7236032]);
        assert_close(&plain.upper, &[0.23919928, 0.5979982, 1.0]);
        let arcsin = bands(&p, &se, ConfType::Arcsin);
        assert_close(&arcsin.lower, &[0.1623028, 0.4026280, 0.6664164]);
        assert_close(&arcsin.upper, &[0.2405760, 0.5973720, 0.9992298]);
        for (kind, lower, upper) in [
            (ConfType::Log, 0.164403, 0.2433045),
            (ConfType::LogLog, 0.1623716, 0.2405312),
            (ConfType::Logit, 0.1636537, 0.242082),
        ] {
            let out = bands(&p[..1], &se[..1], kind);
            assert_close(&out.lower, &[lower]);
            assert_close(&out.upper, &[upper]);
        }
    }

    #[test]
    fn edge_rules_follow_r() {
        // se == 0 -> both limits equal p, even at p = 0
        for kind in [ConfType::Log, ConfType::LogLog, ConfType::Logit] {
            let out = bands(&[0.0], &[0.0], kind);
            assert_close(&out.lower, &[0.0]);
            assert_close(&out.upper, &[0.0]);
        }
        // p == 0 with se > 0 -> NA for the transforms that need log(p)
        let out = bands(&[0.0, 1.0], &[0.1, 0.1], ConfType::LogLog);
        assert!(out.lower.iter().all(|v| v.is_nan()));
        assert!(out.upper.iter().all(|v| v.is_nan()));
        // logse = FALSE rescales se(S) to se(log S)
        let out = survfit_confint(
            &[0.2, 0.5],
            &[0.1, 0.1],
            false,
            ConfType::Plain,
            0.95,
            None,
            true,
        )
        .unwrap();
        assert_close(&out.lower, &[0.004003602, 0.304003602]);
        assert_close(&out.upper, &[0.3959964, 0.695996398]);
        // no upper cap for cumulative hazards
        let out = survfit_confint(&[0.9], &[0.5], true, ConfType::Log, 0.95, None, false).unwrap();
        assert!(out.upper[0] > 1.0);
    }

    #[test]
    fn selow_widens_only_the_lower_limit() {
        let plain = bands(&[0.5], &[0.1], ConfType::Log);
        let out = survfit_confint(
            &[0.5],
            &[0.1],
            true,
            ConfType::Log,
            0.95,
            Some(&[0.2]),
            true,
        )
        .unwrap();
        assert!(out.lower[0] < plain.lower[0]);
        assert_close(&out.upper, &plain.upper);
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(ConfType::parse("bad").is_err());
        assert!(ConfLower::parse("bad").is_err());
        assert_eq!(ConfType::parse("Log-Log").unwrap(), ConfType::LogLog);
        assert!(survfit_confint(&[0.5], &[0.1], true, ConfType::Log, 1.0, None, true).is_err());
        assert!(survfit_confint(&[0.5], &[0.1], true, ConfType::None, 0.95, None, true).is_err());
        assert!(
            survfit_confint(&[0.5], &[0.1, 0.2], true, ConfType::Log, 0.95, None, true).is_err()
        );
    }
}
