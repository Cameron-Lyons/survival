use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const EXP_CLAMP_MIN: f64 = -745.0;
const EXP_CLAMP_MAX: f64 = 709.0;

#[derive(Clone, Copy)]
enum ConfidenceType {
    Plain,
    Log,
    LogLog,
    Logit,
    Arcsin,
}

impl ConfidenceType {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "plain" => Ok(Self::Plain),
            "log" => Ok(Self::Log),
            "log-log" => Ok(Self::LogLog),
            "logit" => Ok(Self::Logit),
            "arcsin" => Ok(Self::Arcsin),
            _ => Err(PyErr::new::<PyValueError, _>("invalid conf.int type")),
        }
    }
}

fn safe_exp(value: f64) -> f64 {
    value.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp()
}

fn r_log(value: f64) -> f64 {
    if value.is_nan() || value <= 0.0 {
        f64::NAN
    } else {
        value.ln()
    }
}

fn r_sqrt(value: f64) -> f64 {
    if value.is_nan() || value < 0.0 {
        f64::NAN
    } else {
        value.sqrt()
    }
}

fn r_asin(value: f64) -> f64 {
    if value.is_nan() || !(-1.0..=1.0).contains(&value) {
        f64::NAN
    } else {
        value.asin()
    }
}

fn r_pmax_zero(value: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else {
        value.max(0.0)
    }
}

fn r_pmin(value: f64, limit: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else {
        value.min(limit)
    }
}

fn prepared_se_at(p: &[f64], se: &[f64], logse: bool, index: usize) -> f64 {
    let vector_index = index % se.len();
    let value = se[vector_index];
    if logse {
        value
    } else if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        0.0
    } else {
        value / p[vector_index % p.len()]
    }
}

fn scale_at(selow: Option<&[f64]>, se: &[f64], index: usize) -> f64 {
    let Some(values) = selow else {
        return 1.0;
    };
    let vector_index = index % values.len();
    let value = values[vector_index];
    if value.is_nan() {
        f64::NAN
    } else if value == 0.0 {
        1.0
    } else {
        value / se[vector_index % se.len()]
    }
}

fn plain_intervals(
    p: &[f64],
    se: &[f64],
    logse: bool,
    z: f64,
    selow: Option<&[f64]>,
    ulimit: bool,
) -> (Vec<f64>, Vec<f64>) {
    let base_len = p.len().max(se.len());
    let scale_len = selow.map_or(1, <[f64]>::len);
    let lower_len = if scale_len == 0 {
        0
    } else {
        base_len.max(scale_len)
    };

    let lower = (0..lower_len)
        .map(|index| {
            let base_index = index % base_len;
            let se2 = prepared_se_at(p, se, logse, base_index) * p[base_index % p.len()] * z;
            let scale = scale_at(selow, se, index % scale_len);
            r_pmax_zero(p[index % p.len()] - se2 * scale)
        })
        .collect();
    let upper = (0..base_len)
        .map(|index| {
            let se2 = prepared_se_at(p, se, logse, index) * p[index % p.len()] * z;
            let value = p[index % p.len()] + se2;
            if ulimit { r_pmin(value, 1.0) } else { value }
        })
        .collect();
    (lower, upper)
}

fn log_xx(value: f64, exclude_one: bool) -> f64 {
    if value.is_nan() || value == 0.0 || (exclude_one && value == 1.0) {
        f64::NAN
    } else {
        r_log(value)
    }
}

fn log_intervals(
    p: &[f64],
    se: &[f64],
    logse: bool,
    z: f64,
    selow: Option<&[f64]>,
    ulimit: bool,
) -> (Vec<f64>, Vec<f64>) {
    let scale_len = selow.map_or(1, <[f64]>::len);
    let lower = if scale_len == 0 {
        Vec::new()
    } else {
        (0..se.len())
            .map(|index| {
                let prepared = prepared_se_at(p, se, logse, index);
                if prepared.is_nan() {
                    return f64::NAN;
                }
                if prepared == 0.0 {
                    return p[index % p.len()];
                }
                let log_p = log_xx(p[index % p.len()], false);
                let scale = scale_at(selow, se, index % scale_len);
                safe_exp(log_p - z * prepared * scale)
            })
            .collect()
    };
    let upper = (0..se.len())
        .map(|index| {
            let prepared = prepared_se_at(p, se, logse, index);
            if prepared.is_nan() {
                return f64::NAN;
            }
            if prepared == 0.0 {
                return p[index % p.len()];
            }
            let value = safe_exp(log_xx(p[index % p.len()], false) + z * prepared);
            if ulimit { r_pmin(value, 1.0) } else { value }
        })
        .collect();
    (lower, upper)
}

fn log_log_intervals(
    p: &[f64],
    se: &[f64],
    logse: bool,
    z: f64,
    selow: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let scale_len = selow.map_or(1, <[f64]>::len);
    let calculate = |index: usize, lower: bool| {
        let prepared = prepared_se_at(p, se, logse, index);
        if prepared.is_nan() {
            return f64::NAN;
        }
        if prepared == 0.0 {
            return p[index % p.len()];
        }
        let log_p = log_xx(p[index % p.len()], true);
        let se2 = z * prepared / log_p;
        let transformed = r_log(-log_p);
        let adjusted = if lower {
            transformed - se2 * scale_at(selow, se, index % scale_len)
        } else {
            transformed + se2
        };
        safe_exp(-safe_exp(adjusted))
    };
    let lower = if scale_len == 0 {
        Vec::new()
    } else {
        (0..se.len()).map(|index| calculate(index, true)).collect()
    };
    let upper = (0..se.len()).map(|index| calculate(index, false)).collect();
    (lower, upper)
}

fn logit_intervals(
    p: &[f64],
    se: &[f64],
    logse: bool,
    z: f64,
    selow: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let scale_len = selow.map_or(1, <[f64]>::len);
    let calculate = |index: usize, lower: bool| {
        let prepared = prepared_se_at(p, se, logse, index);
        if prepared.is_nan() {
            return f64::NAN;
        }
        if prepared == 0.0 {
            return p[index % p.len()];
        }
        let probability = p[index % p.len()];
        let xx = if probability == 0.0 {
            f64::NAN
        } else {
            probability
        };
        let se2 = z * prepared * (1.0 + xx / (1.0 - xx));
        let logit = r_log(probability / (1.0 - probability));
        let adjusted = if lower {
            logit - se2 * scale_at(selow, se, index % scale_len)
        } else {
            logit + se2
        };
        1.0 - 1.0 / (1.0 + safe_exp(adjusted))
    };
    let lower = if scale_len == 0 {
        Vec::new()
    } else {
        (0..se.len()).map(|index| calculate(index, true)).collect()
    };
    let upper = (0..se.len()).map(|index| calculate(index, false)).collect();
    (lower, upper)
}

fn arcsin_intervals(
    p: &[f64],
    se: &[f64],
    logse: bool,
    z: f64,
    selow: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let base_len = p.len().max(se.len());
    let scale_len = selow.map_or(1, <[f64]>::len);
    let se2_at = |index: usize| {
        let probability = p[index % p.len()];
        let xx = if probability == 0.0 {
            f64::NAN
        } else {
            probability
        };
        0.5 * z * prepared_se_at(p, se, logse, index) * r_sqrt(xx / (1.0 - xx))
    };
    let angle_at = |index: usize| {
        let probability = p[index % p.len()];
        let xx = if probability == 0.0 {
            f64::NAN
        } else {
            probability
        };
        r_asin(r_sqrt(xx))
    };
    let lower_len = if scale_len == 0 {
        0
    } else {
        base_len.max(scale_len)
    };
    let lower = (0..lower_len)
        .map(|index| {
            let angle = angle_at(index);
            let se2 = se2_at(index % base_len);
            let scale = scale_at(selow, se, index % scale_len);
            let adjusted = r_pmax_zero(angle - se2 * scale);
            if adjusted.is_nan() {
                f64::NAN
            } else {
                adjusted.sin().powi(2)
            }
        })
        .collect();
    let upper = (0..base_len)
        .map(|index| {
            let adjusted = r_pmin(angle_at(index) + se2_at(index), std::f64::consts::FRAC_PI_2);
            if adjusted.is_nan() {
                f64::NAN
            } else {
                adjusted.sin().powi(2)
            }
        })
        .collect();
    (lower, upper)
}

#[pyfunction]
#[pyo3(signature = (p, se, logse, conf_type, z, selow=None, ulimit=true))]
pub fn survfit_confint_native(
    p: Vec<f64>,
    se: Vec<f64>,
    logse: bool,
    conf_type: &str,
    z: f64,
    selow: Option<Vec<f64>>,
    ulimit: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let interval_type = ConfidenceType::parse(conf_type)?;
    if !z.is_finite() || z < 0.0 {
        return Err(PyErr::new::<PyValueError, _>(
            "z must be finite and non-negative",
        ));
    }
    if se.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if p.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let selow = selow.as_deref();
    Ok(match interval_type {
        ConfidenceType::Plain => plain_intervals(&p, &se, logse, z, selow, ulimit),
        ConfidenceType::Log => log_intervals(&p, &se, logse, z, selow, ulimit),
        ConfidenceType::LogLog => log_log_intervals(&p, &se, logse, z, selow),
        ConfidenceType::Logit => logit_intervals(&p, &se, logse, z, selow),
        ConfidenceType::Arcsin => arcsin_intervals(&p, &se, logse, z, selow),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&left, &right) in actual.iter().zip(expected) {
            if right.is_nan() {
                assert!(left.is_nan());
            } else {
                assert!((left - right).abs() < 1e-7, "{left} != {right}");
            }
        }
    }

    #[test]
    fn matches_r_confidence_transforms() {
        let p = vec![0.2, 0.5, 0.9];
        let z = 1.959_963_984_540_054;
        let cases = [
            (
                "plain",
                vec![0.16080072, 0.4020018, 0.7236032],
                vec![0.23919928, 0.5979982, 1.0],
            ),
            (
                "arcsin",
                vec![0.1623028, 0.4026280, 0.6664164],
                vec![0.2405760, 0.5973720, 0.9992298],
            ),
        ];
        for (kind, expected_lower, expected_upper) in cases {
            let (lower, upper) =
                survfit_confint_native(p.clone(), vec![0.1], true, kind, z, None, true).unwrap();
            assert_close(&lower, &expected_lower);
            assert_close(&upper, &expected_upper);
        }

        for (kind, expected_lower, expected_upper) in [
            ("log", 0.164403, 0.2433045),
            ("log-log", 0.1623716, 0.2405312),
            ("logit", 0.1636537, 0.242082),
        ] {
            let (lower, upper) =
                survfit_confint_native(p.clone(), vec![0.1], true, kind, z, None, true).unwrap();
            assert_close(&lower, &[expected_lower]);
            assert_close(&upper, &[expected_upper]);
        }
    }

    #[test]
    fn preserves_recycling_missing_values_and_asymmetric_lengths() {
        let z = 1.959_963_984_540_054;
        let (lower, upper) = survfit_confint_native(
            vec![0.2, 0.5],
            vec![0.1, 0.2, 0.3],
            true,
            "plain",
            z,
            None,
            true,
        )
        .unwrap();
        assert_close(&lower, &[0.16080072, 0.30400360, 0.08240216]);
        assert_close(&upper, &[0.23919928, 0.69599640, 0.31759784]);

        let (lower, upper) = survfit_confint_native(
            vec![0.2, 0.5],
            vec![0.1],
            true,
            "plain",
            z,
            Some(Vec::new()),
            true,
        )
        .unwrap();
        assert!(lower.is_empty());
        assert_eq!(upper.len(), 2);

        let (lower, upper) = survfit_confint_native(
            vec![0.0, 1.0, f64::NAN],
            vec![0.1, 0.1, 0.1],
            true,
            "log-log",
            z,
            None,
            true,
        )
        .unwrap();
        assert!(lower.iter().all(|value| value.is_nan()));
        assert!(upper.iter().all(|value| value.is_nan()));

        let (lower, upper) =
            survfit_confint_native(vec![0.2, 0.5], vec![0.1], false, "plain", z, None, true)
                .unwrap();
        assert_close(&lower, &[0.004003602, 0.010009004]);
        assert_close(&upper, &[0.3959964, 0.9899910]);

        let (lower, upper) = survfit_confint_native(
            vec![0.2, 0.5],
            vec![0.0, 0.1],
            true,
            "log",
            z,
            Some(Vec::new()),
            true,
        )
        .unwrap();
        assert!(lower.is_empty());
        assert_eq!(upper.len(), 2);

        let (lower, upper) =
            survfit_confint_native(Vec::new(), vec![0.1, 0.2], true, "log-log", z, None, true)
                .unwrap();
        assert!(lower.is_empty());
        assert!(upper.is_empty());

        for kind in ["log", "log-log", "logit"] {
            let (lower, upper) =
                survfit_confint_native(vec![0.0], vec![0.0], true, kind, z, None, true).unwrap();
            assert_close(&lower, &[0.0]);
            assert_close(&upper, &[0.0]);
        }
    }

    #[test]
    fn rejects_invalid_native_options() {
        assert!(
            survfit_confint_native(vec![0.5], vec![0.1], true, "bad", 1.96, None, true).is_err()
        );
        assert!(
            survfit_confint_native(vec![0.5], vec![0.1], true, "plain", f64::NAN, None, true,)
                .is_err()
        );
    }
}
