//! Analysis of deviance tables for Cox models.
//!
//! Port of the table arithmetic of R survival `R/anova.coxph.R` (a single
//! model, terms added sequentially) and `R/anova.coxphlist.R` (a list of
//! nested models).  Refitting the intermediate models of a single-model
//! table needs the design's term assignment and the Cox fitter, so the
//! caller supplies the log-likelihoods and degrees of freedom of every
//! model in order; this module produces R's table shape from them.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pchisq;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;

/// How the models were produced, which decides the sign convention of
/// the chi-square column exactly as R does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnovaKind {
    /// `anova(fit)`: terms added sequentially; `Chisq = 2 * diff(loglik)`.
    Sequential,
    /// `anova(fit1, fit2, ...)`: `Chisq = |2 * diff(loglik)|`, `Df = |diff(df)|`.
    Models,
}

/// One row of the table.  The first model has no test (R's `NA` cells).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct AnovaRow {
    /// R's row name: `"NULL"` then the term labels, or `"1"`, `"2"`, ...
    pub name: String,
    pub loglik: f64,
    pub chisq: Option<f64>,
    pub df: Option<usize>,
    /// `Pr(>|Chi|)`; absent for the first row and when `test` is off.
    pub p_value: Option<f64>,
}

/// R's `anova` data frame for Cox models.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct AnovaCoxphResult {
    pub rows: Vec<AnovaRow>,
    /// `"Chisq"` when p-values were requested (R's `test` argument).
    pub test: Option<String>,
}

/// Build the analysis-of-deviance table from the models' final partial
/// log-likelihoods and their numbers of (non-`NA`) coefficients.
pub fn anova_coxph(
    loglik: &[f64],
    df: &[usize],
    names: &[String],
    kind: AnovaKind,
    test: bool,
) -> SurvivalResult<AnovaCoxphResult> {
    validate_length(loglik.len(), df.len(), "df")?;
    validate_length(loglik.len(), names.len(), "names")?;
    if loglik.is_empty() {
        return Err(SurvivalError::invalid_input(
            "anova needs at least one model",
        ));
    }
    if let Some((index, _)) = loglik.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "loglik[{index}] must be finite"
        )));
    }
    if kind == AnovaKind::Sequential && df.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(SurvivalError::invalid_input(
            "sequential models must not lose degrees of freedom",
        ));
    }
    let rows = (0..loglik.len())
        .map(|i| {
            let (chisq, df_step) = if i == 0 {
                (None, None)
            } else {
                let delta = 2.0 * (loglik[i] - loglik[i - 1]);
                match kind {
                    AnovaKind::Sequential => (Some(delta), Some(df[i] - df[i - 1])),
                    AnovaKind::Models => (Some(delta.abs()), Some(df[i].abs_diff(df[i - 1]))),
                }
            };
            let p_value = match (test, chisq, df_step) {
                (true, Some(chisq), Some(df_step)) => {
                    Some(pchisq(chisq, df_step as f64, false, false))
                }
                _ => None,
            };
            AnovaRow {
                name: names[i].clone(),
                loglik: loglik[i],
                chisq,
                df: df_step,
                p_value,
            }
        })
        .collect();
    Ok(AnovaCoxphResult {
        rows,
        test: test.then(|| "Chisq".to_string()),
    })
}

/// Python entry point: `anova_coxph(loglik, df, names=None, sequential=True,
/// test="Chisq")`.  `names` default to `NULL` and the model index like R.
#[pyfunction(name = "anova_coxph")]
#[pyo3(signature = (loglik, df, names=None, sequential=true, test=Some("Chisq")))]
pub fn anova_coxph_py(
    loglik: Vec<f64>,
    df: Vec<usize>,
    names: Option<Vec<String>>,
    sequential: bool,
    test: Option<&str>,
) -> PyResult<AnovaCoxphResult> {
    let kind = if sequential {
        AnovaKind::Sequential
    } else {
        AnovaKind::Models
    };
    let names = names.unwrap_or_else(|| {
        (0..loglik.len())
            .map(|i| match (kind, i) {
                (AnovaKind::Sequential, 0) => "NULL".to_string(),
                (AnovaKind::Sequential, i) => format!("term {i}"),
                (AnovaKind::Models, i) => (i + 1).to_string(),
            })
            .collect()
    });
    let test = match test {
        None => false,
        Some("Chisq") => true,
        Some(other) => {
            return Err(SurvivalError::invalid_input(format!(
                "test must be 'Chisq' or None, got {other:?}"
            ))
            .into());
        }
    };
    Ok(anova_coxph(&loglik, &df, &names, kind, test)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn sequential_table_matches_r_lung_age_sex() {
        let result = anova_coxph(
            &[-749.9146, -747.7942, -742.8531],
            &[0, 1, 2],
            &names(&["NULL", "age", "sex"]),
            AnovaKind::Sequential,
            true,
        )
        .unwrap();
        assert_eq!(result.test.as_deref(), Some("Chisq"));
        assert_eq!(result.rows[0].chisq, None);
        assert_eq!(result.rows[0].p_value, None);
        assert!((result.rows[1].chisq.unwrap() - 4.2408).abs() < 1e-3);
        assert_eq!(result.rows[1].df, Some(1));
        assert!((result.rows[1].p_value.unwrap() - 0.039461).abs() < 1e-4);
        assert!((result.rows[2].chisq.unwrap() - 9.8822).abs() < 1e-3);
        assert!((result.rows[2].p_value.unwrap() - 0.001669).abs() < 1e-5);
    }

    #[test]
    fn model_list_uses_absolute_differences() {
        let result = anova_coxph(
            &[-742.8531, -747.7942],
            &[2, 1],
            &names(&["1", "2"]),
            AnovaKind::Models,
            true,
        )
        .unwrap();
        assert!((result.rows[1].chisq.unwrap() - 9.8822).abs() < 1e-3);
        assert_eq!(result.rows[1].df, Some(1));
    }

    #[test]
    fn zero_df_steps_follow_pchisq_with_zero_df() {
        let result = anova_coxph(
            &[-10.0, -10.0, -9.5],
            &[1, 1, 1],
            &names(&["1", "2", "3"]),
            AnovaKind::Models,
            true,
        )
        .unwrap();
        assert_eq!(result.rows[1].p_value, Some(1.0));
        assert_eq!(result.rows[2].p_value, Some(0.0));
    }

    #[test]
    fn test_off_drops_p_values_and_inputs_are_validated() {
        let result = anova_coxph(
            &[-10.0, -9.0],
            &[0, 1],
            &names(&["NULL", "x"]),
            AnovaKind::Sequential,
            false,
        )
        .unwrap();
        assert_eq!(result.test, None);
        assert_eq!(result.rows[1].p_value, None);
        assert!(result.rows[1].chisq.is_some());
        assert!(anova_coxph(&[], &[], &[], AnovaKind::Sequential, true).is_err());
        assert!(
            anova_coxph(
                &[f64::NAN, 1.0],
                &[0, 1],
                &names(&["a", "b"]),
                AnovaKind::Sequential,
                true
            )
            .is_err()
        );
    }
}
