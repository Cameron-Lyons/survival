//! Predictions from a parametric survival fit: a port of
//! `R/predict.survreg.R` from the CRAN `survival` package.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{
    ProbabilityBounds, validate_finite, validate_length, validate_probability,
};
use crate::regression::parametric_survival::SurvregFit;
use pyo3::prelude::*;

/// The `type` argument of `predict.survreg` (`link`/`linear` are `lp`).
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregPredictType {
    /// `itrans(eta)`: the prediction on the original response scale.
    Response,
    /// The linear predictor `eta`.
    Lp,
    /// Per-term contributions to `eta`, centred at the training means.
    Terms,
    /// Quantiles of the response distribution, one column per `p`.
    Quantile,
    /// Quantiles on the transformed (linear predictor) scale.
    Uquantile,
}

impl SurvregPredictType {
    const CHOICES: [(&'static str, Self); 7] = [
        ("response", Self::Response),
        ("link", Self::Lp),
        ("lp", Self::Lp),
        ("linear", Self::Lp),
        ("terms", Self::Terms),
        ("quantile", Self::Quantile),
        ("uquantile", Self::Uquantile),
    ];

    /// `match.arg(type)`: an exact name or a unique prefix.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        let key = name.trim().to_lowercase();
        if let Some((_, kind)) = Self::CHOICES.iter().find(|(choice, _)| *choice == key) {
            return Ok(*kind);
        }
        let mut matches: Vec<Self> = Self::CHOICES
            .iter()
            .filter(|(choice, _)| !key.is_empty() && choice.starts_with(key.as_str()))
            .map(|(_, kind)| *kind)
            .collect();
        matches.dedup();
        match matches.as_slice() {
            [kind] => Ok(*kind),
            _ => Err(SurvivalError::invalid_input(format!(
                "prediction type '{name}' should be one of {}",
                Self::CHOICES
                    .iter()
                    .map(|(choice, _)| format!("\"{choice}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

/// The value of `predict.survreg`.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregPrediction {
    #[pyo3(get)]
    pub predict_type: SurvregPredictType,
    /// One row per observation.  `Response`/`Lp` have a single column,
    /// `Quantile`/`Uquantile` one column per requested probability, `Terms`
    /// one column per term.
    #[pyo3(get)]
    pub fit: Vec<Vec<f64>>,
    /// Standard errors laid out like `fit`, when requested.
    #[pyo3(get)]
    pub se_fit: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl SurvregPrediction {
    fn __repr__(&self) -> String {
        format!(
            "SurvregPrediction(type={:?}, n={}, columns={}, has_se={})",
            self.predict_type,
            self.fit.len(),
            self.fit.first().map_or(0, Vec::len),
            self.se_fit.is_some()
        )
    }
}

/// The `newdata` of `predict.survreg`: a design matrix with the columns of
/// the training design, an optional offset and, for a fit with strata, the
/// stratum of every row.
///
/// The offset enters every prediction type the way the training offset
/// enters `linear.predictors`.  (R's `predict.survreg` drops the offset of
/// `newdata` altogether, so its new-data predictions of a model with an
/// offset disagree with its training predictions; that is not reproduced.)
#[derive(Debug, Clone, Copy)]
pub struct SurvregNewdata<'a> {
    pub covariates: &'a [Vec<f64>],
    pub offset: Option<&'a [f64]>,
    pub strata: Option<&'a [usize]>,
}

/// The rows a prediction is evaluated on: the training design or `newdata`.
struct PredictionRows<'a> {
    x: &'a [Vec<f64>],
    /// `x %*% coef + offset`, what `predict.survreg` calls `pred` before any
    /// transform.
    eta: Vec<f64>,
    strata: Vec<usize>,
}

fn prediction_rows<'a>(
    fit: &'a SurvregFit,
    newdata: Option<&SurvregNewdata<'a>>,
) -> SurvivalResult<PredictionRows<'a>> {
    let nvar = fit.nvar();
    let coef = &fit.coefficients[..nvar];
    let Some(newdata) = newdata else {
        return Ok(PredictionRows {
            x: &fit.covariates,
            eta: fit.linear_predictors.clone(),
            strata: fit.strata.clone(),
        });
    };
    let n = newdata.covariates.len();
    for (index, row) in newdata.covariates.iter().enumerate() {
        validate_length(nvar, row.len(), &format!("newdata row {index}"))?;
        validate_finite(row, &format!("newdata row {index}"))?;
    }
    if let Some(offset) = newdata.offset {
        validate_length(n, offset.len(), "offset")?;
        validate_finite(offset, "offset")?;
    }
    let strata = match newdata.strata {
        Some(strata) => {
            validate_length(n, strata.len(), "strata")?;
            if let Some(&bad) = strata.iter().find(|&&s| s >= fit.nstrata()) {
                return Err(SurvivalError::invalid_input(format!(
                    "newdata stratum {bad} is not one of the {} fitted strata",
                    fit.nstrata()
                )));
            }
            strata.to_vec()
        }
        None if fit.nstrata() > 1 => {
            return Err(SurvivalError::invalid_input(
                "the fit has several strata; newdata must give the stratum of every row",
            ));
        }
        None => vec![0; n],
    };
    let eta = newdata
        .covariates
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let lp: f64 = row.iter().zip(coef).map(|(x, b)| x * b).sum();
            lp + newdata.offset.map_or(0.0, |offset| offset[i])
        })
        .collect();
    Ok(PredictionRows {
        x: newdata.covariates,
        eta,
        strata,
    })
}

/// `x_i' V x_i` for the leading block of the variance matrix.
fn quadratic(x: &[f64], variance: &[Vec<f64>]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(j, xj)| {
            xj * x
                .iter()
                .enumerate()
                .map(|(k, xk)| variance[j][k] * xk)
                .sum::<f64>()
        })
        .sum()
}

/// `predict.survreg(object, newdata, type, se.fit, terms, p)`.
///
/// `p` is only used by the quantile types.  `assign` gives, for each design
/// column, its term number in R's `attr(x, "assign")` convention (0 for the
/// intercept); without it every non-intercept column is its own term, with
/// an intercept recognised as a leading column of ones.  `terms` selects a
/// subset of the terms (zero-based, in term order).
pub fn predict_survreg(
    fit: &SurvregFit,
    newdata: Option<&SurvregNewdata<'_>>,
    predict_type: SurvregPredictType,
    se_fit: bool,
    p: &[f64],
    assign: Option<&[usize]>,
    terms: Option<&[usize]>,
) -> SurvivalResult<SurvregPrediction> {
    let nvar = fit.nvar();
    let coef = &fit.coefficients[..nvar];
    let variance = &fit.variance_matrix;
    let fixed_scale = variance.len() == nvar;
    let transform = fit.distribution.transform;

    match predict_type {
        SurvregPredictType::Lp | SurvregPredictType::Response => {
            let rows = prediction_rows(fit, newdata)?;
            let mut pred = rows.eta;
            let mut se = se_fit.then(|| {
                rows.x
                    .iter()
                    .map(|x| quadratic(x, variance).sqrt())
                    .collect::<Vec<f64>>()
            });
            if predict_type == SurvregPredictType::Response {
                pred.iter_mut().for_each(|v| *v = transform.inverse(*v));
                if let Some(se) = se.as_mut() {
                    for (s, &value) in se.iter_mut().zip(&pred) {
                        *s /= transform.derivative(value);
                    }
                }
            }
            Ok(SurvregPrediction {
                predict_type,
                fit: pred.into_iter().map(|v| vec![v]).collect(),
                se_fit: se.map(|se| se.into_iter().map(|v| vec![v]).collect()),
            })
        }
        SurvregPredictType::Quantile | SurvregPredictType::Uquantile => {
            validate_probability(p, "p", ProbabilityBounds::Closed)?;
            let rows = prediction_rows(fit, newdata)?;
            let qq: Vec<f64> = p.iter().map(|&p| fit.distribution.quantile(p)).collect();
            let nstrata = fit.nstrata();
            let mut pred: Vec<Vec<f64>> = rows
                .eta
                .iter()
                .zip(&rows.strata)
                .map(|(&eta, &stratum)| {
                    let scale = fit.scale[stratum];
                    qq.iter().map(|q| eta + q * scale).collect()
                })
                .collect();
            let mut se = se_fit.then(|| {
                rows.x
                    .iter()
                    .zip(&rows.strata)
                    .map(|(x, &stratum)| {
                        if fixed_scale {
                            let se = quadratic(x, variance).sqrt();
                            vec![se; qq.len()]
                        } else {
                            // temp <- cbind(x, (qq[i]*scale) * x.strata)
                            let scale = fit.scale[stratum];
                            qq.iter()
                                .map(|q| {
                                    let mut temp = x.to_vec();
                                    temp.resize(nvar + nstrata, 0.0);
                                    temp[nvar + stratum] = q * scale;
                                    quadratic(&temp, variance).sqrt()
                                })
                                .collect()
                        }
                    })
                    .collect::<Vec<Vec<f64>>>()
            });
            if predict_type == SurvregPredictType::Quantile {
                for row in pred.iter_mut() {
                    row.iter_mut().for_each(|v| *v = transform.inverse(*v));
                }
                if let Some(se) = se.as_mut() {
                    for (se_row, pred_row) in se.iter_mut().zip(&pred) {
                        for (s, &value) in se_row.iter_mut().zip(pred_row) {
                            *s /= transform.derivative(value);
                        }
                    }
                }
            }
            Ok(SurvregPrediction {
                predict_type,
                fit: pred,
                se_fit: se,
            })
        }
        SurvregPredictType::Terms => {
            let rows = prediction_rows(fit, newdata)?;
            let assign: Vec<usize> = match assign {
                Some(assign) => {
                    validate_length(nvar, assign.len(), "assign")?;
                    assign.to_vec()
                }
                None => {
                    let intercept = fit.has_intercept();
                    (0..nvar)
                        .map(|j| if intercept { j } else { j + 1 })
                        .collect()
                }
            };
            let intercept = assign.contains(&0);
            let mut term_ids: Vec<usize> = assign.iter().copied().filter(|&t| t != 0).collect();
            term_ids.sort_unstable();
            term_ids.dedup();
            let selected: Vec<usize> = match terms {
                Some(terms) => terms
                    .iter()
                    .map(|&t| {
                        term_ids.get(t).copied().ok_or_else(|| {
                            SurvivalError::invalid_input(format!(
                                "term {t} is out of range for a model with {} terms",
                                term_ids.len()
                            ))
                        })
                    })
                    .collect::<SurvivalResult<_>>()?,
                None => term_ids,
            };
            let columns: Vec<Vec<usize>> = selected
                .iter()
                .map(|&term| (0..nvar).filter(|&j| assign[j] == term).collect())
                .collect();
            // Centre x at the training means when the model has an intercept.
            let centered: Vec<Vec<f64>> = rows
                .x
                .iter()
                .map(|x| {
                    x.iter()
                        .zip(&fit.means)
                        .map(|(v, m)| if intercept { v - m } else { *v })
                        .collect()
                })
                .collect();
            let pred: Vec<Vec<f64>> = centered
                .iter()
                .map(|x| {
                    columns
                        .iter()
                        .map(|cols| cols.iter().map(|&j| x[j] * coef[j]).sum())
                        .collect()
                })
                .collect();
            let se = se_fit.then(|| {
                centered
                    .iter()
                    .map(|x| {
                        columns
                            .iter()
                            .map(|cols| {
                                // xi <- x[j, ii] * coef[ii];  sqrt(xi %*% R[ii, ii] %*% t(xi))
                                let xi: Vec<f64> = cols.iter().map(|&j| x[j] * coef[j]).collect();
                                cols.iter()
                                    .enumerate()
                                    .map(|(a, &ja)| {
                                        xi[a]
                                            * cols
                                                .iter()
                                                .enumerate()
                                                .map(|(b, &jb)| variance[ja][jb] * xi[b])
                                                .sum::<f64>()
                                    })
                                    .sum::<f64>()
                                    .sqrt()
                            })
                            .collect()
                    })
                    .collect()
            });
            Ok(SurvregPrediction {
                predict_type,
                fit: pred,
                se_fit: se,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_types_parse_like_match_arg() {
        assert_eq!(
            SurvregPredictType::parse("link").unwrap(),
            SurvregPredictType::Lp
        );
        assert_eq!(
            SurvregPredictType::parse("linear").unwrap(),
            SurvregPredictType::Lp
        );
        assert_eq!(
            SurvregPredictType::parse("uq").unwrap(),
            SurvregPredictType::Uquantile
        );
        assert_eq!(
            SurvregPredictType::parse("Response").unwrap(),
            SurvregPredictType::Response
        );
        assert_eq!(
            SurvregPredictType::parse("l").unwrap(),
            SurvregPredictType::Lp
        );
        assert!(SurvregPredictType::parse("").is_err());
        assert!(SurvregPredictType::parse("mystery").is_err());
    }

    #[test]
    fn quadratic_form_matches_manual_expansion() {
        let variance = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
        let x = [1.0, 3.0];
        // 1*2*1 + 2*(1*0.5*3) + 3*1*3
        assert!((quadratic(&x, &variance) - 14.0).abs() < 1e-12);
    }
}
