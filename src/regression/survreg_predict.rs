use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::match_arg::match_arg;
use crate::internal::validation::validate_length;
use crate::regression::parametric_survival::SurvregFit;
use ndarray::{ArrayView1, ArrayView2};
use pyo3::prelude::*;
use std::collections::BTreeMap;

// R's NA_real_ payload, kept distinct from a numerical NaN in new data.
const NA_REAL: f64 = f64::from_bits(0x7ff8_0000_0000_07a2);

// Uncentered R model matrices can carry signaling NA alongside quiet NaN.
// Their product can yield either missing kind in R. Prefer NA here.
fn design_missing(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut nan = None;
    for value in values {
        if value.is_nan() {
            if value.to_bits() as u32 == 1954 {
                return Some(NA_REAL);
            }
            nan = Some(value);
        }
    }
    nan
}

fn response_standard_error(error: f64, derivative: f64) -> f64 {
    if error.is_nan() {
        error
    } else if derivative.is_nan() {
        derivative
    } else {
        error / derivative
    }
}

fn standard_error(variance: f64) -> f64 {
    if variance.is_nan() {
        variance
    } else {
        variance.sqrt()
    }
}

#[doc = " The `type` argument of `predict.survreg` (`link`/`linear` are `lp`)."]
#[pyclass(module = "survival._survival", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurvregPredictType {
    #[doc = " `itrans(eta)`: the prediction on the original response scale."]
    Response,
    #[doc = " The linear predictor `eta`."]
    Lp,
    #[doc = " Per-term contributions to `eta`, centred at the training means."]
    Terms,
    #[doc = " Quantiles of the response distribution, one column per `p`."]
    Quantile,
    #[doc = " Quantiles on the transformed (linear predictor) scale."]
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

    pub fn parse(name: &str) -> SurvivalResult<Self> {
        let index = match_arg(name, &Self::CHOICES.map(|(choice, _)| choice))?;
        Ok(Self::CHOICES[index].1)
    }
}

#[doc = " The value of `predict.survreg`."]
#[pyclass(module = "survival._survival", from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct SurvregPrediction {
    #[doc = " Column count retained independently of the row count."]
    #[pyo3(get)]
    pub n_columns: usize,
    #[pyo3(get)]
    pub predict_type: SurvregPredictType,
    #[doc = " One row per observation.  `Response`/`Lp` have a single column,"]
    #[doc = " `Quantile`/`Uquantile` one column per requested probability, `Terms`"]
    #[doc = " one column per term."]
    #[pyo3(get)]
    pub fit: Vec<Vec<f64>>,
    #[doc = " Standard errors laid out like `fit`, when requested."]
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
            self.n_columns,
            self.se_fit.is_some()
        )
    }

    #[cfg(feature = "python")]
    #[doc = " Independent writable float64 matrices, including empty dimensions."]
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::internal::numpy_utils::prediction_matrix_arrays(
            py,
            &self.fit,
            self.se_fit.as_deref(),
            self.n_columns,
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SurvregNewdata<'a> {
    pub covariates: ArrayView2<'a, f64>,
    pub offset: Option<&'a [f64]>,
    pub strata: Option<&'a [usize]>,
}

struct PredictionRows<'a> {
    x: ArrayView2<'a, f64>,
    eta: Vec<f64>,
    strata: Vec<usize>,
}

fn prediction_design<'a>(
    fit: &'a SurvregFit,
    newdata: Option<&'a SurvregNewdata<'_>>,
) -> SurvivalResult<ArrayView2<'a, f64>> {
    let Some(newdata) = newdata else {
        return Ok(fit.covariates.view());
    };
    let nvar = fit.nvar();
    let n = newdata.covariates.nrows();
    validate_length(nvar, newdata.covariates.ncols(), "newdata columns")?;
    if let Some(((row, column), value)) = newdata
        .covariates
        .indexed_iter()
        .find(|(_, value)| value.is_infinite())
    {
        return Err(SurvivalError::invalid_input(format!(
            "newdata contains non-finite value {value} at row {row}, column {column}"
        )));
    }
    if let Some(offset) = newdata.offset {
        validate_length(n, offset.len(), "offset")?;
        if let Some(value) = offset.iter().find(|value| value.is_infinite()) {
            return Err(SurvivalError::invalid_input(format!(
                "offset contains non-finite value {value}"
            )));
        }
    }
    match newdata.strata {
        Some(strata) => {
            validate_length(n, strata.len(), "strata")?;
            if let Some(&bad) = strata.iter().find(|&&s| s >= fit.nstrata()) {
                return Err(SurvivalError::invalid_input(format!(
                    "newdata stratum {bad} is not one of the {} fitted strata",
                    fit.nstrata()
                )));
            }
        }
        None if fit.nstrata() > 1 => {
            return Err(SurvivalError::invalid_input(
                "the fit has several strata; newdata must give the stratum of every row",
            ));
        }
        None => {}
    }
    Ok(newdata.covariates)
}

fn prediction_rows<'a>(
    fit: &'a SurvregFit,
    newdata: Option<&'a SurvregNewdata<'_>>,
) -> SurvivalResult<PredictionRows<'a>> {
    let x = prediction_design(fit, newdata)?;
    let Some(newdata) = newdata else {
        return Ok(PredictionRows {
            x,
            eta: fit.linear_predictors.clone(),
            strata: fit.strata.clone(),
        });
    };
    let coef = &fit.coefficients[..fit.nvar()];
    let strata = newdata
        .strata
        .map_or_else(|| vec![0; x.nrows()], <[_]>::to_vec);
    let eta = if coef.iter().any(|b| b.is_nan()) {
        // Older serialized fits also represent an aliased coefficient as NaN.
        vec![NA_REAL; x.nrows()]
    } else {
        x.outer_iter()
            .enumerate()
            .map(|(i, row)| {
                let lp: f64 = row.iter().zip(coef).map(|(x, b)| x * b).sum();
                let offset = newdata.offset.map_or(0.0, |offset| offset[i]);
                if lp.is_nan() {
                    design_missing(row.iter().copied().chain([offset])).unwrap_or(lp)
                } else {
                    lp + offset
                }
            })
            .collect()
    };
    Ok(PredictionRows {
        x: newdata.covariates.view(),
        eta,
        strata,
    })
}

fn quadratic(x: ArrayView1<'_, f64>, variance: &[Vec<f64>]) -> f64 {
    let value: f64 = x
        .iter()
        .enumerate()
        .map(|(j, xj)| {
            xj * x
                .iter()
                .enumerate()
                .map(|(k, xk)| variance[j][k] * xk)
                .sum::<f64>()
        })
        .sum();
    if value.is_nan() {
        design_missing(x.iter().copied()).unwrap_or(value)
    } else {
        value
    }
}

/// All quantile gradients for a row share its location covariates. Only one
/// log-scale derivative, q * sigma, varies. Accumulate g' V g in its original
/// order, reusing each covariance row's inner product up to that derivative.
fn quantile_standard_errors(
    x: ArrayView1<'_, f64>,
    variance: &[Vec<f64>],
    stratum: usize,
    sigma: f64,
    quantiles: &[f64],
) -> Vec<f64> {
    if quantiles.is_empty() {
        return Vec::new();
    }
    if let Some(missing) = design_missing(x.iter().copied()) {
        return vec![missing; quantiles.len()];
    }
    let scale_column = x.len() + stratum;
    if let &[q] = quantiles {
        let mut gradient = x.to_vec();
        gradient.resize(variance.len(), 0.0);
        gradient[scale_column] = q * sigma;
        return vec![if q.is_nan() {
            q
        } else {
            standard_error(quadratic(ArrayView1::from(&gradient), variance))
        }];
    }
    let mut sums = vec![-0.0; quantiles.len()];
    for (j, row) in variance.iter().enumerate() {
        let location: f64 = row.iter().zip(x).map(|(v, x)| v * x).sum();
        // Inactive scale derivatives are zero. Keep their products: a
        // nonfinite covariance can still propagate through a zero derivative.
        let prefix = row[x.len()..scale_column]
            .iter()
            .fold(location, |acc, v| acc + v * 0.0);
        let suffix: f64 = row[scale_column + 1..].iter().map(|v| v * 0.0).sum();
        let weight = x.get(j).copied().unwrap_or(0.0);
        let covariance = row[scale_column];
        for (&q, sum) in quantiles.iter().zip(&mut sums) {
            let derivative = q * sigma;
            let inner = (prefix + covariance * derivative) + suffix;
            let outer = if j == scale_column {
                derivative
            } else {
                weight
            };
            *sum += outer * inner;
        }
    }
    for (v, &q) in sums.iter_mut().zip(quantiles) {
        *v = if q.is_nan() { q } else { standard_error(*v) };
    }
    sums
}

pub fn predict_survreg(
    fit: &SurvregFit,
    newdata: Option<&SurvregNewdata<'_>>,
    predict_type: SurvregPredictType,
    se_fit: bool,
    p: &[f64],
    assign: Option<&[usize]>,
    terms: Option<&[usize]>,
) -> SurvivalResult<SurvregPrediction> {
    fit.check_dimensions()?;
    if se_fit {
        fit.check_covariance(&fit.variance_matrix)?;
    }
    let nvar = fit.nvar();
    let variance = &fit.variance_matrix;
    let fixed_scale = variance.len() == nvar;

    match predict_type {
        SurvregPredictType::Lp | SurvregPredictType::Response => {
            let rows = prediction_rows(fit, newdata)?;
            let mut pred = rows.eta;
            let mut se = se_fit.then(|| {
                rows.x
                    .outer_iter()
                    .map(|x| standard_error(quadratic(x, variance)))
                    .collect::<Vec<f64>>()
            });
            if predict_type == SurvregPredictType::Response {
                pred = fit.distribution.inverse_values(&pred)?;
                if let Some(se) = se.as_mut() {
                    let derivatives = fit.distribution.transform_derivatives(&pred)?;
                    for (s, d) in se.iter_mut().zip(derivatives) {
                        *s = response_standard_error(*s, d);
                    }
                }
            }
            Ok(SurvregPrediction {
                n_columns: 1,
                predict_type,
                fit: pred.into_iter().map(|v| vec![v]).collect(),
                se_fit: se.map(|se| se.into_iter().map(|v| vec![v]).collect()),
            })
        }
        SurvregPredictType::Quantile | SurvregPredictType::Uquantile => {
            if newdata.is_none() {
                fit.check_stored_strata()?;
            }
            let rows = prediction_rows(fit, newdata)?;
            let qq = fit.distribution.quantiles(p)?;
            // R's scalar addition takes a missing quantile first. Vector
            // addition takes the left (location) operand when both are missing.
            let scalar = rows.eta.len() == 1 && qq.len() == 1;
            let mut pred: Vec<Vec<f64>> = rows
                .eta
                .iter()
                .zip(&rows.strata)
                .map(|(&eta, &stratum)| {
                    let scale = fit.scale[stratum];
                    qq.iter()
                        .map(|&q| {
                            if q.is_nan() && (scalar || !eta.is_nan()) {
                                q
                            } else if eta.is_nan() {
                                eta
                            } else {
                                eta + q * scale
                            }
                        })
                        .collect()
                })
                .collect();
            let mut se = se_fit.then(|| {
                rows.x
                    .outer_iter()
                    .zip(&rows.strata)
                    .map(|(x, &stratum)| {
                        if fixed_scale {
                            let se = standard_error(quadratic(x, variance));
                            vec![se; qq.len()]
                        } else {
                            quantile_standard_errors(x, variance, stratum, fit.scale[stratum], &qq)
                        }
                    })
                    .collect::<Vec<Vec<f64>>>()
            });
            if predict_type == SurvregPredictType::Quantile {
                if fit.distribution.transform_callbacks.is_some() {
                    let values: Vec<_> = pred.iter().flatten().copied().collect();
                    let values = fit.distribution.inverse_values(&values)?;
                    for (v, value) in pred.iter_mut().flatten().zip(values) {
                        *v = value;
                    }
                } else {
                    for v in pred.iter_mut().flatten() {
                        *v = fit.distribution.transform.inverse(*v)?;
                    }
                }
                if let Some(se) = se.as_mut() {
                    if fit.distribution.transform_callbacks.is_some() {
                        let values: Vec<_> = pred.iter().flatten().copied().collect();
                        let derivatives = fit.distribution.transform_derivatives(&values)?;
                        for (s, d) in se.iter_mut().flatten().zip(derivatives) {
                            *s = response_standard_error(*s, d);
                        }
                    } else {
                        for (s, &value) in se.iter_mut().flatten().zip(pred.iter().flatten()) {
                            *s = response_standard_error(
                                *s,
                                fit.distribution.transform.derivative(value)?,
                            );
                        }
                    }
                }
            }
            Ok(SurvregPrediction {
                n_columns: p.len(),
                predict_type,
                fit: pred,
                se_fit: se,
            })
        }
        SurvregPredictType::Terms => predict_terms(fit, newdata, assign, terms, se_fit),
    }
}

#[derive(Default)]
struct PredictionTerm {
    columns: Vec<usize>,
    outputs: Vec<usize>,
    first_alias: Option<usize>,
}

/// R's term SE uses coefficient-weighted design columns in its covariance
/// quadratic form. Sum the products in the design order for that term.
fn term_with_error(
    columns: &[usize],
    contribution: impl Fn(usize) -> f64,
    variance: &[Vec<f64>],
    products: &mut Vec<f64>,
) -> (f64, f64) {
    if let &[j] = columns {
        let v = contribution(j);
        return (v, (v * (variance[j][j] * v)).sqrt());
    }
    products.clear();
    products.extend(columns.iter().map(|&j| contribution(j)));
    let var: f64 = columns
        .iter()
        .zip(products.iter())
        .map(|(&j, xj)| {
            let inner: f64 = columns
                .iter()
                .zip(products.iter())
                .map(|(&k, xk)| variance[j][k] * xk)
                .sum();
            xj * inner
        })
        .sum();
    (products.iter().sum(), var.sqrt())
}

fn predict_terms(
    fit: &SurvregFit,
    newdata: Option<&SurvregNewdata<'_>>,
    assign: Option<&[usize]>,
    terms: Option<&[usize]>,
    se_fit: bool,
) -> SurvivalResult<SurvregPrediction> {
    // Term contributions use only the design. Keep the ordinary validation
    // of offsets and strata without calculating an unused linear predictor.
    let x = prediction_design(fit, newdata)?;
    let nvar = x.ncols();
    let (mut groups, centered): (Vec<_>, _) = if let Some(assign) = assign {
        validate_length(nvar, assign.len(), "assign")?;
        let mut by_id = BTreeMap::<usize, PredictionTerm>::new();
        for (j, &id) in assign.iter().enumerate() {
            if id != 0 {
                by_id.entry(id).or_default().columns.push(j);
            }
        }
        (by_id.into_values().collect(), assign.contains(&0))
    } else {
        let intercept = fit.has_intercept();
        let groups = (usize::from(intercept)..nvar)
            .map(|j| PredictionTerm {
                columns: vec![j],
                ..Default::default()
            })
            .collect();
        (groups, intercept)
    };
    // Selectors index the sorted distinct positive IDs. Multiple requests for
    // the same term share its calculation but retain their output positions.
    let ngroups = groups.len();
    let n_columns = if let Some(terms) = terms {
        for (out, &index) in terms.iter().enumerate() {
            let group = groups.get_mut(index).ok_or_else(|| {
                SurvivalError::invalid_input(format!(
                    "term {index} is out of range for a model with {ngroups} terms"
                ))
            })?;
            group.outputs.push(out);
        }
        groups.retain(|g| !g.outputs.is_empty());
        terms.len()
    } else {
        for (out, group) in groups.iter_mut().enumerate() {
            group.outputs.push(out);
        }
        ngroups
    };

    for group in &mut groups {
        group.first_alias = group
            .columns
            .iter()
            .position(|&j| fit.coefficients[j].is_nan());
    }
    let mut predictions = Vec::with_capacity(x.nrows());
    let mut errors = se_fit.then(|| Vec::with_capacity(x.nrows()));
    let mut products = Vec::new();
    for row in x.outer_iter() {
        let contribution = |j: usize| {
            let x = if centered {
                row[j] - fit.means[j]
            } else {
                row[j]
            };
            x * fit.coefficients[j]
        };
        let mut pred = vec![0.0; n_columns];
        let mut se = se_fit.then(|| vec![0.0; n_columns]);
        for group in &groups {
            let columns = &group.columns;
            let (mut value, mut error) = if let Some(alias) = group.first_alias {
                // R multiplies the design by coefficients before computing
                // the SE. An earlier missing product keeps its own payload.
                let error = if se_fit {
                    columns[..alias]
                        .iter()
                        .map(|&j| row[j])
                        .find(|x| x.is_nan())
                        .unwrap_or(NA_REAL)
                } else {
                    NA_REAL
                };
                (NA_REAL, error)
            } else if se_fit {
                term_with_error(columns, contribution, &fit.variance_matrix, &mut products)
            } else {
                (columns.iter().map(|&j| contribution(j)).sum(), 0.0)
            };
            if value.is_nan() && group.first_alias.is_none() {
                // Centering and coefficient multiplication quiet the original
                // R NA. For those products the first missing column wins.
                let values = columns.iter().map(|&j| row[j]);
                let first = values.clone().find(|x| x.is_nan());
                if let Some(missing) = first {
                    value = if centered {
                        missing
                    } else {
                        design_missing(values).unwrap_or(missing)
                    };
                    error = missing;
                }
            }
            if let Some(se) = se.as_mut() {
                for &out in &group.outputs {
                    se[out] = error;
                }
            }
            for &out in &group.outputs {
                pred[out] = value;
            }
        }
        predictions.push(pred);
        if let (Some(errors), Some(se)) = (&mut errors, se) {
            errors.push(se);
        }
    }
    Ok(SurvregPrediction {
        n_columns,
        predict_type: SurvregPredictType::Terms,
        fit: predictions,
        se_fit: errors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::parametric_survival::{SurvregControl, SurvregData, survreg_fit};
    use crate::regression::survreg_distributions::SurvregDistribution;

    fn probability_fit(family: &str, scale: f64) -> SurvregFit {
        let data = SurvregData::try_new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0],
            vec![1, 1, 0, 1, 0, 1, 1, 1],
            ndarray::array![
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0]
            ],
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        survreg_fit(
            &data,
            &SurvregDistribution::from_name(family, None).unwrap(),
            None,
            scale,
            &SurvregControl::default(),
            false,
        )
        .unwrap()
    }

    #[test]
    fn aliases_preserve_the_missing_kind_of_independent_terms_and_standard_errors() {
        let mut fit = probability_fit("weibull", 0.0);
        // Simulate a fit saved when aliases used the ordinary NaN payload.
        fit.coefficients[0] = f64::NAN;
        let x = ndarray::array![
            [1.0, 0.0],
            [1.0, f64::NAN],
            [1.0, NA_REAL],
            [f64::NAN, NA_REAL],
            [NA_REAL, f64::NAN]
        ];
        let newdata = SurvregNewdata {
            covariates: x.view(),
            offset: None,
            strata: None,
        };
        let terms = predict_survreg(
            &fit,
            Some(&newdata),
            SurvregPredictType::Terms,
            true,
            &[],
            Some(&[1, 2]),
            None,
        )
        .unwrap();
        for rows in [&terms.fit, terms.se_fit.as_ref().unwrap()] {
            assert!(rows.iter().all(|r| r[0].to_bits() == NA_REAL.to_bits()));
            assert!(rows[0][1].is_finite());
            assert!(rows[1][1].is_nan() && rows[1][1].to_bits() as u32 != 1954);
            assert_eq!(rows[2][1].to_bits(), NA_REAL.to_bits());
        }
        for kind in [
            SurvregPredictType::Lp,
            SurvregPredictType::Response,
            SurvregPredictType::Uquantile,
            SurvregPredictType::Quantile,
        ] {
            let prediction =
                predict_survreg(&fit, Some(&newdata), kind, true, &[0.4], None, None).unwrap();
            assert!(
                prediction
                    .fit
                    .iter()
                    .all(|r| r[0].to_bits() == NA_REAL.to_bits()),
                "{kind:?}"
            );
            let errors = prediction.se_fit.unwrap();
            assert!(
                errors[1][0].is_nan() && errors[1][0].to_bits() as u32 != 1954,
                "{kind:?}"
            );
            assert_eq!(errors[2][0].to_bits(), NA_REAL.to_bits(), "{kind:?}");
            assert_eq!(errors[3][0].to_bits(), NA_REAL.to_bits(), "{kind:?}");
            assert_eq!(errors[4][0].to_bits(), NA_REAL.to_bits(), "{kind:?}");
        }
        fit.coefficients[0] = 1.0;
        fit.coefficients[1] = f64::NAN;
        let grouped = predict_survreg(
            &fit,
            Some(&newdata),
            SurvregPredictType::Terms,
            true,
            &[],
            Some(&[1, 1]),
            None,
        )
        .unwrap();
        assert!(
            grouped
                .fit
                .iter()
                .all(|r| r[0].to_bits() == NA_REAL.to_bits())
        );
        let errors = grouped.se_fit.unwrap();
        assert!(errors[3][0].is_nan() && errors[3][0].to_bits() as u32 != 1954);
        assert_eq!(errors[4][0].to_bits(), NA_REAL.to_bits());
        fit.coefficients[1] = 1.0;
        let full_rank = prediction_rows(&fit, Some(&newdata)).unwrap();
        assert_eq!(full_rank.eta[3].to_bits(), NA_REAL.to_bits());
        assert_eq!(full_rank.eta[4].to_bits(), NA_REAL.to_bits());
    }

    #[test]
    fn scalar_and_vector_quantiles_follow_r_missing_operand_order() {
        let fit = probability_fit("weibull", 0.0);
        let x = ndarray::array![[1.0, f64::NAN], [1.0, NA_REAL]];
        for (row, probability) in [NA_REAL, f64::NAN].into_iter().enumerate() {
            let one = SurvregNewdata {
                covariates: x.slice(ndarray::s![row..row + 1, ..]),
                offset: None,
                strata: None,
            };
            for p in [vec![probability], vec![probability, 0.4]] {
                let prediction = predict_survreg(
                    &fit,
                    Some(&one),
                    SurvregPredictType::Uquantile,
                    false,
                    &p,
                    None,
                    None,
                )
                .unwrap();
                let value = prediction.fit[0][0];
                let expected = if p.len() == 1 {
                    probability
                } else {
                    x[[row, 1]]
                };
                assert!(value.is_nan());
                assert_eq!(
                    value.to_bits() as u32 == 1954,
                    expected.to_bits() as u32 == 1954
                );
            }
        }
    }

    #[test]
    fn missing_probability_columns_preserve_other_quantiles() {
        for family in [
            "weibull",
            "lognormal",
            "loglogistic",
            "gaussian",
            "logistic",
            "t",
        ] {
            let fit = probability_fit(family, 0.0);
            for kind in [SurvregPredictType::Quantile, SurvregPredictType::Uquantile] {
                for errors in [false, true] {
                    let baseline =
                        predict_survreg(&fit, None, kind, errors, &[0.2, 0.8], None, None).unwrap();
                    let actual = predict_survreg(
                        &fit,
                        None,
                        kind,
                        errors,
                        &[
                            0.2,
                            f64::NAN,
                            -0.1,
                            1.1,
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                            0.8,
                        ],
                        None,
                        None,
                    )
                    .unwrap();
                    assert_eq!(actual.n_columns, 7);
                    for (row, expected) in actual.fit.iter().zip(&baseline.fit) {
                        assert_eq!(row[0], expected[0]);
                        assert_eq!(row[6], expected[1]);
                        assert!(row[1..6].iter().all(|value| value.is_nan()));
                    }
                    if let Some(errors) = actual.se_fit {
                        for (row, expected) in errors.iter().zip(baseline.se_fit.unwrap()) {
                            assert_eq!(row[0], expected[0]);
                            assert_eq!(row[6], expected[1]);
                            assert!(row[1..6].iter().all(|value| value.is_nan()));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_scale_probability_errors_keep_location_standard_errors() {
        for family in ["weibull", "gaussian"] {
            let fit = probability_fit(family, 0.7);
            let lp =
                predict_survreg(&fit, None, SurvregPredictType::Lp, true, &[], None, None).unwrap();
            for kind in [SurvregPredictType::Quantile, SurvregPredictType::Uquantile] {
                let actual =
                    predict_survreg(&fit, None, kind, true, &[f64::NAN], None, None).unwrap();
                assert!(actual.fit.iter().all(|row| row[0].is_nan()));
                for (row, expected) in actual
                    .se_fit
                    .unwrap()
                    .iter()
                    .zip(lp.se_fit.as_ref().unwrap())
                {
                    if family == "weibull" && kind == SurvregPredictType::Quantile {
                        assert!(row[0].is_nan());
                    } else {
                        assert_eq!(row[0], expected[0]);
                    }
                }
            }
        }
    }

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
        assert!(SurvregPredictType::parse("Response").is_err());
        assert!(SurvregPredictType::parse("l").is_err());
        assert!(SurvregPredictType::parse("").is_err());
        assert!(SurvregPredictType::parse("mystery").is_err());
    }

    #[test]
    fn sparse_term_assignments_and_repeated_selectors_keep_design_order() {
        let fit = probability_fit("weibull", 0.0);
        let design = ndarray::array![[1.0, 0.0, 3.0], [1.0, 0.0, -0.25]];
        let x = design.slice(ndarray::s![.., ..;2]);
        let newdata = SurvregNewdata {
            covariates: x,
            offset: None,
            strata: None,
        };
        let high = usize::MAX;
        let selected = fit
            .predict(
                Some(&newdata),
                SurvregPredictType::Terms,
                true,
                &[],
                Some(&[high, 2]),
                Some(&[1, 0, 1]),
            )
            .unwrap();
        let grouped = fit
            .predict(
                Some(&newdata),
                SurvregPredictType::Terms,
                true,
                &[],
                Some(&[high, high]),
                Some(&[0, 0]),
            )
            .unwrap();
        for (i, row) in x.outer_iter().enumerate() {
            let weighted = [row[0] * fit.coefficients[0], row[1] * fit.coefficients[1]];
            let v = &fit.variance_matrix;
            let se = |j: usize| (weighted[j] * (v[j][j] * weighted[j])).sqrt();
            assert_eq!(selected.fit[i], [weighted[0], weighted[1], weighted[0]]);
            assert_eq!(selected.se_fit.as_ref().unwrap()[i], [se(0), se(1), se(0)]);
            let variance = quadratic(ArrayView1::from(&weighted[..]), v);
            assert_eq!(grouped.fit[i], [weighted.iter().sum::<f64>(); 2]);
            assert_eq!(grouped.se_fit.as_ref().unwrap()[i], [variance.sqrt(); 2]);
        }
    }

    #[test]
    fn quadratic_form_matches_manual_expansion() {
        let variance = vec![vec![2.0, 0.5], vec![0.5, 1.0]];
        let x = [1.0, 3.0];
        assert!((quadratic(ArrayView1::from(&x), &variance) - 14.0).abs() < 1e-12);
    }

    #[test]
    fn quantile_standard_errors_match_dense_gradients_for_each_stratum() {
        let x = [1.0, -2.75];
        let mut variance = vec![
            vec![1.0, -0.1, 0.0, 0.25, 0.45],
            vec![-0.1, 0.5, -0.1, 0.0, -0.05],
            vec![0.0, -0.1, 0.43, 0.0, 0.025],
            vec![0.25, 0.0, 0.0, 0.52, 0.0],
            vec![0.45, -0.05, 0.025, 0.0, 0.75],
        ];
        let qs = [-2.4, -0.0, 0.3, 3.1, f64::INFINITY, f64::NAN];
        for covariance in [0.45, f64::INFINITY] {
            variance[0][4] = covariance;
            for stratum in 0..3 {
                let actual =
                    quantile_standard_errors((&x[..]).into(), &variance, stratum, 1.3, &qs);
                let mut gradient = [1.0, -2.75, 0.0, 0.0, 0.0];
                for (se, q) in actual.into_iter().zip(qs) {
                    gradient[x.len() + stratum] = q * 1.3;
                    let dense = quadratic((&gradient[..]).into(), &variance).sqrt();
                    assert!(se.to_bits() == dense.to_bits() || (se.is_nan() && dense.is_nan()));
                }
            }
        }
    }
}
