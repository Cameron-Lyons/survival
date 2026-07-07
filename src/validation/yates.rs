use crate::constants::{normal_ci, normal_ci_bounds_95};
use crate::internal::statistical::{normal_cdf, normal_inverse_cdf};
use crate::internal::validation::{
    validate_finite, validate_no_nan, validate_non_empty, validate_non_negative,
};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Result of Yates population prediction
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct YatesResult {
    /// Factor levels being compared
    #[pyo3(get)]
    pub levels: Vec<String>,
    /// Mean population predicted value for each level
    #[pyo3(get)]
    pub means: Vec<f64>,
    /// Standard errors
    #[pyo3(get)]
    pub se: Vec<f64>,
    /// Lower confidence bounds
    #[pyo3(get)]
    pub lower: Vec<f64>,
    /// Upper confidence bounds
    #[pyo3(get)]
    pub upper: Vec<f64>,
    /// Sample size for each level
    #[pyo3(get)]
    pub n: Vec<usize>,
    /// Type of prediction used
    #[pyo3(get)]
    pub predict_type: String,
}

/// Compute population marginal means (Yates-style adjustment).
///
/// This function computes adjusted predictions for each level of a factor,
/// averaging over the distribution of other covariates in the population.
/// This provides a way to estimate "what if everyone had treatment A vs B"
/// effects.
///
/// # Arguments
/// * `predictions` - Predicted values for each observation
/// * `factor` - Factor variable defining groups
/// * `weights` - Optional observation weights
/// * `conf_level` - Confidence level (default: 0.95)
///
/// # Returns
/// * `YatesResult` with adjusted means for each factor level
#[pyfunction]
#[pyo3(signature = (predictions, factor, weights=None, conf_level=None))]
pub fn yates(
    predictions: Vec<f64>,
    factor: Vec<String>,
    weights: Option<Vec<f64>>,
    conf_level: Option<f64>,
) -> PyResult<YatesResult> {
    let n = predictions.len();

    validate_non_empty(&predictions, "predictions")?;
    if factor.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions and factor must have same length",
        ));
    }
    validate_no_nan(&predictions, "predictions")?;
    validate_finite(&predictions, "predictions")?;

    let weights_ref = weights.as_deref();
    if let Some(wts) = weights_ref {
        if wts.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must have same length as predictions",
            ));
        }
        validate_no_nan(wts, "weights")?;
        validate_finite(wts, "weights")?;
        validate_non_negative(wts, "weights")?;
    }

    let conf = conf_level.unwrap_or(0.95);
    let z = z_score(conf)?;

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, level) in factor.iter().enumerate() {
        groups.entry(level.clone()).or_default().push(i);
    }

    let mut levels: Vec<String> = groups.keys().cloned().collect();
    levels.sort();

    let mut means = Vec::with_capacity(levels.len());
    let mut ses = Vec::with_capacity(levels.len());
    let mut lowers = Vec::with_capacity(levels.len());
    let mut uppers = Vec::with_capacity(levels.len());
    let mut ns = Vec::with_capacity(levels.len());

    for level in &levels {
        let Some(indices) = groups.get(level) else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "internal error: missing group level",
            ));
        };
        let group_n = indices.len();

        if group_n == 0 {
            means.push(f64::NAN);
            ses.push(f64::NAN);
            lowers.push(f64::NAN);
            uppers.push(f64::NAN);
            ns.push(0);
            continue;
        }

        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;

        for &i in indices {
            let w = weights_ref.map_or(1.0, |wts| wts[i]);
            let x = predictions[i];
            sum_w += w;
            sum_wx += w * x;
        }

        if sum_w <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must have positive total weight within each factor level",
            ));
        }

        let mean = sum_wx / sum_w;

        let mut sum_w2_dev2 = 0.0;
        for &i in indices {
            let w = weights_ref.map_or(1.0, |wts| wts[i]);
            let dev = predictions[i] - mean;
            sum_w2_dev2 += w * w * dev * dev;
        }

        let variance = sum_w2_dev2 / (sum_w * sum_w);
        let se = variance.sqrt();

        means.push(mean);
        ses.push(se);
        let (lower, upper) = normal_ci(mean, se, z);
        lowers.push(lower);
        uppers.push(upper);
        ns.push(group_n);
    }

    Ok(YatesResult {
        levels,
        means,
        se: ses,
        lower: lowers,
        upper: uppers,
        n: ns,
        predict_type: "linear".to_string(),
    })
}

/// Compute population marginal means with model-based predictions.
///
/// This version takes model coefficients and computes counterfactual
/// predictions for each factor level.
///
/// # Arguments
/// * `x` - Design matrix (flattened, row-major)
/// * `coef` - Model coefficients
/// * `n_obs` - Number of observations
/// * `n_vars` - Number of variables
/// * `factor_col` - Column index of the factor variable
/// * `factor_levels` - Possible levels of the factor
/// * `predict_type` - Type of prediction ("linear", "risk", "survival")
///
/// # Returns
/// * `YatesResult` with adjusted predictions
#[pyfunction]
#[pyo3(signature = (x, coef, n_obs, n_vars, factor_col, factor_levels, predict_type=None))]
pub fn yates_contrast(
    x: Vec<f64>,
    coef: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    factor_col: usize,
    factor_levels: Vec<f64>,
    predict_type: Option<&str>,
) -> PyResult<YatesResult> {
    if n_obs == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_obs must be greater than 0",
        ));
    }
    if n_vars == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_vars must be greater than 0",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }
    if coef.len() != n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coef length must equal n_vars",
        ));
    }
    if factor_col >= n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "factor_col must be < n_vars",
        ));
    }
    validate_non_empty(&factor_levels, "factor_levels")?;
    validate_no_nan(&x, "x")?;
    validate_finite(&x, "x")?;
    validate_no_nan(&coef, "coef")?;
    validate_finite(&coef, "coef")?;
    validate_no_nan(&factor_levels, "factor_levels")?;
    validate_finite(&factor_levels, "factor_levels")?;

    let pred_type = predict_type.unwrap_or("linear");
    validate_predict_type(pred_type)?;

    let mut levels = Vec::with_capacity(factor_levels.len());
    let mut means = Vec::with_capacity(factor_levels.len());
    let mut ses = Vec::with_capacity(factor_levels.len());
    let ns = vec![n_obs; factor_levels.len()];

    for &level in &factor_levels {
        levels.push(format!("{}", level));

        let mut sum_pred = 0.0;
        let mut sum_pred2 = 0.0;

        for i in 0..n_obs {
            let mut eta = 0.0;
            for j in 0..n_vars {
                let x_val = if j == factor_col {
                    level
                } else {
                    x[i * n_vars + j]
                };
                eta += x_val * coef[j];
            }

            let pred = match pred_type {
                "risk" => eta.exp(),
                "survival" => (-eta.exp()).exp(),
                "linear" => eta,
                _ => unreachable!("predict_type was validated"),
            };
            if !pred.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "computed predictions are non-finite; check x and coef scale",
                ));
            }

            sum_pred += pred;
            sum_pred2 += pred * pred;
        }

        let mean = sum_pred / n_obs as f64;
        let variance = (sum_pred2 / n_obs as f64 - mean * mean).max(0.0);
        let se = (variance / n_obs as f64).sqrt();

        means.push(mean);
        ses.push(se);
    }

    let (lower, upper) = normal_ci_bounds_95(&means, &ses);

    Ok(YatesResult {
        levels,
        means,
        se: ses,
        lower,
        upper,
        n: ns,
        predict_type: pred_type.to_string(),
    })
}

/// Compute pairwise contrasts between factor levels
#[pyfunction]
pub fn yates_pairwise(yates_result: &YatesResult) -> PyResult<YatesPairwiseResult> {
    let k = yates_result.levels.len();
    if k < 2 {
        return Ok(YatesPairwiseResult {
            level1: vec![],
            level2: vec![],
            difference: vec![],
            se: vec![],
            z: vec![],
            p_value: vec![],
        });
    }

    let mut level1 = Vec::new();
    let mut level2 = Vec::new();
    let mut difference = Vec::new();
    let mut se = Vec::new();
    let mut z_scores = Vec::new();
    let mut p_values = Vec::new();

    for i in 0..k {
        for j in (i + 1)..k {
            level1.push(yates_result.levels[i].clone());
            level2.push(yates_result.levels[j].clone());

            let diff = yates_result.means[i] - yates_result.means[j];
            difference.push(diff);

            let se_diff = (yates_result.se[i].powi(2) + yates_result.se[j].powi(2)).sqrt();
            se.push(se_diff);

            let z = if se_diff > 0.0 { diff / se_diff } else { 0.0 };
            z_scores.push(z);

            let p = 2.0 * (1.0 - normal_cdf(z.abs()));
            p_values.push(p);
        }
    }

    Ok(YatesPairwiseResult {
        level1,
        level2,
        difference,
        se,
        z: z_scores,
        p_value: p_values,
    })
}

/// Result of pairwise comparisons
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct YatesPairwiseResult {
    #[pyo3(get)]
    pub level1: Vec<String>,
    #[pyo3(get)]
    pub level2: Vec<String>,
    #[pyo3(get)]
    pub difference: Vec<f64>,
    #[pyo3(get)]
    pub se: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
    #[pyo3(get)]
    pub p_value: Vec<f64>,
}

fn z_score(conf_level: f64) -> PyResult<f64> {
    if !conf_level.is_finite() || conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "conf_level must be a finite value between 0 and 1",
        ));
    }
    Ok(normal_inverse_cdf(0.5 + conf_level / 2.0))
}

fn validate_predict_type(predict_type: &str) -> PyResult<()> {
    match predict_type {
        "linear" | "risk" | "survival" => Ok(()),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predict_type must be one of 'linear', 'risk', or 'survival'",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yates_basic() {
        let predictions = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];

        let result = yates(predictions, factor, None, None).unwrap();

        assert_eq!(result.levels.len(), 2);
        assert_eq!(result.means.len(), 2);

        let a_idx = result.levels.iter().position(|l| l == "A").unwrap();
        let b_idx = result.levels.iter().position(|l| l == "B").unwrap();

        assert!((result.means[a_idx] - 1.5).abs() < 0.01);
        assert!((result.means[b_idx] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_yates_weighted() {
        let predictions = vec![1.0, 2.0, 3.0];
        let factor = vec!["A".to_string(), "A".to_string(), "A".to_string()];
        let weights = vec![1.0, 2.0, 1.0];

        let result = yates(predictions, factor, Some(weights), None).unwrap();

        assert!((result.means[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_yates_unweighted_matches_unit_weights() {
        let predictions = vec![1.0, 2.0, 4.0, 8.0];
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let weights = vec![1.0; predictions.len()];

        let unweighted = yates(predictions.clone(), factor.clone(), None, Some(0.95)).unwrap();
        let weighted = yates(predictions, factor, Some(weights), Some(0.95)).unwrap();

        assert_eq!(unweighted.levels, weighted.levels);
        assert_eq!(unweighted.n, weighted.n);
        for i in 0..unweighted.levels.len() {
            assert!((unweighted.means[i] - weighted.means[i]).abs() < 1e-12);
            assert!((unweighted.se[i] - weighted.se[i]).abs() < 1e-12);
            assert!((unweighted.lower[i] - weighted.lower[i]).abs() < 1e-12);
            assert!((unweighted.upper[i] - weighted.upper[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_yates_confidence_level_controls_interval_width() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0];
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
        ];

        let narrow = yates(predictions.clone(), factor.clone(), None, Some(0.80)).unwrap();
        let wide = yates(predictions, factor, None, Some(0.99)).unwrap();

        let narrow_width = narrow.upper[0] - narrow.lower[0];
        let wide_width = wide.upper[0] - wide.lower[0];
        assert!(wide_width > narrow_width);
    }

    #[test]
    fn test_yates_rejects_malformed_inputs() {
        let err = yates(vec![], vec![], None, None).unwrap_err();
        assert!(err.to_string().contains("predictions cannot be empty"));

        let err = yates(vec![f64::NAN], vec!["A".to_string()], None, None).unwrap_err();
        assert!(err.to_string().contains("predictions contains NaN"));

        let err = yates(vec![1.0], vec!["A".to_string()], Some(vec![-1.0]), None).unwrap_err();
        assert!(err.to_string().contains("weights contains negative value"));

        let err = yates(vec![1.0], vec!["A".to_string()], Some(vec![0.0]), None).unwrap_err();
        assert!(
            err.to_string()
                .contains("weights must have positive total weight")
        );

        let err = yates(vec![1.0], vec!["A".to_string()], None, Some(1.0)).unwrap_err();
        assert!(err.to_string().contains("conf_level"));
    }

    #[test]
    fn test_yates_contrast() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let coef = vec![0.5, 1.0];
        let factor_levels = vec![0.0, 1.0];

        let result = yates_contrast(x, coef, 3, 2, 0, factor_levels, Some("linear")).unwrap();

        assert_eq!(result.levels.len(), 2);
    }

    #[test]
    fn test_yates_contrast_rejects_malformed_inputs() {
        let err =
            yates_contrast(vec![], vec![0.5], 0, 1, 0, vec![0.0], Some("linear")).unwrap_err();
        assert!(err.to_string().contains("n_obs must be greater than 0"));

        let err =
            yates_contrast(vec![1.0], vec![0.5], 1, 1, 0, vec![], Some("linear")).unwrap_err();
        assert!(err.to_string().contains("factor_levels cannot be empty"));

        let err = yates_contrast(
            vec![f64::NAN],
            vec![0.5],
            1,
            1,
            0,
            vec![0.0],
            Some("linear"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("x contains NaN"));

        let err = yates_contrast(
            vec![1.0],
            vec![f64::INFINITY],
            1,
            1,
            0,
            vec![0.0],
            Some("linear"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("coef contains non-finite"));

        let err = yates_contrast(
            vec![1.0],
            vec![0.5],
            1,
            1,
            0,
            vec![f64::NAN],
            Some("linear"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("factor_levels contains NaN"));

        let err =
            yates_contrast(vec![1.0], vec![0.5], 1, 1, 0, vec![0.0], Some("bogus")).unwrap_err();
        assert!(err.to_string().contains("predict_type must be one of"));
    }

    #[test]
    fn test_yates_pairwise() {
        let result = YatesResult {
            levels: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            means: vec![1.0, 2.0, 3.0],
            se: vec![0.1, 0.1, 0.1],
            lower: vec![0.8, 1.8, 2.8],
            upper: vec![1.2, 2.2, 3.2],
            n: vec![10, 10, 10],
            predict_type: "linear".to_string(),
        };

        let pairwise = yates_pairwise(&result).unwrap();

        assert_eq!(pairwise.level1.len(), 3);
        assert_eq!(pairwise.difference[0], -1.0);
    }
}
