use crate::utilities::statistical::normal_cdf;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Result of Yates population prediction
#[derive(Debug, Clone)]
#[pyclass]
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

    if factor.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predictions and factor must have same length",
        ));
    }

    let wts = weights.unwrap_or_else(|| vec![1.0; n]);
    if wts.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must have same length as predictions",
        ));
    }

    let conf = conf_level.unwrap_or(0.95);
    let z = z_score(conf);

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
        let indices = groups.get(level).unwrap();
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
            let w = wts[i];
            let x = predictions[i];
            sum_w += w;
            sum_wx += w * x;
        }

        let mean = if sum_w > 0.0 { sum_wx / sum_w } else { 0.0 };

        let mut sum_w2_dev2 = 0.0;
        for &i in indices {
            let w = wts[i];
            let dev = predictions[i] - mean;
            sum_w2_dev2 += w * w * dev * dev;
        }

        let variance = if sum_w > 0.0 {
            sum_w2_dev2 / (sum_w * sum_w)
        } else {
            0.0
        };
        let se = variance.sqrt();

        means.push(mean);
        ses.push(se);
        lowers.push(mean - z * se);
        uppers.push(mean + z * se);
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

    let pred_type = predict_type.unwrap_or("linear");

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
                _ => eta,
            };

            sum_pred += pred;
            sum_pred2 += pred * pred;
        }

        let mean = sum_pred / n_obs as f64;
        let variance = sum_pred2 / n_obs as f64 - mean * mean;
        let se = (variance / n_obs as f64).sqrt();

        means.push(mean);
        ses.push(se);
    }

    let z = 1.96;
    let lower: Vec<f64> = means
        .iter()
        .zip(ses.iter())
        .map(|(&m, &s)| m - z * s)
        .collect();
    let upper: Vec<f64> = means
        .iter()
        .zip(ses.iter())
        .map(|(&m, &s)| m + z * s)
        .collect();

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
#[pyclass]
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

fn z_score(conf_level: f64) -> f64 {
    match conf_level {
        c if (c - 0.90).abs() < 0.001 => 1.645,
        c if (c - 0.95).abs() < 0.001 => 1.96,
        c if (c - 0.99).abs() < 0.001 => 2.576,
        _ => 1.96,
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

        // Group A mean should be 1.5, Group B mean should be 3.0
        let a_idx = result.levels.iter().position(|l| l == "A").unwrap();
        let b_idx = result.levels.iter().position(|l| l == "B").unwrap();

        assert!((result.means[a_idx] - 1.5).abs() < 0.01);
        assert!((result.means[b_idx] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_yates_weighted() {
        let predictions = vec![1.0, 2.0, 3.0];
        let factor = vec!["A".to_string(), "A".to_string(), "A".to_string()];
        let weights = vec![1.0, 2.0, 1.0]; // More weight on middle value

        let result = yates(predictions, factor, Some(weights), None).unwrap();

        // Weighted mean: (1*1 + 2*2 + 1*3) / 4 = 8/4 = 2.0
        assert!((result.means[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_yates_contrast() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2 matrix
        let coef = vec![0.5, 1.0];
        let factor_levels = vec![0.0, 1.0];

        let result = yates_contrast(x, coef, 3, 2, 0, factor_levels, Some("linear")).unwrap();

        assert_eq!(result.levels.len(), 2);
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

        assert_eq!(pairwise.level1.len(), 3); // 3 pairs
        assert_eq!(pairwise.difference[0], -1.0); // A - B
    }
}
