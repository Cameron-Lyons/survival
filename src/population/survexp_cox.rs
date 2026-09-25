//! Expected population survival from fitted Cox curves (`survexp.cfit`).
use super::survexp::SurvExpResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use crate::regression::CoxSurvfitCurve;
use pyo3::prelude::*;

/// Average one predicted curve per subject. A curve block can contain several
/// columns; blocks and columns follow the subject order in `group`.
pub fn survexp_cox(
    curves: &[CoxSurvfitCurve],
    group: &[usize],
    weights: &[f64],
    y: Option<&[f64]>,
    times: Option<&[f64]>,
    method: &str,
) -> SurvivalResult<SurvExpResult> {
    let n = group.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input("Data set has 0 rows"));
    }
    validate_length(n, weights.len(), "weights")?;
    validate_finite(weights, "weights")?;
    validate_non_negative(weights, "weights")?;
    if !matches!(method, "ederer" | "hakulinen" | "conditional") {
        return Err(SurvivalError::invalid_input(
            "invalid cohort survival method",
        ));
    }
    if method != "ederer" && y.is_none() {
        return Err(SurvivalError::invalid_input(
            "a response is required for this method",
        ));
    }
    if let Some(y) = y {
        validate_length(n, y.len(), "y")?;
        validate_finite(y, "y")?;
        validate_non_negative(y, "y")?;
    }
    let Some(groups) = group
        .iter()
        .max()
        .unwrap()
        .checked_add(1)
        .filter(|&groups| groups <= n)
    else {
        return Err(SurvivalError::invalid_input(
            "group codes must be contiguous",
        ));
    };
    let mut totals = vec![0.0; groups];
    let mut counts = vec![0.0; groups];
    for (&g, &w) in group.iter().zip(weights) {
        totals[g] += w;
        counts[g] += 1.0;
    }
    if totals.iter().any(|&w| w <= 0.0 || !w.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "every group must have positive finite total weight",
        ));
    }
    let mut subjects = Vec::with_capacity(n);
    let mut grid = Vec::new();
    for curve in curves {
        validate_finite(&curve.time, "curve time")?;
        if curve.time.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(SurvivalError::invalid_input(
                "curve times must be increasing",
            ));
        }
        validate_length(curve.time.len(), curve.surv.len(), "surv")?;
        validate_length(curve.time.len(), curve.cumhaz.len(), "cumhaz")?;
        let width = curve.surv.first().map_or(0, Vec::len);
        for row in curve.surv.iter().chain(&curve.cumhaz) {
            validate_length(width, row.len(), "curve row")?;
        }
        subjects.extend((0..width).map(|column| (curve, column)));
        grid.extend_from_slice(&curve.time);
    }
    validate_length(n, subjects.len(), "predicted curves")?;
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let mut survival = vec![vec![0.0; groups]; grid.len()];
    let mut risk = vec![vec![0.0; groups]; grid.len()];
    let mut cumulative = vec![0.0; groups];
    let mut previous_h = vec![0.0; n];
    let mut previous_s = vec![1.0; n];
    let mut positions = vec![0; n];
    for (t, &time) in grid.iter().enumerate() {
        let mut numerator = vec![0.0; groups];
        let mut denominator = vec![0.0; groups];
        for (i, &(curve, col)) in subjects.iter().enumerate() {
            while positions[i] < curve.time.len() && curve.time[positions[i]] <= time {
                positions[i] += 1;
            }
            let (s, h) = if positions[i] == 0 {
                (1.0, 0.0)
            } else {
                (
                    curve.surv[positions[i] - 1][col],
                    curve.cumhaz[positions[i] - 1][col],
                )
            };
            let g = group[i];
            if method == "ederer" {
                numerator[g] += weights[i] * s;
                denominator[g] += weights[i];
                risk[t][g] += 1.0;
            } else if y.unwrap()[i] >= time {
                let weight = weights[i]
                    * if method == "hakulinen" {
                        previous_s[i]
                    } else {
                        1.0
                    };
                numerator[g] += weight * (h - previous_h[i]);
                denominator[g] += weight;
                risk[t][g] += 1.0;
            }
            previous_s[i] = s;
            previous_h[i] = h;
        }
        for g in 0..groups {
            survival[t][g] = if method == "ederer" {
                numerator[g] / denominator[g]
            } else {
                if denominator[g] > 0.0 {
                    cumulative[g] += numerator[g] / denominator[g];
                }
                (-cumulative[g]).exp()
            };
        }
    }
    let requested = times.unwrap_or(&grid);
    validate_finite(requested, "times")?;
    validate_non_negative(requested, "times")?;
    if requested.is_empty() || requested.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(SurvivalError::invalid_input(
            "times must be nonempty and increasing",
        ));
    }
    let mut surv = Vec::with_capacity(requested.len());
    let mut n_risk = Vec::with_capacity(requested.len());
    for &time in requested {
        let index = grid.partition_point(|&observed| observed <= time);
        surv.push(if index == 0 {
            vec![1.0; groups]
        } else {
            survival[index - 1].clone()
        });
        n_risk.push(if index == 0 {
            risk.first().cloned().unwrap_or_else(|| counts.clone())
        } else {
            risk[index - 1].clone()
        });
    }
    Ok(SurvExpResult {
        time: requested.to_vec(),
        surv,
        n_risk,
        method: if y.is_none() {
            "Ederer"
        } else if method == "conditional" {
            "conditional"
        } else {
            "cohort"
        }
        .into(),
    })
}

#[pyfunction(name = "survexp_cox")]
#[pyo3(signature=(curves, group, weights, y=None, times=None, method="ederer"))]
pub fn survexp_cox_py(
    py: Python<'_>,
    curves: Vec<CoxSurvfitCurve>,
    group: Vec<usize>,
    weights: Vec<f64>,
    y: Option<Vec<f64>>,
    times: Option<Vec<f64>>,
    method: &str,
) -> PyResult<SurvExpResult> {
    py.detach(|| {
        survexp_cox(
            &curves,
            &group,
            &weights,
            y.as_deref(),
            times.as_deref(),
            method,
        )
    })
    .map_err(Into::into)
}
