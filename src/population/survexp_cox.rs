//! Expected population survival from fitted Cox curves (`survexp.cfit`).
use super::survexp::SurvExpResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use crate::regression::CoxSurvfitCurve;
use pyo3::prelude::*;

/// One event-time baseline, on the same centering scale as the population risks.
/// An empty baseline represents a stratum without observed events.
pub struct CoxExpectedBaseline {
    pub time: Vec<f64>,
    pub cumhaz: Vec<f64>,
}

/// Aggregate expected Cox curves without allocating one curve per subject.
/// Baselines are selected by zero-based `strata`; output columns by `group`.
/// Storage is O(subjects + baseline points + output times * groups).
#[allow(clippy::too_many_arguments)]
pub fn survexp_cox_prepared(
    baselines: &[CoxExpectedBaseline],
    risk: &[f64],
    strata: &[usize],
    group: &[usize],
    weights: &[f64],
    y: Option<&[f64]>,
    times: Option<&[f64]>,
    method: &str,
) -> SurvivalResult<SurvExpResult> {
    let n = risk.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input("Data set has 0 rows"));
    }
    validate_finite(risk, "risk")?;
    validate_non_negative(risk, "risk")?;
    validate_length(n, strata.len(), "strata")?;
    validate_length(n, group.len(), "group")?;
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
    if let Some(times) = times {
        validate_finite(times, "times")?;
        validate_non_negative(times, "times")?;
        if times.is_empty() || times.windows(2).any(|w| w[1] < w[0]) {
            return Err(SurvivalError::invalid_input(
                "times must be nonempty and increasing",
            ));
        }
    }
    let groups = group
        .iter()
        .max()
        .unwrap()
        .checked_add(1)
        .filter(|&v| v <= n)
        .ok_or_else(|| SurvivalError::invalid_input("group codes must be contiguous"))?;
    let mut totals = vec![0.0; groups];
    let mut counts = vec![0.0; groups];
    let mut used = vec![false; baselines.len()];
    for i in 0..n {
        let present = used
            .get_mut(strata[i])
            .ok_or_else(|| SurvivalError::invalid_input("stratum has no baseline"))?;
        *present = true;
        totals[group[i]] += weights[i];
        counts[group[i]] += 1.0;
    }
    if totals.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "every group must have positive finite total weight",
        ));
    }
    let mut grid = Vec::new();
    for (baseline, &used) in baselines.iter().zip(&used) {
        validate_length(
            baseline.time.len(),
            baseline.cumhaz.len(),
            "baseline cumulative hazard",
        )?;
        validate_finite(&baseline.time, "baseline time")?;
        validate_finite(&baseline.cumhaz, "baseline cumulative hazard")?;
        validate_non_negative(&baseline.cumhaz, "baseline cumulative hazard")?;
        if baseline.time.windows(2).any(|w| w[1] <= w[0])
            || baseline.cumhaz.windows(2).any(|w| w[1] < w[0])
        {
            return Err(SurvivalError::invalid_input(
                "baseline times must increase and cumulative hazards must not decrease",
            ));
        }
        if used {
            grid.extend_from_slice(&baseline.time);
        }
    }
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let requested = times.unwrap_or(&grid);
    let mut surv = Vec::with_capacity(requested.len());
    let mut n_risk = Vec::with_capacity(requested.len());
    let mut positions = vec![0; baselines.len()];
    let mut current_h = vec![0.0; baselines.len()];
    let mut previous_h = current_h.clone();
    let mut current_s = vec![1.0; baselines.len()];
    let mut previous_s = current_s.clone();
    let mut cumulative = vec![0.0; groups];
    let mut survival = vec![1.0; groups];
    let mut at_risk = counts.clone();
    let mut numerator = vec![0.0; groups];
    let mut denominator = vec![0.0; groups];
    let mut output = 0;
    for &time in &grid {
        if output == requested.len() {
            break;
        }
        previous_h.copy_from_slice(&current_h);
        previous_s.copy_from_slice(&current_s);
        for (s, baseline) in baselines.iter().enumerate() {
            while positions[s] < baseline.time.len() && baseline.time[positions[s]] <= time {
                current_h[s] = baseline.cumhaz[positions[s]];
                positions[s] += 1;
            }
            if method != "conditional" {
                current_s[s] = (-current_h[s]).exp();
            }
        }
        numerator.fill(0.0);
        denominator.fill(0.0);
        at_risk.fill(0.0);
        for i in 0..n {
            let g = group[i];
            let s = strata[i];
            if method == "ederer" {
                numerator[g] += weights[i] * current_s[s].powf(risk[i]);
                denominator[g] += weights[i];
                at_risk[g] += 1.0;
            } else if y.unwrap()[i] >= time {
                let weight = weights[i]
                    * if method == "hakulinen" {
                        previous_s[s].powf(risk[i])
                    } else {
                        1.0
                    };
                // Subtract subject hazards as in expanded curves, preserving
                // rounding and the reference's exp(-H).powf(risk) convention.
                numerator[g] += weight * (current_h[s] * risk[i] - previous_h[s] * risk[i]);
                denominator[g] += weight;
                at_risk[g] += 1.0;
            }
        }
        // Before the first event, R uses that event's risk counts and survival 1.
        while output < requested.len() && requested[output] < time {
            surv.push(survival.clone());
            n_risk.push(if grid[0] == time {
                at_risk.clone()
            } else {
                counts.clone()
            });
            output += 1;
        }
        for g in 0..groups {
            survival[g] = if method == "ederer" {
                numerator[g] / denominator[g]
            } else {
                cumulative[g] += numerator[g] / denominator[g];
                (-cumulative[g]).exp()
            };
        }
        counts.copy_from_slice(&at_risk);
        while output < requested.len() && requested[output] == time {
            surv.push(survival.clone());
            n_risk.push(at_risk.clone());
            output += 1;
        }
    }
    while output < requested.len() {
        surv.push(survival.clone());
        n_risk.push(counts.clone());
        output += 1;
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

/// Numeric entry point for baselines prepared outside a fitted Rust model.
#[pyfunction(name = "survexp_cox_prepared")]
#[pyo3(signature=(time, cumhaz, lengths, risk, strata, group, weights, y=None, times=None, method="ederer"))]
#[allow(clippy::too_many_arguments)]
pub fn survexp_cox_prepared_py(
    py: Python<'_>,
    time: FloatVec,
    cumhaz: FloatVec,
    lengths: IntVec,
    risk: FloatVec,
    strata: IntVec,
    group: IntVec,
    weights: FloatVec,
    y: Option<FloatVec>,
    times: Option<FloatVec>,
    method: &str,
) -> PyResult<SurvExpResult> {
    let codes = |values: &[i32]| -> SurvivalResult<Vec<usize>> {
        values
            .iter()
            .map(|&v| {
                usize::try_from(v).map_err(|_| {
                    SurvivalError::invalid_input("codes and lengths must be nonnegative")
                })
            })
            .collect()
    };
    let (lengths, strata, group) = (codes(&lengths)?, codes(&strata)?, codes(&group)?);
    Ok(py.detach(|| {
        validate_length(time.len(), cumhaz.len(), "cumhaz")?;
        let mut baselines = Vec::with_capacity(lengths.len());
        let mut start: usize = 0;
        for length in lengths {
            let end = start
                .checked_add(length)
                .filter(|&v| v <= time.len())
                .ok_or_else(|| {
                    SurvivalError::invalid_input("baseline lengths do not match time")
                })?;
            baselines.push(CoxExpectedBaseline {
                time: time[start..end].to_vec(),
                cumhaz: cumhaz[start..end].to_vec(),
            });
            start = end;
        }
        validate_length(time.len(), start, "baseline lengths")?;
        survexp_cox_prepared(
            &baselines,
            &risk,
            &strata,
            &group,
            &weights,
            y.as_deref(),
            times.as_deref(),
            method,
        )
    })?)
}

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
                // R's `hazard %*% tmat / colSums(tmat)`: once nobody in the
                // group is at risk this is 0/0, and the NaN carries through
                // the cumulative sum.
                cumulative[g] += numerator[g] / denominator[g];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_curves_match_expansion_with_interleaved_strata() {
        let baselines = vec![
            CoxExpectedBaseline {
                time: vec![1., 3., 4.],
                cumhaz: vec![0.1, 0.5, 0.8],
            },
            CoxExpectedBaseline {
                time: vec![2., 3., 5.],
                cumhaz: vec![0.2, 0.6, 1.2],
            },
            // No population member uses this stratum, so its time is not output.
            CoxExpectedBaseline {
                time: vec![1.5],
                cumhaz: vec![0.3],
            },
        ];
        let risk = [0.5, 1.2, 2., 0., 3.];
        let strata = [1, 0, 1, 0, 1];
        let group = [0, 1, 1, 0, 1];
        let weights = [0.5, 2., 0., 1., 3.];
        let y = [4., 5., 0., 3., 5.];
        let expanded = risk
            .iter()
            .zip(strata)
            .map(|(&risk, stratum)| {
                let b = &baselines[stratum];
                CoxSurvfitCurve {
                    stratum: stratum as i32,
                    n: 1,
                    time: b.time.clone(),
                    n_risk: vec![],
                    n_event: vec![],
                    n_censor: vec![],
                    surv: b
                        .cumhaz
                        .iter()
                        .map(|&h| vec![(-h).exp().powf(risk)])
                        .collect(),
                    cumhaz: b.cumhaz.iter().map(|&h| vec![h * risk]).collect(),
                    std_err: None,
                }
            })
            .collect::<Vec<_>>();
        let requested = [0., 0.5, 1., 1., 1.5, 2., 3., 5., 8.];
        for method in ["ederer", "hakulinen", "conditional"] {
            for times in [None, Some(requested.as_slice())] {
                let actual = survexp_cox_prepared(
                    &baselines,
                    &risk,
                    &strata,
                    &group,
                    &weights,
                    Some(&y),
                    times,
                    method,
                )
                .unwrap();
                let expected =
                    survexp_cox(&expanded, &group, &weights, Some(&y), times, method).unwrap();
                assert_eq!(actual.time, expected.time);
                assert_eq!(actual.n_risk, expected.n_risk);
                for (actual, expected) in actual
                    .surv
                    .iter()
                    .flatten()
                    .zip(expected.surv.iter().flatten())
                {
                    assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
                }
            }
        }
    }

    #[test]
    fn empty_baseline_keeps_survival_one_and_invalid_inputs_fail() {
        let baselines = [CoxExpectedBaseline {
            time: vec![],
            cumhaz: vec![],
        }];
        let out = survexp_cox_prepared(
            &baselines,
            &[1., 2.],
            &[0, 0],
            &[0, 0],
            &[1., 1.],
            None,
            Some(&[0., 1.]),
            "ederer",
        )
        .unwrap();
        assert_eq!(out.surv, vec![vec![1.], vec![1.]]);
        assert_eq!(out.n_risk, vec![vec![2.], vec![2.]]);
        for (risk, strata, group, weights) in [
            (vec![f64::NAN], vec![0], vec![0], vec![1.]),
            (vec![1.], vec![1], vec![0], vec![1.]),
            (vec![1.], vec![0], vec![usize::MAX], vec![1.]),
            (vec![1.], vec![0], vec![0], vec![0.]),
        ] {
            assert!(
                survexp_cox_prepared(
                    &baselines,
                    &risk,
                    &strata,
                    &group,
                    &weights,
                    None,
                    Some(&[1.]),
                    "ederer"
                )
                .is_err()
            );
        }
    }

    fn curve(cumhaz: [[f64; 2]; 3]) -> CoxSurvfitCurve {
        CoxSurvfitCurve {
            stratum: 0,
            n: 2,
            time: vec![1.0, 2.0, 3.0],
            n_risk: vec![2.0, 1.0, 1.0],
            n_event: vec![1.0, 0.0, 1.0],
            n_censor: vec![0.0, 1.0, 0.0],
            surv: cumhaz
                .iter()
                .map(|row| row.iter().map(|h| (-h).exp()).collect())
                .collect(),
            cumhaz: cumhaz.iter().map(|row| row.to_vec()).collect(),
            std_err: None,
        }
    }

    #[test]
    fn a_group_with_nobody_at_risk_turns_missing_like_r() {
        // Subject 1 (group 0) leaves at time 1, subject 2 (group 1) stays.
        let curves = [curve([[0.1, 0.2], [0.3, 0.4], [0.6, 0.9]])];
        for method in ["conditional", "hakulinen"] {
            let out = survexp_cox(
                &curves,
                &[0, 1],
                &[1.0, 1.0],
                Some(&[1.0, 3.0]),
                None,
                method,
            )
            .unwrap();
            assert!((out.surv[0][0] - (-0.1f64).exp()).abs() < 1e-15);
            assert!(out.surv[1][0].is_nan() && out.surv[2][0].is_nan());
            let group1: Vec<f64> = out.surv.iter().map(|row| row[1]).collect();
            for (value, expected) in group1.iter().zip([0.2f64, 0.4, 0.9]) {
                assert!((value - (-expected).exp()).abs() < 1e-15, "{method}");
            }
        }
    }
}
