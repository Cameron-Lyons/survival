//! Formula-independent Fine–Gray preparation using the shared Kaplan–Meier
//! and interval-expansion kernels. Callers retain covariates by source row.

use std::collections::BTreeMap;

use crate::core::strata_order::validate_intervals;
use crate::data_prep::aeq_counting;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::validation::{validate_finite, validate_length};
use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, survfitkm};
use pyo3::prelude::*;

use super::finegray_data::{FineGrayOutput, finegray};

/// Prepared response and grouping. Status 0 is censoring, positive codes are
/// competing endpoints. Strata and subject codes need not be consecutive.
#[derive(Debug, Clone, Copy)]
pub struct FineGrayInput<'a> {
    pub time: &'a [f64],
    pub status: &'a [i32],
    pub event_type: i32,
    pub start: Option<&'a [f64]>,
    pub strata: Option<&'a [i32]>,
    pub id: Option<&'a [i32]>,
    /// As in R, weights multiply the expansion only, not the censoring fits.
    /// R indexes weights by row within each stratum; that convention is retained.
    pub weights: Option<&'a [f64]>,
    pub timefix: bool,
}

#[derive(Default)]
struct Curve {
    time: Vec<f64>,
    survival: Vec<f64>,
}

fn censoring_curves(
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    strata: Option<&[i32]>,
) -> SurvivalResult<Vec<Curve>> {
    let data = SurvfitKMData::try_new(
        Some(start),
        stop,
        event,
        None,
        strata.map(<[i32]>::to_vec),
        None,
        None,
    )?;
    let fit = survfitkm(
        &data,
        &SurvfitKMOptions {
            se_fit: false,
            timefix: false,
            ..Default::default()
        },
    )?;
    let counts = fit.strata.unwrap_or_else(|| vec![fit.time.len()]);
    let mut offset = 0;
    Ok(counts
        .into_iter()
        .map(|count| {
            let mut curve = Curve::default();
            for i in offset..offset + count {
                if fit.n_event[i] > 0.0 {
                    curve.time.push(fit.time[i]);
                    curve.survival.push(fit.surv[i]);
                }
            }
            offset += count;
            curve
        })
        .collect())
}

fn subject_layout(
    start: Option<&[f64]>,
    time: &[f64],
    status: &[i32],
    id: Option<&[i32]>,
) -> SurvivalResult<(Vec<bool>, Vec<bool>, bool)> {
    let n = time.len();
    let Some(start) = start else {
        return Ok((vec![false; n], vec![true; n], false));
    };
    let id =
        id.ok_or_else(|| SurvivalError::invalid_input("(start, stop] data requires a subject id"))?;
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| id[i].cmp(&id[j]).then_with(|| time[i].total_cmp(&time[j])));
    let mut first = vec![false; n];
    let mut last = vec![false; n];
    let minimum = time.iter().copied().fold(f64::INFINITY, f64::min);
    let mut delay = false;
    for (p, &row) in order.iter().enumerate() {
        if p == 0 || id[row] != id[order[p - 1]] {
            first[row] = true;
            delay |= start[row] > minimum;
        } else {
            let previous = order[p - 1];
            if status[previous] != 0 {
                return Err(SurvivalError::invalid_input(
                    "a subject has a transition before their last time point",
                ));
            }
            if time[previous] != start[row] {
                return Err(SurvivalError::invalid_input("a subject has gaps in time"));
            }
        }
        last[row] = p + 1 == n || id[row] != id[order[p + 1]];
    }
    Ok((first, last, delay))
}

/// Expand a multi-state response into Fine–Gray rows. The result's one-based
/// source rows refer to the full input. Output is grouped by ascending stratum,
/// then input row, then added interval. `wt` includes the optional user weights.
/// Near ties are resolved before subject checks when `timefix` is true.
pub fn finegray_expand(input: &FineGrayInput<'_>) -> SurvivalResult<FineGrayOutput> {
    let n = input.time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, input.status.len(), "status")?;
    validate_finite(input.time, "time")?;
    if input.event_type <= 0 || input.status.iter().any(|&v| v < 0) {
        return Err(SurvivalError::invalid_input(
            "status must be non-negative and event_type must be positive",
        ));
    }
    for (name, values) in [("strata", input.strata), ("id", input.id)] {
        if let Some(values) = values {
            validate_length(n, values.len(), name)?;
        }
    }
    if let Some(weights) = input.weights {
        validate_length(n, weights.len(), "weights")?;
        validate_finite(weights, "weights")?;
    }
    if let Some(start) = input.start {
        validate_length(n, start.len(), "start")?;
        validate_finite(start, "start")?;
        validate_intervals(start, input.time)?;
    }
    let (start, time) = if input.timefix {
        aeq_counting(input.start, input.time)?
    } else {
        (input.start.map(<[f64]>::to_vec), input.time.to_vec())
    };
    let (first, last, delay) = subject_layout(start.as_deref(), &time, input.status, input.id)?;
    if !input.status.contains(&input.event_type) {
        return Err(SurvivalError::invalid_input(
            "selected endpoint has no events",
        ));
    }
    let start = start.unwrap_or_else(|| {
        let minimum = time.iter().copied().fold(f64::INFINITY, f64::min);
        vec![
            if minimum > 0.0 {
                0.0
            } else {
                2.0 * minimum - 1.0
            };
            n
        ]
    });
    let mut utime: Vec<_> = start.iter().chain(&time).copied().collect();
    utime.sort_by(f64::total_cmp);
    utime.dedup();
    let rank1: Vec<_> = start
        .iter()
        .map(|&t| utime.partition_point(|&v| v <= t) as f64)
        .collect();
    let rank2: Vec<_> = time
        .iter()
        .zip(input.status)
        .map(|(&t, &status)| {
            utime.partition_point(|&v| v <= t) as f64 - if status == 0 { 0.0 } else { 0.2 }
        })
        .collect();
    let entry = if delay {
        Some(censoring_curves(
            rank2.iter().map(|&t| -t).collect(),
            rank1.iter().map(|&t| -t).collect(),
            first.iter().map(|&v| i32::from(v)).collect(),
            input.strata,
        )?)
    } else {
        None
    };
    let censoring = censoring_curves(
        rank1,
        rank2,
        last.iter()
            .zip(input.status)
            .map(|(&last, &status)| i32::from(last && status == 0))
            .collect(),
        input.strata,
    )?;
    let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for row in 0..n {
        groups
            .entry(input.strata.map_or(0, |s| s[row]))
            .or_default()
            .push(row);
    }
    let mut output = FineGrayOutput {
        row: Vec::new(),
        start: Vec::new(),
        end: Vec::new(),
        wt: Vec::new(),
        add: Vec::new(),
    };
    for (group, rows) in groups.values().enumerate() {
        if !rows.iter().any(|&i| input.status[i] == input.event_type) {
            continue;
        }
        let curve = &censoring[group];
        let (mut ctime, probabilities) = if let Some(entry) = &entry {
            let entry = &entry[group];
            let dtime: Vec<_> = entry.time.iter().rev().map(|&t| -t).collect();
            let dprob: Vec<_> = entry
                .survival
                .iter()
                .rev()
                .skip(1)
                .copied()
                .chain([1.0])
                .collect();
            let mut combined: Vec<_> = dtime.iter().chain(&curve.time).copied().collect();
            combined.sort_by(f64::total_cmp);
            combined.dedup();
            let probabilities = combined
                .iter()
                .map(|&t| {
                    let d = dtime.partition_point(|&v| v <= t).saturating_sub(1);
                    let g = curve.time.partition_point(|&v| v <= t);
                    dprob[d] * if g == 0 { 1.0 } else { curve.survival[g - 1] }
                })
                .collect::<Vec<_>>();
            (
                combined
                    .iter()
                    .map(|&t| utime[t as usize - 1])
                    .collect::<Vec<_>>(),
                probabilities,
            )
        } else {
            (
                curve.time.iter().map(|&t| utime[t as usize - 1]).collect(),
                curve.survival.clone(),
            )
        };
        ctime.push(
            rows.iter()
                .map(|&i| time[i])
                .fold(f64::NEG_INFINITY, f64::max),
        );
        let cprob: Vec<_> = [1.0].into_iter().chain(probabilities).collect();
        let mut keep = vec![false; ctime.len()];
        keep[0] = true;
        for &row in rows {
            if input.status[row] == input.event_type {
                let cut = ctime.partition_point(|&t| t < time[row]);
                if cut < keep.len() {
                    keep[cut] = true;
                }
            }
        }
        let mut split = finegray(
            &rows.iter().map(|&i| start[i]).collect::<Vec<_>>(),
            &rows.iter().map(|&i| time[i]).collect::<Vec<_>>(),
            &ctime,
            &cprob,
            &rows
                .iter()
                .map(|&i| input.status[i] != 0 && input.status[i] != input.event_type && last[i])
                .collect::<Vec<_>>(),
            &keep,
        )?;
        for (row, weight) in split.row.iter_mut().zip(&mut split.wt) {
            if let Some(weights) = input.weights {
                *weight *= weights[*row - 1];
            }
            *row = rows[*row - 1] + 1;
        }
        output.row.append(&mut split.row);
        output.start.append(&mut split.start);
        output.end.append(&mut split.end);
        output.wt.append(&mut split.wt);
        output.add.append(&mut split.add);
    }
    Ok(output)
}

/// Shared numerical preparation for R/Python formula interfaces. Integer id
/// and strata codes are arbitrary. Covariates remain in the calling language.
#[pyfunction(name = "finegray_expand")]
#[pyo3(signature = (time, status, event_type=1, start=None, strata=None, id=None, weights=None, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn finegray_expand_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    event_type: i32,
    start: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<IntVec>,
    weights: Option<FloatVec>,
    timefix: bool,
) -> PyResult<FineGrayOutput> {
    Ok(py.detach(|| {
        finegray_expand(&FineGrayInput {
            time: &time,
            status: &status,
            event_type,
            start: start.as_deref(),
            strata: strata.as_deref(),
            id: id.as_deref(),
            weights: weights.as_deref(),
            timefix,
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(time: &'a [f64], status: &'a [i32]) -> FineGrayInput<'a> {
        FineGrayInput {
            time,
            status,
            event_type: 1,
            start: None,
            strata: None,
            id: None,
            weights: None,
            timefix: true,
        }
    }

    #[test]
    fn prepared_expansion_matches_hand_censoring_weights() {
        let output = finegray_expand(&input(
            &[1., 2., 2., 3., 4., 5., 6., 7.],
            &[1, 2, 0, 1, 2, 0, 1, 2],
        ))
        .unwrap();
        assert_eq!(output.row, vec![1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8]);
        assert_eq!(
            output.start,
            vec![0., 0., 2., 5., 0., 0., 0., 5., 0., 0., 0.]
        );
        assert_eq!(output.end, vec![1., 2., 5., 7., 2., 3., 5., 7., 5., 6., 7.]);
        for (&value, expected) in
            output
                .wt
                .iter()
                .zip([1., 1., 5. / 6., 5. / 9., 1., 1., 1., 2. / 3., 1., 1., 1.])
        {
            assert!((value - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn arbitrary_strata_keep_source_rows_and_r_weight_convention() {
        let mut input = input(&[1., 2., 2., 3., 4., 5., 6., 7.], &[1, 2, 0, 1, 2, 0, 1, 2]);
        input.strata = Some(&[-17, 80, -17, 80, -17, 80, -17, 80]);
        input.weights = Some(&[1., 2., 1., 2., 1., 2., 1., 2.]);
        input.event_type = 2;
        let output = finegray_expand(&input).unwrap();
        assert_eq!(output.row, vec![1, 1, 3, 5, 7, 2, 4, 4, 6, 8]);
        for (&value, expected) in
            output
                .wt
                .iter()
                .zip([1., 2. / 3., 2., 1., 2., 1., 2., 1., 1., 2.])
        {
            assert!((value - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn public_preparation_validates_response_and_subjects() {
        assert!(finegray_expand(&input(&[], &[])).is_err());
        assert!(finegray_expand(&input(&[1.], &[])).is_err());
        assert!(finegray_expand(&input(&[f64::NAN], &[1])).is_err());
        assert!(finegray_expand(&input(&[1.], &[-1])).is_err());
        assert!(
            finegray_expand(&input(&[1.], &[2]))
                .unwrap_err()
                .to_string()
                .contains("no events")
        );
        let mut data = input(&[1., 3.], &[0, 1]);
        data.start = Some(&[0., 2.]);
        assert!(
            finegray_expand(&data)
                .unwrap_err()
                .to_string()
                .contains("subject id")
        );
        data.id = Some(&[1, 1]);
        assert!(
            finegray_expand(&data)
                .unwrap_err()
                .to_string()
                .contains("gaps in time")
        );
        data.status = &[1, 1];
        assert!(
            finegray_expand(&data)
                .unwrap_err()
                .to_string()
                .contains("transition before")
        );
    }
}
