use crate::core::strata_order::stratum_groups;
use crate::error::SurvivalResult;
use crate::internal::typed_inputs::{AndersenGillInput, CountingProcessData, Weights};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_non_negative,
};
use crate::regression::TieMethod;

pub fn agmart(input: &AndersenGillInput, method: TieMethod) -> SurvivalResult<Vec<f64>> {
    let n = input.counting.len();
    CountingProcessData::validate_parts(
        &input.counting.start,
        &input.counting.stop,
        &input.counting.event,
    )?;
    validate_binary_i32(&input.counting.event, "event")?;
    validate_length(n, input.score.len(), "score")?;
    validate_finite(&input.score, "score")?;
    validate_non_negative(&input.score, "score")?;
    if let Some(weights) = &input.weights {
        validate_length(n, weights.values.len(), "weights")?;
        Weights::validate_values(&weights.values)?;
    }
    if let Some(strata) = &input.strata {
        validate_length(n, strata.len(), "strata")?;
    }
    let weights = input.weights_or_unit_cow();
    let strata = input.strata_or_default_cow();
    if super::martingale_log::needs_log_space(&input.score, &weights) {
        return super::martingale_log::martingale_log(
            Some(&input.counting.start),
            &input.counting.stop,
            &input.counting.event,
            &input.score,
            &weights,
            &strata,
            method,
        );
    }
    Ok(agmart_rows(
        &input.counting.start,
        &input.counting.stop,
        &input.counting.event,
        &input.score,
        &weights,
        &strata,
        method,
    ))
}

pub(crate) fn agmart_rows(
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Vec<f64> {
    let n = start.len();

    let mut ignore = vec![true; n];
    for (_, rows) in stratum_groups(strata) {
        let mut event_times: Vec<f64> = rows
            .iter()
            .filter(|&&i| event[i] == 1)
            .map(|&i| stop[i])
            .collect();
        event_times.sort_by(f64::total_cmp);
        event_times.dedup();
        for &i in &rows {
            let below_start = event_times.partition_point(|&t| t <= start[i]);
            let below_stop = event_times.partition_point(|&t| t <= stop[i]);
            ignore[i] = below_start == below_stop;
        }
    }
    let nused = ignore.iter().filter(|&&flag| !flag).count();

    let mut sort_stop: Vec<usize> = (0..n).collect();
    sort_stop.sort_by(|&a, &b| {
        ignore[a]
            .cmp(&ignore[b])
            .then_with(|| strata[a].cmp(&strata[b]))
            .then_with(|| stop[b].total_cmp(&stop[a]))
    });
    let mut sort_start: Vec<usize> = (0..n).collect();
    sort_start.sort_by(|&a, &b| {
        ignore[a]
            .cmp(&ignore[b])
            .then_with(|| strata[a].cmp(&strata[b]))
            .then_with(|| start[b].total_cmp(&start[a]))
    });

    agmart3(
        nused,
        start,
        stop,
        event,
        score,
        weights,
        strata,
        &sort_start,
        &sort_stop,
        method,
    )
}

/// [`agmart_rows`] for rows whose (stratum, stop, row) order is known, as a
/// fit's sorted rows are.  Walking that order gives the used rows (those with
/// an event time of their stratum in `(start, stop]`) and their descending
/// stop order without a sort; only the start order is sorted, per stratum.
/// Both orders list just the used rows, the only ones [`agmart3`] reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agmart_rows_ordered(
    order: &[usize],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> Vec<f64> {
    let mut sort_stop = Vec::with_capacity(order.len());
    let mut sort_start = Vec::with_capacity(order.len());
    let mut first = 0;
    while first < order.len() {
        let label = strata[order[first]];
        let end = first + order[first..].partition_point(|&row| strata[row] == label);
        let rows = &order[first..end];
        let used = spans_an_event(rows, start, stop, event);
        // Descending stop as `total_cmp` orders it (+0.0 before -0.0), ties
        // in row order as a stable sort leaves them.
        let mut run_end = rows.len();
        while run_end > 0 {
            let bits = stop[rows[run_end - 1]].to_bits();
            let run_start = rows[..run_end]
                .iter()
                .rposition(|&row| stop[row].to_bits() != bits)
                .map_or(0, |position| position + 1);
            sort_stop.extend((run_start..run_end).filter(|&p| used[p]).map(|p| rows[p]));
            run_end = run_start;
        }
        let stratum_start = sort_start.len();
        sort_start.extend(
            rows.iter()
                .zip(&used)
                .filter(|(_, used)| **used)
                .map(|(&row, _)| row),
        );
        sort_start[stratum_start..]
            .sort_unstable_by(|&a, &b| start[b].total_cmp(&start[a]).then(a.cmp(&b)));
        first = end;
    }
    agmart3(
        sort_stop.len(),
        start,
        stop,
        event,
        score,
        weights,
        strata,
        &sort_start,
        &sort_stop,
        method,
    )
}

/// Whether each of a stratum's `rows`, in ascending stop order, has an event
/// time in `(start, stop]`: the latest event time up to its stop exceeds its
/// start.  Times compare as IEEE values, as the event-time search of
/// [`agmart_rows`] does.
fn spans_an_event(rows: &[usize], start: &[f64], stop: &[f64], event: &[i32]) -> Vec<bool> {
    let mut used = vec![false; rows.len()];
    let mut last_event = f64::NEG_INFINITY;
    let mut first = 0;
    while first < rows.len() {
        let time = stop[rows[first]];
        let end = rows[first..]
            .iter()
            .position(|&row| stop[row] != time)
            .map_or(rows.len(), |length| first + length);
        if rows[first..end].iter().any(|&row| event[row] == 1) {
            last_event = time;
        }
        for position in first..end {
            used[position] = start[rows[position]] < last_event;
        }
        first = end;
    }
    used
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn agmart3(
    nused: usize,
    tstart: &[f64],
    tstop: &[f64],
    event: &[i32],
    score: &[f64],
    weight: &[f64],
    strata: &[i32],
    sort1: &[usize],
    sort2: &[usize],
    method: TieMethod,
) -> Vec<f64> {
    let nr = tstart.len();
    let mut resid = vec![0.0; nr];
    let mut atrisk = vec![false; nr];
    if nused == 0 {
        return resid;
    }

    let mut person1 = 0;
    let mut denom = 0.0;
    let mut cumhaz = 0.0;
    let mut istrat = strata[sort2[0]];
    let mut person2 = 0;
    while person2 < nused {
        let mut dtime = 0.0;
        let mut k = person2;
        while k < nused {
            let p2 = sort2[k];
            if strata[p2] != istrat {
                while person1 < nused {
                    let p1 = sort1[person1];
                    if strata[p1] != istrat {
                        break;
                    }
                    resid[p1] -= cumhaz * score[p1];
                    person1 += 1;
                }
                cumhaz = 0.0;
                denom = 0.0;
                istrat = strata[p2];
                person2 = person1;
            }
            if event[p2] > 0 {
                dtime = tstop[p2];
                break;
            }
            k += 1;
        }
        if k == nused {
            break;
        }

        while person1 < nused {
            let p1 = sort1[person1];
            if tstart[p1] < dtime || strata[p1] != istrat {
                break;
            }
            if atrisk[p1] {
                denom -= score[p1] * weight[p1];
                resid[p1] -= cumhaz * score[p1];
            }
            person1 += 1;
        }

        let mut deaths = 0.0;
        let mut e_denom = 0.0;
        let mut wtsum = 0.0;
        k = person2;
        while k < nused {
            let p2 = sort2[k];
            if tstop[p2] < dtime || strata[p2] != istrat {
                break;
            }
            if event[p2] == 1 {
                atrisk[p2] = true;
                resid[p2] = 1.0 + cumhaz * score[p2];
                deaths += 1.0;
                denom += score[p2] * weight[p2];
                e_denom += score[p2] * weight[p2];
                wtsum += weight[p2];
            } else if tstart[p2] < dtime {
                denom += score[p2] * weight[p2];
                atrisk[p2] = true;
                resid[p2] = cumhaz * score[p2];
            }
            k += 1;
        }

        let hazard;
        if !method.is_efron() || deaths == 1.0 {
            hazard = wtsum / denom;
            person2 = k;
        } else {
            let mut total = 0.0;
            let mut e_hazard = 0.0;
            wtsum /= deaths;
            for i in 0..deaths as usize {
                let temp = i as f64 / deaths;
                total += wtsum / (denom - temp * e_denom);
                e_hazard += wtsum * (1.0 - temp) / (denom - temp * e_denom);
            }
            hazard = total;
            let temp = hazard - e_hazard;
            while person2 < k {
                let p2 = sort2[person2];
                if event[p2] > 0 {
                    resid[p2] += temp * score[p2];
                }
                person2 += 1;
            }
        }
        cumhaz += hazard;
    }

    while person1 < nused {
        let p1 = sort1[person1];
        if atrisk[p1] {
            resid[p1] -= cumhaz * score[p1];
        }
        person1 += 1;
    }
    resid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::typed_inputs::{CountingProcessData, CoxMartInput, SurvivalData, Weights};
    use crate::residuals::coxmart;

    fn ag_input(
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<i32>,
        score: Vec<f64>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
    ) -> AndersenGillInput {
        AndersenGillInput::try_new(
            CountingProcessData::try_new(start, stop, event).unwrap(),
            score,
            weights.map(|w| Weights::try_new(w).unwrap()),
            strata,
        )
        .unwrap()
    }

    #[test]
    fn zero_start_times_reduce_to_coxmart() {
        let time = vec![5.0, 3.0, 3.0, 8.0, 1.0, 3.0, 8.0];
        let status = vec![1, 1, 0, 1, 0, 1, 0];
        let score = vec![1.0, 0.5, 2.0, 1.5, 0.7, 1.1, 0.3];
        let weights = vec![1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 3.0];
        let strata = vec![1, 1, 1, 2, 2, 2, 2];
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            let right = coxmart(
                &CoxMartInput::try_new(
                    SurvivalData::try_new(time.clone(), status.clone()).unwrap(),
                    score.clone(),
                    Some(Weights::try_new(weights.clone()).unwrap()),
                    Some(strata.clone()),
                )
                .unwrap(),
                method,
            )
            .unwrap();
            let counting = agmart(
                &ag_input(
                    vec![0.0; 7],
                    time.clone(),
                    status.clone(),
                    score.clone(),
                    Some(weights.clone()),
                    Some(strata.clone()),
                ),
                method,
            )
            .unwrap();
            for (a, b) in right.iter().zip(&counting) {
                assert!((a - b).abs() < 1e-12, "{a} != {b} ({method:?})");
            }
        }
    }

    #[test]
    fn delayed_entry_subtracts_the_hazard_before_entry() {
        let resid = agmart(
            &ag_input(
                vec![0.0, 2.0, 0.0],
                vec![1.0, 4.0, 4.0],
                vec![1, 1, 0],
                vec![1.0; 3],
                None,
                None,
            ),
            TieMethod::Breslow,
        )
        .unwrap();
        let h1 = 1.0 / 2.0;
        let h4 = 1.0 / 2.0;
        assert!((resid[0] - (1.0 - h1)).abs() < 1e-12);
        assert!((resid[1] - (1.0 - h4)).abs() < 1e-12);
        assert!((resid[2] - (0.0 - h1 - h4)).abs() < 1e-12);
    }

    #[test]
    fn interval_without_events_has_zero_residual() {
        let resid = agmart(
            &ag_input(
                vec![0.0, 5.0, 0.0],
                vec![2.0, 9.0, 3.0],
                vec![1, 0, 1],
                vec![1.0, 1e300, 1.0],
                None,
                None,
            ),
            TieMethod::Efron,
        )
        .unwrap();
        assert_eq!(resid[1], 0.0);
        assert!((resid[0] - 0.5).abs() < 1e-12);
        assert!((resid[2] - (1.0 - 0.5 - 1.0)).abs() < 1e-12);
    }
}
