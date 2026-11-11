//! With every row entered before the earliest stop, risk sets are nested.
//! Build their moments by positive additions in reverse time, then emit the
//! recorded times in ascending order. No large risk is ever subtracted to
//! expose a smaller one, and no risk-scale or precision guard is relaxed.

use super::{LogMoments, LogRiskSweep, LogRiskTime, LogScale};
use crate::error::{SurvivalError, SurvivalResult};

pub(crate) fn has_common_entry(rows: &[usize], stop: &[f64], entry: Option<&[f64]>) -> bool {
    entry.is_none_or(|entry| {
        let first_stop = rows
            .iter()
            .map(|&row| stop[row])
            .fold(f64::INFINITY, f64::min);
        rows.iter().all(|&row| entry[row] < first_stop)
    })
}

struct RiskTime {
    time: f64,
    rows: Vec<usize>,
    deaths: Vec<usize>,
    nrisk: usize,
    weight: LogScale,
    risk: LogMoments,
    nondeath: LogMoments,
    death: LogMoments,
    death_weight: LogScale,
}

impl LogRiskSweep<'_> {
    pub(super) fn walk_common_entry(
        &self,
        order: &[usize],
        log_risk: &[LogScale],
        mut callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        let p = self.x.ncols();
        let mut risk = LogMoments::new(p);
        let mut weight = LogScale::EMPTY;
        let mut records = Vec::new();
        let mut covariates = vec![0.0; p];
        let mut end = order.len();
        while end > 0 {
            let time = self.stop[self.rows[order[end - 1]]];
            let mut first = end - 1;
            while first > 0 && self.stop[self.rows[order[first - 1]]] == time {
                first -= 1;
            }
            let rows: Vec<_> = order[first..end].iter().map(|&i| self.rows[i]).collect();
            let mut nondeath = risk;
            let mut death = LogMoments::new(p);
            let mut death_weight = LogScale::EMPTY;
            let mut deaths = Vec::new();
            for (&position, &row) in order[first..end].iter().zip(&rows) {
                for (out, &value) in covariates.iter_mut().zip(self.x.row(row)) {
                    *out = value;
                }
                let row_weight = LogScale::from_log(self.weights[row].ln());
                weight = weight.mix(row_weight).0;
                if self.status[row] == 1 {
                    deaths.push(row);
                    death.add(log_risk[position], &covariates);
                    death_weight = death_weight.mix(row_weight).0;
                } else {
                    nondeath.add(log_risk[position], &covariates);
                }
            }
            risk = nondeath.clone();
            risk.add(death.weight, &death.mean);
            records.push(RiskTime {
                time,
                rows,
                deaths,
                nrisk: order.len() - first,
                weight,
                risk: risk.clone(),
                nondeath,
                death,
                death_weight,
            });
            end = first;
        }
        for record in records.iter().rev() {
            if record.risk.weight.is_zero() && !record.death_weight.is_zero() {
                return Err(SurvivalError::invalid_input(
                    "positive-weight events require a positive risk-set sum",
                ));
            }
            callback(LogRiskTime {
                time: record.time,
                rows_at_time: &record.rows,
                deaths: &record.deaths,
                nrisk: record.nrisk,
                risk_weight: record.weight.exp_at(0.0),
                denom: record.risk.weight,
                mean: &record.risk.mean,
                death_risk: record.death.weight,
                nondeath_risk: record.nondeath.weight,
                nondeath_mean: &record.nondeath.mean,
                death_mean: &record.death.mean,
                death_weight: record.death_weight.exp_at(0.0),
                log_death_weight: record.death_weight.log_value(),
                covariance: None,
                nondeath_covariance: None,
                death_covariance: None,
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[derive(Debug)]
    struct Snapshot {
        time: f64,
        rows: Vec<usize>,
        deaths: Vec<usize>,
        nrisk: usize,
        weights: [f64; 2],
        risks: [LogScale; 3],
        means: Vec<f64>,
    }

    fn collect(sweep: &LogRiskSweep<'_>, scales: &[LogScale], covariance: bool) -> Vec<Snapshot> {
        let mut out = Vec::new();
        sweep
            .walk(Some(scales), covariance, |event| {
                let mut means = event.mean.to_vec();
                means.extend_from_slice(event.nondeath_mean);
                means.extend_from_slice(event.death_mean);
                out.push(Snapshot {
                    time: event.time,
                    rows: event.rows_at_time.to_vec(),
                    deaths: event.deaths.to_vec(),
                    nrisk: event.nrisk,
                    weights: [event.risk_weight, event.death_weight],
                    risks: [event.denom, event.nondeath_risk, event.death_risk],
                    means,
                });
                Ok(())
            })
            .unwrap();
        out
    }

    #[test]
    fn reverse_positive_moments_match_tree_risk_sets_and_callback_order() {
        let stop = [1., 2., 2., 2., 4., 5., 5., 7.];
        let entry = [0.; 8];
        let status = [0, 1, 1, 0, 1, 0, 1, 0];
        let weights = [1., 0.5, 0., 0., 1.5, 3., 1., 2.];
        let x = array![
            [1e16, -1.],
            [1e16, 0.5],
            [1e16, 1.],
            [1e16, -0.2],
            [1e16, 0.7],
            [1e16, -0.5],
            [1e16, 0.1],
            [1e16, 1.5]
        ];
        let rows = [5, 2, 7, 0, 4, 1, 6, 3];
        for counting in [false, true] {
            for compensated in [false, true] {
                let scales: Vec<_> = (0..8)
                    .map(|row| {
                        if compensated {
                            LogScale::new_parts(1e308, 1e292, row as f64 * 0.1)
                        } else {
                            LogScale::from_log(if row < 3 { 1000. } else { -1000. })
                        }
                    })
                    .collect();
                let sweep = LogRiskSweep {
                    stop: &stop,
                    entry: counting.then_some(entry.as_slice()),
                    status: &status,
                    x: x.view(),
                    weights: &weights,
                    log_risk: &[],
                    rows: &rows,
                };
                let common = collect(&sweep, &scales, false);
                let tree = collect(&sweep, &scales, true);
                assert_eq!(common.len(), tree.len());
                for (a, b) in common.iter().zip(&tree) {
                    assert_eq!(
                        (a.time, &a.rows, &a.deaths, a.nrisk),
                        (b.time, &b.rows, &b.deaths, b.nrisk)
                    );
                    for (left, right) in a.weights.iter().zip(&b.weights) {
                        assert!((left - right).abs() <= 2e-13 * (1. + right.abs()));
                    }
                    for (&left, &right) in a.risks.iter().zip(&b.risks) {
                        assert_eq!(left.is_zero(), right.is_zero());
                        if !left.is_zero() {
                            assert!(left.ratio_log(right).abs() < 2e-13);
                        }
                    }
                    for (left, right) in a.means.iter().zip(&b.means) {
                        assert!((left - right).abs() <= 2e-13 * (1. + right.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn common_entry_requires_entry_before_censor_times_too() {
        let rows = [0, 1, 2];
        let stop = [1., 2., 3.];
        assert!(has_common_entry(&rows, &stop, None));
        assert!(has_common_entry(&rows, &stop, Some(&[-2., 0., 0.5])));
        assert!(!has_common_entry(&rows, &stop, Some(&[0., 1., 0.])));
        assert!(!has_common_entry(&rows, &stop, Some(&[0., 1.5, 0.])));
        let entry = [0., 1.5, 0.];
        let sweep = LogRiskSweep {
            stop: &stop,
            entry: Some(&entry),
            status: &[0, 1, 0],
            x: ndarray::ArrayView2::from_shape((3, 0), &[]).unwrap(),
            weights: &[1.; 3],
            log_risk: &[0.; 3],
            rows: &rows,
        };
        let events = collect(&sweep, &[LogScale::from_log(0.); 3], false);
        assert_eq!(events[0].time, 1.);
        assert_eq!(events[0].nrisk, 2);
        assert_eq!(events[0].weights[0], 2.);
    }

    #[test]
    fn common_entry_retains_risk_validation_and_zero_weight_semantics() {
        let x = array![[0.], [1.]];
        for weights in [[0., 0.], [1., 0.], [-1., 1.]] {
            for origin in [0., f64::NAN, f64::INFINITY] {
                let sweep = LogRiskSweep {
                    stop: &[1., 2.],
                    entry: None,
                    status: &[1, 0],
                    x: x.view(),
                    weights: &weights,
                    log_risk: &[],
                    rows: &[0, 1],
                };
                let scales = [LogScale::from_log(origin); 2];
                let common = sweep.walk(Some(&scales), false, |_| Ok(()));
                let tree = sweep.walk(Some(&scales), true, |_| Ok(()));
                assert_eq!(common.is_ok(), tree.is_ok());
            }
        }
    }
}
