use crate::constants::TIME_EPSILON;
use std::collections::BTreeMap;

pub(super) struct StratifiedBaselineLookup {
    curves: BTreeMap<i32, (Vec<f64>, Vec<f64>)>,
}

impl StratifiedBaselineLookup {
    pub(super) fn from_components(times: &[f64], hazards: &[f64], strata: &[i32]) -> Self {
        let mut curves: BTreeMap<i32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for ((&time, &hazard), &stratum) in times.iter().zip(hazards.iter()).zip(strata.iter()) {
            let (curve_times, curve_hazards) = curves.entry(stratum).or_default();
            curve_times.push(time);
            curve_hazards.push(hazard);
        }
        Self { curves }
    }

    pub(super) fn cumulative_hazard_at(&self, stratum: i32, time: f64) -> f64 {
        let Some((times, hazards)) = self.curves.get(&stratum) else {
            return 0.0;
        };
        cumulative_step_at(times, hazards, time)
    }

    pub(super) fn times_for_strata(&self, strata: &[i32]) -> Vec<f64> {
        let mut times = Vec::new();
        for stratum in strata {
            if let Some((curve_times, _)) = self.curves.get(stratum) {
                times.extend(curve_times.iter().copied());
            }
        }
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);
        times
    }
}

#[derive(Clone, Copy)]
pub(super) struct CoxSweepRow {
    pub(super) original_idx: usize,
    pub(super) stop: f64,
    pub(super) entry: f64,
    pub(super) risk: f64,
    pub(super) weight: f64,
    pub(super) status: i32,
}

pub(super) struct ActiveRiskSet<'a> {
    rows: &'a [CoxSweepRow],
    use_entry_times: bool,
    entry_order: Vec<usize>,
    stop_order: Vec<usize>,
    active_rows: Vec<bool>,
    entry_pos: usize,
    stop_pos: usize,
    pub(super) risk_sum: f64,
}

impl<'a> ActiveRiskSet<'a> {
    pub(super) fn new(rows: &'a [CoxSweepRow], use_entry_times: bool) -> Self {
        let mut stop_order: Vec<usize> = (0..rows.len()).collect();
        stop_order.sort_by(|&lhs, &rhs| {
            rows[lhs]
                .stop
                .total_cmp(&rows[rhs].stop)
                .then_with(|| lhs.cmp(&rhs))
        });
        let entry_order = if use_entry_times {
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.sort_by(|&lhs, &rhs| {
                rows[lhs]
                    .entry
                    .total_cmp(&rows[rhs].entry)
                    .then_with(|| lhs.cmp(&rhs))
            });
            order
        } else {
            Vec::new()
        };
        let active_rows = if use_entry_times {
            vec![false; rows.len()]
        } else {
            Vec::new()
        };
        let risk_sum = if use_entry_times {
            0.0
        } else {
            rows.iter().map(|row| row.risk).sum()
        };

        Self {
            rows,
            use_entry_times,
            entry_order,
            stop_order,
            active_rows,
            entry_pos: 0,
            stop_pos: 0,
            risk_sum,
        }
    }

    pub(super) fn advance_to<Change>(&mut self, event_time: f64, mut on_change: Change)
    where
        Change: FnMut(usize, bool),
    {
        if self.use_entry_times {
            while self.entry_pos < self.entry_order.len()
                && self.rows[self.entry_order[self.entry_pos]].entry < event_time
            {
                let row_idx = self.entry_order[self.entry_pos];
                if !self.active_rows[row_idx] {
                    self.active_rows[row_idx] = true;
                    self.risk_sum += self.rows[row_idx].risk;
                    on_change(row_idx, true);
                }
                self.entry_pos += 1;
            }
            while self.stop_pos < self.stop_order.len()
                && self.rows[self.stop_order[self.stop_pos]].stop < event_time
            {
                let row_idx = self.stop_order[self.stop_pos];
                if self.active_rows[row_idx] {
                    self.active_rows[row_idx] = false;
                    self.risk_sum -= self.rows[row_idx].risk;
                    on_change(row_idx, false);
                }
                self.stop_pos += 1;
            }
        } else {
            while self.stop_pos < self.stop_order.len()
                && self.rows[self.stop_order[self.stop_pos]].stop < event_time
            {
                let row_idx = self.stop_order[self.stop_pos];
                self.risk_sum -= self.rows[row_idx].risk;
                on_change(row_idx, false);
                self.stop_pos += 1;
            }
        }
    }
}

pub(super) fn cumulative_step_at(times: &[f64], values: &[f64], time: f64) -> f64 {
    match times.binary_search_by(|probe| probe.total_cmp(&time)) {
        Ok(idx) => values[idx],
        Err(0) => 0.0,
        Err(idx) => values[idx - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_step_at_uses_last_prior_value() {
        let times = [1.0, 2.0, 4.0];
        let values = [0.2, 0.5, 0.9];

        assert_eq!(cumulative_step_at(&times, &values, 0.5), 0.0);
        assert_eq!(cumulative_step_at(&times, &values, 1.0), 0.2);
        assert_eq!(cumulative_step_at(&times, &values, 3.0), 0.5);
        assert_eq!(cumulative_step_at(&times, &values, 5.0), 0.9);
    }

    #[test]
    fn stratified_lookup_times_are_sorted_and_deduped() {
        let lookup = StratifiedBaselineLookup::from_components(
            &[2.0, 1.0 + TIME_EPSILON / 2.0, 1.0, 3.0],
            &[0.2, 0.15, 0.1, 0.3],
            &[0, 0, 0, 1],
        );

        assert_eq!(lookup.times_for_strata(&[0]), vec![1.0, 2.0]);
        assert_eq!(lookup.times_for_strata(&[1, 0]), vec![1.0, 2.0, 3.0]);
    }
}
