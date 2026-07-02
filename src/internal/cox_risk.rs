#[derive(Debug, Clone)]
pub(crate) struct CoxRiskSetData {
    pub(crate) cumsum_exp_eta: Vec<f64>,
    pub(crate) cumsum_weighted_x: Vec<f64>,
    pub(crate) cumsum_weighted_x_sq: Vec<f64>,
    pub(crate) risk_set_pos: Vec<usize>,
}

impl CoxRiskSetData {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            cumsum_exp_eta: Vec::with_capacity(n),
            cumsum_weighted_x: Vec::with_capacity(n * p),
            cumsum_weighted_x_sq: Vec::with_capacity(n * p),
            risk_set_pos: Vec::with_capacity(n),
        }
    }

    fn resize_for(&mut self, n: usize, p: usize) {
        self.cumsum_exp_eta.resize(n, 0.0);
        self.cumsum_weighted_x.resize(n * p, 0.0);
        self.cumsum_weighted_x_sq.resize(n * p, 0.0);
        self.risk_set_pos.resize(n, 0);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CoxRiskSetFirstOrderData {
    pub(crate) cumsum_exp_eta: Vec<f64>,
    pub(crate) cumsum_weighted_x: Vec<f64>,
    pub(crate) risk_set_pos: Vec<usize>,
}

impl CoxRiskSetFirstOrderData {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            cumsum_exp_eta: Vec::with_capacity(n),
            cumsum_weighted_x: Vec::with_capacity(n * p),
            risk_set_pos: Vec::with_capacity(n),
        }
    }

    fn resize_for(&mut self, n: usize, p: usize) {
        self.cumsum_exp_eta.resize(n, 0.0);
        self.cumsum_weighted_x.resize(n * p, 0.0);
        self.risk_set_pos.resize(n, 0);
    }
}

#[derive(Debug, Default)]
pub(crate) struct CoxRiskSetScratch {
    sorted_indices: Vec<usize>,
    running_x: Vec<f64>,
    running_x_sq: Vec<f64>,
}

impl CoxRiskSetScratch {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            sorted_indices: Vec::with_capacity(n),
            running_x: Vec::with_capacity(p),
            running_x_sq: Vec::with_capacity(p),
        }
    }

    fn reset_for(&mut self, n: usize, p: usize) {
        self.sorted_indices.clear();
        self.sorted_indices.extend(0..n);
        self.running_x.resize(p, 0.0);
        self.running_x.fill(0.0);
        self.running_x_sq.resize(p, 0.0);
        self.running_x_sq.fill(0.0);
    }
}

#[derive(Debug, Default)]
pub(crate) struct CoxRiskSetFirstOrderScratch {
    sorted_indices: Vec<usize>,
    running_x: Vec<f64>,
}

impl CoxRiskSetFirstOrderScratch {
    pub(crate) fn with_capacity(n: usize, p: usize) -> Self {
        Self {
            sorted_indices: Vec::with_capacity(n),
            running_x: Vec::with_capacity(p),
        }
    }

    fn reset_for(&mut self, n: usize, p: usize) {
        self.sorted_indices.clear();
        self.sorted_indices.extend(0..n);
        self.running_x.resize(p, 0.0);
        self.running_x.fill(0.0);
    }
}

pub(crate) fn cox_risk_shift(eta: &[f64], weights: &[f64]) -> f64 {
    let shift = eta
        .iter()
        .zip(weights.iter())
        .filter_map(|(&eta_i, &weight)| {
            if weight > 0.0 && eta_i.is_finite() {
                Some(eta_i)
            } else {
                None
            }
        })
        .fold(f64::NEG_INFINITY, f64::max);

    if shift.is_finite() { shift } else { 0.0 }
}

pub(crate) fn shifted_exp_eta(eta: &[f64], weights: &[f64]) -> Vec<f64> {
    let shift = cox_risk_shift(eta, weights);
    shifted_exp_eta_with_shift(eta, weights, shift)
}

pub(crate) fn shifted_exp_eta_with_shift(eta: &[f64], weights: &[f64], shift: f64) -> Vec<f64> {
    eta.iter()
        .zip(weights.iter())
        .map(|(&eta_i, &weight)| {
            if weight > 0.0 {
                (eta_i - shift).exp()
            } else {
                0.0
            }
        })
        .collect()
}

pub(crate) fn shifted_weighted_exp_eta_with_shift(
    eta: &[f64],
    weights: &[f64],
    shift: f64,
) -> Vec<f64> {
    eta.iter()
        .zip(weights.iter())
        .map(|(&eta_i, &weight)| {
            if weight > 0.0 {
                weight * (eta_i - shift).exp()
            } else {
                0.0
            }
        })
        .collect()
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn precompute_cox_risk_set_cumsum(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
) -> CoxRiskSetData {
    let mut risk_data = CoxRiskSetData::with_capacity(n, p);
    let mut scratch = CoxRiskSetScratch::with_capacity(n, p);
    precompute_cox_risk_set_cumsum_into(
        x,
        n,
        p,
        time,
        weights,
        exp_eta,
        &mut risk_data,
        &mut scratch,
    );
    risk_data
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn precompute_cox_risk_set_cumsum_into(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
    risk_data: &mut CoxRiskSetData,
    scratch: &mut CoxRiskSetScratch,
) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(time.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(exp_eta.len(), n);

    risk_data.resize_for(n, p);
    scratch.reset_for(n, p);
    scratch
        .sorted_indices
        .sort_by(|&a, &b| time[b].total_cmp(&time[a]).then_with(|| a.cmp(&b)));

    let mut running_exp = 0.0;
    for (pos, &idx) in scratch.sorted_indices.iter().enumerate() {
        let weighted_exp = weights[idx] * exp_eta[idx];
        running_exp += weighted_exp;

        for col in 0..p {
            let xij = x[idx * p + col];
            scratch.running_x[col] += weighted_exp * xij;
            scratch.running_x_sq[col] += weighted_exp * xij * xij;
        }

        risk_data.cumsum_exp_eta[pos] = running_exp;
        let row_start = pos * p;
        risk_data.cumsum_weighted_x[row_start..row_start + p].copy_from_slice(&scratch.running_x);
        risk_data.cumsum_weighted_x_sq[row_start..row_start + p]
            .copy_from_slice(&scratch.running_x_sq);
    }

    let mut start = 0;
    while start < n {
        let current_time = time[scratch.sorted_indices[start]];
        let mut end = start + 1;
        while end < n
            && crate::constants::same_time(time[scratch.sorted_indices[end]], current_time)
        {
            end += 1;
        }

        let pos = end - 1;
        for &idx in &scratch.sorted_indices[start..end] {
            risk_data.risk_set_pos[idx] = pos;
        }
        start = end;
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn precompute_cox_risk_set_first_order_cumsum_into(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
    risk_data: &mut CoxRiskSetFirstOrderData,
    scratch: &mut CoxRiskSetFirstOrderScratch,
) {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(time.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(exp_eta.len(), n);

    risk_data.resize_for(n, p);
    scratch.reset_for(n, p);
    scratch
        .sorted_indices
        .sort_by(|&a, &b| time[b].total_cmp(&time[a]).then_with(|| a.cmp(&b)));

    let mut running_exp = 0.0;
    for (pos, &idx) in scratch.sorted_indices.iter().enumerate() {
        let weighted_exp = weights[idx] * exp_eta[idx];
        running_exp += weighted_exp;

        for col in 0..p {
            scratch.running_x[col] += weighted_exp * x[idx * p + col];
        }

        risk_data.cumsum_exp_eta[pos] = running_exp;
        let row_start = pos * p;
        risk_data.cumsum_weighted_x[row_start..row_start + p].copy_from_slice(&scratch.running_x);
    }

    let mut start = 0;
    while start < n {
        let current_time = time[scratch.sorted_indices[start]];
        let mut end = start + 1;
        while end < n
            && crate::constants::same_time(time[scratch.sorted_indices[end]], current_time)
        {
            end += 1;
        }

        let pos = end - 1;
        for &idx in &scratch.sorted_indices[start..end] {
            risk_data.risk_set_pos[idx] = pos;
        }
        start = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cox_risk_shift_ignores_zero_weight_and_non_finite_values() {
        let eta = [1.0, 5.0, f64::INFINITY, 3.0];
        let weights = [1.0, 0.0, 1.0, 2.0];

        assert_eq!(cox_risk_shift(&eta, &weights), 3.0);
        assert_eq!(cox_risk_shift(&[f64::INFINITY], &[1.0]), 0.0);
        assert_eq!(cox_risk_shift(&[10.0], &[0.0]), 0.0);
    }

    #[test]
    fn shifted_exp_eta_skips_zero_weight_values() {
        let eta = [2.0, f64::INFINITY, 4.0];
        let weights = [1.0, 0.0, 0.0];

        let exp_eta = shifted_exp_eta(&eta, &weights);

        assert_eq!(exp_eta, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn shifted_weighted_exp_eta_applies_weights_after_shift() {
        let eta = [2.0, f64::INFINITY, 4.0];
        let weights = [2.0, 0.0, 3.0];

        let exp_eta = shifted_weighted_exp_eta_with_shift(&eta, &weights, 2.0);

        assert_eq!(exp_eta, vec![2.0, 0.0, 3.0 * (2.0_f64).exp()]);
    }

    #[test]
    fn precompute_cox_risk_set_cumsum_ignores_zero_weight_non_finite_eta() {
        let x = [1.0, 2.0];
        let time = [1.0, 2.0];
        let weights = [1.0, 0.0];
        let eta = [0.0, f64::INFINITY];
        let exp_eta = shifted_exp_eta(&eta, &weights);

        let risk_data = precompute_cox_risk_set_cumsum(&x, 2, 1, &time, &weights, &exp_eta);

        assert_eq!(risk_data.cumsum_exp_eta, vec![0.0, 1.0]);
        assert_eq!(risk_data.cumsum_weighted_x, vec![0.0, 1.0]);
        assert!(
            risk_data
                .cumsum_exp_eta
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            risk_data
                .cumsum_weighted_x
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn precompute_cox_risk_set_cumsum_groups_equal_times() {
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let time = [1.0, 2.0, 2.0];
        let weights = [1.0, 1.0, 1.0];
        let exp_eta = [1.0, 1.0, 1.0];

        let risk_data = precompute_cox_risk_set_cumsum(&x, 3, 2, &time, &weights, &exp_eta);

        assert_eq!(risk_data.cumsum_exp_eta, vec![1.0, 2.0, 3.0]);
        assert_eq!(risk_data.risk_set_pos, vec![2, 1, 1]);
        assert_eq!(risk_data.cumsum_weighted_x[2..4], [5.0, 50.0]);
        assert_eq!(risk_data.cumsum_weighted_x_sq[2..4], [13.0, 1300.0]);
    }

    #[test]
    fn precompute_cox_risk_set_first_order_cumsum_groups_equal_times() {
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let time = [1.0, 2.0, 2.0];
        let weights = [1.0, 1.0, 1.0];
        let exp_eta = [1.0, 1.0, 1.0];
        let mut risk_data = CoxRiskSetFirstOrderData::with_capacity(3, 2);
        let mut scratch = CoxRiskSetFirstOrderScratch::with_capacity(3, 2);

        precompute_cox_risk_set_first_order_cumsum_into(
            &x,
            3,
            2,
            &time,
            &weights,
            &exp_eta,
            &mut risk_data,
            &mut scratch,
        );

        assert_eq!(risk_data.cumsum_exp_eta, vec![1.0, 2.0, 3.0]);
        assert_eq!(risk_data.risk_set_pos, vec![2, 1, 1]);
        assert_eq!(risk_data.cumsum_weighted_x[2..4], [5.0, 50.0]);
    }

    #[test]
    fn precompute_cox_risk_set_cumsum_into_reuses_buffers() {
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let time = [1.0, 2.0, 2.0];
        let weights = [1.0, 1.0, 1.0];
        let exp_eta = [1.0, 1.0, 1.0];
        let mut risk_data = CoxRiskSetData::with_capacity(3, 2);
        let mut scratch = CoxRiskSetScratch::with_capacity(3, 2);

        precompute_cox_risk_set_cumsum_into(
            &x,
            3,
            2,
            &time,
            &weights,
            &exp_eta,
            &mut risk_data,
            &mut scratch,
        );

        assert_eq!(risk_data.cumsum_exp_eta, vec![1.0, 2.0, 3.0]);
        assert_eq!(risk_data.risk_set_pos, vec![2, 1, 1]);
        assert_eq!(risk_data.cumsum_weighted_x[2..4], [5.0, 50.0]);
        assert_eq!(risk_data.cumsum_weighted_x_sq[2..4], [13.0, 1300.0]);
    }
}
