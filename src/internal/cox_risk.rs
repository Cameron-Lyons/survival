#[derive(Debug, Clone)]
pub(crate) struct CoxRiskSetData {
    pub(crate) cumsum_exp_eta: Vec<f64>,
    pub(crate) cumsum_weighted_x: Vec<f64>,
    pub(crate) cumsum_weighted_x_sq: Vec<f64>,
    pub(crate) risk_set_pos: Vec<usize>,
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

#[allow(clippy::needless_range_loop)]
pub(crate) fn precompute_cox_risk_set_cumsum(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
) -> CoxRiskSetData {
    debug_assert_eq!(x.len(), n * p);
    debug_assert_eq!(time.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert_eq!(exp_eta.len(), n);

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].total_cmp(&time[a]).then_with(|| a.cmp(&b)));

    let mut cumsum_exp_eta = vec![0.0; n];
    let mut cumsum_weighted_x = vec![0.0; n * p];
    let mut cumsum_weighted_x_sq = vec![0.0; n * p];
    let mut risk_set_pos = vec![0usize; n];

    let mut running_exp = 0.0;
    let mut running_x = vec![0.0; p];
    let mut running_x_sq = vec![0.0; p];

    for (pos, &idx) in sorted_indices.iter().enumerate() {
        let weighted_exp = weights[idx] * exp_eta[idx];
        running_exp += weighted_exp;

        for col in 0..p {
            let xij = x[idx * p + col];
            running_x[col] += weighted_exp * xij;
            running_x_sq[col] += weighted_exp * xij * xij;
        }

        cumsum_exp_eta[pos] = running_exp;
        let row_start = pos * p;
        cumsum_weighted_x[row_start..row_start + p].copy_from_slice(&running_x);
        cumsum_weighted_x_sq[row_start..row_start + p].copy_from_slice(&running_x_sq);
    }

    let mut start = 0;
    while start < n {
        let current_time = time[sorted_indices[start]];
        let mut end = start + 1;
        while end < n && crate::constants::same_time(time[sorted_indices[end]], current_time) {
            end += 1;
        }

        let pos = end - 1;
        for &idx in &sorted_indices[start..end] {
            risk_set_pos[idx] = pos;
        }
        start = end;
    }

    CoxRiskSetData {
        cumsum_exp_eta,
        cumsum_weighted_x,
        cumsum_weighted_x_sq,
        risk_set_pos,
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
}
