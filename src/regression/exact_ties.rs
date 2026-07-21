use ndarray::Array2;

pub(crate) struct ExactTiedMoments {
    pub log_denom: f64,
    pub mean: Vec<f64>,
    pub covariance: Array2<f64>,
}

pub(crate) struct ExactRiskAccumulator {
    pub log_denom: f64,
    pub mean: Vec<f64>,
    pub covariance: Array2<f64>,
    delta: Vec<f64>,
}

impl ExactRiskAccumulator {
    pub fn new(nvar: usize) -> Self {
        Self {
            log_denom: f64::NEG_INFINITY,
            mean: vec![0.0; nvar],
            covariance: Array2::zeros((nvar, nvar)),
            delta: vec![0.0; nvar],
        }
    }

    pub fn add(&mut self, person: usize, log_weight: f64, covar: &Array2<f64>) {
        if self.log_denom == f64::NEG_INFINITY {
            self.log_denom = log_weight;
            for variable in 0..self.mean.len() {
                self.mean[variable] = covar[(person, variable)];
            }
            return;
        }

        let total_log_weight = log_add_exp(self.log_denom, log_weight);
        let added_fraction = (log_weight - total_log_weight).exp();
        let existing_fraction = 1.0 - added_fraction;
        for (variable, difference) in self.delta.iter_mut().enumerate() {
            *difference = covar[(person, variable)] - self.mean[variable];
        }
        for row in 0..self.mean.len() {
            for column in 0..self.mean.len() {
                self.covariance[(row, column)] = existing_fraction * self.covariance[(row, column)]
                    + existing_fraction * added_fraction * self.delta[row] * self.delta[column];
            }
        }
        for (mean, &difference) in self.mean.iter_mut().zip(&self.delta) {
            *mean += added_fraction * difference;
        }
        self.log_denom = total_log_weight;
    }
}

fn log_add_exp(lhs: f64, rhs: f64) -> f64 {
    if lhs == f64::NEG_INFINITY {
        return rhs;
    }
    if rhs == f64::NEG_INFINITY {
        return lhs;
    }
    let largest = lhs.max(rhs);
    largest + ((lhs - largest).exp() + (rhs - largest).exp()).ln()
}

/// Marginal probability that each row is included in an exact tied-death set.
pub(crate) fn exact_inclusion_probabilities(
    risk_indices: &[usize],
    deaths: usize,
    log_risk: &[f64],
) -> Option<Vec<(usize, f64)>> {
    if deaths == 0 || deaths > risk_indices.len() {
        return None;
    }
    let shift = risk_indices
        .iter()
        .map(|&person| log_risk[person])
        .fold(f64::NEG_INFINITY, f64::max);
    if !shift.is_finite() {
        return None;
    }

    let n_risk = risk_indices.len();
    let mut prefix = vec![vec![f64::NEG_INFINITY; deaths + 1]; n_risk + 1];
    prefix[0][0] = 0.0;
    for pos in 0..n_risk {
        let (previous_rows, current_rows) = prefix.split_at_mut(pos + 1);
        let previous = &previous_rows[pos];
        let current = &mut current_rows[0];
        current.copy_from_slice(previous);
        let log_weight = log_risk[risk_indices[pos]] - shift;
        for size in 1..=deaths.min(pos + 1) {
            current[size] = log_add_exp(current[size], previous[size - 1] + log_weight);
        }
    }

    let mut suffix = vec![vec![f64::NEG_INFINITY; deaths + 1]; n_risk + 1];
    suffix[n_risk][0] = 0.0;
    for pos in (0..n_risk).rev() {
        let (current_rows, following_rows) = suffix.split_at_mut(pos + 1);
        let current = &mut current_rows[pos];
        let following = &following_rows[0];
        current.copy_from_slice(following);
        let log_weight = log_risk[risk_indices[pos]] - shift;
        for size in 1..=deaths.min(n_risk - pos) {
            current[size] = log_add_exp(current[size], following[size - 1] + log_weight);
        }
    }

    let log_denom = prefix[n_risk][deaths];
    if !log_denom.is_finite() {
        return None;
    }
    Some(
        (0..n_risk)
            .map(|pos| {
                let mut excluded_log_weight = f64::NEG_INFINITY;
                for (left_size, &left_log_weight) in prefix[pos].iter().take(deaths).enumerate() {
                    let right_size = deaths - 1 - left_size;
                    excluded_log_weight = log_add_exp(
                        excluded_log_weight,
                        left_log_weight + suffix[pos + 1][right_size],
                    );
                }
                let log_weight = log_risk[risk_indices[pos]] - shift;
                (
                    risk_indices[pos],
                    (log_weight + excluded_log_weight - log_denom)
                        .exp()
                        .clamp(0.0, 1.0),
                )
            })
            .collect(),
    )
}

/// Exact conditional moments for selecting `deaths` rows from a risk set.
///
/// The dynamic program stores each subset-size distribution as a log total,
/// mean, and central covariance. This avoids enumerating subsets, raw moment
/// cancellation, and overflow when the number of possible subsets is large.
pub(crate) fn exact_tied_moments(
    risk_indices: &[usize],
    deaths: usize,
    log_risk: &[f64],
    covar: &Array2<f64>,
) -> ExactTiedMoments {
    let nvar = covar.ncols();
    let states = deaths + 1;
    let mut log_denoms = vec![f64::NEG_INFINITY; states];
    let mut means = vec![0.0; states * nvar];
    let mut covariances = vec![0.0; states * nvar * nvar];
    let mut delta = vec![0.0; nvar];
    log_denoms[0] = 0.0;

    for (seen, &person) in risk_indices.iter().enumerate() {
        let max_size = deaths.min(seen + 1);
        for size in (1..=max_size).rev() {
            let added_log_weight = log_denoms[size - 1] + log_risk[person];
            if added_log_weight == f64::NEG_INFINITY {
                continue;
            }

            let existing_log_weight = log_denoms[size];
            let total_log_weight = log_add_exp(existing_log_weight, added_log_weight);
            let added_fraction = (added_log_weight - total_log_weight).exp();
            let existing_fraction = 1.0 - added_fraction;
            let current_mean_offset = size * nvar;
            let previous_mean_offset = (size - 1) * nvar;

            for variable in 0..nvar {
                delta[variable] = means[previous_mean_offset + variable]
                    + covar[(person, variable)]
                    - means[current_mean_offset + variable];
            }

            let current_covariance_offset = size * nvar * nvar;
            let previous_covariance_offset = (size - 1) * nvar * nvar;
            for row in 0..nvar {
                for column in 0..nvar {
                    let current = current_covariance_offset + row * nvar + column;
                    let previous = previous_covariance_offset + row * nvar + column;
                    covariances[current] = existing_fraction * covariances[current]
                        + added_fraction * covariances[previous]
                        + existing_fraction * added_fraction * delta[row] * delta[column];
                }
            }
            for variable in 0..nvar {
                means[current_mean_offset + variable] += added_fraction * delta[variable];
            }
            log_denoms[size] = total_log_weight;
        }
    }

    let mean_offset = deaths * nvar;
    let covariance_offset = deaths * nvar * nvar;
    let mut covariance = Array2::zeros((nvar, nvar));
    for row in 0..nvar {
        for column in 0..nvar {
            covariance[(row, column)] = covariances[covariance_offset + row * nvar + column];
        }
    }

    ExactTiedMoments {
        log_denom: log_denoms[deaths],
        mean: means[mean_offset..mean_offset + nvar].to_vec(),
        covariance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn moments_match_hand_computed_two_death_set() {
        let covar = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 2.0, 3.0, 4.0]).unwrap();
        let log_risk = [2.0_f64.ln(), 3.0_f64.ln(), 5.0_f64.ln()];

        let moments = exact_tied_moments(&[0, 1, 2], 2, &log_risk, &covar);

        assert_close(moments.log_denom, 31.0_f64.ln(), 1e-14);
        assert_close(moments.mean[0], 91.0 / 31.0, 1e-14);
        assert_close(moments.mean[1], 142.0 / 31.0, 1e-14);
        assert_close(
            moments.covariance[(0, 0)],
            301.0 / 31.0 - (91.0 / 31.0_f64).powi(2),
            1e-14,
        );
        assert_close(
            moments.covariance[(1, 0)],
            442.0 / 31.0 - 91.0 * 142.0 / 31.0_f64.powi(2),
            1e-14,
        );
        assert_close(
            moments.covariance[(1, 1)],
            724.0 / 31.0 - (142.0 / 31.0_f64).powi(2),
            1e-14,
        );
    }

    #[test]
    fn log_weight_shift_changes_only_the_denominator() {
        let covar = Array2::from_shape_vec((4, 1), vec![-1.0, 0.5, 2.0, 4.0]).unwrap();
        let base_logs = vec![-2.0, -0.5, 0.25, 1.5];
        let shifted_logs: Vec<f64> = base_logs.iter().map(|value| value + 1_000.0).collect();

        let base = exact_tied_moments(&[0, 1, 2, 3], 2, &base_logs, &covar);
        let shifted = exact_tied_moments(&[0, 1, 2, 3], 2, &shifted_logs, &covar);

        assert_close(shifted.log_denom - base.log_denom, 2_000.0, 1e-12);
        assert_close(shifted.mean[0], base.mean[0], 1e-12);
        assert_close(shifted.covariance[(0, 0)], base.covariance[(0, 0)], 1e-12);
    }

    #[test]
    fn singleton_accumulator_matches_dynamic_programming() {
        let covar = Array2::from_shape_vec((4, 2), vec![-1.0, 0.25, 0.5, 2.0, 2.0, -0.5, 4.0, 1.5])
            .unwrap();
        let log_risk = vec![998.0, 999.5, 1_000.25, 1_001.5];
        let expected = exact_tied_moments(&[0, 1, 2, 3], 1, &log_risk, &covar);
        let mut actual = ExactRiskAccumulator::new(2);
        for (person, &log_weight) in log_risk.iter().enumerate() {
            actual.add(person, log_weight, &covar);
        }

        assert_close(actual.log_denom, expected.log_denom, 1e-12);
        for variable in 0..2 {
            assert_close(actual.mean[variable], expected.mean[variable], 1e-12);
            for other in 0..2 {
                assert_close(
                    actual.covariance[(variable, other)],
                    expected.covariance[(variable, other)],
                    1e-12,
                );
            }
        }
    }

    #[test]
    fn inclusion_probabilities_match_pairwise_weights_and_log_shifts() {
        let base_logs = vec![2.0_f64.ln(), 3.0_f64.ln(), 5.0_f64.ln()];
        let shifted_logs: Vec<f64> = base_logs.iter().map(|value| value + 1_000.0).collect();
        let expected = [16.0 / 31.0, 21.0 / 31.0, 25.0 / 31.0];

        let base = exact_inclusion_probabilities(&[0, 1, 2], 2, &base_logs)
            .expect("pairwise inclusion probabilities should compute");
        let shifted = exact_inclusion_probabilities(&[0, 1, 2], 2, &shifted_logs)
            .expect("shifted inclusion probabilities should compute");

        for (((_, base_value), (_, shifted_value)), expected_value) in
            base.iter().zip(&shifted).zip(expected)
        {
            assert_close(*base_value, expected_value, 1e-12);
            assert_close(*shifted_value, expected_value, 1e-12);
        }
        assert_close(base.iter().map(|(_, value)| value).sum(), 2.0, 1e-12);
    }

    #[test]
    fn selecting_the_entire_risk_set_has_zero_covariance() {
        let covar = Array2::from_shape_vec((3, 1), vec![0.2, 1.5, -0.4]).unwrap();
        let log_risk = vec![700.0, -700.0, 0.0];

        let moments = exact_tied_moments(&[0, 1, 2], 3, &log_risk, &covar);

        assert_close(moments.log_denom, 0.0, 1e-12);
        assert_close(moments.mean[0], 1.3, 1e-12);
        assert_eq!(moments.covariance[(0, 0)], 0.0);
    }

    #[test]
    fn large_balanced_tie_remains_finite() {
        let n = 1_050;
        let deaths = n / 2;
        let covar =
            Array2::from_shape_vec((n, 1), (0..n).map(|idx| (idx % 2) as f64).collect()).unwrap();
        let log_risk = vec![0.0; n];

        let moments = exact_tied_moments(&(0..n).collect::<Vec<_>>(), deaths, &log_risk, &covar);

        assert!(moments.log_denom.is_finite());
        assert!(moments.mean[0].is_finite());
        assert!(moments.covariance[(0, 0)].is_finite());
        assert!(moments.covariance[(0, 0)] > 0.0);
    }
}
