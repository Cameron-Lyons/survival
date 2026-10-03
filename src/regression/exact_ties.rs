use ndarray::{Array2, ArrayView2};

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

    pub fn clear(&mut self) {
        self.log_denom = f64::NEG_INFINITY;
        self.mean.fill(0.0);
        self.covariance.fill(0.0);
    }

    pub fn add(&mut self, person: usize, log_weight: f64, covar: &Array2<f64>) {
        if self.log_denom == f64::NEG_INFINITY {
            self.log_denom = log_weight;
            for variable in 0..self.mean.len() {
                self.mean[variable] = covar[(person, variable)];
            }
            return;
        }

        let (total_log_weight, existing_fraction, added_fraction) =
            log_weighted_mix(self.log_denom, log_weight);
        for (variable, difference) in self.delta.iter_mut().enumerate() {
            *difference = covar[(person, variable)] - self.mean[variable];
        }
        for row in 0..self.mean.len() {
            for column in 0..self.mean.len() {
                self.covariance[(row, column)] = existing_fraction * self.covariance[(row, column)]
                    + existing_fraction * added_fraction * self.delta[row] * self.delta[column];
            }
        }
        for (variable, (mean, &difference)) in self.mean.iter_mut().zip(&self.delta).enumerate() {
            *mean = if existing_fraction >= added_fraction {
                *mean + added_fraction * difference
            } else {
                covar[(person, variable)] - existing_fraction * difference
            };
        }
        self.log_denom = total_log_weight;
    }
}

/// Dynamic singleton moments for counting-process risk sets. Leaves contain
/// 16 input rows: changing membership rebuilds that small block, then merges
/// its ancestors. Removal never subtracts a large weight from rounded totals.
/// Dirty blocks accumulate until the next death time, batching censor changes.
pub(crate) struct ExactRiskTree {
    first_row: usize,
    active: Vec<bool>,
    active_count: usize,
    leaf_base: usize,
    dirty: Vec<bool>,
    pending: Vec<usize>,
    log_denoms: Vec<f64>,
    means: Vec<f64>,
    covariances: Vec<f64>,
    scratch: ExactRiskAccumulator,
}

const EXACT_RISK_BLOCK_SIZE: usize = 16;

impl ExactRiskTree {
    pub fn new(first_row: usize, nrows: usize, nvar: usize) -> Self {
        let blocks = nrows.div_ceil(EXACT_RISK_BLOCK_SIZE);
        let leaf_base = blocks.next_power_of_two();
        let nodes = 2 * leaf_base;
        Self {
            first_row,
            active: vec![false; nrows],
            active_count: 0,
            leaf_base,
            dirty: vec![false; blocks],
            pending: Vec::new(),
            log_denoms: vec![f64::NEG_INFINITY; nodes],
            means: vec![0.0; nodes * nvar],
            covariances: vec![0.0; nodes * nvar * nvar],
            scratch: ExactRiskAccumulator::new(nvar),
        }
    }

    pub fn len(&self) -> usize {
        self.active_count
    }

    pub fn active_rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(row, &active)| active.then_some(self.first_row + row))
    }

    pub fn set_active(&mut self, row: usize, active: bool) {
        let position = row - self.first_row;
        debug_assert_ne!(self.active[position], active);
        self.active[position] = active;
        if active {
            self.active_count += 1;
        } else {
            self.active_count -= 1;
        }
        let block = position / EXACT_RISK_BLOCK_SIZE;
        if !self.dirty[block] {
            self.dirty[block] = true;
            self.pending.push(block);
        }
    }

    pub fn refresh(&mut self, log_risk: &[f64], covar: &Array2<f64>) {
        let nvar = self.scratch.mean.len();
        let square = nvar * nvar;
        while let Some(block) = self.pending.pop() {
            self.dirty[block] = false;
            self.scratch.clear();
            let start = block * EXACT_RISK_BLOCK_SIZE;
            let end = (start + EXACT_RISK_BLOCK_SIZE).min(self.active.len());
            for position in start..end {
                if self.active[position] {
                    let row = self.first_row + position;
                    self.scratch.add(row, log_risk[row], covar);
                }
            }
            let mut node = self.leaf_base + block;
            self.log_denoms[node] = self.scratch.log_denom;
            self.means[node * nvar..(node + 1) * nvar].copy_from_slice(&self.scratch.mean);
            self.covariances[node * square..(node + 1) * square]
                .copy_from_slice(self.scratch.covariance.as_slice().expect("standard layout"));
            while node > 1 {
                node /= 2;
                self.merge_children(node);
            }
        }
    }

    fn merge_children(&mut self, node: usize) {
        let (left, right) = (2 * node, 2 * node + 1);
        let nvar = self.scratch.mean.len();
        let square = nvar * nvar;
        // An empty child contributes no moments. Copying the other child
        // also preserves tiny contributions after its dominant sibling leaves.
        if self.log_denoms[left] == f64::NEG_INFINITY || self.log_denoms[right] == f64::NEG_INFINITY
        {
            let source = if self.log_denoms[left] == f64::NEG_INFINITY {
                right
            } else {
                left
            };
            self.log_denoms[node] = self.log_denoms[source];
            self.means
                .copy_within(source * nvar..(source + 1) * nvar, node * nvar);
            self.covariances
                .copy_within(source * square..(source + 1) * square, node * square);
            return;
        }
        let (total, a, b) = log_weighted_mix(self.log_denoms[left], self.log_denoms[right]);
        self.log_denoms[node] = total;
        for i in 0..nvar {
            self.scratch.delta[i] = self.means[right * nvar + i] - self.means[left * nvar + i];
            self.means[node * nvar + i] = if a >= b {
                self.means[left * nvar + i] + b * self.scratch.delta[i]
            } else {
                self.means[right * nvar + i] - a * self.scratch.delta[i]
            };
        }
        for i in 0..nvar {
            for j in 0..nvar {
                let cell = i * nvar + j;
                self.covariances[node * square + cell] = a * self.covariances[left * square + cell]
                    + b * self.covariances[right * square + cell]
                    + a * b * self.scratch.delta[i] * self.scratch.delta[j];
            }
        }
    }

    pub fn log_denom(&self) -> f64 {
        self.log_denoms[1]
    }

    pub fn mean(&self) -> &[f64] {
        let nvar = self.scratch.mean.len();
        &self.means[nvar..2 * nvar]
    }

    pub fn covariance(&self) -> ArrayView2<'_, f64> {
        let nvar = self.scratch.mean.len();
        let square = nvar * nvar;
        ArrayView2::from_shape((nvar, nvar), &self.covariances[square..2 * square])
            .expect("one root covariance matrix")
    }
}

/// Total log weight and the two mixing probabilities. Compute the smaller
/// probability directly, since subtracting the larger from one can erase
/// the covariance when the weights differ by more than about exp(36).
fn log_weighted_mix(lhs: f64, rhs: f64) -> (f64, f64, f64) {
    if lhs == f64::NEG_INFINITY {
        return (rhs, 0.0, 1.0);
    }
    if rhs == f64::NEG_INFINITY {
        return (lhs, 1.0, 0.0);
    }
    let ratio = (-(lhs - rhs).abs()).exp();
    let total = lhs.max(rhs) + ratio.ln_1p();
    let smaller = ratio / (1.0 + ratio);
    let larger = 1.0 / (1.0 + ratio);
    if lhs >= rhs {
        (total, larger, smaller)
    } else {
        (total, smaller, larger)
    }
}

pub(crate) fn exact_tied_moments(
    risk_indices: &[usize],
    deaths: usize,
    log_risk: &[f64],
    covar: &Array2<f64>,
) -> ExactTiedMoments {
    let nvar = covar.ncols();
    debug_assert!(deaths <= risk_indices.len());
    // A death subset A has weight exp(sum(eta[A])). Its complement has
    // weight exp(-sum(eta[Ac])), multiplied by the same exp(sum(eta))
    // for every subset. Compute whichever subset is smaller: this changes
    // O(n * deaths * p^2) work to O(n * min(deaths, n - deaths) * p^2).
    // Selected and complementary covariate sums have equal covariances.
    // Track selected-so-far means even on the complementary path: subtracting
    // an almost-certainly excluded large covariate from total_x can erase the
    // moderate selected mean.
    let complementary = deaths > risk_indices.len() / 2;
    // Remove a common shift before adding or negating log risks. Otherwise
    // the complement's negative log denominator can cancel against a large
    // total even though all risk ratios are small and well conditioned.
    let log_shift = if complementary {
        log_risk[risk_indices[0]]
    } else {
        0.0
    };
    let subset_size = if complementary {
        risk_indices.len() - deaths
    } else {
        deaths
    };
    let states = subset_size + 1;
    let mut log_denoms = vec![f64::NEG_INFINITY; states];
    let mut means = vec![0.0; states * nvar];
    let mut covariances = vec![0.0; states * nvar * nvar];
    let mut delta = vec![0.0; nvar];
    let mut total_log_risk = 0.0;
    log_denoms[0] = 0.0;

    for (seen, &person) in risk_indices.iter().enumerate() {
        let log_weight = if complementary {
            let centered = log_risk[person] - log_shift;
            total_log_risk += centered;
            -centered
        } else {
            log_risk[person]
        };
        let max_size = subset_size.min(seen + 1);
        for size in (1..=max_size).rev() {
            let added_log_weight = log_denoms[size - 1] + log_weight;
            if added_log_weight == f64::NEG_INFINITY {
                continue;
            }

            let existing_log_weight = log_denoms[size];
            let (total_log_weight, existing_fraction, added_fraction) =
                log_weighted_mix(existing_log_weight, added_log_weight);
            let current_mean_offset = size * nvar;
            let previous_mean_offset = (size - 1) * nvar;

            for variable in 0..nvar {
                let x = covar[(person, variable)];
                let existing =
                    means[current_mean_offset + variable] + if complementary { x } else { 0.0 };
                let added =
                    means[previous_mean_offset + variable] + if complementary { 0.0 } else { x };
                delta[variable] = added - existing;
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
                // Anchor at the more probable branch. In particular, when
                // added_fraction rounds to 1, existing + delta can cancel the
                // added mean across a large covariate contrast.
                let x = covar[(person, variable)];
                means[current_mean_offset + variable] = if existing_fraction >= added_fraction {
                    means[current_mean_offset + variable]
                        + if complementary { x } else { 0.0 }
                        + added_fraction * delta[variable]
                } else {
                    means[previous_mean_offset + variable] + if complementary { 0.0 } else { x }
                        - existing_fraction * delta[variable]
                };
            }
            log_denoms[size] = total_log_weight;
        }
        if complementary {
            // The zero-exclusion state selects every row seen so far. Update
            // it after the descending-size walk has read its previous mean.
            for variable in 0..nvar {
                means[variable] += covar[(person, variable)];
            }
        }
    }

    let mean_offset = subset_size * nvar;
    let covariance_offset = subset_size * nvar * nvar;
    let mut covariance = Array2::zeros((nvar, nvar));
    for row in 0..nvar {
        for column in 0..nvar {
            covariance[(row, column)] = covariances[covariance_offset + row * nvar + column];
        }
    }

    ExactTiedMoments {
        log_denom: log_denoms[subset_size] + total_log_risk + deaths as f64 * log_shift,
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

    /// Independent reference: enumerate every possible death subset and
    /// compute its weighted mean and centred covariance directly.
    fn enumerated_moments(
        risk_indices: &[usize],
        deaths: usize,
        log_risk: &[f64],
        covar: &Array2<f64>,
    ) -> ExactTiedMoments {
        let nvar = covar.ncols();
        let subsets: Vec<(f64, Vec<f64>)> = (0u32..1 << risk_indices.len())
            .filter(|mask| mask.count_ones() as usize == deaths)
            .map(|mask| {
                let mut log_weight = 0.0;
                let mut x = vec![0.0; nvar];
                for (i, &row) in risk_indices.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        log_weight += log_risk[row];
                        for (j, value) in x.iter_mut().enumerate() {
                            *value += covar[(row, j)];
                        }
                    }
                }
                (log_weight, x)
            })
            .collect();
        let shift = subsets
            .iter()
            .map(|(log_weight, _)| *log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = subsets
            .iter()
            .map(|(log_weight, _)| (log_weight - shift).exp())
            .sum();
        let mut mean = vec![0.0; nvar];
        for (log_weight, x) in &subsets {
            let weight = (log_weight - shift).exp() / total;
            for (j, value) in mean.iter_mut().enumerate() {
                *value += weight * x[j];
            }
        }
        let mut covariance = Array2::zeros((nvar, nvar));
        for (log_weight, x) in &subsets {
            let weight = (log_weight - shift).exp() / total;
            for i in 0..nvar {
                for j in 0..nvar {
                    covariance[(i, j)] += weight * (x[i] - mean[i]) * (x[j] - mean[j]);
                }
            }
        }
        ExactTiedMoments {
            log_denom: shift + total.ln(),
            mean,
            covariance,
        }
    }

    #[test]
    fn all_subset_sizes_match_exhaustive_weighted_enumeration() {
        let covar = Array2::from_shape_vec(
            (7, 2),
            vec![
                -1.0, 0.25, 0.5, 2.0, 2.0, -0.5, 4.0, 1.5, 0.25, -2.0, 0.75, 3.0, -0.5, 0.0,
            ],
        )
        .unwrap();
        // Exercise noncontiguous indices and an order independent of row order.
        let risk_indices = [6, 0, 4, 2, 5];
        for shift in [-1_000.0, 0.0, 1_000.0] {
            let log_risk: Vec<f64> = [-2.0, -0.5, 0.25, 1.5, 3.0, -1.25, 0.75]
                .iter()
                .map(|weight| weight + shift)
                .collect();
            for deaths in 0..=risk_indices.len() {
                let actual = exact_tied_moments(&risk_indices, deaths, &log_risk, &covar);
                let expected = enumerated_moments(&risk_indices, deaths, &log_risk, &covar);
                assert_close(actual.log_denom, expected.log_denom, 1e-11);
                for i in 0..2 {
                    assert_close(actual.mean[i], expected.mean[i], 1e-11);
                    for j in 0..2 {
                        assert_close(
                            actual.covariance[(i, j)],
                            expected.covariance[(i, j)],
                            1e-11,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn complementary_moments_keep_tiny_covariance_across_orders_and_log_shifts() {
        let covar = Array2::from_shape_vec(
            (5, 2),
            vec![-1.0, 0.25, 0.5, 2.0, 2.0, -0.5, 0.25, -2.0, 0.75, 3.0],
        )
        .unwrap();
        // Row 0 almost certainly survives. Its arrival last must still
        // retain the small covariance of the other surviving subjects.
        for shift in [-10_000.0, 0.0, 10_000.0] {
            let log_risk: Vec<f64> = [-40.0, 0.0, 40.0, 1.0, -1.0]
                .iter()
                .map(|value| value + shift)
                .collect();
            let expected = enumerated_moments(&[0, 1, 2, 3, 4], 4, &log_risk, &covar);
            for rows in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]] {
                let actual = exact_tied_moments(&rows, 4, &log_risk, &covar);
                assert_close(actual.log_denom, expected.log_denom, 1e-10);
                for i in 0..2 {
                    assert_close(actual.mean[i], expected.mean[i], 1e-12);
                    for j in 0..2 {
                        let reference = expected.covariance[(i, j)];
                        assert_close(
                            actual.covariance[(i, j)],
                            reference,
                            reference.abs() * 1e-12,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn singleton_moments_keep_small_probability_when_the_dominant_row_arrives_last() {
        let covar = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let log_risk = [0.0, 40.0];
        let minority_probability = (-40.0_f64).exp();
        let expected = minority_probability / (1.0 + minority_probability).powi(2);
        let mut moments = ExactRiskAccumulator::new(1);
        moments.add(0, log_risk[0], &covar);
        moments.add(1, log_risk[1], &covar);
        assert_close(moments.covariance[(0, 0)], expected, expected * 1e-12);
        let tied = exact_tied_moments(&[0, 1], 1, &log_risk, &covar);
        assert_close(tied.covariance[(0, 0)], expected, expected * 1e-12);
    }

    #[test]
    fn dominant_probability_preserves_mean_across_large_covariate_contrasts() {
        let mut covar = Array2::zeros((17, 1));
        covar[(0, 0)] = 1e20;
        covar[(16, 0)] = 2.0;
        for shift in [-1_000.0, 0.0, 1_000.0] {
            let mut log_risk = vec![shift; 17];
            log_risk[0] -= 100.0;
            let expected = enumerated_moments(&[0, 16], 1, &log_risk, &covar);
            assert_eq!(expected.mean[0], 2.0);
            for rows in [[0, 16], [16, 0]] {
                let mut accumulator = ExactRiskAccumulator::new(1);
                for row in rows {
                    accumulator.add(row, log_risk[row], &covar);
                }
                assert_close(accumulator.mean[0], expected.mean[0], 1e-12);
                assert_close(
                    accumulator.covariance[(0, 0)],
                    expected.covariance[(0, 0)],
                    expected.covariance[(0, 0)] * 1e-12,
                );
                let tied = exact_tied_moments(&rows, 1, &log_risk, &covar);
                assert_close(tied.mean[0], expected.mean[0], 1e-12);
                assert_close(
                    tied.covariance[(0, 0)],
                    expected.covariance[(0, 0)],
                    expected.covariance[(0, 0)] * 1e-12,
                );
            }
            // Opposite blocks exercise the same merge in the tree ancestors.
            let mut tree = ExactRiskTree::new(0, 17, 1);
            tree.set_active(0, true);
            tree.set_active(16, true);
            tree.refresh(&log_risk, &covar);
            assert_close(tree.mean()[0], expected.mean[0], 1e-12);
            assert_close(
                tree.covariance()[(0, 0)],
                expected.covariance[(0, 0)],
                expected.covariance[(0, 0)] * 1e-12,
            );
        }
    }

    #[test]
    fn complementary_mean_preserves_selected_rows_when_excluded_covariates_are_large() {
        let covar = Array2::from_shape_vec((3, 1), vec![1e20, 2.0, 3.0]).unwrap();
        for shift in [-1_000.0, 0.0, 1_000.0] {
            let log_risk = [shift - 100.0, shift, shift];
            let expected = enumerated_moments(&[0, 1, 2], 2, &log_risk, &covar);
            assert_eq!(expected.mean[0], 5.0);
            for rows in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let actual = exact_tied_moments(&rows, 2, &log_risk, &covar);
                assert_close(actual.log_denom, expected.log_denom, 1e-11);
                assert_close(actual.mean[0], expected.mean[0], 1e-12);
                assert_close(
                    actual.covariance[(0, 0)],
                    expected.covariance[(0, 0)],
                    expected.covariance[(0, 0)] * 1e-12,
                );
            }
        }
    }

    #[test]
    fn nearly_complete_large_tie_matches_uniform_subset_moments() {
        // Omitting one row uniformly leaves the negative of that row's
        // covariates (both population means are zero), with its covariance.
        let n = 10_000;
        let covar = Array2::from_shape_fn((n, 2), |(row, column)| match column {
            0 => {
                if row % 2 == 0 {
                    -1.0
                } else {
                    1.0
                }
            }
            _ => (row % 5) as f64 - 2.0,
        });
        let moments = exact_tied_moments(&(0..n).collect::<Vec<_>>(), n - 1, &vec![0.0; n], &covar);
        assert_close(moments.log_denom, (n as f64).ln(), 1e-11);
        assert_close(moments.mean[0], 0.0, 1e-11);
        assert_close(moments.mean[1], 0.0, 1e-11);
        assert_close(moments.covariance[(0, 0)], 1.0, 1e-11);
        assert_close(moments.covariance[(1, 1)], 2.0, 1e-11);
        assert_close(moments.covariance[(0, 1)], 0.0, 1e-11);
        assert_close(
            moments.covariance[(0, 1)],
            moments.covariance[(1, 0)],
            1e-11,
        );
    }

    #[test]
    fn blocked_tree_preserves_survivors_when_a_dominant_other_block_leaves() {
        let n = 66;
        let mut covar = Array2::zeros((n, 2));
        for (row, values) in [
            (0, [100.0, -50.0]),
            (17, [0.0, 0.0]),
            (34, [1.0, 0.0]),
            (65, [-1.0, 1.0]),
        ] {
            for column in 0..2 {
                covar[(row, column)] = values[column];
            }
        }
        for shift in [-1_000.0, 0.0, 1_000.0] {
            let log_risk: Vec<f64> = (0..n).map(|row| covar[(row, 0)] + shift).collect();
            let mut tree = ExactRiskTree::new(0, n, 2);
            for row in [0, 17, 34, 65] {
                tree.set_active(row, true);
            }
            tree.refresh(&log_risk, &covar);
            let expected = enumerated_moments(&[0, 17, 34, 65], 1, &log_risk, &covar);
            for i in 0..2 {
                for j in 0..2 {
                    let reference = expected.covariance[(i, j)];
                    assert_close(
                        tree.covariance()[(i, j)],
                        reference,
                        reference.abs() * 1e-12,
                    );
                }
            }
            tree.set_active(0, false);
            tree.refresh(&log_risk, &covar);
            let expected = enumerated_moments(&[17, 34, 65], 1, &log_risk, &covar);
            assert_eq!(tree.len(), 3);
            assert_eq!(tree.active_rows().collect::<Vec<_>>(), vec![17, 34, 65]);
            assert_close(tree.log_denom(), expected.log_denom, 1e-11);
            for i in 0..2 {
                assert_close(tree.mean()[i], expected.mean[i], 1e-11);
                for j in 0..2 {
                    assert_close(
                        tree.covariance()[(i, j)],
                        expected.covariance[(i, j)],
                        1e-11,
                    );
                }
            }
            // Empty blocks and the whole tree reset; membership changes may
            // accumulate without refreshing until the next death time.
            for row in [17, 34, 65] {
                tree.set_active(row, false);
            }
            tree.set_active(17, true);
            tree.set_active(17, false);
            tree.refresh(&log_risk, &covar);
            assert_eq!(tree.len(), 0);
            assert_eq!(tree.log_denom(), f64::NEG_INFINITY);
            assert!(tree.mean().iter().all(|&value| value == 0.0));
            assert!(tree.covariance().iter().all(|&value| value == 0.0));
            tree.set_active(65, true);
            tree.refresh(&log_risk, &covar);
            assert_eq!(tree.len(), 1);
            assert_eq!(tree.log_denom(), log_risk[65]);
            assert_eq!(tree.mean(), &[-1.0, 1.0]);
            assert!(tree.covariance().iter().all(|&value| value == 0.0));
        }
    }

    #[test]
    fn blocked_tree_retains_scalar_denominators_for_zero_column_models() {
        let covar = Array2::zeros((40, 0));
        let log_risk = vec![1_000.0; 40];
        let mut tree = ExactRiskTree::new(0, 40, 0);
        for row in 0..40 {
            tree.set_active(row, true);
        }
        tree.refresh(&log_risk, &covar);
        assert_close(tree.log_denom(), 1_000.0 + 40.0_f64.ln(), 1e-11);
        assert!(tree.mean().is_empty());
        assert_eq!(tree.covariance().dim(), (0, 0));
    }
}
