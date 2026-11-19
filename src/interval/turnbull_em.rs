use crate::error::{SurvivalError, SurvivalResult};
use rayon::prelude::*;
use std::ops::Range;

/// Aitken acceleration can select a different iterate after even a one-ulp
/// change. Each interval sum and each output weight must retain its row order.
pub(super) struct TurnbullWeights<'a> {
    supports: &'a [Range<usize>],
    weights: &'a [f64],
    dense: Option<DenseIntervals>,
}

impl<'a> TurnbullWeights<'a> {
    pub(super) fn new(supports: &'a [Range<usize>], weights: &'a [f64], jumps: usize) -> Self {
        let visits = supports
            .iter()
            .map(Range::len)
            .fold(0, usize::saturating_add);
        // For narrow intervals a single pass is cheaper than sorting the rows
        // and gathering their sums on each iteration.
        let dense = (visits > 4096 && visits / 16 > supports.len().saturating_add(jumps))
            .then(|| DenseIntervals::new(supports, jumps, visits));
        Self {
            supports,
            weights,
            dense,
        }
    }

    pub(super) fn loglik(&mut self, jumps: &[f64]) -> f64 {
        if let Some(dense) = &mut self.dense {
            dense
                .masses(self.supports, jumps)
                .iter()
                .map(|p| p.ln())
                .sum()
        } else {
            self.supports
                .iter()
                .map(|range| jumps[range.clone()].iter().sum::<f64>().ln())
                .sum()
        }
    }

    pub(super) fn redistribute(&mut self, jumps: &[f64], out: &mut [f64]) -> SurvivalResult<()> {
        out.fill(0.0);
        if let Some(dense) = &mut self.dense {
            let tile_size = dense.tile_size;
            let factors = dense.masses(self.supports, jumps);
            for (mass, &weight) in factors.iter_mut().zip(self.weights) {
                *mass = multiplier(weight, *mass)?;
            }

            if tile_size < out.len() {
                out.par_chunks_mut(tile_size)
                    .enumerate()
                    .for_each(|(i, tile)| {
                        let offset = i * tile_size;
                        let end = offset + tile.len();
                        for (range, &factor) in self.supports.iter().zip(factors.iter()) {
                            let start = range.start.max(offset);
                            let stop = range.end.min(end);
                            if start < stop {
                                add_mass(
                                    &mut tile[start - offset..stop - offset],
                                    &jumps[start..stop],
                                    factor,
                                );
                            }
                        }
                    });
            } else {
                for (range, &factor) in self.supports.iter().zip(factors.iter()) {
                    add_mass(&mut out[range.clone()], &jumps[range.clone()], factor);
                }
            }
        } else {
            for (range, &weight) in self.supports.iter().zip(self.weights) {
                let slice = &jumps[range.clone()];
                let factor = multiplier(weight, slice.iter().sum())?;
                add_mass(&mut out[range.clone()], slice, factor);
            }
        }
        Ok(())
    }
}

fn multiplier(weight: f64, mass: f64) -> SurvivalResult<f64> {
    let factor = weight / mass;
    if factor.is_finite() {
        Ok(factor)
    } else {
        Err(SurvivalError::invalid_input(
            "interval-censored observation has no finite probability mass",
        ))
    }
}

fn add_mass(out: &mut [f64], jumps: &[f64], factor: f64) {
    for (weight, &jump) in out.iter_mut().zip(jumps) {
        *weight += factor * jump;
    }
}

struct DenseIntervals {
    order: Vec<usize>,
    sorted_masses: Vec<f64>,
    row_masses: Vec<f64>,
    rows_per_task: usize,
    tile_size: usize,
}

impl DenseIntervals {
    fn new(supports: &[Range<usize>], jumps: usize, visits: usize) -> Self {
        let mut order: Vec<_> = (0..supports.len()).collect();
        order.sort_unstable_by_key(|&i| (supports[i].start, supports[i].end));
        // Amortize scheduling and keep each output tile large enough to
        // vectorize the within-row updates.
        let tasks = if visits >= 256 * 1024 {
            rayon::current_num_threads().min(8)
        } else {
            1
        };
        Self {
            order,
            sorted_masses: vec![0.0; supports.len()],
            row_masses: vec![0.0; supports.len()],
            rows_per_task: supports.len().div_ceil(tasks).max(128),
            tile_size: jumps.div_ceil(tasks).max(128),
        }
    }

    fn masses(&mut self, supports: &[Range<usize>], jumps: &[f64]) -> &mut [f64] {
        // Starting at the same jump gives identical partial sums. Extend the
        // running sum one jump at a time and restore input order for the log
        // likelihood and the distribution of weights.
        let sum_rows = |order: &[usize], masses: &mut [f64]| {
            let mut previous_start = usize::MAX;
            let mut end = 0;
            let mut mass = -0.0;
            for (&i, value) in order.iter().zip(masses) {
                let range = &supports[i];
                if range.start != previous_start {
                    previous_start = range.start;
                    end = range.start;
                    // The identity used by Iterator::sum::<f64>().
                    mass = -0.0;
                }
                for &jump in &jumps[end..range.end] {
                    mass += jump;
                }
                end = range.end;
                *value = mass;
            }
        };
        if self.rows_per_task < self.order.len() {
            self.order
                .par_chunks(self.rows_per_task)
                .zip(self.sorted_masses.par_chunks_mut(self.rows_per_task))
                .for_each(|(order, masses)| sum_rows(order, masses));
        } else {
            sum_rows(&self.order, &mut self.sorted_masses);
        }
        for (&i, &mass) in self.order.iter().zip(&self.sorted_masses) {
            self.row_masses[i] = mass;
        }
        &mut self.row_masses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_intervals_preserve_every_interval_sum_and_weight() {
        let n = 1031;
        let mut jumps: Vec<_> = (0..n)
            .map(|i| 1e-120 * ((i % 29 + 1) as f64).sqrt())
            .collect();
        jumps[0] = 0.75;
        jumps[n - 1] = 0.25;
        let supports: Vec<_> = (0..n).map(|i| i % 11..n - i % 13).collect();
        let weights: Vec<_> = (0..n).map(|i| ((i % 7) as f64).sqrt()).collect();
        let mut em = TurnbullWeights::new(&supports, &weights, n);
        let dense = em.dense.as_mut().unwrap();
        dense.rows_per_task = 128;
        dense.tile_size = 128;
        for (mass, range) in dense.masses(&supports, &jumps).iter().zip(&supports) {
            assert_eq!(*mass, jumps[range.clone()].iter().sum::<f64>());
        }
        let likelihood = em.loglik(&jumps);
        let mut fitted = vec![0.0; n];
        em.redistribute(&jumps, &mut fitted).unwrap();

        em.dense = None;
        assert_eq!(em.loglik(&jumps), likelihood);
        let mut direct = vec![0.0; n];
        em.redistribute(&jumps, &mut direct).unwrap();
        assert_eq!(direct, fitted);
    }

    #[test]
    fn dense_intervals_check_zero_mass_even_for_zero_weight() {
        let n = 128;
        let jumps = vec![-0.0; n];
        let mut supports = vec![0..n; n];
        supports[0] = 0..0;
        let weights = vec![0.0; n];
        let mut em = TurnbullWeights::new(&supports, &weights, n);
        let masses = em.dense.as_mut().unwrap().masses(&supports, &jumps);
        assert!(
            masses
                .iter()
                .all(|mass| mass.to_bits() == (-0.0_f64).to_bits())
        );
        assert!(em.redistribute(&jumps, &mut vec![0.0; n]).is_err());
    }
}
