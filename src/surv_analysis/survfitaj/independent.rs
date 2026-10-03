//! Independent competing-risk IJ from departed influence moments.
//!
//! All rows still at risk have influence equal to their normalized weight
//! times a shared coefficient. Departed rows only undergo the AJ transition
//! and area updates. Small triangular QR factors retain their second moments
//! without subtracting nearly equal squares or storing one vector per row.

use super::AJKernelData;
use ndarray::Array2;

/// Upper triangular factor of a Gram matrix; three columns suffice for
/// [source probability influence, target probability influence, target area].
#[derive(Clone, Copy)]
struct Moments<const N: usize> {
    factor: [[f64; N]; N],
}

impl<const N: usize> Default for Moments<N> {
    fn default() -> Self {
        Self {
            factor: [[0.0; N]; N],
        }
    }
}

impl<const N: usize> Moments<N> {
    fn add(&mut self, mut row: [f64; N]) {
        for col in 0..N {
            let old = self.factor[col][col];
            let norm = old.hypot(row[col]);
            if norm == 0.0 {
                continue;
            }
            let cosine = old / norm;
            let sine = row[col] / norm;
            self.factor[col][col] = norm;
            for (next, value) in row.iter_mut().enumerate().skip(col + 1) {
                let before = self.factor[col][next];
                self.factor[col][next] = cosine * before + sine * *value;
                *value = cosine * *value - sine * before;
            }
        }
    }

    fn norm(&self, col: usize) -> f64 {
        (0..=col).fold(0.0_f64, |norm, row| norm.hypot(self.factor[row][col]))
    }
}

pub(super) struct Estimates {
    pub pstate: Array2<f64>,
    pub cumhaz: Array2<f64>,
    pub std_err: Array2<f64>,
    pub std_chaz: Array2<f64>,
    pub std_auc: Array2<f64>,
}

struct Influence {
    probability: Vec<f64>,
    area: Vec<f64>,
    hazard: Vec<f64>,
    source: Moments<2>,
    targets: Vec<Moments<3>>,
    hazard_norm: Vec<f64>,
}

fn safe_variance_norm(norm: f64) -> bool {
    // R's general IJ squares row influences. Retain its overflow and
    // underflow behavior rather than changing missing confidence limits.
    norm == 0.0 || (norm >= f64::MIN_POSITIVE.sqrt() && norm <= f64::MAX.sqrt())
}

impl Influence {
    fn new(states: usize, hazards: usize) -> Self {
        Self {
            probability: vec![0.0; states],
            area: vec![0.0; states],
            hazard: vec![0.0; hazards],
            source: Moments::default(),
            targets: vec![Moments::default(); states],
            hazard_norm: vec![0.0; hazards],
        }
    }

    fn integrate(&mut self, delta: f64, origin: usize) {
        self.source.factor[0][1] += delta * self.source.factor[0][0];
        for state in 0..self.probability.len() {
            self.area[state] += delta * self.probability[state];
            if state != origin {
                let factor = &mut self.targets[state].factor;
                factor[0][2] += delta * factor[0][1];
                factor[1][2] += delta * factor[1][1];
            }
        }
    }

    fn transition(&mut self, change: &[f64], origin: usize) {
        let multiplier = 1.0 + change[origin];
        self.source.factor[0][0] *= multiplier;
        let before = self.probability[origin];
        for (state, &increment) in change.iter().enumerate() {
            if state == origin {
                self.probability[state] += before * increment;
            } else {
                let factor = &mut self.targets[state].factor;
                factor[0][1] += increment * factor[0][0];
                factor[0][0] *= multiplier;
                self.probability[state] += before * increment;
            }
        }
    }

    fn exit(
        &mut self,
        d: &AJKernelData<'_>,
        row: usize,
        origin: usize,
        scale: f64,
        event_derivative: f64,
        hazard_derivative: f64,
    ) {
        let weight = d.wt[row] / scale;
        let destination = d.state[row].checked_sub(1);
        let event = destination.is_some_and(|state| state != origin);
        let source =
            weight * (self.probability[origin] - if event { event_derivative } else { 0.0 });
        self.source.add([source, weight * self.area[origin]]);
        for state in 0..self.probability.len() {
            if state != origin {
                let probability = weight
                    * (self.probability[state]
                        + if event && destination == Some(state) {
                            event_derivative
                        } else {
                            0.0
                        });
                self.targets[state].add([source, probability, weight * self.area[state]]);
            }
        }
        let event_hazard = destination.and_then(|state| d.hindx[[origin, state]]);
        for hazard in 0..self.hazard.len() {
            let value = weight
                * (self.hazard[hazard]
                    + if event_hazard == Some(hazard) {
                        hazard_derivative
                    } else {
                        0.0
                    });
            self.hazard_norm[hazard] = self.hazard_norm[hazard].hypot(value);
        }
    }
}

/// `None` retains the general IJ for clusters, delayed entry, initial-state
/// uncertainty, explicit influences, and inputs outside finite moment arithmetic.
pub(super) fn estimates(
    d: &AJKernelData<'_>,
    risk: &Array2<f64>,
    transitions: &Array2<f64>,
) -> Option<Estimates> {
    if !d.right || d.sefit != 1 || d.ngrp != d.sort2.len() || d.i0.iter().any(|&v| v != 0.0) {
        return None;
    }
    let origin = d.cstate[*d.sort2.first()?];
    if d.sort2.iter().any(|&row| d.cstate[row] != origin) {
        return None;
    }
    let scale = d.sort2.iter().map(|&row| d.wt[row]).fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let states = d.p0.len();
    let hazards = d.trmat.len();
    let times = d.utime.len();
    let first_time = d.utime.first().copied().unwrap_or(f64::INFINITY);
    let mut destinations = d.sort2.iter().filter_map(|&row| {
        (d.wt[row] > 0.0 && d.time2[row] >= first_time)
            .then_some(d.state[row])?
            .checked_sub(1)
            .filter(|&state| state != origin)
    });
    let only_destination = destinations
        .next()
        .filter(|&destination| destinations.all(|other| other == destination));
    // Backwards norms avoid cancellation after a heavy subject exits and
    // avoid overflow/underflow from explicitly squaring normalized weights.
    let mut risk_norm = vec![0.0_f64; d.sort2.len() + 1];
    for position in (0..d.sort2.len()).rev() {
        let weight = d.wt[d.sort2[position]];
        let normalized = weight / scale;
        if weight > 0.0 && normalized == 0.0 {
            return None;
        }
        risk_norm[position] = risk_norm[position + 1].hypot(normalized);
    }
    let mut result = Estimates {
        pstate: Array2::zeros((times, states)),
        cumhaz: Array2::zeros((times, hazards)),
        std_err: Array2::zeros((times, states)),
        std_chaz: Array2::zeros((times, hazards)),
        std_auc: Array2::zeros((times, states)),
    };
    let mut influence = Influence::new(states, hazards);
    let mut probability = d.p0.to_vec();
    let mut cumulative = vec![0.0; hazards];
    let mut change = vec![0.0; states];
    let mut position = 0;
    for (time_index, &time) in d.utime.iter().enumerate() {
        while position < d.sort2.len() && d.time2[d.sort2[position]] < time {
            influence.exit(d, d.sort2[position], origin, scale, 0.0, 0.0);
            position += 1;
        }
        let delta = time
            - if time_index == 0 {
                d.t0
            } else {
                d.utime[time_index - 1]
            };
        influence.integrate(delta, origin);
        let start = position;
        let mut end = position;
        while end < d.sort2.len() && d.time2[d.sort2[end]] <= time {
            end += 1;
        }
        let denominator = risk[[time_index, origin]];
        let event = d.sort2[position..end].iter().any(|&row| d.state[row] > 0);
        if !denominator.is_finite() || (event && denominator <= 0.0) {
            // R's zero-weight events on an empty weighted risk set produce
            // NaN standard errors. Preserve this with its general calculation.
            return None;
        }
        let inverse = if event {
            1.0 / (denominator / scale)
        } else {
            0.0
        };
        if !inverse.is_finite() {
            return None;
        }
        change.fill(0.0);
        for &row in &d.sort2[position..end] {
            if let Some(destination) = d.state[row].checked_sub(1)
                && destination != origin
            {
                let hazard = d.wt[row] / denominator;
                change[origin] -= hazard;
                change[destination] += hazard;
            }
        }
        influence.transition(&change, origin);
        for (hazard, &(from, to)) in d.trmat.iter().enumerate() {
            let events = transitions[[time_index, hazard]];
            if events > 0.0 {
                let increment = events / risk[[time_index, from]];
                if !(increment / risk[[time_index, from]]).is_finite() {
                    // Normalization must preserve missing values caused by
                    // overflow in the general kernel's scaled hazard.
                    return None;
                }
                let derivative = increment * inverse;
                influence.hazard[hazard] -= derivative;
                if from != to {
                    let term = probability[from] * derivative;
                    influence.probability[from] += term;
                    influence.probability[to] -= term;
                }
            }
        }
        let event_derivative = probability[origin] * inverse;
        if !event_derivative.is_finite()
            || influence
                .probability
                .iter()
                .chain(&influence.area)
                .chain(&influence.hazard)
                .any(|value| !value.is_finite())
        {
            return None;
        }
        for &row in &d.sort2[position..end] {
            influence.exit(d, row, origin, scale, event_derivative, inverse);
        }
        position = end;
        let before = probability.clone();
        for (hazard, &(from, to)) in d.trmat.iter().enumerate() {
            let events = transitions[[time_index, hazard]];
            if events > 0.0 {
                let increment = events / risk[[time_index, from]];
                cumulative[hazard] += increment;
                probability[from] -= before[from] * increment;
                probability[to] += before[from] * increment;
            }
            let se =
                influence.hazard_norm[hazard].hypot(risk_norm[position] * influence.hazard[hazard]);
            if !safe_variance_norm(se) {
                return None;
            }
            result.cumhaz[[time_index, hazard]] = cumulative[hazard];
            result.std_chaz[[time_index, hazard]] = se;
        }
        for (state, &estimate) in probability.iter().enumerate() {
            let (probability_norm, area_norm) = if state == origin {
                (influence.source.norm(0), influence.source.norm(1))
            } else {
                (
                    influence.targets[state].norm(1),
                    influence.targets[state].norm(2),
                )
            };
            let se = probability_norm.hypot(risk_norm[position] * influence.probability[state]);
            let area_se = area_norm.hypot(risk_norm[position] * influence.area[state]);
            if !safe_variance_norm(se) || !safe_variance_norm(area_se) {
                return None;
            }
            result.pstate[[time_index, state]] = estimate;
            result.std_err[[time_index, state]] = se;
            result.std_auc[[time_index, state]] = area_se;
        }
        // With a single destination its influence is exactly the negative
        // of the source influence, including when p0 is nondegenerate.
        // Retain that identity through complete absorption rather than
        // introducing residual variance through independent QR updates.
        if let Some(destination) = only_destination {
            result.std_err[[time_index, destination]] = result.std_err[[time_index, origin]];
            result.std_auc[[time_index, destination]] = result.std_auc[[time_index, origin]];
        }
        if change[origin] == -1.0 {
            // Complete absorption annihilates previous source influences.
            // Check the current-row arithmetic in its original order: a
            // residual of a single ulp can change log confidence limits,
            // including when the probability itself rounded below zero.
            // Use the general kernel when the exact-zero error mask differs,
            // even if the errors agree numerically.
            let mut squared = 0.0;
            for &row in &d.sort2[start..] {
                let mut value = 0.0;
                if d.time2[row] == time && d.state[row] > 0 && d.state[row] - 1 != origin {
                    value -= d.wt[row] * before[origin] / denominator;
                }
                if d.wt[row] > 0.0 {
                    for (hazard, &(from, to)) in d.trmat.iter().enumerate() {
                        let events = transitions[[time_index, hazard]];
                        if from == origin && to != origin && events > 0.0 {
                            let scaled = (events / denominator) / denominator;
                            value += d.wt[row] * before[origin] * scaled;
                        }
                    }
                }
                squared += value * value;
            }
            if (squared == 0.0) != (result.std_err[[time_index, origin]] == 0.0) {
                return None;
            }
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::{Influence, Moments};

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 2e-13 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual:.17e} differs from {expected:.17e}"
        );
    }

    fn assert_gram<const N: usize>(moments: &Moments<N>, rows: &[[f64; N]], scale: f64) {
        for first in 0..N {
            let expected_norm = rows
                .iter()
                .fold(0.0_f64, |norm, row| norm.hypot(row[first]));
            assert_close(moments.norm(first) / scale, expected_norm / scale);
            for second in 0..N {
                let expected: f64 = rows
                    .iter()
                    .map(|row| (row[first] / scale) * (row[second] / scale))
                    .sum();
                let actual: f64 = (0..N)
                    .map(|row| {
                        (moments.factor[row][first] / scale) * (moments.factor[row][second] / scale)
                    })
                    .sum();
                assert_close(actual, expected);
                if second < first {
                    assert_eq!(moments.factor[first][second], 0.0);
                }
            }
        }
    }

    #[test]
    fn qr_preserves_dependent_rows_and_zero_leading_pivots() {
        let rows = [
            [0.0, 3.0, -6.0],
            [0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0],
            [2.0, -4.0, 1.0],
            [-4.0, 8.0, -2.0],
            [0.0, 2.0, 1.0],
        ];
        let mut moments = Moments::<3>::default();
        for (index, &row) in rows.iter().enumerate() {
            moments.add(row);
            assert_gram(&moments, &rows[..=index], 1.0);
        }
    }

    #[test]
    fn qr_norms_remain_finite_when_explicit_squares_do_not() {
        let unscaled = [[3.0, -4.0, 0.0], [-2.0, 0.0, 5.0], [0.0, 1.0, -2.0]];
        for scale in [1e-200, 1e200] {
            let rows = unscaled.map(|row| row.map(|value| value * scale));
            let mut moments = Moments::<3>::default();
            for row in rows {
                moments.add(row);
            }
            assert_gram(&moments, &rows, scale);
            assert!((0..3).all(|column| moments.norm(column).is_finite()));
        }
    }

    fn assert_departed(
        influence: &Influence,
        probabilities: &[[f64; 4]],
        areas: &[[f64; 4]],
        origin: usize,
    ) {
        let source: Vec<[f64; 2]> = probabilities
            .iter()
            .zip(areas)
            .map(|(probability, area)| [probability[origin], area[origin]])
            .collect();
        assert_gram(&influence.source, &source, 1.0);
        for state in 0..4 {
            if state != origin {
                let target: Vec<[f64; 3]> = probabilities
                    .iter()
                    .zip(areas)
                    .map(|(probability, area)| {
                        [probability[origin], probability[state], area[state]]
                    })
                    .collect();
                assert_gram(&influence.targets[state], &target, 1.0);
            }
        }
    }

    #[test]
    fn departed_factors_match_rowwise_area_and_probability_transforms() {
        let origin = 1;
        let mut probabilities = vec![
            [-0.3, 0.5, -0.2, 0.0],
            [0.8, -0.5, -0.3, 0.0],
            [0.1, 0.0, -0.1, 0.0],
        ];
        let mut areas = vec![
            [1.0, -1.25, 0.25, 0.0],
            [-0.4, 0.2, 0.2, 0.0],
            [2.0, 0.0, -2.0, 0.0],
        ];
        let mut influence = Influence::new(4, 0);
        influence.probability = vec![0.25, -0.75, 0.5, 0.0];
        influence.area = vec![-2.0, 0.5, 1.5, 0.0];
        let mut active_probability = influence.probability.clone();
        let mut active_area = influence.area.clone();
        for (probability, area) in probabilities.iter().zip(&areas) {
            influence.source.add([probability[origin], area[origin]]);
            for state in 0..4 {
                if state != origin {
                    influence.targets[state].add([
                        probability[origin],
                        probability[state],
                        area[state],
                    ]);
                }
            }
        }
        assert_departed(&influence, &probabilities, &areas, origin);

        for (delta, change) in [
            (2.75, [0.2, -0.7, 0.5, 0.0]),
            (0.125, [0.25, -1.0, 0.75, 0.0]),
        ] {
            for (probability, area) in probabilities.iter().zip(&mut areas) {
                for state in 0..4 {
                    area[state] += delta * probability[state];
                }
            }
            for state in 0..4 {
                active_area[state] += delta * active_probability[state];
            }
            influence.integrate(delta, origin);
            assert_departed(&influence, &probabilities, &areas, origin);
            for (actual, expected) in influence.area.iter().zip(&active_area) {
                assert_close(*actual, *expected);
            }

            for probability in &mut probabilities {
                let source = probability[origin];
                for state in 0..4 {
                    probability[state] += source * change[state];
                }
            }
            let source = active_probability[origin];
            for state in 0..4 {
                active_probability[state] += source * change[state];
            }
            influence.transition(&change, origin);
            assert_departed(&influence, &probabilities, &areas, origin);
            for (actual, expected) in influence.probability.iter().zip(&active_probability) {
                assert_close(*actual, *expected);
            }
        }
        assert_eq!(influence.source.norm(0), 0.0);

        // Complete absorption leaves a zero source pivot, while existing
        // target and area columns continue to contain departed influence.
        let probability = [-0.4, 0.0, 0.4, 0.0];
        let area = [0.75, -0.25, -0.5, 0.0];
        influence.source.add([probability[origin], area[origin]]);
        for state in 0..4 {
            if state != origin {
                influence.targets[state].add([
                    probability[origin],
                    probability[state],
                    area[state],
                ]);
            }
        }
        probabilities.push(probability);
        areas.push(area);
        assert_departed(&influence, &probabilities, &areas, origin);
    }
}
