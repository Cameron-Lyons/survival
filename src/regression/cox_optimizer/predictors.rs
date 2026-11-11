use super::CoxData;
use crate::regression::log_scale::LogScale;
use ndarray::{ArrayView1, ArrayView2};

// At this magnitude a dot-product term can erase an otherwise well-resolved
// covariate effect. Keep this precision threshold separate from risk overflow:
// existing counting-process recentering remains unchanged for ordinary trials.
pub(super) const PREDICTOR_PRECISION_LIMIT: f64 = 67_108_864.0;

pub(crate) fn predictor_bound_needs_anchor(bound: f64) -> bool {
    bound > PREDICTOR_PRECISION_LIMIT
}

pub(crate) fn bounds_need_anchor(bounds: &[f64], beta: &[f64]) -> bool {
    bounds
        .iter()
        .zip(beta)
        .any(|(&bound, &b)| predictor_bound_needs_anchor(bound * b.abs()))
}

pub(crate) fn covariate_anchor_needed(x: ArrayView2<'_, f64>, beta: &[f64]) -> bool {
    x.rows().into_iter().any(|row| {
        row.iter()
            .zip(beta)
            .any(|(&x, &b)| predictor_bound_needs_anchor((x * b).abs()))
    })
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    if !sum.is_finite() {
        return (sum, 0.0);
    }
    let part = sum - a;
    (sum, (a - (sum - part)) + (b - part))
}

fn compensated_dot(terms: impl Iterator<Item = (f64, f64)> + Clone) -> LogScale {
    let original = terms.clone();
    let (mut high, mut low) = (0.0, 0.0);
    for (difference, b) in terms {
        let product = difference * b;
        if !product.is_finite() {
            return LogScale::from_log(product);
        }
        let (sum, error) = two_sum(high, product);
        if !sum.is_finite() {
            return balanced_dot(original);
        }
        high = sum;
        low += error + difference.mul_add(b, -product);
    }
    let (origin, log_mass) = two_sum(high, low);
    normalized_parts(origin, log_mass)
}

// Only used if the ordinary compensated accumulation overflowed. Add an
// opposite-signed term whenever possible, so finite cancellations happen
// before a same-signed remainder can legitimately exceed f64's range.
fn balanced_dot(terms: impl Iterator<Item = (f64, f64)>) -> LogScale {
    let (mut positive, mut negative) = (Vec::new(), Vec::new());
    for (difference, b) in terms {
        let product = difference * b;
        if !product.is_finite() {
            return LogScale::from_log(product);
        }
        let term = (product, difference.mul_add(b, -product));
        if product >= 0.0 {
            positive.push(term);
        } else {
            negative.push(term);
        }
    }
    let (mut high, mut low) = (0.0, 0.0);
    while !positive.is_empty() || !negative.is_empty() {
        let next = if high > 0.0 {
            negative.pop().or_else(|| positive.pop())
        } else {
            positive.pop().or_else(|| negative.pop())
        };
        let (product, product_error) = next.expect("remaining product");
        let (sum, error) = two_sum(high, product);
        if !sum.is_finite() {
            return LogScale::from_log(sum);
        }
        high = sum;
        low += error + product_error;
    }
    let (origin, log_mass) = two_sum(high, low);
    normalized_parts(origin, log_mass)
}

fn normalized_parts(origin: f64, correction: f64) -> LogScale {
    if origin.is_finite() {
        LogScale::from_predictor_parts(origin, correction)
    } else {
        LogScale::from_log(origin)
    }
}

/// Retain both the large component and its correction until offsets cancel.
pub(crate) fn relative_covariate_scale(
    row: ArrayView1<'_, f64>,
    anchor: &[f64],
    beta: &[f64],
) -> LogScale {
    compensated_dot(
        row.iter()
            .zip(anchor)
            .zip(beta)
            .map(|((&x, &origin), &b)| (x - origin, b)),
    )
}

pub(crate) fn add_predictor_parts(left: LogScale, right: LogScale) -> LogScale {
    if left.log_mass.is_nan() {
        return left;
    }
    if right.log_mass.is_nan() {
        return right;
    }
    let (origin, error) = two_sum(left.origin, right.origin);
    if !origin.is_finite() {
        return LogScale::from_log(origin);
    }
    let (correction, correction_error) = two_sum(left.origin_correction, right.origin_correction);
    let (origin, correction) = two_sum(origin, correction + error);
    let (origin, correction) = two_sum(origin, correction + correction_error);
    if !origin.is_finite() {
        return LogScale::from_log(origin);
    }
    LogScale::new_parts(origin, correction, left.log_mass + right.log_mass)
}

impl CoxData {
    pub(super) fn needs_paired_predictors(&self, beta: &[f64]) -> bool {
        self.large_offset || bounds_need_anchor(&self.predictor_bounds, beta)
    }

    pub(super) fn predictor_scales(&self, beta: &[f64]) -> Vec<LogScale> {
        let mut values = vec![LogScale::EMPTY; self.time.len()];
        let mut start = 0;
        for end in 0..values.len() {
            if self.stratum_end[end] != 1 {
                continue;
            }
            let anchor = self.covar.row(end);
            for (person, value) in values.iter_mut().enumerate().take(end + 1).skip(start) {
                *value = add_predictor_parts(
                    LogScale::from_log(self.offset[person]),
                    relative_covariate_scale(
                        self.covar.row(person),
                        anchor.as_slice().expect("row-major covariates"),
                        beta,
                    ),
                );
            }
            start = end + 1;
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn covariate_low_parts_survive_offset_cancellation_and_are_renormalized() {
        let row = array![1.0, 1.0];
        for (large, small) in [(1e16, 0.3), (1e16, 1e8), (1e308, 1e292)] {
            let covariate = relative_covariate_scale(row.view(), &[0.0, 0.0], &[large, small]);
            let predictor = add_predictor_parts(LogScale::from_log(-large), covariate);
            assert_eq!(predictor.origin, small);
            assert_eq!(predictor.log_mass, 0.0);
            let (two, a, b) = predictor.mix(predictor);
            assert_eq!((a, b), (0.5, 0.5));
            assert_eq!(two.log_mass, 2.0_f64.ln());
        }
    }

    #[test]
    fn compensated_dot_recovers_finite_cancellation_before_reporting_overflow() {
        let row = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let result =
            relative_covariate_scale(row.view(), &[0.; 5], &[1e308, 1e308, 0.3, -1e308, -1e308]);
        assert_eq!(result.log_value(), 0.3);
        let overflow = relative_covariate_scale(row.view(), &[0.; 5], &[1e308; 5]);
        assert_eq!(overflow.log_value(), f64::INFINITY);
        assert_eq!(
            add_predictor_parts(LogScale::from_log(-1e308), LogScale::from_log(-1e308)).log_value(),
            f64::NEG_INFINITY
        );
    }
}
