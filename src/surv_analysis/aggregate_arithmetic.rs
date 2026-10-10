//! Ordinary aggregation with R's extended exceptional-case arithmetic.
//!
//! Same-sign normal inputs use a compensated binary64 sum. Its relative error
//! is bounded by a few binary64 ulps in the certified domain below; ordinary
//! results need not match the reference's final bit. Exceptional groups replay
//! R's 64-bit extended arithmetic, with a separately stored exponent so an
//! intermediate sum may exceed the binary64 range. This is not an exact
//! mathematical sum: terms below that precision can still be lost.

/// Accurate ordinary mean with R's finite-sum and scaled-sum exceptional paths.
pub(crate) fn r_mean(values: &[f64]) -> f64 {
    if let [value] = values {
        return if value.is_nan() {
            f64::NAN
        } else if *value == 0.0 {
            0.0
        } else {
            *value
        };
    }
    if let Some(mean) = normal_mean(values) {
        return mean;
    }
    extended_mean(values)
}

/// R's default ungrouped survival aggregation uses `rowMeans`, which divides
/// the sequential extended sum without mean's residual refinement. Ordinary
/// same-sign groups use the same bounded fast path as the explicit mean.
pub(crate) fn r_row_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if let Some(mean) = normal_mean(values) {
        return mean;
    }
    match extended_sum(values) {
        Ok(sum) => sum.divide(values.len() as u64).to_f64(),
        Err(nonfinite) => nonfinite,
    }
}

/// Ordinary totals use a compensated binary64 sum; exceptional groups replay
/// the reference's sequential 64-bit-significand accumulator. In particular,
/// intermediate overflow and very large cancellation retain R's behavior.
pub(crate) fn r_sum(values: &[f64]) -> f64 {
    if let Some(sum) = normal_sum(values) {
        return sum;
    }
    match extended_sum(values) {
        Ok(sum) => sum.to_f64(),
        Err(nonfinite) => nonfinite,
    }
}

fn normal_sum(values: &[f64]) -> Option<f64> {
    if values.len() > 1 << 20 {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    let mut positive = false;
    let mut negative = false;
    for &value in values {
        if value != 0.0 && !value.is_normal() {
            return None;
        }
        positive |= value > 0.0;
        negative |= value < 0.0;
        if positive && negative {
            return None;
        }
        let next = sum + value;
        if next.abs() > f64::MAX / 4.0 {
            return None;
        }
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    Some(sum + correction)
}

/// Neumaier's compensated sum in a conservative noncancelling domain.
///
/// The error-free addition residuals have absolute sum <= n*u*sum|x|,
/// u = 2^-53. Adding these residuals introduces error <= gamma(n)*n*u*sum|x|,
/// gamma(n) = n*u/(1-n*u). For n <= 2^20 this coefficient is < u/4096.
/// Division and the final addition add only a few further u terms. Inputs
/// have one sign, so sum|x| = |sum x|; a normal result bounds division's
/// gradual-underflow error by u*|mean|. The MAX/4 guard leaves room for both
/// the reference's residual sum and rounding error. The mathematical mean's
/// relative error is conservatively bounded by 8*EPSILON; comparison with
/// R's sequential residual sum additionally allows 8*n*2^-64. Mixed signs,
/// subnormal inputs/results, or any failed certificate use exact replay.
fn normal_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.len() > 1 << 20 {
        return None;
    }
    if let &[first, second] = values {
        // In this domain, addition has relative error <= u and multiplying
        // its normal mean by one half is exact. No residual is needed for
        // the documented ordinary bound. Exceptional pairs still replay R.
        if (first != 0.0 && !first.is_normal())
            || (second != 0.0 && !second.is_normal())
            || (first != 0.0
                && second != 0.0
                && first.is_sign_negative() != second.is_sign_negative())
        {
            return None;
        }
        let sum = first + second;
        if sum.abs() > f64::MAX / 4.0 {
            return None;
        }
        if sum == 0.0 {
            return Some(0.0);
        }
        let mean = sum * 0.5;
        return mean.is_normal().then_some(mean);
    }
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    let mut positive = false;
    let mut negative = false;
    for &value in values {
        if value != 0.0 && !value.is_normal() {
            return None;
        }
        positive |= value > 0.0;
        negative |= value < 0.0;
        if positive && negative {
            return None;
        }
        let next = sum + value;
        if !next.is_finite() {
            return None;
        }
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    if sum.abs() > f64::MAX / 4.0 || !correction.is_finite() {
        return None;
    }
    let n = values.len() as f64;
    let mean = sum / n + correction / n;
    if sum != 0.0 && !mean.is_normal() {
        return None;
    }
    Some(mean)
}

/// Literal sequential replay of the reference's 64-bit significand arithmetic.
fn extended_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let sum = match extended_sum(values) {
        Ok(sum) => sum,
        Err(nonfinite) => return nonfinite,
    };

    let n = values.len() as u64;
    let finite_sum = sum.to_f64().is_finite();
    let mut mean = if finite_sum {
        sum.divide(n)
    } else {
        // R divides the binary64 input before promoting each smaller term.
        values.iter().fold(Extended::ZERO, |sum, &value| {
            sum.add(Extended::from_f64(value / values.len() as f64))
        })
    };
    if mean.to_f64().is_finite() {
        let residual = values.iter().fold(Extended::ZERO, |sum, &value| {
            let residual = Extended::from_f64(value).subtract(mean);
            sum.add(if finite_sum {
                residual
            } else {
                residual.divide(n)
            })
        });
        mean = mean.add(if finite_sum {
            residual.divide(n)
        } else {
            residual
        });
    }
    mean.to_f64()
}

fn extended_sum(values: &[f64]) -> Result<Extended, f64> {
    let mut sum = Extended::ZERO;
    let mut positive_infinity = false;
    let mut negative_infinity = false;
    for &value in values {
        if value.is_finite() {
            sum = sum.add(Extended::from_f64(value));
        } else if value.is_nan() {
            return Err(f64::NAN);
        } else if value.is_sign_negative() {
            negative_infinity = true;
        } else {
            positive_infinity = true;
        }
    }
    match (positive_infinity, negative_infinity) {
        (true, true) => return Err(f64::NAN),
        (true, false) => return Err(f64::INFINITY),
        (false, true) => return Err(f64::NEG_INFINITY),
        (false, false) => {}
    }

    Ok(sum)
}

/// A finite number `(-1)^negative * significand * 2^exponent`.
/// Nonzero significands always have their highest bit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extended {
    negative: bool,
    significand: u64,
    exponent: i32,
}

impl Extended {
    const ZERO: Self = Self {
        negative: false,
        significand: 0,
        exponent: 0,
    };

    fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits();
        let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
        let mut significand = bits & ((1 << 52) - 1);
        let exponent = if raw_exponent == 0 {
            -1074
        } else {
            significand |= 1 << 52;
            raw_exponent - 1023 - 52
        };
        Self::round(significand as u128, exponent, bits >> 63 != 0, 0, 1)
    }

    /// Rounds the exact value `(magnitude + remainder/denominator) * 2^exponent`.
    /// Division supplies a quotient with at least 64 bits; smaller magnitudes
    /// arise only from exact addition/subtraction and have no remainder.
    fn round(
        magnitude: u128,
        exponent: i32,
        negative: bool,
        remainder: u64,
        denominator: u64,
    ) -> Self {
        if magnitude == 0 {
            return Self::ZERO;
        }
        let bits = 128 - magnitude.leading_zeros();
        if bits < 64 {
            debug_assert_eq!(remainder, 0);
            let shift = 64 - bits;
            return Self {
                negative,
                significand: (magnitude << shift) as u64,
                exponent: exponent - shift as i32,
            };
        }
        let mut shift = bits - 64;
        let mut top = magnitude >> shift;
        let increment = if shift == 0 {
            let twice = 2 * remainder as u128;
            twice > denominator as u128 || (twice == denominator as u128 && top & 1 != 0)
        } else {
            let low = magnitude & ((1u128 << shift) - 1);
            let half = 1u128 << (shift - 1);
            low > half || (low == half && (remainder != 0 || top & 1 != 0))
        };
        top += u128::from(increment);
        if top == 1u128 << 64 {
            top >>= 1;
            shift += 1;
        }
        Self {
            negative,
            significand: top as u64,
            exponent: exponent + shift as i32,
        }
    }

    fn add(self, other: Self) -> Self {
        if self.significand == 0 {
            return other;
        }
        if other.significand == 0 {
            return self;
        }
        if self.exponent < other.exponent {
            return other.add(self);
        }
        let delta = (self.exponent - other.exponent) as u32;
        if delta > 64 {
            // Below a power of two, spacing is half that above it. An opposite
            // sign term between a quarter and half of the larger unit rounds
            // to the immediately preceding value; the exact quarter tie keeps
            // the even power of two. Every other far term rounds away.
            if self.negative != other.negative
                && self.significand == 1 << 63
                && delta == 65
                && other.significand > 1 << 63
            {
                return Self {
                    negative: self.negative,
                    significand: u64::MAX,
                    exponent: self.exponent - 1,
                };
            }
            return self;
        }
        let left = (self.significand as u128) << delta;
        let right = other.significand as u128;
        if self.negative == other.negative {
            Self::round(left + right, other.exponent, self.negative, 0, 1)
        } else if left >= right {
            Self::round(left - right, other.exponent, self.negative, 0, 1)
        } else {
            Self::round(right - left, other.exponent, other.negative, 0, 1)
        }
    }

    fn subtract(self, other: Self) -> Self {
        self.add(Self {
            negative: !other.negative,
            ..other
        })
    }

    fn divide(self, denominator: u64) -> Self {
        debug_assert_ne!(denominator, 0);
        if self.significand == 0 {
            return self;
        }
        let numerator = (self.significand as u128) << 64;
        Self::round(
            numerator / denominator as u128,
            self.exponent - 64,
            self.negative,
            (numerator % denominator as u128) as u64,
            denominator,
        )
    }

    fn to_f64(self) -> f64 {
        if self.significand == 0 {
            return 0.0;
        }
        let sign = u64::from(self.negative) << 63;
        let mut exponent = self.exponent + 63;
        if exponent < -1022 {
            let shift = (-self.exponent - 1074) as u32;
            let fraction = if shift > 64 {
                0
            } else {
                let magnitude = self.significand as u128;
                let top = magnitude >> shift;
                let low = magnitude & ((1u128 << shift) - 1);
                let half = 1u128 << (shift - 1);
                (top + u128::from(low > half || (low == half && top & 1 != 0))) as u64
            };
            return f64::from_bits(sign | fraction);
        }
        let mut top = self.significand >> 11;
        let low = self.significand & 0x7ff;
        top += u64::from(low > 0x400 || (low == 0x400 && top & 1 != 0));
        if top == 1 << 53 {
            top >>= 1;
            exponent += 1;
        }
        if exponent > 1023 {
            return f64::from_bits(sign | (0x7ff << 52));
        }
        f64::from_bits(sign | (((exponent + 1023) as u64) << 52) | (top & ((1 << 52) - 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_rounds_ties_carries_and_the_power_of_two_boundary() {
        let one = Extended::from_f64(1.0);
        let half = Extended::from_f64(2.0f64.powi(-64));
        assert_eq!(one.add(half), one);
        let odd = Extended {
            significand: one.significand + 1,
            ..one
        };
        assert_eq!(odd.add(half).significand, one.significand + 2);
        let before = Extended {
            negative: false,
            significand: u64::MAX,
            exponent: -64,
        };
        assert_eq!(one.subtract(half), before);
        let quarter = Extended::from_f64(2.0f64.powi(-65));
        assert_eq!(one.subtract(quarter), one);
        assert_eq!(
            one.subtract(Extended {
                significand: quarter.significand + 1,
                ..quarter
            }),
            before
        );
        assert_eq!(one.subtract(Extended::from_f64(2.0f64.powi(-66))), one);
        assert_eq!(
            Extended {
                negative: false,
                significand: u64::MAX,
                exponent: -63,
            }
            .add(half),
            Extended::from_f64(2.0)
        );
        assert_eq!(one.subtract(one), Extended::ZERO);
        let maximum = Extended::from_f64(f64::MAX);
        assert!(maximum.add(maximum).to_f64().is_infinite());
        assert_eq!(maximum.add(maximum).subtract(maximum), maximum);
    }

    #[test]
    fn division_and_binary64_conversion_round_without_losing_subnormals() {
        let one = Extended::from_f64(1.0);
        assert_eq!(
            one.divide(3),
            Extended {
                negative: false,
                significand: 0xaaaa_aaaa_aaaa_aaab,
                exponent: -65,
            }
        );
        assert_eq!(
            Extended {
                negative: false,
                significand: u64::MAX,
                exponent: 0,
            }
            .divide(u64::MAX),
            one
        );
        let even_midpoint = Extended {
            significand: one.significand + 1024,
            ..one
        };
        assert_eq!(even_midpoint.to_f64().to_bits(), 1.0f64.to_bits());
        let odd_midpoint = Extended {
            significand: one.significand + 3072,
            ..one
        };
        assert_eq!(odd_midpoint.to_f64().to_bits(), 1.0f64.to_bits() + 2);
        let half_minimum = Extended {
            exponent: -1138,
            ..one
        };
        assert_eq!(half_minimum.to_f64().to_bits(), 0);
        assert_eq!(
            Extended {
                significand: half_minimum.significand + 1,
                ..half_minimum
            }
            .to_f64()
            .to_bits(),
            1
        );
        assert_eq!(
            Extended {
                negative: true,
                ..half_minimum
            }
            .to_f64()
            .to_bits(),
            1 << 63
        );
        for value in [
            0.0,
            1.0,
            -1.0,
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            assert_eq!(
                Extended::from_f64(value).to_f64().to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn mean_keeps_infinities_and_cancellation_after_wide_intermediate_sums() {
        assert_eq!(r_mean(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert_eq!(r_mean(&[f64::NEG_INFINITY, 1.0]), f64::NEG_INFINITY);
        assert!(r_mean(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(r_mean(&[f64::NAN, f64::INFINITY]).is_nan());
        assert_eq!(r_mean(&[f64::MAX, f64::MAX]), f64::MAX);
        assert_eq!(
            r_mean(&[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1.0]),
            0.36
        );
        let tiny = f64::from_bits(1);
        assert_eq!(r_mean(&[tiny, tiny, tiny]), tiny);
        // The reference accumulator rounds each operation to 64 bits; it is
        // intentionally distinct from the exact mathematical mean 1/3 here.
        assert_eq!(r_mean(&[1e300, 1.0, -1e300]), 0.0);
    }

    #[test]
    fn sum_replays_sequential_extended_rounding_on_exceptional_groups() {
        for (values, expected) in [
            (vec![1e16, 1.0, -1e16], 1.0),
            (vec![1e300, 1.0, -1e300], 0.0),
            (vec![1e308, 1e308, -1e308], 1e308),
            (vec![f64::MAX, f64::MAX, -f64::MAX], f64::MAX),
            (vec![f64::INFINITY, 1.0], f64::INFINITY),
            (vec![f64::NEG_INFINITY, 1.0], f64::NEG_INFINITY),
            (vec![f64::INFINITY, f64::NEG_INFINITY], f64::NAN),
            (vec![f64::NAN, 1.0], f64::NAN),
            (vec![f64::from_bits(1); 3], f64::from_bits(3)),
        ] {
            let actual = r_sum(&values);
            assert!(
                (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
            );
        }
        assert_eq!(r_sum(&[]), 0.0);
        assert_eq!(r_sum(&[-0.0]).to_bits(), 0);
        assert!(normal_sum(&vec![1.0; (1 << 20) + 1]).is_none());
        for values in [vec![1.0, 2.0, 3.0], vec![0.0; 3], vec![-1.0; 7]] {
            assert_eq!(normal_sum(&values).unwrap(), r_sum(&values));
        }
    }

    #[test]
    fn row_mean_divides_extended_sum_without_residual_refinement() {
        assert_eq!(
            r_row_mean(&[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1.0]),
            0.2
        );
        assert_eq!(
            r_mean(&[f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, 1.0]),
            0.36
        );
        assert_eq!(r_row_mean(&[1e308, 1e308, -1e308]), 1e308 / 3.0);
        assert_eq!(r_row_mean(&[1e300, 1.0, -1e300]), 0.0);
        assert_eq!(r_row_mean(&[f64::INFINITY, 1.0]), f64::INFINITY);
        assert!(r_row_mean(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(r_row_mean(&[]).is_nan());
        assert_eq!(r_row_mean(&[0.1, 0.2, 0.3]), r_mean(&[0.1, 0.2, 0.3]));
    }

    #[test]
    fn ordinary_compensated_domain_has_bounded_error_and_exceptional_replay() {
        for values in [
            vec![f64::INFINITY, 1.0],
            vec![f64::NAN, 1.0],
            vec![1.0, -1.0],
            vec![f64::MAX, f64::MAX],
            vec![f64::from_bits(1), f64::from_bits(1)],
            vec![f64::MIN_POSITIVE, 0.0, 0.0],
        ] {
            assert!(normal_mean(&values).is_none(), "{values:?}");
            let actual = r_mean(&values);
            let replay = extended_mean(&values);
            assert!((actual.is_nan() && replay.is_nan()) || actual.to_bits() == replay.to_bits());
        }
        assert!(normal_mean(&vec![1.0; (1 << 20) + 1]).is_none());
        assert_eq!(r_mean(&[-0.0]).to_bits(), 0);
        assert_eq!(r_mean(&[f64::from_bits(1)]).to_bits(), 1);
        for bits in [0x7ff8_0000_0000_0001, 0xfff8_0000_0000_0001] {
            assert_eq!(
                r_mean(&[f64::from_bits(bits)]).to_bits(),
                f64::NAN.to_bits()
            );
        }
        for values in [
            [0.0, -0.0],
            [0.0, f64::MIN_POSITIVE * 2.0],
            [1.0, f64::from_bits(1.0_f64.to_bits() + 1)],
            [f64::MAX / 16.0, f64::MAX / 16.0],
            [-1.0, -f64::from_bits(1.0_f64.to_bits() + 1)],
        ] {
            let actual = normal_mean(&values).unwrap();
            let reference = extended_mean(&values);
            assert!((actual - reference).abs() <= f64::EPSILON * reference.abs());
            assert_eq!(actual.to_bits(), r_mean(&values).to_bits());
        }
        for values in [
            [f64::MIN_POSITIVE, 0.0],
            [f64::MAX / 4.0, f64::MAX / 4.0],
            [-1.0, 1.0],
            [f64::INFINITY, 1.0],
        ] {
            assert!(normal_mean(&values).is_none());
            assert_eq!(r_mean(&values).to_bits(), extended_mean(&values).to_bits());
        }

        let mut seed = 5781_u64;
        for n in [2, 3, 7, 16, 64, 257, 4096] {
            for scale in [1e-100, 1.0, 1e100, -1e-100, -1.0, -1e100] {
                for _ in 0..8 {
                    let values: Vec<_> = (0..n)
                        .map(|_| {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                            scale * ((seed >> 12) as f64 / (1_u64 << 52) as f64)
                        })
                        .collect();
                    let actual = normal_mean(&values).unwrap();
                    let reference = extended_mean(&values);
                    let bound = 8.0 * f64::EPSILON + 8.0 * n as f64 * 2.0_f64.powi(-64);
                    assert!(
                        (actual - reference).abs() <= bound * reference.abs(),
                        "n={n} scale={scale}: {actual} != {reference}"
                    );
                    assert_eq!(actual.to_bits(), r_mean(&values).to_bits());
                }
            }
        }
    }
}
