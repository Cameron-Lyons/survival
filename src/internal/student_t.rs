//! Student-t probabilities from the beta integral and density integration.
//!
//! These formulas are derived from the defining density and incomplete beta
//! identities. No implementation code from R or other GPL sources is used.

use std::f64::consts::{LN_2, PI, SQRT_2};

const LOG_SQRT_2_PI: f64 = 0.918_938_533_204_672_8;

#[derive(Clone, Copy, Debug)]
pub(crate) struct StudentT {
    df: f64,
    sqrt_df: f64,
    log_df: f64,
    log_normalizer: f64,
    log_double_tail_coefficient: f64,
}

impl StudentT {
    pub(crate) fn new(df: f64) -> Self {
        let log_df = df.ln();
        let a = 0.5 * df;
        let (log_normalizer, log_double_tail_coefficient) = if a < 0.001 {
            // log Γ(a+1/2)-log Γ(a+1)-log Γ(1/2), expanded at zero.
            // Keeping log(df) separately also handles df/2 underflow.
            let correction = a
                * (-2.0 * LN_2
                    + a * (PI * PI / 6.0
                        + a * (-2.404_113_806_319_188_5
                            + a * (7.0 * PI.powi(4) / 180.0 - a * 6.221_566_530_860_22))));
            let tail = -LN_2 + correction;
            (tail + 0.5 * log_df, correction)
        } else {
            let mut shifted = a;
            let mut recurrence = 0.0;
            while shifted < 32.0 {
                recurrence += 0.5 * (1.0 / shifted).ln_1p() - (0.5 / shifted).ln_1p();
                shifted += 1.0;
            }
            // Bernoulli expansion of log Γ(a+1/2)-log Γ(a)-log(a)/2.
            let inverse = shifted.recip();
            let square = inverse * inverse;
            let ratio = inverse
                * (-1.0 / 8.0
                    + square
                        * (1.0 / 192.0
                            + square
                                * (-1.0 / 640.0
                                    + square * (17.0 / 14_336.0 - square * 31.0 / 18_432.0))));
            let normalizer = -LOG_SQRT_2_PI + ratio + recurrence;
            (normalizer, normalizer - 0.5 * log_df + LN_2)
        };
        Self {
            df,
            sqrt_df: df.sqrt(),
            log_df,
            log_normalizer,
            log_double_tail_coefficient,
        }
    }

    pub(crate) fn degrees_of_freedom(self) -> f64 {
        self.df
    }

    fn is_valid(self) -> bool {
        self.df.is_finite() && self.df > 0.0
    }

    fn log_one_plus_square(self, x: f64) -> f64 {
        if x <= self.sqrt_df {
            let ratio = x / self.sqrt_df;
            (ratio * ratio).ln_1p()
        } else {
            let inverse = self.sqrt_df / x;
            2.0 * x.ln() - self.log_df + (inverse * inverse).ln_1p()
        }
    }

    pub(crate) fn log_pdf(self, x: f64) -> f64 {
        if !self.is_valid() || x.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let logarithm = self.log_one_plus_square(x.abs());
        // Splitting the exponent preserves the df term when df+1 rounds to 1.
        self.log_normalizer - 0.5 * logarithm - (0.5 * self.df) * logarithm
    }

    pub(crate) fn pdf(self, x: f64) -> f64 {
        self.log_pdf(x).exp()
    }

    fn center_limit(self) -> f64 {
        self.sqrt_df.min(1.0)
    }

    /// Integral of the density from zero to x, with x <= min(1,sqrt(df)).
    fn center_mass(self, x: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        if self.df == 1.0 {
            return x.atan() / PI;
        }
        if self.df == 2.0 {
            return 0.5 * (x / x.hypot(SQRT_2));
        }
        let ratio = x / self.sqrt_df;
        let square = ratio * ratio;
        let y = square / (1.0 + square);
        // Euler's positive series for I_y(1/2,df/2). Avoid sqrt(y), which
        // would discard a representable central mass when x*x underflows.
        let a_y = (0.5 * x) * x / (1.0 + square);
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut compensation = 0.0;
        for k in 1..=256 {
            let k = f64::from(k);
            term *= (a_y + (k - 0.5) * y) / (k + 0.5);
            let adjusted = term - compensation;
            let next = sum + adjusted;
            compensation = (next - sum) - adjusted;
            sum = next;
            if term <= f64::EPSILON * sum {
                break;
            }
        }
        x * self.pdf(x) * sum
    }

    fn log_tail(self, x: f64) -> f64 {
        self.log_double_tail(x) - LN_2
    }

    // Keeping log(2*tail) avoids subtracting numbers near -ln(2) when
    // df is tiny and finite quantiles lie far outside the local center.
    fn log_double_tail(self, x: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        if x.is_infinite() {
            return f64::NEG_INFINITY;
        }
        if self.df == 1.0 {
            return if x <= 1.0 {
                (-2.0 * x.atan() / PI).ln_1p()
            } else {
                (x.recip().atan() / PI).ln() + LN_2
            };
        }
        if self.df == 2.0 {
            let radius = x.hypot(SQRT_2);
            return -2.0 * radius.ln() - (x / radius).ln_1p() + LN_2;
        }
        if x <= self.center_limit() {
            return (-2.0 * self.center_mass(x)).ln_1p();
        }
        if x >= 9.0 {
            return self.log_tail_series(x);
        }
        if self.df >= 100_000.0 {
            return self.normal_limit_tail(x).ln() + LN_2;
        }
        if self.df >= 1_000.0 {
            return self.normal_integral_tail(x).ln() + LN_2;
        }

        let a = 0.5 * self.df;
        let logarithm = self.log_one_plus_square(x);
        let log_z = -logarithm;
        let z = log_z.exp();
        if z <= 0.5 {
            // I_z(a,1/2)=z^a/(a B(a,1/2))*sum positive hypergeometric terms.
            // log(z) remains usable even if z itself underflows to zero.
            let mut term = 1.0;
            let mut sum = 0.0;
            for k in 1..=256 {
                let k = f64::from(k);
                term *= ((a + (k - 1.0)) / (a + k)) * ((k - 0.5) / k) * z;
                sum += term;
                if term <= f64::EPSILON * sum.abs() {
                    break;
                }
            }
            return self.log_double_tail_coefficient + a * log_z + sum.ln_1p();
        }
        let ratio = x / self.sqrt_df;
        let square = ratio * ratio;
        let y = square / (1.0 + square);
        let log_y = y.ln();
        let log_beta = -self.log_normalizer - 0.5 * self.log_df;
        let front = a * log_z + 0.5 * log_y - log_beta;
        if z < (a + 1.0) / (a + 2.5) {
            front + beta_fraction(a, 0.5, z).ln() - a.ln()
        } else {
            let log_center = front + beta_fraction(0.5, a, y).ln() + LN_2;
            log_one_minus_exp(log_center)
        }
    }

    fn log_tail_series(self, x: f64) -> f64 {
        // Integration by parts gives Q/f=(1+x²/df)/x * sum, with term ratio
        // -(2k-1)/(x²*(1+2k/df)). Its alternating remainder is bounded by the
        // next term. For x>=9 it reaches floating-point accuracy before terms
        // can increase, including in the normal limit.
        let inverse = x.recip();
        let inverse_square = inverse * inverse;
        let mut term = 1.0;
        let mut sum = 0.0;
        for k in 1..=256 {
            let k = f64::from(k);
            term *= -((2.0 * k - 1.0) / (1.0 + 2.0 * k / self.df)) * inverse_square;
            sum += term;
            if term.abs() <= f64::EPSILON * self.df.min(1.0) {
                break;
            }
        }
        let logarithm = self.log_one_plus_square(x);
        if x <= self.sqrt_df {
            self.log_pdf(x) + logarithm - x.ln() + sum.ln_1p() + LN_2
        } else {
            let ratio = self.sqrt_df / x;
            self.log_double_tail_coefficient - (0.5 * self.df) * logarithm
                + 0.5 * (ratio * ratio).ln_1p()
                + sum.ln_1p()
        }
    }

    fn normal_limit_tail(self, x: f64) -> f64 {
        // Integrating the density's expansion in 1/df yields Q_normal+phi*P.
        // Independent high-precision density quadrature sampled df>=1e5,
        // x<=9; the largest observed relative remainder was 1.4e-17.
        const POLYNOMIALS: [&[f64]; 6] = [
            &[1.0 / 4.0, 1.0 / 4.0],
            &[-1.0 / 32.0, -5.0 / 96.0, -7.0 / 96.0, 1.0 / 32.0],
            &[
                -5.0 / 128.0,
                -1.0 / 128.0,
                1.0 / 64.0,
                7.0 / 192.0,
                -11.0 / 384.0,
                1.0 / 384.0,
            ],
            &[
                21.0 / 2048.0,
                61.0 / 6144.0,
                -71.0 / 30720.0,
                -313.0 / 30720.0,
                -2141.0 / 92160.0,
                445.0 / 18432.0,
                -25.0 / 6144.0,
                1.0 / 6144.0,
            ],
            &[
                399.0 / 8192.0,
                119.0 / 8192.0,
                1.0 / 2048.0,
                19.0 / 6144.0,
                83.0 / 12288.0,
                333.0 / 20480.0,
                -1879.0 / 92160.0,
                49.0 / 10240.0,
                -133.0 / 368640.0,
                1.0 / 122880.0,
            ],
            &[
                -869.0 / 65536.0,
                -2465.0 / 196608.0,
                137.0 / 196608.0,
                1949.0 / 1376256.0,
                -12805.0 / 6193152.0,
                -146047.0 / 30965760.0,
                -75113.0 / 6193152.0,
                231253.0 / 13271040.0,
                -135149.0 / 26542080.0,
                107.0 / 196608.0,
                -23.0 / 983040.0,
                1.0 / 2949120.0,
            ],
        ];
        let square = x * x;
        let inverse_df = self.df.recip();
        let mut power = inverse_df;
        let mut correction = 0.0;
        for coefficients in POLYNOMIALS {
            let polynomial = coefficients
                .iter()
                .rev()
                .fold(0.0, |value, &coefficient| value * square + coefficient);
            correction += x * polynomial * power;
            power *= inverse_df;
        }
        0.5 * libm::erfc(x / SQRT_2) + (-0.5 * square - LOG_SQRT_2_PI).exp() * correction
    }

    fn normal_integral_tail(self, x: f64) -> f64 {
        // Under w²=df*log(1+x²/df), the Student density becomes
        // sqrt(2π)*c_df*phi(w)*g(w²/df), g(v)=sqrt(v/(1-exp(-v))).
        // Here df>=1e3 and x<9, so v<.081 at the lower integration limit.
        // Integrating g through v^8 has relative truncation error below
        // 5e-16; see docs/student-t-normal-limit.md. Rounding dominates.
        let ratio = (x / self.sqrt_df).powi(2);
        let w = if ratio < 1e-8 {
            // The ratio can underflow even though x is representable.
            x * (1.0 + ratio * (-0.25 + ratio * (13.0 / 96.0)))
        } else {
            (self.df * ratio.ln_1p()).sqrt()
        };
        let square = w * w;
        let tail = 0.5 * libm::erfc(w / SQRT_2);
        let density = (-0.5 * square - LOG_SQRT_2_PI).exp();
        let coefficients = [
            0.25,
            1.0 / 96.0,
            -1.0 / 384.0,
            -1.0 / 10_240.0,
            19.0 / 368_640.0,
            79.0 / 61_931_520.0,
            -55.0 / 49_545_216.0,
            -2_339.0 / 118_908_518_400.0,
        ];
        let mut moment = tail;
        let mut power = w;
        let inverse_df = self.df.recip();
        let mut inverse_power = 1.0;
        let mut correction = 0.0;
        for (index, coefficient) in coefficients.into_iter().enumerate() {
            // M_2k = w^(2k-1)*phi(w) + (2k-1)*M_(2k-2).
            moment = power * density + (2 * index + 1) as f64 * moment;
            inverse_power *= inverse_df;
            correction += coefficient * inverse_power * moment;
            power *= square;
        }
        (self.log_normalizer + LOG_SQRT_2_PI).exp() * (tail + correction)
    }

    pub(crate) fn cdf(self, x: f64) -> f64 {
        if !self.is_valid() || x.is_nan() {
            return f64::NAN;
        }
        if x == 0.0 {
            return 0.5;
        }
        let absolute = x.abs();
        if absolute <= self.center_limit() {
            return 0.5 + self.center_mass(absolute).copysign(x);
        }
        let log_tail = if self.df < 0.001 {
            let log_double_tail = self.log_double_tail(absolute);
            if log_double_tail > -LN_2 {
                let mass = -0.5 * log_double_tail.exp_m1();
                return 0.5 + mass.copysign(x);
            }
            log_double_tail - LN_2
        } else {
            self.log_sf(absolute)
        };
        if x < 0.0 {
            log_tail.exp()
        } else {
            -log_tail.exp_m1()
        }
    }

    pub(crate) fn log_sf(self, x: f64) -> f64 {
        if !self.is_valid() || x.is_nan() {
            return f64::NAN;
        }
        if x >= 0.0 {
            self.log_tail(x)
        } else {
            log_one_minus_exp(self.log_tail(-x))
        }
    }

    pub(crate) fn inverse_cdf(self, probability: f64) -> f64 {
        if !self.is_valid() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return f64::NAN;
        }
        if probability == 0.0 {
            return f64::NEG_INFINITY;
        }
        if probability == 1.0 {
            return f64::INFINITY;
        }
        if probability == 0.5 {
            return 0.0;
        }
        let sign = if probability < 0.5 { -1.0 } else { 1.0 };
        let tail = if probability < 0.5 {
            probability
        } else {
            1.0 - probability
        };
        let center = (probability - 0.5).abs();
        if self.df == 1.0 {
            return sign
                * if center <= 0.25 {
                    (PI * center).tan()
                } else {
                    (PI * tail).tan().recip()
                };
        }
        if self.df == 2.0 {
            return sign * (2.0 * center) / ((2.0 * tail) * (1.0 - tail)).sqrt();
        }
        let center_limit = self.center_limit();
        if center <= self.center_mass(center_limit) {
            let mut low = 0.0;
            let mut high = center_limit;
            let mut x = center / self.log_normalizer.exp();
            for _ in 0..64 {
                let error = self.center_mass(x) - center;
                if error > 0.0 {
                    high = x;
                } else {
                    low = x;
                }
                let correction = error / self.pdf(x);
                if correction.abs() <= 4.0 * f64::EPSILON * x {
                    break;
                }
                let candidate = x - correction;
                x = if candidate > low && candidate < high {
                    candidate
                } else {
                    0.5 * (low + high)
                };
            }
            return sign * x;
        }
        let target = if center < 0.25 {
            (-2.0 * center).ln_1p()
        } else {
            tail.ln() + LN_2
        };
        let max_log = f64::MAX.ln();
        if self.log_double_tail(f64::MAX) > target {
            return sign * f64::INFINITY;
        }
        let mut low = center_limit.ln();
        let mut high = max_log;
        let guess = if self.df <= 32.0 {
            0.5 * self.log_df + (self.log_double_tail_coefficient - target) / self.df
        } else {
            super::statistical::normal_inverse_cdf(tail).abs().ln()
        };
        let mut log_x = guess.clamp(low, high);
        for _ in 0..128 {
            let x = log_x.exp();
            let log_tail = self.log_double_tail(x);
            let error = log_tail - target;
            if error > 0.0 {
                low = log_x;
            } else {
                high = log_x;
            }
            let slope = (log_x + self.log_pdf(x) - log_tail + LN_2).exp();
            let correction = error / slope;
            if correction.abs() <= 4.0 * f64::EPSILON * log_x.abs().max(1.0) {
                break;
            }
            let candidate = log_x + correction;
            log_x = if candidate.is_finite() && candidate > low && candidate < high {
                candidate
            } else {
                0.5 * (low + high)
            };
        }
        sign * log_x.exp()
    }
}

fn log_one_minus_exp(value: f64) -> f64 {
    if value < -LN_2 {
        (-value.exp()).ln_1p()
    } else {
        (-value.exp_m1()).ln()
    }
}

fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    // Modified Lentz evaluation of the incomplete-beta continued fraction.
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let safe = |value: f64| {
        if value.abs() < tiny {
            tiny.copysign(value)
        } else {
            value
        }
    };
    let mut c = 1.0;
    let mut d = safe(1.0 - (a + b) * x / (a + 1.0)).recip();
    let mut fraction = d;
    for k in 1..=1024 {
        let k = f64::from(k);
        let twice = 2.0 * k;
        let even = k * (b - k) * x / ((a + twice - 1.0) * (a + twice));
        d = safe(1.0 + even * d).recip();
        c = safe(1.0 + even / c);
        fraction *= c * d;
        let odd = -(a + k) * (a + b + k) * x / ((a + twice) * (a + twice + 1.0));
        d = safe(1.0 + odd * d).recip();
        c = safe(1.0 + odd / c);
        let delta = c * d;
        fraction *= delta;
        if (delta - 1.0).abs() <= 4.0 * f64::EPSILON {
            break;
        }
    }
    fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_number(value: &serde_json::Value) -> f64 {
        match value.as_str() {
            Some("Inf") => f64::INFINITY,
            Some("-Inf") => f64::NEG_INFINITY,
            Some(number) => number.parse().expect("numeric reference"),
            None => value.as_f64().expect("numeric reference"),
        }
    }

    #[test]
    fn student_t_normal_limit_matches_r_at_method_boundaries() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/student_t_normal_reference.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let df = reference_number(&case["df"]);
            let distribution = StudentT::new(df);
            let mut previous = 0.0;
            for (index, x) in case["x"].as_array().unwrap().iter().enumerate() {
                let x = reference_number(x);
                let expected = reference_number(&case["cdf"][index]);
                let actual = distribution.cdf(x);
                assert!(actual > 0.0 && actual >= previous, "df={df}, x={x}");
                assert!(
                    (actual - expected).abs() <= 1e-13 * expected,
                    "df={df}, x={x}: actual={actual:e}, expected={expected:e}"
                );
                let expected_log = reference_number(&case["log_cdf"][index]);
                assert!(
                    (distribution.log_sf(-x) - expected_log).abs() <= 1e-13,
                    "log tail df={df}, x={x}"
                );
                previous = actual;
            }
            let mut previous = f64::NEG_INFINITY;
            for (index, probability) in case["p"].as_array().unwrap().iter().enumerate() {
                let probability = reference_number(probability);
                let expected = reference_number(&case["quantile"][index]);
                let actual = distribution.inverse_cdf(probability);
                assert!(actual >= previous, "df={df}, p={probability}");
                assert!(
                    (actual - expected).abs() <= 1e-13 * expected.abs(),
                    "df={df}, p={probability}: actual={actual:e}, expected={expected:e}"
                );
                previous = actual;
            }
        }
    }

    #[test]
    fn student_t_independent_reference_grid() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/student_t_numerics_reference.json"
        ))
        .unwrap();
        let mut failures = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let df = reference_number(&case["df"]);
            let distribution = StudentT::new(df);
            for point in case["points"].as_array().unwrap() {
                let x = reference_number(&point["x"]);
                for (name, actual, relative) in [
                    ("pdf", distribution.pdf(x), 3e-12),
                    ("cdf", distribution.cdf(x), 3e-11),
                ] {
                    let expected = reference_number(&point[name]);
                    let ulp = expected.next_up() - expected;
                    let tolerance = if name == "cdf" && expected >= 0.25 {
                        2.0 * ulp
                    } else {
                        (relative * expected).max(2.0 * ulp)
                    };
                    if (actual - expected).abs() > tolerance
                        || actual.is_nan()
                        || (expected > 0.0 && actual == 0.0)
                    {
                        failures.push(format!(
                            "df={df}, x={x}, {name}: actual {actual:e}, expected {expected:e}"
                        ));
                    }
                }
                for (name, actual) in [
                    ("log_pdf", distribution.log_pdf(x)),
                    ("log_cdf", distribution.log_sf(-x)),
                ] {
                    let expected = reference_number(&point[name]);
                    let matches = if expected.is_infinite() {
                        actual == expected
                    } else {
                        (actual - expected).abs() <= 3e-12 + 5e-14 * expected.abs()
                    };
                    if !matches {
                        failures.push(format!(
                            "df={df}, x={x}, {name}: actual {actual:e}, expected {expected:e}"
                        ));
                    }
                }
            }
            for quantile in case["quantiles"].as_array().unwrap() {
                let probability = reference_number(&quantile["p"]);
                let expected = reference_number(&quantile["expected"]);
                let actual = distribution.inverse_cdf(probability);
                let matches = if expected.is_infinite() || expected == 0.0 {
                    actual == expected
                } else {
                    (actual - expected).abs() <= 3e-11 * expected.abs()
                };
                if !matches {
                    failures.push(format!(
                        "df={df}, p={probability:e}: actual {actual:e}, expected {expected:e}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn student_t_tiny_df_quantiles_outside_local_center() {
        // Independent high-precision density integration after the change of
        // variable x=sqrt(df)*sinh(u), which resolves these very broad tails.
        for (df, probability, expected) in [
            (2e-19, 0.5_f64.next_down(), -2.699_924_628_446_226_5e231),
            (1e-18, 0.5_f64.next_down(), -8.228_929_318_229_573e38),
            (1e-18, 0.5_f64.next_up(), 1.354_305_554_488_381_2e87),
            (1e-16, 0.5_f64.next_down(), -1.352_774_868_521_158_2e-8),
            (1e-16, 0.5_f64.next_up(), 4.551_439_015_188_660_5e-8),
            (1e-16, 0.5 - 1e-15, -2.387_356_098_652_416_5),
        ] {
            let distribution = StudentT::new(df);
            let actual = distribution.inverse_cdf(probability);
            assert!(
                (actual - expected).abs() <= 3e-11 * expected.abs(),
                "df={df}, p={probability}: actual={actual:e}, expected={expected:e}"
            );
            assert_eq!(distribution.cdf(actual), probability);
        }
    }

    #[test]
    fn student_t_all_finite_positive_df_boundaries() {
        let smallest = StudentT::new(f64::from_bits(1));
        for x in [-f64::MAX, -1.0, 0.0, 1.0, f64::MAX] {
            assert_eq!(smallest.cdf(x), 0.5);
            assert!(!smallest.pdf(x).is_nan());
        }
        assert_eq!(smallest.inverse_cdf(0.5_f64.next_up()), f64::INFINITY);
        assert_eq!(smallest.inverse_cdf(0.5_f64.next_down()), f64::NEG_INFINITY);
        let largest = StudentT::new(f64::MAX);
        for x in [0.0, 1.0, 9.0, 38.0, 1e155, f64::MAX] {
            assert!(!largest.pdf(x).is_nan());
            assert!(!largest.cdf(x).is_nan());
            assert!(!largest.log_sf(x).is_nan());
        }
        for p in [f64::from_bits(1), 1e-300, 0.25, 0.5_f64.next_up()] {
            assert!(largest.inverse_cdf(p).is_finite());
        }
        for df in [f64::from_bits(1), 0.1, 2.0, f64::MAX] {
            let distribution = StudentT::new(df);
            assert_eq!(distribution.inverse_cdf(0.0), f64::NEG_INFINITY);
            assert_eq!(distribution.inverse_cdf(1.0), f64::INFINITY);
            assert_eq!(distribution.inverse_cdf(0.5), 0.0);
            assert!(distribution.pdf(f64::NAN).is_nan());
            assert!(distribution.cdf(f64::NAN).is_nan());
            assert!(distribution.inverse_cdf(f64::NAN).is_nan());
        }
        assert!(StudentT::new(2.0).log_sf(f64::MAX).is_finite());
    }
}
