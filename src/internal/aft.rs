use crate::internal::statistical::ln_gamma;

/// Width on the model response scale, preserving adjacent log-time endpoints.
pub(crate) fn transformed_interval_width(lower: f64, upper: f64, uses_log_time: bool) -> f64 {
    if uses_log_time {
        let relative_width = (upper - lower) / lower;
        if relative_width.is_finite() {
            relative_width.ln_1p()
        } else {
            upper.ln() - lower.ln()
        }
    } else {
        upper - lower
    }
}

/// Log-likelihood calculations shared by AFT fitting and diagnostics. The two tails are
/// evaluated directly, so a small survival probability is never `1 - cdf`.
#[derive(Clone, Copy)]
pub(crate) enum AftDistribution {
    Extreme,
    Logistic,
    Gaussian,
    StudentT {
        df: f64,
        root_df: f64,
        log_df: f64,
        log_normalizer: f64,
    },
}

#[derive(Clone, Copy)]
struct AftDensity {
    log_density: f64,
    score: f64,
    curvature: f64,
}

#[derive(Clone, Copy)]
struct StudentCoordinates {
    // z/sqrt(df+z²), sqrt(df)/sqrt(df+z²), and 1/sqrt(df+z²).
    location: f64,
    remainder: f64,
    inverse_root: f64,
    log_denominator: f64,
}

fn student_log_normalizer(df: f64) -> f64 {
    if df >= 256.0 {
        // Gamma-ratio expansion. The first omitted term is 17/(112*df^7),
        // below 3e-18 here; subtracting separate lgamma values loses accuracy.
        let inverse = df.recip();
        let square = inverse * inverse;
        -0.5 * std::f64::consts::TAU.ln() - inverse / 4.0
            + inverse * square * (1.0 / 24.0 - square / 20.0)
    } else {
        ln_gamma((df + 1.0) / 2.0)
            - ln_gamma(df / 2.0)
            - 0.5 * (df.ln() + std::f64::consts::PI.ln())
    }
}

fn student_coordinates(z: f64, root_df: f64, log_df: f64) -> StudentCoordinates {
    if z.abs() > root_df {
        let relative = root_df / z.abs();
        let square = relative * relative;
        let root = (1.0 + square).sqrt();
        StudentCoordinates {
            location: z.signum() / root,
            remainder: relative / root,
            inverse_root: z.abs().recip() / root,
            log_denominator: 2.0 * z.abs().ln() - log_df + square.ln_1p(),
        }
    } else {
        let relative = z / root_df;
        let square = relative * relative;
        let inverse_root = (1.0 + square).sqrt().recip();
        StudentCoordinates {
            location: relative * inverse_root,
            remainder: inverse_root,
            inverse_root: inverse_root / root_df,
            log_denominator: square.ln_1p(),
        }
    }
}

fn student_density(df: f64, log_normalizer: f64, values: StudentCoordinates) -> AftDensity {
    AftDensity {
        log_density: log_normalizer - 0.5 * (df + 1.0) * values.log_denominator,
        score: -(df + 1.0) * values.location * values.inverse_root,
        curvature: (df + 1.0)
            * (values.location * values.location - values.remainder * values.remainder)
            * values.inverse_root
            * values.inverse_root,
    }
}

/// Continued fraction for incomplete beta. The beta normalizer is supplied by
/// the distribution, so evaluating observations never recomputes gamma values.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const FLOOR: f64 = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < FLOOR {
        d = FLOOR;
    }
    d = d.recip();
    let mut value = d;
    for index in 1..=200 {
        let index = f64::from(index);
        let twice = 2.0 * index;
        for (half, numerator) in [
            (index / (a + twice)) * ((b - index) / (a + twice - 1.0)) * x,
            -((a + index) / (a + twice)) * ((a + b + index) / (a + twice + 1.0)) * x,
        ]
        .into_iter()
        .enumerate()
        {
            d = 1.0 + numerator * d;
            if d.abs() < FLOOR {
                d = FLOOR;
            }
            c = 1.0 + numerator / c;
            if c.abs() < FLOOR {
                c = FLOOR;
            }
            d = d.recip();
            let change = d * c;
            value *= change;
            // Only the second half-step is a full continued-fraction iteration.
            if half == 1 && (change - 1.0).abs() <= 3e-14 {
                return value;
            }
        }
    }
    value
}

/// Returns log of the smaller tail and, in distant tails, its scale score and
/// curvature. With x=df/(df+z²), expand the incomplete beta integral as
/// H(x)=sum ((1/2)_k/k!) * (df/(df+2k)) * x^k. At x<=0.1 every term ratio
/// is below 0.1, bounding the omitted positive remainder by term/9.
fn student_tail(
    df: f64,
    log_df: f64,
    log_normalizer: f64,
    values: StudentCoordinates,
) -> (f64, Option<(f64, f64)>) {
    let x = values.remainder * values.remainder;
    let y = values.location * values.location;
    let a = df / 2.0;
    if x <= 0.1 {
        let mut term = 1.0;
        let mut remainder = 0.0;
        let mut slope = 0.0;
        for index in 1..=200 {
            let index = f64::from(index);
            term *= x * (index - 0.5) / index * (a + index - 1.0) / (a + index);
            remainder += term;
            slope += index * term;
            if index * term <= f64::EPSILON * (1.0 + slope) {
                break;
            }
        }
        let log_tail =
            log_normalizer - 0.5 * log_df - a * values.log_denominator + remainder.ln_1p();
        // Differentiating H preserves small curvature without subtracting the
        // scale score from df, including when df itself is very large.
        let score = df * values.location.abs() / (1.0 + remainder);
        let curvature = -score * (x + 2.0 * y * slope / (1.0 + remainder));
        return (log_tail, Some((score, curvature)));
    }
    let z = (values.location / values.inverse_root).abs();
    if z >= 10.0 {
        // Expand the incomplete-beta integral at its dominant endpoint:
        // H(x)=(1-x)^(-1/2)*J. Taylor's alternating remainder for
        // (1+w)^(-1/2), w>=0, is bounded by the next term; integration
        // preserves that bound. These terms reach f64 precision for z>=10.
        let mut term = 1.0;
        let mut remainder = 0.0;
        for index in 1..=100 {
            let index = f64::from(index);
            term *= -(df / (a + index)) * ((index - 0.5) / z) / z;
            remainder += term;
            if term.abs() <= f64::EPSILON * (1.0 + remainder) {
                break;
            }
        }
        let log_tail =
            log_normalizer - z.ln() - (a - 0.5) * values.log_denominator + remainder.ln_1p();
        let score = df * y / (1.0 + remainder);
        let curvature = -score * (x - score * remainder);
        return (log_tail, Some((score, curvature)));
    }
    if df >= 1e10 && (1.0 + z * z).powi(2) / df <= 1e-5 {
        // Integrating the Student density expansion through df^-2 gives
        // Q_t=Q_normal+phi*(z(z²+1)/(4df)
        // +(3z^7-7z^5-5z³-3z)/(96df²)). Under this bound and z<10,
        // the omitted relative correction is below f64 precision. A bound on
        // z is essential: distant Student tails retain their power-law decay.
        let log_tail = normal_log_upper_tail(z);
        let square = z * z;
        let inverse = df.recip();
        let correction = z * (square + 1.0) * inverse / 4.0
            + z * (square * (square * (3.0 * square - 7.0) - 5.0) - 3.0) * inverse / 96.0 * inverse;
        let hazard = (-0.5 * square - 0.5 * std::f64::consts::TAU.ln() - log_tail).exp();
        return (log_tail + (hazard * correction).ln_1p(), None);
    }
    if y == 0.0 {
        return (-std::f64::consts::LN_2, None);
    }
    let log_beta_normalizer = log_normalizer + 0.5 * log_df;
    let log_front = -a * values.log_denominator + 0.5 * y.ln() + log_beta_normalizer;
    if y > 3.0 / (df + 5.0) {
        (log_front + beta_fraction(a, 0.5, x).ln() - log_df, None)
    } else {
        let log_complement = log_front + beta_fraction(0.5, a, y).ln() + std::f64::consts::LN_2;
        (
            -std::f64::consts::LN_2 + (-log_complement.exp()).ln_1p(),
            None,
        )
    }
}

fn student_single(
    z: f64,
    scale: f64,
    status: i32,
    df: f64,
    root_df: f64,
    log_df: f64,
    log_normalizer: f64,
) -> [f64; 6] {
    let values = student_coordinates(z, root_df, log_df);
    let density = student_density(df, log_normalizer, values);
    if status == 1 {
        let fraction = values.location * values.location;
        let remainder = values.remainder * values.remainder;
        let inverse_root = values.inverse_root / scale;
        let location = values.location * inverse_root;
        return [
            density.log_density - scale.ln(),
            (df + 1.0) * location,
            (df + 1.0) * (fraction - remainder) * inverse_root * inverse_root,
            if fraction > 0.5 {
                df - (df + 1.0) * remainder
            } else {
                (df + 1.0) * fraction - 1.0
            },
            -2.0 * ((df + 1.0) * remainder * fraction),
            -2.0 * ((df + 1.0) * remainder * location),
        ];
    }
    let (log_small, scale_derivatives) = student_tail(df, log_df, log_normalizer, values);
    let log_large = (-log_small.exp()).ln_1p();
    let small_tail = (status == 0 && z >= 0.0) || (status == 2 && z <= 0.0);
    if let Some((score, curvature)) = scale_derivatives {
        let inverse_location = z.recip() / scale;
        let rare = [
            log_small,
            score * inverse_location,
            (score + curvature) * inverse_location * inverse_location,
            score,
            curvature,
            curvature * inverse_location,
        ];
        if small_tail {
            return rare;
        }
        let odds = (log_small - log_large).exp();
        let covariance = odds * (1.0 + odds);
        return [
            log_large,
            -odds * rare[1],
            -odds * rare[2] - covariance * rare[1] * rare[1],
            -odds * rare[3],
            -odds * rare[4] - covariance * rare[3] * rare[3],
            -odds * rare[5] - covariance * rare[1] * rare[3],
        ];
    }
    let g = if small_tail { log_small } else { log_large };
    let ratio = (density.log_density - g).exp();
    let score = if status == 0 { -ratio } else { ratio };
    let curvature = score * (density.score - score);
    [
        g,
        -score / scale,
        curvature / scale / scale,
        -z * score,
        z * (score + z * curvature),
        (score + z * curvature) / scale,
    ]
}

fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// The normal hazard and its small difference from z. Retaining the continued
/// fraction's correction avoids both subtracting large log probabilities and
/// subtracting almost equal hazard and z values in the tail curvature.
fn normal_tail_hazard(z: f64) -> (f64, f64) {
    let mut denominator = z;
    for numerator in (2..=32).rev() {
        denominator = z + f64::from(numerator) / denominator;
    }
    let correction = denominator.recip();
    (z + correction, correction)
}

/// Log of the upper normal tail for a nonnegative argument. The continued
/// fraction also covers tails smaller than f64 probabilities.
fn normal_log_upper_tail(z: f64) -> f64 {
    if z < 20.0 {
        return (0.5 * libm::erfc(z / std::f64::consts::SQRT_2)).ln();
    }
    let (hazard, _) = normal_tail_hazard(z);
    -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - hazard.ln()
}

impl AftDistribution {
    pub(crate) fn from_key(key: &str, parameter: Option<f64>) -> Self {
        match key {
            "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value"
            | "extremevalue" => Self::Extreme,
            "logistic" | "loglogistic" | "log_logistic" => Self::Logistic,
            "gaussian" | "normal" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
                Self::Gaussian
            }
            "t" | "student" | "student_t" | "studentt" => {
                let df = parameter.expect("Student-t df was validated");
                Self::StudentT {
                    df,
                    root_df: df.sqrt(),
                    log_df: df.ln(),
                    log_normalizer: student_log_normalizer(df),
                }
            }
            _ => unreachable!("distribution was validated"),
        }
    }

    fn density(self, z: f64) -> AftDensity {
        let (log_density, score, curvature) = match self {
            Self::Extreme => {
                let exponential = z.exp();
                (z - exponential, 1.0 - exponential, -exponential)
            }
            Self::Logistic => {
                let log_density = -z.abs() - 2.0 * (-z.abs()).exp().ln_1p();
                (log_density, -(z / 2.0).tanh(), -2.0 * log_density.exp())
            }
            Self::Gaussian => (-0.5 * z * z - 0.5 * std::f64::consts::TAU.ln(), -z, -1.0),
            Self::StudentT {
                df,
                root_df,
                log_df,
                log_normalizer,
            } => {
                return student_density(
                    df,
                    log_normalizer,
                    student_coordinates(z, root_df, log_df),
                );
            }
        };
        AftDensity {
            log_density,
            score,
            curvature,
        }
    }

    fn log_tails(self, z: f64) -> (f64, f64) {
        match self {
            Self::Extreme => {
                let exponential = z.exp();
                // exp(z) may underflow even while log(F(z)) is representable.
                let log_cdf = if exponential == 0.0 {
                    z
                } else {
                    (-(-exponential).exp_m1()).ln()
                };
                (log_cdf, -exponential)
            }
            Self::Logistic => (-softplus(-z), -softplus(z)),
            Self::Gaussian => {
                let log_small = normal_log_upper_tail(z.abs());
                let log_large = (-log_small.exp()).ln_1p();
                if z < 0.0 {
                    (log_small, log_large)
                } else {
                    (log_large, log_small)
                }
            }
            Self::StudentT {
                df,
                root_df,
                log_df,
                log_normalizer,
            } => {
                let (log_small, _) = student_tail(
                    df,
                    log_df,
                    log_normalizer,
                    student_coordinates(z, root_df, log_df),
                );
                let log_large = (-log_small.exp()).ln_1p();
                if z < 0.0 {
                    (log_small, log_large)
                } else {
                    (log_large, log_small)
                }
            }
        }
    }

    /// Log likelihood and derivatives with respect to location eta and log(scale):
    /// [loglik, eta, eta-eta, logscale, logscale-logscale, eta-logscale].
    pub(crate) fn single(self, z: f64, scale: f64, status: i32) -> [f64; 6] {
        if let Self::StudentT {
            df,
            root_df,
            log_df,
            log_normalizer,
        } = self
        {
            return student_single(z, scale, status, df, root_df, log_df, log_normalizer);
        }
        if matches!(self, Self::Gaussian) && status != 1 {
            let signed_z = if status == 0 { z } else { -z };
            let (g, score, curvature) = if signed_z >= 20.0 {
                let (hazard, correction) = normal_tail_hazard(signed_z);
                (
                    -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - hazard.ln(),
                    if status == 0 { -hazard } else { hazard },
                    -hazard * correction,
                )
            } else {
                let probability = 0.5 * libm::erfc(signed_z / std::f64::consts::SQRT_2);
                let density = (-0.5 * z * z).exp() / std::f64::consts::TAU.sqrt();
                if density == 0.0 {
                    return [probability.ln(), 0.0, 0.0, 0.0, 0.0, 0.0];
                }
                let ratio = density / probability;
                let score = if status == 0 { -ratio } else { ratio };
                (probability.ln(), score, score * (-z - score))
            };
            return [
                g,
                -score / scale,
                curvature / scale / scale,
                -z * score,
                z * (score + z * curvature),
                (score + z * curvature) / scale,
            ];
        }
        let density = self.density(z);
        let (g, score, curvature) = if status == 1 {
            (
                density.log_density - scale.ln(),
                density.score,
                density.curvature,
            )
        } else {
            let (log_cdf, log_survival) = self.log_tails(z);
            match self {
                Self::Extreme if status == 0 => (-z.exp(), -z.exp(), -z.exp()),
                Self::Extreme => {
                    let exponential = z.exp();
                    if exponential.is_infinite() {
                        return [0.0; 6];
                    }
                    let ratio = if exponential < 1e-4 {
                        1.0 - exponential / 2.0 + exponential * exponential / 12.0
                    } else {
                        (density.log_density - log_cdf).exp()
                    };
                    let curvature = if exponential < 1e-4 {
                        -exponential / 2.0 + exponential * exponential / 6.0
                    } else {
                        ratio * (1.0 - exponential - ratio)
                    };
                    (log_cdf, ratio, curvature)
                }
                Self::Logistic => {
                    let score = if status == 0 {
                        -log_cdf.exp()
                    } else {
                        log_survival.exp()
                    };
                    (
                        if status == 0 { log_survival } else { log_cdf },
                        score,
                        -density.log_density.exp(),
                    )
                }
                Self::Gaussian if (status == 0 && z >= 20.0) || (status == 2 && z <= -20.0) => {
                    let (hazard, correction) = normal_tail_hazard(z.abs());
                    let g = if status == 0 { log_survival } else { log_cdf };
                    let score = if status == 0 { -hazard } else { hazard };
                    (g, score, -hazard * correction)
                }
                _ => {
                    let g = if status == 0 { log_survival } else { log_cdf };
                    let ratio = (density.log_density - g).exp();
                    let score = if status == 0 { -ratio } else { ratio };
                    (g, score, score * (density.score - score))
                }
            }
        };
        [
            g,
            -score / scale,
            curvature / scale / scale,
            -z * score - f64::from(status == 1),
            z * (score + z * curvature),
            (score + z * curvature) / scale,
        ]
    }

    /// Accept a width on the transformed response scale before standardization.
    /// The density limit preserves log probability when width/scale underflows.
    pub(crate) fn interval_from_response_width(
        self,
        lower: f64,
        width: f64,
        scale: f64,
    ) -> [f64; 6] {
        let standardized_width = width / scale;
        if width > 0.0 && standardized_width == 0.0 {
            let mut row = self.single(lower, scale, 1);
            row[0] += width.ln();
            row
        } else {
            self.interval(lower, standardized_width, scale)
        }
    }

    /// True likelihood derivatives for an interval with standardized lower bound
    /// and width. Residual-specific conventions must be applied by the caller.
    pub(crate) fn interval(self, lower: f64, width: f64, scale: f64) -> [f64; 6] {
        let upper = lower + width;
        let lower_density = self.density(lower);
        let upper_density = self.density(upper);
        // Integrating the conditional scores avoids subtracting almost equal
        // endpoint densities. Check log-density variation as well as width:
        // the extreme-value density changes at exp(z), much faster than z.
        let density_variation = width
            * (lower_density.score.abs().max(upper_density.score.abs())
                + lower_density
                    .curvature
                    .abs()
                    .max(upper_density.curvature.abs())
                    .sqrt());
        if density_variation < 1e-3 {
            return self.narrow_interval(lower, width, scale);
        }
        // P = A - B, using upper tails on the right and lower tails on
        // the left. Combine their log-likelihood derivatives directly: forming
        // density/probability ratios loses curvature in distant Gaussian tails.
        let (larger, smaller) = if lower > 0.0 {
            (self.single(lower, scale, 0), self.single(upper, scale, 0))
        } else {
            (self.single(upper, scale, 2), self.single(lower, scale, 2))
        };
        let log_ratio = if matches!(self, Self::Gaussian) && (lower >= 20.0 || upper <= -20.0) {
            let (near, far) = if lower > 0.0 {
                (lower, upper)
            } else {
                (-upper, -lower)
            };
            let (near_hazard, near_correction) = normal_tail_hazard(near);
            let (_, far_correction) = normal_tail_hazard(far);
            let hazard_difference = width + far_correction - near_correction;
            -0.5 * width * (near + far) - (hazard_difference / near_hazard).ln_1p()
        } else {
            smaller[0] - larger[0]
        };
        let ratio = (-log_ratio).exp_m1().recip();
        if ratio == 0.0 {
            return larger;
        }
        let location_difference = larger[1] - smaller[1];
        let scale_difference = larger[3] - smaller[3];
        let covariance_weight = ratio * (1.0 + ratio);
        let g = larger[0] + (-log_ratio.exp_m1()).ln();
        let dg = larger[1] + ratio * location_difference;
        let ddg = larger[2] + ratio * (larger[2] - smaller[2])
            - covariance_weight * location_difference * location_difference;
        let ds = larger[3] + ratio * scale_difference;
        let dds = larger[4] + ratio * (larger[4] - smaller[4])
            - covariance_weight * scale_difference * scale_difference;
        let dsg = larger[5] + ratio * (larger[5] - smaller[5])
            - covariance_weight * location_difference * scale_difference;
        [g, dg, ddg, ds, dds, dsg]
    }

    fn narrow_interval(self, lower: f64, width: f64, scale: f64) -> [f64; 6] {
        const NODES: [f64; 4] = [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ];
        const WEIGHTS: [f64; 4] = [
            0.34785484513745385,
            0.6521451548625461,
            0.6521451548625461,
            0.34785484513745385,
        ];
        let half_width = width / 2.0;
        let center = lower + half_width;
        let rows = NODES.map(|node| self.single(center + node * half_width, scale, 1));
        let maximum = rows
            .iter()
            .map(|row| row[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: [f64; 4] = std::array::from_fn(|i| WEIGHTS[i] * (rows[i][0] - maximum).exp());
        let total = weights.iter().sum::<f64>();
        let dg = (0..4).map(|i| weights[i] * rows[i][1]).sum::<f64>() / total;
        let ds_true = (0..4).map(|i| weights[i] * rows[i][3]).sum::<f64>() / total;
        let mut ddg = 0.0;
        let mut dds_true = 0.0;
        let mut dsg_true = 0.0;
        for i in 0..4 {
            let location_delta = rows[i][1] - dg;
            let scale_delta = rows[i][3] - ds_true;
            ddg += weights[i] * (rows[i][2] + location_delta * location_delta);
            dds_true += weights[i] * (rows[i][4] + scale_delta * scale_delta);
            dsg_true += weights[i] * (rows[i][5] + location_delta * scale_delta);
        }
        let g = width.ln() - std::f64::consts::LN_2 + scale.ln() + maximum + total.ln();
        [
            g,
            dg,
            ddg / total,
            ds_true,
            dds_true / total,
            dsg_true / total,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
        if expected == 0.0 {
            assert_eq!(actual, expected);
        } else {
            assert!(
                ((actual - expected) / expected).abs() <= tolerance,
                "actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn likelihood_hessians_match_finite_differences_for_all_censoring_types() {
        for key in ["extreme", "logistic", "gaussian", "t"] {
            let family = AftDistribution::from_key(key, (key == "t").then_some(4.0));
            for status in [0, 1, 2, 3] {
                for width in [0.4, 1e-8] {
                    let eta = 0.2;
                    let rho = 0.4_f64;
                    let h = 1e-5;
                    let row = |location: f64, log_scale: f64| {
                        let scale = log_scale.exp();
                        let z = (1.3 - location) / scale;
                        if status == 3 {
                            family.interval(z, width / scale, scale)
                        } else {
                            family.single(z, scale, status)
                        }
                    };
                    let center = row(eta, rho);
                    let eta_plus = row(eta + h, rho);
                    let eta_minus = row(eta - h, rho);
                    let rho_plus = row(eta, rho + h);
                    let rho_minus = row(eta, rho - h);
                    for (actual, numeric) in [
                        (center[1], (eta_plus[0] - eta_minus[0]) / (2.0 * h)),
                        (center[2], (eta_plus[1] - eta_minus[1]) / (2.0 * h)),
                        (center[3], (rho_plus[0] - rho_minus[0]) / (2.0 * h)),
                        (center[4], (rho_plus[3] - rho_minus[3]) / (2.0 * h)),
                        (center[5], (rho_plus[1] - rho_minus[1]) / (2.0 * h)),
                        (center[5], (eta_plus[3] - eta_minus[3]) / (2.0 * h)),
                    ] {
                        assert!(
                            (actual - numeric).abs() < 2e-8,
                            "{key}, status={status}, width={width}: {actual} vs {numeric}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn student_log_tails_match_r_across_beta_evaluation_branches() {
        // R 4.6.1: pt(-c(0,.1,1,2,3,4,6,20,40), df, log.p=TRUE).
        let z_values = [0.0, 0.1, 1.0, 2.0, 3.0, 4.0, 6.0, 20.0, 40.0];
        for (df, expected) in [
            (
                3.0,
                [
                    -std::f64::consts::LN_2,
                    -0.7693239925792584,
                    -1.6321892244941227,
                    -2.6640861743156115,
                    -3.546184675450908,
                    -4.268395994597445,
                    -5.373826254202845,
                    -8.89844171394626,
                    -10.971162936870325,
                ],
            ),
            (
                4.0,
                [
                    -std::f64::consts::LN_2,
                    -0.7709402595289603,
                    -1.6769114931813456,
                    -2.8463082595453697,
                    -3.9134748570618902,
                    -4.820215993949168,
                    -6.244413641509423,
                    -10.90090412963439,
                    -13.66106721128188,
                ],
            ),
            (
                7.0,
                [
                    -std::f64::consts::LN_2,
                    -0.7731128266309448,
                    -1.7412089616007083,
                    -3.150991400914242,
                    -4.608068074213525,
                    -5.95418545820755,
                    -8.212915204882439,
                    -16.140912634356177,
                    -20.952541904706948,
                ],
            ),
        ] {
            let family = AftDistribution::from_key("t", Some(df));
            for (z, reference) in z_values.into_iter().zip(expected) {
                let actual = family.single(z, 1.0, 0);
                assert!((actual[0] - reference).abs() < 2e-12);
            }
        }
        let family = AftDistribution::from_key("t", Some(1000.0));
        for (z, expected) in [(60.0, -767.2789337230811), (80.0, -1005.0406311503176)] {
            assert!((family.single(z, 1.0, 0)[0] - expected).abs() < 2e-9);
        }
    }

    #[test]
    fn student_tail_hessians_retain_small_scale_curvature() {
        // Independent 480-digit series evaluation; log probabilities also agree
        // with R dt/pt(log=TRUE). The probability itself underflows for df=4.
        let family = AftDistribution::from_key("t", Some(4.0));
        for (status, expected) in [
            (
                1,
                [
                    -1_148.807_639_847_235,
                    5e-100,
                    5e-200,
                    4.0,
                    -4e-199,
                    -4e-299,
                ],
            ),
            (
                0,
                [
                    -919.9354249089502,
                    4e-100,
                    4e-200,
                    4.0,
                    -2.666_666_666_666_667e-199,
                    -2.6666666666666667e-299,
                ],
            ),
        ] {
            let row = family.single(1e100, 1.0, status);
            for (actual, reference) in row.into_iter().zip(expected) {
                assert_relative(actual, reference, 2e-12);
            }
        }
        let row = family.single(1e200, 1.7, 0);
        for (actual, expected) in
            row.into_iter()
                .zip([-1840.9694621065684, 4e-200 / 1.7, 0.0, 4.0, -0.0, -0.0])
        {
            assert_relative(actual, expected, 2e-12);
        }
        let mirrored = family.single(-1e100, 1.0, 2);
        let right = family.single(1e100, 1.0, 0);
        for column in 0..6 {
            let sign = if column == 1 || column == 5 {
                -1.0
            } else {
                1.0
            };
            assert_eq!(mirrored[column], sign * right[column]);
        }
    }

    #[test]
    fn common_student_tail_keeps_scale_derivatives_when_density_underflows() {
        let family = AftDistribution::from_key("t", Some(3.0));
        let row = family.single(-1e100, 1.0, 0);
        // For df=3, Q(z) ~ (2 sqrt(3)/pi) z^-3.
        let tail = 2.0 * 3.0_f64.sqrt() / std::f64::consts::PI * 1e-300;
        assert_relative(row[0], -tail, 2e-12);
        assert_relative(row[3], -3.0 * tail, 2e-12);
        assert_relative(row[4], -9.0 * tail, 2e-12);
        assert!(row.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn large_student_degrees_of_freedom_preserve_density_normalization() {
        // R dt(0, df, log=TRUE); subtraction of two lgamma values is unstable.
        for (df, expected) in [
            (256.0, -0.9199150932211914),
            (1e4, -0.9189635332046311),
            (1e16, -0.9189385332046728),
            (1e300, -0.9189385332046728),
        ] {
            let family = AftDistribution::from_key("t", Some(df));
            assert!((family.single(0.0, 1.0, 1)[0] - expected).abs() < 5e-16);
        }
        // In this far Student tail, x=df/(df+z²) ~ 1e-100. The next relative
        // terms are smaller than f64 precision, despite df being very large.
        let row = AftDistribution::from_key("t", Some(1e300)).single(1e200, 1.0, 0);
        for (actual, expected) in row[1..]
            .iter()
            .copied()
            .zip([1e100, 1e-100, 1e300, -2e200, -2.0])
        {
            assert_relative(actual, expected, 2e-14);
        }
        assert!(row[0].is_finite());
        let family = AftDistribution::from_key("t", Some(1e308));
        assert!(
            family
                .single(0.0, 1.0, 1)
                .iter()
                .all(|value| value.is_finite())
        );
        let row = family.single(1.0, 1.0, 1);
        assert_relative(row[4], -2.0, 2e-14);
        assert_relative(row[5], -2.0, 2e-14);
    }

    #[test]
    fn saturated_extreme_left_tail_has_zero_likelihood_derivatives() {
        let family = AftDistribution::from_key("extreme", None);
        for z in [710.0, 1000.0] {
            assert_eq!(family.single(z, 1.7, 2), [0.0; 6]);
        }
    }

    #[test]
    fn very_large_student_df_keeps_central_and_endpoint_tail_branches_consistent() {
        // R pt(-z, df, log.p=TRUE). Values straddle the endpoint expansion's
        // z=10 boundary and include x=df/(df+z²) rounding to one.
        let z_values = [0.1, 1.0, 2.0, 9.999999, 10.0, 10.000001, 40.0];
        for (df, expected) in [
            (
                1e12,
                [
                    -0.7761545927302516,
                    -1.8410216450085,
                    -3.7831843336760986,
                    -53.23127504986997,
                    -53.23128514796271,
                    -53.231295246056426,
                    -804.608441372954,
                ],
            ),
            (
                1e300,
                [
                    -0.776154592730276,
                    -1.8410216450092634,
                    -3.783184333682032,
                    -53.23127505241974,
                    -53.23128515051247,
                    -53.231295248606195,
                    -804.6084420137537,
                ],
            ),
        ] {
            let family = AftDistribution::from_key("t", Some(df));
            for (z, expected) in z_values.into_iter().zip(expected) {
                let row = family.single(z, 1.0, 0);
                assert!(row.iter().all(|value| value.is_finite()));
                assert!((row[0] - expected).abs() < 2e-12);
            }
        }
    }

    #[test]
    fn smallest_representable_interval_retains_finite_log_likelihood() {
        let width = f64::from_bits(1);
        for family in [AftDistribution::Gaussian, AftDistribution::Logistic] {
            let row = family.interval(0.0, width, 1.0);
            let expected = family.single(0.0, 1.0, 1);
            assert!((row[0] - (expected[0] + width.ln())).abs() < 2e-13);
            assert_eq!(&row[1..], &expected[1..]);
        }
    }

    #[test]
    fn interval_width_underflow_retains_the_response_density_limit() {
        let family = AftDistribution::Gaussian;
        let row = family.interval_from_response_width(0.0, 1e-300, 1e100);
        // log(width/scale)-log(sqrt(2*pi)), evaluated without width/scale.
        assert!((row[0] - (-921.9529757308229)).abs() < 2e-13);
        assert_eq!(row[3], -1.0);
        assert_relative(row[2], -1e-200, 1e-14);
        assert!(row.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn transformed_interval_width_retains_adjacent_and_wide_bounds() {
        let lower = 1e100_f64;
        let upper = f64::from_bits(lower.to_bits() + 1);
        let expected = ((upper - lower) / lower).ln_1p();
        assert_eq!(transformed_interval_width(lower, upper, true), expected);
        assert!(expected > 0.0);
        assert_eq!(
            transformed_interval_width(1e-300, 1e300, true),
            1e300_f64.ln() - 1e-300_f64.ln()
        );
        assert_eq!(transformed_interval_width(-2.0, -0.5, false), 1.5);
    }
}
