//! A positive quantity with a separately retained logarithmic origin.
//!
//! A compensated predictor origin stays separate from the relative risk mass.
//! This preserves multiplicities even when the predictor's rounding correction
//! is itself large. Variance sums use degree two on both origin components.

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct LogScale {
    pub origin: f64,
    pub origin_correction: f64,
    pub log_mass: f64,
}

impl LogScale {
    pub const EMPTY: Self = Self::new(0.0, f64::NEG_INFINITY);

    pub const fn new(origin: f64, log_mass: f64) -> Self {
        Self::new_parts(origin, 0.0, log_mass)
    }

    pub const fn new_parts(origin: f64, origin_correction: f64, log_mass: f64) -> Self {
        Self {
            origin,
            origin_correction,
            log_mass,
        }
    }

    pub const fn from_predictor_parts(origin: f64, origin_correction: f64) -> Self {
        Self::new_parts(origin, origin_correction, 0.0)
    }

    pub fn from_log(value: f64) -> Self {
        if value.is_finite() {
            Self::new(value, 0.0)
        } else {
            Self::new(0.0, value)
        }
    }

    pub fn is_zero(self) -> bool {
        self.log_mass == f64::NEG_INFINITY
    }

    /// Collapse only at a public boundary, never before taking a ratio.
    pub fn log_value(self) -> f64 {
        if self.log_mass.is_infinite() {
            self.log_mass
        } else {
            self.origin + (self.origin_correction + self.log_mass)
        }
    }

    pub fn shifted(self, log_factor: f64) -> Self {
        if self.is_zero() {
            self
        } else {
            Self::new_parts(
                self.origin,
                self.origin_correction,
                self.log_mass + log_factor,
            )
        }
    }

    pub fn ratio_log(self, denominator: Self) -> f64 {
        ((self.origin - denominator.origin)
            + (self.origin_correction - denominator.origin_correction))
            + (self.log_mass - denominator.log_mass)
    }

    pub fn reciprocal(self, log_numerator: f64) -> Self {
        Self::new_parts(
            -self.origin,
            -self.origin_correction,
            log_numerator - self.log_mass,
        )
    }

    pub fn exp_at(self, predictor: f64) -> f64 {
        self.exp_at_degree(predictor, 1.0)
    }

    pub fn exp_at_degree(self, predictor: f64, degree: f64) -> f64 {
        self.exp_at_scale_degree(Self::from_log(predictor), degree)
    }

    pub fn exp_at_scale(self, predictor: Self) -> f64 {
        self.exp_at_scale_degree(predictor, 1.0)
    }

    pub fn exp_at_scale_degree(self, predictor: Self, degree: f64) -> f64 {
        self.log_at_scale_degree(predictor, degree).exp()
    }

    pub fn log_at_scale_degree(self, predictor: Self, degree: f64) -> f64 {
        // Keep R's NA payload, and preserve exact zero/terminal hazard limits.
        if predictor.origin.is_nan() {
            return predictor.origin;
        }
        if predictor.origin_correction.is_nan() {
            return predictor.origin_correction;
        }
        if predictor.log_mass.is_nan() {
            return predictor.log_mass;
        }
        if self.log_mass.is_infinite() {
            return self.log_mass;
        }
        degree
            * ((self.origin + predictor.origin)
                + (self.origin_correction + predictor.origin_correction))
            + (self.log_mass + degree * predictor.log_mass)
    }

    pub fn mix(self, other: Self) -> (Self, f64, f64) {
        self.mix_degree(other, 1.0)
    }

    /// Add two positive quantities and return their fractions of the sum.
    pub fn mix_degree(self, other: Self, degree: f64) -> (Self, f64, f64) {
        if self.is_zero() {
            return (other, 0.0, 1.0);
        }
        if other.is_zero() {
            return (self, 1.0, 0.0);
        }
        // Multiple terminal K-P increments still sum to an infinite hazard.
        // Their relative finite masses are immaterial at this exact boundary.
        if self.log_mass == f64::INFINITY || other.log_mass == f64::INFINITY {
            return match (
                self.log_mass == f64::INFINITY,
                other.log_mass == f64::INFINITY,
            ) {
                (true, true) => (Self::from_log(f64::INFINITY), 0.5, 0.5),
                (true, false) => (self, 1.0, 0.0),
                (false, true) => (other, 0.0, 1.0),
                (false, false) => unreachable!(),
            };
        }
        let difference = degree
            * ((self.origin - other.origin) + (self.origin_correction - other.origin_correction))
            + (self.log_mass - other.log_mass);
        let ratio = (-difference.abs()).exp();
        let smaller = ratio / (1.0 + ratio);
        let larger = 1.0 / (1.0 + ratio);
        if difference >= 0.0 {
            (self.shifted(ratio.ln_1p()), larger, smaller)
        } else {
            (other.shifted(ratio.ln_1p()), smaller, larger)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LogScale;

    #[test]
    fn multiplicities_and_weights_survive_huge_origins() {
        for origin in [-1e308, -1e16, 1e16, 1e308] {
            let one = LogScale::from_log(origin);
            let (two, a, b) = one.mix(one);
            assert_eq!((a, b), (0.5, 0.5));
            assert_eq!(two.log_mass, 2.0_f64.ln());
            let (three, a, b) = two.mix(one);
            assert!((a - 2.0 / 3.0).abs() < 1e-15);
            assert!((b - 1.0 / 3.0).abs() < 1e-15);
            assert!((three.reciprocal(0.0).exp_at(origin) - 1.0 / 3.0).abs() < 1e-15);
            let (_, a, b) = one.mix(one.shifted(3.0_f64.ln()));
            assert!((a - 0.25).abs() < 1e-15);
            assert!((b - 0.75).abs() < 1e-15);
        }
    }

    #[test]
    fn variance_origins_cancel_before_doubling() {
        for origin in [-1e308, 1e308] {
            let variance = LogScale::new(-origin, 0.25_f64.ln());
            let (sum, a, b) = variance.mix_degree(variance, 2.0);
            assert_eq!((a, b), (0.5, 0.5));
            assert_eq!(sum.exp_at_degree(origin, 2.0), 0.5);
            let prediction = LogScale::new(origin, 0.4);
            assert!((sum.exp_at_scale_degree(prediction, 2.0) - 0.5 * 0.8_f64.exp()).abs() < 1e-15);
        }
    }

    #[test]
    fn compensated_origins_leave_a_separate_slot_for_risk_multiplicity() {
        for sign in [-1.0, 1.0] {
            let risk = LogScale::from_predictor_parts(sign * 1e308, sign * 1e292);
            let (two, a, b) = risk.mix(risk);
            assert_eq!((a, b), (0.5, 0.5));
            assert_eq!(two.log_mass, 2.0_f64.ln());
            assert_eq!(two.origin_correction, risk.origin_correction);
            assert!((risk.ratio_log(two).exp() - 0.5).abs() < 1e-15);
            assert!((two.reciprocal(0.0).exp_at_scale(risk) - 0.5).abs() < 1e-15);
            let variance =
                LogScale::new_parts(-risk.origin, -risk.origin_correction, 0.25_f64.ln());
            assert_eq!(variance.exp_at_scale_degree(risk, 2.0), 0.25);
            assert_eq!(
                variance
                    .mix_degree(variance, 2.0)
                    .0
                    .exp_at_scale_degree(risk, 2.0),
                0.5
            );
        }
    }

    #[test]
    fn empty_terms_and_distant_origins_have_exact_limits() {
        let one = LogScale::new(1e308, 0.0);
        let tiny = LogScale::new(-1e308, 0.0);
        assert_eq!(one.mix(tiny), (one, 1.0, 0.0));
        assert_eq!(tiny.mix(one), (one, 0.0, 1.0));
        assert_eq!(LogScale::EMPTY.mix(one), (one, 0.0, 1.0));
        assert_eq!(LogScale::EMPTY.exp_at(f64::INFINITY), 0.0);
        assert_eq!(
            LogScale::from_log(f64::INFINITY).exp_at(f64::NEG_INFINITY),
            f64::INFINITY
        );
        let terminal = LogScale::from_log(f64::INFINITY);
        assert_eq!(terminal.mix(terminal), (terminal, 0.5, 0.5));
        assert_eq!(terminal.mix(one), (terminal, 1.0, 0.0));
        assert_eq!(one.mix(terminal), (terminal, 0.0, 1.0));
    }
}
