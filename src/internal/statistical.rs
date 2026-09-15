use crate::constants::{DEFAULT_CONCORDANCE, LCG64_INCREMENT, LCG64_MULTIPLIER, TIME_EPSILON};
use crate::internal::dist::{lgammafn, pchisq, pgamma, pnorm, pt, qgamma, qnorm, qt};

#[inline]
pub(crate) fn sample_normal(rng: &mut crate::internal::rng::Rng) -> f64 {
    let u1: f64 = rng.f64().max(1e-10);
    let u2: f64 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Standard normal quantile (R's `qnorm(p)`); `p` outside `(0, 1)` maps to
/// the corresponding infinity so callers may pass slightly out-of-range
/// probabilities without triggering NaN.
#[inline]
pub(crate) fn probit(p: f64) -> f64 {
    normal_inverse_cdf(p)
}

/// Error function, `erf(x) = 2 pnorm(x sqrt 2) - 1`.
#[inline]
pub(crate) fn erf(x: f64) -> f64 {
    crate::internal::dist::erf(x)
}

/// Complementary error function, `erfc(x) = 2 pnorm(x sqrt 2, lower = FALSE)`.
#[inline]
pub(crate) fn erfc(x: f64) -> f64 {
    crate::internal::dist::erfc(x)
}

/// Standard normal distribution function (R's `pnorm(x)`).
#[inline]
pub(crate) fn normal_cdf(x: f64) -> f64 {
    pnorm(x, true, false)
}

/// Standard normal survival function (R's `pnorm(x, lower.tail = FALSE)`).
#[inline]
pub(crate) fn normal_sf(x: f64) -> f64 {
    pnorm(x, false, false)
}

/// Harrell's C of a risk score against right-censored outcomes, as R's
/// `concordance(Surv(time, event) ~ risk, reverse = TRUE, ymax = horizon)`:
/// `(concordant + tied / 2) / comparable` over the pairs whose earlier
/// time is an event, events after `horizon` (when given) counting no
/// pairs.  Rows with a non-finite time or score are dropped;
/// [`DEFAULT_CONCORDANCE`] is returned when no pair is comparable.
pub(crate) fn concordance_index_with_horizon(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    horizon: Option<f64>,
) -> f64 {
    use crate::concordance::{ConcordanceOptions, concordancefit};
    use crate::core::SurvResponse;
    use crate::internal::typed_inputs::SurvivalData;
    use ndarray::Array2;

    let n = risk_scores.len();
    if n < 2 || time.len() != n || event.len() != n {
        return DEFAULT_CONCORDANCE;
    }
    let keep: Vec<usize> = (0..n)
        .filter(|&i| risk_scores[i].is_finite() && time[i].is_finite())
        .collect();
    if keep.len() < 2 {
        return DEFAULT_CONCORDANCE;
    }
    let Ok(survival) = SurvivalData::try_new(
        keep.iter().map(|&i| time[i]).collect(),
        keep.iter().map(|&i| i32::from(event[i] == 1)).collect(),
    ) else {
        return DEFAULT_CONCORDANCE;
    };
    let x = Array2::from_shape_fn((keep.len(), 1), |(row, _)| risk_scores[keep[row]]);
    let options = ConcordanceOptions {
        reverse: true,
        ymax: horizon,
        std_err: false,
        ..ConcordanceOptions::default()
    };
    match concordancefit(
        SurvResponse::Right(&survival),
        x.view(),
        None,
        None,
        None,
        &options,
    ) {
        Ok(fit) if fit.concordance[0].is_finite() => fit.concordance[0],
        _ => DEFAULT_CONCORDANCE,
    }
}

#[inline]
pub(crate) fn lcg64_next(state: &mut u64) {
    *state = state
        .wrapping_mul(LCG64_MULTIPLIER)
        .wrapping_add(LCG64_INCREMENT);
}

#[inline]
#[cfg(feature = "ml")]
pub(crate) fn lcg64_shuffle_with_state(indices: &mut [usize], state: &mut u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        lcg64_next(state);
        let j = (*state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[inline]
pub(crate) fn lcg64_shuffle_per_index_seed(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let mut state = seed.wrapping_add(i as u64);
        lcg64_next(&mut state);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[inline]
pub(crate) fn compute_censoring_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;
        let mut total_at_time = 0;

        while i < n && (time[indices[i]] - current_time).abs() < TIME_EPSILON {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);
        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

#[inline]
pub(crate) fn km_step_prob_at(t: f64, unique_times: &[f64], km_values: &[f64]) -> f64 {
    if unique_times.is_empty() {
        return 1.0;
    }
    if t < unique_times[0] {
        return 1.0;
    }

    let mut left = 0;
    let mut right = unique_times.len();
    while left < right {
        let mid = (left + right) / 2;
        if unique_times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 { 1.0 } else { km_values[left - 1] }
}

/// Standard normal quantile (R's `qnorm(p)`); `p <= 0` gives `-Inf` and
/// `p >= 1` gives `+Inf` instead of R's NaN for out-of-range input.
#[inline]
pub(crate) fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    qnorm(p, true, false)
}

/// `qnorm(1 - alpha/2)`, the two-sided normal critical value, or `None`
/// when `alpha` is not in `(0, 1)`.
#[inline]
pub(crate) fn two_sided_normal_quantile(alpha: f64) -> Option<f64> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }

    // Halving in log space keeps subnormal alpha values representable: R's
    // qnorm(log(alpha) - log(2), lower.tail = FALSE, log.p = TRUE).
    let z = qnorm(alpha.ln() - std::f64::consts::LN_2, false, true);
    z.is_finite().then_some(z)
}

/// Gamma quantile with unit scale (R's `qgamma(p, a)`); `p <= 0` gives 0
/// and `p >= 1` gives `+Inf`.
#[inline]
pub(crate) fn gamma_inverse_cdf(p: f64, a: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    qgamma(p, a, 1.0, true, false)
}

/// Chi-squared survival function (R's `pchisq(x, df, lower.tail = FALSE)`),
/// evaluated directly in the upper tail; 1 for `x <= 0` or `df == 0`.
#[inline]
pub(crate) fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    pchisq(x, df as f64, false, false)
}

/// Chi-squared distribution function (R's `pchisq(x, df)`); zero for
/// `x <= 0` or a non-positive `df`.
#[inline]
pub(crate) fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    pchisq(x, df, true, false)
}

/// `log|gamma(x)|` (R's `lgamma(x)`).
#[inline]
pub(crate) fn ln_gamma(x: f64) -> f64 {
    lgammafn(x)
}

/// Student t density (R's `dt(x, df)`).
#[inline]
pub(crate) fn student_t_pdf(value: f64, df: f64) -> f64 {
    crate::internal::dist::dt(value, df, false)
}

/// Student t distribution function (R's `pt(x, df)`).
#[inline]
pub(crate) fn student_t_cdf(value: f64, df: f64) -> f64 {
    pt(value, df, true, false)
}

/// Student t quantile (R's `qt(p, df)`); NaN outside `[0, 1]`.
#[inline]
pub(crate) fn student_t_inverse_cdf(probability: f64, df: f64) -> f64 {
    qt(probability, df, true, false)
}

/// Regularized lower incomplete gamma function `P(a, x)` (R's
/// `pgamma(x, a)`); zero for `x < 0` or a non-positive shape.
#[inline]
pub(crate) fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    pgamma(x, a, 1.0, true, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concordance_index_with_horizon_follows_r_reverse_concordance() {
        // R: concordance(Surv(time, status) ~ x, d, reverse = TRUE) = 0.5,
        // with counts 9 concordant, 9 discordant, 3 tied on x.
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
        let status = [1, 1, 0, 1, 1, 1, 0, 1];
        let x = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9];
        assert!((concordance_index_with_horizon(&x, &time, &status, None) - 0.5).abs() < 1e-14);
        // ymax = 3: 7 concordant, 7 discordant, 3 tied on x (R).
        let capped = concordance_index_with_horizon(&x, &time, &status, Some(3.0));
        assert!((capped - (7.0 + 1.5) / 17.0).abs() < 1e-14);
        // A perfect risk score on untied times.
        let time = [1.0, 2.0, 3.0, 4.0, 5.0];
        let status = [1, 0, 1, 1, 0];
        let perfect: Vec<f64> = time.iter().map(|t| -t).collect();
        let c = concordance_index_with_horizon(&perfect, &time, &status, None);
        assert!((c - 1.0).abs() < 1e-14);
        let worst: Vec<f64> = time.to_vec();
        assert!(concordance_index_with_horizon(&worst, &time, &status, None).abs() < 1e-14);
    }

    #[test]
    fn concordance_index_with_horizon_degenerate_inputs_give_the_default() {
        assert_eq!(
            concordance_index_with_horizon(&[1.0], &[1.0], &[1], None),
            DEFAULT_CONCORDANCE
        );
        assert_eq!(
            concordance_index_with_horizon(&[1.0, 2.0], &[1.0, 2.0], &[0, 0], None),
            DEFAULT_CONCORDANCE
        );
        // The NaN row is dropped: the remaining pair is discordant.
        assert_eq!(
            concordance_index_with_horizon(
                &[1.0, f64::NAN, 2.0],
                &[1.0, 2.0, 3.0],
                &[1, 1, 1],
                None
            ),
            0.0
        );
        assert_eq!(
            concordance_index_with_horizon(&[1.0, 2.0], &[1.0, 2.0], &[1, 1, 1], None),
            DEFAULT_CONCORDANCE
        );
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_chi2_sf_basic() {
        assert!((chi2_sf(0.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(-1.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(1.0, 0) - 1.0).abs() < 1e-10);
        // R: pchisq(3.84, 1, lower.tail = FALSE)
        assert!((chi2_sf(3.84, 1) - 0.050043521248705224).abs() < 1e-16);
        assert!((chi2_cdf(3.84, 1.0) - 0.94995647875129474).abs() < 1e-15);
        assert_eq!(chi2_cdf(0.0, 1.0), 0.0);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_ln_gamma() {
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
        // R: lgamma(0.5) = log(sqrt(pi))
        assert!((ln_gamma(0.5) - 0.57236494292470008).abs() < 1e-15);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn normal_helpers_match_r() {
        // R: pnorm(1.96), qnorm(0.975), pnorm(-10)
        assert!((normal_cdf(1.96) - 0.97500210485177963).abs() < 1e-15);
        assert!((normal_inverse_cdf(0.975) - 1.9599639845400536).abs() < 1e-15);
        assert!((normal_cdf(-10.0) / 7.6198530241605269e-24 - 1.0).abs() < 1e-14);
        assert_eq!(normal_inverse_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_inverse_cdf(1.0), f64::INFINITY);
        assert_eq!(normal_inverse_cdf(-0.1), f64::NEG_INFINITY);
        assert!(normal_inverse_cdf(f64::NAN).is_nan());
        assert!((probit(0.025) + 1.9599639845400536).abs() < 1e-15);
        assert!((two_sided_normal_quantile(0.05).unwrap() - 1.9599639845400536).abs() < 1e-15);
        assert_eq!(two_sided_normal_quantile(0.0), None);
        assert_eq!(two_sided_normal_quantile(1.0), None);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn erf_helpers_match_reference_values() {
        // R: 2 * pnorm(x * sqrt(2)) - 1 and 2 * pnorm(x * sqrt(2), lower = FALSE),
        // erf(3) against its true value 0.99997790950300141456...; erf(1e-8) is
        // compared with 2/sqrt(pi) * 1e-8, which R's own expression cannot
        // resolve.
        assert!((erf(0.5) - 0.52049987781304652).abs() < 1e-16);
        assert!((erf(-0.5) + 0.52049987781304652).abs() < 1e-16);
        assert!((erf(3.0) - 0.99997790950300141).abs() < 1.2e-16);
        assert!((erfc(3.0) / 2.2090496998585394e-05 - 1.0).abs() < 1e-15);
        assert!((erf(1e-8) / 1.1283791670955126e-08 - 1.0).abs() < 1e-15);
        assert!((erfc(-3.0) - (2.0 - 2.2090496998585394e-05)).abs() < 1e-15);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erfc(0.0), 1.0);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn student_t_helpers_match_reference_values_and_boundaries() {
        assert!((student_t_pdf(1.0, 5.0) - 0.21967979735098059).abs() < 1e-16);
        assert!((student_t_cdf(1.0, 5.0) - 0.81839126617543867).abs() < 1e-15);
        assert_eq!(student_t_pdf(f64::INFINITY, 5.0), 0.0);
        assert!(student_t_pdf(f64::NAN, 5.0).is_nan());
        assert_eq!(student_t_cdf(f64::NEG_INFINITY, 5.0), 0.0);
        assert_eq!(student_t_cdf(f64::INFINITY, 5.0), 1.0);
        assert_eq!(student_t_inverse_cdf(0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(student_t_inverse_cdf(1.0, 5.0), f64::INFINITY);
        assert!(student_t_inverse_cdf(1.5, 5.0).is_nan());
        assert!(student_t_inverse_cdf(f64::NAN, 5.0).is_nan());

        for probability in [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
            let quantile = student_t_inverse_cdf(probability, 5.0);
            assert!((student_t_cdf(quantile, 5.0) - probability).abs() < 1e-15);
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_gamma_helpers() {
        // R: qgamma(0.475, 5), qgamma(0.525, 6), pgamma(4.5, 5)
        assert!((gamma_inverse_cdf(0.475, 5.0) - 4.5375048990088311).abs() < 1e-14);
        assert!((gamma_inverse_cdf(0.525, 6.0) - 5.8200445519969533).abs() < 1e-14);
        assert!((lower_incomplete_gamma(5.0, 4.5) - 0.46789642362528439).abs() < 1e-15);
        assert_eq!(gamma_inverse_cdf(0.0, 5.0), 0.0);
        assert_eq!(gamma_inverse_cdf(1.0, 5.0), f64::INFINITY);
        assert_eq!(lower_incomplete_gamma(5.0, 0.0), 0.0);
        assert_eq!(lower_incomplete_gamma(0.0, 1.0), 0.0);
        assert_eq!(lower_incomplete_gamma(5.0, -1.0), 0.0);
    }
}
