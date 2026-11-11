use ndarray::Array2;
use survival::regression::{CoxPHFit, CoxphData, CoxphFitResult, CoxphOptions, TieMethod};

fn close(actual: f64, expected: f64) {
    let tolerance = 2e-10 * expected.abs().max(1e-300);
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{actual:e} != {expected:e} (tolerance {tolerance:e})"
    );
}

fn fixture() -> CoxphData {
    CoxphData::try_new(
        [1., 1., 2., 3., 4., 4.].repeat(2),
        None,
        [1, 1, 1, 0, 1, 0].repeat(2),
        Array2::from_shape_vec(
            (12, 1),
            vec![-1., 0.5, 1., -0.5, 0.2, 1.4, 0.2, -0.8, 0.4, 1.2, -0.4, 0.7],
        )
        .unwrap(),
        Some(vec![
            0.5, 1., 2., 1.5, 0.75, 1.25, 1.5, 0.75, 1.25, 2., 1., 0.5,
        ]),
        Some(vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),
        Some(vec![
            0.2, -0.1, 0.3, 0., -0.2, 0.1, -0.3, 0.2, 0.1, -0.1, 0.4, 0.,
        ]),
    )
    .unwrap()
}

fn raw_fit(data: CoxphData, method: TieMethod, beta: f64, iterations: usize) -> CoxphFitResult {
    CoxphFitResult::fit(
        data,
        CoxphOptions {
            method,
            init: Some(vec![beta]),
            iter_max: iterations,
            ..CoxphOptions::default()
        },
        false,
    )
    .unwrap()
}

// Independent O(n²) one-covariate oracle: rebuild each event risk set and
// directly calculate its normalized probabilities, score, and covariance.
fn oracle(data: &CoxphData, beta: f64, method: TieMethod) -> (f64, f64, f64) {
    let n = data.n();
    let weights = data.weights.as_ref().unwrap();
    let offset = data.offset.as_ref().unwrap();
    let strata = data.strata.as_ref().unwrap();
    let eta: Vec<_> = (0..n).map(|i| beta * data.x[(i, 0)] + offset[i]).collect();
    let mut events: Vec<_> = (0..n).filter(|&i| data.status[i] == 1).collect();
    events.sort_by(|&i, &j| {
        strata[i]
            .cmp(&strata[j])
            .then(data.time[i].total_cmp(&data.time[j]))
    });
    events.dedup_by(|i, j| strata[*i] == strata[*j] && data.time[*i] == data.time[*j]);
    let (mut loglik, mut score, mut info) = (0.0, 0.0, 0.0);
    for event in events {
        let deaths: Vec<_> = (0..n)
            .filter(|&i| {
                strata[i] == strata[event]
                    && data.time[i] == data.time[event]
                    && data.status[i] == 1
            })
            .collect();
        let at_risk: Vec<_> = (0..n)
            .filter(|&i| strata[i] == strata[event] && data.time[i] >= data.time[event])
            .collect();
        let max_log = at_risk
            .iter()
            .map(|&i| eta[i] + weights[i].ln())
            .fold(f64::NEG_INFINITY, f64::max);
        let steps = if method == TieMethod::Efron {
            deaths.len()
        } else {
            1
        };
        for step in 0..steps {
            let mut risks: Vec<_> = at_risk
                .iter()
                .map(|&i| {
                    let fraction = if deaths.contains(&i) {
                        1.0 - step as f64 / deaths.len() as f64
                    } else {
                        1.0
                    };
                    (eta[i] + weights[i].ln() - max_log).exp() * fraction
                })
                .collect();
            let total: f64 = risks.iter().sum();
            for risk in &mut risks {
                *risk /= total;
            }
            let mean: f64 = at_risk
                .iter()
                .zip(&risks)
                .map(|(&i, &p)| p * data.x[(i, 0)])
                .sum();
            let covariance: f64 = at_risk
                .iter()
                .zip(&risks)
                .map(|(&i, &p)| p * (data.x[(i, 0)] - mean).powi(2))
                .sum();
            for &i in &deaths {
                let weight = weights[i] / steps as f64;
                loglik += weight * (eta[i] - max_log - total.ln());
                score += weight * (data.x[(i, 0)] - mean);
                info += weight * covariance;
            }
        }
    }
    (loglik, score, info)
}

fn check_oracle(data: CoxphData, method: TieMethod, beta: f64) {
    let (loglik, score, info) = oracle(&data, beta, method);
    let fit = raw_fit(data, method, beta, 0);
    close(fit.loglik[0], loglik);
    close(fit.loglik[1], loglik);
    close(fit.first[0], score);
    close(fit.var[(0, 0)], 1.0 / info);
    close(fit.score, score * (score / info));
}

#[test]
fn opposite_stratum_offsets_have_the_null_control_likelihood() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let data = CoxphData::try_new(
            vec![1., 2., 1., 2.],
            None,
            vec![1, 0, 1, 0],
            Array2::zeros((4, 0)),
            None,
            Some(vec![1, 1, 2, 2]),
            Some(vec![1000., 1000., -1000., -1000.]),
        )
        .unwrap();
        let fit = CoxPHFit::fit(
            data,
            CoxphOptions {
                method,
                ..CoxphOptions::default()
            },
        )
        .unwrap();
        close(fit.loglik[0], -2.0 * 2.0_f64.ln());
        close(fit.loglik[1], -2.0 * 2.0_f64.ln());
    }
}

#[test]
fn shifted_weighted_tied_covariate_fits_match_stock_r_controls() {
    // R 4.6.2, survival 3.8-11: coxph(Surv(t,s) ~ x + strata(g) +
    // offset(o), data, weights=w, ties=method, robust=FALSE), using fixture().
    for (method, beta, variance, loglik, score) in [
        (
            TieMethod::Breslow,
            -0.471810935418853,
            0.239341457912726,
            [-13.7453143277682, -13.2811408528274],
            0.955499138372156,
        ),
        (
            TieMethod::Efron,
            -0.511122859100819,
            0.241657642959352,
            [-13.4972917005859, -12.9567188163440],
            1.11614783277186,
        ),
    ] {
        for shifts in [[0., 0.], [1000., 1000.], [-1000., -1000.], [1000., -1000.]] {
            let mut data = fixture();
            for (i, offset) in data.offset.as_mut().unwrap().iter_mut().enumerate() {
                *offset += shifts[i / 6];
            }
            let fit = raw_fit(data, method, 0., 40);
            close(fit.coefficients[0], beta);
            close(fit.var[(0, 0)], variance);
            close(fit.loglik[0], loglik[0]);
            close(fit.loglik[1], loglik[1]);
            close(fit.score, score);
        }
    }
}

#[test]
fn late_tiny_risk_sets_and_dominant_tied_deaths_match_direct_moments() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for reversed in [false, true] {
            let mut data = fixture();
            data.strata = Some(vec![0; 12]);
            data.time = vec![1., 1., 2., 3., 4., 4., 5., 5., 6., 7., 8., 8.];
            data.offset = Some(
                (0..12)
                    .map(|i| if (i < 2) != reversed { 1000. } else { -1000. })
                    .collect(),
            );
            check_oracle(data, method, 0.4);
        }
    }
}

#[test]
fn extreme_case_weights_and_risks_keep_their_score_and_information() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for weight_scale in [1e-250, 1e250] {
            for offset_shift in [-700., 700.] {
                let mut data = fixture();
                for weight in data.weights.as_mut().unwrap() {
                    *weight *= weight_scale;
                }
                for offset in data.offset.as_mut().unwrap() {
                    *offset += offset_shift;
                }
                check_oracle(data, method, -0.2);
            }
        }
    }
}

#[test]
fn efron_all_deaths_at_a_time_retains_nonzero_remaining_risk() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let mut data = fixture();
        data.time.fill(1.);
        data.status.fill(1);
        for (i, offset) in data.offset.as_mut().unwrap().iter_mut().enumerate() {
            *offset += if i < 6 { 1000. } else { -1000. };
        }
        check_oracle(data, method, 0.6);
    }
}

#[test]
fn overflowing_sum_of_finite_censored_risks_has_finite_event_moments() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let data = CoxphData::try_new(
            vec![1., 2., 2.],
            None,
            vec![1, 0, 0],
            Array2::from_shape_vec((3, 1), vec![0., 0., 1.]).unwrap(),
            Some(vec![1., 1e308, 1e308]),
            Some(vec![0; 3]),
            Some(vec![0.; 3]),
        )
        .unwrap();
        check_oracle(data, method, 0.);
    }
}

#[test]
fn ordinary_fit_arithmetic_is_unchanged() {
    // Captured from the native extension before adding this fallback.
    for (method, beta, variance, loglik, score, first) in [
        (
            TieMethod::Breslow,
            -0.47181093541885316,
            0.23934145791272635,
            [-13.745314327768245, -13.281140852827377],
            0.9554991383721557,
            -2.893219224288793e-11,
        ),
        (
            TieMethod::Efron,
            -0.5111228591008189,
            0.2416576429593524,
            [-13.49729170058594, -12.956718816344019],
            1.1161478327718557,
            -1.1269683652834717e-10,
        ),
    ] {
        let fit = raw_fit(fixture(), method, 0., 40);
        assert_eq!(fit.coefficients, [beta]);
        assert_eq!(fit.var[(0, 0)], variance);
        assert_eq!(fit.loglik, loglik);
        assert_eq!(fit.score, score);
        assert_eq!(fit.first, [first]);
        assert_eq!((fit.iter, fit.flag), (3, 1));
    }
}

#[test]
fn two_covariate_information_and_iterations_are_invariant_to_stratum_shifts() {
    let mut ordinary = fixture();
    let second = [
        0.4, -0.3, 0.8, 1.3, -0.6, 0.2, 0.7, 1.1, -0.1, 0.5, 0.9, -0.4,
    ];
    ordinary.x = Array2::from_shape_fn((12, 2), |(i, j)| {
        if j == 0 {
            ordinary.x[(i, 0)]
        } else {
            second[i]
        }
    });
    let mut shifted = ordinary.clone();
    for (i, offset) in shifted.offset.as_mut().unwrap().iter_mut().enumerate() {
        *offset += if i < 6 { 1000. } else { -1000. };
    }
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let options = CoxphOptions {
            method,
            iter_max: 40,
            ..CoxphOptions::default()
        };
        let control = CoxphFitResult::fit(ordinary.clone(), options.clone(), false).unwrap();
        let fit = CoxphFitResult::fit(shifted.clone(), options, false).unwrap();
        for (actual, expected) in fit.coefficients.iter().zip(&control.coefficients) {
            close(*actual, *expected);
        }
        for (actual, expected) in fit.var.iter().zip(&control.var) {
            close(*actual, *expected);
        }
        for (actual, expected) in fit.loglik.iter().zip(&control.loglik) {
            close(*actual, *expected);
        }
        close(fit.score, control.score);
        assert_eq!((fit.iter, fit.flag), (control.iter, control.flag));
    }
}

fn anchored_fit(
    mut data: CoxphData,
    method: TieMethod,
    counting: bool,
    beta: Vec<f64>,
    iterations: usize,
) -> CoxphFitResult {
    if method == TieMethod::Exact {
        data.weights = None;
    }
    if counting {
        data.entry = Some(
            data.time
                .iter()
                .enumerate()
                .map(|(i, &time)| if i % 3 == 0 { time - 0.5 } else { 0.0 })
                .collect(),
        );
    }
    CoxphFitResult::fit(
        data,
        CoxphOptions {
            method,
            init: Some(beta),
            iter_max: iterations,
            ..CoxphOptions::default()
        },
        false,
    )
    .unwrap()
}

fn anchored_fixture(opposite: bool, cancellation: bool) -> CoxphData {
    let mut data = fixture();
    let columns = if cancellation { 3 } else { 2 };
    data.x = Array2::from_shape_fn((12, columns), |(i, j)| {
        if j == 1 {
            data.x[(i, 0)]
        } else if opposite && i >= 6 {
            -1.0
        } else {
            1.0
        }
    });
    data
}

fn same_anchored_fit(actual: &CoxphFitResult, control: &CoxphFitResult) {
    for (a, b) in actual.loglik.iter().zip(control.loglik) {
        close(*a, b);
    }
    close(actual.coefficients[1], control.coefficients[1]);
    for (a, b) in actual.var.iter().zip(&control.var) {
        assert!((a - b).abs() <= 2e-10 * b.abs().max(1.0), "{a} != {b}");
    }
    close(actual.score, control.score);
    assert_eq!(actual.flag, control.flag);
}

#[test]
fn huge_constant_covariate_effects_preserve_small_terms_in_every_fitter() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            for opposite in [false, true] {
                let data = anchored_fixture(opposite, false);
                let control = anchored_fit(data.clone(), method, counting, vec![0., 0.3], 0);
                for large in [-1e308, -1e16, 1e16, 1e308] {
                    let actual = anchored_fit(data.clone(), method, counting, vec![large, 0.3], 0);
                    same_anchored_fit(&actual, &control);
                    close(actual.first[1], control.first[1]);
                }
            }
        }
    }
}

#[test]
fn cancelling_large_covariate_terms_do_not_erase_the_middle_term() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let data = anchored_fixture(true, true);
            let control = anchored_fit(data.clone(), method, counting, vec![0., 0.3, 0.], 0);
            for large in [1e16, 1e308] {
                let actual =
                    anchored_fit(data.clone(), method, counting, vec![large, 0.3, -large], 0);
                same_anchored_fit(&actual, &control);
                close(actual.first[1], control.first[1]);
            }
        }
    }
}

#[test]
fn iterative_fits_with_a_huge_unidentifiable_constant_match_zero_constant_controls() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let data = anchored_fixture(false, false);
            let control = anchored_fit(data.clone(), method, counting, vec![0., 0.3], 30);
            let actual = anchored_fit(data, method, counting, vec![1e16, 0.3], 30);
            same_anchored_fit(&actual, &control);
            assert_eq!(actual.iter, control.iter);
        }
    }
}

#[test]
fn huge_within_stratum_offsets_retain_tied_risk_multiplicities_and_covariate_effects() {
    // Each risk set contains two rows with equal offset. The early huge-risk
    // pair completely dominates the late tiny pair; removing the latter from
    // that event produces the independently fitted two-stratum control.
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let mut data = CoxphData::try_new(
                vec![1., 1., 2., 2.],
                None,
                vec![1, 0, 1, 0],
                Array2::from_shape_vec((4, 1), vec![0., 1., 0., 1.]).unwrap(),
                None,
                None,
                None,
            )
            .unwrap();
            if counting {
                data.entry = Some(vec![0., 0., 1.5, 1.5]);
            }
            let options = CoxphOptions {
                method,
                init: Some(vec![0.3]),
                iter_max: 0,
                ..CoxphOptions::default()
            };
            let mut reference = data.clone();
            reference.strata = Some(vec![0, 0, 1, 1]);
            let expected = CoxphFitResult::fit(reference, options.clone(), false).unwrap();
            for large in [1e16, 1e308] {
                data.offset = Some(vec![large, large, -large, -large]);
                let actual = CoxphFitResult::fit(data.clone(), options.clone(), false).unwrap();
                close(actual.loglik[0], expected.loglik[0]);
                close(actual.first[0], expected.first[0]);
                close(actual.var[(0, 0)], expected.var[(0, 0)]);
                close(actual.score, expected.score);
            }
        }
    }
}

#[test]
fn varying_large_covariate_terms_cancel_without_erasing_a_small_effect() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let mut data = anchored_fixture(false, true);
            for i in 0..data.n() {
                let value = if i % 3 != 0 { 1.0 } else { 0.0 };
                data.x[(i, 0)] = value;
                data.x[(i, 2)] = value;
            }
            let control = anchored_fit(data.clone(), method, counting, vec![0., 0.3, 0.], 0);
            let actual = anchored_fit(data, method, counting, vec![1e16, 0.3, -1e16], 0);
            same_anchored_fit(&actual, &control);
            close(actual.first[1], control.first[1]);
        }
    }
}

#[test]
fn opposite_extreme_deaths_have_negative_infinite_likelihood_instead_of_nan() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let data = CoxphData::try_new(
                vec![1., 1., 2., 2.],
                counting.then(|| vec![0.; 4]),
                vec![1, 1, 0, 0],
                Array2::zeros((4, 0)),
                None,
                None,
                Some(vec![1e308, -1e308, 0., 0.]),
            )
            .unwrap();
            let result = CoxphFitResult::fit(
                data,
                CoxphOptions {
                    method,
                    ..CoxphOptions::default()
                },
                false,
            )
            .unwrap();
            assert_eq!(result.loglik, [f64::NEG_INFINITY; 2]);
        }
    }
}

#[test]
fn varying_offsets_cancel_large_covariate_terms_before_small_effects_are_rounded() {
    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            let mut data = anchored_fixture(false, false);
            data.offset = Some(vec![0.; data.n()]);
            for i in 0..data.n() {
                data.x[(i, 0)] = if i % 3 == 0 { 0. } else { 1. };
            }
            let control = anchored_fit(data.clone(), method, counting, vec![0., 0.3], 0);
            for large in [1e16, 1e308] {
                for (i, offset) in data.offset.as_mut().unwrap().iter_mut().enumerate() {
                    *offset = -large * data.x[(i, 0)];
                }
                let actual = anchored_fit(data.clone(), method, counting, vec![large, 0.3], 0);
                same_anchored_fit(&actual, &control);
                close(actual.first[1], control.first[1]);
            }
        }
    }
}
