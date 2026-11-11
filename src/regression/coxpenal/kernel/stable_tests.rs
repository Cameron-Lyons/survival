use super::*;
use ndarray::array;

fn fixture(counting: bool, efron: bool, unsafe_offsets: bool) -> Kernel {
    let stop = [1., 1., 2., 3., 3., 4., 5., 5., 6., 7., 8., 8.];
    let start = [0., 0., 0.5, 0., 1., 0., 2., 1., 4., 0., 5., 0.];
    let x = array![
        [1., -1.],
        [1., 0.5],
        [1., 1.],
        [1., -0.5],
        [1., 0.2],
        [1., 1.4],
        [1., 0.2],
        [1., -0.8],
        [1., 0.4],
        [1., 1.2],
        [1., -0.4],
        [1., 0.7]
    ];
    Kernel::new(KernelData {
        stop: &stop,
        start: counting.then_some(start.as_slice()),
        status: &[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        x: &x,
        weights: &[0.5, 1., 2., 1.5, 0.75, 1.25, 1.5, 0.75, 1.25, 2., 1., 0.5],
        offset: &[0.; 12],
        strata: None,
        frail: Some(&[0, 0, 1, 1, 2, 2, 0, 2, 1, 1, 2, 0]),
        nfrail: 3,
        efron,
        docenter: &[false, true],
        unsafe_offsets,
    })
}

// Independent test-only expansion of the sparse indicators. Production retains
// only the frailty diagonal and dense cross block and never allocates this q².
fn oracle(kernel: &Kernel, beta: &[f64], fbeta: &[f64]) -> (f64, Vec<f64>, Array2<f64>, Vec<f64>) {
    let n = kernel.n;
    let q = kernel.nvar + kernel.nf;
    let mut strata = vec![0; n];
    let mut first = 0;
    for (s, &end) in kernel.strata_end.iter().enumerate() {
        for &row in &kernel.sort1[first..end] {
            strata[row] = s;
        }
        first = end;
    }
    let mut z = Array2::zeros((n, q));
    let mut eta = vec![0.; n];
    for row in 0..n {
        if let Some(frail) = &kernel.frail {
            z[(row, frail[row])] = 1.;
            eta[row] += fbeta[frail[row]];
        }
        eta[row] += kernel.offset[row];
        for (i, &b) in beta.iter().enumerate() {
            z[(row, kernel.nf + i)] = kernel.covar[(row, i)];
            eta[row] += b * kernel.covar[(row, i)];
        }
    }
    let mut times: Vec<_> = (0..n).filter(|&row| kernel.status[row] == 1).collect();
    times.sort_by(|&a, &b| {
        strata[a]
            .cmp(&strata[b])
            .then(kernel.stop[a].total_cmp(&kernel.stop[b]))
    });
    times.dedup_by(|a, b| strata[*a] == strata[*b] && kernel.stop[*a] == kernel.stop[*b]);
    let (mut loglik, mut score, mut info, mut expected) =
        (0., vec![0.; q], Array2::zeros((q, q)), vec![0.; n]);
    for event in times {
        let deaths: Vec<_> = (0..n)
            .filter(|&row| {
                strata[row] == strata[event]
                    && kernel.stop[row] == kernel.stop[event]
                    && kernel.status[row] == 1
            })
            .collect();
        let risk: Vec<_> = (0..n)
            .filter(|&row| {
                strata[row] == strata[event]
                    && kernel.stop[row] >= kernel.stop[event]
                    && kernel
                        .start
                        .as_ref()
                        .is_none_or(|start| start[row] < kernel.stop[event])
            })
            .collect();
        let max_log = risk
            .iter()
            .map(|&row| eta[row] + kernel.weights[row].ln())
            .fold(f64::NEG_INFINITY, f64::max);
        let steps = if kernel.method == 1. { deaths.len() } else { 1 };
        for step in 0..steps {
            let fractions: Vec<_> = risk
                .iter()
                .map(|row| {
                    if deaths.contains(row) {
                        1. - step as f64 / deaths.len() as f64
                    } else {
                        1.
                    }
                })
                .collect();
            let risks: Vec<_> = risk
                .iter()
                .zip(&fractions)
                .map(|(&row, &fraction)| {
                    (eta[row] + kernel.weights[row].ln() - max_log).exp() * fraction
                })
                .collect();
            let denominator: f64 = risks.iter().sum();
            let average: f64 =
                deaths.iter().map(|&row| kernel.weights[row]).sum::<f64>() / steps as f64;
            let mean: Vec<f64> = (0..q)
                .map(|j| {
                    risk.iter()
                        .zip(&risks)
                        .map(|(&row, &weight)| weight * z[(row, j)] / denominator)
                        .sum()
                })
                .collect();
            for &row in &deaths {
                let weight = kernel.weights[row] / steps as f64;
                loglik += weight * (eta[row] - max_log - denominator.ln());
                for j in 0..q {
                    score[j] += weight * (z[(row, j)] - mean[j]);
                }
            }
            for ((&row, &weight), &fraction) in risk.iter().zip(&risks).zip(&fractions) {
                expected[row] += average * (eta[row] - max_log).exp() * fraction / denominator;
                for i in 0..q {
                    for j in 0..q {
                        info[(i, j)] += average * weight / denominator
                            * (z[(row, i)] - mean[i])
                            * (z[(row, j)] - mean[j]);
                    }
                }
            }
        }
    }
    (loglik, score, info, expected)
}

fn check(kernel: &mut Kernel, beta: &[f64], reference_beta: &[f64], fbeta: &[f64]) {
    let (expected_lk, expected_u, expected_info, expected_events) =
        oracle(kernel, reference_beta, fbeta);
    let mut u = vec![0.; kernel.nf + kernel.nvar];
    let mut info = Array2::zeros((kernel.nvar, kernel.nf + kernel.nvar));
    let mut diagonal = vec![0.; kernel.nf + kernel.nvar];
    let actual = kernel
        .evaluate(beta, fbeta, &mut u, &mut info, &mut diagonal)
        .unwrap();
    let scale: f64 = kernel.weights.iter().sum();
    let close =
        |a: f64, b: f64| assert!((a - b).abs() <= 2e-10 * (b.abs() + scale), "{a:e} != {b:e}");
    close(actual, expected_lk);
    for (a, b) in u.iter().zip(expected_u) {
        close(*a, b);
    }
    for (g, &value) in diagonal[..kernel.nf].iter().enumerate() {
        close(value, expected_info[(g, g)]);
    }
    for i in 0..kernel.nvar {
        for j in 0..=kernel.nf + i {
            close(info[(i, j)], expected_info[(kernel.nf + i, j)]);
        }
    }
    for (a, b) in kernel
        .expected_events()
        .unwrap()
        .iter()
        .zip(expected_events)
    {
        assert!(
            (a - b).abs() < 2e-10 * (1. + b.abs()),
            "expected count {a} != {b}"
        );
    }
}

#[test]
fn sparse_stable_moments_match_explicit_indicators_with_weights_ties_and_entry() {
    for counting in [false, true] {
        for efron in [false, true] {
            for weight_scale in [1e-200, 1., 1e200] {
                let mut kernel = fixture(counting, efron, true);
                for weight in &mut kernel.weights {
                    *weight *= weight_scale;
                }
                check(&mut kernel, &[1e16, 0.3], &[0., 0.3], &[0.2, -0.1, 0.3]);
            }
        }
    }
}

#[test]
fn tied_sparse_groups_mix_deaths_and_survivors_against_explicit_indicators() {
    for counting in [false, true] {
        for efron in [false, true] {
            let mut kernel = fixture(counting, efron, true);
            kernel.stop.fill(1.);
            if let Some(start) = &mut kernel.start {
                start.fill(0.);
            }
            check(&mut kernel, &[1e16, 0.3], &[0., 0.3], &[0.2, -0.1, 0.3]);
        }
    }
}

#[test]
fn large_tied_sparse_block_matches_uniform_group_information() {
    let n = 1024;
    let x = Array2::ones((n, 1));
    let stop = vec![1.; n];
    let status = vec![1; n];
    let weights = vec![1.; n];
    let offset = vec![0.; n];
    let frail: Vec<_> = (0..n).collect();
    let mut kernel = Kernel::new(KernelData {
        stop: &stop,
        start: None,
        status: &status,
        x: &x,
        weights: &weights,
        offset: &offset,
        strata: None,
        frail: Some(&frail),
        nfrail: n,
        efron: true,
        docenter: &[false],
        unsafe_offsets: false,
    });
    let mut u = vec![0.; n + 1];
    let mut info = Array2::zeros((1, n + 1));
    let mut diagonal = vec![0.; n + 1];
    let loglik = kernel
        .evaluate(&[1e16], &vec![0.; n], &mut u, &mut info, &mut diagonal)
        .unwrap();
    let expected_loglik: f64 = -(1..=n).map(|value| (value as f64).ln()).sum::<f64>();
    assert!((loglik - expected_loglik).abs() < 1e-9);
    assert!(u.iter().all(|value| value.abs() < 1e-10));
    assert!(info.iter().all(|value| value.abs() < 1e-10));
    let expected = 1. - 1. / n as f64;
    assert!(
        diagonal[..n]
            .iter()
            .all(|value| (value - expected).abs() < 1e-10)
    );
    assert!(
        kernel
            .expected_events()
            .unwrap()
            .iter()
            .all(|value| (value - 1.).abs() < 1e-10)
    );
}

#[test]
fn stable_sparse_evaluation_does_not_contaminate_a_following_ordinary_trial() {
    for efron in [false, true] {
        let mut kernel = fixture(false, efron, false);
        check(&mut kernel, &[1e16, 0.3], &[0., 0.3], &[0.2, -0.1, 0.3]);
        check(&mut kernel, &[0., 0.3], &[0., 0.3], &[0.2, -0.1, 0.3]);
        assert!(kernel.stable_score.is_none());
    }
}

#[test]
fn large_relative_predictors_use_the_likelihood_without_clipping() {
    for counting in [false, true] {
        for efron in [false, true] {
            let x = array![[1.], [0.]];
            let mut kernel = Kernel::new(KernelData {
                stop: &[1., 1.],
                start: counting.then_some(&[0., 0.][..]),
                status: &[0, 1],
                x: &x,
                weights: &[1., 1.],
                offset: &[0., 0.],
                strata: None,
                frail: None,
                nfrail: 0,
                efron,
                docenter: &[false],
                unsafe_offsets: false,
            });
            let mut u = vec![0.];
            let mut info = Array2::zeros((1, 1));
            let mut diagonal = vec![0.];
            assert_eq!(
                kernel
                    .evaluate(&[1e16], &[], &mut u, &mut info, &mut diagonal)
                    .unwrap(),
                -1e16
            );
            assert_eq!(u, [-1.]);
            assert_eq!(info[(0, 0)], 0.);
            assert_eq!(kernel.expected_events().unwrap(), [1., 0.]);
            let ordinary = kernel
                .evaluate(&[30.], &[], &mut u, &mut info, &mut diagonal)
                .unwrap();
            assert!((ordinary - (-22. - (-22.0_f64).exp().ln_1p())).abs() < 1e-12);
            assert!(kernel.stable_score.is_none());
        }
    }
}

#[test]
fn dominant_non_death_frailty_group_retains_positive_information() {
    for counting in [false, true] {
        let x = array![[1.], [1.]];
        let mut kernel = Kernel::new(KernelData {
            stop: &[2., 1.],
            start: counting.then_some(&[0., 0.][..]),
            status: &[0, 1],
            x: &x,
            weights: &[1., 1.],
            offset: &[40., 0.],
            strata: None,
            frail: Some(&[0, 1]),
            nfrail: 2,
            efron: true,
            docenter: &[false],
            unsafe_offsets: false,
        });
        let mut u = vec![0.; 3];
        let mut info = Array2::zeros((1, 3));
        let mut diagonal = vec![0.; 3];
        kernel
            .evaluate(&[1e16], &[0., 0.], &mut u, &mut info, &mut diagonal)
            .unwrap();
        let ratio = (-40.0_f64).exp();
        let expected = ratio / (1. + ratio).powi(2);
        for &value in &diagonal[..2] {
            assert!(
                (value / expected - 1.).abs() < 1e-13,
                "{value:e} != {expected:e}"
            );
        }
    }
}

#[test]
fn negligible_sparse_groups_do_not_create_nan_information() {
    for counting in [false, true] {
        for sign in [-1., 1.] {
            let x = array![[1.], [1.]];
            let mut kernel = Kernel::new(KernelData {
                stop: &[2., 1.],
                start: counting.then_some(&[0., 0.][..]),
                status: &[0, 1],
                x: &x,
                weights: &[1., 1.],
                offset: &[sign * 1e308, -sign * 1e308],
                strata: None,
                frail: Some(&[0, 1]),
                nfrail: 2,
                efron: true,
                docenter: &[false],
                unsafe_offsets: true,
            });
            let mut u = vec![0.; 3];
            let mut info = Array2::zeros((1, 3));
            let mut diagonal = vec![0.; 3];
            let loglik = kernel
                .evaluate(&[0.], &[0., 0.], &mut u, &mut info, &mut diagonal)
                .unwrap();
            assert_eq!(loglik, if sign < 0. { 0. } else { f64::NEG_INFINITY });
            assert!(u.iter().all(|value| value.is_finite()));
            assert!(info.iter().all(|&value| value == 0.));
            assert!(diagonal.iter().all(|&value| value == 0.));
            assert_eq!(
                kernel.expected_events().unwrap(),
                if sign < 0. { [0., 1.] } else { [1., 0.] }
            );
        }
    }
}

#[test]
fn overflowing_unweighted_expected_counts_return_an_error_without_panicking() {
    let x = array![[1.0], [1.0]];
    let mut kernel = Kernel::new(KernelData {
        stop: &[1.0, 1.0],
        start: None,
        status: &[0, 1],
        x: &x,
        weights: &[1e-310, 1.0],
        offset: &[500.0, -500.0],
        strata: None,
        frail: None,
        nfrail: 0,
        efron: false,
        docenter: &[false],
        unsafe_offsets: true,
    });
    let mut score = vec![0.0];
    let mut information = Array2::zeros((1, 1));
    let mut diagonal = vec![0.0];
    let likelihood = kernel
        .evaluate(&[0.0], &[], &mut score, &mut information, &mut diagonal)
        .unwrap();
    assert!(likelihood.is_finite());
    assert!(score.iter().all(|value| value.is_finite()));
    assert!(kernel.expected_events().is_err());
}
