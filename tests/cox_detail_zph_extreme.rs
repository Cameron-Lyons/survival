use ndarray::Array2;
use survival::regression::{
    CoxPHFit, CoxZph, CoxphData, CoxphOptions, TieMethod, ZphTransform, cox_zph, coxph_detail,
};

fn fitted(method: TieMethod, counting: bool, strata: bool, shift: f64, common: bool) -> CoxPHFit {
    let time = [1., 2., 2., 3., 4., 4., 5., 6., 7., 8.];
    let entry = [0., 0., 1., 0., 2., 0., 1., 0., 3., 0.];
    let status = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1];
    let x1 = [0., 0.4, 0.8, 0.2, 1., 1.4, 0.6, 1.2, 1.6, 1.8];
    let x2 = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97];
    let times = (0..20)
        .map(|i| time[i % 10] + if !strata && i >= 10 { 20. } else { 0. })
        .collect();
    let starts = (0..20)
        .map(|i| entry[i % 10] + if !strata && i >= 10 { 20. } else { 0. })
        .collect();
    let data = CoxphData::try_new(
        times,
        counting.then_some(starts),
        (0..20).map(|i| status[i % 10]).collect(),
        Array2::from_shape_fn(
            (20, 2),
            |(i, j)| if j == 0 { x1[i % 10] } else { x2[i % 10] },
        ),
        Some((0..20).map(|i| 0.5 + (i % 4) as f64 / 2.).collect()),
        strata.then_some((0..20).map(|i| if i < 10 { -4 } else { 11 }).collect()),
        Some(
            (0..20)
                .map(|i| if common || i < 10 { shift } else { -shift })
                .collect(),
        ),
    )
    .unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method,
            init: Some(vec![0.2, -0.1]),
            iter_max: 0,
            robust: Some(false),
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite(), "nonfinite {a} vs {b}");
    assert!((a - b).abs() < 3e-8 * (1. + b.abs()), "{a} != {b}");
}

fn zph(fit: &CoxPHFit, transform: &ZphTransform) -> CoxZph {
    cox_zph(fit, transform, true, false, true, None).unwrap()
}

fn compare_zph(a: &CoxZph, b: &CoxZph) {
    assert_eq!(a.time, b.time);
    assert_eq!(a.strata, b.strata);
    for (a, b) in a
        .table
        .iter()
        .chain(&a.global_test)
        .zip(b.table.iter().chain(&b.global_test))
    {
        close(a.chisq, b.chisq);
        close(a.df, b.df);
        close(a.p, b.p);
    }
    for (a, b) in a.y.iter().flatten().zip(b.y.iter().flatten()) {
        close(*a, *b);
    }
    for (a, b) in a.var.iter().flatten().zip(b.var.iter().flatten()) {
        close(*a, *b);
    }
}

#[test]
fn detail_and_ph_tests_preserve_stratum_and_disjoint_interval_shifts() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for (counting, strata) in [(false, true), (true, true), (true, false)] {
            let base = fitted(method, counting, strata, 0., false);
            let expected = coxph_detail(&base, true).unwrap();
            for shift in [100., 1000.] {
                let fit = fitted(method, counting, strata, shift, false);
                let actual = coxph_detail(&fit, true).unwrap();
                assert_eq!(actual.time, expected.time);
                assert_eq!(actual.nevent, expected.nevent);
                assert_eq!(actual.nrisk, expected.nrisk);
                assert_eq!(actual.riskmat, expected.riskmat);
                for (a, b) in actual
                    .means
                    .iter()
                    .flatten()
                    .zip(expected.means.iter().flatten())
                {
                    close(*a, *b);
                }
                for (a, b) in actual
                    .score
                    .iter()
                    .flatten()
                    .zip(expected.score.iter().flatten())
                {
                    close(*a, *b);
                }
                for (a, b) in actual
                    .imat
                    .iter()
                    .flatten()
                    .flatten()
                    .zip(expected.imat.iter().flatten().flatten())
                {
                    close(*a, *b);
                }
                assert!(
                    actual
                        .hazard
                        .iter()
                        .chain(&actual.varhaz)
                        .chain(&actual.wtrisk)
                        .all(|value| !value.is_nan())
                );
                for transform in [
                    ZphTransform::Km,
                    ZphTransform::Rank,
                    ZphTransform::Identity,
                    ZphTransform::Log,
                ] {
                    compare_zph(&zph(&fit, &transform), &zph(&base, &transform));
                }
            }
        }
    }
}

#[test]
fn detail_late_risk_sets_match_direct_positive_moments() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let fit = fitted(method, false, false, 1000., false);
        let actual = coxph_detail(&fit, false).unwrap();
        for (g, &time) in actual.time.iter().enumerate() {
            let risk: Vec<usize> = (0..fit.n).filter(|&i| fit.time[i] >= time).collect();
            let deaths: Vec<usize> = risk
                .iter()
                .copied()
                .filter(|&i| fit.time[i] == time && fit.status[i] == 1)
                .collect();
            let high = risk
                .iter()
                .map(|&i| fit.linear_predictors[i] + fit.weights[i].ln())
                .fold(f64::NEG_INFINITY, f64::max);
            let death_weight: f64 = deaths.iter().map(|&i| fit.weights[i]).sum();
            let d = deaths.len();
            let mut mean = [0.; 2];
            let mut info = [[0.; 2]; 2];
            let mut score = [0.; 2];
            for &i in &deaths {
                for (j, value) in score.iter_mut().enumerate() {
                    *value += fit.weights[i] * fit.x[(i, j)];
                }
            }
            for step in 0..d {
                let fraction = if method == TieMethod::Efron {
                    step as f64 / d as f64
                } else {
                    0.
                };
                let weight: Vec<f64> = risk
                    .iter()
                    .map(|&i| {
                        (fit.linear_predictors[i] + fit.weights[i].ln() - high).exp()
                            * if deaths.contains(&i) {
                                1. - fraction
                            } else {
                                1.
                            }
                    })
                    .collect();
                let denom: f64 = weight.iter().sum();
                let xbar: Vec<f64> = (0..2)
                    .map(|j| {
                        risk.iter()
                            .zip(&weight)
                            .map(|(&i, &w)| w * fit.x[(i, j)])
                            .sum::<f64>()
                            / denom
                    })
                    .collect();
                for j in 0..2 {
                    mean[j] += xbar[j] / d as f64;
                    score[j] -= death_weight * xbar[j] / d as f64;
                    for k in 0..2 {
                        info[j][k] += death_weight / d as f64
                            * risk
                                .iter()
                                .zip(&weight)
                                .map(|(&i, &w)| {
                                    w * (fit.x[(i, j)] - xbar[j]) * (fit.x[(i, k)] - xbar[k])
                                })
                                .sum::<f64>()
                            / denom;
                    }
                }
            }
            for j in 0..2 {
                close(actual.means[g][j], mean[j]);
                close(actual.score[g][j], score[j]);
                for (k, &expected) in info[j].iter().enumerate() {
                    close(actual.imat[g][j][k], expected);
                }
            }
        }
    }
}

#[test]
fn enormous_common_offsets_preserve_detail_and_ph_covariate_information() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let base = fitted(method, true, false, 0., true);
        let expected = coxph_detail(&base, false).unwrap();
        for offset in [1e308, -1e308] {
            let fit = fitted(method, true, false, offset, true);
            let actual = coxph_detail(&fit, false).unwrap();
            for (a, b) in actual
                .means
                .iter()
                .flatten()
                .zip(expected.means.iter().flatten())
            {
                close(*a, *b);
            }
            for (a, b) in actual
                .score
                .iter()
                .flatten()
                .zip(expected.score.iter().flatten())
            {
                close(*a, *b);
            }
            for (a, b) in actual
                .imat
                .iter()
                .flatten()
                .flatten()
                .zip(expected.imat.iter().flatten().flatten())
            {
                close(*a, *b);
            }
            assert!(
                actual
                    .hazard
                    .iter()
                    .chain(&actual.varhaz)
                    .chain(&actual.wtrisk)
                    .all(|value| !value.is_nan())
            );
            compare_zph(
                &zph(&fit, &ZphTransform::Km),
                &zph(&base, &ZphTransform::Km),
            );
        }
    }
}

fn dynamic_fit(method: TieMethod, counting: bool, shift: f64) -> CoxPHFit {
    CoxPHFit::fit(
        CoxphData::try_new(
            vec![1., 1., 1., 1., 3., 3., 3., 3.],
            counting.then_some(vec![0., 0., 0., 0., 2., 2., 2., 2.]),
            vec![1, 1, 0, 0, 1, 1, 0, 0],
            Array2::from_shape_vec((8, 1), vec![-1., 0.5, 1.5, 0.25, -1., 0.5, 1.5, 0.25]).unwrap(),
            Some(vec![0.5, 2., 3., 1., 0.5, 2., 3., 1.]),
            None,
            Some(vec![shift, shift, shift, shift, 0., 0., 0., 0.]),
        )
        .unwrap(),
        CoxphOptions {
            method,
            init: Some(vec![0.]),
            iter_max: 0,
            robust: Some(false),
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn dynamic_origins_preserve_detail_covariances_and_ph_statistics() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let base = dynamic_fit(method, counting, if counting { 0. } else { 1000. });
            let expected = coxph_detail(&base, true).unwrap();
            for shift in [1e16, 1e308] {
                let fit = dynamic_fit(method, counting, shift);
                let actual = coxph_detail(&fit, true).unwrap();
                assert_eq!(actual.riskmat, expected.riskmat);
                for (a, b) in actual
                    .means
                    .iter()
                    .flatten()
                    .zip(expected.means.iter().flatten())
                {
                    close(*a, *b);
                }
                for (a, b) in actual
                    .score
                    .iter()
                    .flatten()
                    .zip(expected.score.iter().flatten())
                {
                    close(*a, *b);
                }
                for (a, b) in actual
                    .imat
                    .iter()
                    .flatten()
                    .flatten()
                    .zip(expected.imat.iter().flatten().flatten())
                {
                    close(*a, *b);
                }
                // The final block has zero absolute offset: its baseline
                // hazard and weighted risk count must recover their full mass.
                close(actual.hazard[1], expected.hazard[1]);
                close(actual.varhaz[1], expected.varhaz[1]);
                close(actual.wtrisk[1], expected.wtrisk[1]);
                assert!(
                    actual
                        .hazard
                        .iter()
                        .chain(&actual.varhaz)
                        .chain(&actual.wtrisk)
                        .all(|x| !x.is_nan())
                );
                for transform in [ZphTransform::Km, ZphTransform::Rank, ZphTransform::Identity] {
                    compare_zph(&zph(&fit, &transform), &zph(&base, &transform));
                }
            }
        }
    }
}

#[test]
fn ph_tests_do_not_overflow_global_centering_of_finite_public_predictors() {
    let base = dynamic_fit(TieMethod::Efron, true, 0.);
    let expected = zph(&base, &ZphTransform::Identity);
    for origin in [1e308, -1e308] {
        let mut fit = base.clone();
        fit.linear_predictors.fill(origin);
        compare_zph(&zph(&fit, &ZphTransform::Identity), &expected);
    }
}
