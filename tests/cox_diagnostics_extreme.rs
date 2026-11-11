use ndarray::{Array2, array};
use survival::regression::{CoxPHFit, CoxphData, CoxphOptions, TieMethod};

fn fit(method: TieMethod, counting: bool, strata: bool, shift: f64, robust: bool) -> CoxPHFit {
    let mut time = vec![2., 2., 3., 5., 2., 3., 3., 5.];
    let mut entry = vec![0., 1., 0., 2., 0., 1., 0., 3.];
    if !strata {
        for value in &mut time[4..] {
            *value += 10.;
        }
        for value in &mut entry[4..] {
            *value += 10.;
        }
    }
    let offset = (0..8)
        .map(|row| if row < 4 { shift } else { -shift })
        .collect();
    let data = CoxphData::try_new(
        time,
        counting.then_some(entry),
        vec![1, 1, 0, 1, 0, 1, 1, 0],
        array![
            [-1.2, 0.5],
            [0.4, -1.0],
            [1.1, 0.3],
            [-0.3, 1.2],
            [0.8, -0.7],
            [1.7, 0.9],
            [-0.9, 0.1],
            [0.2, -1.3]
        ],
        Some(vec![1., 0.5, 2., 0.75, 0.25, 1.5, 1., 2.]),
        strata.then_some(vec![-4, -4, -4, -4, 11, 11, 11, 11]),
        Some(offset),
    )
    .unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method,
            init: Some(vec![0.2, -0.1]),
            iter_max: 0,
            robust: Some(robust),
            cluster: robust.then_some(vec![0, 0, 1, 1, 2, 2, 3, 3]),
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= 2e-10 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

fn close_matrix(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (&actual, &expected) in actual.iter().zip(expected) {
        close(actual, expected);
    }
}

struct DirectResiduals {
    martingale: Vec<f64>,
    score: Array2<f64>,
    schoenfeld: Vec<(usize, Vec<f64>)>,
}

/// Independent per-time definition: normalize each active set, then distribute
/// its event and expected increments directly to every subject. No cumulative
/// integrals or tree operations are shared with the implementation.
fn direct(model: &CoxPHFit) -> DirectResiduals {
    let n = model.n;
    let p = model.x.ncols();
    let strata = model.strata.clone().unwrap_or(vec![0; n]);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        strata[a]
            .cmp(&strata[b])
            .then_with(|| model.time[a].total_cmp(&model.time[b]))
    });
    let mut martingale: Vec<f64> = model.status.iter().map(|&x| f64::from(x)).collect();
    let mut score = Array2::zeros((n, p));
    let mut schoenfeld = Vec::new();
    let mut first = 0;
    while first < n {
        let row = order[first];
        let time = model.time[row];
        let label = strata[row];
        let mut end = first + 1;
        while end < n && strata[order[end]] == label && model.time[order[end]] == time {
            end += 1;
        }
        let deaths: Vec<usize> = order[first..end]
            .iter()
            .copied()
            .filter(|&i| model.status[i] == 1)
            .collect();
        first = end;
        if deaths.is_empty() {
            continue;
        }
        let risk: Vec<usize> = (0..n)
            .filter(|&i| {
                strata[i] == label
                    && model.time[i] >= time
                    && model.entry.as_ref().is_none_or(|entry| entry[i] < time)
            })
            .collect();
        let maximum = risk
            .iter()
            .map(|&i| model.linear_predictors[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let normalized: Vec<f64> = (0..n)
            .map(|i| (model.linear_predictors[i] - maximum).exp() * model.weights[i])
            .collect();
        let denominator: f64 = risk.iter().map(|&i| normalized[i]).sum();
        let death_risk: f64 = deaths.iter().map(|&i| normalized[i]).sum();
        let death_weight: f64 = deaths.iter().map(|&i| model.weights[i]).sum();
        let steps = if model.method == TieMethod::Efron {
            deaths.len()
        } else {
            1
        };
        let mut average = vec![0.; p];
        for step in 0..steps {
            let fraction = step as f64 / deaths.len() as f64;
            let denom = denominator - fraction * death_risk;
            let mean: Vec<f64> = (0..p)
                .map(|j| {
                    (risk
                        .iter()
                        .map(|&i| normalized[i] * model.x[(i, j)])
                        .sum::<f64>()
                        - fraction
                            * deaths
                                .iter()
                                .map(|&i| normalized[i] * model.x[(i, j)])
                                .sum::<f64>())
                        / denom
                })
                .collect();
            for j in 0..p {
                average[j] += mean[j] / steps as f64;
            }
            for &i in &risk {
                let down = if model.status[i] == 1 && model.time[i] == time {
                    1. - fraction
                } else {
                    1.
                };
                let expected = (model.linear_predictors[i] - maximum).exp()
                    * down
                    * (death_weight / steps as f64)
                    / denom;
                martingale[i] -= expected;
                for j in 0..p {
                    score[(i, j)] -= expected * (model.x[(i, j)] - mean[j]);
                }
            }
        }
        for i in deaths {
            let value: Vec<f64> = (0..p).map(|j| model.x[(i, j)] - average[j]).collect();
            for j in 0..p {
                score[(i, j)] += value[j];
            }
            schoenfeld.push((i, value));
        }
    }
    DirectResiduals {
        martingale,
        score,
        schoenfeld,
    }
}

fn check_risk_definition(model: &CoxPHFit) {
    let DirectResiduals {
        martingale,
        score,
        schoenfeld,
    } = direct(model);
    for (&actual, expected) in model.residuals.iter().zip(martingale) {
        close(actual, expected);
    }
    close_matrix(&model.score_residuals(false, None).unwrap(), &score);
    let actual = model.schoenfeld_residuals(false).unwrap();
    assert_eq!(
        actual.rows,
        schoenfeld.iter().map(|(row, _)| *row).collect::<Vec<_>>()
    );
    for (actual, (_, expected)) in actual.residuals.iter().zip(&schoenfeld) {
        for (&actual, &expected) in actual.iter().zip(expected) {
            close(actual, expected);
        }
    }
    let weighted = model.score_residuals(true, None).unwrap();
    for i in 0..model.n {
        for j in 0..model.x.ncols() {
            close(weighted[(i, j)], score[(i, j)] * model.weights[i]);
        }
    }
}

fn check_definition(model: &CoxPHFit) {
    check_risk_definition(model);
    for values in [
        model.dfbeta(true, None).unwrap(),
        model.dfbetas(true, None).unwrap(),
    ] {
        assert!(values.iter().all(|value| value.is_finite()));
    }
    assert!(
        model
            .scaled_schoenfeld_residuals(false)
            .unwrap()
            .residuals
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(
        model
            .deviance_residuals(false, None)
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn full_cox_diagnostics_match_direct_definition_at_extreme_log_risks() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            for strata in [false, true] {
                for shift in [100., 1000.] {
                    check_definition(&fit(method, counting, strata, shift, false));
                }
            }
        }
    }
}

#[test]
fn stratum_and_disjoint_interval_shifts_preserve_robust_diagnostics() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for (counting, strata) in [(false, true), (true, true), (true, false)] {
            let base = fit(method, counting, strata, 0., true);
            let shifted = fit(method, counting, strata, 1000., true);
            for (&actual, &expected) in shifted.residuals.iter().zip(&base.residuals) {
                close(actual, expected);
            }
            close_matrix(&shifted.var, &base.var);
            close_matrix(
                &shifted.score_residuals(false, None).unwrap(),
                &base.score_residuals(false, None).unwrap(),
            );
            close_matrix(
                &shifted.dfbeta(true, None).unwrap(),
                &base.dfbeta(true, None).unwrap(),
            );
            close_matrix(
                &shifted.dfbetas(true, None).unwrap(),
                &base.dfbetas(true, None).unwrap(),
            );
            let actual = shifted.scaled_schoenfeld_residuals(false).unwrap();
            let expected = base.scaled_schoenfeld_residuals(false).unwrap();
            assert_eq!(actual.rows, expected.rows);
            for (a, b) in actual.residuals.iter().zip(&expected.residuals) {
                for (&a, &b) in a.iter().zip(b) {
                    close(a, b);
                }
            }
        }
    }
}

fn dynamic_fit(method: TieMethod, counting: bool, shift: f64) -> CoxPHFit {
    let data = CoxphData::try_new(
        vec![1., 1., 1., 1., 3., 3., 3., 3.],
        counting.then_some(vec![0., 0., 0., 0., 2., 2., 2., 2.]),
        vec![1, 1, 0, 0, 1, 1, 0, 0],
        array![[-1.], [0.5], [1.5], [0.25], [-1.], [0.5], [1.5], [0.25]],
        Some(vec![0.5, 2., 3., 1., 0.5, 2., 3., 1.]),
        None,
        Some(vec![shift, shift, shift, shift, 0., 0., 0., 0.]),
    )
    .unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method,
            init: Some(vec![0.]),
            iter_max: 0,
            robust: Some(true),
            cluster: Some(vec![0, 0, 1, 1, 2, 2, 3, 3]),
            ..CoxphOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn dynamic_huge_risk_origins_preserve_weights_and_residual_integrals() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            // Large early risks suppress the later block until the early rows
            // depart. Counting intervals make the blocks fully disjoint.
            let base = dynamic_fit(method, counting, if counting { 0. } else { 1000. });
            for shift in [1e16, 1e308] {
                let fit = dynamic_fit(method, counting, shift);
                check_definition(&fit);
                close_matrix(&fit.var, &base.var);
                close_matrix(
                    &fit.score_residuals(false, None).unwrap(),
                    &base.score_residuals(false, None).unwrap(),
                );
                close_matrix(
                    &fit.dfbeta(true, None).unwrap(),
                    &base.dfbeta(true, None).unwrap(),
                );
                close_matrix(
                    &fit.dfbetas(true, None).unwrap(),
                    &base.dfbetas(true, None).unwrap(),
                );
            }
        }
    }
}

#[test]
fn common_entry_diagnostics_keep_small_early_expected_counts() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let data = CoxphData::try_new(
                vec![1., 1., 1., 2., 3.],
                counting.then_some(vec![0.; 5]),
                vec![1, 0, 0, 1, 0],
                array![[-1.], [0.5], [1.], [-0.5], [0.2]],
                None,
                None,
                Some(vec![0., 0., 0., -100., 1e-10_f64.ln()]),
            )
            .unwrap();
            let model = CoxPHFit::fit(
                data,
                CoxphOptions {
                    method,
                    init: Some(vec![0.]),
                    iter_max: 0,
                    robust: Some(false),
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            let expected = 1. / (3. + 1e-10 + (-100_f64).exp());
            close(model.residuals[0], 1. - expected);
            close(model.residuals[1], -expected);
            close(model.residuals[2], -expected);
            // A tiny expected count rounds the later event's martingale to
            // one, where the existing deviance reconstruction is infinite.
            check_risk_definition(&model);
        }
    }
}

#[test]
fn common_entry_score_moments_keep_large_constant_design_exactly_zero() {
    for counting in [false, true] {
        for n in [8, 12] {
            let data = CoxphData::try_new(
                (1..=n).map(|time| time as f64).collect(),
                counting.then_some(vec![0.; n]),
                (0..n).map(|row| i32::from(row < n - 3)).collect(),
                Array2::from_elem((n, 1), 1e16),
                None,
                None,
                Some(
                    (0..n)
                        .map(|row| if row < n - 3 { -100. } else { 0. })
                        .collect(),
                ),
            )
            .unwrap();
            let model = CoxPHFit::fit(
                data,
                CoxphOptions {
                    init: Some(vec![0.]),
                    iter_max: 0,
                    robust: Some(false),
                    ..CoxphOptions::default()
                },
            )
            .unwrap();
            assert!(
                model
                    .score_residuals(false, None)
                    .unwrap()
                    .iter()
                    .all(|&score| score == 0.)
            );
            assert!(
                model
                    .schoenfeld_residuals(false)
                    .unwrap()
                    .residuals
                    .iter()
                    .flatten()
                    .all(|&score| score == 0.)
            );
        }
    }
}

#[test]
fn common_entry_weighted_tied_and_stratified_diagnostics_match_definition() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            for stratified in [false, true] {
                let rows: Vec<_> = (0..12).rev().collect();
                let stop = [1., 1., 2., 3., 3., 4.];
                let status = [1, 1, 0, 1, 1, 0];
                let weights = [0.5, 2., 1.5, 3., 1., 2.];
                let values = [
                    [-1., 0.2],
                    [0.5, -0.5],
                    [1., 0.3],
                    [-0.5, 1.2],
                    [0.2, -0.9],
                    [0.7, 0.4],
                ];
                let data = CoxphData::try_new(
                    rows.iter().map(|&row| stop[row % 6]).collect(),
                    counting.then_some(vec![0.; 12]),
                    rows.iter().map(|&row| status[row % 6]).collect(),
                    Array2::from_shape_fn((12, 2), |(i, j)| values[rows[i] % 6][j]),
                    Some(rows.iter().map(|&row| weights[row % 6]).collect()),
                    stratified.then(|| {
                        rows.iter()
                            .map(|&row| if row < 6 { -4 } else { 19 })
                            .collect()
                    }),
                    Some(
                        rows.iter()
                            .map(|&row| if status[row % 6] == 1 { -100. } else { 0. })
                            .collect(),
                    ),
                )
                .unwrap();
                let model = CoxPHFit::fit(
                    data,
                    CoxphOptions {
                        method,
                        init: Some(vec![0.2, -0.1]),
                        iter_max: 0,
                        robust: Some(false),
                        ..CoxphOptions::default()
                    },
                )
                .unwrap();
                // Only two distinct censor covariates carry representable
                // information, so standardized residuals can be undefined.
                check_risk_definition(&model);
            }
        }
    }
}
