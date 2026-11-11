use ndarray::{Array2, array};
use survival::regression::{CoxPHFit, CoxphData, CoxphOptions, TieMethod, coxph_detail};

fn fitted(counting: bool, stratified: bool, method: TieMethod) -> CoxPHFit {
    let data = CoxphData::try_new(
        vec![1., 1., 2., 3., 3., 4., 5., 5.],
        counting.then(|| vec![0., 0., 1., 1., 0., 2., 3., 2.]),
        vec![1, 1, 0, 1, 1, 0, 1, 0],
        array![
            [0.2, 1.0],
            [0.8, 0.2],
            [0.4, 0.7],
            [1.1, 1.3],
            [0.7, 0.4],
            [0.3, 1.1],
            [1.3, 0.5],
            [0.5, 0.9]
        ],
        None,
        stratified.then(|| vec![2, 1, 2, 1, 2, 1, 2, 1]),
        None,
    )
    .unwrap();
    CoxPHFit::fit(
        data,
        CoxphOptions {
            method,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn detail_rejects_modified_fit_dimensions() {
    let original = fitted(true, true, TieMethod::Efron);
    let mutations: &[fn(&mut CoxPHFit)] = &[
        |fit| fit.time.clear(),
        |fit| fit.status.clear(),
        |fit| fit.weights.clear(),
        |fit| fit.linear_predictors.clear(),
        |fit| fit.means.clear(),
        |fit| fit.coefficients.clear(),
        |fit| fit.x = Array2::zeros((1, fit.nvar())),
        |fit| fit.x = Array2::zeros((fit.n, fit.nvar() + 1)),
        |fit| fit.entry = Some(vec![0.]),
        |fit| fit.strata = Some(vec![1]),
        |fit| fit.n += 1,
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut fit = original.clone();
        mutate(&mut fit);
        for riskmat in [false, true] {
            assert!(
                coxph_detail(&fit, riskmat).is_err(),
                "case {case}, riskmat={riskmat}"
            );
        }
    }
}

#[test]
fn detail_rejects_invalid_risk_set_values() {
    let original = fitted(true, true, TieMethod::Breslow);
    let mutations: &[fn(&mut CoxPHFit)] = &[
        |fit| fit.time[0] = f64::NAN,
        |fit| fit.time[0] = f64::INFINITY,
        |fit| fit.status[0] = 2,
        |fit| fit.weights[0] = 0.,
        |fit| fit.weights[0] = -1.,
        |fit| fit.weights[0] = f64::INFINITY,
        |fit| fit.linear_predictors[0] = f64::NAN,
        |fit| fit.linear_predictors[0] = f64::INFINITY,
        |fit| fit.means[0] = f64::NAN,
        |fit| fit.x[(0, 0)] = f64::NAN,
        |fit| fit.entry.as_mut().unwrap()[0] = f64::NEG_INFINITY,
        |fit| fit.entry.as_mut().unwrap()[0] = fit.time[0],
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut fit = original.clone();
        mutate(&mut fit);
        assert!(coxph_detail(&fit, false).is_err(), "case {case}");
    }
}

#[test]
fn detail_rejects_stale_time_and_stratum_order() {
    let original = fitted(false, true, TieMethod::Efron);
    let mutations: &[fn(&mut CoxPHFit)] = &[
        |fit| fit.time.swap(0, 6),
        |fit| fit.strata.as_mut().unwrap().swap(0, 1),
        |fit| fit.strata.as_mut().unwrap()[0] = 99,
        |fit| fit.strata = None,
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut fit = original.clone();
        mutate(&mut fit);
        for riskmat in [false, true] {
            assert!(
                coxph_detail(&fit, riskmat).is_err(),
                "case {case}, riskmat={riskmat}"
            );
        }
    }
}

#[test]
fn valid_detail_risk_sets_match_the_fitted_rows() {
    for counting in [false, true] {
        for stratified in [false, true] {
            for method in [TieMethod::Efron, TieMethod::Breslow] {
                let fit = fitted(counting, stratified, method);
                let detail = coxph_detail(&fit, true).unwrap();
                let riskmat = detail.riskmat.as_ref().unwrap();
                for (column, &time) in detail.time.iter().enumerate() {
                    let mut count = 0;
                    for (row, values) in riskmat.iter().enumerate() {
                        let expected = fit.time[row] >= time
                            && fit.entry.as_ref().is_none_or(|entry| entry[row] < time)
                            && fit.strata.as_ref().is_none_or(|strata| {
                                strata[row] == detail.strata.as_ref().unwrap()[column]
                            });
                        assert_eq!(values[column], i32::from(expected));
                        count += usize::from(expected);
                    }
                    assert_eq!(detail.nrisk[column], count);
                }
                let mut without_riskmat = detail;
                without_riskmat.riskmat = None;
                assert_eq!(coxph_detail(&fit, false).unwrap(), without_riskmat);
            }
        }
    }
}
