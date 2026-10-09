//! Public Rust inputs remain checked when callers mutate their fields.

use survival::surv_analysis::{
    SurvdiffData, SurvfitAJData, SurvfitAJOptions, SurvfitKMData, SurvfitKMOptions, survdiff,
    survfitaj, survfitkm,
};

#[test]
fn kaplan_meier_rejects_modified_inputs_before_fitting() {
    let original = SurvfitKMData::right_censored(vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap();
    let mutations: &[fn(&mut SurvfitKMData)] = &[
        |d| d.time.clear(),
        |d| d.time[0] = f64::NAN,
        |d| d.status.clear(),
        |d| d.status[0] = 2,
        |d| d.start = Some(vec![0.0]),
        |d| d.start = Some(vec![0.0, 2.0, 0.0]),
        |d| d.start = Some(vec![0.0, f64::NEG_INFINITY, 0.0]),
        |d| d.weights = Some(vec![1.0]),
        |d| d.weights = Some(vec![1.0, -1.0, 1.0]),
        |d| d.weights = Some(vec![1.0, f64::INFINITY, 1.0]),
        |d| d.strata = Some(vec![1]),
        |d| d.id = Some(vec![1]),
        |d| d.cluster = Some(vec![1]),
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut data = original.clone();
        mutate(&mut data);
        for timefix in [false, true] {
            let options = SurvfitKMOptions {
                timefix,
                ..Default::default()
            };
            assert!(survfitkm(&data, &options).is_err(), "case {case}");
        }
    }
    assert!(survfitkm(&original, &SurvfitKMOptions::default()).is_ok());
}

#[test]
fn aalen_johansen_rejects_modified_inputs_before_fitting() {
    let original = SurvfitAJData::try_new(
        None,
        vec![1.0, 2.0, 3.0],
        vec![1, 0, 2],
        vec!["a".into(), "b".into()],
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let mutations: &[fn(&mut SurvfitAJData)] = &[
        |d| d.time.clear(),
        |d| d.time[0] = f64::INFINITY,
        |d| d.state.clear(),
        |d| d.state[0] = -1,
        |d| d.state[0] = 3,
        |d| d.states.clear(),
        |d| d.start = Some(vec![0.0]),
        |d| d.start = Some(vec![0.0, 3.0, 0.0]),
        |d| d.start = Some(vec![f64::NAN, 0.0, 0.0]),
        |d| d.weights = Some(vec![1.0]),
        |d| d.weights = Some(vec![1.0, -1.0, 1.0]),
        |d| d.weights = Some(vec![1.0, f64::NAN, 1.0]),
        |d| d.strata = Some(vec![1]),
        |d| d.id = Some(vec![1]),
        |d| d.cluster = Some(vec![1]),
        |d| d.istate = Some(vec!["a".into()]),
        |d| {
            d.istate = Some(vec!["a".into(); 3]);
            d.istate_levels = Some(vec!["b".into()]);
        },
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut data = original.clone();
        mutate(&mut data);
        for timefix in [false, true] {
            let options = SurvfitAJOptions {
                timefix,
                ..Default::default()
            };
            assert!(survfitaj(&data, &options).is_err(), "case {case}");
        }
    }
    assert!(survfitaj(&original, &SurvfitAJOptions::default()).is_ok());
}

#[test]
fn logrank_rejects_modified_inputs_before_fitting() {
    let original = SurvdiffData::try_new(
        None,
        vec![1.0, 2.0, 3.0],
        vec![1, 0, 1],
        vec![0, 1, 0],
        None,
    )
    .unwrap();
    let mutations: &[fn(&mut SurvdiffData)] = &[
        |d| d.time.clear(),
        |d| d.time[0] = f64::NAN,
        |d| d.status.clear(),
        |d| d.status[0] = 2,
        |d| d.group.clear(),
        |d| d.start = Some(vec![0.0]),
        |d| d.start = Some(vec![0.0, 2.0, 0.0]),
        |d| d.start = Some(vec![0.0, f64::INFINITY, 0.0]),
        |d| d.strata = Some(vec![1]),
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let mut data = original.clone();
        mutate(&mut data);
        for timefix in [false, true] {
            for rho in [0.0, 1.0] {
                assert!(survdiff(&data, rho, timefix).is_err(), "case {case}");
            }
        }
    }
    assert!(survdiff(&original, 0.0, true).is_ok());
}
