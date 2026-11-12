//! Downstream multistate Cox fitting and curves, with no Python runtime.

use ndarray::{Array2, array};
use survival::regression::{CoxphOptions, CoxphmsData, CoxphmsDesign, TieMethod, coxphms_fit};
use survival::surv_analysis::{
    CoxphmsCurveData, CoxphmsCurveOptions, CoxphmsNewData, coxphms_curves,
};

fn observations() -> CoxphmsData {
    CoxphmsData {
        time: vec![1., 2., 2., 2., 3., 3., 4., 5., 5., 6., 7., 8.],
        entry: None,
        endpoint: vec![2, 2, 2, 3, 2, 0, 3, 2, 3, 3, 0, 2],
        istate: vec![1; 12],
        x: array![
            [0.],
            [1.],
            [0.],
            [1.],
            [0.],
            [1.],
            [1.],
            [0.],
            [0.],
            [1.],
            [0.],
            [1.]
        ],
        strata_terms: vec![],
        id: (1..=12).collect(),
        x_assign: vec![1],
        weights: None,
        offset: None,
        cluster: None,
    }
}

fn design() -> CoxphmsDesign {
    CoxphmsDesign {
        cmap: array![[1, 2]],
        nx: 1,
        baseline: vec![1, 2],
        strata_use: Array2::from_elem((0, 2), false),
        from: vec![1, 1],
        to: vec![2, 3],
    }
}

fn fit_options() -> CoxphOptions {
    CoxphOptions {
        method: TieMethod::Efron,
        robust: Some(false),
        ..CoxphOptions::default()
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.abs().max(1.),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn downstream_fit_and_curves_match_stock_r() {
    // R survival 3.8-11: event <- factor(endpoint, levels=c(0,2,3),
    // labels=c("censor","a","b")); coxph(Surv(time, event) ~ x,
    // data, id=id, ties="efron", robust=FALSE), followed by
    // survfit(fit, newdata=data.frame(x=c(0,1)), stype=2, ctype=2).
    let observations = observations();
    let design = design();
    let model = coxphms_fit(&observations, &design, fit_options()).unwrap();
    close(
        &model.fit.coefficients,
        &[-1.3959169955233046, 1.0198235800357465],
    );
    close(
        &model.fit.loglik,
        &[-17.77197141282299, -16.371331604004798],
    );
    close(
        &[model.fit.var[(0, 0)], model.fit.var[(1, 1)]],
        &[1.2583287343436285, 1.3703332854792003],
    );
    assert_eq!(model.stack.rindex, (0..12).chain(0..12).collect::<Vec<_>>());
    assert_eq!(model.stack.hazard, [vec![0; 12], vec![1; 12]].concat());
    assert_eq!(model.stack.block, [vec![1; 12], vec![2; 12]].concat());
    assert!(model.share.is_none());

    let states = vec!["(s0)".into(), "a".into(), "b".into()];
    let data = CoxphmsCurveData {
        observations: &observations,
        design: &design,
        states: &states,
        strata: None,
        beta: &model.fit.coefficients,
        means: &[0.0],
        share_scale: None,
    };
    let x = array![[0.0], [1.0]];
    let newdata = CoxphmsNewData {
        x: x.view(),
        offset: None,
    };
    let curves = coxphms_curves(
        &data,
        &newdata,
        &CoxphmsCurveOptions {
            ctype: 2,
            ..CoxphmsCurveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(curves.baseline.time, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    assert_eq!(curves.pstate.shape(), &[8, 2, 3]);
    assert_eq!(curves.cumhaz.shape(), &[8, 2, 2]);
    close(
        &curves.pstate.slice(ndarray::s![7, 0, ..]).to_vec(),
        &[
            0.005042970825669924,
            0.8181738099512157,
            0.17678321922311435,
        ],
    );
    close(
        &curves.pstate.slice(ndarray::s![7, 1, ..]).to_vec(),
        &[0.09721223781002042, 0.3231754860107074, 0.5796122761792721],
    );
    close(
        &curves.cumhaz.slice(ndarray::s![1, 0, ..]).to_vec(),
        &[0.3602207547102648, 0.046218765683776246],
    );
    close(
        &curves.cumhaz.slice(ndarray::s![7, 1, ..]).to_vec(),
        &[1.2096504348346935, 1.121208237210946],
    );

    let empty = Array2::zeros((0, 1));
    let curves = coxphms_curves(
        &data,
        &CoxphmsNewData {
            x: empty.view(),
            offset: None,
        },
        &CoxphmsCurveOptions::default(),
    )
    .unwrap();
    assert_eq!(curves.pstate.shape(), &[8, 0, 3]);
    assert_eq!(curves.cumhaz.shape(), &[8, 0, 2]);
}

#[test]
fn shared_baselines_use_observed_subject_and_state_codes() {
    let data = observations();
    let mut design = design();
    design.baseline = vec![7, 7];
    let reference = coxphms_fit(&data, &design, fit_options()).unwrap();
    let mut sparse = data;
    sparse.id = (0..12).map(|i| i32::MAX - i).collect();
    sparse.istate.fill(usize::MAX);
    design.from.fill(usize::MAX);
    let fit = coxphms_fit(&sparse, &design, fit_options()).unwrap();
    assert_eq!(fit.fit.coefficients, reference.fit.coefficients);
    assert_eq!(fit.fit.loglik, reference.fit.loglik);
    assert_eq!(fit.stack.rindex, reference.stack.rindex);
    assert_eq!(fit.share, reference.share);
}

#[test]
fn public_fit_revalidates_design_and_observation_fields() {
    let data = observations();
    let original = design();
    for kind in 0..9 {
        let mut design = original.clone();
        match kind {
            0 => {
                design.baseline.pop();
            }
            1 => {
                design.from.pop();
            }
            2 => design.to[0] = 0,
            3 => design.to[1] = design.to[0],
            4 => design.cmap[(0, 0)] = -1,
            5 => design.cmap[(0, 0)] = i32::MAX,
            6 => design.strata_use = Array2::from_elem((0, 1), false),
            7 => design.nx = 2,
            _ => design.baseline[0] = 0,
        }
        assert!(
            coxphms_fit(&data, &design, fit_options()).is_err(),
            "design case {kind}"
        );
    }
    for kind in 0..13 {
        let mut data = data.clone();
        match kind {
            0 => {
                data.endpoint.pop();
            }
            1 => {
                data.istate.pop();
            }
            2 => {
                data.id.pop();
            }
            3 => data.x = Array2::zeros((1, 1)),
            4 => data.x_assign.clear(),
            5 => data.weights = Some(vec![1.0]),
            6 => data.weights = Some(vec![0.0; 12]),
            7 => data.offset = Some(vec![f64::NAN; 12]),
            8 => data.entry = Some(data.time.clone()),
            9 => data.time[0] = f64::NAN,
            10 => data.id[0] = 0,
            11 => data.istate[0] = 0,
            _ => data.endpoint[0] = 4,
        }
        assert!(
            coxphms_fit(&data, &original, fit_options()).is_err(),
            "data case {kind}"
        );
    }
    assert!(
        coxphms_fit(
            &data,
            &original,
            CoxphOptions {
                cluster: Some(vec![1]),
                ..fit_options()
            }
        )
        .is_err()
    );
    assert!(
        coxphms_fit(
            &data,
            &original,
            CoxphOptions {
                method: TieMethod::Exact,
                ..fit_options()
            }
        )
        .is_err()
    );
}

#[test]
fn public_curves_validate_shapes_codes_and_prediction_values() {
    let observations = observations();
    let design = design();
    let states = vec!["(s0)".into(), "a".into(), "b".into()];
    let x = array![[0.0], [1.0]];
    let newdata = CoxphmsNewData {
        x: x.view(),
        offset: None,
    };
    let options = CoxphmsCurveOptions::default();
    let data = CoxphmsCurveData {
        observations: &observations,
        design: &design,
        states: &states,
        strata: None,
        beta: &[0.5, -0.3],
        means: &[0.0],
        share_scale: None,
    };
    assert!(
        coxphms_curves(
            &CoxphmsCurveData {
                beta: &[0.5],
                ..data
            },
            &newdata,
            &options
        )
        .is_err()
    );
    assert!(coxphms_curves(&CoxphmsCurveData { means: &[], ..data }, &newdata, &options).is_err());
    assert!(
        coxphms_curves(
            &CoxphmsCurveData {
                strata: Some(&[0]),
                ..data
            },
            &newdata,
            &options
        )
        .is_err()
    );
    assert!(
        coxphms_curves(
            &CoxphmsCurveData {
                states: &states[..2],
                ..data
            },
            &newdata,
            &options
        )
        .is_err()
    );
    assert!(
        coxphms_curves(
            &CoxphmsCurveData {
                beta: &[f64::NAN, 0.],
                ..data
            },
            &newdata,
            &options
        )
        .is_err()
    );
    assert!(
        coxphms_curves(
            &CoxphmsCurveData {
                share_scale: Some(&[1.]),
                ..data
            },
            &newdata,
            &options
        )
        .is_err()
    );
    assert!(
        coxphms_curves(
            &data,
            &CoxphmsNewData {
                offset: Some(&[0.]),
                ..newdata
            },
            &options
        )
        .is_err()
    );
    let bad_x = array![[f64::INFINITY]];
    assert!(
        coxphms_curves(
            &data,
            &CoxphmsNewData {
                x: bad_x.view(),
                offset: None
            },
            &options
        )
        .is_err()
    );
    assert!(
        coxphms_curves(
            &data,
            &newdata,
            &CoxphmsCurveOptions {
                ctype: 3,
                ..options
            }
        )
        .is_err()
    );
}
