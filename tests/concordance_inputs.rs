//! Concordance checks public response fields before ordering or sweeping rows.

use ndarray::Array2;
use survival::concordance::{ConcordanceOptions, concordancefit};
use survival::core::SurvResponse;
use survival::data_types::{CountingProcessData, SurvivalData};

fn predictor() -> Array2<f64> {
    Array2::from_shape_vec((3, 1), vec![0.2, 0.5, 0.3]).unwrap()
}

fn options() -> impl Iterator<Item = ConcordanceOptions> {
    [false, true].into_iter().flat_map(|timefix| {
        [false, true]
            .into_iter()
            .map(move |std_err| ConcordanceOptions {
                timefix,
                std_err,
                ..Default::default()
            })
    })
}

#[test]
fn concordance_rejects_modified_right_censored_inputs() {
    let original = SurvivalData::try_new(vec![1.0, 2.0, 3.0], vec![1, 0, 1]).unwrap();
    let mutations: &[fn(&mut SurvivalData)] = &[
        |d| d.time.clear(),
        |d| d.time.push(4.0),
        |d| d.time[0] = f64::NAN,
        |d| d.time[0] = f64::INFINITY,
        |d| d.status.clear(),
        |d| d.status.push(1),
        |d| d.status[0] = -1,
        |d| d.status[0] = 2,
    ];
    let x = predictor();
    for (case, mutate) in mutations.iter().enumerate() {
        let mut data = original.clone();
        mutate(&mut data);
        for options in options() {
            assert!(
                concordancefit(
                    SurvResponse::Right(&data),
                    x.view(),
                    None,
                    None,
                    None,
                    &options
                )
                .is_err(),
                "case {case}, timefix={}, std_err={}",
                options.timefix,
                options.std_err
            );
        }
    }
    for options in options() {
        assert!(
            concordancefit(
                SurvResponse::Right(&original),
                x.view(),
                None,
                None,
                None,
                &options
            )
            .is_ok()
        );
    }
}

#[test]
fn concordance_rejects_modified_counting_process_inputs() {
    let original =
        CountingProcessData::try_new(vec![0.0, 0.5, 1.0], vec![1.0, 2.0, 3.0], vec![1, 0, 1])
            .unwrap();
    let mutations: &[fn(&mut CountingProcessData)] = &[
        |d| d.start.clear(),
        |d| d.start.push(0.0),
        |d| d.start[0] = f64::NAN,
        |d| d.start[0] = f64::NEG_INFINITY,
        |d| d.start[0] = 1.0,
        |d| d.stop.clear(),
        |d| d.stop.push(4.0),
        |d| d.stop[0] = f64::NAN,
        |d| d.stop[0] = f64::INFINITY,
        |d| d.event.clear(),
        |d| d.event.push(1),
        |d| d.event[0] = -1,
        |d| d.event[0] = 2,
    ];
    let x = predictor();
    for (case, mutate) in mutations.iter().enumerate() {
        let mut data = original.clone();
        mutate(&mut data);
        for options in options() {
            assert!(
                concordancefit(
                    SurvResponse::Counting(&data),
                    x.view(),
                    None,
                    None,
                    None,
                    &options
                )
                .is_err(),
                "case {case}, timefix={}, std_err={}",
                options.timefix,
                options.std_err
            );
        }
    }
    for options in options() {
        assert!(
            concordancefit(
                SurvResponse::Counting(&original),
                x.view(),
                None,
                None,
                None,
                &options
            )
            .is_ok()
        );
    }
}
