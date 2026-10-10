//! Public population kernels recheck mutable rate tables before indexed work.

use ndarray::{Array2, array};
use std::panic::{AssertUnwindSafe, catch_unwind};
use survival::SurvivalResult;
use survival::population::{
    DimType, PyearsCategories, PyearsExpect, PyearsFollowup, PyearsRatetable, RateTable,
    RatetableColumn, SurvexpInput, align_us_year_axis, match_ratetable, pyears, survexp,
    survexp_fit, survexp_mn_table, survexp_us_table, survexp_usr_table,
};

fn table() -> RateTable {
    RateTable::try_new(
        vec![2, 2],
        vec!["age".into(), "sex".into()],
        vec![
            vec!["young".into(), "old".into()],
            vec!["male".into(), "female".into()],
        ],
        vec![Some(vec![0.0, 10.0]), None],
        vec![DimType::Continuous, DimType::Factor],
        vec![0.1, 0.2, 0.3, 0.4],
    )
    .unwrap()
}

fn no_categories(n: usize) -> PyearsCategories {
    PyearsCategories {
        factors: vec![],
        dims: vec![],
        cuts: vec![],
        data: Array2::zeros((n, 0)),
    }
}

fn invalid<T>(label: &str, operation: impl FnOnce() -> SurvivalResult<T>) {
    let result = catch_unwind(AssertUnwindSafe(operation));
    assert!(result.is_ok(), "{label} panicked");
    assert!(result.unwrap().is_err(), "{label} accepted invalid input");
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs() * 1e-12 + 1e-300,
        "{actual} != {expected}"
    );
}

#[test]
fn population_boundaries_reject_tables_mutated_after_construction() {
    let mutations: [fn(&mut RateTable); 25] = [
        |t| t.dims.clear(),
        |t| t.dims.push(1),
        |t| t.dims[0] = 0,
        |t| t.dims = vec![usize::MAX, 2],
        |t| t.dimid.clear(),
        |t| t.dimid.push("extra".into()),
        |t| t.dimid[0].clear(),
        |t| t.dimnames.clear(),
        |t| {
            t.dimnames[0].pop();
        },
        |t| t.cutpoints.clear(),
        |t| {
            t.cutpoints[0].as_mut().unwrap().pop();
        },
        |t| t.cutpoints[0] = None,
        |t| t.cutpoints[0].as_mut().unwrap()[0] = f64::NAN,
        |t| t.cutpoints[0].as_mut().unwrap()[0] = f64::INFINITY,
        |t| t.cutpoints[0].as_mut().unwrap()[1] = -1.0,
        |t| t.cutpoints[1] = Some(vec![]),
        |t| t.types.clear(),
        |t| t.types.push(DimType::Factor),
        |t| t.types = vec![DimType::UsYear, DimType::UsYear],
        |t| t.rates.clear(),
        |t| t.rates.push(0.5),
        |t| t.rates[0] = f64::NAN,
        |t| t.rates[0] = f64::INFINITY,
        |t| t.rates[0] = -0.1,
        |t| {
            t.types[0] = DimType::Date;
            t.cutpoints[0].as_mut().unwrap()[1] = f64::MAX;
        },
    ];
    let positions = array![[2.0, 1.0]];
    let names = vec!["age".into(), "sex".into()];
    let columns = vec![
        RatetableColumn::Numeric(vec![2.0]),
        RatetableColumn::Labels(vec!["male".into()]),
    ];
    let followup = PyearsFollowup {
        start: None,
        stop: vec![2.0],
        event: Some(vec![1.0]),
    };
    for (case, mutate) in mutations.iter().enumerate() {
        let mut changed = table();
        mutate(&mut changed);
        let label = format!("mutation {case}");
        // 25 mutations through eight checked entry points: 200 invalid calls.
        invalid(&label, || changed.validate());
        invalid(&label, || changed.validate_positions(&positions));
        invalid(&label, || changed.match_levels(1, &["male".into()]));
        invalid(&label, || match_ratetable(&changed, &names, &columns));
        invalid(&label, || {
            align_us_year_axis(&changed, &mut positions.clone())
        });
        invalid(&label, || {
            survexp_fit(&[0], &positions, None, &[2.0], false, &changed)
        });
        invalid(&label, || {
            survexp(
                &changed,
                SurvexpInput {
                    positions: &positions,
                    y: None,
                    group: None,
                    times: Some(&[2.0]),
                    method: None,
                    cohort: true,
                    conditional: false,
                    scale: 1.0,
                },
            )
        });
        invalid(&label, || {
            pyears(
                &followup,
                None,
                &no_categories(1),
                Some(PyearsRatetable {
                    table: &changed,
                    positions: positions.clone(),
                }),
                PyearsExpect::Event,
                1.0,
            )
        });
        // Reporting must also stay safe when only some attributes remain.
        assert!(catch_unwind(AssertUnwindSafe(|| changed.to_string())).is_ok());
    }
}

#[test]
fn extreme_groups_and_public_rate_or_alignment_indices_stay_bounded() {
    let rt = table();
    let positions = array![[2.0, 1.0]];
    for code in [usize::MAX, usize::MAX - 1, isize::MAX as usize / 8 + 1] {
        invalid("group capacity", || {
            survexp_fit(&[code], &positions, None, &[1.0, 2.0], false, &rt)
        });
        invalid("group capacity through survexp", || {
            survexp(
                &rt,
                SurvexpInput {
                    positions: &positions,
                    y: None,
                    group: Some(&[code]),
                    times: Some(&[1.0, 2.0]),
                    method: None,
                    cohort: true,
                    conditional: false,
                    scale: 1.0,
                },
            )
        });
    }
    let mut oversized = rt.clone();
    oversized.dims = vec![usize::MAX, 2];
    for index in [[0, 0], [usize::MAX - 1, 1]] {
        let result = catch_unwind(AssertUnwindSafe(|| oversized.rate(&index)));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }
    assert_eq!(rt.rate(&[0]), None);
    assert_eq!(rt.rate(&[2, 0]), None);
    for dim in [2, usize::MAX] {
        invalid("factor dimension", || {
            rt.match_levels(dim, &["male".into()])
        });
    }
    for width in 0..3 {
        invalid("US alignment matrix", || {
            align_us_year_axis(survexp_us_table(), &mut Array2::zeros((1, width)))
        });
    }
    let mut missing_age = survexp_us_table().clone();
    missing_age.dimid[0] = "other".into();
    invalid("US alignment names", || {
        align_us_year_axis(&missing_age, &mut array![[0.0, 1.0, 0.0]])
    });
    // A table without a US-year axis performs no position arithmetic.
    align_us_year_axis(&rt, &mut array![[f64::NAN, f64::INFINITY]]).unwrap();
}

#[test]
fn weighted_person_years_and_cohorts_match_independent_hazard_integrals() {
    let rt = table();
    let positions = array![[9.0, 1.0], [0.0, 2.0]];
    // The first subject spends one day at .1 then one at .2; the second
    // spends one day at .3. The cohort loses its second member after day one.
    let first_mean = ((-0.1f64).exp() + (-0.3f64).exp()) / 2.0;
    for conditional in [false, true] {
        let fit = survexp_fit(
            &[0, 0],
            &positions,
            Some(&[2.0, 1.0]),
            &[0.0, 1.0, 2.0],
            conditional,
            &rt,
        )
        .unwrap();
        assert_eq!(fit.n, array![[2], [2], [1]]);
        assert_eq!(fit.surv[[0, 0]], 1.0);
        near(
            fit.surv[[1, 0]],
            if conditional {
                (-0.2f64).exp()
            } else {
                first_mean
            },
        );
        near(
            fit.surv[[2, 0]],
            if conditional {
                (-0.4f64).exp()
            } else {
                first_mean * (-0.2f64).exp()
            },
        );
    }
    for delayed_entry in [false, true] {
        let followup = PyearsFollowup {
            start: delayed_entry.then(|| vec![1.0, 0.0]),
            stop: if delayed_entry {
                vec![3.0, 1.0]
            } else {
                vec![2.0, 1.0]
            },
            event: Some(vec![1.0, 0.0]),
        };
        for method in [PyearsExpect::Event, PyearsExpect::Pyears] {
            let result = pyears(
                &followup,
                Some(&[2.0, 0.5]),
                &no_categories(2),
                Some(PyearsRatetable {
                    table: &rt,
                    positions: positions.clone(),
                }),
                method,
                1.0,
            )
            .unwrap();
            assert_eq!(result.pyears, [4.5]);
            assert_eq!(result.n, [2.0]);
            assert_eq!(result.event, Some(vec![2.0]));
            assert_eq!(result.offtable, 0.0);
            let integrated_survival =
                |hazard: f64, duration: f64| -(-hazard * duration).exp_m1() / hazard;
            let expected = if method == PyearsExpect::Event {
                if delayed_entry { 0.95 } else { 0.75 }
            } else {
                let male = if delayed_entry {
                    integrated_survival(0.2, 2.0)
                } else {
                    integrated_survival(0.1, 1.0) + (-0.1f64).exp() * integrated_survival(0.2, 1.0)
                };
                2.0 * male + 0.5 * integrated_survival(0.3, 1.0)
            };
            near(result.expected.unwrap()[0], expected);
        }
    }
    // The low-level fit has always skipped finite nonpositive follow-up,
    // while the higher-level survexp validates its response separately.
    let skipped =
        survexp_fit(&[0], &array![[0.0, 1.0]], Some(&[-1.0]), &[1.0], false, &rt).unwrap();
    assert_eq!(skipped.n[[0, 0]], 0);
    assert_eq!(skipped.surv[[0, 0]], 0.0);
}

#[test]
#[allow(clippy::excessive_precision)] // Keep the stock R reference decimals verbatim.
fn builtin_sparse_groups_and_date_types_match_stock_r() {
    // survival 3.8-12, survival:::survexp.fit(group=c(1L,3L), x,
    // y=c(365,180), times=c(0,100,365), death=FALSE/TRUE, ratetable).
    // x: ages c(40,60)*365.25, sexes c(1,2), year 2000-01-01;
    // survexp.usr additionally has race c(1,2).
    for (rt, expected) in [
        (
            survexp_us_table(),
            [
                [0.99929022877547202, 0.99779231183227612],
                [0.99741177046224949, 0.99602967101359352],
            ],
        ),
        (
            survexp_usr_table(),
            [
                [0.99935057020557561, 0.99647361486945141],
                [0.99763162025410712, 0.99366146235444175],
            ],
        ),
        (
            survexp_mn_table(),
            [
                [0.99951425047514508, 0.99832885973054064],
                [0.99822815505520424, 0.99699395849015815],
            ],
        ),
    ] {
        rt.validate().unwrap();
        let sex = rt.dimid.iter().position(|name| name == "sex").unwrap();
        assert_eq!(
            rt.match_levels(sex, &["ma".into(), "FEMALE".into()])
                .unwrap(),
            [1, 2]
        );
        let positions = if rt.ndim() == 4 {
            array![[14610.0, 1.0, 1.0, 10957.0], [21915.0, 2.0, 2.0, 10957.0]]
        } else {
            array![[14610.0, 1.0, 10957.0], [21915.0, 2.0, 10957.0]]
        };
        for conditional in [false, true] {
            let fit = survexp_fit(
                &[0, 2],
                &positions,
                Some(&[365.0, 180.0]),
                &[0.0, 100.0, 365.0],
                conditional,
                rt,
            )
            .unwrap();
            assert_eq!(fit.n, array![[1, 0, 1], [1, 0, 1], [1, 0, 1]]);
            assert_eq!(fit.surv[[0, 0]], 1.0);
            assert_eq!(fit.surv[[0, 2]], 1.0);
            for row in 0..3 {
                assert_eq!(fit.surv[[row, 1]], if conditional { 1.0 } else { 0.0 });
            }
            for (row, values) in expected.iter().enumerate() {
                near(fit.surv[[row + 1, 0]], values[0]);
                near(fit.surv[[row + 1, 2]], values[1]);
            }
        }
    }
    let date = RateTable::try_new(
        vec![2],
        vec!["date".into()],
        vec![vec!["early".into(), "late".into()]],
        vec![Some(vec![0.0, 10.0])],
        vec![DimType::Date],
        vec![0.1, 0.2],
    )
    .unwrap();
    let fit = survexp_fit(
        &[0],
        &array![[9.0]],
        Some(&[2.0]),
        &[0.0, 1.0, 2.0],
        false,
        &date,
    )
    .unwrap();
    for (row, expected) in [1.0, 0.90483741803595952, 0.74081822068171788]
        .iter()
        .enumerate()
    {
        near(fit.surv[[row, 0]], *expected);
    }
    let us = RateTable::try_new(
        vec![1, 2],
        vec!["age".into(), "year".into()],
        vec![vec!["all".into()], vec!["1999".into(), "2000".into()]],
        vec![Some(vec![0.0]), Some(vec![10592.0, 10957.0])],
        vec![DimType::Continuous, DimType::UsYear],
        vec![0.1, 0.2],
    )
    .unwrap();
    let mut aligned = array![[14428.0, 10957.0]];
    align_us_year_axis(&us, &mut aligned).unwrap();
    assert_eq!(aligned, array![[14428.0, 10775.0]]);
    let fit = survexp_fit(
        &[0],
        &array![[14428.0, 10957.0]],
        Some(&[365.0]),
        &[0.0, 100.0, 365.0],
        false,
        &us,
    )
    .unwrap();
    for (row, expected) in [1.0, 4.5399929762484854e-5, 1.5873123369578814e-24]
        .iter()
        .enumerate()
    {
        near(fit.surv[[row, 0]], *expected);
    }
}
