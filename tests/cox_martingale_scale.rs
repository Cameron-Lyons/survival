use survival::prelude::{
    AndersenGillInput, CountingProcessData, CoxMartInput, SurvivalData, Weights,
};
use survival::regression::TieMethod;
use survival::residuals::{agmart, coxmart};

const STOP: [f64; 8] = [2., 2., 3., 5., 2., 3., 3., 5.];
const START: [f64; 8] = [0., 1., 0., 2., 0., 1., 0., 3.];
const EVENT: [i32; 8] = [1, 1, 0, 1, 0, 1, 1, 0];
const SCORE: [f64; 8] = [0.25, 1., 0.5, 0.75, 1., 0.5, 0.75, 0.25];
const WEIGHT: [f64; 8] = [1., 0.5, 2., 0.75, 0.25, 1.5, 1., 2.];
const STRATA: [i32; 8] = [-4, -4, -4, -4, 11, 11, 11, 11];

fn calculate(counting: bool, score: Vec<f64>, weights: Vec<f64>, method: TieMethod) -> Vec<f64> {
    let weights = Some(Weights::try_new(weights).unwrap());
    let strata = Some(STRATA.to_vec());
    if counting {
        agmart(
            &AndersenGillInput::try_new(
                CountingProcessData::try_new(START.to_vec(), STOP.to_vec(), EVENT.to_vec())
                    .unwrap(),
                score,
                weights,
                strata,
            )
            .unwrap(),
            method,
        )
        .unwrap()
    } else {
        coxmart(
            &CoxMartInput::try_new(
                SurvivalData::try_new(STOP.to_vec(), EVENT.to_vec()).unwrap(),
                score,
                weights,
                strata,
            )
            .unwrap(),
            method,
        )
        .unwrap()
    }
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (row, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-12 * (1. + expected.abs()),
            "row {row}: {actual} vs {expected}"
        );
    }
}

fn direct_risk_set_residuals(counting: bool, method: TieMethod) -> Vec<f64> {
    let mut times = STOP.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    (0..STOP.len())
        .map(|row| {
            let mut expected = 0.0;
            for &time in &times {
                if time > STOP[row] || (counting && time <= START[row]) {
                    continue;
                }
                let deaths: Vec<usize> = (0..STOP.len())
                    .filter(|&i| STRATA[i] == STRATA[row] && EVENT[i] == 1 && STOP[i] == time)
                    .collect();
                if deaths.is_empty() {
                    continue;
                }
                let denominator: f64 = (0..STOP.len())
                    .filter(|&i| {
                        STRATA[i] == STRATA[row]
                            && STOP[i] >= time
                            && (!counting || START[i] < time)
                    })
                    .map(|i| SCORE[i] * WEIGHT[i])
                    .sum();
                let death_risk: f64 = deaths.iter().map(|&i| SCORE[i] * WEIGHT[i]).sum();
                let death_weight: f64 = deaths.iter().map(|&i| WEIGHT[i]).sum();
                let steps = if method == TieMethod::Efron {
                    deaths.len()
                } else {
                    1
                };
                for step in 0..steps {
                    let fraction = step as f64 / deaths.len() as f64;
                    let correction = if EVENT[row] == 1 && STOP[row] == time {
                        1.0 - fraction
                    } else {
                        1.0
                    };
                    expected += SCORE[row] * correction * (death_weight / steps as f64)
                        / (denominator - fraction * death_risk);
                }
            }
            f64::from(EVENT[row]) - expected
        })
        .collect()
}

#[test]
fn martingale_residuals_are_invariant_to_extreme_stratum_risk_scales() {
    for counting in [false, true] {
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            let expected = direct_risk_set_residuals(counting, method);
            assert_close(
                &calculate(counting, SCORE.to_vec(), WEIGHT.to_vec(), method),
                &expected,
            );
            for (first, second) in [(1e308, 1e308), (1e-300, 1e-300), (1e308, 1e-300)] {
                let score = SCORE
                    .iter()
                    .enumerate()
                    .map(|(row, score)| score * if row < 4 { first } else { second })
                    .collect();
                assert_close(
                    &calculate(counting, score, WEIGHT.to_vec(), method),
                    &expected,
                );
            }
        }
    }
}

#[test]
fn martingale_residuals_are_invariant_to_extreme_case_weight_scales() {
    for counting in [false, true] {
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            let expected = direct_risk_set_residuals(counting, method);
            for scale in [5e307, 1e-300] {
                let weights = WEIGHT.iter().map(|weight| weight * scale).collect();
                assert_close(
                    &calculate(counting, SCORE.to_vec(), weights, method),
                    &expected,
                );
            }
        }
    }
}

#[test]
fn rescaled_martingale_residuals_match_stock_r_weighted_ties_and_entry() {
    // survival 3.8-11: coxph(Surv(...) ~ offset(log(score)) + strata(group),
    // weights=weight, ties=method, robust=FALSE, control=coxph.control(iter.max=0)).
    let cases = [
        (
            false,
            TieMethod::Breslow,
            [
                0.837837837837838,
                0.351351351351351,
                -0.324324324324324,
                -0.486486486486487,
                0.,
                0.375,
                0.0625,
                -0.3125,
            ],
        ),
        (
            false,
            TieMethod::Efron,
            [
                0.870531822144725,
                0.482127288578902,
                -0.355710549258936,
                -0.533565823888405,
                0.,
                0.4375,
                0.15625,
                -0.40625,
            ],
        ),
        (
            true,
            TieMethod::Breslow,
            [
                0.785714285714286,
                0.142857142857143,
                -0.428571428571428,
                0.,
                0.,
                0.166666666666667,
                -0.25,
                0.,
            ],
        ),
        (
            true,
            TieMethod::Efron,
            [
                0.824675324675325,
                0.298701298701298,
                -0.487012987012987,
                0.,
                0.,
                0.166666666666667,
                -0.25,
                0.,
            ],
        ),
    ];
    for (counting, method, expected) in cases {
        for scale in [1e308, 1e-300] {
            let score = SCORE.iter().map(|risk| risk * scale).collect();
            assert_close(
                &calculate(counting, score, WEIGHT.to_vec(), method),
                &expected,
            );
        }
    }
}

#[test]
fn subnormal_scores_keep_a_finite_expected_event_count() {
    for score in [1e-320, f64::from_bits(1)] {
        let input = CoxMartInput::try_new(
            SurvivalData::try_new(vec![1., 2.], vec![1, 0]).unwrap(),
            vec![score; 2],
            None,
            None,
        )
        .unwrap();
        assert_close(&coxmart(&input, TieMethod::Efron).unwrap(), &[0.5, -0.5]);
    }
}

#[test]
fn zero_risks_and_zero_weight_deaths_keep_other_risk_sets_valid() {
    let survival = SurvivalData::try_new(vec![1., 1., 2., 3.], vec![1, 1, 0, 1]).unwrap();
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let input = CoxMartInput::try_new(
            survival.clone(),
            vec![1., 0., 1., 0.],
            Some(Weights::try_new(vec![1., 0., 1., 0.]).unwrap()),
            Some(vec![1, 1, 1, 2]),
        )
        .unwrap();
        let (event, censored) = if method == TieMethod::Efron {
            (1. - (0.25 + 1. / 6.), -(0.25 + 1. / 3.))
        } else {
            (0.5, -0.5)
        };
        assert_close(
            &coxmart(&input, method).unwrap(),
            &[event, 1., censored, 1.],
        );
    }
}

#[test]
fn disjoint_counting_risk_sets_preserve_tiny_and_huge_risks() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for scores in [
            vec![1e308, 1e308, 1e-300, 1e-300],
            vec![1e-300, 1e-300, 1e308, 1e308],
        ] {
            let input = AndersenGillInput::try_new(
                CountingProcessData::try_new(
                    vec![0., 0., 3., 3.],
                    vec![1., 2., 4., 5.],
                    vec![1, 0, 1, 0],
                )
                .unwrap(),
                scores,
                None,
                None,
            )
            .unwrap();
            assert_close(&agmart(&input, method).unwrap(), &[0.5, -0.5, 0.5, -0.5]);
        }
    }
}

#[test]
fn late_tiny_risk_sets_survive_earlier_huge_scores() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let input = CoxMartInput::try_new(
            SurvivalData::try_new(vec![1., 2., 3.], vec![1, 1, 0]).unwrap(),
            vec![1e308, 1e-300, 1e-300],
            None,
            None,
        )
        .unwrap();
        assert_close(&coxmart(&input, method).unwrap(), &[0., 0.5, -0.5]);
    }
}

#[test]
fn positive_events_with_no_risk_return_errors_but_no_event_strata_are_zero() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for events in [vec![1, 0], vec![0, 0]] {
            let right = CoxMartInput::try_new(
                SurvivalData::try_new(vec![1., 2.], events.clone()).unwrap(),
                vec![0., 0.],
                None,
                None,
            )
            .unwrap();
            let counting = AndersenGillInput::try_new(
                CountingProcessData::try_new(vec![0., 0.], vec![1., 2.], events.clone()).unwrap(),
                vec![0., 0.],
                None,
                None,
            )
            .unwrap();
            for result in [coxmart(&right, method), agmart(&counting, method)] {
                if events[0] == 1 {
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("positive risk-set sum")
                    );
                } else {
                    assert_eq!(result.unwrap(), vec![0., 0.]);
                }
            }
        }
    }
}

#[test]
fn raw_martingale_kernels_reject_negative_risks() {
    let right = CoxMartInput::try_new(
        SurvivalData::try_new(vec![1., 2.], vec![1, 0]).unwrap(),
        vec![1., -1.],
        None,
        None,
    )
    .unwrap();
    let counting = AndersenGillInput::try_new(
        CountingProcessData::try_new(vec![0., 0.], vec![1., 2.], vec![1, 0]).unwrap(),
        vec![1., -1.],
        None,
        None,
    )
    .unwrap();
    assert!(coxmart(&right, TieMethod::Efron).is_err());
    assert!(agmart(&counting, TieMethod::Efron).is_err());
}
