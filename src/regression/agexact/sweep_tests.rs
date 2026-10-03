use super::{AgexactOptions, agexact_fit};
use crate::regression::CoxphData;
use ndarray::Array2;
use serde_json::Value;

fn numbers(value: &Value) -> Vec<f64> {
    serde_json::from_value(value.clone()).unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "actual={actual:.16e}, expected={expected:.16e}",
    );
}

fn reference_input(case: &Value) -> CoxphData {
    let rows: Vec<Vec<f64>> = serde_json::from_value(case["x"].clone()).unwrap();
    let n = rows.len();
    let p = rows[0].len();
    CoxphData::try_new(
        numbers(&case["stop"]),
        Some(numbers(&case["start"])),
        serde_json::from_value(case["status"].clone()).unwrap(),
        Array2::from_shape_vec((n, p), rows.into_iter().flatten().collect()).unwrap(),
        Some(vec![1.0; n]),
        Some(serde_json::from_value(case["strata"].clone()).unwrap()),
        Some(numbers(&case["offset"])),
    )
    .unwrap()
}

#[test]
fn counting_sweep_matches_stock_r_fits_and_zero_iteration_information() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../test/r/kernel_references/exact_counting_sweep.json"
    ))
    .unwrap();
    assert_eq!(reference["survival_version"], "3.8.12");
    for case in reference["cases"].as_array().unwrap() {
        for (key, iterations) in [("initial", 0), ("fitted", 50)] {
            let expected = &case[key];
            if expected.is_null() {
                continue;
            }
            let fit = agexact_fit(
                reference_input(case),
                &AgexactOptions {
                    init: Some(numbers(&case["init"])),
                    iter_max: iterations,
                    eps: 1e-9,
                    toler_chol: 1e-10,
                    nocenter: None,
                },
            )
            .unwrap_or_else(|error| panic!("{} {key}: {error}", case["name"]));
            for (actual, expected) in fit
                .coefficients
                .iter()
                .zip(numbers(&expected["coefficients"]))
            {
                close(*actual, expected, 2e-9);
            }
            for (actual, expected) in fit.loglik.iter().zip(numbers(&expected["loglik"])) {
                close(*actual, expected, 2e-10);
            }
            for (actual, expected) in fit.means.iter().zip(numbers(&expected["means"])) {
                close(*actual, expected, 2e-11);
            }
            let variance: Vec<Vec<f64>> =
                serde_json::from_value(expected["variance"].clone()).unwrap();
            for (actual, expected) in fit.var.iter().flatten().zip(variance.iter().flatten()) {
                close(*actual, *expected, 2e-9);
            }
            close(fit.sctest, expected["score"].as_f64().unwrap(), 2e-9);
            assert_eq!(fit.iter, expected["iter"].as_u64().unwrap() as usize);
            if iterations == 0 {
                assert_eq!(fit.flag, 0);
            } else {
                assert_eq!(fit.flag, 2);
            }
        }
    }
}

/// Direct active-set enumeration, with fresh centred moments at every event.
/// It shares no walk, tree, accumulator, matrix inversion or subset DP code.
fn enumerate_untied(data: &CoxphData, beta: &[f64]) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let entry = data.entry.as_ref().unwrap();
    let strata = data.strata.as_ref().unwrap();
    let offset = data.offset.as_ref().unwrap();
    let eta: Vec<f64> = (0..data.n())
        .map(|row| offset[row] + data.x[(row, 0)] * beta[0] + data.x[(row, 1)] * beta[1])
        .collect();
    let mut events: Vec<(i32, f64, usize)> = (0..data.n())
        .filter(|&row| data.status[row] == 1)
        .map(|row| (strata[row], data.time[row], row))
        .collect();
    events.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
    assert!(
        events
            .windows(2)
            .all(|pair| { pair[0].0 != pair[1].0 || pair[0].1 != pair[1].1 })
    );
    let mut likelihood = 0.0;
    let mut score = [0.0; 2];
    let mut information = [[0.0; 2]; 2];
    for (code, time, death) in events {
        let active: Vec<usize> = (0..data.n())
            .filter(|&row| strata[row] == code && entry[row] < time && time <= data.time[row])
            .collect();
        assert!(active.contains(&death));
        let shift = active
            .iter()
            .map(|&row| eta[row])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = active.iter().map(|&row| (eta[row] - shift).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mut mean = [0.0; 2];
        for (&row, &weight) in active.iter().zip(&weights) {
            for (column, value) in mean.iter_mut().enumerate() {
                *value += weight * data.x[(row, column)] / total;
            }
        }
        likelihood += (eta[death] - shift) - total.ln();
        for i in 0..2 {
            score[i] += data.x[(death, i)] - mean[i];
            for j in 0..2 {
                for (&row, &weight) in active.iter().zip(&weights) {
                    information[i][j] += weight / total
                        * (data.x[(row, i)] - mean[i])
                        * (data.x[(row, j)] - mean[j]);
                }
            }
        }
    }
    let determinant = information[0][0] * information[1][1] - information[0][1] * information[1][0];
    assert!(determinant > 0.0);
    let variance = [
        [
            information[1][1] / determinant,
            -information[0][1] / determinant,
        ],
        [
            -information[1][0] / determinant,
            information[0][0] / determinant,
        ],
    ];
    (likelihood, score, variance)
}

fn generated_input(seed: usize, scheme: usize, shift: f64, permutation: usize) -> CoxphData {
    let n = 73;
    let status: Vec<i32> = (0..n)
        .map(|row| i32::from(!(row + seed).is_multiple_of(4)))
        .collect();
    let strata: Vec<i32> = (0..n).map(|row| (row % 3) as i32 - 1).collect();
    let mut stop: Vec<f64> = (0..n).map(|row| row as f64 + 1.0).collect();
    // Censors share an event time in the same stratum. No event ties arise.
    for row in 0..n - 3 {
        if status[row] == 0 && status[row + 3] == 1 {
            stop[row] = stop[row + 3];
        }
    }
    let mut entry: Vec<f64> = (0..n)
        .map(|row| stop[row] * ((row * 37 + seed * 11) % 101 + 1) as f64 / 103.0)
        .collect();
    for row in 0..n {
        if scheme == 2 || (scheme == 1 && strata[row] == -1) {
            entry[row] = -2.0 + (row % 17) as f64 * 0.01;
        }
    }
    let x = Array2::from_shape_fn((n, 2), |(row, column)| {
        if column == 0 {
            ((row * 7 + seed) % 19) as f64 / 7.0 - 1.0
        } else {
            ((row * 11 + seed * 3) % 23) as f64 / 9.0 - 1.5
        }
    });
    let offset: Vec<f64> = (0..n)
        .map(|row| shift + ((row * 17 + seed) % 13) as f64 * 0.03)
        .collect();
    let order: Vec<usize> = (0..n).map(|row| (row * permutation) % n).collect();
    CoxphData::try_new(
        order.iter().map(|&row| stop[row]).collect(),
        Some(order.iter().map(|&row| entry[row]).collect()),
        order.iter().map(|&row| status[row]).collect(),
        Array2::from_shape_fn((n, 2), |(row, column)| x[(order[row], column)]),
        Some(vec![1.0; n]),
        Some(order.iter().map(|&row| strata[row]).collect()),
        Some(order.iter().map(|&row| offset[row]).collect()),
    )
    .unwrap()
}

#[test]
fn general_and_growing_sweeps_match_independent_risk_set_enumeration() {
    let beta = [0.3, -0.2];
    for seed in 0..5 {
        for scheme in 0..3 {
            for shift in [-1_000.0, 0.0, 1_000.0] {
                for permutation in [1, 19] {
                    let data = generated_input(seed, scheme, shift, permutation);
                    let (likelihood, score, variance) = enumerate_untied(&data, &beta);
                    let fit = agexact_fit(
                        data,
                        &AgexactOptions {
                            init: Some(beta.to_vec()),
                            iter_max: 0,
                            ..AgexactOptions::default()
                        },
                    )
                    .unwrap();
                    close(fit.loglik[0], likelihood, 2e-10);
                    for (i, expected) in variance.iter().enumerate() {
                        close(fit.u[i], score[i], 2e-10);
                        for (j, value) in expected.iter().enumerate() {
                            close(fit.var[i][j], *value, 2e-10);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn dominant_risk_removal_matches_fresh_information_and_surviving_rows() {
    for shift in [-1_000.0, 0.0, 1_000.0] {
        // The exp(100) row leaves at time 2. The remaining exp(1), exp(-1)
        // and exp(0) risks must survive its removal, with full-rank covariance.
        let data = CoxphData::try_new(
            vec![1.0, 3.0, 4.0, 4.0],
            Some(vec![0.0, 0.0, 0.0, 2.0]),
            vec![1, 1, 0, 0],
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 100.0, -50.0])
                .unwrap(),
            None,
            Some(vec![7; 4]),
            Some(vec![shift; 4]),
        )
        .unwrap();
        let beta = [1.0, 0.0];
        let (likelihood, score, variance) = enumerate_untied(&data, &beta);
        let fit = agexact_fit(
            data,
            &AgexactOptions {
                init: Some(beta.to_vec()),
                iter_max: 0,
                ..AgexactOptions::default()
            },
        )
        .unwrap();
        close(fit.loglik[0], likelihood, 2e-12);
        for (i, expected) in variance.iter().enumerate() {
            close(fit.u[i], score[i], 2e-12);
            for (j, value) in expected.iter().enumerate() {
                close(fit.var[i][j], *value, 2e-12);
            }
        }
    }
}

#[test]
fn entry_at_a_death_time_is_excluded_while_censors_at_that_time_remain() {
    let data = CoxphData::try_new(
        vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
        Some(vec![0.0, 0.0, 0.0, 2.0, 2.0, 1.0]),
        vec![1, 1, 0, 1, 0, 0],
        Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 2.0, -1.0, -2.0, 2.0,
            ],
        )
        .unwrap(),
        None,
        Some(vec![0; 6]),
        Some(vec![0.0; 6]),
    )
    .unwrap();
    let beta = [0.3, -0.2];
    let (likelihood, score, variance) = enumerate_untied(&data, &beta);
    let fit = agexact_fit(
        data,
        &AgexactOptions {
            init: Some(beta.to_vec()),
            iter_max: 0,
            ..AgexactOptions::default()
        },
    )
    .unwrap();
    close(fit.loglik[0], likelihood, 2e-12);
    for (i, expected) in variance.iter().enumerate() {
        close(fit.u[i], score[i], 2e-12);
        for (j, value) in expected.iter().enumerate() {
            close(fit.var[i][j], *value, 2e-12);
        }
    }
}
