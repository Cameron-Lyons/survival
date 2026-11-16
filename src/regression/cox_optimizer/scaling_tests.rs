//! Pins the row-major centring and scaling bit for bit to the column-wise
//! sweeps it replaced.

use super::*;
use crate::internal::rng::Rng;
use ndarray::ArrayView1;

fn reference_mean_abs_scale(
    weights: &Array1<f64>,
    column: ArrayView1<'_, f64>,
    rows: impl Iterator<Item = usize>,
) -> f64 {
    let (weight, abs_sum) = rows.fold((0.0, 0.0), |(weight, abs_sum), person| {
        (
            weight + weights[person],
            abs_sum + weights[person] * column[person].abs(),
        )
    });
    if abs_sum > 0.0 { weight / abs_sum } else { 1.0 }
}

fn reference_scale_center(fit: &mut CoxFit, doscale: &[bool]) {
    let CoxData {
        ref mut covar,
        ref weights,
        ref mut predictor_bounds,
        ..
    } = fit.data;
    let fitter = &fit.fitter;
    let nused = covar.nrows();
    let total_weight: f64 = weights.sum();
    for (i, &scale_column) in doscale.iter().enumerate() {
        if !scale_column {
            fit.means[i] = 0.0;
            fit.scale[i] = 1.0;
            predictor_bounds[i] = covar.column(i).iter().map(|x| x.abs()).fold(0.0, f64::max);
            continue;
        }
        let plain_mean = covar.column(i).sum() / nused as f64;
        let (center, mean) = match fitter {
            Fitter::Coxfit6 { .. } => {
                let weighted_mean = (0..nused)
                    .map(|person| weights[person] * covar[(person, i)])
                    .sum::<f64>()
                    / total_weight;
                (weighted_mean, weighted_mean)
            }
            Fitter::Agfit4 { walk, .. } => (
                walk.by_stop
                    .first()
                    .map_or(0.0, |first| covar[(first.row, i)]),
                plain_mean,
            ),
            Fitter::Coxexact | Fitter::Agexact { .. } => (plain_mean, plain_mean),
        };
        covar.column_mut(i).mapv_inplace(|value| value - center);
        fit.means[i] = mean;
        let column = covar.column(i);
        let scale = match fitter {
            Fitter::Coxfit6 { .. } => reference_mean_abs_scale(weights, column, 0..nused),
            Fitter::Agfit4 { walk, .. } => {
                reference_mean_abs_scale(weights, column, walk.by_stop.iter().map(|j| j.row))
            }
            Fitter::Coxexact => {
                let sd = (column.mapv(|v| v * v).sum() / (nused as f64 - 1.0)).sqrt();
                if sd > 0.0 { 1.0 / sd } else { 1.0 }
            }
            Fitter::Agexact { .. } => 1.0,
        };
        let mut bound = 0.0_f64;
        covar.column_mut(i).mapv_inplace(|value| {
            let value = value * scale;
            bound = bound.max(value.abs());
            value
        });
        predictor_bounds[i] = bound;
        fit.scale[i] = scale;
    }
    for (beta, &scale) in fit.beta.iter_mut().zip(&fit.scale) {
        *beta /= scale;
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

struct Case {
    time: Vec<f64>,
    entry: Vec<f64>,
    status: Vec<i32>,
    strata: Vec<i32>,
    weights: Vec<f64>,
    covar: Array2<f64>,
    beta: Vec<f64>,
}

const SPECIAL_VALUES: [f64; 6] = [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300];

fn random_case(rng: &mut Rng, n: usize, nvar: usize) -> Case {
    let distinct = [1, 3, 50][rng.usize(0..3)];
    let time: Vec<f64> = (0..n)
        .map(|_| 1.0 + rng.usize(0..distinct) as f64)
        .collect();
    let entry = time
        .iter()
        .map(|&t| t - 0.5 * (1 + rng.usize(0..4)) as f64)
        .collect();
    let status = (0..n).map(|_| i32::from(rng.bool())).collect();
    let nstrata = rng.usize(1..4);
    let strata = (0..n).map(|_| rng.usize(0..nstrata) as i32).collect();
    let weights = (0..n)
        .map(|_| match rng.usize(0..5) {
            0 => 0.0,
            1 => 1.0,
            _ => rng.f64() * 3.0,
        })
        .collect();
    // Some columns are constant, all negative zeros, or carry non-finite
    // values, besides ordinary draws.
    let kinds: Vec<usize> = (0..nvar).map(|_| rng.usize(0..8)).collect();
    let covar = Array2::from_shape_fn((n, nvar), |(_, j)| match kinds[j] {
        0 => 2.5,
        1 => -0.0,
        2 if rng.usize(0..10) == 0 => SPECIAL_VALUES[rng.usize(0..SPECIAL_VALUES.len())],
        _ => (rng.f64() - 0.3) * 10.0_f64.powi(rng.usize(0..4) as i32 - 1),
    });
    let beta = (0..nvar).map(|_| rng.f64() - 0.5).collect();
    Case {
        time,
        entry,
        status,
        strata,
        weights,
        covar,
        beta,
    }
}

fn build(case: &Case, method: TieMethod, counting: bool, doscale: Vec<bool>) -> CoxFit {
    let mut builder = CoxFitBuilder::new(
        Array1::from_vec(case.time.clone()),
        Array1::from_vec(case.status.clone()),
        case.covar.clone(),
    )
    .strata(Array1::from_vec(case.strata.clone()))
    .method(method)
    .doscale(doscale)
    .initial_beta(case.beta.clone());
    if method != TieMethod::Exact {
        builder = builder.weights(Array1::from_vec(case.weights.clone()));
    }
    if counting {
        builder = builder.entry_times(Array1::from_vec(case.entry.clone()));
    }
    builder.build().unwrap()
}

#[test]
fn row_major_scaling_matches_the_column_sweeps() {
    let mut rng = Rng::with_seed(0x00c0_fe5c_a1e5);
    for case_index in 0..600 {
        let n = [1, 2, 7, 8, 9, 30, 120][case_index % 7];
        let nvar = [0, 1, 2, 3, 5][case_index % 5];
        let case = random_case(&mut rng, n, nvar);
        let doscale: Vec<bool> = (0..nvar).map(|_| rng.usize(0..4) != 0).collect();
        for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
            for counting in [false, true] {
                let mut expected = build(&case, method, counting, vec![false; nvar]);
                reference_scale_center(&mut expected, &doscale);
                let actual = build(&case, method, counting, doscale.clone());
                let label = format!("case {case_index} {method:?} counting {counting}");
                assert_eq!(
                    bits(actual.data.covar.as_slice().unwrap()),
                    bits(expected.data.covar.as_slice().unwrap()),
                    "{label}: covariates"
                );
                assert_eq!(bits(&actual.means), bits(&expected.means), "{label}");
                assert_eq!(bits(&actual.scale), bits(&expected.scale), "{label}");
                assert_eq!(
                    bits(&actual.data.predictor_bounds),
                    bits(&expected.data.predictor_bounds),
                    "{label}: bounds"
                );
                assert_eq!(bits(&actual.beta), bits(&expected.beta), "{label}: beta");
            }
        }
    }
}
