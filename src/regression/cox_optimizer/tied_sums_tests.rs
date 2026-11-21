//! Pins the Breslow and Efron evaluations, which accumulate the deaths'
//! risk-weighted sums only where an Efron step reads them, to the sweeps that
//! accumulated them for every death and added R's information one Efron step
//! at a time. Log-likelihoods, scores and Breslow information match bit for
//! bit. Efron tie sets collapse their information steps, so each entry must
//! match within `1e-12 * sqrt(s_ii s_kk)` for the diagonal second-moment
//! magnitudes `s` the reference sums, which bound every term either sweep
//! adds to it.

use super::*;
use crate::internal::rng::Rng;

fn reference_death_time_update(
    efron: bool,
    risk_set: &RiskSetSums,
    tied: &RiskSetSums,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    scale: &mut [f64],
) -> f64 {
    let nvar = u.len();
    let ndead = tied.count;
    let steps = if efron { ndead } else { 1 };
    let wtave = tied.weight / steps as f64;
    let (cmat, cmat2) = (&risk_set.cmat, &tied.cmat);
    let imat = imat.as_slice_mut().expect("standard layout");
    let mut loglik = 0.0;
    for j in 0..steps {
        let fraction = j as f64 / ndead as f64;
        let sum = |all: f64, deaths: f64| if j == 0 { all } else { all - fraction * deaths };
        let denom = sum(risk_set.denom, tied.denom);
        loglik -= wtave * denom.ln();
        for i in 0..nvar {
            let mean = sum(risk_set.a[i], tied.a[i]) / denom;
            u[i] -= wtave * mean;
            for k in 0..=i {
                let second = sum(cmat[i * nvar + k], cmat2[i * nvar + k]);
                let first = sum(risk_set.a[k], tied.a[k]);
                imat[k * nvar + i] += wtave * ((second - mean * first) / denom);
            }
            // Non-finite tie sets take the per-step path in both sweeps.
            let magnitude = (wtave * sum(cmat[i * nvar + i], cmat2[i * nvar + i]) / denom).abs();
            if magnitude.is_finite() {
                scale[i] += magnitude;
            }
        }
    }
    loglik
}

fn reference_coxfit6(
    data: &CoxData,
    beta: &[f64],
    u: &mut [f64],
    imat: &mut Array2<f64>,
    scale: &mut [f64],
) -> f64 {
    if data.needs_paired_predictors(beta) {
        return data.coxfit6_log(beta, u, imat);
    }
    let row = data.rows();
    let eta = data.linear_predictors(beta);
    let mut risk_set = RiskSetSums::zeros(u.len(), true);
    let mut tied = RiskSetSums::zeros(u.len(), true);
    let mut loglik = 0.0;
    let mut person = data.time.len();
    while person > 0 {
        if data.stratum_end[person - 1] == 1 {
            risk_set.clear();
        }
        let time = data.time[person - 1];
        loop {
            person -= 1;
            let (weight, x) = (data.weights[person], row(person));
            let risk = eta[person].exp() * weight;
            if !(1e-100..=1e100).contains(&risk) {
                u.fill(0.0);
                imat.fill(0.0);
                return data.coxfit6_log(beta, u, imat);
            }
            risk_set.add(weight, risk, x);
            if data.status[person] != 0 {
                tied.add(weight, risk, x);
                loglik += weight * eta[person];
                add_scaled(u, weight, x);
            }
            if person == 0 || data.stratum_end[person - 1] == 1 || data.time[person - 1] != time {
                break;
            }
        }
        if tied.count > 0 {
            loglik += reference_death_time_update(data.efron, &risk_set, &tied, u, imat, scale);
            tied.clear();
        }
    }
    loglik
}

fn reference_agfit4_walk(
    data: &CoxData,
    walk: &AgWalk,
    eta: &[f64],
    mut risk_set: impl AgRiskSet,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    scale: &mut [f64],
) -> SurvivalResult<f64> {
    let row = data.rows();
    let mut tied = RiskSetSums::zeros(u.len(), true);
    let mut loglik = 0.0;
    for &(start, end) in &walk.bounds {
        risk_set.clear();
        let (by_stop, by_entry) = (&walk.by_stop[start..end], &walk.by_entry[start..end]);
        let (mut joined, mut left) = (0, 0);
        while let Some(next) = by_stop[joined..].iter().position(|j| j.death) {
            let first_death = joined + next;
            let time = by_stop[first_death].time;
            while left < by_entry.len() && by_entry[left].0 >= time {
                let person = by_entry[left].1;
                risk_set.remove(person, row(person));
                left += 1;
            }
            while joined < by_stop.len() && by_stop[joined].time >= time {
                let person = by_stop[joined].row;
                risk_set.add(person, row(person))?;
                joined += 1;
            }
            tied.clear();
            for joining in by_stop[first_death..joined].iter().filter(|j| j.death) {
                let person = joining.row;
                let (weight, x) = (data.weights[person], row(person));
                tied.add(weight, risk_set.risk(person), x);
                loglik += weight * (eta[person] - risk_set.recenter());
                add_scaled(u, weight, x);
            }
            loglik +=
                reference_death_time_update(data.efron, risk_set.sums(), &tied, u, imat, scale);
        }
    }
    Ok(loglik)
}

fn reference_agfit4(
    data: &CoxData,
    beta: &[f64],
    walk: &AgWalk,
    u: &mut [f64],
    imat: &mut Array2<f64>,
    scale: &mut [f64],
) -> SurvivalResult<f64> {
    if data.needs_paired_predictors(beta) {
        return data.agfit4_scales(beta, walk, u, imat);
    }
    let eta = data.linear_predictors(beta);
    let mut risk_set = RecenteredRiskSet::new(u.len());
    if eta.iter().all(|eta| eta.abs() < 199.0) {
        let risk: Vec<f64> = eta
            .iter()
            .zip(data.weights.iter())
            .map(|(eta, weight)| eta.exp() * weight)
            .collect();
        let centred = CentredAtZero {
            sums: &mut risk_set.sums,
            risk: &risk,
            weights: &data.weights,
        };
        reference_agfit4_walk(data, walk, &eta, centred, u, imat, scale)
    } else {
        let recentred = Recentred {
            set: &mut risk_set,
            eta: &eta,
            weights: &data.weights,
        };
        reference_agfit4_walk(data, walk, &eta, recentred, u, imat, scale)
    }
}

struct Reference {
    loglik: SurvivalResult<f64>,
    u: Vec<f64>,
    imat: Array2<f64>,
    scale: Vec<f64>,
}

fn reference_evaluate(fit: &CoxFit, beta: &[f64]) -> Reference {
    let nvar = beta.len();
    let (mut u, mut imat) = (vec![0.0; nvar], Array2::zeros((nvar, nvar)));
    let mut scale = vec![0.0; nvar];
    let loglik = match &fit.fitter {
        Fitter::Coxfit6 { .. } => Ok(reference_coxfit6(
            &fit.data, beta, &mut u, &mut imat, &mut scale,
        )),
        Fitter::Agfit4 { walk, .. } => {
            reference_agfit4(&fit.data, beta, walk, &mut u, &mut imat, &mut scale)
        }
        _ => unreachable!("Breslow and Efron fits only"),
    };
    Reference {
        loglik,
        u,
        imat,
        scale,
    }
}

/// The bits of each value, with every NaN as `f64::NAN`. Rust leaves the
/// sign and payload of a NaN that arithmetic produces to the target (x86
/// gives `inf - inf` the sign bit, AArch64 does not), so only NaN-ness is
/// comparable across platforms.
fn bits(values: &[f64]) -> Vec<u64> {
    values
        .iter()
        .map(|value| if value.is_nan() { f64::NAN } else { *value }.to_bits())
        .collect()
}

/// Efron information within `1e-12 * sqrt(s_ii s_kk)` of R's per-step sums,
/// and Breslow information bit for bit; non-finite entries must agree exactly.
fn assert_information(fit: &CoxFit, reference: &Reference, label: &str) {
    let nvar = reference.u.len();
    let actual = fit.imat.as_slice().unwrap();
    let expected = reference.imat.as_slice().unwrap();
    if !fit.data.efron {
        assert_eq!(bits(actual), bits(expected), "{label}: information");
        return;
    }
    for row in 0..nvar {
        for col in 0..nvar {
            let (a, e) = (actual[row * nvar + col], expected[row * nvar + col]);
            if bits(&[a]) == bits(&[e]) {
                continue;
            }
            if !(a.is_finite() && e.is_finite()) {
                assert_eq!(bits(&[a]), bits(&[e]), "{label}: information[{row}, {col}]");
                continue;
            }
            let tolerance = 1e-12 * (reference.scale[row] * reference.scale[col]).sqrt();
            assert!(
                (a - e).abs() <= tolerance,
                "{label}: information[{row}, {col}] {a} vs {e} (tolerance {tolerance})"
            );
        }
    }
}

fn assert_same_evaluation(fit: &mut CoxFit, beta: &[f64], label: &str) {
    let reference = reference_evaluate(fit, beta);
    let evaluated = fit.evaluate(beta);
    match (evaluated, &reference.loglik) {
        (Ok(actual), Ok(expected)) => {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{label}: loglik");
            assert_eq!(bits(&fit.u), bits(&reference.u), "{label}: score");
            assert_information(fit, &reference, label);
        }
        (Err(actual), Err(expected)) => {
            assert_eq!(actual.to_string(), expected.to_string(), "{label}");
        }
        (actual, expected) => panic!("{label}: {actual:?} vs {expected:?}"),
    }
}

struct Case {
    time: Vec<f64>,
    entry: Vec<f64>,
    status: Vec<i32>,
    strata: Vec<i32>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    covar: Array2<f64>,
}

/// Zero case weights send a right-censored sweep to its log-space fallback,
/// so only some cases carry them.
fn random_case(rng: &mut Rng, nvar: usize, zero_weights: bool) -> Case {
    let n = rng.usize(1..70);
    let distinct = [2, 5, 40, 1000][rng.usize(0..4)];
    let time: Vec<f64> = (0..n)
        .map(|_| 1.0 + rng.usize(0..distinct) as f64 / 8.0)
        .collect();
    let entry = time
        .iter()
        .map(|&t| {
            if rng.usize(0..3) == 0 {
                0.0
            } else {
                (t - 0.125 * (1 + rng.usize(0..12)) as f64).max(0.0)
            }
        })
        .collect();
    let death_rate = [0.0, 0.3, 0.7, 1.0][rng.usize(0..4)];
    let status = (0..n).map(|_| i32::from(rng.f64() < death_rate)).collect();
    let nstrata = rng.usize(1..4);
    let strata = (0..n)
        .map(|_| rng.usize(0..nstrata) as i32 * 3 - 2)
        .collect();
    let weights = (0..n)
        .map(|_| match rng.usize(0..6) {
            0 => 1.0,
            1 if zero_weights => 0.0,
            _ => 0.25 + rng.f64() * 2.0,
        })
        .collect();
    let offset = (0..n).map(|_| rng.f64() - 0.5).collect();
    let covar = Array2::from_shape_fn((n, nvar), |(_, j)| match rng.usize(0..10) {
        0 => 0.0,
        1 => -0.0,
        _ => (rng.f64() - 0.5) * (1.0 + j as f64),
    });
    Case {
        time,
        entry,
        status,
        strata,
        weights,
        offset,
        covar,
    }
}

fn build(case: &Case, method: TieMethod, counting: bool, offset_shift: f64) -> CoxFit {
    let mut builder = CoxFitBuilder::new(
        Array1::from_vec(case.time.clone()),
        Array1::from_vec(case.status.clone()),
        case.covar.clone(),
    )
    .strata(Array1::from_vec(case.strata.clone()))
    .weights(Array1::from_vec(case.weights.clone()))
    .offset(Array1::from_iter(
        case.offset.iter().map(|o| o + offset_shift),
    ))
    .method(method);
    if counting {
        builder = builder.entry_times(Array1::from_vec(case.entry.clone()));
    }
    builder.build().unwrap()
}

#[test]
fn evaluations_match_sweeps_that_sum_every_death() {
    let mut rng = Rng::with_seed(0x5eed_0f7e_d5a5);
    for case_index in 0..300 {
        let nvar = case_index % 4;
        let case = random_case(&mut rng, nvar, case_index % 5 == 4);
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            for counting in [false, true] {
                // The shift moves counting-process fits onto the recentred
                // risk set; right-censored fits are invariant to it.
                for offset_shift in [0.0, -300.0] {
                    let mut fit = build(&case, method, counting, offset_shift);
                    let label =
                        format!("case {case_index} {method:?} counting {counting} {offset_shift}");
                    for scale in [0.0, 0.5, 2.0, 40.0] {
                        let beta: Vec<f64> = (0..nvar).map(|_| (rng.f64() - 0.5) * scale).collect();
                        assert_same_evaluation(&mut fit, &beta, &label);
                    }
                    if fit.fit().is_ok() {
                        let beta = fit.beta.clone();
                        assert_same_evaluation(&mut fit, &beta, &label);
                    }
                }
            }
        }
    }
}

#[test]
fn fully_tied_deaths_and_extreme_risks_match() {
    let time = Array1::from_vec(vec![2.0; 6]);
    let status = Array1::from_vec(vec![1, 1, 0, 1, 1, 1]);
    let covar = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.1, 0.4, -0.3, 0.2, 0.5, -0.1, 0.0, 0.7, -0.6, 0.3, 0.2, -0.4,
        ],
    )
    .unwrap();
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let mut builder =
                CoxFitBuilder::new(time.clone(), status.clone(), covar.clone()).method(method);
            if counting {
                builder = builder.entry_times(Array1::zeros(6));
            }
            let mut fit = builder.build().unwrap();
            for beta in [
                [0.3, -0.2],
                [500.0, -400.0],
                [f64::NAN, 0.0],
                [f64::INFINITY, 1.0],
            ] {
                assert_same_evaluation(&mut fit, &beta, &format!("{method:?} {counting}"));
            }
        }
    }
}

/// Large tie sets in which nearly everyone at risk dies are where collapsing
/// Efron's steps cancels most: the survivors' sums are differences of nearly
/// equal risk-set and death sums. Skewed weights, a dominant subject and
/// covariates far from zero make the moments cancel further.
#[test]
fn large_nearly_complete_tie_sets_keep_their_information() {
    let mut rng = Rng::with_seed(0x000e_7a11_d1e5);
    for (n, survivors, dominant, offset) in [
        (2_000, 0, false, 0.0),
        (2_000, 1, false, 50.0),
        (3_000, 3, true, 0.0),
        (3_000, 40, true, 50.0),
        (600, 2, false, -1e3),
    ] {
        let nvar = 3;
        let time: Vec<f64> = (0..n)
            .map(|i| if i < n - survivors { 5.0 } else { 9.0 })
            .collect();
        let status = vec![1; n];
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if dominant && i == 7 {
                    1e8
                } else {
                    10f64.powf(rng.f64() * 12.0 - 6.0)
                }
            })
            .collect();
        let covar = Array2::from_shape_fn((n, nvar), |(_, j)| {
            offset + (rng.f64() - 0.5) * (1.0 + j as f64)
        });
        for counting in [false, true] {
            let mut builder = CoxFitBuilder::new(
                Array1::from_vec(time.clone()),
                Array1::from_vec(status.clone()),
                covar.clone(),
            )
            .weights(Array1::from_vec(weights.clone()))
            .method(TieMethod::Efron);
            if counting {
                builder = builder.entry_times(Array1::zeros(n));
            }
            let mut fit = builder.build().unwrap();
            let label = format!("n {n} survivors {survivors} dominant {dominant} {offset}");
            for scale in [0.0, 0.3, 2.0] {
                let beta: Vec<f64> = (0..nvar).map(|_| (rng.f64() - 0.5) * scale).collect();
                assert_same_evaluation(&mut fit, &beta, &label);
            }
        }
    }
}
