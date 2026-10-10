use super::*;
use ndarray::s;

fn assert_same_results(actual: CoxFitResults, expected: CoxFitResults) {
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&actual.coefficients), bits(&expected.coefficients));
    assert_eq!(bits(&actual.means), bits(&expected.means));
    assert_eq!(bits(&actual.score), bits(&expected.score));
    assert_eq!(actual.var.shape(), expected.var.shape());
    assert_eq!(
        bits(actual.var.as_slice().unwrap()),
        bits(expected.var.as_slice().unwrap())
    );
    assert_eq!(bits(&actual.loglik), bits(&expected.loglik));
    assert_eq!(actual.sctest.to_bits(), expected.sctest.to_bits());
    assert_eq!(actual.flag, expected.flag);
    assert_eq!(actual.iter, expected.iter);
    assert_eq!(actual.info, expected.info);
    assert_eq!(actual.order, expected.order);
}

#[test]
fn borrowed_strided_inputs_match_owned_fits_for_every_fitter() {
    // Every input has padding, and the covariate view strides both rows and
    // columns. This checks gathering without assuming contiguous transport.
    let pad = |values: &[f64]| Array1::from_iter(values.iter().flat_map(|&v| [v, f64::NAN]));
    let pad_codes = |values: &[i32]| Array1::from_iter(values.iter().flat_map(|&v| [v, -99]));
    let time = pad(&[4.0, 2.0, 6.0, 3.0, 2.0, 5.0, 1.0, 4.0]);
    let status = pad_codes(&[1, 1, 0, 1, 1, 0, 1, 1]);
    let entry = pad(&[1.0, 0.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0]);
    let strata = pad_codes(&[1, 0, 1, 0, 1, 0, 0, 1]);
    let offset = pad(&[0.1, -0.2, 0.0, 0.3, 0.1, 0.0, -0.1, 0.2]);
    let weights = pad(&[1.0, 1.5, 0.75, 2.0, 1.0, 0.5, 1.25, 1.0]);
    let mut covar = Array2::from_elem((16, 4), f64::NAN);
    covar.slice_mut(s![..;2, ..;2]).assign(
        &Array2::from_shape_vec(
            (8, 2),
            vec![
                0.6, 0.1, 0.5, 0.4, 0.8, 0.7, 1.0, 0.2, 0.3, 0.6, 0.4, 0.5, 0.2, 0.3, 0.9, 0.8,
            ],
        )
        .unwrap(),
    );
    let (time, status, covar, entry, strata, offset, weights) = (
        time.slice(s![..;2]),
        status.slice(s![..;2]),
        covar.slice(s![..;2, ..;2]),
        entry.slice(s![..;2]),
        strata.slice(s![..;2]),
        offset.slice(s![..;2]),
        weights.slice(s![..;2]),
    );

    for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
        for counting in [false, true] {
            for max_iter in [0, 1, 20] {
                let mut borrowed = CoxFitBuilder::new(time, status, covar)
                    .strata(strata)
                    .offset(offset);
                // The builder retains each borrowed view instead of making
                // an intermediate copy, before build owns its sorted data.
                assert!(borrowed.time.is_view());
                assert!(borrowed.status.is_view());
                assert!(borrowed.covar.is_view());
                assert_eq!(borrowed.covar.as_ptr(), covar.as_ptr());
                let mut owned =
                    CoxFitBuilder::new(time.to_owned(), status.to_owned(), covar.to_owned())
                        .strata(strata.to_owned())
                        .offset(offset.to_owned());
                if counting {
                    borrowed = borrowed.entry_times(entry);
                    owned = owned.entry_times(entry.to_owned());
                }
                if method != TieMethod::Exact {
                    borrowed = borrowed.weights(weights);
                    owned = owned.weights(weights.to_owned());
                }
                let configure = |builder: CoxFitBuilder<'_>| {
                    builder
                        .method(method)
                        .max_iter(max_iter)
                        .eps(1e-9)
                        .doscale(vec![true, false])
                        .initial_beta(vec![0.1, -0.15])
                        .build()
                        .unwrap()
                };
                let mut borrowed = configure(borrowed);
                let mut owned = configure(owned);
                assert_eq!(borrowed.data.covar, owned.data.covar);
                borrowed.fit().unwrap();
                owned.fit().unwrap();
                assert_same_results(borrowed.results(), owned.results());
            }
        }
    }
    // Centre/scale applies only to the gathered matrix, preserving inputs.
    assert_eq!(covar[(0, 0)], 0.6);
    assert_eq!(covar[(7, 1)], 0.8);
}

#[test]
fn borrowed_column_major_and_reversed_rows_match_owned_inputs() {
    let time = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0]);
    let status = Array1::from_vec(vec![1, 1, 0, 1, 0, 1]);
    let covar = Array2::from_shape_vec(
        (2, 6),
        vec![0.0, 0.5, 1.0, 0.3, 0.7, 0.2, 0.2, 0.1, 0.8, 0.6, 0.4, 0.9],
    )
    .unwrap()
    .reversed_axes();
    for reverse in [false, true] {
        let (time, status, covar) = if reverse {
            (
                time.slice(s![..;-1]),
                status.slice(s![..;-1]),
                covar.slice(s![..;-1, ..]),
            )
        } else {
            (time.view(), status.view(), covar.view())
        };
        let mut borrowed = CoxFitBuilder::new(time, status, covar)
            .method(TieMethod::Efron)
            .build()
            .unwrap();
        let mut owned = CoxFitBuilder::new(time.to_owned(), status.to_owned(), covar.to_owned())
            .method(TieMethod::Efron)
            .build()
            .unwrap();
        borrowed.fit().unwrap();
        owned.fit().unwrap();
        assert_same_results(borrowed.results(), owned.results());
    }
}

#[test]
fn borrowed_input_lifetime_ends_after_build() {
    let mut fit = {
        let time = [1.0, 2.0, 3.0];
        let status = [1, 0, 1];
        let covar = Array2::from_shape_vec((3, 1), vec![0.2, 0.7, 0.4]).unwrap();
        CoxFitBuilder::new(time.as_slice(), status.as_slice(), covar.view())
            .build()
            .unwrap()
    };
    fit.fit().unwrap();
    assert!(fit.results().loglik[0].is_finite());
}

fn workspace(fit: &CoxFit) -> &EvaluationWorkspace {
    match &fit.fitter {
        Fitter::Coxfit6 { workspace } => workspace,
        Fitter::Agfit4 { workspace, .. } => workspace,
        _ => panic!("Breslow and Efron fit expected"),
    }
}

fn workspace_addresses(fit: &CoxFit) -> [*const f64; 6] {
    let workspace = workspace(fit);
    let running = match &fit.fitter {
        Fitter::Agfit4 { risk_set, .. } => &risk_set.sums,
        _ => &workspace.risk_set,
    };
    [
        workspace.eta.as_ptr(),
        workspace.risk.as_ptr(),
        running.a.as_ptr(),
        running.cmat.as_ptr(),
        workspace.tied.a.as_ptr(),
        workspace.tied.cmat.as_ptr(),
    ]
}

fn reset_fixture(counting: bool, method: TieMethod) -> CoxFit {
    let mut builder = CoxFitBuilder::new(
        Array1::from_vec(vec![3.0, 1.0, 2.0, 2.0, 4.0, 2.0, 1.0, 3.0]),
        Array1::from_vec(vec![1, 1, 1, 1, 0, 1, 0, 1]),
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.5, 0.3, 0.2, 0.4, 1.0, 0.8, 0.7, 0.1, 0.1, 0.5, 0.6, 0.7, 0.3, 0.2, 0.9, 0.6,
            ],
        )
        .unwrap(),
    )
    .strata(Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1]))
    .weights(Array1::from_vec(vec![
        1.0, 2.0, 1.5, 0.5, 0.75, 1.25, 1.0, 1.0,
    ]))
    .method(method);
    if counting {
        builder = builder.entry_times(Array1::from_vec(vec![
            1.0, 0.0, 0.5, 0.0, 3.0, 0.5, 0.0, 2.0,
        ]));
    }
    builder.build().unwrap()
}

#[test]
fn scratch_buffers_are_reused_and_reset_between_trial_coefficients() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let mut fit = reset_fixture(counting, method);
            fit.evaluate(&[0.2, -0.15]).unwrap();
            let addresses = workspace_addresses(&fit);
            for beta in [[0.2, -0.15], [-1.0, 0.5], [0.0, 0.0], [0.2, -0.15]] {
                let loglik = fit.evaluate(&beta).unwrap();
                let mut fresh = reset_fixture(counting, method);
                assert_eq!(loglik.to_bits(), fresh.evaluate(&beta).unwrap().to_bits());
                assert_eq!(fit.u, fresh.u);
                assert_eq!(fit.imat, fresh.imat);
                assert_eq!(workspace_addresses(&fit), addresses);
            }
        }
    }
}

#[test]
fn counting_scratch_resets_when_switching_recentred_and_fast_paths() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        let mut fit = reset_fixture(true, method);
        fit.evaluate(&[0.2, -0.15]).unwrap();
        let addresses = workspace_addresses(&fit);
        for offset in [0.0, -750.0, 0.0, 750.0, 0.0] {
            fit.data.offset.fill(offset);
            let beta = [0.2, -0.15];
            let loglik = fit.evaluate(&beta).unwrap();
            let mut fresh = reset_fixture(true, method);
            fresh.data.offset.fill(offset);
            assert_eq!(loglik.to_bits(), fresh.evaluate(&beta).unwrap().to_bits());
            assert_eq!(fit.u, fresh.u);
            assert_eq!(fit.imat, fresh.imat);
            assert_eq!(workspace_addresses(&fit), addresses);
        }
    }
}

#[test]
fn rejected_nonfinite_and_overflow_trials_do_not_contaminate_next_evaluation() {
    for method in [TieMethod::Breslow, TieMethod::Efron] {
        for counting in [false, true] {
            let mut builder = CoxFitBuilder::new(
                Array1::from_vec(vec![1.0, 2.0, 3.0]),
                Array1::from_vec(vec![1, 1, 1]),
                Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap(),
            )
            .method(method);
            if counting {
                builder = builder.entry_times(Array1::zeros(3));
            }
            let mut fit = builder.build().unwrap();
            let loglik = fit.evaluate(&[0.2]).unwrap();
            let addresses = workspace_addresses(&fit);
            let score = fit.u.clone();
            let information = fit.imat.clone();
            let rejected = fit.evaluate(&[100_000.0]);
            if counting {
                assert_eq!(
                    rejected.unwrap_err().to_string(),
                    "exp overflow due to covariates"
                );
            } else {
                assert!(!rejected.unwrap().is_finite());
            }
            assert_eq!(fit.evaluate(&[0.2]).unwrap().to_bits(), loglik.to_bits());
            assert_eq!(fit.u, score);
            assert_eq!(fit.imat, information);
            assert_eq!(workspace_addresses(&fit), addresses);
        }
    }
}
