use std::hint::black_box;
use survival::concordance::{ConcordanceOptions, concordancefit};
use survival::core::SurvResponse;
use survival::data_types::SurvivalData;
use survival::regression::{
    CoxPHFit, aareg_fit, agexact_py, cch_borgan_fit, cch_fit, coxph_fit, finegray, survreg,
};
use survival::surv_analysis::{
    self, ResidualType, RmeanOption, SurvfitKMData, SurvfitKMOptions, nelson_aalen, pseudo,
    survmean,
};
use survival::validation::{BrierInput, brier, uno_c_index};

fn generate_survival_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
    let mut time = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    let mut status_i32 = Vec::with_capacity(n);

    for i in 0..n {
        time.push((i as f64 + 1.0) * 0.5 + (i % 7) as f64 * 0.1);
        let s = if i % 3 == 0 { 0.0 } else { 1.0 };
        status.push(s);
        status_i32.push(s as i32);
    }

    (time, status, status_i32)
}

fn generate_group_data(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 2) as i32).collect()
}

fn generate_predictions(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.1 + (i % 8) as f64 * 0.1).collect()
}

fn generate_covariates(n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..p)
                .map(|j| {
                    let centered_i = (i % 17) as f64 - 8.0;
                    let centered_j = (j % 5) as f64 - 2.0;
                    centered_i * 0.03 + centered_j * 0.1 + ((i * (j + 3)) % 11) as f64 * 0.01
                })
                .collect()
        })
        .collect()
}

fn generate_tied_regression_data(n: usize, p: usize) -> (Vec<f64>, Vec<i32>, Vec<Vec<f64>>) {
    let time = (0..n)
        .map(|i| 1.0 + (i % 80) as f64 * 0.25 + (i / 80) as f64 * 0.01)
        .collect();
    let status = (0..n).map(|i| if i % 4 == 0 { 0 } else { 1 }).collect();
    let covariates = generate_covariates(n, p);
    (time, status, covariates)
}

fn generate_case_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.75 + (i % 7) as f64 * 0.1).collect()
}

fn generate_entry_times(time: &[f64]) -> Vec<f64> {
    time.iter()
        .enumerate()
        .map(|(i, &stop)| {
            let scrambled_fraction = (i.wrapping_mul(37).wrapping_add(17) % 101) as f64 + 1.0;
            stop * scrambled_fraction / 103.0
        })
        .collect()
}

fn generate_strata(n: usize, n_strata: usize) -> Vec<i32> {
    (0..n).map(|i| (i % n_strata) as i32).collect()
}

fn fitted_coxph_model(n: usize, p: usize) -> CoxPHFit {
    let (time, status, covariates) = generate_tied_regression_data(n, p);
    coxph_fit(
        time,
        status,
        covariates,
        None,
        None,
        None,
        None,
        "breslow",
        None,
        Some(20),
        None,
        None,
        None,
        None,
        None,
    )
    .expect("benchmark Cox PH fit should converge")
}

mod kaplan_meier {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn survfitkm(bencher: divan::Bencher, n: usize) {
        let (time, _, status) = generate_survival_data(n);
        let data = SurvfitKMData::right_censored(time, status)
            .expect("benchmark survival data should be valid");
        let options = SurvfitKMOptions::default();

        bencher.bench_local(|| surv_analysis::survfitkm(&data, &options));
    }

    /// Fifty curves from one call: the rows are bucketed by stratum once.
    #[divan::bench(args = [1000, 10000, 100000])]
    fn survfitkm_strata(bencher: divan::Bencher, n: usize) {
        let (time, _, status) = generate_survival_data(n);
        let strata = generate_strata(n, 50);
        let data = SurvfitKMData::try_new(None, time, status, None, Some(strata), None, None)
            .expect("benchmark survival data should be valid");
        let options = SurvfitKMOptions::default();

        bencher.bench_local(|| surv_analysis::survfitkm(&data, &options));
    }
}

mod nelson_aalen_bench {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn nelson_aalen_estimator(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);

        bencher.bench_local(|| nelson_aalen(&time, &status_i32, None, 0.95));
    }
}

mod pseudo_bench {
    use super::*;

    fn inputs(n: usize) -> (Vec<f64>, Vec<i32>, Vec<f64>) {
        let (time, _, status) = generate_survival_data(n);
        let eval_times = (0..n).map(|idx| idx as f64 * 0.5 + 0.25).collect();
        (time, status, eval_times)
    }

    fn run(bencher: divan::Bencher, n: usize, type_: &'static str) {
        let (time, status, eval_times) = inputs(n);
        let data = SurvfitKMData::right_censored(time, status)
            .expect("benchmark pseudo-value inputs should be valid");
        let options = SurvfitKMOptions::default();
        let kind = ResidualType::parse(type_).expect("known residual type");
        bencher.bench_local(|| {
            black_box(
                pseudo(&data, &options, &eval_times, kind)
                    .expect("benchmark pseudo-value inputs should be valid"),
            )
        });
    }

    #[divan::bench(args = [100, 500])]
    fn survival_time_grid(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "survival");
    }

    #[divan::bench(args = [100, 500])]
    fn cumulative_hazard_time_grid(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "cumhaz");
    }

    #[divan::bench(args = [100, 500, 1000])]
    fn restricted_mean_time_grid(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "rmst");
    }
}

mod fitted_survfit_residuals {
    use super::*;
    use survival::surv_analysis::survfit_residuals_at_times;

    fn run(bencher: divan::Bencher, n: usize, type_: &'static str) {
        let (time, status, status_i32) = generate_survival_data(n);
        let weights = generate_case_weights(n);
        let curve = compute_survfitkm(
            &time,
            &status,
            &weights,
            None,
            &vec![0; n],
            &KaplanMeierConfig::default(),
        );
        let eval_times: Vec<f64> = [0.1, 0.3, 0.5, 0.7, 0.9]
            .iter()
            .map(|q| q * time[n - 1])
            .collect();
        bencher.bench_local(|| {
            black_box(
                survfit_residuals_at_times(
                    time.clone(),
                    status_i32.clone(),
                    curve.time.clone(),
                    curve.n_risk.clone(),
                    curve.n_event.clone(),
                    curve.estimate.clone(),
                    curve.cumhaz.clone(),
                    eval_times.clone(),
                    type_,
                    None,
                    1,
                )
                .expect("fitted survival residual inputs should be valid"),
            );
        });
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn survival(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "survival");
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn cumulative_hazard(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "cumhaz");
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn restricted_mean(bencher: divan::Bencher, n: usize) {
        run(bencher, n, "rmst");
    }
}

mod aareg_bench {
    use super::*;

    type AaregInputs = (Vec<f64>, Vec<i32>, Vec<Vec<f64>>, Vec<f64>);

    fn inputs(n: usize, p: usize) -> AaregInputs {
        let (stop, status, covariates) = generate_tied_regression_data(n, p);
        let weights = generate_case_weights(n);
        (stop, status, covariates, weights)
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn weighted_risk_sweep(bencher: divan::Bencher, n: usize) {
        let inputs = inputs(n, 4);
        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(stop, status, covariates, weights)| {
                black_box(
                    aareg_fit(
                        stop,
                        status,
                        covariates,
                        None,
                        Some(weights),
                        None,
                        1e-7,
                        Some(12),
                        false,
                        None,
                        "aalen".to_string(),
                        None,
                    )
                    .expect("benchmark Aalen inputs should be full rank"),
                )
            },
        );
    }

    #[divan::bench(args = [100, 500, 1000])]
    fn clustered_influence(bencher: divan::Bencher, n: usize) {
        let inputs = inputs(n, 3);
        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(stop, status, covariates, weights)| {
                let clusters = (0..n).map(|idx| (idx % 50) as i32).collect();
                black_box(
                    aareg_fit(
                        stop,
                        status,
                        covariates,
                        None,
                        Some(weights),
                        Some(clusters),
                        1e-7,
                        Some(9),
                        true,
                        None,
                        "aalen".to_string(),
                        None,
                    )
                    .expect("benchmark Aalen influence inputs should be full rank"),
                )
            },
        );
    }

    #[divan::bench(args = [100, 500, 1000])]
    fn unclustered_influence(bencher: divan::Bencher, n: usize) {
        let inputs = inputs(n, 3);
        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(stop, status, covariates, weights)| {
                black_box(
                    aareg_fit(
                        stop,
                        status,
                        covariates,
                        None,
                        Some(weights),
                        None,
                        1e-7,
                        Some(9),
                        true,
                        None,
                        "aalen".to_string(),
                        None,
                    )
                    .expect("benchmark Aalen influence inputs should be full rank"),
                )
            },
        );
    }
}

mod logrank {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn logrank(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let group = generate_group_data(n);

        bencher.bench_local(|| {
            survival::validation::logrank_test(&time, &status_i32, &group, None, None, 0.0, true)
        });
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn g_rho(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let group = generate_group_data(n);

        bencher.bench_local(|| {
            survival::validation::logrank_test(&time, &status_i32, &group, None, None, 1.0, true)
        });
    }

    /// `survdiff(Surv(time, status) ~ group + strata(s))` with 49 strata.
    #[divan::bench(args = [1000, 10000, 100000])]
    fn survdiff_strata(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let group = generate_group_data(n);
        let strata = generate_strata(n, 49);
        let data =
            surv_analysis::SurvdiffData::try_new(None, time, status_i32, group, Some(strata))
                .expect("benchmark survival data should be valid");

        bencher.bench_local(|| surv_analysis::survdiff(&data, 0.0, true));
    }
}

mod survreg_residuals {
    use super::*;
    use survival::residuals::survreg_residual_matrix;

    #[divan::bench(args = [100, 1000, 10000])]
    fn gaussian_mixed_censoring(bencher: divan::Bencher, n: usize) {
        let time: Vec<f64> = (0..n).map(|i| 1.0 + (i % 41) as f64 * 0.1).collect();
        let time2: Vec<f64> = time.iter().map(|time| time + 0.25).collect();
        let status: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
        let linear_pred = vec![2.5; n];
        bencher
            .with_inputs(|| {
                (
                    time.clone(),
                    time2.clone(),
                    status.clone(),
                    linear_pred.clone(),
                )
            })
            .bench_local_values(|(time, time2, status, linear_pred)| {
                black_box(
                    survreg_residual_matrix(
                        time,
                        status,
                        linear_pred,
                        1.3,
                        "gaussian".to_string(),
                        Some(time2),
                        None,
                    )
                    .expect("benchmark residual inputs should be valid"),
                )
            });
    }
}

mod brier_score {
    use super::*;

    fn brier_inputs(n: usize) -> (Vec<f64>, Vec<i32>, Vec<f64>, Vec<Vec<f64>>) {
        let (time, _, status_i32) = generate_survival_data(n);
        let max_time = time.iter().cloned().fold(0.0_f64, f64::max);
        let times: Vec<f64> = (1..=4).map(|k| max_time * k as f64 / 5.0).collect();
        let predictions = generate_predictions(n);
        let phat: Vec<Vec<f64>> = times
            .iter()
            .enumerate()
            .map(|(k, _)| {
                predictions
                    .iter()
                    .map(|p| p * (k + 1) as f64 / 4.0)
                    .collect()
            })
            .collect();
        (time, status_i32, times, phat)
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn brier_ipcw(bencher: divan::Bencher, n: usize) {
        let (time, status, times, phat) = brier_inputs(n);

        bencher.bench_local(|| {
            brier(&BrierInput {
                time: &time,
                status: &status,
                weights: None,
                times: &times,
                phat: &phat,
                ties: true,
                efron: false,
                timefix: true,
            })
        });
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn brier_ipcw_weighted(bencher: divan::Bencher, n: usize) {
        let (time, status, times, phat) = brier_inputs(n);
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 5) as f64 * 0.1).collect();

        bencher.bench_local(|| {
            brier(&BrierInput {
                time: &time,
                status: &status,
                weights: Some(&weights),
                times: &times,
                phat: &phat,
                ties: true,
                efron: false,
                timefix: true,
            })
        });
    }
}

mod rmst_bench {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn rmst(bencher: divan::Bencher, n: usize) {
        let (time, _, status) = generate_survival_data(n);
        let tau = time.iter().cloned().fold(0.0_f64, f64::max) * 0.8;
        let data = SurvfitKMData::right_censored(time, status)
            .expect("benchmark survival data should be valid");
        let options = SurvfitKMOptions::default();

        bencher.bench_local(|| {
            let km = surv_analysis::survfitkm(&data, &options).expect("valid curve");
            survmean(&km, 1.0, RmeanOption::At(tau))
        });
    }
}

mod concordance_bench {
    use super::*;

    fn risk_column(n: usize, levels: usize) -> ndarray::Array2<f64> {
        ndarray::Array2::from_shape_fn((n, 1), |(i, _)| (i % levels) as f64)
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn concordance(bencher: divan::Bencher, n: usize) {
        let (time, _, status) = generate_survival_data(n);
        let data = SurvivalData::try_new(time, status).unwrap();
        let x = risk_column(n, 10);
        let options = ConcordanceOptions::default();

        bencher.bench_local(|| {
            concordancefit(
                SurvResponse::Right(&data),
                x.view(),
                None,
                None,
                None,
                &options,
            )
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn concordance_tied_events(bencher: divan::Bencher, n: usize) {
        let data = SurvivalData::try_new(vec![1.0; n], vec![1; n]).unwrap();
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.1).collect();
        let x = risk_column(n, 16);
        let options = ConcordanceOptions::default();

        bencher.bench_local(|| {
            concordancefit(
                SurvResponse::Right(&data),
                x.view(),
                Some(&weights),
                None,
                None,
                &options,
            )
        });
    }
}

mod uno_c_index_bench {
    use super::*;

    #[divan::bench(args = [100, 1000, 5000, 20000])]
    fn mixed_censoring_and_tied_risk(bencher: divan::Bencher, n: usize) {
        let time: Vec<f64> = (0..n)
            .map(|idx| 1.0 + (idx % 250) as f64 * 0.25 + (idx / 250) as f64 * 0.01)
            .collect();
        let status: Vec<i32> = (0..n).map(|idx| i32::from(idx % 4 != 0)).collect();
        let risk_score: Vec<f64> = (0..n)
            .map(|idx| ((idx.wrapping_mul(37) + 11) % 64) as f64 / 16.0)
            .collect();
        let inputs = (time, status, risk_score);

        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|(time, status, risk_score)| {
                black_box(
                    uno_c_index(time, status, risk_score, None)
                        .expect("benchmark Uno C-index inputs should be valid"),
                )
            });
    }

    #[divan::bench(args = [100, 1000, 5000, 20000])]
    fn mixed_censoring_and_distinct_risk(bencher: divan::Bencher, n: usize) {
        let time: Vec<f64> = (0..n)
            .map(|idx| 1.0 + (idx % 250) as f64 * 0.25 + (idx / 250) as f64 * 0.01)
            .collect();
        let status: Vec<i32> = (0..n).map(|idx| i32::from(idx % 4 != 0)).collect();
        let risk_score: Vec<f64> = (0..n).map(|idx| idx as f64).collect();
        let inputs = (time, status, risk_score);

        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|(time, status, risk_score)| {
                black_box(
                    uno_c_index(time, status, risk_score, None)
                        .expect("benchmark Uno C-index inputs should be valid"),
                )
            });
    }
}

mod finegray_interval_expansion {
    use super::*;

    #[divan::bench(args = [1000, 5000, 20000])]
    fn sparse_kept_cuts(bencher: divan::Bencher, n: usize) {
        let tstart = vec![0.0; n];
        let tstop: Vec<f64> = (0..n).map(|idx| (idx % (n - 1)) as f64 + 0.5).collect();
        let ctime: Vec<f64> = (0..n).map(|idx| idx as f64 + 1.0).collect();
        let cprob = vec![1.0; n];
        let extend = vec![true; n];
        let mut keep = vec![false; n];
        keep[n - 1] = true;
        let inputs = (tstart, tstop, ctime, cprob, extend, keep);

        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(tstart, tstop, ctime, cprob, extend, keep)| {
                black_box(
                    finegray(tstart, tstop, ctime, cprob, extend, keep)
                        .expect("benchmark Fine-Gray inputs should be valid"),
                )
            },
        );
    }

    #[divan::bench(args = [1000, 5000, 20000])]
    fn dense_cut_output(bencher: divan::Bencher, n: usize) {
        let tstart = vec![0.0];
        let tstop = vec![0.5];
        let ctime: Vec<f64> = (0..n).map(|idx| idx as f64 + 1.0).collect();
        let cprob = vec![1.0; n];
        let extend = vec![true];
        let keep = vec![true; n];
        let inputs = (tstart, tstop, ctime, cprob, extend, keep);

        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(tstart, tstop, ctime, cprob, extend, keep)| {
                black_box(
                    finegray(tstart, tstop, ctime, cprob, extend, keep)
                        .expect("benchmark Fine-Gray inputs should be valid"),
                )
            },
        );
    }
}

mod exact_counting_process_cox {
    use super::*;

    #[divan::bench(args = [1000, 2000, 4000])]
    fn untied_scaling(bencher: divan::Bencher, n: usize) {
        #[cfg(feature = "python")]
        pyo3::Python::initialize();

        let start = vec![0.0; n];
        let stop: Vec<f64> = (1..=n).map(|value| value as f64).collect();
        let event = vec![1; n];
        let x: Vec<Vec<f64>> = (0..n).map(|value| vec![(value % 17) as f64]).collect();
        let inputs = (start, stop, event, x);

        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|(start, stop, event, x)| {
                black_box(
                    agexact_py(
                        start,
                        stop,
                        event,
                        x,
                        None,
                        None,
                        None,
                        Some(0),
                        Some(1e-9),
                        Some(1e-9),
                        None,
                    )
                    .expect("untied exact counting-process benchmark should succeed"),
                )
            });
    }

    #[divan::bench]
    fn tied_24_of_12(bencher: divan::Bencher) {
        #[cfg(feature = "python")]
        pyo3::Python::initialize();

        const N: usize = 24;
        const DEATHS: usize = 12;
        let start = vec![0.0; N];
        let stop = vec![1.0; N];
        let event: Vec<i32> = (0..N).map(|person| i32::from(person < DEATHS)).collect();
        let x: Vec<Vec<f64>> = (0..N).map(|value| vec![value as f64]).collect();
        let inputs = (start, stop, event, x);

        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|(start, stop, event, x)| {
                black_box(
                    agexact_py(
                        start,
                        stop,
                        event,
                        x,
                        None,
                        None,
                        None,
                        Some(0),
                        Some(1e-9),
                        Some(1e-9),
                        None,
                    )
                    .expect("benchmark exact counting-process fit should succeed"),
                )
            });
    }
}

mod cox_regression {
    use super::*;

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_efron(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);

        bencher.bench_local(|| {
            let fit = coxph_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                None,
                None,
                None,
                None,
                "efron",
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark Cox PH Efron fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [1000, 5000, 20000])]
    fn coxph_counting_efron(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let entry_times = generate_entry_times(&time);

        bencher.bench_local(|| {
            let fit = coxph_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                Some(entry_times.clone()),
                None,
                None,
                None,
                "efron",
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark counting-process Cox PH Efron fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_breslow(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);

        bencher.bench_local(|| {
            let fit = coxph_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                None,
                None,
                None,
                None,
                "breslow",
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark Cox PH Breslow fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn weighted_stratified_coxph_efron(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let weights = generate_case_weights(n);
        let strata = generate_strata(n, 3);

        bencher.bench_local(|| {
            let fit = coxph_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                None,
                Some(strata.clone()),
                Some(weights.clone()),
                None,
                "efron",
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark weighted stratified Cox PH fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_expected_events(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let weights = generate_case_weights(n);
        let strata = generate_strata(n, 3);
        let entry_times: Vec<f64> = time.iter().map(|time| (time - 0.5).max(0.0)).collect();
        let fit = coxph_fit(
            time,
            status,
            covariates,
            Some(entry_times),
            Some(strata),
            Some(weights),
            None,
            "efron",
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            None,
            None,
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let expected = fit
                .predict_expected(None, false)
                .expect("benchmark expected event prediction should succeed");
            black_box(expected);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_stratified_survival_curve(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let weights = generate_case_weights(n);
        let strata = generate_strata(n, 3);
        let fit = coxph_fit(
            time,
            status,
            covariates,
            None,
            Some(strata),
            Some(weights),
            None,
            "efron",
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            None,
            None,
            None,
        )
        .expect("benchmark Cox PH fit should converge");
        let rows = generate_covariates(3, 4);
        let newdata = survival::regression::CoxNewData::try_new(
            ndarray::Array2::from_shape_vec((3, 4), rows.into_iter().flatten().collect())
                .expect("rectangular rows"),
            Some(vec![0, 1, 2]),
            None,
            None,
            None,
        )
        .expect("benchmark newdata should be valid");

        bencher.bench_local(|| {
            let curves = fit
                .survfit(
                    Some(&newdata),
                    survival::regression::SurvfitOptions::default(),
                )
                .expect("benchmark stratified survival curve should succeed");
            black_box(curves);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_schoenfeld_residuals(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let weights = generate_case_weights(n);
        let strata = generate_strata(n, 3);
        let entry_times: Vec<f64> = time.iter().map(|time| (time - 0.5).max(0.0)).collect();
        let fit = coxph_fit(
            time,
            status,
            covariates,
            Some(entry_times),
            Some(strata),
            Some(weights),
            None,
            "efron",
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            None,
            None,
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let residuals = fit
                .residuals(
                    survival::regression::ResidualType::Schoenfeld,
                    None,
                    None,
                    None,
                )
                .expect("benchmark Schoenfeld residuals should succeed");
            black_box(residuals);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_counting_score_residuals(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 4);
        let weights = generate_case_weights(n);
        let strata = generate_strata(n, 3);
        let entry_times: Vec<f64> = time.iter().map(|time| (time - 0.5).max(0.0)).collect();
        let fit = coxph_fit(
            time,
            status,
            covariates,
            Some(entry_times),
            Some(strata),
            Some(weights),
            None,
            "efron",
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            None,
            None,
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let residuals = fit
                .residuals(survival::regression::ResidualType::Score, None, None, None)
                .expect("benchmark score residuals should succeed");
            black_box(residuals);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_dfbeta(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let residuals = black_box(&model)
                .residuals(survival::regression::ResidualType::Dfbeta, None, None, None)
                .expect("benchmark dfbeta residuals should succeed");
            black_box(residuals);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_basehaz(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let basehaz = black_box(&model)
                .basehaz(true)
                .expect("benchmark baseline hazard should succeed");
            black_box(basehaz);
        });
    }
}

/// The residual kernels behind `residuals.coxph` on their own: the C ports
/// (`coxmart`, `agmart3`, `coxscore2`, `agscore3`) and `coxscho` on the
/// shared backward sweep, with 1000 distinct times so that ties exercise
/// the Efron branches.
mod cox_residual_kernels {
    use super::*;
    use survival::core::schoenfeld_residuals;
    use survival::data_types::{AndersenGillInput, CountingProcessData, CoxMartInput, Weights};
    use survival::regression::TieMethod;
    use survival::residuals::{agmart, coxmart};
    use survival::scoring::{agscore3, coxscore2};

    struct KernelInputs {
        time: Vec<f64>,
        entry: Vec<f64>,
        status: Vec<i32>,
        covariates: ndarray::Array2<f64>,
        score: Vec<f64>,
        weights: Vec<f64>,
        strata: Vec<i32>,
    }

    fn inputs(n: usize) -> KernelInputs {
        let p = 4;
        let time: Vec<f64> = (0..n)
            .map(|i| ((i.wrapping_mul(7919) % 1000) + 1) as f64)
            .collect();
        let entry: Vec<f64> = time
            .iter()
            .enumerate()
            .map(|(i, &t)| (t * ((i.wrapping_mul(31) % 50) as f64) / 100.0).floor())
            .collect();
        let status: Vec<i32> = (0..n).map(|i| i32::from(i % 5 != 0)).collect();
        let covariates = generate_covariates(n, p);
        let score: Vec<f64> = covariates
            .iter()
            .map(|row| (0.3 * row[0] - 0.2 * row[1]).exp())
            .collect();
        let covariates =
            ndarray::Array2::from_shape_vec((n, p), covariates.into_iter().flatten().collect())
                .expect("rectangular covariates");
        KernelInputs {
            time,
            entry,
            status,
            covariates,
            score,
            weights: generate_case_weights(n),
            strata: generate_strata(n, 3),
        }
    }

    #[divan::bench(args = [1000, 10000, 100000])]
    fn coxmart_efron(bencher: divan::Bencher, n: usize) {
        let data = inputs(n);
        let input = CoxMartInput::try_new(
            SurvivalData::try_new(data.time, data.status).expect("valid data"),
            data.score,
            Some(Weights::try_new(data.weights).expect("valid weights")),
            Some(data.strata),
        )
        .expect("valid input");
        bencher.bench_local(|| coxmart(black_box(&input), TieMethod::Efron));
    }

    #[divan::bench(args = [1000, 10000, 100000])]
    fn agmart_efron(bencher: divan::Bencher, n: usize) {
        let data = inputs(n);
        let input = AndersenGillInput::try_new(
            CountingProcessData::try_new(data.entry, data.time, data.status).expect("valid data"),
            data.score,
            Some(Weights::try_new(data.weights).expect("valid weights")),
            Some(data.strata),
        )
        .expect("valid input");
        bencher.bench_local(|| agmart(black_box(&input), TieMethod::Efron));
    }

    #[divan::bench(args = [1000, 10000, 100000])]
    fn coxscore2_efron(bencher: divan::Bencher, n: usize) {
        let data = inputs(n);
        let survival = SurvivalData::try_new(data.time, data.status).expect("valid data");
        bencher.bench_local(|| {
            coxscore2(
                black_box(&survival),
                data.covariates.view(),
                &data.score,
                Some(&data.weights),
                Some(&data.strata),
                TieMethod::Efron,
            )
        });
    }

    #[divan::bench(args = [1000, 10000, 100000])]
    fn agscore3_efron(bencher: divan::Bencher, n: usize) {
        let data = inputs(n);
        let counting =
            CountingProcessData::try_new(data.entry, data.time, data.status).expect("valid data");
        bencher.bench_local(|| {
            agscore3(
                black_box(&counting),
                data.covariates.view(),
                &data.score,
                Some(&data.weights),
                Some(&data.strata),
                TieMethod::Efron,
            )
        });
    }

    #[divan::bench(args = [1000, 10000, 100000])]
    fn coxscho_counting_efron(bencher: divan::Bencher, n: usize) {
        let data = inputs(n);
        let counting =
            CountingProcessData::try_new(data.entry, data.time, data.status).expect("valid data");
        bencher.bench_local(|| {
            schoenfeld_residuals(
                SurvResponse::Counting(black_box(&counting)),
                data.covariates.view(),
                &data.score,
                Some(&data.weights),
                Some(&data.strata),
                TieMethod::Efron,
            )
        });
    }
}

mod ridge_cox {
    use super::*;
    use survival::regression::{CoxPHFit, coxph_penalized_fit, coxph_ridge_fit};

    #[derive(Clone)]
    struct Inputs {
        time: Vec<f64>,
        status: Vec<i32>,
        covariates: Vec<Vec<f64>>,
        weights: Vec<f64>,
        strata: Vec<i32>,
        penalty: Vec<f64>,
        groups: Vec<Vec<usize>>,
    }

    fn inputs(n: usize, penalized_columns: usize) -> Inputs {
        const P: usize = 8;
        let covariates: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let index = (i + 1) as f64;
                (0..P)
                    .map(|j| (index * (j + 1) as f64 * 0.31).sin() + 0.55 * (index * 0.21).sin())
                    .collect()
            })
            .collect();
        // The same weighted, stratified data are used for each comparison.
        // theta=20 uses R's unweighted sample-variance scaling before fitting.
        let mut penalty = vec![0.0; P];
        for column in (P - penalized_columns)..P {
            let mean = covariates.iter().map(|row| row[column]).sum::<f64>() / n as f64;
            penalty[column] = 20.0
                * covariates
                    .iter()
                    .map(|row| (row[column] - mean).powi(2))
                    .sum::<f64>()
                / (n - 1) as f64;
        }
        let mut groups: Vec<Vec<usize>> = (0..(P - penalized_columns)).map(|i| vec![i]).collect();
        if penalized_columns != 0 {
            groups.push(((P - penalized_columns)..P).collect());
        }
        Inputs {
            time: (0..n)
                .map(|i| 1.0 + ((i * 37) % 201) as f64 / 10.0)
                .collect(),
            status: (0..n).map(|i| i32::from(i % 4 != 0)).collect(),
            covariates,
            weights: generate_case_weights(n),
            strata: generate_strata(n, 3),
            penalty,
            groups,
        }
    }

    fn fit(inputs: Inputs, penalized: bool) -> CoxPHFit {
        if penalized {
            let (fit, diagnostics) = coxph_penalized_fit(
                inputs.time,
                inputs.status,
                inputs.covariates,
                inputs.penalty,
                inputs.groups,
                Some(inputs.strata),
                Some(inputs.weights),
                None,
                None,
                Some(30),
                Some(1e-9),
                Some(1e-11),
                Some("efron"),
                None,
                None,
            )
            .expect("benchmark ridge Cox inputs should be valid");
            black_box(diagnostics);
            fit
        } else {
            coxph_fit(
                inputs.time,
                inputs.status,
                inputs.covariates,
                Some(inputs.strata),
                Some(inputs.weights),
                None,
                None,
                Some(30),
                Some(1e-9),
                Some(1e-11),
                Some("efron"),
                None,
                None,
            )
            .expect("benchmark ordinary Cox inputs should be valid")
        }
    }

    fn run(bencher: divan::Bencher, n: usize, penalized_columns: usize) {
        let inputs = inputs(n, penalized_columns);
        let penalized = penalized_columns != 0;
        let check = fit(inputs.clone(), penalized);
        assert_eq!(
            check.convergence_flag, 8,
            "benchmark Cox fit must converge at full rank"
        );
        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|inputs| {
                black_box(fit(inputs, penalized));
            });
    }

    #[divan::bench(args = [1000, 10000])]
    fn ordinary(bencher: divan::Bencher, n: usize) {
        run(bencher, n, 0);
    }

    #[divan::bench(args = [1000, 10000])]
    fn mixed_ridge(bencher: divan::Bencher, n: usize) {
        run(bencher, n, 4);
    }

    #[divan::bench(args = [1000, 10000])]
    fn grouped_ridge(bencher: divan::Bencher, n: usize) {
        run(bencher, n, 8);
    }

    #[divan::bench(args = [1000, 10000])]
    fn automatic_df_grouped(bencher: divan::Bencher, n: usize) {
        let inputs = inputs(n, 8);
        let fit_selected = |inputs: Inputs| {
            let (fit, diagnostics, selection) = coxph_ridge_fit(
                inputs.time,
                inputs.status,
                inputs.covariates,
                inputs
                    .penalty
                    .into_iter()
                    .map(|penalty| penalty / 20.0)
                    .collect(),
                inputs.groups,
                vec![None],
                vec![Some(4.0)],
                vec![0.1],
                Some(inputs.strata),
                Some(inputs.weights),
                None,
                None,
                Some(30),
                Some(1e-9),
                Some(1e-11),
                Some("efron"),
                None,
                None,
                None,
            )
            .expect("benchmark automatic ridge Cox inputs should be valid");
            (fit, diagnostics, selection)
        };
        let (check, diagnostics, selection) = fit_selected(inputs.clone());
        assert_eq!(check.convergence_flag, 8);
        assert!(selection.done[0]);
        assert!((diagnostics.term_df[0] - 4.0).abs() < 0.1);
        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|inputs| {
                black_box(fit_selected(inputs));
            });
    }
}

mod case_cohort_bench {
    use super::*;

    type CaseCohortData = (Vec<f64>, Vec<i32>, Vec<Vec<f64>>, Vec<i32>, Vec<i64>);

    fn case_cohort_data(n: usize, p: usize) -> CaseCohortData {
        // Use strictly spaced event times so Prentice's entry-delta stays representable,
        // and offset the subcohort pattern from censoring so sampled noncases remain.
        let time = (0..n)
            .map(|idx| (idx as f64 + 1.0) * 0.25)
            .collect::<Vec<_>>();
        let mut status = (0..n)
            .map(|idx| if idx % 4 == 0 { 0 } else { 1 })
            .collect::<Vec<_>>();
        let covariates = generate_covariates(n, p);
        let subcohort = (0..n)
            .map(|idx| i32::from(idx % 5 != 0))
            .collect::<Vec<_>>();
        for idx in 0..n {
            if subcohort[idx] == 0 {
                status[idx] = 1;
            }
        }
        let id = (0..n).map(|idx| idx as i64).collect();
        (time, status, covariates, subcohort, id)
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn prentice(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates, subcohort, id) = case_cohort_data(n, 4);
        bencher.bench_local(|| {
            let fit = cch_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                n * 4,
                None,
                "Prentice",
                false,
            )
            .expect("benchmark Prentice fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn lin_ying_robust(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates, subcohort, id) = case_cohort_data(n, 4);
        bencher.bench_local(|| {
            let fit = cch_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                n * 4,
                None,
                "LinYing",
                true,
            )
            .expect("benchmark Lin-Ying fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn borgan_i(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates, subcohort, id) = case_cohort_data(n, 4);
        let stratum = (0..n).map(|idx| (idx / 4) % 2).collect::<Vec<_>>();
        bencher.bench_local(|| {
            let fit = cch_borgan_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                stratum.clone(),
                vec![n * 2, n * 2],
                None,
                "I.Borgan",
            )
            .expect("benchmark I.Borgan fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn borgan_ii(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates, subcohort, id) = case_cohort_data(n, 4);
        let stratum = (0..n).map(|idx| (idx / 4) % 2).collect::<Vec<_>>();
        bencher.bench_local(|| {
            let fit = cch_borgan_fit(
                time.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                stratum.clone(),
                vec![n * 2, n * 2],
                None,
                "II.Borgan",
            )
            .expect("benchmark II.Borgan fit should converge");
            black_box(fit);
        });
    }
}

mod survreg_bench {
    use super::*;

    fn mixed_censored_probe(bencher: divan::Bencher, n: usize, distribution: &str) {
        let time: Vec<f64> = (0..n)
            .map(|i| 3.0 + ((i % 19) as f64 - 9.0) * 0.2)
            .collect();
        let upper: Vec<f64> = time
            .iter()
            .enumerate()
            .map(|(i, &value)| value + if i % 3 == 0 { 1e-7 } else { 0.2 })
            .collect();
        let status: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![1.0, (i % 17) as f64 * 0.05, (i % 7) as f64 * -0.1])
            .collect();
        bencher.bench_local(|| {
            black_box(
                survreg(
                    time.clone(),
                    status.clone(),
                    rows.clone(),
                    None,
                    None,
                    Some(vec![3.0, 0.0, 0.0, 0.0]),
                    None,
                    Some(distribution),
                    Some(0),
                    None,
                    None,
                    Some(upper.clone()),
                    None,
                    None,
                )
                .expect("mixed-censoring likelihood probe should be finite"),
            );
        });
    }

    #[divan::bench(args = [1000, 9999, 10000, 100000])]
    fn mixed_censored_gaussian(bencher: divan::Bencher, n: usize) {
        mixed_censored_probe(bencher, n, "gaussian");
    }

    #[divan::bench(args = [1000, 9999, 10000, 100000])]
    fn mixed_censored_student_t(bencher: divan::Bencher, n: usize) {
        mixed_censored_probe(bencher, n, "t");
    }

    fn status_as_survreg(status: &[i32]) -> Vec<f64> {
        status.iter().map(|&value| f64::from(value)).collect()
    }

    #[divan::bench(args = [100, 1000, 5000, 10000])]
    fn survreg_weibull(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 3);
        let status = status_as_survreg(&status);

        bencher.bench_local(|| {
            let fit = survreg(
                time.clone(),
                status.clone(),
                covariates.clone(),
                None,
                None,
                None,
                None,
                Some("weibull"),
                Some(30),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark Weibull survreg fit should converge");
            black_box(fit);
        });
    }

    #[divan::bench(args = [100, 1000, 5000, 10000])]
    fn weighted_stratified_survreg_lognormal(bencher: divan::Bencher, n: usize) {
        bench_weighted_stratified_lognormal(bencher, n, false);
    }

    #[divan::bench(args = [100, 1000, 5000, 10000])]
    fn weighted_stratified_survreg_lognormal_duplicate(bencher: divan::Bencher, n: usize) {
        bench_weighted_stratified_lognormal(bencher, n, true);
    }

    fn bench_weighted_stratified_lognormal(bencher: divan::Bencher, n: usize, duplicate: bool) {
        let (time, status, mut covariates) = generate_tied_regression_data(n, 3);
        if duplicate {
            // Keep the same matrix size while making the final location
            // column an exact alias of the preceding column.
            for row in &mut covariates {
                row[2] = row[1];
            }
        }
        let status = status_as_survreg(&status);
        let weights = generate_case_weights(n);
        let strata: Vec<usize> = generate_strata(n, 3)
            .into_iter()
            .map(|value| value as usize)
            .collect();

        bencher.bench_local(|| {
            let fit = survreg(
                time.clone(),
                status.clone(),
                covariates.clone(),
                Some(weights.clone()),
                None,
                None,
                Some(strata.clone()),
                Some("lognormal"),
                Some(30),
                Some(1e-7),
                Some(1e-9),
                None,
                None,
                None,
            )
            .expect("benchmark weighted stratified lognormal survreg fit should converge");
            black_box(fit);
        });
    }
}

mod gaussian_distribution_bench {
    use super::*;
    use survival::regression::survreg_distribution;

    fn run(bencher: divan::Bencher, values: Vec<f64>, kind: &str) {
        let n = values.len();
        let inputs = (
            values,
            vec![0.0; n],
            vec![1.0; n],
            "gaussian".to_string(),
            kind.to_string(),
        );
        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(values, mean, scale, distribution, kind)| {
                black_box(
                    survreg_distribution(values, mean, scale, distribution, kind, None)
                        .expect("benchmark Gaussian distribution inputs should be valid"),
                )
            },
        );
    }

    #[divan::bench(args = [100, 10000])]
    fn central_probabilities(bencher: divan::Bencher, n: usize) {
        let values = (0..n)
            .map(|idx| -4.0 + 8.0 * idx as f64 / (n - 1) as f64)
            .collect();
        run(bencher, values, "distribution");
    }

    #[divan::bench(args = [100, 10000])]
    fn tail_probabilities(bencher: divan::Bencher, n: usize) {
        let tails = [-5.0, -8.0, -9.0, -12.0, -20.0, -30.0, -37.5, -38.0];
        let values = (0..n).map(|idx| tails[idx % tails.len()]).collect();
        run(bencher, values, "distribution");
    }

    #[divan::bench(args = [100, 10000])]
    fn central_quantiles(bencher: divan::Bencher, n: usize) {
        let values = (0..n)
            .map(|idx| 0.001 + 0.998 * idx as f64 / (n - 1) as f64)
            .collect();
        run(bencher, values, "quantile");
    }

    #[divan::bench(args = [100, 10000])]
    fn tail_quantiles(bencher: divan::Bencher, n: usize) {
        let tails = [
            f64::from_bits(1),
            1e-320,
            1e-300,
            1e-200,
            1e-100,
            1e-20,
            1e-10,
            1.0_f64.next_down(),
        ];
        let values = (0..n).map(|idx| tails[idx % tails.len()]).collect();
        run(bencher, values, "quantile");
    }
}

mod simd_bench {
    use survival::simd_ops::{dot_product_simd, sum_simd, variance_simd};

    fn generate_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.1 + 0.5).collect()
    }

    fn sum_scalar(values: &[f64]) -> f64 {
        values.iter().sum()
    }

    fn dot_product_scalar(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn variance_scalar(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn sum_scalar_bench(bencher: divan::Bencher, n: usize) {
        let data = generate_data(n);
        bencher.bench_local(|| sum_scalar(&data));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn sum_simd_bench(bencher: divan::Bencher, n: usize) {
        let data = generate_data(n);
        bencher.bench_local(|| sum_simd(&data));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn dot_product_scalar_bench(bencher: divan::Bencher, n: usize) {
        let a = generate_data(n);
        let b = generate_data(n);
        bencher.bench_local(|| dot_product_scalar(&a, &b));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn dot_product_simd_bench(bencher: divan::Bencher, n: usize) {
        let a = generate_data(n);
        let b = generate_data(n);
        bencher.bench_local(|| dot_product_simd(&a, &b));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn variance_scalar_bench(bencher: divan::Bencher, n: usize) {
        let data = generate_data(n);
        bencher.bench_local(|| variance_scalar(&data));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn variance_simd_bench(bencher: divan::Bencher, n: usize) {
        let data = generate_data(n);
        bencher.bench_local(|| variance_simd(&data));
    }
}

mod timeline_range_bench {
    use super::*;
    use survival::data_prep::totimeline;

    #[divan::bench(args = [1_000, 10_000, 100_000])]
    fn interval_projection(bencher: divan::Bencher, n: usize) {
        let subjects = 100;
        let id: Vec<i64> = (0..n).map(|row| (row / subjects) as i64).collect();
        let time1: Vec<f64> = (0..n).map(|row| (row % subjects) as f64).collect();
        let time2: Vec<f64> = time1.iter().map(|time| time + 1.0).collect();
        let status: Vec<i32> = (0..n).map(|row| (row % 4) as i32).collect();
        let istate = vec![1; n];

        bencher.bench_local(|| {
            black_box(
                totimeline(&id, &time1, &time2, &status, &istate)
                    .expect("benchmark timeline intervals should be valid"),
            )
        });
    }
}

mod tmerge_bench {
    use super::*;
    use survival::data_prep::{tmerge_cumulative, tmerge_lookup};

    #[divan::bench(args = [1_000, 10_000, 100_000])]
    fn last_value_sweep(bencher: divan::Bencher, n: usize) {
        let id: Vec<usize> = (0..n).map(|row| row / 10).collect();
        let time: Vec<f64> = (0..n).map(|row| (row % 10) as f64).collect();
        let update_id = id.clone();
        let update_time: Vec<f64> = time.iter().map(|value| value - 0.5).collect();

        bencher.bench_local(|| {
            black_box(
                tmerge_lookup(&id, &time, &update_id, &update_time)
                    .expect("benchmark tmerge inputs should be valid"),
            )
        });
    }

    #[divan::bench(args = [1_000, 10_000, 100_000])]
    fn cumulative_sweep(bencher: divan::Bencher, n: usize) {
        let id: Vec<usize> = (0..n).map(|row| row / 10).collect();
        let time: Vec<f64> = (0..n).map(|row| (row % 10) as f64).collect();
        let initial = vec![f64::NAN; n];
        let update_id = id.clone();
        let update_time: Vec<f64> = time.iter().map(|value| value - 0.5).collect();
        let increment = vec![1.0; n];

        bencher.bench_local(|| {
            black_box(
                tmerge_cumulative(&id, &time, &initial, &update_id, &update_time, &increment)
                    .expect("benchmark tmerge inputs should be valid"),
            )
        });
    }
}

mod surv2counting_bench {
    use super::*;
    use survival::data_prep::{Repeated, surv2counting};

    #[divan::bench(args = [1_000, 10_000, 100_000])]
    fn timeline_to_counting(bencher: divan::Bencher, n: usize) {
        let id: Vec<i64> = (0..n).map(|row| (row / 10) as i64).collect();
        let time: Vec<f64> = (0..n).map(|row| (9 - row % 10) as f64).collect();
        let status: Vec<Option<i32>> = (0..n)
            .map(|row| {
                Some(if row % 10 == 9 {
                    1
                } else {
                    (row % 3 + 1) as i32
                })
            })
            .collect();

        bencher.bench_local(|| {
            black_box(
                surv2counting(&id, &time, &status, true, Repeated::No, &[])
                    .expect("benchmark timeline data should be valid"),
            )
        });
    }
}

mod rttright_counting_bench {
    use super::*;
    use survival::data_prep::rttright_counting;

    #[divan::bench(args = [1_000, 10_000, 100_000])]
    fn counting_time_matrix(bencher: divan::Bencher, n: usize) {
        let subjects = n / 2;
        let id: Vec<i64> = (0..subjects)
            .flat_map(|subject| [subject as i64; 2])
            .collect();
        let start: Vec<f64> = (0..subjects)
            .flat_map(|subject| [0.0, 1.0 + (subject % 7) as f64 * 0.01])
            .collect();
        let stop: Vec<f64> = (0..subjects)
            .flat_map(|subject| {
                [
                    1.0 + (subject % 7) as f64 * 0.01,
                    3.0 + subject as f64 * 0.001,
                ]
            })
            .collect();
        let status: Vec<i32> = (0..subjects)
            .flat_map(|subject| [0, i32::from(subject % 3 != 0)])
            .collect();
        let times: Vec<f64> = (0..16).map(|index| 0.5 + index as f64 * 0.25).collect();

        bencher.bench_local(|| {
            black_box(
                rttright_counting(
                    start.clone(),
                    stop.clone(),
                    status.clone(),
                    id.clone(),
                    Some(times.clone()),
                    None,
                    None,
                    true,
                    true,
                )
                .expect("benchmark counting-process histories should be valid"),
            )
        });
    }
}

mod student_t_normal_limit_bench {
    use super::black_box;
    use survival::regression::survreg_distribution;

    fn distribution(bencher: divan::Bencher, df: f64, kind: &str) {
        let values: Vec<f64> = (0..1024)
            .map(|i| {
                let fraction = i as f64 / 1023.0;
                if kind == "quantile" {
                    0.001 + 0.998 * fraction
                } else {
                    -8.5 + 17.0 * fraction
                }
            })
            .collect();
        let mean = vec![0.0; values.len()];
        let scale = vec![1.0; values.len()];
        bencher.bench_local(|| {
            black_box(
                survreg_distribution(
                    values.clone(),
                    mean.clone(),
                    scale.clone(),
                    "t".to_string(),
                    kind.to_string(),
                    Some(df),
                )
                .expect("ordinary Student distribution values should be valid"),
            )
        });
    }

    #[divan::bench(args = [4.5, 999.0, 1000.0, 3000.0, 9999.0, 10000.0, 30000.0, 100000.0, 1000000.0])]
    fn cdf_1024(bencher: divan::Bencher, df: f64) {
        distribution(bencher, df, "distribution");
    }

    #[divan::bench(args = [4.5, 999.0, 1000.0, 3000.0, 9999.0, 10000.0, 30000.0, 100000.0, 1000000.0])]
    fn quantile_1024(bencher: divan::Bencher, df: f64) {
        distribution(bencher, df, "quantile");
    }
}

fn main() {
    divan::main();
}
