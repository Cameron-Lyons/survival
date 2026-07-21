use std::hint::black_box;
use survival::regression::{CoxPHModel, agexact, coxph_fit, finegray, survreg};
use survival::{
    KaplanMeierConfig, WeightType, compute_brier, compute_rmst, compute_survfitkm, concordance1,
    nelson_aalen, uno_c_index, weighted_logrank_test,
};

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

fn fitted_coxph_model(n: usize, p: usize) -> CoxPHModel {
    let (time, status, covariates) = generate_tied_regression_data(n, p);
    let status: Vec<u8> = status.into_iter().map(|value| value as u8).collect();
    let mut model = CoxPHModel::new_with_data(covariates, time, status)
        .expect("benchmark CoxPHModel data should be valid");
    model
        .fit(20)
        .expect("benchmark CoxPHModel fit should converge");
    model
}

mod kaplan_meier {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn survfitkm(bencher: divan::Bencher, n: usize) {
        let (time, status, _) = generate_survival_data(n);
        let weights: Vec<f64> = vec![1.0; n];
        let position: Vec<i32> = vec![0; n];
        let config = KaplanMeierConfig::default();

        bencher
            .bench_local(|| compute_survfitkm(&time, &status, &weights, None, &position, &config));
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

mod logrank {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn logrank_test(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let group = generate_group_data(n);

        bencher
            .bench_local(|| weighted_logrank_test(&time, &status_i32, &group, WeightType::LogRank));
    }

    #[divan::bench(args = [100, 1000, 10000])]
    fn fleming_harrington_test(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let group = generate_group_data(n);

        bencher.bench_local(|| {
            weighted_logrank_test(
                &time,
                &status_i32,
                &group,
                WeightType::FlemingHarrington { p: 0.5, q: 0.5 },
            )
        });
    }
}

mod brier_score {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn brier(bencher: divan::Bencher, n: usize) {
        let predictions = generate_predictions(n);
        let (_, _, outcomes) = generate_survival_data(n);

        bencher.bench_local(|| compute_brier(&predictions, &outcomes, None));
    }

    #[divan::bench(args = [100, 1000, 10000, 100000])]
    fn brier_weighted(bencher: divan::Bencher, n: usize) {
        let predictions = generate_predictions(n);
        let (_, _, outcomes) = generate_survival_data(n);
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 5) as f64 * 0.1).collect();

        bencher.bench_local(|| compute_brier(&predictions, &outcomes, Some(&weights)));
    }
}

mod rmst_bench {
    use super::*;

    #[divan::bench(args = [100, 1000, 10000])]
    fn rmst(bencher: divan::Bencher, n: usize) {
        let (time, _, status_i32) = generate_survival_data(n);
        let tau = time.iter().cloned().fold(0.0_f64, f64::max) * 0.8;

        bencher.bench_local(|| compute_rmst(&time, &status_i32, tau, 0.95));
    }
}

mod concordance_bench {
    use super::*;

    #[divan::bench(args = [100, 1000, 5000])]
    fn concordance(bencher: divan::Bencher, n: usize) {
        let (time, status, _) = generate_survival_data(n);
        let mut y = Vec::with_capacity(2 * n);
        y.extend_from_slice(&time);
        y.extend_from_slice(&status);

        let weights: Vec<f64> = vec![1.0; n];
        let ntree = 10i32;
        let indx: Vec<i32> = (0..n).map(|i| (i % ntree as usize) as i32).collect();

        bencher.bench_local(|| concordance1(&y, &weights, &indx, ntree));
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn concordance_tied_events(bencher: divan::Bencher, n: usize) {
        let mut y = vec![1.0; n];
        y.extend(vec![1.0; n]);

        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.1).collect();
        let ntree = 16i32;
        let indx: Vec<i32> = (0..n).map(|i| (i % ntree as usize) as i32).collect();

        bencher.bench_local(|| concordance1(&y, &weights, &indx, ntree));
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
        let covar: Vec<f64> = (0..n).map(|value| (value % 17) as f64).collect();
        let offset = vec![0.0; n];
        let strata = vec![0; n];
        let work = vec![0.0; n + 4];
        let work2 = vec![0; 2 * n];
        let inputs = (start, stop, event, covar, offset, strata, work, work2);

        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(start, stop, event, covar, offset, strata, work, work2)| {
                black_box(
                    agexact(
                        0,
                        n as i32,
                        1,
                        start,
                        stop,
                        event,
                        covar,
                        offset,
                        strata,
                        vec![0.0],
                        vec![0.0],
                        vec![0.0],
                        vec![0.0],
                        vec![0.0; 2],
                        work,
                        work2,
                        1e-9,
                        1e-9,
                        vec![0],
                    )
                    .expect("untied exact counting-process benchmark should succeed"),
                )
            },
        );
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
        let covar: Vec<f64> = (0..N).map(|value| value as f64).collect();
        let offset = vec![0.0; N];
        let strata = vec![0; N];
        let work = vec![0.0; N + 4];
        let work2 = vec![0; 2 * N];
        let inputs = (start, stop, event, covar, offset, strata, work, work2);

        bencher.with_inputs(|| inputs.clone()).bench_local_values(
            |(start, stop, event, covar, offset, strata, work, work2)| {
                black_box(
                    agexact(
                        0,
                        N as i32,
                        1,
                        start,
                        stop,
                        event,
                        covar,
                        offset,
                        strata,
                        vec![0.0],
                        vec![0.0],
                        vec![0.0],
                        vec![0.0],
                        vec![0.0; 2],
                        work,
                        work2,
                        1e-9,
                        1e-9,
                        vec![0],
                    )
                    .expect("benchmark exact counting-process fit should succeed"),
                )
            },
        );
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
                Some(20),
                Some(1e-7),
                Some(1e-9),
                Some("efron"),
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
                None,
                None,
                None,
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                Some("efron"),
                Some(entry_times.clone()),
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
                Some(20),
                Some(1e-7),
                Some(1e-9),
                Some("breslow"),
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
                Some(strata.clone()),
                Some(weights.clone()),
                None,
                None,
                Some(20),
                Some(1e-7),
                Some(1e-9),
                Some("efron"),
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
            Some(strata),
            Some(weights),
            None,
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            Some("efron"),
            Some(entry_times),
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let expected = fit
                .expected_events()
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
            Some(strata),
            Some(weights),
            None,
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            Some("efron"),
            None,
            None,
        )
        .expect("benchmark Cox PH fit should converge");
        let rows = generate_covariates(3, 4);
        let prediction_strata = vec![0, 1, 2];

        bencher.bench_local(|| {
            let curves = fit
                .survival_curve_with_strata(rows.clone(), prediction_strata.clone(), true)
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
            Some(strata),
            Some(weights),
            None,
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            Some("efron"),
            Some(entry_times),
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let residuals = fit
                .schoenfeld_residuals()
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
            Some(strata),
            Some(weights),
            None,
            None,
            Some(20),
            Some(1e-7),
            Some(1e-9),
            Some("efron"),
            Some(entry_times),
            None,
        )
        .expect("benchmark Cox PH fit should converge");

        bencher.bench_local(|| {
            let residuals = fit
                .score_residuals()
                .expect("benchmark score residuals should succeed");
            black_box(residuals);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_log_likelihood(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let log_likelihood = black_box(&model).log_likelihood();
            black_box(log_likelihood);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_brier_score(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let score = model
                .brier_score(None)
                .expect("benchmark Brier score should succeed");
            black_box(score);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_std_errors(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let standard_errors = black_box(&model).std_errors();
            black_box(standard_errors);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_dfbeta(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let residuals = black_box(&model).dfbeta();
            black_box(residuals);
        });
    }

    #[divan::bench(args = [100, 1000, 5000])]
    fn coxph_model_vcov(bencher: divan::Bencher, n: usize) {
        let model = fitted_coxph_model(n, 4);

        bencher.bench_local(|| {
            let variance = black_box(&model).vcov();
            black_box(variance);
        });
    }
}

mod survreg_bench {
    use super::*;

    fn status_as_survreg(status: &[i32]) -> Vec<f64> {
        status.iter().map(|&value| f64::from(value)).collect()
    }

    #[divan::bench(args = [100, 1000, 5000])]
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

    #[divan::bench(args = [100, 1000, 5000])]
    fn weighted_stratified_survreg_lognormal(bencher: divan::Bencher, n: usize) {
        let (time, status, covariates) = generate_tied_regression_data(n, 3);
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

fn main() {
    divan::main();
}
