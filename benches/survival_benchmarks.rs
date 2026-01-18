use survival::{
    KaplanMeierConfig, WeightType, compute_brier, compute_rmst, compute_survfitkm, concordance1,
    nelson_aalen, weighted_logrank_test,
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
