use std::hint::black_box;

use survival::surv_analysis::{SurvfitAJData, SurvfitAJOptions, survfitaj};

/// Competing-risks data: one interval per subject, a quarter of them ending
/// in the second state, on a grid of `max_times` distinct times.
fn benchmark_inputs(n: usize, max_times: usize) -> SurvfitAJData {
    let n_times = n.min(max_times);
    let time: Vec<f64> = (0..n).map(|idx| (idx % n_times + 1) as f64).collect();
    let state: Vec<i32> = (0..n).map(|idx| if idx % 4 == 0 { 1 } else { 0 }).collect();
    SurvfitAJData::try_new(
        None,
        time,
        state,
        vec!["event".to_string()],
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("benchmark inputs should define a valid multistate curve")
}

fn run_benchmark(data: SurvfitAJData, options: SurvfitAJOptions) {
    black_box(
        survfitaj(&data, &options)
            .expect("benchmark inputs should define a valid multistate curve"),
    );
}

#[divan::bench(args = [100, 1000, 10000])]
fn point_estimates(bencher: divan::Bencher, n: usize) {
    let data = benchmark_inputs(n, 250);
    let options = SurvfitAJOptions {
        se_fit: false,
        ..Default::default()
    };
    bencher
        .with_inputs(|| (data.clone(), options.clone()))
        .bench_local_values(|(data, options)| run_benchmark(data, options));
}

#[divan::bench(args = [1000, 10000])]
fn point_estimates_long_grid(bencher: divan::Bencher, n: usize) {
    let data = benchmark_inputs(n, n);
    let options = SurvfitAJOptions {
        se_fit: false,
        ..Default::default()
    };
    bencher
        .with_inputs(|| (data.clone(), options.clone()))
        .bench_local_values(|(data, options)| run_benchmark(data, options));
}

#[divan::bench(args = [100, 1000, 5000])]
fn standard_errors(bencher: divan::Bencher, n: usize) {
    let data = benchmark_inputs(n, 250);
    let options = SurvfitAJOptions::default();
    bencher
        .with_inputs(|| (data.clone(), options.clone()))
        .bench_local_values(|(data, options)| run_benchmark(data, options));
}

#[divan::bench(args = [100, 1000])]
fn influence_output(bencher: divan::Bencher, n: usize) {
    let data = benchmark_inputs(n, 250);
    let options = SurvfitAJOptions {
        influence: true,
        ..Default::default()
    };
    bencher
        .with_inputs(|| (data.clone(), options.clone()))
        .bench_local_values(|(data, options)| run_benchmark(data, options));
}

fn main() {
    divan::main();
}
