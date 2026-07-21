use std::hint::black_box;

use survival::surv_analysis::survfitaj;

#[derive(Clone)]
struct SurvfitAjInputs {
    y: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    utime: Vec<f64>,
    cstate: Vec<usize>,
    weights: Vec<f64>,
    groups: Vec<usize>,
    n_groups: usize,
    p0: Vec<f64>,
    i0: Vec<f64>,
    sefit: i32,
    position: Vec<usize>,
    hindx: Vec<Vec<usize>>,
    trmat: Vec<Vec<usize>>,
}

fn benchmark_inputs(n: usize, sefit: i32, max_times: usize) -> SurvfitAjInputs {
    let n_times = n.min(max_times);
    let mut y = Vec::with_capacity(3 * n);
    for idx in 0..n {
        y.push(0.0);
        y.push((idx % n_times + 1) as f64);
        y.push(if idx % 4 == 0 { 2.0 } else { 0.0 });
    }
    let sort1: Vec<usize> = (0..n).collect();
    let mut sort2: Vec<usize> = (0..n).collect();
    sort2.sort_unstable_by_key(|&idx| (idx % n_times, idx));

    SurvfitAjInputs {
        y,
        sort1,
        sort2,
        utime: (1..=n_times).map(|time| time as f64).collect(),
        cstate: vec![0; n],
        weights: vec![1.0; n],
        groups: (0..n).collect(),
        n_groups: n,
        p0: vec![1.0, 0.0],
        i0: if sefit == 0 {
            Vec::new()
        } else {
            vec![0.0; 2 * n]
        },
        sefit,
        position: vec![3; n],
        hindx: vec![vec![1, 0], vec![1, 1]],
        trmat: vec![vec![0, 1]],
    }
}

fn run_benchmark(inputs: SurvfitAjInputs) {
    black_box(
        survfitaj(
            inputs.y,
            inputs.sort1,
            inputs.sort2,
            inputs.utime,
            inputs.cstate,
            inputs.weights,
            inputs.groups,
            inputs.n_groups,
            inputs.p0,
            inputs.i0,
            inputs.sefit,
            false,
            inputs.position,
            inputs.hindx,
            inputs.trmat,
            0.0,
        )
        .expect("benchmark inputs should define a valid multistate curve"),
    );
}

#[divan::bench(args = [100, 1000, 10000])]
fn point_estimates(bencher: divan::Bencher, n: usize) {
    let inputs = benchmark_inputs(n, 0, 250);
    bencher
        .with_inputs(|| inputs.clone())
        .bench_local_values(run_benchmark);
}

#[divan::bench(args = [1000, 10000])]
fn point_estimates_long_grid(bencher: divan::Bencher, n: usize) {
    let inputs = benchmark_inputs(n, 0, n);
    bencher
        .with_inputs(|| inputs.clone())
        .bench_local_values(run_benchmark);
}

#[divan::bench(args = [100, 1000, 5000])]
fn standard_errors(bencher: divan::Bencher, n: usize) {
    let inputs = benchmark_inputs(n, 1, 250);
    bencher
        .with_inputs(|| inputs.clone())
        .bench_local_values(run_benchmark);
}

#[divan::bench(args = [100, 1000])]
fn influence_output(bencher: divan::Bencher, n: usize) {
    let inputs = benchmark_inputs(n, 3, 250);
    bencher
        .with_inputs(|| inputs.clone())
        .bench_local_values(run_benchmark);
}

fn main() {
    divan::main();
}
