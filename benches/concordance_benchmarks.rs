//! Shared outcome preparation across multiple concordance predictors.
use ndarray::Array2;
use survival::concordance::{ConcordanceOptions, TimeWeight, concordancefit};
use survival::core::SurvResponse;
use survival::data_types::{CountingProcessData, SurvivalData};

fn main() {
    divan::main();
}

const ROWS: usize = 20_000;

fn inputs(columns: usize) -> (SurvivalData, Array2<f64>, Vec<f64>) {
    let time: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + ((i * 4999) % ROWS / 4) as f64)
        .collect();
    let status = (0..ROWS).map(|i| i32::from(i % 4 != 0)).collect();
    let x = Array2::from_shape_fn((ROWS, columns), |(i, column)| {
        ((i * (37 + column * 8) + column * 11) % 127) as f64
    });
    let weights = (0..ROWS).map(|i| 0.5 + (i % 7) as f64 / 4.0).collect();
    (SurvivalData::try_new(time, status).unwrap(), x, weights)
}

#[divan::bench(args = [1, 8, 32])]
fn counts(bencher: divan::Bencher, columns: usize) {
    let (data, x, weights) = inputs(columns);
    let options = ConcordanceOptions {
        std_err: false,
        ..ConcordanceOptions::default()
    };
    bencher.bench_local(|| {
        concordancefit(
            SurvResponse::Right(&data),
            x.view(),
            Some(&weights),
            None,
            None,
            &options,
        )
        .unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn weighted_variance(bencher: divan::Bencher, columns: usize) {
    let (data, x, weights) = inputs(columns);
    let options = ConcordanceOptions {
        timewt: TimeWeight::S,
        ..ConcordanceOptions::default()
    };
    bencher.bench_local(|| {
        concordancefit(
            SurvResponse::Right(&data),
            x.view(),
            Some(&weights),
            None,
            None,
            &options,
        )
        .unwrap()
    });
}

#[divan::bench(args = [1, 8, 32])]
fn counting_influence_and_ranks(bencher: divan::Bencher, columns: usize) {
    let (data, x, weights) = inputs(columns);
    let start = data
        .time
        .iter()
        .map(|&time| (time - 50.0).max(0.0))
        .collect();
    let data = CountingProcessData::try_new(start, data.time, data.status).unwrap();
    let options = ConcordanceOptions {
        timewt: TimeWeight::I,
        influence: 3,
        ranks: true,
        ..ConcordanceOptions::default()
    };
    bencher.bench_local(|| {
        concordancefit(
            SurvResponse::Counting(&data),
            x.view(),
            Some(&weights),
            None,
            None,
            &options,
        )
        .unwrap()
    });
}
