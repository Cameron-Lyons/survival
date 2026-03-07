use anyhow::{Context, Result};
use csv;
use linfa::prelude::*;
use linfa::traits::Fit;
use linfa_survival::{CoxPHModel, CoxPHParams};
use ndarray::{array, s, Array1, Array2};

#[derive(Debug, serde::Deserialize)]
struct RotterdamRow {
    recur: u32,
    death: u32,
    rtime: f64,
    dtime: f64,
    age: f64,
    meno: u32,
    size: u32,
    nodes: u32,
}

struct ProcessedRow {
    rfs: u32,
    rfstime: f64,
    age: f64,
    meno: u32,
    size: u32,
    capped_nodes: u32,
}

fn main() -> Result<()> {
    let mut processed = Vec::new();
    let mut rdr = csv::Reader::from_path("rotterdam.csv")?;

    for result in rdr.deserialize() {
        let row: RotterdamRow = result?;
        let ignore = row.recur == 0 && row.death == 1 && row.rtime < row.dtime;
        let rfs = if row.recur == 1 || ignore {
            1
        } else {
            row.death
        };
        let rfstime = (if row.recur == 1 || ignore {
            row.rtime
        } else {
            row.dtime
        }) / 365.25;

        processed.push(ProcessedRow {
            rfs,
            rfstime,
            age: row.age,
            meno: row.meno,
            size: row.size,
            capped_nodes: row.nodes.min(12),
        });
    }

    let features = processed
        .iter()
        .map(|row| {
            vec![
                row.age,
                row.meno as f64,
                row.size as f64,
                row.capped_nodes as f64,
            ]
        })
        .collect::<Array2<f64>>();

    let times = processed.iter().map(|r| r.rfstime).collect::<Vec<_>>();
    let status = processed.iter().map(|r| r.rfs).collect::<Vec<_>>();

    let dataset = Dataset::new(features, (times, status));
    let model = CoxPHParams::default()
        .fit(&dataset)
        .context("Failed to fit Cox model")?;

    let tau = array![2.0, 4.0, 6.0, 8.0];

    let weights = calculate_weights(&processed, &tau);
    let predictions = model.predict_survival(&tau);

    let y = create_outcome_matrix(&processed, &tau);
    let (ss1, ss2) = calculate_scores(&y, &predictions, &weights, &tau);

    println!("Brier scores: {:?}", ss1);
    println!("R-squared: {:?}", 1.0 - ss1.clone() / ss2);

    Ok(())
}

fn calculate_weights(data: &[ProcessedRow], tau: &Array1<f64>) -> Array2<f64> {
    Array2::ones((data.len(), tau.len()))
}

fn create_outcome_matrix(data: &[ProcessedRow], tau: &Array1<f64>) -> Array2<u32> {
    Array2::from_shape_fn((data.len(), tau.len()), |(i, j)| {
        (data[i].rfstime <= tau[j] && data[i].rfs == 1) as u32
    })
}

fn calculate_scores(
    y: &Array2<u32>,
    pred: &Array2<f64>,
    weights: &Array2<f64>,
    tau: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let y_float = y.mapv(|x| x as f64);
    let ss1 = (weights * &(&y_float - pred).mapv(|x| x.powi(2))).sum_axis(ndarray::Axis(0));
    let ss2 = (weights * &(&y_float - 0.5).mapv(|x| x.powi(2))).sum_axis(ndarray::Axis(0));
    (ss1, ss2)
}
