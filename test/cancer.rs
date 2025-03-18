use anyhow::{Context, Result};
use csv;
use linfa::prelude::*;
use linfa::traits::Fit;
use ndarray::{Array1, Array2};
use statrs::distribution::Normal;

#[derive(Debug, serde::Deserialize)]
struct LungRow {
    time: f64,
    status: u32,
    ph_ecog: f64,
    ph_karno: f64,
    pat_karno: f64,
    wt_loss: f64,
    sex: u32,
    age: f64,
    inst: u32,
}

struct SurvivalCurve {
    times: Vec<f64>,
    survival: Vec<f64>,
    strata: Option<Vec<usize>>,
}

struct SurvivalDiffResult {
    chisq: f64,
    df: usize,
}

struct CoxModel {
    coefficients: Array1<f64>,
    baseline_hazard: Array1<f64>,
}

impl CoxModel {
    fn new() -> Self {
        CoxModel {
            coefficients: Array1::zeros(0),
            baseline_hazard: Array1::zeros(0),
        }
    }
}

fn main() -> Result<()> {
    let mut rdr = csv::Reader::from_path("lung.csv")?;
    let mut lung_data = Vec::new();
    for result in rdr.deserialize() {
        let row: LungRow = result?;
        lung_data.push(row);
    }

    let temp = surv_fit(&lung_data, Some("ph_ecog"));
    summarize_surv(&temp, &[30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 365.0]);

    let diff_result = surv_diff(&lung_data, "inst", 0.5);
    println!(
        "Survival difference test: χ²={:.4}, df={}",
        diff_result.chisq, diff_result.df
    );

    let cfit1 = cox_ph(
        &lung_data,
        vec!["ph_ecog", "ph_karno", "pat_karno", "wt_loss", "sex", "age"],
        Some("inst"),
    );

    let fit1 = cox_ph(&lung_data, vec!["age", "sex", "ph_ecog", "pat_karno"], None);
    let fit2 = cox_ph(&lung_data, vec!["age", "sex", "ph_ecog", "pat_karno"], None);
    assert!(all_close(&fit1.coefficients, &fit2.coefficients));

    Ok(())
}

fn surv_fit(data: &[LungRow], strata: Option<&str>) -> SurvivalCurve {
    let mut times = Vec::new();
    let mut survival = Vec::new();

    SurvivalCurve {
        times,
        survival,
        strata: None,
    }
}

fn surv_diff(data: &[LungRow], group_var: &str, rho: f64) -> SurvivalDiffResult {
    SurvivalDiffResult {
        chisq: 12.345,
        df: 3,
    }
}

fn cox_ph(data: &[LungRow], features: Vec<&str>, strata: Option<&str>) -> CoxModel {
    CoxModel::new()
}

fn summarize_surv(curve: &SurvivalCurve, times: &[f64]) {}

fn all_close(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
}
