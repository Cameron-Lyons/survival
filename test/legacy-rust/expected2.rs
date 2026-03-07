use chrono::{Date, Datelike, Duration, NaiveDate, Utc};
use ndarray::{Array1, Array2, Axis};
use survival::{
    coxph::{CoxPHModel, SurvivalData},
    prelude::ScaleType,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Surv {
    time: Vec<f64>,
    event: Vec<bool>,
}

fn coxph(formula: &str, data: &PbcData) -> CoxPHModel {
    let event: Vec<bool> = data.status.iter().map(|&s| s > 0).collect();
    
    let covariates = prepare_covariates(data);
    
    CoxPHModel::new()
        .add_covariates(covariates)
        .fit(Surv {
            time: data.time.clone(),
            event,
        })
        .expect("Model fitting failed")
}

fn survexp(formula: &str, ratetable: &CoxPHModel, data: &PbcData) -> ExpectedSurvival {
    ExpectedSurvival::new()
}

fn datedate(date: Date<Utc>) -> i64 {
    let origin_1960 = NaiveDate::from_ymd(1960, 1, 1).and_hms(0, 0, 0);
    let origin_1970 = NaiveDate::from_ymd(1970, 1, 1).and_hms(0, 0, 0);
    let offset = origin_1970.signed_duration_since(origin_1960);
    date.timestamp() + offset.num_seconds()
}

#[derive(Debug, Deserialize)]
struct LungData {
    age: Vec<f64>,
    sex: Vec<u8>,
    ph_ecog: Vec<f64>,
    time: Vec<f64>,
    status: Vec<u8>,
    entry: Vec<Date<Utc>>,
}

fn pyears(
    formula: &str,
    data: &LungData,
    ratetable: &HashMap<&str, Array2<f64>>,
    rmap: HashMap<&str, Array1<f64>>,
) -> PyearsResult {
    PyearsResult::new()
}

fn main() {
    let pbc_data = load_pbc_data();
    let pfit2 = coxph("Surv(time, status) ~ trt + log(bili) + log(protime) + age + platelet + sex", &pbc_data);
    let esurv = survexp("~ trt", &pfit2, &pbc_data);

    let mut temp = pbc_data.clone();
    temp.sex2 = temp.sex
        .iter()
        .map(|&s| match s {
            2 => "f",
            1 => "m",
            _ => "unknown",
        })
        .collect();
    
    let esurv2 = survexp("~ trt", &pfit2, &temp);
    
    let lung_data = load_lung_data();
    let n = lung_data.time.len();
    
    let entry: Vec<Date<Utc>> = (0..n)
        .map(|i| NaiveDate::from_ymd(1940, 1, 1) + Duration::days((n - i) * 50))
        .collect();
    
    let entry2 = entry.iter().map(|d| d.and_hms(0, 0, 0)).collect();
    let entry3 = entry.iter().map(|&d| datedate(d)).collect();
    
    let p1 = pyears("Surv(time, status) ~ ph_ecog", &lung_data, survexp_us(), rmap! {
        "age" => lung_data.age * 365.25,
        "sex" => lung_data.sex,
        "year" => entry
    });
    
}
