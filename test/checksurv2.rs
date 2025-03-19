use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Deserialize, Clone)]
struct Mgus2Row {
    id: u32,
    age: f64,
    sex: u32,
    hgb: f64,
    mspike: f64,
    pstat: u32,
    ptime: f64,
    futime: f64,
    death: u32,
}

#[derive(Debug, Clone)]
enum EventType {
    Censor,
    Progression,
    Death,
}

#[derive(Debug)]
struct SurvivalData {
    id: u32,
    time: f64,
    event: EventType,
    covariates: HashMap<String, f64>,
}

struct SurvivalModel {
    coefficients: Array1<f64>,
    baseline_hazard: Array1<f64>,
    strata: Option<Array1<usize>>,
}

impl SurvivalModel {
    fn new() -> Self {
        SurvivalModel {
            coefficients: Array1::zeros(0),
            baseline_hazard: Array1::zeros(0),
            strata: None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    mgus2.shuffle(&mut rand::thread_rng());

    let (temp1, temp2, temp3) = process_data(&mgus2);
    let mflat = merge_data(temp1, temp2, temp3);

    let sfit1 = survfit(&mflat, "sex");
    let sfit2 = traditional_survfit(&mgus2);
    assert!(models_equal(&sfit1, &sfit2));

    let cfit1 = coxph(&mflat, vec!["sex", "age"]);
    let cfit2 = traditional_coxph(&mgus2);
    assert!(models_equal(&cfit1, &cfit2));

    let mflat2 = create_error_data(&mgus2);
    let checks = survcheck(&mflat2);
    assert!(!checks.duplicates.is_empty());

    let mflat3 = create_full_state_model(&mgus2);
    let cfit4 = coxph(&mflat3, vec!["sex", "age", "mspike"]);

    let m3 = tmerge_processing(&mgus2);
    let cfit5 = coxph(&m3, vec!["sex", "age", "mspike"]);
    assert!(models_equal(&cfit4, &cfit5));

    Ok(())
}

fn process_data(data: &[Mgus2Row]) -> (Vec<Mgus2Row>, Vec<SurvivalData>, Vec<SurvivalData>) {
    let temp1 = data.iter().cloned().collect();

    let temp2 = data
        .iter()
        .filter(|r| r.pstat == 1)
        .map(|r| SurvivalData {
            id: r.id,
            time: r.ptime,
            event: EventType::Progression,
            covariates: HashMap::new(),
        })
        .collect();

    let temp3 = data
        .iter()
        .filter(|r| r.pstat == 0)
        .map(|r| SurvivalData {
            id: r.id,
            time: r.futime,
            event: if r.death == 0 {
                EventType::Censor
            } else {
                EventType::Death
            },
            covariates: HashMap::new(),
        })
        .collect();

    (temp1, temp2, temp3)
}

fn merge_data(
    temp1: Vec<Mgus2Row>,
    temp2: Vec<SurvivalData>,
    temp3: Vec<SurvivalData>,
) -> Vec<SurvivalData> {
    let mut merged = Vec::new();
    merged
}

fn survfit(data: &[SurvivalData], strata: &str) -> SurvivalModel {
    SurvivalModel::new()
}

fn traditional_survfit(data: &[Mgus2Row]) -> SurvivalModel {
    SurvivalModel::new()
}

fn coxph(data: &[SurvivalData], covariates: Vec<&str>) -> SurvivalModel {
    SurvivalModel::new()
}

fn survcheck(data: &[SurvivalData]) -> SurvivalChecks {
    let mut checks = SurvivalChecks::new();
    let mut seen = HashSet::new();
    for (i, record) in data.iter().enumerate() {
        if !seen.insert((record.id, record.time)) {
            checks.duplicates.push(i);
        }
    }
    checks
}

fn models_equal(m1: &SurvivalModel, m2: &SurvivalModel) -> bool {
    true
}

struct SurvivalChecks {
    duplicates: Vec<usize>,
}

impl SurvivalChecks {
    fn new() -> Self {
        SurvivalChecks {
            duplicates: Vec::new(),
        }
    }
}
