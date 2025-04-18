use ndarray::{Array2, ArrayView2};
use serde::Deserialize;

struct CoxModel {
    coefficients: Vec<f64>,
}

struct SurvivalFit {
    survival: Vec<f64>,
    times: Vec<f64>,
}

impl CoxModel {
    fn fit(formula: Formula, data: &Dataset) -> Self {
        CoxModel {
            coefficients: vec![],
        }
    }
}

impl SurvivalFit {
    fn new(model: &CoxModel, new_data: &Dataset) -> Self {
        SurvivalFit {
            survival: vec![],
            times: vec![],
        }
    }
}

struct Formula;
struct Dataset {
    time: Vec<f64>,
    status: Vec<f64>,
    features: Array2<f64>,
}

fn tfun(fit: &CoxModel, mydata: &Dataset) -> SurvivalFit {
    SurvivalFit::new(fit, mydata)
}

#[derive(Debug, Deserialize)]
struct LungRecord {
    time: f64,
    status: f64,
    age: f64,
    sex: f64,
}

fn main() {
    let lung_data = load_lung_data();

    let features = Array2::from_shape_vec(
        (lung_data.len(), 2),
        lung_data
            .iter()
            .flat_map(|r| vec![r.age, if r.sex == 1.0 { 0.0 } else { 1.0 }])
            .collect(),
    )
    .unwrap();

    let dataset = Dataset {
        time: lung_data.iter().map(|r| r.time).collect(),
        status: lung_data.iter().map(|r| r.status).collect(),
        features,
    };

    let myfit = CoxModel::fit(Formula, &dataset);

    let subset = Dataset {
        time: dataset.time[0..5].to_vec(),
        status: dataset.status[0..5].to_vec(),
        features: dataset.features.slice(s![0..5, ..]).to_owned(),
    };

    let temp1 = tfun(&myfit, &subset);
    let temp2 = SurvivalFit::new(&myfit, &subset);

    assert_eq!(temp1.times, temp2.times);
    assert_eq!(temp1.survival, temp2.survival);
}

fn load_lung_data() -> Vec<LungRecord> {
    vec![]
}
