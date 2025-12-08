use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray::Array2;
use qstats::survival::{self, coxph};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

#[derive(Debug, serde::Deserialize)]
struct LungRecord {
    time: f64,
    status: u8,
    sex: u8,
    age: f64,
}

fn read_lung_data(file_path: &str) -> Result<Vec<LungRecord>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut data = Vec::new();
    for result in rdr.deserialize() {
        let record: LungRecord = result?;
        data.push(record);
    }
    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = read_lung_data("lung.csv")?;

    {
        let (male, female): (Vec<_>, Vec<_>) = data.iter().partition(|r| r.sex == 1);
        let km_male = survival::kaplan_meier(
            &male.iter().map(|r| r.time).collect::<Vec<_>>(),
            &male.iter().map(|r| r.status == 1).collect::<Vec<_>>(),
            None,
        );
        let km_female = survival::kaplan_meier(
            &female.iter().map(|r| r.time).collect::<Vec<_>>(),
            &female.iter().map(|r| r.status == 1).collect::<Vec<_>>(),
            None,
        );

        let times: Vec<f64> = data.iter().map(|r| r.time).collect();
        let status: Vec<bool> = data.iter().map(|r| r.status == 1).collect();
        let age: Vec<f64> = data.iter().map(|r| r.age).collect();
        let strata: Vec<i32> = data.iter().map(|r| r.sex as i32).collect();

        let params = coxph::Params {
            max_iter: 0,
            tol: 1e-9,
            initial_coefs: Some(vec![0.0]),
            ties: coxph::TiesMethod::Breslow,
            strata: Some(&strata),
            weights: None,
        };

        let mut cox_model = survival::coxph(
            ×,
            &status,
            &[age.as_slice()], 
            params,
        )?;

        cox_model.variance = Array2::zeros((1, 1));

        let baseline_hazards = cox_model.baseline_hazard();
        
    }

    Ok(())
}
