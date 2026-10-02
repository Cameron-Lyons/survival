//! Run with `cargo run --example rust_package_layout --no-default-features`.

use ndarray::array;
use survival::SurvivalResult;
use survival::regression::{CoxPHFit, CoxphData, CoxphOptions};
use survival::surv_analysis::{SurvfitKMData, SurvfitKMOptions, survfitkm};

fn main() -> SurvivalResult<()> {
    let time = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0];
    let status = vec![1, 1, 0, 1, 0, 1, 1, 0];
    let x = array![[0.2], [-0.5], [1.0], [0.1], [-0.2], [0.8], [-0.4], [0.3]];

    let km_data = SurvfitKMData::right_censored(time.clone(), status.clone())?;
    let km = survfitkm(&km_data, &SurvfitKMOptions::default())?;
    println!("Kaplan-Meier times: {:?}", km.time);
    println!("Survival probabilities: {:?}", km.surv);

    let cox_data = CoxphData::try_new(time, None, status, x, None, None, None)?;
    let cox = CoxPHFit::fit(cox_data, CoxphOptions::default())?;
    println!("Cox coefficients: {:?}", cox.coefficients);
    Ok(())
}
