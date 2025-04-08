use polars::prelude::*;
use approx::assert_abs_diff_eq;

use survival::coxph::{CoxPH, PredictType};
use survival::survreg::SurvReg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lung_df = LazyCsvReader::new("lung.csv").finish()?.collect()?;

    let lung_df = lung_df
        .lazy()
        .with_columns([
            col("ph.ecog").cast(DataType::Categorical(None)),
            col("sex").cast(DataType::Categorical(None)),
        ])
        .drop_nulls(None)?
        .collect()?;

    fn aeq(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    let fit = CoxPH::new()
        .time_col("time")
        .status_col("status")
        .add_categorical_feature("ph.ecog")
        .fit(&lung_df)?;

    let tdata = df! [
        "ph.ecog" => [0, 1, 2, 3],
    ]?
    .cast("ph.ecog", DataType::Categorical(None))?;

    let p1 = fit.predict(&tdata, PredictType::LinearPredictor)?;
    let p2 = fit.predict(&lung_df, PredictType::LinearPredictor)?;

    let ph_ecog_values = lung_df.column("ph.ecog")?.i32()?;
    let ref_indices: Vec<usize> = [0, 1, 2, 3]
        .iter()
        .filter_map(|v| ph_ecog_values.into_iter().position(|x| x == *v))
        .collect();

    let p2_subset: Vec<f64> = ref_indices.iter().map(|&i| p2[i]).collect();
    assert!(aeq(&p1, &p2_subset), "First model prediction mismatch");

    let fit2 = CoxPH::new()
        .time_col("time")
        .status_col("status")
        .add_categorical_feature("ph.ecog")
        .add_categorical_feature("sex")
        .fit(&lung_df)?;

    let tdata = df! [
        "ph.ecog" => [0, 1, 2, 3, 0, 1, 2, 3],
        "sex" => [1, 1, 1, 1, 2, 2, 2, 2],
    ]?
    .cast("ph.ecog", DataType::Categorical(None))?
    .cast("sex", DataType::Categorical(None))?;

    let p1 = fit2.predict(&tdata, PredictType::Risk)?;

    let xdata = df! [
        "ph.ecog" => [1, 2, 3, 1, 2, 3],
        "sex" => [1, 1, 1, 2, 2, 2],
    ]?
    .cast("ph.ecog", DataType::Categorical(None))?
    .cast("sex", DataType::Categorical(None))?;

    let p2 = fit2.predict(&xdata, PredictType::Risk)?;

    let p1_subset = vec![p1[1], p1[2], p1[3], p1[5], p1[6], p1[7]];
    assert!(aeq(&p2, &p1_subset), "Second model prediction mismatch";

    let fit3 = SurvReg::new()
        .time_col("time")
        .status_col("status")
        .add_categorical_feature("ph.ecog")
        .add_feature("age")
        .fit(&lung_df)?;

    let tdata = df! [
        "ph.ecog" => [0, 1, 2, 3],
        "age" => [50; 4],
    ]?
    .cast("ph.ecog", DataType::Categorical(None))?;

    let _ = fit3.predict(&tdata, PredictType::LinearPredictor)?;

    Ok(())
}
