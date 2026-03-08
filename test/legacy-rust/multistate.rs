use linfa::traits::{Fit, Predict};
use linfa::Dataset;
use linfa_survival::CoxPHParams;
use ndarray::{array, concatenate, Axis};
use polars::prelude::*;
use approx::assert_abs_diff_eq;

fn main() {
    let df = CsvReader::from_path("mgus2.csv")
        .unwrap()
        .finish()
        .unwrap();

    let data1 = df.clone().lazy()
        .with_column(
            when(col("pstat").eq(1))
                .then(col("ptime"))
                .otherwise(col("futime"))
                .alias("etime")
        )
        .with_column(
            when(col("pstat").eq(1))
                .then(lit(1))
                .otherwise(col("death") * 2)
                .cast(DataType::UInt32)
                .alias("event_code")
        )
        .with_column(
            col("event_code")
                .cast(DataType::Categorical(None))
                .alias("event")
        )
        .collect()
        .unwrap();

    let data2 = df.clone().lazy()
        .select(&["id", "age", "sex", "mspike"])
        .sort_by_exprs(&[col("id")], &[false], false)
        .collect()
        .unwrap()
        .repeat(2, true)
        .unwrap()
        .with_column(
            concatenate(&[
                data1.column("etime").unwrap().clone(),
                data1.column("etime").unwrap().clone()
            ], true).unwrap().alias("time")
        )
        .with_column(
            concatenate(&[
                Series::new("status_pcm", vec![1; data1.height()]),
                Series::new("status_death", vec![0; data1.height()])
            ], true).unwrap().alias("status")
        )
        .with_column(
            Series::new("type", [vec![2; data1.height()], vec![3; data1.height()]].concat())
        )
        .collect()
        .unwrap();

    let features_pcm = data1.lazy()
        .filter(col("event").eq(lit("PCM")))
        .select(&["age", "sex", "mspike"])
        .collect()
        .unwrap();
    let times_pcm = data1.lazy()
        .filter(col("event").eq(lit("PCM")))
        .select(&["etime"])
        .collect()
        .unwrap();
    
    let dataset_pcm = Dataset::new(
        features_pcm.to_ndarray::<Float64Type>().unwrap(),
        times_pcm.to_ndarray::<Float64Type>().unwrap()
    );
    
    let params = CoxPHParams::new().ties("breslow");
    let model_pcm = params.fit(&dataset_pcm).unwrap();

}
