use ndarray::{Array1, Array2};
use polars::prelude::*;
use survival::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let df = DataFrame::default(); 

    let n = df.height();
    let occupation = df.column("occupation")?.unique()?.sort(false);
    let nresp = occupation.len();

    let indices: Vec<usize> = (0..n)
        .flat_map(|i| std::iter::repeat(i).take(nresp))
        .collect();
    let expanded_df = df.take(&indices.into())?;

    let occ2_values = occupation
        .utf8()?
        .into_no_null_iter()
        .cycle()
        .take(n * nresp)
        .collect();
    let mut expanded_df = expanded_df.with_column(Series::new("occ2", occ2_values))?;

    let y = expanded_df
        .column("occupation")?
        .equal_to(expanded_df.column("occ2")?)?
        .cast(&DataType::UInt32)?;
    let expanded_df = expanded_df.with_column(y.alias("y"))?;

    let expanded_df = expanded_df.with_column(Series::new("id", indices))?;

    let y_data: Array1<f64> = expanded_df
        .column("y")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let strata: Array1<usize> = expanded_df
        .column("id")?
        .u32()?
        .into_no_null_iter()
        .map(|x| x as usize)
        .collect();

    let mut covariates = Vec::new();
    for row in expanded_df.iter_rows() {
        let mut features = vec![/];
        features.extend(/);
        features.extend(/);
        covariates.extend(features);
    }
    let covariates =
        Array2::from_shape_vec((y_data.len(), covariates.len() / y_data.len()), covariates)?;

    let model =
        CoxPH::new()
            .strata(strata)
            .fit(&covariates, &Array1::ones(y_data.len()), &y_data)?;

    println!("Coefficients: {:?}", model.coefficients());
    println!("Log-likelihood: {}", model.log_likelihood());

    Ok(())
}
