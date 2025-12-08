use linfa::traits::Fit;
use linfa::Dataset;
use ndarray::{array, Array2, Axis};
use survival::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bladder = load_bladder_data()?;
    let bladder2 = load_bladder2_data()?;

    let options = CoxOptions::default()
        .missing_data_handling(MissingData::Exclude)
        .contrasts(Contrasts::Treatment)
        .tie_method(TieMethod::Breslow);

    let formula = "Surv(stop, event) ~ (rx + size + number) * strata(enum)";
    let wfit = CoxPH::new()
        .formula(formula.parse()?)
        .cluster_by("id")
        .fit(&bladder, options.clone())?;

    println!("Wei et al. model:");
    println!("{:?}", wfit);

    let rx_indices = vec![1, 4, 5, 6];
    let cmat = Array2::from_shape_fn((4, 4), |(i, j)| if i == 0 || j == i { 1.0 } else { 0.0 });

    let coefficients = wfit.coefficients().select(Axis(0), &rx_indices);
    let transformed_coef = coefficients.dot(&cmat);
    println!("Transformed coefficients:\n{}", transformed_coef);

    let variance = wfit
        .variance()
        .select(Axis(0), &rx_indices)
        .select(Axis(1), &rx_indices);
    let transformed_var = cmat.t().dot(&variance).dot(&cmat);
    println!("Transformed variance:\n{}", transformed_var);

    let fita = CoxPH::new()
        .formula("Surv(start, stop, event) ~ rx + size + number".parse()?)
        .cluster_by("id")
        .fit(&bladder2, options.clone())?;
    println!("\nAnderson-Gill model:");
    println!("{:?}", fita);

    let fit1p = fit_prentice_model(&bladder2, 1)?;
    let fit2pa = fit_prentice_model(&bladder2, 2)?;
    let fit2pb = fit_prentice_model_time_since(&bladder2, 2)?;
    let fit3pa = fit_prentice_model(&bladder2, 3)?;

    println!("\nPrentice models:");
    println!("Model 1: {:?}", fit1p);
    println!("Model 2a: {:?}", fit2pa);
    println!("Model 2b: {:?}", fit2pb);
    println!("Model 3a: {:?}", fit3pa);

    Ok(())
}

fn fit_prentice_model(
    data: &Dataset,
    enum_value: u32,
) -> Result<CoxPH, Box<dyn std::error::Error>> {
    let filtered_data = data.filter(|row| row["enum"] == enum_value);
    CoxPH::new()
        .formula("Surv(stop, event) ~ rx + size + number".parse()?)
        .fit(&filtered_data, CoxOptions::default())
}

fn fit_prentice_model_time_since(
    data: &Dataset,
    enum_value: u32,
) -> Result<CoxPH, Box<dyn std::error::Error>> {
    let mut modified_data = data.clone();
    modified_data["time_since"] = modified_data["stop"] - modified_data["start"];

    let filtered_data = modified_data.filter(|row| row["enum"] == enum_value);
    CoxPH::new()
        .formula("Surv(time_since, event) ~ rx + size + number".parse()?)
        .fit(&filtered_data, CoxOptions::default())
}
