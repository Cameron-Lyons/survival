use ndarray::{Array, Array2, Axis, concatenate};
use polars::prelude::*;
use approx::assert_abs_diff_eq;

use survival::prelude::*;

fn aeq(x: &Array<f64, ndarray::Ix1>, y: &Array<f64, ndarray::Ix1>, tol: f64) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < tol)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tdata = DataFrame::from_csv("myeloid.csv")?;
    
    process_tmerge(&mut tdata)?;
    create_event_factor(&mut tdata)?;

    let formula = "Surv(tstart, tstop, event) ~ trt + sex";
    let fit = CoxPH::fit(formula, &tdata)
        .id("id")
        .iter(4)
        .robust(false)
        .build()?;

    let fit12 = CoxPH::fit("Surv(tstart, tstop, sct) ~ trt + sex", &tdata)
        .subset("priortx == 0")
        .method(Method::Breslow)
        .build()?;

    let fit13 = CoxPH::fit("Surv(tstart, tstop, death) ~ trt + sex", &tdata)
        .subset("priortx == 0")
        .method(Method::Breslow)
        .build()?;

    let fit23 = CoxPH::fit("Surv(tstart, tstop, death) ~ trt + sex", &tdata)
        .subset("priortx == 1")
        .method(Method::Breslow)
        .build()?;

    let combined_coef = concatenate![
        Axis(0),
        fit12.coefficients().view(),
        fit13.coefficients().view(),
        fit23.coefficients().view()
    ];
    assert!(aeq(&fit.coefficients(), &combined_coef, 1e-6));

    let total_ll = fit12.loglik() + fit13.loglik() + fit23.loglik();
    assert_abs_diff_eq!(fit.loglik(), total_ll, epsilon = 1e-6);

    let mut combined_var = Array2::zeros((6, 6));
    combined_var.slice_mut(s![0..2, 0..2]).assign(&fit12.var());
    combined_var.slice_mut(s![2..4, 2..4]).assign(&fit13.var());
    combined_var.slice_mut(s![4..6, 4..6]).assign(&fit23.var());
    assert!(aeq(&fit.var().into_owned(), &combined_var.into_raw_vec(), 1e-6));

    let res = fit.residuals(ResidualType::Martingale);
    let res12 = fit12.residuals(ResidualType::Martingale);
    let res13 = fit13.residuals(ResidualType::Martingale);
    let res23 = fit23.residuals(ResidualType::Martingale);
    let combined_res = concatenate![Axis(0), res12, res13, res23];
    assert!(aeq(&res, &combined_res, 1e-6));

    let score_res = fit.score_residuals();
    let x = fit.model_matrix();
    let x12 = fit12.model_matrix();
    let x13 = fit13.model_matrix();
    let x23 = fit23.model_matrix();

    Ok(())
}

fn process_tmerge(df: &mut DataFrame) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

fn create_event_factor(df: &mut DataFrame) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
