use approx::assert_abs_diff_eq;
use linfa::prelude::*;
use linfa_survival::CoxPHParams;
use ndarray::{Array, Array1, Array2, Axis};
use survival::prelude::{DataFrame, SurvivalData, SurvivalRegression, SurvivorData};

fn aeq(x: &Array1<f64>, y: &Array1<f64>, tol: f64) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < tol)
}

fn main() {
    let lung = load_lung_data();
    let aml = load_aml_data();
    let nafld1 = load_nafld_data();
    let cgd = load_cgd_data();

    let fit1 = fit_cox_model(&lung, true);
    let fit2 = fit_cox_model(&lung, false);

    let test = concordance_comparison(&fit1, &fit2);

    let ksex = lung.column("sex").to_vec();
    let test1 = stratified_concordance(&fit1, &ksex);
    let test2 = stratified_concordance(&fit2, &ksex);

    assert!(aeq(
        &test.concordance,
        &array![test1.concordance, test2.concordance],
        1e-6
    ));
    assert!(aeq(
        &test.var.diag(),
        &array![test1.var[[0, 0]], test2.var[[0, 0]]],
        1e-6
    ));

    analyze_aml(&aml);

    analyze_nafld(&nafld1);

    check_interval_censoring(&cgd);
}

fn fit_cox_model(data: &DataFrame, include_offset: bool) -> SurvivalRegression {
    let time = data.column("time").as_f64();
    let status = data.column("status").as_bool();
    let age = data.column("age").as_f64();
    let ph_ecog = data.column("ph.ecog").as_f64();
    let sex = data.column("sex").as_factor();

    let mut builder = SurvivalRegression::cox()
        .add_covariate("age", age)
        .strata(sex);

    if include_offset {
        let offset: Vec<f64> = ph_ecog.iter().map(|&x| x * 0.0).collect();
        builder = builder.offset(offset);
    } else {
        builder = builder.add_covariate("ph.ecog", ph_ecog);
    }

    builder.fit(&time, &status).unwrap()
}

struct ConcordanceResult {
    concordance: Array1<f64>,
    var: Array2<f64>,
    dfbeta: Array2<f64>,
}

fn concordance_comparison(
    fit1: &SurvivalRegression,
    fit2: &SurvivalRegression,
) -> ConcordanceResult {
    unimplemented!()
}

fn stratified_concordance(model: &SurvivalRegression, strata: &[f64]) -> ConcordanceResult {
    unimplemented!()
}

fn analyze_aml(data: &DataFrame) {
    let time = data.column("time").as_f64();
    let status = data.column("status").as_bool();
    let x = data.column("x").as_factor();

    let params = CoxPHParams::new().with_ranks(true).with_timewt("n");

    let concordance = linfa_survival::concordance(&time, &status, &x.view(), params).unwrap();

    unimplemented!()
}

fn analyze_nafld(data: &DataFrame) {
    
    let time = data.column("futime").as_f64();
    let status = data.column("status").as_bool();
    let male = data.column("male").as_bool();
    let age = data.column("age").as_f64();

    let model = SurvivalRegression::cox()
        .add_covariate("male", male)
        .add_spline("age", age, 3)
        .fit(&time, &status)
        .unwrap();

    unimplemented!()
}

fn check_interval_censoring(data: &DataFrame) {
    
    let tstart = data.column("tstart").as_f64();
    let tstop = data.column("tstop").as_f64();
    let status = data.column("status").as_bool();
    let id = data.column("id").as_usize();

    let model = SurvivalRegression::cox()
        .interval_censored()
        .fit(&(tstart, tstop), &status)
        .unwrap();

    unimplemented!()
}

fn load_lung_data() -> DataFrame {
    unimplemented!()
}
fn load_aml_data() -> DataFrame {
    unimplemented!()
}
fn load_nafld_data() -> DataFrame {
    unimplemented!()
}
fn load_cgd_data() -> DataFrame {
    unimplemented!()
}
