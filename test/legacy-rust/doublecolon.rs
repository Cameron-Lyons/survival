use survival::prelude::*;
use survival::{coxph, survfit, survreg, concordance, survdiff};
use survival::lung::load_lung_dataset;

fn aeq(x: &[f64], y: &[f64], tol: f64) -> bool {
    x.iter().zip(y.iter()).all(|(a, b)| (a - b).abs() < tol)
}

#[test]
fn test_survival_functions() {
    let lung = load_lung_dataset();

    let c1 = coxph(Surv::new("time", "status") ~ age + strata("inst"), &lung).unwrap();
    
    let c3 = coxph(survival::Surv::new("time", "status") ~ age + survival::strata("inst"), &lung).unwrap();
    
    assert!(aeq(c1.coefficients(), c3.coefficients(), 1e-6));

    let fit5a = survfit(Surv::new("time", "status") ~ sex, &lung).unwrap();
    let fit5c = survfit(survival::Surv::new("time", "status") ~ survival::strata("sex"), &lung).unwrap();
    assert_eq!(fit5a.survival_function(), fit5c.survival_function());

}
