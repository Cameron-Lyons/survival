use approx::assert_abs_diff_eq;
use survival::prelude::*;
use survival::stats::{CoxModel, ModelComparison};

#[test]
fn test_cox_anova() {
    let lung_data = load_lung_data().expect("Failed to load lung data");
    let tdata = lung_data.na_omit();

    let formula1 = "Surv(time, status) ~ ph.ecog + wt.loss + strata(sex) + poly(age, 3)";
    let fit1 = CoxModel::fit(formula1, &lung_data).expect("fit1 failed");

    let formula1_clean = "Surv(time, status) ~ ph.ecog + wt.loss + poly(age, 3) + strata(sex)";
    let fit1_clean = CoxModel::fit(formula1_clean, &tdata).expect("fit1_clean failed");

    let anova1 = fit1.anova().expect("anova1 failed");
    let anova1_clean = fit1_clean.anova().expect("anova1_clean failed");
    assert!(
        anova1.approx_eq(&anova1_clean, 1e-6),
        "ANOVA results differ"
    );

    let fit2 = CoxModel::fit(
        "Surv(time, status) ~ ph.ecog + wt.loss + strata(sex)",
        &tdata,
    )
    .expect("fit2 failed");
    let fit3 =
        CoxModel::fit("Surv(time, status) ~ ph.ecog + strata(sex)", &tdata).expect("fit3 failed");

    let expected_loglik = [
        fit1.null_loglik(),
        fit3.loglik(),
        fit2.loglik(),
        fit1.loglik(),
    ];
    assert_abs_diff_eq!(anova1.loglik(), &expected_loglik[1..], epsilon = 1e-6);

    let chisq: Vec<_> = anova1.loglik_diff().iter().map(|d| 2.0 * d).collect();
    assert_abs_diff_eq!(anova1.chisq(), &chisq, epsilon = 1e-6);

    let expected_df = vec![1, 1, 3];
    assert_eq!(anova1.df(), expected_df);

    let seq_test = ModelComparison::new()
        .add_model(fit3)
        .add_model(fit2)
        .add_model(fit1)
        .compare()
        .expect("Model comparison failed");

    assert_abs_diff_eq!(seq_test.loglik(), anova1.loglik()[1..], epsilon = 1e-6);
    assert_abs_diff_eq!(seq_test.chisq()[1..], anova1.chisq()[2..], epsilon = 1e-6);
    assert_abs_diff_eq!(
        seq_test.p_values()[1..],
        anova1.p_values()[2..],
        epsilon = 1e-6
    );
}
