test_that("survival:: specials in formulas fit as stock R's", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  lung <- survival::lung
  cox <- c("age + survival::strata(sex) + survival::cluster(inst)",
           "survival::pspline(age, df = 2) + sex",
           "survival::ridge(age, sex, theta = 1)",
           "age + survival::frailty(inst)")
  for (rhs in cox) {
    formula <- as.formula(paste("Surv(time, status) ~", rhs))
    actual <- coxph(formula, lung)
    expected <- survival::coxph(formula, lung)
    expect_identical(names(coef(actual)), names(coef(expected)), info = rhs)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-7, info = rhs)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-7, info = rhs)
  }
  actual <- survreg(Surv(time, status) ~ age + survival::strata(sex), lung)
  expected <- survival::survreg(Surv(time, status) ~ age + survival::strata(sex), lung)
  expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-7)
  expect_equal(unname(actual$scale), unname(expected$scale), tolerance = 1e-7)
  times <- c(100, 300, 500)
  actual <- summary(survfit(Surv(time, status) ~ survival::strata(sex), lung), times = times)
  expected <- summary(survival::survfit(Surv(time, status) ~ survival::strata(sex), lung),
    times = times)
  expect_equal(actual$surv, expected$surv, tolerance = 1e-12)
  expect_equal(actual$std.err, expected$std.err, tolerance = 1e-10)
  plain <- summary(survfit(Surv(time, status) ~ strata(sex), lung), times = times)
  expect_identical(as.character(actual$strata), as.character(plain$strata))
  actual <- survdiff(Surv(time, status) ~ sex + survival::strata(ph.ecog), lung)
  expected <- survival::survdiff(Surv(time, status) ~ sex + survival::strata(ph.ecog), lung)
  expect_equal(actual$chisq, expected$chisq, tolerance = 1e-10)
})
