test_that("namespaced calls in formulas fit and name as stock R's", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  lung <- survival::lung
  for (rhs in c("splines::ns(age, df = 2) + sex", "stats::poly(age, 2) + sex",
                "survival::nsk(age, df = 3) + sex", "base::log(age) + sex",
                "splines::bs(age, df = 3) + sex", "splines::ns(age, df = 2):sex",
                "sex + splines:::ns(age, df = 2)")) {
    formula <- as.formula(paste("Surv(time, status) ~", rhs))
    actual <- coxph(formula, lung)
    expected <- survival::coxph(formula, lung)
    expect_identical(names(coef(actual)), names(coef(expected)), info = rhs)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-9, info = rhs)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-9, info = rhs)
  }
})
