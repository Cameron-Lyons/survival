test_that("summary.coxph scale vectors and init-aware Wald tests match stock R", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  lung <- survival::lung
  for (args in list(list(), list(init = c(0.02, -0.3)), list(init = c(0.02, -0.3), iter.max = 0),
                    list(robust = TRUE, init = c(0.02, -0.3)))) {
    actual <- do.call(coxph, c(list(Surv(time, status) ~ age + sex, data = lung), args))
    expected <- do.call(survival::coxph, c(list(Surv(time, status) ~ age + sex, data = lung), args))
    expect_equal(unname(summary(actual)$waldtest), unname(summary(expected)$waldtest),
      tolerance = 1e-8, info = paste(names(args), collapse = "+"))
  }
  actual <- coxph(Surv(time, status) ~ age + sex + ph.ecog, lung)
  expected <- survival::coxph(Surv(time, status) ~ age + sex + ph.ecog, lung)
  for (scale in list(c(10, 1, 1), c(10, 2))) {
    expect_equal(summary(actual, scale = scale)$coefficients,
      suppressWarnings(summary(expected, scale = scale)$coefficients), tolerance = 1e-8)
    expect_equal(summary(actual, scale = scale)$conf.int,
      suppressWarnings(summary(expected, scale = scale)$conf.int), tolerance = 1e-8)
  }
  actual <- coxph(Surv(time, status) ~ age + sex, lung, robust = TRUE)
  expected <- survival::coxph(Surv(time, status) ~ age + sex, lung, robust = TRUE)
  expect_equal(summary(actual, scale = c(10, 1))$coefficients,
    summary(expected, scale = c(10, 1))$coefficients, tolerance = 1e-8)
})
