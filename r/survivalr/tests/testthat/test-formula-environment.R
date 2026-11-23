test_that("formula variables come from data, then from the formula's environment", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  lung <- survival::lung
  tm <- lung$time; st <- lung$status; sx <- lung$sex; ag <- lung$age
  times <- c(100, 300)
  expect_equal(summary(survfit(Surv(tm, st) ~ sx), times = times)$surv,
    summary(survival::survfit(survival::Surv(tm, st) ~ sx), times = times)$surv, tolerance = 1e-10)
  expect_equal(summary(survfit("Surv(tm, st) ~ sx"), times = times)$surv,
    summary(survival::survfit(survival::Surv(tm, st) ~ sx), times = times)$surv, tolerance = 1e-10)
  expect_equal(unname(coef(coxph(Surv(tm, st) ~ ag + sx))),
    unname(coef(survival::coxph(survival::Surv(tm, st) ~ ag + sx))), tolerance = 1e-8)
  expect_equal(unname(coef(survreg(Surv(tm, st) ~ ag))),
    unname(coef(survival::survreg(survival::Surv(tm, st) ~ ag))), tolerance = 1e-8)
  expect_equal(concordance(Surv(tm, st) ~ ag)$concordance,
    survival::concordance(survival::Surv(tm, st) ~ ag)$concordance, tolerance = 1e-10)
  # A vector from the environment beside a data frame, including an offset.
  expect_equal(survdiff(Surv(time, status) ~ sx, lung)$chisq,
    survival::survdiff(Surv(time, status) ~ sx, lung)$chisq, tolerance = 1e-10)
  ex <- 1 - exp(-lung$time / 400)
  expect_equal(survdiff(Surv(time, status) ~ offset(ex), lung)$chisq,
    survival::survdiff(Surv(time, status) ~ offset(ex), lung)$chisq, tolerance = 1e-10)
  # Variables local to a calling function.
  inside <- function(fit) { t2 <- lung$time; s2 <- lung$status; fit(Surv(t2, s2) ~ 1) }
  expect_equal(summary(inside(survfit), times = 200)$surv,
    summary(inside(survival::survfit), times = 200)$surv, tolerance = 1e-10)
  # A scalar recycles as R's arithmetic does.
  k <- 3
  expect_equal(unname(coef(coxph(Surv(time, status) ~ I(age / k), lung))),
    unname(coef(survival::coxph(Surv(time, status) ~ I(age / k), lung))), tolerance = 1e-8)
  # A data column wins over an environment variable of the same name.
  age <- rep(1, nrow(lung))
  expect_equal(unname(coef(coxph(Surv(time, status) ~ age, lung))),
    unname(coef(survival::coxph(Surv(time, status) ~ age, lung))), tolerance = 1e-8)
})
