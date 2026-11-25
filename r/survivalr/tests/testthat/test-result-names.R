# Stock fits get survival's own Surv; a bare Surv here is survivalr's.
stock_formula <- function(form) {
  environment(form) <- list2env(list(Surv = survival::Surv), parent = environment(form))
  form
}

test_that("Cox residuals carry R's row names, term columns and constant", {
  lung <- survival::lung
  forms <- list(
    Surv(time, status) ~ age + ph.ecog + strata(sex),
    Surv(time, status) ~ age + sex
  )
  for (form in forms) {
    fit <- coxph(form, data = lung)
    stock <- survival::coxph(stock_formula(form), data = lung)
    for (type in c("martingale", "deviance", "score", "dfbeta", "dfbetas", "partial")) {
      expect_equal(residuals(fit, type = type), residuals(stock, type = type), tolerance = 1e-6)
    }
  }
  # naresid pads the excluded rows, names them, and drops the partial constant.
  fit <- coxph(Surv(time, status) ~ age + ph.ecog, data = lung, na.action = na.exclude)
  stock <- survival::coxph(stock_formula(Surv(time, status) ~ age + ph.ecog), data = lung,
    na.action = na.exclude)
  for (type in c("martingale", "dfbeta", "partial")) {
    expect_equal(residuals(fit, type = type), residuals(stock, type = type), tolerance = 1e-6)
  }
  fit <- coxph(Surv(time, status) ~ age + ph.ecog, data = lung)
  stock <- survival::coxph(stock_formula(Surv(time, status) ~ age + ph.ecog), data = lung)
  expect_equal(names(residuals(fit)), names(residuals(stock)))
})

test_that("survreg residuals carry R's row names", {
  lung <- survival::lung
  fit <- survreg(Surv(time, status) ~ age + sex, data = lung)
  stock <- survival::survreg(stock_formula(Surv(time, status) ~ age + sex), data = lung)
  for (type in c("response", "deviance", "dfbeta", "dfbetas", "working", "ldcase", "matrix")) {
    expect_equal(residuals(fit, type = type), residuals(stock, type = type), tolerance = 1e-6)
  }
  fit <- survreg(Surv(time, status) ~ age + ph.ecog, data = lung, na.action = na.exclude)
  stock <- survival::survreg(stock_formula(Surv(time, status) ~ age + ph.ecog), data = lung,
    na.action = na.exclude)
  expect_equal(residuals(fit, type = "deviance"), residuals(stock, type = "deviance"),
    tolerance = 1e-6)
})

test_that("d/p/qsurvreg keep the names R's arithmetic gives", {
  expect_equal(psurvreg(c(a = 1, b = 2), 0, 1), survival::psurvreg(c(a = 1, b = 2), 0, 1))
  expect_equal(
    qsurvreg(c(lo = 0.1, hi = 0.9), 0, 1, "lognormal"),
    survival::qsurvreg(c(lo = 0.1, hi = 0.9), 0, 1, "lognormal")
  )
  expect_equal(dsurvreg(1:2, c(m1 = 0, m2 = 1), 1),
    survival::dsurvreg(1:2, c(m1 = 0, m2 = 1), 1))
  expect_null(names(dsurvreg(1:2, 0, 1)))
})

test_that("cipoisson names and shapes its result as R does", {
  calls <- list(
    quote(cipoisson(c(a = 3, b = 4))),
    quote(cipoisson(c(a = 3))),
    quote(cipoisson(c(a = 3), method = "anscombe")),
    quote(cipoisson(3, c(x = 1, y = 2))),
    quote(cipoisson(c(1, 2), c(0, 1))),
    quote(cipoisson(c(1, NA, 3))),
    quote(cipoisson(matrix(1:4, 2)))
  )
  for (call in calls) {
    expect_equal(eval(call), eval(call, list(cipoisson = survival::cipoisson)))
  }
  # Anscombe's interval at k = 0 is NaN, with R's warning.
  expect_warning(
    result <- cipoisson(c(0, 3), c(2, 2), method = "anscombe"),
    "NaNs produced"
  )
  expect_equal(result,
    suppressWarnings(survival::cipoisson(c(0, 3), c(2, 2), method = "anscombe")))
})

test_that("Cox and survreg accessors have R's shapes", {
  lung <- survival::lung
  fit <- coxph(Surv(time, status) ~ age + sex + cluster(inst), data = lung)
  stock <- survival::coxph(stock_formula(Surv(time, status) ~ age + sex + cluster(inst)), data = lung)
  expect_equal(extractAIC(fit), extractAIC(stock))
  expect_equal(summary(fit)$robscore, summary(stock)$robscore)
  expect_null(df.residual(fit))
  aft <- survreg(Surv(time, status) ~ age + sex, data = lung)
  expect_equal(extractAIC(aft),
    extractAIC(survival::survreg(stock_formula(Surv(time, status) ~ age + sex), data = lung)))
  censored <- lung
  censored$status <- 0
  expect_equal(
    suppressWarnings(coef(coxph(Surv(time, status) ~ age + sex, data = censored))),
    suppressWarnings(coef(survival::coxph(stock_formula(Surv(time, status) ~ age + sex), data = censored)))
  )
})
