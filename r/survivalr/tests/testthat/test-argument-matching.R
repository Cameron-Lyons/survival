zph_chisq <- function(zph) vapply(zph$table, function(row) row$chisq, numeric(1))

test_that("coxph and survreg match partial argument names as R does", {
  lung <- survival::lung
  # coxph's own formals, then coxph.control's names for the rest of ...
  expect_equal(
    unname(coef(coxph(Surv(time, status) ~ age, lung, meth = "breslow"))),
    unname(coef(survival::coxph(survival::Surv(time, status) ~ age, lung, meth = "breslow")))
  )
  expect_warning(
    fit <- coxph(Surv(time, status) ~ age + sex, lung, iter = 2),
    "Ran out of iterations"
  )
  stock <- suppressWarnings(
    survival::coxph(survival::Surv(time, status) ~ age + sex, lung, iter = 2)
  )
  expect_equal(unname(coef(fit)), unname(coef(stock)))
  expect_error(coxph(Surv(time, status) ~ age, lung, bogus = 1), "not matched")
  # survreg's own formals, then survreg.control's.
  expect_equal(
    unname(coef(survreg(Surv(time, status) ~ age, lung, di = "lognormal"))),
    unname(coef(survival::survreg(survival::Surv(time, status) ~ age, lung, di = "lognormal")))
  )
  expect_equal(
    unname(suppressWarnings(coef(survreg(Surv(time, status) ~ age, lung, maxit = 1)))),
    unname(suppressWarnings(coef(
      survival::survreg(survival::Surv(time, status) ~ age, lung, maxit = 1)
    )))
  )
})

test_that("brier takes a single time", {
  # Stock brier re-evaluates fit$call$data, so the data is named where it can find it.
  fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, survival::lung)
  stock <- survival::coxph(survival::Surv(time, status) ~ age + sex + ph.ecog, survival::lung)
  expected <- survival::brier(stock, times = 300)
  for (times in list(300, 300L)) {
    result <- brier(fit, times = times)
    expect_equal(result$brier, expected$brier)
    expect_equal(result$rsquared, expected$rsquared)
  }
})

test_that("concordance matches timewt partially and ignores other arguments", {
  lung <- survival::lung
  stock <- function(...) survival::concordance(survival::Surv(time, status) ~ ph.karno, lung, ...)
  expect_equal(
    concordance(Surv(time, status) ~ ph.karno, lung, timewt = "n/G")$concordance,
    stock(timewt = "n/G")$concordance
  )
  expect_equal(
    concordance(Surv(time, status) ~ ph.karno, lung, timew = "S")$concordance,
    stock(timewt = "S")$concordance
  )
  result <- concordance(Surv(time, status) ~ ph.karno, lung, std.err = FALSE)
  expect_equal(result$concordance, stock()$concordance)
  expect_equal(result$var, stock(std.err = FALSE)$var)
})

test_that("survfit0 ignores its other arguments", {
  heart <- survival::heart
  fit <- survfit(Surv(start, stop, event) ~ transplant, data = heart, id = id)
  expect_equal(
    summary(survfit0(fit, start.time = -5), times = 10)$surv,
    summary(survfit0(fit), times = 10)$surv
  )
})

test_that("cox.zph takes any R function as its transform", {
  lung <- survival::lung
  fit <- coxph(Surv(time, status) ~ age + sex, data = lung)
  stock <- survival::coxph(survival::Surv(time, status) ~ age + sex, data = lung)
  for (transform in list(sqrt, log, log1p)) {
    expect_equal(
      unname(zph_chisq(cox.zph(fit, transform = transform))),
      unname(survival::cox.zph(stock, transform = transform)$table[, "chisq"])
    )
  }
  expect_equal(cox.zph(fit, transform = sqrt)$transform, "sqrt")
  expect_equal(
    cox.zph(fit, transform = function(t) sqrt(t))$transform,
    survival::cox.zph(stock, transform = function(t) sqrt(t))$transform
  )
})
