require_survreg_rank_bridge <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
}

test_that("survreg aliases match R while preserving stored linear predictors", {
  require_survreg_rank_bridge()
  data <- data.frame(time = 1:6, status = rep(1, 6), x = -2:3, z = -2:3)
  for (iterations in c(0, 20)) {
    for (initial in list(c(0, 0, 0), c(0, 0, 2))) {
      bridged <- survreg(Surv(time, status) ~ x + z, data = data,
                         dist = "gaussian", scale = 1, init = initial,
                         max_iter = iterations, score = TRUE)
      reference <- survival::survreg(
        survival::Surv(time, status) ~ x + z, data = data,
        dist = "gaussian", scale = 1, init = initial, score = TRUE,
        control = survival::survreg.control(maxiter = iterations)
      )
      expect_equal(coef(bridged), coef(reference), tolerance = 1e-10)
      expect_identical(unname(coef(bridged)[[3L]]), NA_real_)
      expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-10)
      expect_equal(vcov(bridged, complete = FALSE), vcov(reference, complete = FALSE),
                   tolerance = 1e-10)
      expect_equal(as.numeric(logLik(bridged)), as.numeric(logLik(reference)), tolerance = 1e-10)
      expect_equal(unlist(bridged$score_vector), reference$score, tolerance = 1e-10)
      expect_equal(unname(predict(bridged, type = "lp")),
                   unname(predict(reference, type = "lp")), tolerance = 1e-10)
      expect_equal(unname(predict(bridged, newdata = data, type = "lp")),
                   unname(predict(reference, newdata = data, type = "lp")))
      expect_false(any(is.nan(predict(bridged, newdata = data, type = "lp"))))
      expect_equal(summary(bridged)$table, summary(reference)$table, tolerance = 1e-9)
      expect_false(any(is.nan(summary(bridged)$table["z", ])))
      expect_equal(confint(bridged), confint(reference), tolerance = 1e-8)
      expect_false(any(is.nan(confint(bridged)["z", ])))
      actual_terms <- predict(bridged, type = "terms", se.fit = TRUE)
      reference_terms <- predict(reference, type = "terms", se.fit = TRUE)
      expect_equal(unname(actual_terms$fit), unname(reference_terms$fit), tolerance = 1e-9)
      expect_equal(unname(actual_terms$se.fit), unname(reference_terms$se.fit), tolerance = 1e-9)
      expect_false(any(is.nan(actual_terms$fit[, "z"])))
      expect_equal(unname(residuals(bridged, type = "response")),
                   unname(residuals(reference, type = "response")), tolerance = 1e-10)
      expect_equal(unname(residuals(bridged, type = "dfbeta")[, 3L]), rep(0, 6))
      expect_true(all(is.nan(residuals(bridged, type = "dfbetas")[, 3L])))
      expect_equal(attr(logLik(bridged), "df"), attr(logLik(reference), "df"))
    }
  }
})

test_that("survreg absolute rank tolerance controls generalized covariance", {
  require_survreg_rank_bridge()
  data <- data.frame(time = 1:3, status = rep(1, 3))
  for (tolerance in c(1e-10, 1e-14)) {
    bridged <- survreg(Surv(time, status) ~ 1, data = data,
                       dist = "gaussian", scale = 1e6, init = 2, max_iter = 0,
                       tol_chol = tolerance)
    reference <- survival::survreg(
      survival::Surv(time, status) ~ 1, data = data,
      dist = "gaussian", scale = 1e6, init = 2,
      control = survival::survreg.control(maxiter = 0, toler.chol = tolerance)
    )
    expect_equal(coef(bridged), coef(reference))
    expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
    expect_equal(vcov(bridged, complete = FALSE), vcov(reference, complete = FALSE),
                 tolerance = 1e-12)
    if (tolerance == 1e-10) {
      expect_warning(result <- withVisible(summary(bridged)), "zero rank")
      expect_false(result$visible)
      expect_identical(result$value, bridged)
    }
  }
})

test_that("survreg scale covariance stays available with and without location aliases", {
  require_survreg_rank_bridge()
  data <- data.frame(
    time = c(1.3, 2.1, 1.8, 3.4, 2.9, 4.5, 3.8, 5.2), status = rep(1, 8),
    x = c(-1.2, -0.5, 0.3, 1.1, -0.8, 0.6, 1.4, 0.2),
    group = rep(c("a", "b"), 4)
  )
  data$z <- data$x
  for (rhs in c("x", "x + z", "x + z + strata(group)")) {
    initial <- if (rhs == "x") c(2, 0.5, log(1.2)) else
      if (rhs == "x + z") c(2, 0.2, 0.3, log(1.2)) else
        c(2, 0.2, 0.3, log(1.2), log(0.9))
    bridged <- survreg(as.formula(paste("Surv(time, status) ~", rhs)),
                       data = data, dist = "gaussian", init = initial, max_iter = 0)
    reference <- survival::survreg(
      as.formula(paste("survival::Surv(time, status) ~", rhs)),
      data = data, dist = "gaussian", init = initial,
      control = survival::survreg.control(maxiter = 0)
    )
    expect_equal(coef(bridged), coef(reference), tolerance = 1e-10)
    expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 1e-9)
    expect_equal(unname(vcov(bridged, complete = FALSE)),
                 unname(vcov(reference, complete = FALSE)), tolerance = 1e-9)
    expect_equal(unname(summary(bridged)$table), unname(summary(reference)$table),
                 tolerance = 1e-8)
    expect_equal(ncol(vcov(bridged)), length(initial))
  }
})

test_that("survreg estimated scales keep summaries available when all locations alias", {
  require_survreg_rank_bridge()
  data <- data.frame(time = 1:4, status = rep(1, 4), x = 0)
  bridged <- survreg(Surv(time, status) ~ 0 + x, data = data,
                     dist = "gaussian", init = c(0, 0), max_iter = 0)
  reference <- survival::survreg(
    survival::Surv(time, status) ~ 0 + x, data = data,
    dist = "gaussian", init = c(0, 0),
    control = survival::survreg.control(maxiter = 0)
  )
  expect_identical(unname(coef(bridged)), NA_real_)
  expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
  expect_no_warning(actual <- withVisible(summary(bridged)))
  expect_true(actual$visible)
  expect_equal(actual$value$table, summary(reference)$table, tolerance = 1e-12)
  expect_identical(unname(actual$value$table["Log(scale)", "Value"]), 0)
})
