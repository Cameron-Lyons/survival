test_that("prepared Yates prediction preserves R values and names", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  fit <- survival::coxph(survival::Surv(time, status) ~ age, survival::lung)
  for (horizon in c(0, 100, Inf)) {
    setup <- yates_setup(fit, "survival", options = list(rmean = horizon))
    reference <- survival::yates_setup(fit, "survival", options = list(rmean = horizon))
    for (eta in list(0, c(a = -1, b = .5), matrix(c(-1, .5), ncol = 1,
                     dimnames = list(c("a", "b"), NULL)), matrix(c(-1, .5), nrow = 1))) {
      expect_equal(setup$predict(eta), reference$predict(eta), tolerance = 1e-11)
    }
    expect_error(setup$predict(matrix(1, 2, 2)), "vector or single-row/column")
  }
  own_fit <- coxph(Surv(time, status) ~ age, survival::lung)
  own_setup <- yates_setup(own_fit, "survival", options = list(rmean = 100))
  reference <- survival::yates_setup(fit, "survival", options = list(rmean = 100))
  expect_equal(own_setup$predict(c(-1, 0, .5)), reference$predict(c(-1, 0, .5)), tolerance = 1e-10)
})

test_that("Yates summary aligns baseline times and derived curve uncertainty", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (kind in c("ordinary", "single", "empty")) for (nlevel in c(1L, 3L)) {
    data <- switch(kind,
      ordinary = survival::lung,
      single = data.frame(time = c(2, 2, 2), status = c(1, 1, 0)),
      empty = data.frame(time = c(1, 2, 3), status = c(0, 0, 0)))
    fit <- survival::coxph(survival::Surv(time, status) ~ 1, data)
    setup <- yates_setup(fit, "survival")
    mean <- setup$predict(seq_len(nlevel) / 10)
    variance <- matrix(.0001, nrow(mean), ncol(mean))
    result <- setup$summary(mean, variance)
    expected <- t(mean[, -c(1L, 2L), drop = FALSE])
    std <- t(sqrt(variance[, -c(1L, 2L), drop = FALSE]))
    expect_equal(result$surv, expected, tolerance = 1e-12)
    expect_equal(result$cumhaz, -log(expected), tolerance = 1e-12)
    expect_equal(result$std.err, std / expected, tolerance = 1e-12)
    expect_equal(result$std.chaz, result$std.err)
    expect_equal(nrow(result$surv), length(result$time))
    expect_equal(ncol(result$surv), nlevel)
    expect_equal(setup$summary(mean, variance), result)
  }
})
