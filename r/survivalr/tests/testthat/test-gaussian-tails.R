test_that("Gaussian distribution helpers preserve small R tail probabilities", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  z <- c(-30, -20, -10, -8, -1, 0, 1, 8)
  probabilities <- c(.Machine$double.xmin * .Machine$double.eps,
                     1e-300, 1e-30, 1e-12, 0.025, 0.5, 0.975, 1 - 2^-53)
  for (distribution in c("gaussian", "lognormal")) {
    x <- if (distribution == "gaussian") z else exp(z)
    expected <- survival::psurvreg(x, mean = 0, distribution = distribution)
    actual <- psurvreg(x, mean = 0, distribution = distribution)
    # Ratios make this a relative check even for probabilities near 1e-198.
    expect_equal(actual / expected, rep(1, length(x)), tolerance = 3e-12)
    expect_true(all(actual > 0))

    expected_quantiles <- survival::qsurvreg(
      probabilities, mean = 0.4, scale = 1.25, distribution = distribution
    )
    actual_quantiles <- qsurvreg(
      probabilities, mean = 0.4, scale = 1.25, distribution = distribution
    )
    expect_equal(actual_quantiles / expected_quantiles,
                 rep(1, length(probabilities)), tolerance = 3e-12)
    expect_true(all(is.finite(actual_quantiles)))
  }
})

test_that("Gaussian AFT tail likelihood and score match R at prescribed parameters", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(time = c(39, 40, 41, 50), status = c(1, 1, 1, 0))
  bridged <- survreg(
    Surv(time, status) ~ 1, data = data, dist = "gaussian",
    init = c(40, 0), score = TRUE, max_iter = 0
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ 1, data = data, dist = "gaussian",
    init = c(40, 0), score = TRUE,
    control = survival::survreg.control(maxiter = 0)
  )
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 1e-14)
  expect_equal(as.numeric(logLik(bridged)), reference$loglik[[2L]], tolerance = 2e-13)
  expect_equal(unlist(bridged$score_vector), reference$score, tolerance = 2e-11)
  expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 5e-10)
})

test_that("model summaries retain small two-sided Gaussian p-values", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(time = c(8, 9, 10), status = c(1, 1, 1))
  bridged <- survreg(Surv(time, status) ~ 1, data = data, dist = "gaussian",
                    init = c(9, 0), max_iter = 0)
  reference <- survival::survreg(
    survival::Surv(time, status) ~ 1, data = data, dist = "gaussian",
    init = c(9, 0), control = survival::survreg.control(maxiter = 0)
  )
  actual_p <- summary(bridged)$table["(Intercept)", "p"]
  reference_p <- summary(reference)$table["(Intercept)", "p"]
  expect_gt(actual_p, 0)
  expect_equal(actual_p / reference_p, 1, tolerance = 3e-12)
})
