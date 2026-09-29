test_that("direct Cox curves preserve vector, matrix and list metadata", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  args <- list(ctype = 2L, stype = 2L, se.fit = TRUE, varmat = matrix(.2),
               y = cbind(c(1, 2, 3), c(1, 1, 0)), x = cbind(c(0, 1, 2)),
               wt = NULL, risk = rep(1, 3), strata = NULL,
               x2 = rbind(first = .5, second = 1.5), risk2 = c(low = 1, high = 2))
  for (grouped in c(FALSE, TRUE)) for (se in c(FALSE, TRUE)) {
    for (multiple in c(FALSE, TRUE)) for (flat in c(FALSE, TRUE)) {
      current <- args
      current$strata <- if (grouped) factor(c("b", "a", "b"), levels = c("b", "a")) else NULL
      current$se.fit <- se
      current$unlist <- flat
      if (!multiple) {
        current$x2 <- current$x2[1L, , drop = FALSE]
        current$risk2 <- current$risk2[1L]
      }
      expect_equal(do.call(coxsurv.fit, current), do.call(survival::coxsurv.fit, current),
                   tolerance = 1e-11)
    }
  }
  args$x2 <- unname(args$x2)
  expect_equal(do.call(coxsurv.fit, args), do.call(survival::coxsurv.fit, args), tolerance = 1e-11)
})

test_that("direct Cox curves accept single rows and omit unused arguments", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  args <- list(ctype = 1L, stype = 2L, se.fit = TRUE, varmat = matrix(.2),
               y = cbind(1, 1), x = cbind(0), wt = NULL, risk = 1,
               x2 = matrix(.5), risk2 = 1.2, strata = 1)
  expect_equal(do.call(coxsurv.fit, args), do.call(survival::coxsurv.fit, args), tolerance = 1e-11)
  actual <- coxsurv.fit(1L, 2L, FALSE, y = args$y, x = args$x, risk = 1,
                        x2 = args$x2, risk2 = 1.2,
                        cluster = stop("unused"), oldid = stop("unused"), position = stop("unused"))
  expected <- survival::coxsurv.fit(1L, 2L, FALSE, y = args$y, x = args$x, wt = NULL,
                                    risk = 1, x2 = args$x2, risk2 = 1.2)
  expect_equal(actual, expected, tolerance = 1e-11)
  expect_error(survfitcoxph.fit(y = args$y, x = args$x, wt = NULL, x2 = args$x2,
                               risk = 1, se.fit = FALSE), "newrisk")
})

test_that("direct Cox curves support empty levels, zero covariates and empty trajectories", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  y <- cbind(c(1, 2, 3), c(1, 1, 0))
  x <- matrix(numeric(), 3, 0)
  x2 <- matrix(numeric(), 1, 0)
  strata <- factor(rep("a", 3), levels = c("empty", "a"))
  fit <- coxsurv.fit(1L, 2L, TRUE, matrix(numeric(), 0, 0),
                     y = y, x = x, wt = NULL, risk = rep(1, 3), strata = strata,
                     x2 = x2, risk2 = 2)
  expect_equal(fit$n, c(0L, 3L))
  expect_equal(fit$strata, c(empty = 0L, a = 3L))
  expect_equal(fit$std.err[3], 2 * sqrt(1/9 + 1/4))
  empty <- coxsurv.fit(1L, 2L, TRUE, matrix(numeric(), 0, 0),
                       y = y, x = x, risk = rep(1, 3), strata = strata,
                       x2 = x2, risk2 = 2, y2 = cbind(7, 8), strata2 = 1L, id2 = "only")
  expect_equal(empty$n, 0L)
  expect_equal(empty$time, numeric())
  expect_equal(empty$surv, numeric())
  expect_equal(empty$std.err, numeric())
})

test_that("individual curves retain names from single-time interval risks", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  args <- list(ctype = 1L, stype = 2L, se.fit = TRUE, varmat = matrix(.2),
               y = cbind(c(1, 2, 3), c(1, 1, 0)), x = cbind(c(0, 1, 2)),
               wt = NULL, risk = rep(1, 3), strata = NULL,
               x2 = cbind(c(.5, .7, .3)), risk2 = c(single = 1.2, double = .8, empty = 1),
               y2 = rbind(c(0, 1), c(1, 3), c(3, 4)), strata2 = rep(1L, 3),
               id2 = rep("subject", 3))
  for (se in c(FALSE, TRUE)) for (flat in c(FALSE, TRUE)) {
    args$se.fit <- se
    args$unlist <- flat
    expect_equal(do.call(coxsurv.fit, args), do.call(survival::coxsurv.fit, args), tolerance = 1e-11)
  }
})
