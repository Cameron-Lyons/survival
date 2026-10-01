aft_matrix_data <- function() {
  list(
    x = cbind(Intercept = 1, age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
              group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0)),
    y = cbind(time = c(1.2, 2.5, .9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9),
              status = c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1))
  )
}

aft_matrix_call <- function(fun, args) {
  messages <- character()
  fit <- withCallingHandlers(do.call(fun, args), warning = function(w) {
    messages <<- c(messages, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(fit = fit, warnings = messages)
}

test_that("the compact AFT bridge preserves R fit components and diagnostics", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- aft_matrix_data()
  for (dist in c("extreme", "logistic", "gaussian", "t")) {
    for (kind in c("default", "weighted", "fixed", "strata", "initial", "partial",
                    "zero", "one", "two", "mean", "mean_fixed", "no_intercept",
                    "alias", "binary", "interval", "unnamed")) {
      args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                          controlvals = survival::survreg.control(), dist = dist,
                          parms = if (dist == "t") 4 else NULL))
      if (kind == "weighted") {
        args$weights <- rep(c(.5, 1, 2), 4)
        args$offset <- seq(.1, 1.2, length.out = 12)
      }
      if (kind == "fixed") args$scale <- 1.5
      if (kind == "strata") { args$nstrat <- 2; args$strata <- rep(1:2, 6) }
      if (kind == "initial") args$init <- c(2, 0, 0, .1)
      if (kind == "partial") args$init <- c(2, 0, 0)
      if (kind %in% c("zero", "one", "two")) {
        args$controlvals <- survival::survreg.control(iter.max = match(kind, c("zero", "one", "two")) - 1L)
      }
      if (kind %in% c("mean", "mean_fixed")) args$x <- data$x[, 1L, drop = FALSE]
      if (kind == "mean_fixed") args$scale <- 1
      if (kind == "no_intercept") args$x <- data$x[, 2:3]
      if (kind == "alias") args$x <- cbind(data$x, duplicate = data$x[, 2L])
      if (kind == "binary") args$x <- data$x[, c(1L, 3L)]
      if (kind == "interval") args$y <- cbind(data$y[, 1L], data$y[, 1L] + .4, rep(c(1, 0, 2, 3), 3))
      if (kind == "unnamed") args$x <- unname(data$x)
      actual <- aft_matrix_call(survreg.fit, args)
      expected <- aft_matrix_call(survival::survreg.fit, args)
      expect_equal(actual$warnings, expected$warnings, info = paste(dist, kind))
      expect_equal(names(actual$fit), names(expected$fit))
      for (name in names(expected$fit)) {
        expect_equal(actual$fit[[name]], expected$fit[[name]], tolerance = 2e-7,
                     info = paste(dist, kind, name))
      }
    }
  }
})

test_that("AFT matrix callbacks execute through the shared Python fitter", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- aft_matrix_data()
  control <- survival::survreg.control()
  reference <- survival::survreg.fit(data$x, data$y, NULL, NULL, NULL, control, "gaussian")
  bridge <- survivalr::survreg.fit
  local_mocked_bindings(survreg.fit = function(...) stop("R reference fitter called"), .package = "survival")
  original <- .call_r_api
  used <- character()
  local_mocked_bindings(.call_r_api = function(name, ...) {
    used <<- c(used, name)
    original(name, ...)
  }, .package = "survivalr")
  count <- 0L
  seen <- list()
  distribution <- list(
    name = "Custom Gaussian",
    init = function(y, weights, parms) {
      expect_identical(parms, c(unused = 3))
      survival::survreg.distributions$gaussian$init(y, weights, parms)
    },
    density = function(z, parms) {
      count <<- count + 1L
      seen[[length(seen) + 1L]] <<- length(z)
      expect_identical(parms, c(unused = 3))
      survival::survreg.distributions$gaussian$density(z, parms)
    },
    trans = function(...) stop("unexpected response transform"), scale = 100
  )
  fit <- bridge(data$x, data$y, NULL, NULL, NULL, control, distribution, parms = c(unused = 3))
  expect_equal(fit, reference, tolerance = 2e-7)
  expect_identical(used, "survreg_fit")
  expect_gt(count, 0L)
  expect_true(all(unlist(seen) == nrow(data$x)))
  expect_equal(unserialize(serialize(fit, NULL)), fit)
  # Display names do not override an explicitly supplied callback.
  distribution$name <- "Gaussian"
  count <- 0L
  expect_equal(bridge(data$x, data$y, NULL, NULL, NULL, control, distribution,
                         parms = c(unused = 3)), reference, tolerance = 2e-7)
  expect_gt(count, 0L)
  # A complete intercept-only start needs no initializer or distribution probe.
  distribution$init <- NULL
  expect_length(bridge(data$x[, 1L, drop = FALSE], data$y, NULL, NULL,
                           c(2, .1), control, distribution, parms = c(unused = 3))$coefficients, 2L)
})

test_that("AFT callback intervals and validation do not fall back to R", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- aft_matrix_data()
  control <- survival::survreg.control()
  # Stock R's callback-density interval path has an unsafe C indexing defect;
  # use its equivalent built-in Gaussian path as the reference.
  interval <- cbind(data$y[, 1L], data$y[, 1L] + .4, rep(c(1, 0, 2, 3), 3))
  expected <- survival::survreg.fit(data$x, interval, NULL, NULL, NULL, control, "gaussian")
  custom <- survival::survreg.distributions$gaussian
  custom$name <- "Custom Gaussian"
  actual <- survreg.fit(data$x, interval, NULL, NULL, NULL, control, custom)
  expect_equal(actual, expected, tolerance = 2e-7)
  custom$density <- function(z) matrix(0, length(z), 4)
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, custom), "five-column")
  custom$density <- function(z) stop("density failure detail")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, custom), "density failure detail")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, "t"), "explicit parms")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, "weibull"), "Missing density")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, "unknown"), "Unrecognized")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, list()), "Missing density")
  expect_error(survreg.fit(data$x, data$y, c(0, rep(1, 11)), NULL, NULL, control, "gaussian"), "Invalid weights")
  expect_error(survreg.fit(data$x, data$y, NULL, NULL, NULL, control, "gaussian",
                         nstrat = 2, strata = rep(1.5, 12)), "Invalid strata")
})

test_that("AFT matrix dimensions retain unused scales and singleton vectors", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- aft_matrix_data()
  control <- survival::survreg.control()
  args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                      controlvals = control, dist = "extreme",
                      nstrat = 3, strata = rep(1:2, 6)))
  expect_equal(aft_matrix_call(survreg.fit, args), aft_matrix_call(survival::survreg.fit, args), tolerance = 2e-7)
  args$nstrat <- 1
  args$strata <- c(NA, rep(-5, 11))
  expect_equal(do.call(survreg.fit, args), do.call(survival::survreg.fit, args), tolerance = 2e-7)
  args <- list(x = matrix(1, 1, 1, dimnames = list(NULL, "Intercept")),
               y = matrix(c(2.1, 1), 1, 2), weights = 1, offset = 0, init = 0,
               controlvals = survival::survreg.control(iter.max = 0), dist = "gaussian", scale = 1)
  expect_equal(do.call(survreg.fit, args), do.call(survival::survreg.fit, args), tolerance = 2e-7)
})
