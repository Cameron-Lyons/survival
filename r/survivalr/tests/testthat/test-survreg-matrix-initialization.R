.require_survreg_matrix_initialization <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
}

.survreg_matrix_initialization_data <- function() {
  i <- 0:15
  lower <- 1.1 + ((i * 7) %% 17) * 0.22 + i * 0.03
  list(
    x = cbind(`(Intercept)` = 1,
              x = ((i * 5) %% 13 - 6) * 0.19,
              z = ((i * 3) %% 11 - 5) * 0.17),
    y = cbind(time = lower, time2 = lower + 0.15 + (i %% 3) * 0.2,
              status = rep(c(1, 0, 2, 3), 4)),
    weights = 0.6 + (i %% 5) * 0.27,
    offset = ((i %% 4) - 1.5) * 0.09,
    strata = rep(1:2, 8)
  )
}

.compare_survreg_matrix_initialization <- function(arguments, label) {
  actual <- do.call(survreg.fit, arguments)
  expected <- do.call(survival::survreg.fit, arguments)
  # Mixed Gaussian censoring retains the existing approximate normal CDF;
  # the other kernels and complete-event Gaussian references are tighter.
  tolerance <- if (identical(arguments$dist, "gaussian") &&
                    any(arguments$y[, ncol(arguments$y)] != 1)) 3e-5 else 3e-7
  expect_identical(names(actual), names(expected), info = label)
  for (field in c("coefficients", "icoef", "var", "loglik", "linear.predictors", "score")) {
    expect_equal(actual[[field]], expected[[field]], tolerance = tolerance,
                 info = paste(label, field))
  }
  expect_identical(actual$df, expected$df, info = label)
  if (arguments$controlvals$iter.max == 0L) {
    expect_identical(actual$iter, 0L, info = label)
  }
  actual
}

test_that("survreg.fit derives location-only scale starts from the native null fit", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  for (distribution in c("extreme", "logistic", "gaussian", "t")) {
    for (stratified in c(FALSE, TRUE)) {
      for (intercept in c(FALSE, TRUE)) {
        x <- if (intercept) data$x else data$x[, -1L, drop = FALSE]
        if (!intercept) x[, 1L] <- x[, 1L] + 2
        initial <- if (intercept) c(2.8, 0.1, -0.2) else c(1.3, -0.2)
        for (maxiter in c(0L, 200L)) {
          arguments <- list(
            x = x, y = data$y, weights = data$weights, offset = data$offset,
            init = initial, dist = distribution, parms = if (distribution == "t") 5 else NULL,
            nstrat = if (stratified) 2L else 1L, strata = data$strata,
            controlvals = survival::survreg.control(maxiter = maxiter, rel.tolerance = 1e-10)
          )
          label <- paste(distribution, stratified, intercept, maxiter)
          actual <- .compare_survreg_matrix_initialization(arguments, label)
          if (maxiter == 0L) {
            expect_identical(unname(head(actual$coefficients, ncol(x))), initial,
                             info = label)
          }
          arguments$init <- c(initial, unname(actual$icoef[-1L]))
          expanded <- do.call(survreg.fit, arguments)
          expect_identical(actual, expanded, info = paste(label, "expanded full start"))
        }
      }
    }
  }
})

test_that("survreg.fit omitted and full starts preserve R coordinates and null metadata", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  for (distribution in c("extreme", "logistic", "gaussian", "t")) {
    for (fixed in c(FALSE, TRUE)) {
      for (maxiter in c(0L, 200L)) {
        for (initial in list(NULL, c(2.8, 0.1, -0.2,
                                   if (fixed) numeric() else log(1.1)))) {
          arguments <- list(
            x = data$x, y = data$y, weights = data$weights, offset = data$offset,
            init = initial, dist = distribution, parms = if (distribution == "t") 5 else NULL,
            scale = if (fixed) 1.1 else 0,
            controlvals = survival::survreg.control(maxiter = maxiter, rel.tolerance = 1e-10)
          )
          .compare_survreg_matrix_initialization(
            arguments, paste(distribution, fixed, maxiter, is.null(initial))
          )
        }
      }
    }
  }
})

test_that("survreg.fit mean-only metadata comes from the main fit", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  for (mode in c("fixed", "estimated", "stratified")) {
    for (maxiter in c(0L, 200L)) {
      nscale <- switch(mode, fixed = 0L, estimated = 1L, stratified = 2L)
      for (initial in list(NULL, c(1.2, log(c(0.9, 1.4)[seq_len(nscale)])))) {
        arguments <- list(
          x = data$x[, 1L, drop = FALSE], y = data$y,
          weights = data$weights, offset = data$offset,
          init = initial, dist = "logistic", scale = if (mode == "fixed") 1.1 else 0,
          nstrat = max(1L, nscale), strata = data$strata,
          controlvals = survival::survreg.control(maxiter = maxiter, rel.tolerance = 1e-10)
        )
        actual <- .compare_survreg_matrix_initialization(
          arguments, paste(mode, maxiter, is.null(initial))
        )
        expect_identical(actual$icoef, actual$coefficients)
        expect_identical(actual$loglik[[1L]], actual$loglik[[2L]])
      }
    }
  }
})

test_that("survreg.fit validates partial starts before fitting", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  arguments <- list(
    x = data$x, y = data$y, weights = data$weights, offset = data$offset,
    dist = "logistic", controlvals = survival::survreg.control(maxiter = 0)
  )
  for (initial in list(numeric(), c(1, 2), rep(0, 5))) {
    arguments$init <- initial
    expect_error(do.call(survreg.fit, arguments), "Wrong length for initial parameters")
  }
  for (initial in list(c(0, NA, 0), c(0, Inf, 0), c(0, 0, 0, NaN))) {
    arguments$init <- initial
    expect_error(do.call(survreg.fit, arguments), "only finite values")
  }
  arguments$x <- data$x[, 1L, drop = FALSE]
  arguments$init <- 1.2
  for (nstrat in 1:2) {
    arguments$nstrat <- nstrat
    arguments$strata <- data$strata
    expect_error(do.call(survreg.fit, arguments), "complete initial vector including log-scale")
  }
  arguments$nstrat <- 1L
  arguments$scale <- 1.1
  expect_identical(unname(do.call(survreg.fit, arguments)$coefficients), 1.2)
})

test_that("survreg.fit retains custom distribution fallback", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  distribution <- survival::survreg.distributions$logistic
  distribution$name <- "Custom logistic"
  arguments <- list(
    x = data$x, y = data$y, weights = data$weights, offset = data$offset,
    init = c(1.2, 0.1, -0.2), dist = distribution,
    controlvals = survival::survreg.control(maxiter = 0)
  )
  expect_identical(do.call(survreg.fit, arguments), do.call(survival::survreg.fit, arguments))
})

test_that("difficult Gaussian matrix starts retain the existing tail accuracy", {
  .require_survreg_matrix_initialization()
  data <- .survreg_matrix_initialization_data()
  # At these poor prescribed locations, the existing normal CDF approximation
  # affects density ratios, scores, and curvature. Bounds below were measured
  # against R at identical full parameter vectors with the preceding kernel,
  # including the smaller difference in preliminary scale-fit termination.
  # Keep these trials separate from the tight initialization parity checks.
  limits <- list(
    estimated_no_intercept = c(var = 2.9e-4, loglik = 3.4e-4, score = 4e-3),
    estimated_intercept = c(var = 1.1e-5, loglik = 1.9e-5, score = 1.3e-4),
    stratified_no_intercept = c(var = 8.1e-4, loglik = 5e-4, score = 6.3e-3),
    stratified_intercept = c(var = 6.5e-6, loglik = 1.2e-5, score = 9.2e-5),
    fixed_intercept = c(var = 2.4e-5, loglik = 4.2e-5, score = 1.3e-4)
  )
  for (mode in c("estimated", "stratified", "fixed")) {
    for (intercept in c(FALSE, TRUE)) {
      if (mode == "fixed" && !intercept) next
      initial <- if (intercept) c(1.2, 0.1, -0.2) else c(0.1, -0.2)
      arguments <- list(
        x = if (intercept) data$x else data$x[, -1L, drop = FALSE],
        y = data$y, weights = data$weights, offset = data$offset,
        init = initial, dist = "gaussian", scale = if (mode == "fixed") 1.1 else 0,
        nstrat = if (mode == "stratified") 2L else 1L, strata = data$strata,
        controlvals = survival::survreg.control(maxiter = 0, rel.tolerance = 1e-10)
      )
      label <- paste(mode, if (intercept) "intercept" else "no_intercept", sep = "_")
      actual <- do.call(survreg.fit, arguments)
      expected <- do.call(survival::survreg.fit, arguments)
      expect_identical(unname(head(actual$coefficients, length(initial))), initial)
      expect_lt(max(abs(actual$coefficients - expected$coefficients)), 1e-7, label = label)
      expect_lt(max(abs(actual$icoef - expected$icoef)), 1e-7, label = label)
      expect_lt(max(abs(actual$linear.predictors - expected$linear.predictors)), 5e-15,
                label = label)
      if (mode != "fixed") {
        arguments$init <- c(initial, unname(actual$icoef[-1L]))
        expect_identical(actual, do.call(survreg.fit, arguments), info = label)
        # Evaluate at R's exact starting scales as well: this removes scale
        # estimation from the measured inherited Gaussian error.
        arguments$init <- c(initial, unname(expected$icoef[-1L]))
      }
      point <- do.call(survreg.fit, arguments)
      expect_identical(unname(point$coefficients), unname(expected$coefficients), info = label)
      for (field in names(limits[[label]])) {
        expect_lt(max(abs(actual[[field]] - expected[[field]])), limits[[label]][[field]],
                  label = paste(label, field))
        expect_lt(max(abs(point[[field]] - expected[[field]])), limits[[label]][[field]],
                  label = paste(label, "exact parameters", field))
      }
    }
  }
})
