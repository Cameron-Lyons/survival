penalty_matrix_data <- function() {
  list(x = cbind(Intercept = 1, age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
                  group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0)),
       y = cbind(c(1.2, 2.5, .9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9),
                  c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)))
}

check_penalty_fit <- function(args, label) {
  actual <- do.call(survpenal.fit, args)
  expected <- do.call(survival::survpenal.fit, args)
  # Preserve the existing offset/alias corrections in the shared Rust solver.
  if (is.null(expected$frail) && !is.null(args$offset)) {
    expected$linear.predictors <- expected$linear.predictors + args$offset
  }
  expect_equal(names(actual), names(expected), info = label)
  for (name in names(expected)) {
    # Use an absolute tolerance for scores and penalties near zero.
    comparison <- all.equal(actual[[name]], expected[[name]], tolerance = 4e-7, scale = 1)
    expect_true(isTRUE(comparison), info = paste(label, name, paste(comparison, collapse = "; ")))
  }
  invisible(actual)
}

test_that("R ridge controllers use the shared penalized fitter", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  for (dist in c("extreme", "gaussian", "logistic", "t")) {
    for (kind in c("fixed", "df", "weighted", "scale", "strata", "initial", "zero", "one")) {
      attribute <- attributes(if (kind == "df") survival::ridge(data$x[, 2:3], df = 1)
                              else survival::ridge(data$x[, 2:3], theta = 1))
      args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                          controlvals = survival::survreg.control(), dist = dist,
                          pcols = list(2:3), pattr = list(attribute),
                          assign = list(Intercept = 1, ridge = 2:3),
                          parms = if (dist == "t") 4 else NULL))
      if (kind == "weighted") {
        args$weights <- rep(c(.5, 1, 2), 4)
        args$offset <- seq(.1, 1.2, length.out = 12)
      }
      if (kind == "scale") args$scale <- 1.5
      if (kind == "strata") { args$nstrat <- 2; args$strata <- rep(1:2, 6) }
      if (kind == "initial") args$init <- c(2, 0, 0, .1)
      if (kind %in% c("zero", "one")) args$controlvals$iter.max <- as.integer(kind == "one")
      check_penalty_fit(args, paste(dist, kind))
    }
  }
})

test_that("R spline and frailty controllers preserve their histories", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  for (family in c("gamma", "gaussian", "t")) for (sparse in c(FALSE, TRUE)) {
    for (search in c(FALSE, TRUE)) {
      options <- if (!search) list(theta = .4) else if (family == "t") list(df = 1) else list()
      term <- do.call(survival::frailty, c(list(rep(1:3, 4), distribution = family,
                                               sparse = sparse), options))
      columns <- if (sparse) 3L else 3:5
      x <- cbind(data$x[, 1:2], if (sparse) term else model.matrix(~ term - 1))
      colnames(x)[columns] <- paste0("frail", seq_along(columns))
      args <- list(x = x, y = data$y, weights = NULL, offset = NULL, init = NULL,
                   controlvals = survival::survreg.control(), dist = "gaussian",
                   pcols = list(columns), pattr = list(attributes(term)),
                   assign = list(Intercept = 1, age = 2, frailty = columns))
      check_penalty_fit(args, paste(family, sparse, search))
    }
  }
  for (kind in c("df", "fixed", "aic")) {
    term <- switch(kind,
                   df = survival::pspline(seq(.1, 1.2, length.out = 12), df = 2, nterm = 4),
                   fixed = survival::pspline(seq(.1, 1.2, length.out = 12), theta = .4, nterm = 4),
                   aic = survival::pspline(seq(.1, 1.2, length.out = 12), df = 0, nterm = 4))
    x <- cbind(Intercept = 1, term)
    colnames(x)[-1L] <- paste0("spline", seq_len(ncol(term)))
    args <- list(x = x, y = data$y, weights = NULL, offset = NULL, init = NULL,
                 controlvals = survival::survreg.control(), dist = "gaussian",
                 pcols = list(2:ncol(x)), pattr = list(attributes(term)),
                 assign = list(Intercept = 1, spline = 2:ncol(x)))
    check_penalty_fit(args, paste("spline", kind))
  }
})

test_that("custom R penalty searches run while the reference fitter is disabled", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  steps <- list()
  attribute <- attributes(survival::ridge(data$x[, 2:3], theta = 1, scale = FALSE))
  attribute$cargs <- c("x", "coef", "plik", "loglik", "status", "neff", "df", "trH")
  attribute$cfun <- function(parms, iter, old, x, coef, plik, loglik, status, neff, df, trH) {
    if (iter == 0L) return(list(theta = .25, marker = "custom state"))
    expect_identical(old$marker, "custom state")
    expect_equal(x, data$x[, 2:3])
    expect_equal(status, data$y[, 2L])
    expect_length(coef, 2L)
    expect_true(all(is.finite(c(plik, loglik, neff, df, trH))))
    steps[[length(steps) + 1L]] <<- iter
    list(theta = 1, done = iter == 2L, marker = old$marker)
  }
  args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                      controlvals = survival::survreg.control(), dist = "gaussian",
                      pcols = list(2:3), pattr = list(attribute),
                      assign = list(Intercept = 1, ridge = 2:3)))
  expected <- do.call(survival::survpenal.fit, args)
  steps <- list()
  bridge <- survivalr::survpenal.fit
  local_mocked_bindings(survpenal.fit = function(...) stop("R reference fitter called"), .package = "survival")
  actual <- do.call(bridge, args)
  expect_equal(actual, expected, tolerance = 4e-7)
  expect_equal(unlist(steps), 1:2)
  expect_equal(unserialize(serialize(actual, NULL)), actual)
})

test_that("R custom densities retain their effective sample size variance callback", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  attribute <- attributes(survival::ridge(data$x[, 2:3], df = 1))
  for (family in c("extreme", "gaussian", "logistic", "t")) {
    definition <- survival::survreg.distributions[[family]]
    definition$name <- paste("Custom", family)
    args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                        controlvals = survival::survreg.control(), dist = definition,
                        pcols = list(2:3), pattr = list(attribute),
                        assign = list(Intercept = 1, ridge = 2:3),
                        parms = if (family == "t") c(df = 4) else NULL))
    check_penalty_fit(args, paste("custom", family))
  }
  seen <- numeric()
  definition <- survival::survreg.distributions$gaussian
  definition$variance <- function(scale_squared) {
    seen <<- c(seen, scale_squared)
    1
  }
  args$dist <- definition
  args$parms <- NULL
  check_penalty_fit(args, "custom variance")
  expect_length(seen, 2L)
  expect_equal(seen[1], seen[2], tolerance = 1e-9)
})

test_that("R penalty callback failures retain their messages", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  attribute <- attributes(survival::ridge(data$x[, 2:3], theta = 1))
  args <- c(data, list(weights = NULL, offset = NULL, init = NULL,
                      controlvals = survival::survreg.control(), dist = "gaussian",
                      pcols = list(2:3), pattr = list(attribute),
                      assign = list(Intercept = 1, ridge = 2:3)))
  args$pattr[[1]]$cfun <- function(...) list(theta = 1)
  expect_error(do.call(survpenal.fit, args), "must return done")
  args$pattr[[1]] <- attribute
  args$pattr[[1]]$pfun <- function(...) stop("my penalty error")
  expect_error(do.call(survpenal.fit, args), "my penalty error")
  args$pattr[[1]] <- attribute
  args$pattr[[1]]$cfun <- function(...) stop("my controller error")
  expect_error(do.call(survpenal.fit, args), "my controller error")
  args$pattr[[1]] <- attribute
  args$pattr[[1]]$cargs <- "invalid"
  expect_error(do.call(survpenal.fit, args), "invalid not matched")
  args$pattr[[1]] <- attribute
  args$assign <- list(Intercept = 1, age = 2, group = 3)
  expect_error(do.call(survpenal.fit, args), "pcols and assign arguments disagree")
})

test_that("sparse-first penalties preserve dense column labels and term ordering", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  frailty <- survival::frailty(rep(c(10, 20, 30), 4), distribution = "gaussian",
                              theta = .4, sparse = TRUE)
  ridge <- survival::ridge(data$x[, 2], theta = 1, scale = FALSE)
  attr(ridge, "varname") <- "penalized age"
  # Stock R misindexes a dense penalty after removing an earlier sparse column.
  # Use its equivalent dense-first layout as the numerical reference.
  reference <- list(x = cbind(Intercept = 1, age = data$x[, 2], group = frailty),
                    y = data$y, weights = NULL, offset = seq(.1, 1.2, length.out = 12),
                    init = NULL, controlvals = survival::survreg.control(), dist = "gaussian",
                    pcols = list(2L, 3L), pattr = list(attributes(ridge), attributes(frailty)),
                    assign = list(Intercept = 1L, ridge = 2L, frailty = 3L))
  expected <- do.call(survival::survpenal.fit, reference)
  reference$x <- reference$x[, c(3, 1, 2)]
  reference$pcols <- list(3L, 1L)
  reference$assign <- list(frailty = 1L, Intercept = 2L, ridge = 3L)
  actual <- do.call(survpenal.fit, reference)
  for (field in c("coefficients", "icoef", "var", "var2", "loglik", "linear.predictors",
                  "frail", "fvar", "penalty", "score")) {
    expect_equal(actual[[field]], expected[[field]], tolerance = 4e-7, info = field)
  }
  expect_equal(actual$history, expected$history[c("frailty", "ridge")])
  expect_equal(actual$pterms, expected$pterms[c("frailty", "Intercept", "ridge")])
  expect_equal(actual$df, expected$df[c(3, 1, 2, 4)], tolerance = 4e-7)
  expected_assign <- expected$assign2[c("frailty", "Intercept", "ridge", "sigma")]
  expected_assign$frailty <- 1L
  expect_equal(actual$assign2, expected_assign)
})

test_that("custom interval densities and aliased predictions use the corrected solver", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- penalty_matrix_data()
  attribute <- attributes(survival::ridge(data$x[, 2:3], theta = 1))
  interval <- cbind(data$y[, 1], data$y[, 1] + .7, rep(c(1, 3, 2), 4))
  args <- list(x = data$x, y = interval, weights = NULL, offset = NULL, init = NULL,
               controlvals = survival::survreg.control(), dist = "gaussian",
               pcols = list(2:3), pattr = list(attribute),
               assign = list(Intercept = 1, ridge = 2:3))
  expected <- do.call(survival::survpenal.fit, args)
  definition <- survival::survreg.distributions$gaussian
  calls <- 0L
  definition$density <- function(z) {
    calls <<- calls + 1L
    survival::survreg.distributions$gaussian$density(z)
  }
  args$dist <- definition
  actual <- do.call(survpenal.fit, args)
  expect_equal(actual, expected, tolerance = 4e-7)
  expect_gt(calls, 0L)
  args$x <- cbind(data$x, alias = data$x[, 1])
  args$y <- data$y
  args$dist <- "gaussian"
  args$assign <- c(args$assign, list(alias = 4))
  args$offset <- seq(.1, 1.2, length.out = 12)
  actual <- do.call(survpenal.fit, args)
  coefficient <- actual$coefficients[1:4]
  expect_true(anyNA(coefficient))
  coefficient[is.na(coefficient)] <- 0
  expect_equal(actual$linear.predictors, drop(args$x %*% coefficient) + args$offset)
})
