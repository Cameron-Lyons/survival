.capture_origin_quantile <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = warnings)
}

.origin_quantile_data <- function() {
  i <- 1:30
  data.frame(time = as.numeric(i %% 11 + 1), status = as.integer(i %% 4 != 0),
             start = as.numeric((i %% 3) / 4), x = cos(i * .47),
             g = rep(c("a", "b"), 15))
}

.origin_quantile_newdata <- function() {
  data.frame(x = c(.2, .7, -.4), g = c("a", "b", "a"),
             row.names = c("first", "second", "third"))
}

.origin_quantile_bare_groups <- function(captured, actual_fit) {
  # Split KM curves retain the bridge's public bare-level names.
  if (!is.list(actual_fit) || inherits(actual_fit, "python.builtin.object")) return(captured)
  labels <- names(actual_fit)
  if (is.list(captured$value)) {
    captured$value <- lapply(captured$value, function(value) {
      if (is.matrix(value)) rownames(value) <- labels
      value
    })
  } else if (is.matrix(captured$value)) rownames(captured$value) <- labels
  captured
}

.origin_quantile_stock_formula <- function(formula) {
  environment(formula) <- asNamespace("survival")
  formula
}

.origin_stock_quantile <- function(...) {
  getFromNamespace("quantile.survfit", "survival")(...)
}

.origin_stock_median <- function(...) {
  getFromNamespace("median.survfit", "survival")(...)
}

.origin_expect_stock_fit <- function(fit) {
  expect_s3_class(fit, "survfit")
  expect_false(inherits(fit, "survival_py_survfit"))
}

test_that("Cox quantiles and medians preserve conditional origins and curve dimensions", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- .origin_quantile_data()
  newdata <- .origin_quantile_newdata()
  formulas <- list(Surv(time, status) ~ x, Surv(time, status) ~ x + strata(g),
                   Surv(start, time, status) ~ x, Surv(start, time, status) ~ x + strata(g))
  requests <- list(NULL, newdata[1L, , drop = FALSE], newdata,
                   newdata[1L, "x", drop = FALSE], newdata["x"])
  for (formula in formulas) {
    actual_model <- coxph(formula, data)
    stock_model <- survival::coxph(.origin_quantile_stock_formula(formula), data,
                                  x = TRUE, y = TRUE, model = TRUE)
    for (origin in list(NULL, 0, 3)) for (request in requests) for (errors in c(FALSE, TRUE)) {
      arguments <- list(se.fit = errors)
      if (!is.null(request)) arguments$newdata <- request
      if (!is.null(origin)) arguments$start.time <- origin
      actual_fit <- do.call(survfit, c(list(actual_model), arguments))
      stock_fit <- do.call(getFromNamespace("survfit.coxph", "survival"),
                          c(list(stock_model), arguments))
      .origin_expect_stock_fit(stock_fit)
      for (probs in list(c(0, .01, .5, 1), c(1, .5, .5, 0), numeric(0))) {
        for (bounds in c(FALSE, TRUE)) {
          actual <- .capture_origin_quantile(quantile(actual_fit, probs, conf.int = bounds))
          expected <- .capture_origin_quantile(.origin_stock_quantile(stock_fit, probs,
                                                                      conf.int = bounds))
          expect_equal(actual, expected, tolerance = 4e-7)
        }
      }
      for (scale in c(1, 2)) {
        expect_equal(.capture_origin_quantile(median(actual_fit, scale = scale)),
                     .capture_origin_quantile(.origin_stock_median(stock_fit, scale = scale)),
                     tolerance = 4e-7)
      }
    }
  }
})

test_that("KM start.time truncates data while quantile origins retain zero", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- .origin_quantile_data()
  formulas <- list(Surv(time, status) ~ 1, Surv(time, status) ~ g,
                   Surv(start, time, status) ~ 1, Surv(start, time, status) ~ g)
  for (formula in formulas) for (origin in list(NULL, 0, 3)) for (errors in c(FALSE, TRUE)) {
    arguments <- list(formula, data = data, se.fit = errors)
    if (!is.null(origin)) arguments$start.time <- origin
    actual_fit <- do.call(survfit, arguments)
    stock_arguments <- arguments
    stock_arguments[[1L]] <- .origin_quantile_stock_formula(formula)
    stock_fit <- do.call(getFromNamespace("survfit.formula", "survival"), stock_arguments)
    .origin_expect_stock_fit(stock_fit)
    for (probs in list(c(0, .01, .5, 1), numeric(0))) for (bounds in c(FALSE, TRUE)) {
      actual <- .capture_origin_quantile(quantile(actual_fit, probs, conf.int = bounds))
      expected <- .origin_quantile_bare_groups(
        .capture_origin_quantile(.origin_stock_quantile(stock_fit, probs, conf.int = bounds)),
        actual_fit)
      expect_equal(actual, expected, tolerance = 4e-7)
    }
    expected <- .origin_quantile_bare_groups(
      .capture_origin_quantile(.origin_stock_median(stock_fit)), actual_fit)
    expect_equal(.capture_origin_quantile(median(actual_fit)), expected, tolerance = 4e-7)
  }
})

test_that("conditional Cox layouts retain scalar scales and infinite tolerance behavior", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- .origin_quantile_data()
  newdata <- .origin_quantile_newdata()
  actual_model <- coxph(Surv(time, status) ~ x + strata(g), data)
  stock_model <- survival::coxph(
    .origin_quantile_stock_formula(Surv(time, status) ~ x + strata(g)), data,
    x = TRUE, y = TRUE, model = TRUE)
  fits <- list()
  for (request in list(newdata[1L, , drop = FALSE], newdata, newdata["x"])) {
    fits[[length(fits) + 1L]] <- list(
      actual = survfit(actual_model, newdata = request, start.time = 3),
      stock = getFromNamespace("survfit.coxph", "survival")(
        stock_model, newdata = request, start.time = 3))
    .origin_expect_stock_fit(fits[[length(fits)]]$stock)
  }
  for (fit in fits) for (scale in c(0, -1, 1, 2, Inf, -Inf, NA_real_, NaN)) {
    for (probs in list(c(0, .01, .5, 1), numeric(0))) for (bounds in c(FALSE, TRUE)) {
      expect_equal(.capture_origin_quantile(quantile(fit$actual, probs, conf.int = bounds, scale = scale)),
                   .capture_origin_quantile(.origin_stock_quantile(
                     fit$stock, probs, conf.int = bounds, scale = scale)),
                   tolerance = 4e-7)
    }
    expect_equal(.capture_origin_quantile(median(fit$actual, scale = scale)),
                 .capture_origin_quantile(.origin_stock_median(fit$stock, scale = scale)),
                 tolerance = 4e-7)
  }
  for (fit in fits) for (tolerance in c(sqrt(.Machine$double.eps), -Inf, Inf)) {
    for (probs in list(c(0, .5, 1), numeric(0))) for (bounds in c(FALSE, TRUE)) {
      expect_equal(.capture_origin_quantile(quantile(fit$actual, probs, conf.int = bounds,
                                                    tolerance = tolerance)),
                   .capture_origin_quantile(.origin_stock_quantile(
                     fit$stock, probs, conf.int = bounds, tolerance = tolerance)), tolerance = 4e-7)
    }
    expect_equal(.capture_origin_quantile(median(fit$actual, tolerance = tolerance)),
                 .capture_origin_quantile(.origin_stock_median(fit$stock, tolerance = tolerance)),
                 tolerance = 4e-7)
  }
})
