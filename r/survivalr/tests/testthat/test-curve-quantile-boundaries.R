.capture_curve_quantile <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = warnings)
}

.stock_curve_quantile_labels <- function(captured, actual_fit) {
  # Split bridge KM curves expose bare group levels instead of formula labels.
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

test_that("curve quantile tolerance boundaries retain stock values, shapes and warnings", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = c(1, 2, 3, 4, 5, 7, 8, 10),
                     status = c(1, 1, 0, 1, 0, 1, 1, 1), g = rep(c("a", "b"), 4))
  data$start <- pmax(0, data$time - 2)
  for (formula in list(Surv(time, status) ~ 1, Surv(time, status) ~ g,
                       Surv(start, time, status) ~ g)) {
    actual_fit <- survfit(formula, data)
    stock_formula <- formula
    environment(stock_formula) <- asNamespace("survival")
    stock_fit <- getFromNamespace("survfit.formula", "survival")(stock_formula, data)
    expect_s3_class(stock_fit, "survfit")
    expect_false(inherits(stock_fit, "survival_py_survfit"))
    stock_quantile <- getFromNamespace("quantile.survfit", "survival")
    stock_median <- getFromNamespace("median.survfit", "survival")
    for (tolerance in c(sqrt(.Machine$double.eps), -Inf, Inf)) {
      for (probs in list(numeric(0), c(0, .5, 1))) for (bounds in c(FALSE, TRUE)) {
        actual <- .capture_curve_quantile(
          quantile(actual_fit, probs = probs, tolerance = tolerance, conf.int = bounds))
        expected <- .stock_curve_quantile_labels(.capture_curve_quantile(
          stock_quantile(stock_fit, probs = probs, tolerance = tolerance, conf.int = bounds)),
          actual_fit)
        expect_equal(actual, expected, tolerance = 4e-7)
      }
      expect_equal(.capture_curve_quantile(median(actual_fit, tolerance = tolerance)),
                   .stock_curve_quantile_labels(
                     .capture_curve_quantile(stock_median(stock_fit, tolerance = tolerance)),
                     actual_fit),
                   tolerance = 4e-7)
    }
  }
})

test_that("curve quantiles refuse logical, character and missing probabilities", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = 1:6, status = c(1, 0, 1, 1, 0, 1))
  fit <- survfit(Surv(time, status) ~ 1, data)
  stock_formula <- Surv(time, status) ~ 1
  environment(stock_formula) <- asNamespace("survival")
  stock <- getFromNamespace("survfit.formula", "survival")(stock_formula, data)
  expect_s3_class(stock, "survfit")
  expect_false(inherits(stock, "survival_py_survfit"))
  stock_quantile <- getFromNamespace("quantile.survfit", "survival")
  for (probs in list(TRUE, c(FALSE, TRUE), "0.5", c(.2, NA_real_, .8), NaN)) {
    expect_error(quantile(fit, probs = probs), "invalid probability", fixed = TRUE)
    expect_error(stock_quantile(stock, probs = probs), "invalid probability", fixed = TRUE)
  }
  for (probs in list(-.1, 1.1, Inf, -Inf)) {
    expect_error(quantile(fit, probs = probs), "Invalid probability", fixed = TRUE)
    expect_error(stock_quantile(stock, probs = probs), "Invalid probability", fixed = TRUE)
  }
})
