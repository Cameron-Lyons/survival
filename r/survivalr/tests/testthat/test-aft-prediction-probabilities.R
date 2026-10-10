.aft_probability_compare <- function(actual, expected) {
  if (is.list(expected)) {
    expect_named(actual, names(expected))
    for (field in names(expected)) .aft_probability_compare(actual[[field]], expected[[field]])
    return(invisible(NULL))
  }
  expect_identical(dim(actual), dim(expected))
  expect_identical(is.na(actual), is.na(expected), ignore_attr = TRUE)
  expect_equal(as.numeric(actual), as.numeric(expected), tolerance = 4e-7)
}

test_that("AFT prediction probabilities preserve missing columns and ignore unused arguments", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = c(1, 2, 3, 4, 5, 7, 8, 10),
                     status = c(1, 1, 0, 1, 0, 1, 1, 1), x = c(0, 1, 0, 1, 0, 0, 1, 0))
  newdata <- data.frame(x = c(0, 1))
  for (family in c("weibull", "gaussian")) {
    options <- list(formula = Surv(time, status) ~ x, data = data, dist = family)
    if (family == "gaussian") options$scale <- .7
    fit <- do.call(survreg, options)
    stock <- do.call(survival::survreg, options)
    for (type in c("quantile", "uquantile")) {
      for (p in list(c(.2, NA_real_, .8), NA_real_)) for (se in c(FALSE, TRUE)) {
        .aft_probability_compare(
          predict(fit, newdata, type = type, p = p, se.fit = se),
          predict(stock, newdata, type = type, p = p, se.fit = se))
      }
    }
    for (type in c("response", "link", "lp", "linear", "terms")) {
      for (se in c(FALSE, TRUE)) {
        .aft_probability_compare(
          predict(fit, newdata, type = type, p = "unused", se.fit = se),
          predict(stock, newdata, type = type, p = "unused", se.fit = se))
      }
    }
  }
})

test_that("empty AFT probabilities preserve both fit and standard-error matrix widths", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = c(1, 2, 3, 4, 5, 7, 8, 10),
                     status = c(1, 1, 0, 1, 0, 1, 1, 1), x = c(0, 1, 0, 1, 0, 0, 1, 0))
  newdata <- data.frame(x = c(0, 1))
  specs <- lapply(c("weibull", "lognormal", "loglogistic", "gaussian", "logistic",
                    "t", "exponential", "rayleigh"), function(family) list(dist = family))
  specs <- c(specs, list(list(dist = "weibull", scale = .7), list(dist = "gaussian", scale = .7)))
  empty <- matrix(numeric(0), nrow = nrow(newdata), ncol = 0L)
  for (spec in specs) {
    options <- c(list(formula = Surv(time, status) ~ x, data = data), spec)
    if (spec$dist == "t") options$parms <- 5
    fit <- do.call(survreg, options)
    stock <- do.call(survival::survreg, options)
    fixed <- spec$dist %in% c("exponential", "rayleigh") || !is.null(spec$scale)
    for (type in c("quantile", "uquantile")) for (kind in c("NULL", "numeric(0)")) {
      p <- if (kind == "NULL") NULL else numeric(0)
      actual <- predict(fit, newdata, type = type, p = p, se.fit = FALSE)
      .aft_probability_compare(actual, empty)
      null_error <- is.null(p) && spec$dist %in% c("gaussian", "lognormal", "t")
      if (null_error) {
        # qnorm(NULL) and qt(NULL) reject NULL even without standard errors.
        expect_error(predict(stock, newdata, type = type, p = p, se.fit = FALSE),
                     "Non-numeric argument to mathematical function", fixed = TRUE)
      } else {
        .aft_probability_compare(actual, predict(stock, newdata, type = type, p = p, se.fit = FALSE))
      }
      # The independently specified oracle retains two rows and zero columns
      # for both outputs, including stock's failing estimated-scale SE path.
      actual_se <- predict(fit, newdata, type = type, p = p, se.fit = TRUE)
      .aft_probability_compare(actual_se, list(fit = empty, se.fit = empty))
      if (null_error) {
        expect_error(predict(stock, newdata, type = type, p = p, se.fit = TRUE),
                     "Non-numeric argument to mathematical function", fixed = TRUE)
      } else if (!fixed) {
        expect_error(predict(stock, newdata, type = type, p = p, se.fit = TRUE),
                     "subscript out of bounds", fixed = TRUE)
      } else {
        stock_se <- predict(stock, newdata, type = type, p = p, se.fit = TRUE)
        if (type == "quantile" && spec$dist != "gaussian") {
          .aft_probability_compare(actual_se, stock_se)
        } else {
          # Fixed-scale uquantiles (and identity quantiles) incorrectly retain
          # one location SE per row despite an empty probability vector.
          expect_identical(dim(stock_se$fit), c(2L, 0L))
          expect_null(dim(stock_se$se.fit))
          expect_length(stock_se$se.fit, 2L)
        }
      }
    }
  }
})

test_that("AFT distribution queries retain missing numeric values", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (family in c("weibull", "lognormal", "loglogistic", "gaussian", "logistic",
                   "t", "exponential", "rayleigh")) {
    for (routine in c("dsurvreg", "psurvreg", "qsurvreg")) {
      arguments <- list(c(.2, NA_real_, .8), mean = .3, scale = .8,
                        distribution = family, parms = if (family == "t") 5 else NULL)
      .aft_probability_compare(
        do.call(get(routine), arguments),
        do.call(get(routine, envir = asNamespace("survival")), arguments))
    }
  }
})
