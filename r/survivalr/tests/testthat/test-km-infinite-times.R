.capture_infinite_km <- function(expr) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = warnings)
}

.infinite_km_inputs <- function() {
  list(
    finite = list(time = c(1, 2, 2, 3, 4, 5), status = c(1, 0, 1, 0, 1, 1)),
    positive = list(time = c(1, 2, 2, 3, Inf, Inf), status = c(1, 0, 1, 0, 1, 0)),
    negative = list(time = c(-Inf, -Inf, 1, 2, 3, 4), status = c(1, 0, 1, 0, 1, 1)),
    mixed_near = list(time = c(-Inf, 1, 1 + 1e-12, 2, 3, Inf), status = c(1, 0, 1, 0, 1, 1)),
    positive_only = list(time = rep(Inf, 4), status = c(1, 0, 1, 0)),
    negative_only = list(time = rep(-Inf, 4), status = c(1, 0, 1, 0)),
    counting = list(start = c(-Inf, -Inf, 0, 1, 2, 3), time = c(1, 2, 2, 3, Inf, Inf),
                    status = c(1, 0, 1, 0, 1, 0)),
    counting_near = list(start = c(-Inf, 0, 0, 1, 2, 3), time = c(1, 1 + 1e-12, 2, 3, Inf, Inf),
                         status = c(1, 0, 1, 0, 1, 0)),
    counting_positive_only = list(start = c(-Inf, 0, 1, 2), time = rep(Inf, 4),
                                  status = c(1, 0, 1, 0)))
}

.infinite_km_response <- function(input) {
  constructor <- getFromNamespace("Surv", "survival")
  if (is.null(input$start)) constructor(input$time, input$status) else
    constructor(input$start, input$time, input$status)
}

.infinite_km_finite_aeq <- function(response) {
  # Stock aeqSurv owns finite cutpoint selection. Its result is inserted into
  # the original matrix positions, preserving infinite endpoints and status rows.
  columns <- seq_len(ncol(response) - 1L)
  times <- unclass(response)[, columns, drop = FALSE]
  finite <- is.finite(times)
  if (any(finite)) {
    constructor <- getFromNamespace("Surv", "survival")
    fixed <- getFromNamespace("aeqSurv", "survival")(
      constructor(times[finite], rep(0, sum(finite))))
    times[finite] <- unclass(fixed)[, 1L]
  }
  response[, columns] <- times
  response
}

.infinite_km_stock_fit <- function(input, grouped, weighted, timefix, type, start = NULL) {
  response <- .infinite_km_response(input)
  if (timefix) response <- .infinite_km_finite_aeq(response)
  rows <- length(input$time)
  data <- list(response = response)
  if (grouped) data$group <- rep(c("a", "b"), length.out = rows)
  formula <- as.formula(if (grouped) "response ~ group" else "response ~ 1",
                        env = asNamespace("survival"))
  arguments <- list(formula = formula, data = data, timefix = FALSE,
                    stype = type[[1L]], ctype = type[[2L]], id = seq_len(rows),
                    entry = !is.null(input$start))
  if (weighted) {
    arguments$weights <- rep(c(.5, 1, 2), length.out = rows)
    arguments$influence <- 3
  }
  if (!is.null(start)) arguments$start.time <- start
  do.call(getFromNamespace("survfit.formula", "survival"), arguments)
}

.infinite_km_actual_fit <- function(input, grouped, weighted, timefix, type, start = NULL) {
  data <- as.data.frame(input)
  rows <- nrow(data)
  if (grouped) data$group <- rep(c("a", "b"), length.out = rows)
  formula <- paste(if (is.null(input$start)) "Surv(time,status)" else "Surv(start,time,status)",
                   if (grouped) "~group" else "~1")
  before <- serialize(data, NULL)
  arguments <- list(formula = as.formula(formula), data = data, timefix = timefix,
                    stype = type[[1L]], ctype = type[[2L]], id = seq_len(rows),
                    entry = !is.null(input$start))
  if (weighted) {
    arguments$weights <- rep(c(.5, 1, 2), length.out = rows)
    arguments$influence <- 3
  }
  if (!is.null(start)) arguments$start.time <- start
  fit <- do.call(survfit, arguments)
  expect_identical(serialize(data, NULL), before)
  fit
}

.infinite_km_expected_columns <- function(fit, summary = FALSE) {
  fields <- c("time", "n.risk", "n.event", "n.censor", "n.enter", "surv", "cumhaz",
              "std.err", "std.chaz", "lower", "upper")
  values <- lapply(fields, function(field) fit[[field]])
  names(values) <- fields
  values <- Filter(Negate(is.null), values)
  if (!summary && isTRUE(fit$logse)) values$std.err <- values$std.err * fit$surv
  if (!is.null(fit$strata)) values$strata <- if (summary) {
    sub("^group=", "", as.character(fit$strata))
  } else rep(sub("^group=", "", names(fit$strata)), fit$strata)
  values
}

.infinite_km_compare_columns <- function(actual, expected, info) {
  expect_true(all(names(expected) %in% names(actual)), info = info)
  normalize <- function(value) {
    if (!is.numeric(value)) return(as.character(value))
    value <- as.numeric(value)
    # Numerical curve outputs use NaN for R's undefined NA estimates/bounds.
    value[is.na(value)] <- NA_real_
    value
  }
  expect_equal(lapply(actual[names(expected)], normalize), lapply(expected, normalize),
               tolerance = 1e-8, info = info)
}

.infinite_km_signed <- function(value) {
  if (is.list(value)) return(lapply(value, .infinite_km_signed))
  result <- lapply(as.numeric(value), function(number) {
    if (is.na(number)) return(NULL)
    if (is.infinite(number)) return(if (number > 0) "Inf" else "-Inf")
    if (number == 0 && 1 / number < 0) return("-0")
    number
  })
  attributes(result) <- attributes(value)[intersect(names(attributes(value)),
                                                   c("names", "dim", "dimnames"))]
  result
}

.infinite_km_quantile_labels <- function(value) {
  if (is.list(value)) return(lapply(value, .infinite_km_quantile_labels))
  if (is.matrix(value)) rownames(value) <- sub("^group=", "", rownames(value))
  value
}

test_that("infinite right and counting KM curves retain stock counts summaries and signed quantiles", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  stock_summary <- getFromNamespace("summary.survfit", "survival")
  stock_quantile <- getFromNamespace("quantile.survfit", "survival")
  inputs <- .infinite_km_inputs()
  for (name in names(inputs)) for (timefix in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE))
    for (weighted in c(FALSE, TRUE)) for (type in list(c(1, 1), c(2, 2))) {
      info <- paste(name, timefix, grouped, weighted, paste(type, collapse = ""))
      input <- inputs[[name]]
      stock <- .infinite_km_stock_fit(input, grouped, weighted, timefix, type)
      expect_s3_class(stock, "survfit")
      expect_false(inherits(stock, "survival_py_survfit"), info = info)
      actual <- .infinite_km_actual_fit(input, grouped, weighted, timefix, type)
      .infinite_km_compare_columns(as.data.frame(actual), .infinite_km_expected_columns(stock), info)
      requests <- list(list(times = c(-Inf, 0, 2, Inf), extend = FALSE),
                       list(times = c(-Inf, 0, 2, Inf), extend = TRUE),
                       list(times = c(Inf, 2, 2, -Inf, 0), extend = TRUE, dosum = FALSE))
      for (request in requests) {
        expected <- do.call(stock_summary, c(list(object = stock, rmean = "none"), request))
        value <- do.call(summary, c(list(object = actual), request))
        .infinite_km_compare_columns(value, .infinite_km_expected_columns(expected, TRUE), info)
      }
      for (scale in c(1, -2, Inf)) for (confidence in c(FALSE, TRUE)) {
        expected <- .capture_infinite_km(stock_quantile(stock, c(0, .25, .5, .75, 1),
          scale = scale, conf.int = confidence))
        expected$value <- .infinite_km_quantile_labels(expected$value)
        value <- .capture_infinite_km(quantile(actual, c(0, .25, .5, .75, 1),
          scale = scale, conf.int = confidence))
        expected$value <- .infinite_km_signed(expected$value)
        value$value <- .infinite_km_signed(value$value)
        expect_equal(value, expected, tolerance = 1e-8, info = paste(info, scale, confidence))
      }
    }
})

test_that("finite-only stock timefix preserves infinite endpoints and source row positions", {
  skip_if_not_installed("survival")
  inputs <- .infinite_km_inputs()
  stock_aeq <- getFromNamespace("aeqSurv", "survival")
  for (input in inputs) {
    source <- .infinite_km_response(input)
    fixed <- .infinite_km_finite_aeq(source)
    original <- unclass(source)
    result <- unclass(fixed)
    columns <- seq_len(ncol(original) - 1L)
    expect_identical(dim(result), dim(original))
    expect_identical(result[, ncol(result)], original[, ncol(original)])
    expect_identical(is.infinite(result[, columns, drop = FALSE]),
                     is.infinite(original[, columns, drop = FALSE]))
    expect_identical(result[is.infinite(original)], original[is.infinite(original)])
  }
  source <- .infinite_km_response(inputs$mixed_near)
  raw <- .capture_infinite_km(stock_aeq(source))
  expect_equal(unclass(raw$value)[, 1L], c(1, 1, 2, 3, 3, 1))
  expect_identical(raw$warnings,
    "number of rows of result is not a multiple of vector length (arg 1)")
  expect_equal(unclass(.infinite_km_finite_aeq(source))[, 1L], c(-Inf, 1, 1, 2, 3, Inf))
  raw <- .capture_infinite_km(stock_aeq(.infinite_km_response(inputs$counting_near)))
  expect_identical(raw$value$error, "aeqSurv exception, an interval has effective length 0")
  source <- .infinite_km_response(list(time = c(1, 1 + 1e-12, 2, 3, Inf, Inf),
    status = c(1, 0, 1, 0, 1, 0)))
  raw <- .capture_infinite_km(stock_aeq(source))
  expect_equal(unclass(raw$value)[, 1L], c(1, 1, 2, 3, 3, 3))
  expect_length(raw$warnings, 0L)
  expect_equal(unclass(.infinite_km_finite_aeq(source))[, 1L], c(1, 1, 2, 3, Inf, Inf))
})

test_that("infinite summary rows retain values lost by stock empty-stratum assembly", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = c(1, 2, 1, Inf), status = c(1, 0, 1, 1),
                     group = c("a", "a", "b", "b"))
  formula <- as.formula("Surv(time,status)~group", env = asNamespace("survival"))
  stock <- getFromNamespace("survfit.formula", "survival")(formula, data, timefix = FALSE)
  expect_s3_class(stock, "survfit")
  expect_false(inherits(stock, "survival_py_survfit"))
  stock_summary <- getFromNamespace("summary.survfit", "survival")
  stock_subset <- getFromNamespace("[.survfit", "survival")
  raw <- stock_summary(stock, times = Inf, extend = FALSE, dosum = TRUE, rmean = "none")
  expect_identical(raw$time, Inf)
  expect_null(raw$surv)
  expect_null(raw$n.event)
  expect_null(raw$n.censor)
  parts <- lapply(seq_along(stock$strata), function(i) {
    stock_summary(stock_subset(stock, i), times = Inf, extend = FALSE,
                  dosum = TRUE, rmean = "none")
  })
  # Leading NULL fields make stock unlistsurv discard later strata. Retain
  # every original stock field and use its direct per-stratum kernels for the
  # missing values, rather than implementing the event-count arithmetic here.
  repaired <- raw
  for (field in c("surv", "n.event", "n.censor", "n.enter", "std.err", "std.chaz",
                  "cumhaz", "lower", "upper")) {
    if (is.null(raw[[field]]) && !is.null(stock[[field]])) {
      repaired[[field]] <- do.call(c, lapply(parts, `[[`, field))
    }
  }
  expect_equal(repaired$n.event, 2)
  expect_equal(repaired$n.censor, 0)
  expect_identical(do.call(c, lapply(parts, `[[`, "time")), raw$time)
  actual <- survfit(Surv(time, status) ~ group, data, timefix = FALSE)
  value <- summary(actual, times = Inf, extend = FALSE, dosum = TRUE)
  .infinite_km_compare_columns(value, .infinite_km_expected_columns(repaired, TRUE),
                               "empty first stratum")
})

test_that("KM infinite start times and summary queries retain stock behavior", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  inputs <- .infinite_km_inputs()[c("finite", "positive", "negative", "counting")]
  stock_summary <- getFromNamespace("summary.survfit", "survival")
  for (input in inputs) for (start in c(-Inf, Inf)) {
    expected <- .capture_infinite_km(.infinite_km_stock_fit(input, FALSE, FALSE, FALSE, c(1, 1), start))
    actual <- .capture_infinite_km(.infinite_km_actual_fit(input, FALSE, FALSE, FALSE, c(1, 1), start))
    if (is.list(expected$value) && !is.null(expected$value$error)) {
      expect_match(actual$value$error, "start.time", fixed = TRUE)
    } else {
      expect_s3_class(expected$value, "survfit")
      expect_false(inherits(expected$value, "survival_py_survfit"))
      .infinite_km_compare_columns(as.data.frame(actual$value),
                                   .infinite_km_expected_columns(expected$value), paste("start", start))
      value <- .capture_infinite_km(summary(actual$value, times = c(-Inf, Inf), extend = TRUE))
      reference <- .capture_infinite_km(stock_summary(expected$value,
        times = c(-Inf, Inf), extend = TRUE, rmean = "none"))
      expect_identical(value$warnings, reference$warnings)
      if (is.list(reference$value) && !is.null(reference$value$error)) {
        # Stock counting curves with an infinite conditional origin may have
        # earlier entry rows; survfit0 then produces an unsorted time grid.
        expect_identical(value$value$error, reference$value$error)
      } else .infinite_km_compare_columns(value$value,
        .infinite_km_expected_columns(reference$value, TRUE), paste("start", start))
    }
  }
  input <- inputs$finite
  expected <- .infinite_km_stock_fit(input, FALSE, FALSE, FALSE, c(1, 1))
  actual <- .infinite_km_actual_fit(input, FALSE, FALSE, FALSE, c(1, 1))
  for (times in list(NA_real_, NaN, c(-Inf, NA_real_, Inf))) {
    expect_error(stock_summary(expected, times = times), "times contains missing values", fixed = TRUE)
    expect_error(summary(actual, times = times), "times contains missing values", fixed = TRUE)
  }
})
