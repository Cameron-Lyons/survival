.capture_quantile_argument <- function(expr) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = warnings)
}

.quantile_argument_labels <- function(captured, actual_fit) {
  # Preserve the bridge's established bare labels for split ordinary KM curves.
  if (!is.list(actual_fit) || inherits(actual_fit, "python.builtin.object")) return(captured)
  labels <- names(actual_fit)
  if (is.list(captured$value) && is.null(captured$value$error)) {
    captured$value <- lapply(captured$value, function(value) {
      if (is.matrix(value)) rownames(value) <- labels
      value
    })
  } else if (is.matrix(captured$value)) rownames(captured$value) <- labels
  captured
}

.quantile_argument_pairs <- function() {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  i <- 1:30
  data <- data.frame(time = as.numeric(i %% 11 + 1), status = as.integer(i %% 4 != 0),
                     start = as.numeric((i %% 3) / 4), x = cos(i * .47),
                     g = rep(c("a", "b"), 15), zero = 0)
  newdata <- data.frame(x = c(.2, .7, -.4), g = c("a", "b", "a"),
                        row.names = c("first", "second", "third"))
  stock_formula <- function(text) as.formula(text, env = asNamespace("survival"))
  pairs <- list()
  for (name in c("km", "km_group", "km_counting", "km_none", "km_group_none", "km_censor")) {
    formula <- switch(name, km = "Surv(time,status)~1", km_group = "Surv(time,status)~g",
                       km_counting = "Surv(start,time,status)~g", km_none = "Surv(time,status)~1",
                       km_group_none = "Surv(time,status)~g", km_censor = "Surv(time,zero)~1")
    arguments <- list(formula = stock_formula(formula), data = data)
    if (grepl("none", name)) arguments$conf.type <- "none"
    stock <- do.call(getFromNamespace("survfit.formula", "survival"), arguments)
    arguments$formula <- as.formula(formula)
    pairs[[name]] <- list(stock = stock, actual = do.call(survfit, arguments))
  }
  for (name in c("cox_one", "cox_many", "cox_none", "cox_strat_one", "cox_strat_many",
                 "cox_array", "cox_array_none")) {
    formula <- if (grepl("strat|array", name)) "Surv(time,status)~x+strata(g)" else
      "Surv(time,status)~x"
    stock_model <- survival::coxph(stock_formula(formula), data, x = TRUE, y = TRUE, model = TRUE)
    actual_model <- coxph(as.formula(formula), data)
    request <- if (grepl("array", name)) newdata["x"] else if (grepl("one|none", name))
      newdata[1L, , drop = FALSE] else newdata
    arguments <- list(newdata = request, start.time = 3)
    if (grepl("none", name)) arguments$se.fit <- FALSE
    pairs[[name]] <- list(
      stock = do.call(getFromNamespace("survfit.coxph", "survival"), c(list(stock_model), arguments)),
      actual = do.call(survfit, c(list(actual_model), arguments)))
  }
  for (pair in pairs) {
    expect_s3_class(pair$stock, "survfit")
    expect_false(inherits(pair$stock, "survival_py_survfit"))
  }
  pairs
}

.quantile_confidence_arguments <- function() {
  list(true = TRUE, false = FALSE, one = 1, zero = 0, two = 2, negative = -1,
       infinity = Inf, negative_infinity = -Inf, nan = NaN, missing = NA,
       missing_numeric = NA_real_, many = c(TRUE, FALSE), many_numeric = c(1, 0),
       empty = logical(0), empty_numeric = numeric(0), null = NULL, text_true = "TRUE",
       text_t = "T", text_false = "false", text_f = "F", text_one = "1", text_bad = "unused",
       text_empty = "", factor = factor("TRUE"), factor_false = factor("FALSE"),
       factor_missing = factor(NA_character_), date = as.Date("1970-01-02"), complex = 1 + 1i,
       raw = as.raw(1), list_true = list(TRUE), list_many = list(TRUE, FALSE))
}

test_that("curve quantile confidence conditions retain stock coercion and errors", {
  pairs <- .quantile_argument_pairs()
  stock_quantile <- getFromNamespace("quantile.survfit", "survival")
  for (name in names(pairs)) for (probs in list(c(0, .5, 1), numeric(0))) {
    pair <- pairs[[name]]
    for (label in names(.quantile_confidence_arguments())) {
      confidence <- .quantile_confidence_arguments()[[label]]
      actual <- .capture_quantile_argument(quantile(pair$actual, probs, conf.int = confidence))
      expected <- .quantile_argument_labels(.capture_quantile_argument(
        stock_quantile(pair$stock, probs, conf.int = confidence)), pair$actual)
      expect_equal(actual, expected, tolerance = 4e-7, info = paste(name, label, length(probs)))
    }
  }
})

test_that("curve quantile tolerance arithmetic retains stock values warnings and errors", {
  pairs <- .quantile_argument_pairs()
  stock_quantile <- getFromNamespace("quantile.survfit", "survival")
  stock_median <- getFromNamespace("median.survfit", "survival")
  tolerances <- list(default = sqrt(.Machine$double.eps), zero = 0, negative = -.1,
    infinity = Inf, negative_infinity = -Inf, nan = NaN, missing = NA, empty = numeric(0),
    null = NULL, many = c(0, .1), many3 = c(0, .1, .2), many4 = c(0, .1, .2, .3),
    true = TRUE, false = FALSE, text = ".01", text_bad = "unused", factor = factor(".01"),
    date = as.Date("1970-01-02"), complex = 1 + 1i, list_one = list(.01), list_empty = list())
  for (name in names(pairs)) for (label in names(tolerances)) {
    pair <- pairs[[name]]
    tolerance <- tolerances[[label]]
    for (probs in list(c(0, .5, 1), numeric(0))) {
      actual <- .capture_quantile_argument(quantile(pair$actual, probs, tolerance = tolerance))
      expected <- .quantile_argument_labels(.capture_quantile_argument(
        stock_quantile(pair$stock, probs, tolerance = tolerance)), pair$actual)
      expect_equal(actual, expected, tolerance = 4e-7, info = paste(name, label, length(probs)))
    }
    actual <- .capture_quantile_argument(median(pair$actual, tolerance = tolerance))
    expected <- .quantile_argument_labels(.capture_quantile_argument(
      stock_median(pair$stock, tolerance = tolerance)), pair$actual)
    expect_equal(actual, expected, tolerance = 4e-7, info = paste(name, label, "median"))
  }
})

test_that("curve quantile confidence promises retain stock evaluation order", {
  pairs <- .quantile_argument_pairs()
  stock_quantile <- getFromNamespace("quantile.survfit", "survival")
  compare <- function(pair, request) {
    actual <- .capture_quantile_argument(request(quantile, pair$actual))
    expected <- .quantile_argument_labels(.capture_quantile_argument(
      request(stock_quantile, pair$stock)), pair$actual)
    expect_equal(actual, expected, tolerance = 4e-7)
  }
  for (pair in pairs) {
    compare(pair, function(fun, fit) fun(fit, c(0, .5, 1),
      conf.int = stop("confidence evaluated")))
    compare(pair, function(fun, fit) fun(fit, numeric(0),
      conf.int = stop("confidence evaluated")))
    compare(pair, function(fun, fit) fun(fit, TRUE,
      conf.int = stop("confidence evaluated"), tolerance = stop("tolerance evaluated")))
    compare(pair, function(fun, fit) fun(fit, c(0, .5, 1),
      conf.int = NA, tolerance = stop("tolerance evaluated")))
    compare(pair, function(fun, fit) fun(fit, c(0, .5, 1),
      conf.int = NA, scale = stop("scale evaluated")))
    compare(pair, function(fun, fit) fun(fit, c(0, .5, 1),
      conf.int = FALSE, tolerance = stop("tolerance evaluated"), scale = stop("scale evaluated")))
  }
})

test_that("fitted-curve medians retain stock refusal of duplicate confidence arguments", {
  pairs <- .quantile_argument_pairs()
  stock_median <- getFromNamespace("median.survfit", "survival")
  for (pair in pairs) for (confidence in list(TRUE, FALSE, NA, numeric(0))) {
    actual <- .capture_quantile_argument(median(pair$actual, conf.int = confidence))
    expected <- .capture_quantile_argument(stock_median(pair$stock, conf.int = confidence))
    expect_equal(actual, expected)
  }
})
