.aggregate_fun_pair <- function(multistate = FALSE, stratified = FALSE) {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  if (multistate) {
    data <- data.frame(id = 1:18, time = 1:18,
      event = factor(c(1, 2, 0, 1, 0, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 1, 2, 0),
        levels = 0:2, labels = c("censor", "A", "B")),
      x = c(0, 1, -1, 2, 0, -1, 2, 0, 1, -2, 1, 2, 0, 1, -1, 2, -2, 0))
    actual_fit <- coxph(Surv(time, event) ~ x, data, id = id,
      init = c(.25, -.125), iter.max = 0)
    stock_fit <- survival::coxph(survival::Surv(time, event) ~ x, data, id = id,
      init = c(.25, -.125), control = survival::coxph.control(iter.max = 0))
  } else {
    data <- data.frame(time = 1:12, status = rep(c(1L, 1L, 0L), 4),
      x = rep(c(0, 1, 2), 4), z = c(0, 1, .5, -1, 2, .7, 1, 0, -1, .5, 2, 1),
      g = factor(rep(c("a", "b"), each = 6L)))
    actual_formula <- if (stratified) Surv(time, status) ~ x + z + strata(g)
      else Surv(time, status) ~ x + z
    stock_formula <- if (stratified) survival::Surv(time, status) ~ x + z + strata(g)
      else survival::Surv(time, status) ~ x + z
    actual_fit <- coxph(actual_formula, data,
      init = c(.25, -.125), iter.max = 0)
    stock_fit <- survival::coxph(stock_formula, data,
      init = c(.25, -.125),
      control = survival::coxph.control(iter.max = 0))
  }
  newdata <- data.frame(x = 0:3, z = c(1, 0, -1, .4),
    row.names = c("b", "a", "d", "c"))
  list(actual = survfit(actual_fit, newdata = newdata, se.fit = FALSE),
    stock = survival::survfit(stock_fit, newdata = newdata, se.fit = FALSE))
}

.aggregate_fun_values <- function(x, multistate) {
  if (multistate) as.numeric(x$pstate) else as.data.frame(x)$surv
}

.aggregate_fun_groups <- function() list(
  none = NULL, constant = rep("whole", 4), bare = c("b", "a", "b", "a"),
  numeric = c(20L, 10L, 20L, 10L),
  factor = factor(c("b", "a", "b", "a"), levels = c("b", "unused", "a")),
  named = list(group = factor(c("b", "a", "b", "a"),
    levels = c("b", "unused", "a"))),
  compound = list(group = factor(c("b", "a", "b", "a"), levels = c("b", "a")),
    number = c(20L, 10L, 10L, 20L)),
  frame = data.frame(group = factor(c("b", "a", "b", "a"), levels = c("b", "a")),
    number = c(20L, 10L, 10L, 20L)))

test_that("aggregate builtins preserve stock values and original group labels", {
  functions <- list(mean = base::mean, median = stats::median,
    min = base::min, max = base::max, sum = base::sum)
  for (multistate in c(FALSE, TRUE)) {
    pair <- .aggregate_fun_pair(multistate)
    before <- as.data.frame(pair$actual)
    for (by in .aggregate_fun_groups()) for (name in names(functions)) {
      expected <- aggregate(pair$stock, by = by, FUN = functions[[name]])
      for (fun in list(functions[[name]], name)) {
        actual <- aggregate(pair$actual, by = by, FUN = fun,
          na.rm = TRUE, ignored = stop("dots must not be evaluated"))
        expect_equal(.aggregate_fun_values(actual, multistate),
          as.numeric(if (multistate) expected$pstate else expected$surv), tolerance = 1e-12)
        expect_identical(actual$newdata, expected$newdata)
        expect_identical(actual[["newdata"]], expected$newdata)
        expect_identical(as.list(actual)$newdata, expected$newdata)
        expect_identical(survfit0(actual)$newdata, expected$newdata)
        expect_identical(summary(actual)$newdata, expected$newdata)
        expect_identical(summary(actual)[["newdata"]], expected$newdata)
        expect_null(as.data.frame(actual)$cumhaz)
      }
    }
    expect_identical(as.data.frame(pair$actual), before)
  }
})

test_that("custom aggregate summaries retain their environment and stock call order", {
  for (multistate in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    pair <- .aggregate_fun_pair(multistate)
    by <- if (grouped) c("b", "a", "b", "a") else NULL
    recorder <- function() {
      calls <- list()
      constant <- .125
      list(fun = function(z) {
        calls[[length(calls) + 1L]] <<- list(values = unname(z),
          names = names(z), kind = typeof(z))
        constant + z[[1L]] + 2 * z[[length(z)]]
      }, calls = function() calls)
    }
    actual_recorder <- recorder()
    stock_recorder <- recorder()
    actual <- aggregate(pair$actual, by = by, FUN = actual_recorder$fun,
      ignored = stop("dots must not be evaluated"))
    stock <- aggregate(pair$stock, by = by, FUN = stock_recorder$fun)
    expect_equal(.aggregate_fun_values(actual, multistate),
      as.numeric(if (multistate) stock$pstate else stock$surv), tolerance = 1e-12)
    expect_equal(actual_recorder$calls(), stock_recorder$calls(), tolerance = 1e-12)
    expect_identical(actual_recorder$calls()[[1L]]$kind, "integer")
    expect_identical(actual$newdata, stock$newdata)
    custom_summary <- function(z) tail(z, 1L)
    named <- aggregate(pair$actual, by = by, FUN = "custom_summary")
    expected_named <- aggregate(pair$stock, by = by, FUN = custom_summary)
    expect_equal(.aggregate_fun_values(named, multistate),
      as.numeric(if (multistate) expected_named$pstate else expected_named$surv),
      tolerance = 1e-12)
  }
})

test_that("aggregate validates every group and propagates callback errors", {
  for (multistate in c(FALSE, TRUE)) {
    pair <- .aggregate_fun_pair(multistate)
    by <- c("b", "a", "b", "a")
    for (answer in list(TRUE, "1", c(1, 2), list(1))) {
      calls <- list()
      fun <- local({ value <- answer; function(z) {
        calls[[length(calls) + 1L]] <<- z
        value
      } })
      expect_error(aggregate(pair$actual, by = by, FUN = fun),
        "FUN must return a single value summary")
      expect_identical(calls, list(c(2L, 4L), c(1L, 3L)))
    }
    calls <- 0L
    fun <- function(z) {
      calls <<- calls + 1L
      if (!is.integer(z)) stop("aggregate callback sentinel")
      sum(z)
    }
    expect_error(aggregate(pair$actual, by = by, FUN = fun), "aggregate callback sentinel")
    expect_identical(calls, 3L)
    expect_error(aggregate(pair$actual, by = by,
      FUN = function(z, scale) sum(z) * scale, scale = 3), "scale.*missing")
    expect_error(aggregate(pair$actual, by = by, FUN = "missing_summary_function"),
      "missing_summary_function")
    expect_error(aggregate(pair$actual, by = 1:3, FUN = sum), "same length")
    missing <- aggregate(pair$actual, by = by, FUN = function(z) NA_integer_)
    expect_true(all(is.na(.aggregate_fun_values(missing, multistate))))
    for (answer in list(factor("a"), as.Date("2026-10-09"))) {
      fun <- local({ value <- answer; function(z) value })
      actual <- aggregate(pair$actual, by = by, FUN = fun)
      stock <- aggregate(pair$stock, by = by, FUN = fun)
      expect_equal(.aggregate_fun_values(actual, multistate),
        as.numeric(if (multistate) stock$pstate else stock$surv), tolerance = 1e-12)
    }
  }
})

test_that("aggregate metadata survives CoxMS selections and saved Python snapshots", {
  pair <- .aggregate_fun_pair(TRUE)
  by <- .aggregate_fun_groups()$frame
  actual <- aggregate(pair$actual, by = by, FUN = function(z) sum(z))
  stock <- aggregate(pair$stock, by = by, FUN = function(z) sum(z))
  selected <- actual[c(4L, 1L, 4L), , drop = FALSE]
  expected <- stock[c(4L, 1L, 4L), , drop = FALSE]
  expect_identical(selected$newdata, expected$newdata)
  expect_identical(summary(selected)$newdata, expected$newdata)
  expect_equal(as.numeric(selected$pstate), as.numeric(expected$pstate), tolerance = 1e-12)
  restored <- unserialize(serialize(actual, NULL))
  expect_identical(restored$newdata, stock$newdata)
  expect_equal(as.numeric(restored$pstate), as.numeric(stock$pstate), tolerance = 1e-12)
})

test_that("aggregate mean and median honor overridden numeric and default methods", {
  pair <- .aggregate_fun_pair()
  with_method <- function(generic, kind, code) {
    fun <- get(generic, mode = "function")
    registry <- get(".__S3MethodsTable__.", envir = environment(fun))
    key <- paste0(generic, ".", kind)
    previous <- if (exists(key, envir = registry, inherits = FALSE)) {
      get(key, envir = registry, inherits = FALSE)
    } else NULL
    on.exit({
      if (is.null(previous)) rm(list = key, envir = registry)
      else assign(key, previous, envir = registry)
    })
    default <- get(paste0(generic, ".default"), envir = environment(fun))
    registerS3method(generic, kind, function(z, ...) default(z, ...) + .125,
      envir = environment(fun))
    force(code)
  }
  for (generic in c("mean", "median")) for (kind in c("integer", "numeric", "default")) {
    values <- with_method(generic, kind, {
      actual <- aggregate(pair$actual, by = c("b", "a", "b", "a"), FUN = generic)
      stock <- aggregate(pair$stock, by = c("b", "a", "b", "a"), FUN = generic)
      list(actual = as.data.frame(actual)$surv, stock = as.numeric(stock$surv))
    })
    expect_equal(values$actual, values$stock, tolerance = 1e-12,
      info = paste(generic, kind))
  }
  for (kind in c("integer", "numeric", "default")) {
    values <- with_method("mean", kind, {
      lapply(list(NULL, rep("whole", 4), c("b", "a", "b", "a")), function(by) {
        actual <- aggregate(pair$actual, by = by)
        stock <- aggregate(pair$stock, by = by)
        list(actual = as.data.frame(actual)$surv, stock = as.numeric(stock$surv))
      })
    })
    for (value in values) expect_equal(value$actual, value$stock, tolerance = 1e-12)
  }
})

test_that("aggregate collapses the data margin of ordinary stratified Cox curves", {
  pair <- .aggregate_fun_pair(stratified = TRUE)
  expect_identical(dim(pair$actual), dim(pair$stock))
  for (by in list(NULL, rep("same", 4), .aggregate_fun_groups()$compound)) {
    actual <- aggregate(pair$actual, by = by, FUN = function(z) max(z) - min(z))
    stock <- aggregate(pair$stock, by = by, FUN = function(z) max(z) - min(z))
    expect_identical(dim(actual), dim(stock))
    expect_equal(as.data.frame(actual)$surv, as.numeric(stock$surv), tolerance = 1e-12)
    expect_identical(actual$newdata, stock$newdata)
    expect_equal(as.data.frame(actual)$time,
      rep(stock$time, if (is.matrix(stock$surv)) ncol(stock$surv) else 1L))
  }
})
