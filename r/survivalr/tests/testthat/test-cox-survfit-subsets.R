.cox_subset_pair <- function(stratified = TRUE, single = FALSE, tiny = FALSE, narrow = FALSE) {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- if (tiny) data.frame(time = 1:8, status = 1L, x = rep(c(0, 1), 4),
    g = factor(c(rep("a", 7), "b"))) else data.frame(time = 1:12,
    status = rep(c(1L, 1L, 0L), 4), x = rep(c(0, 1, 2), 4),
    z = c(0, 1, .5, -1, 2, .7, 1, 0, -1, .5, 2, 1),
    g = factor(rep(c("a", "b"), each = 6L)))
  actual_formula <- if (tiny || narrow) Surv(time, status) ~ x + strata(g)
    else if (stratified) Surv(time, status) ~ x + z + strata(g)
    else Surv(time, status) ~ x + z
  stock_formula <- if (tiny || narrow) survival::Surv(time, status) ~ x + strata(g)
    else if (stratified) survival::Surv(time, status) ~ x + z + strata(g)
    else survival::Surv(time, status) ~ x + z
  initial <- if (tiny || narrow) .25 else c(.25, -.125)
  actual_fit <- coxph(actual_formula, data, init = initial, iter.max = 0)
  stock_fit <- survival::coxph(stock_formula, data, init = initial,
    control = survival::coxph.control(iter.max = 0))
  newdata <- data.frame(x = c(-1, 0, 2), z = c(1, 0, -1),
    category = factor(c("y", "x", "y"), levels = c("x", "unused", "y")),
    row.names = c("two", "one", "three"))
  if (single) newdata <- newdata[1L, , drop = FALSE]
  if (narrow) newdata <- newdata["x"]
  list(actual = survfit(actual_fit, newdata = newdata),
    stock = survival::survfit(stock_fit, newdata = newdata))
}

.cox_subset_expect <- function(actual, stock) {
  expect_identical(dim(actual), dim(stock))
  actual_fields <- as.list(actual)
  for (name in c("n", "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
    "std.err", "std.chaz", "lower", "upper", "strata", "type", "logse", "conf.int",
    "conf.type", "start.time", "newdata")) {
    expect_equal(actual_fields[[name]], stock[[name]], tolerance = 2e-12, info = name)
  }
  expect_identical(actual$newdata, stock$newdata)
  if (length(stock$time) && (!is.matrix(stock$surv) || ncol(stock$surv)) &&
      (is.null(stock$newdata) || is.data.frame(stock$newdata))) {
    actual0 <- as.list(survfit0(actual))
    stock0 <- survival::survfit0(stock)
    for (name in c("time", "n.risk", "n.event", "surv", "strata", "newdata"))
      expect_equal(actual0[[name]], stock0[[name]], tolerance = 2e-12, info = name)
    expect_equal(quantile(actual, probs = .5), quantile(stock, probs = .5), tolerance = 2e-12)
    for (at_times in c(FALSE, TRUE)) {
      actual_summary <- if (at_times) summary(actual, times = c(0, 4, 8, 12), extend = TRUE)
        else summary(actual, censored = TRUE)
      stock_summary <- if (at_times) summary(stock, times = c(0, 4, 8, 12), extend = TRUE,
        data.frame = TRUE) else summary(stock, censored = TRUE, data.frame = TRUE)
      for (name in intersect(names(stock_summary), names(actual_summary)))
        expect_equal(actual_summary[[name]], stock_summary[[name]], tolerance = 2e-12,
          ignore_attr = TRUE, info = name)
    }
  }
}

test_that("ordinary Cox curves select independent strata and prediction margins", {
  pair <- .cox_subset_pair()
  before <- as.list(pair$actual)
  selections <- list(
    function(x) x[c(2L, 1L), c(3L, 2L, 1L), drop = FALSE],
    function(x) x[c(2L, 1L, 2L), c(3L, 1L, 3L), drop = FALSE],
    function(x) x[c("b", "a", "b"), c(3L, 1L), drop = FALSE],
    function(x) x[-1L, c(FALSE, TRUE, FALSE), drop = FALSE],
    function(x) x[logical(), , drop = FALSE],
    function(x) x[, integer(), drop = FALSE])
  for (select in selections) .cox_subset_expect(select(pair$actual), select(pair$stock))
  for (drop in c(FALSE, TRUE)) {
    .cox_subset_expect(pair$actual[2L, 2L, drop = drop], pair$stock[2L, 2L, drop = drop])
    .cox_subset_expect(pair$actual[2L, , drop = drop], pair$stock[2L, , drop = drop])
    .cox_subset_expect(pair$actual[, 2L, drop = drop], pair$stock[, 2L, drop = drop])
  }
  expect_identical(as.list(pair$actual), before)
  expect_identical(pair$actual[], pair$actual)
})

test_that("ordinary Cox single subscripts retain stock curve and drop conventions", {
  for (pair in list(.cox_subset_pair(), .cox_subset_pair(FALSE), .cox_subset_pair(single = TRUE))) {
    for (selection in list(c(2L, 1L), 1L, c(2L, 1L, 2L), integer())) {
      if (!length(selection) && length(dim(pair$stock)) == 2L) next
      for (drop in c(FALSE, TRUE))
        .cox_subset_expect(pair$actual[selection, drop = drop], pair$stock[selection, drop = drop])
    }
  }
  pair <- .cox_subset_pair()
  for (selection in list(1:6, 6:1, c(4L, 1L, 4L)))
    .cox_subset_expect(pair$actual[selection], pair$stock[selection])
  tiny <- .cox_subset_pair(tiny = TRUE)
  .cox_subset_expect(tiny$actual[2L, 2L], tiny$stock[2L, 2L])
  expect_true(is.matrix(tiny$actual[2L, 2L]$surv))
})

test_that("ordinary Cox repeated labels survive selection and saved snapshots", {
  pair <- .cox_subset_pair()
  actual <- pair$actual[c(2L, 1L, 2L), c(3L, 1L), drop = FALSE]
  stock <- pair$stock[c(2L, 1L, 2L), c(3L, 1L), drop = FALSE]
  expect_identical(names(actual$strata), c("b", "a", "b"))
  .cox_subset_expect(actual[c("b", "a"), c(2L, 1L), drop = FALSE],
    stock[c("b", "a"), c(2L, 1L), drop = FALSE])
  restored <- unserialize(serialize(actual, NULL))
  .cox_subset_expect(restored, stock)
  .cox_subset_expect(restored["b", , drop = FALSE], stock["b", , drop = FALSE])
})

test_that("ordinary Cox invalid subscripts fail with stock dimension errors", {
  pair <- .cox_subset_pair()
  expect_error(pair$actual[3L, ], "subscript out of bounds")
  expect_error(pair$actual[, 4L], "subscript out of bounds")
  expect_error(pair$actual["missing", ], "strata.*not matched")
  expect_error(pair$actual[NA_integer_, ], "subscript out of bounds")
  expect_error(pair$actual[, "one"], "no 'dimnames' attribute for array")
  expect_error(pair$actual[1L, 1L, 1L], "incorrect number of dimensions")
  plain <- .cox_subset_pair(FALSE)
  expect_error(plain$actual[, 1L], "incorrect number of dimensions")
})

test_that("one-coefficient Cox initial values retain their vector shape", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(time = 1:8, status = rep(c(1L, 0L), 4), x = rep(c(0, 1), 4))
  for (initial in c(0, .25)) {
    actual <- coxph(Surv(time, status) ~ x, data, init = initial, iter.max = 0)
    stock <- survival::coxph(survival::Surv(time, status) ~ x, data, init = initial,
      control = survival::coxph.control(iter.max = 0))
    expect_equal(coef(actual), coef(stock), tolerance = 2e-12)
    expect_equal(logLik(actual), logLik(stock), tolerance = 2e-12, ignore_attr = TRUE)
  }
})

test_that("ordinary Cox selections drop one-column prediction tables to vectors", {
  pair <- .cox_subset_pair(narrow = TRUE)
  selections <- list(function(x) x[c(2L, 1L), c(3L, 1L), drop = FALSE],
    function(x) x[, 2L, drop = FALSE], function(x) x[2L, , drop = FALSE],
    function(x) x[, integer(), drop = FALSE])
  for (select in selections) .cox_subset_expect(select(pair$actual), select(pair$stock))
  actual <- selections[[1L]](pair$actual)
  stock <- selections[[1L]](pair$stock)
  expect_identical(actual$newdata, c(2, -1))
  expect_error(stock[, 2L, drop = FALSE], "incorrect number of dimensions")
  expect_equal(actual[, 2L, drop = FALSE]$surv, stock$surv[, 2L, drop = FALSE])
  .cox_subset_expect(unserialize(serialize(actual, NULL)), stock)
})
