.fg_data <- function(n = 12L) {
  data.frame(time = seq_len(n),
    event = factor(rep(c("a", "b", "censor"), length.out = n), levels = c("censor", "a", "b")),
    x = seq_len(n) / 2, group = ordered(rep(c("z", "a"), length.out = n), levels = c("z", "a")),
    wt = rep(c(1, 2, 3), length.out = n), id = seq_len(n))
}

# The delay test in stock R indexes unsorted rows by positions in the sorted data.
.fg_reference <- function() {
  fun <- survival::finegray
  code <- paste(deparse(body(fun), width.cutoff = 500L), collapse = "\n")
  stopifnot(grepl("Y[first, 1]", code, fixed = TRUE))
  body(fun) <- str2lang(sub("Y[first, 1]", "Y[index[first], 1]", code, fixed = TRUE))
  fun
}

test_that("Fine-Gray retains arbitrary R formula columns without reparsing", {
  data <- .fg_data()
  shift <- function(x) sqrt(x + 2)
  matrix_term <- function(x) cbind(x, x^2)
  formulas <- list(Surv(time, event) ~ shift(x) + group,
    Surv(time, event) ~ matrix_term(x) + I(x^2),
    Surv(time, event) ~ poly(x, 2) + offset(log(x)),
    Surv(time, event) ~ x*group,
    Surv(time, event) ~ 1,
    Surv(time, event) ~ group + strata(wt),
    Surv(time, event) ~ x + strata(group) + strata(wt))
  for (formula in formulas) {
    expect_equal(finegray(formula, data, weights = wt, count = "extra rows"),
      survival::finegray(formula, data, weights = wt, count = "extra rows"), tolerance = 1e-12)
  }
  env <- list2env(as.list(data), parent = environment())
  formula <- as.formula("Surv(time, event) ~ shift(x)+group", env = env)
  expect_equal(finegray(formula), survival::finegray(formula))
  expect_equal(finegray(formula, data = NULL), survival::finegray(formula, data = NULL))
})

test_that("Fine-Gray evaluates response and transformed covariates once", {
  data <- .fg_data()
  response_calls <- covariate_calls <- 0L
  response <- function() { response_calls <<- response_calls + 1L; Surv(data$time, data$event) }
  transform <- function(x) { covariate_calls <<- covariate_calls + 1L; x + 1 }
  result <- finegray(response() ~ transform(x), data)
  expect_equal(response_calls, 1L)
  expect_equal(covariate_calls, 1L)
  expect_equal(result[[1L]], survival::finegray(Surv(time, event) ~ I(x + 1), data)[[1L]])
})

test_that("Fine-Gray source rows preserve classes, subsets and missing covariates", {
  data <- .fg_data()
  data$date <- as.Date("2020-01-01") + data$id
  data$x[c(2, 5)] <- NA
  formula <- Surv(time, event) ~ group+I(x^2)+date
  for (action in list(na.pass, na.omit, na.exclude)) {
    expect_equal(finegray(formula, data, subset = c(10, 1, 5, 2, 8, 7, 4, 3), na.action = action),
      survival::finegray(formula, data, subset = c(10, 1, 5, 2, 8, 7, 4, 3), na.action = action))
  }
  result <- finegray(formula, data)
  expect_s3_class(result$group, "ordered")
  expect_s3_class(result[["I(x^2)"]], "AsIs")
  expect_s3_class(result$date, "Date")
})

test_that("prepared Fine-Gray numerics match R across randomized strata and delayed entry", {
  reference <- .fg_reference()
  for (seed in seq_len(40L)) {
    set.seed(seed)
    n <- 30L
    data <- .fg_data(n)
    data$time <- sample(3:25, n, replace = TRUE)
    data$event <- factor(sample(c("censor", "a", "b"), n, replace = TRUE), levels = c("censor", "a", "b"))
    data$wt <- runif(n, 0.5, 2)
    data$group <- factor(sample(letters[1:3], n, replace = TRUE))
    counting <- seed %% 2L == 0L
    data$start <- sample(0:2, n, replace = TRUE)
    # An early censoring activates the delayed-entry correction for late subjects.
    data$time[1] <- 1
    data$start[1] <- 0
    data$event[1] <- "censor"
    data$event[2:3] <- c("a", "b")
    formula <- if (counting) Surv(start, time, event) ~ x+strata(group) else Surv(time, event) ~ x+strata(group)
    actual <- tryCatch(finegray(formula, data, id = id, weights = wt, count = "added"), error = identity)
    expected <- tryCatch(reference(formula, data, id = id, weights = wt, count = "added"), error = identity)
    if (inherits(actual, "error")) {
      # Stock R divides by zero on some delayed-entry curves; the port rejects it.
      expect_match(conditionMessage(actual), "censoring probability is zero")
      expect_true(inherits(expected, "error") || any(!is.finite(expected$fgwt)))
    } else {
      expect_false(inherits(expected, "error"))
      expect_equal(actual, expected, tolerance = 1e-12)
    }
  }
})

test_that("Fine-Gray counting histories are checked and near ties are resolved", {
  data <- .fg_data(6L)
  data$id <- rep(1:3, each = 2)
  data$start <- c(0, 1+1e-12, 0, 2, 0, 3)
  data$time <- c(1, 4, 2, 5, 3, 6)
  data$event <- factor(c("censor", "a", "censor", "b", "censor", "a"), levels = c("censor", "a", "b"))
  formula <- Surv(start, time, event) ~ x
  expect_equal(finegray(formula, data, id = id), survival::finegray(formula, data, id = id))
  expect_error(finegray(formula, data, id = id, timefix = FALSE), "gaps in time")
  expect_error(finegray(formula, data), "requires a subject id")
  data$event[1] <- "a"
  expect_error(finegray(formula, data, id = id), "transition before")
})

test_that("Fine-Gray validates formula arguments and uses no reference fallback", {
  data <- .fg_data()
  expect_error(finegray(), "formula argument")
  expect_error(finegray(time ~ x, data), "survival object")
  expect_error(finegray(Surv(time, event) ~ x+cluster(id), data), "cluster")
  expect_error(finegray(Surv(time, event) ~ x, data, subset = FALSE), "No \\(non-missing\\)")
  expect_error(finegray(Surv(time, event) ~ x, data, etype = "absent"), "not in the data")
  expect_warning(result <- finegray(Surv(time, event) ~ x, data, etype = c("b", "a")), "first endpoint")
  expect_identical(attr(result, "event"), "b")
  own <- finegray
  local_mocked_bindings(finegray = function(...) stop("reference called"), .package = "survival")
  local_mocked_bindings(.call_r_api = function(...) stop("Python formula called"),
    .package = "survivalr")
  expect_s3_class(own(Surv(time, event) ~ log(x)+group, data), "data.frame")
})

test_that("one-row and direct Fine-Gray inputs retain vector shape", {
  data <- .fg_data(1L)
  expect_equal(finegray(Surv(time, event) ~ x, data), survival::finegray(Surv(time, event) ~ x, data))
  expect_equal(finegray(tstart = 0, tstop = 1, ctime = 2, cprob = 1, extend = TRUE, keep = TRUE),
    data.frame(row = 1L, start = 0, end = 2, wt = 1, add = 0L))
  data <- .fg_data()
  expect_equal(finegray(Surv(time, event) ~ x, data, timefix = 0),
    survival::finegray(Surv(time, event) ~ x, data, timefix = 0))
  expect_error(finegray(Surv(time, event) ~ x, data, timefix = NA), "missing value")
  data$id[1] <- NA
  expect_error(finegray(Surv(time, event) ~ x, data, id = id), "id must not contain missing")
  data$fgstart <- 8
  expect_equal(finegray(Surv(time, event) ~ fgstart, data), survival::finegray(Surv(time, event) ~ fgstart, data))
})
