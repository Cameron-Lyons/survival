.py_data <- function() {
  data.frame(time = c(10, 30, 20, 5, 0, 12), event = c(1, 0, 1, 0, 0, 1),
    start = c(1, 3, 5, 2, 0, 1), base = c(0, 5, 12, 19, 2, 3),
    group = factor(c("b", "a", "b", "a", "b", "a"), levels = c("b", "a", "unused")),
    sex = c(1, 2, 1, 2, 1, 2), age = c(40, 50, 60, 50, 40, 60)*365.25,
    year = as.Date("2000-01-01") + 0:5, wt = c(1, 2, 0.5, 0, 3, 1),
    row.names = paste0("row", 1:6))
}

.py_compare <- function(actual, expected) {
  actual$call <- expected$call <- NULL
  # R model-frame terms can capture evaluation environments with identical content.
  if (!is.null(actual$terms)) environment(actual$terms) <- environment(expected$terms) <- baseenv()
  if (!is.null(actual$model)) {
    at <- attr(actual$model, "terms")
    et <- attr(expected$model, "terms")
    environment(at) <- environment(et) <- baseenv()
    attr(actual$model, "terms") <- at
    attr(expected$model, "terms") <- et
  }
  expect_equal(actual, expected, tolerance = 1e-12)
}

test_that("all person-years formulas use native tabulation with R metadata", {
  data <- .py_data()
  change <- function(x) factor(x %% 3)
  formulas <- list(time ~ group, cbind(time, event) ~ group,
    Surv(time, event) ~ factor(sex) + group,
    Surv(time, event) ~ change(base),
    Surv(time, event) ~ tcut(base, c(0, 10, 20, 30)) + group,
    Surv(time, event) ~ offset(base) + cluster(sex),
    Surv(time, event) ~ 1)
  for (formula in formulas) for (model in c(FALSE, TRUE)) {
    .py_compare(pyears(formula, data, weights = wt, scale = 1, model = model, x = TRUE, y = TRUE),
      survival::pyears(formula, data, weights = wt, scale = 1, model = model, x = TRUE, y = TRUE))
  }
  for (formula in formulas[-length(formulas)]) {
    .py_compare(pyears(formula, data, weights = wt, scale = 1, data.frame = TRUE),
      survival::pyears(formula, data, weights = wt, scale = 1, data.frame = TRUE))
  }
})

test_that("person-years formula environments and response calls are evaluated once", {
  data <- .py_data()
  env <- list2env(as.list(data), parent = environment())
  formula <- as.formula("Surv(time, event) ~ group", env = env)
  .py_compare(pyears(formula, scale = 1), survival::pyears(formula, scale = 1))
  .py_compare(pyears(formula, data = NULL, scale = 1), survival::pyears(formula, data = NULL, scale = 1))
  response_calls <- column_calls <- 0L
  response <- function() { response_calls <<- response_calls + 1L; Surv(data$time, data$event) }
  term <- function(x) { column_calls <<- column_calls + 1L; factor(x) }
  actual <- pyears(response() ~ term(sex), data, model = TRUE)
  expect_identical(response_calls, 1L)
  expect_identical(column_calls, 1L)
  expect_equal(as.numeric(actual$pyears), as.numeric(survival::pyears(Surv(time, event) ~ sex, data)$pyears))
  expect_identical(attr(attr(actual$model, "terms"), "predvars")[[2L]], quote(response()))
})

test_that("rate-table formulas use native positions and handle numeric matrices", {
  data <- .py_data()
  for (lhs in c("time", "cbind(time)", "cbind(start, time)", "Surv(time, event)")) {
    formula <- as.formula(paste(lhs, "~group"))
    for (expect in c("event", "pyears")) {
      .py_compare(pyears(formula, data, ratetable = survival::survexp.us, expect = expect, model = TRUE),
        survival::pyears(formula, data, ratetable = survival::survexp.us, expect = expect, model = TRUE))
    }
  }
  env <- list2env(as.list(data), parent = environment())
  formula <- as.formula("Surv(time, event) ~ tcut(base,c(0,10,20,30))", env = env)
  .py_compare(pyears(formula, ratetable = survival::survexp.us, rmap = list(age = age + 365.25)),
    survival::pyears(formula, ratetable = survival::survexp.us, rmap = list(age = age + 365.25)))
})

test_that("subsets and missing actions preserve model and factor levels", {
  data <- .py_data()
  data$base[c(2, 4)] <- NA
  formula <- Surv(time, event) ~ factor(base) + group
  for (action in list(na.omit, na.exclude)) {
    .py_compare(pyears(formula, data, subset = c(6, 2, 5, 1, 4), na.action = action, model = TRUE),
      survival::pyears(formula, data, subset = c(6, 2, 5, 1, 4), na.action = action, model = TRUE))
  }
  data$group <- factor(rep("only", nrow(data)))
  .py_compare(pyears(time ~ group, data), survival::pyears(time ~ group, data))
  result <- pyears(time ~ 1, data, data.frame = TRUE, scale = 1)
  expect_identical(names(result$data), c("pyears", "n"))
  expect_equal(result$data$pyears, sum(data$time))
  expect_equal(result$data$n, sum(data$time > 0))
})

test_that("person-years errors and zero-time warnings are explicit", {
  data <- .py_data()
  expect_error(pyears(), "formula argument")
  expect_error(pyears(~group, data), "Follow-up time")
  expect_error(pyears(time ~ group*sex, data), "interaction")
  expect_error(pyears(time ~ group, data, subset = FALSE), "0 observations")
  expect_error(pyears(time ~ group, data, rmap = list(age = age)), "No rate table")
  expect_error(pyears(time ~ group, data, ratetable = NULL), "Invalid rate table")
  expect_error(pyears(time ~ group, data, ratetable = survival::survexp.us, rmap = age), "Invalid rcall")
  expect_error(pyears(time ~ group, data, ratetable = survival::survexp.us, rmap = list(bogus = age)), "Variable not found")
  expect_error(pyears(cbind(start, time, event) ~ group, data), "too many columns")
  data$event[5] <- 1
  expect_warning(pyears(Surv(time, event) ~ group, data), "event and 0 follow-up")
  data$time[5] <- -1
  expect_error(pyears(Surv(time, event) ~ group, data), "Negative survival time")
  expect_error(pyears(time ~ group, data), "Negative follow up time")
})

test_that("person-years does not forward formulas to reference or Python parsers", {
  data <- .py_data()
  bridge <- pyears
  local_mocked_bindings(pyears = function(...) stop("reference called"), .package = "survival")
  local_mocked_bindings(.call_r_api = function(...) stop("Python formula called"))
  expect_s3_class(bridge(time ~ group, data), "pyears")
  expect_s3_class(bridge(time ~ group, data, ratetable = survival::survexp.us), "pyears")
})
