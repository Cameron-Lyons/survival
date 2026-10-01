.obrien_data <- function() {
  data.frame(time = c(4, 1, 3, 2, 4, 5), status = c(1, 1, 0, 1, 1, 0),
             start = c(1, 0, 0, 1, 0, 3), x = c(2, 5, 1, 5, 3, 2), z = c(4, 1, 2, 4, 1, 2),
             group = factor(c("b", "a", "b", "a", "a", "b"), levels = c("b", "a")),
             id = letters[1:6], row.names = paste0("r", 1:6))
}

.obrien_reference <- function() {
  reference <- survival::survobrien
  body <- paste(deparse(body(reference), width.cutoff = 500L), collapse = "\n")
  corrections <- c(
    "y[, 2] >= temp[x, 1] & strata.keep == temp[x, 2]" = "y[, 1] >= temp[x, 1] & strata.keep == temp[x, 2]",
    "!strata.keep == temp[x, 2]" = "strata.keep == temp[x, 2]",
    "names(m)[stemp$vars]" = "stemp$vars",
    "names(m)[cluster$vars]" = "cluster$vars",
    "Terms[-cluster$tvar]" = "Terms[-cluster$terms]"
  )
  for (from in names(corrections)) {
    stopifnot(grepl(from, body, fixed = TRUE))
    body <- sub(from, corrections[[from]], body, fixed = TRUE)
  }
  body(reference) <- str2lang(body)
  reference
}

test_that("O'Brien evaluates general R formula terms with the shared expansion", {
  data <- .obrien_data()
  shifted <- function(x) log(x + 3)
  matrix_term <- function(x) cbind(x, 2*x)
  tt <- function(x) sin(x)
  formulas <- list(
    Surv(time, status) ~ x + z,
    Surv(time, status) ~ log(x + 1) + shifted(z),
    Surv(time, status) ~ x + I(z),
    Surv(time, status) ~ x + I(z^2),
    Surv(time, status) ~ x + group,
    Surv(time, status) ~ matrix_term(x) + z,
    Surv(time, status) ~ tt(x) + z,
    Surv(start, time, status) ~ x + I(z)
  )
  for (formula in formulas) {
    expect_equal(survobrien(formula, data), survival::survobrien(formula, data), tolerance = 1e-13)
  }
})

test_that("stratified and clustered expansions preserve risk-set order and keepers", {
  data <- .obrien_data()
  reference <- .obrien_reference()
  for (lhs in c("Surv(time, status)", "Surv(start, time, status)")) {
    for (rhs in c("x + strata(group)", "x + group + cluster(id)",
                  "group + x + strata(group) + cluster(id)",
                  "I(z) + cluster(id) + x + strata(group)",
                  "x + strata(group) + strata(z)")) {
      formula <- as.formula(paste(lhs, rhs, sep = "~"))
      expect_equal(survobrien(formula, data), reference(formula, data), tolerance = 1e-13)
    }
  }
  result <- survobrien(Surv(time, status) ~ x+strata(group), data)
  expect_equal(result$.id., c(1L, 6L, 2L, 4L, 5L, 4L, 5L, 5L))
  expect_equal(result$.strata., c(1L, 1L, 2L, 2L, 2L, 3L, 3L, 4L))
})

test_that("protected and cluster columns follow subset and missing-data removal", {
  data <- .obrien_data()
  data$x[3] <- NA
  formula <- Surv(time, status) ~ x+group+I(z)+cluster(id)
  selected <- c(6, 4, 3, 2, 4)
  expected_data <- data[selected, ]
  expected_data <- expected_data[!is.na(expected_data$x), ]
  expect_equal(survobrien(formula, data, subset = selected), .obrien_reference()(formula, expected_data))
  own <- survobrien(Surv(time, status) ~ x+group, data, na.action = na.omit)
  expected <- survival::survobrien(Surv(time, status) ~ x+group, data[!is.na(data$x), ])
  expect_equal(own, expected)
  expect_s3_class(own$group, "factor")
  expect_identical(levels(own$group), levels(data$group))
})

test_that("missing and infinite covariates use R's nonmissing rank denominator", {
  data <- .obrien_data()
  data$x <- c(NA, Inf, -Inf, 2, NaN, 2)
  for (formula in list(Surv(time, status) ~ x+z, Surv(start, time, status) ~ x+z)) {
    expect_equal(survobrien(formula, data, na.action = na.pass),
                 survival::survobrien(formula, data, na.action = na.pass), tolerance = 1e-13)
  }
})

test_that("custom transforms preserve batches, order, names and return types", {
  data <- .obrien_data()
  for (transform in list(function(x) 2*x, function(x) rank(x),
                        function(x) setNames(x - mean(x), paste0("v", seq_along(x))),
                        function(x) as.character(x))) {
    expect_equal(survobrien(Surv(time, status) ~ x+z, data, transform = transform),
                 survival::survobrien(Surv(time, status) ~ x+z, data, transform = transform))
  }
  seen <- list()
  transform <- function(x) { seen[[length(seen) + 1L]] <<- x; x }
  actual <- survobrien(Surv(time, status) ~ x+z, data, transform = transform)
  batches <- seen
  seen <- list()
  expected <- survival::survobrien(Surv(time, status) ~ x+z, data, transform = transform)
  expect_identical(seen, batches)
  expect_equal(actual, expected)
  expect_error(survobrien(Surv(time, status) ~ x, data, transform = function(x) 1), "1 to 1")
  expect_error(survobrien(Surv(time, status) ~ x, data, transform = function(x) {
    if (length(x) == 10L) x else numeric()
  }), "1 to 1")
})

test_that("formula environments work without data and responses are evaluated once", {
  data <- .obrien_data()
  env <- list2env(as.list(data), parent = environment())
  formula <- as.formula("Surv(time, status) ~ x+group+I(z)", env = env)
  expect_equal(survobrien(formula), survival::survobrien(formula, as.data.frame(as.list(env))))
  calls <- 0L
  response <- function() { calls <<- calls + 1L; Surv(data$time, data$status) }
  formula <- response() ~ x
  actual <- survobrien(formula, data)
  expect_equal(calls, 1L)
  expect_equal(actual, survival::survobrien(Surv(time, status) ~ x, data))
})

test_that("no-event data retain empty output columns and invalid inputs fail clearly", {
  data <- .obrien_data()
  data$status <- 0L
  result <- survobrien(Surv(time, status) ~ x+group+I(z), data)
  expect_equal(nrow(result), 0L)
  expect_identical(names(result), c("time", "status", "group", "z", ".id.", "x", ".strata."))
  expect_s3_class(result$group, "factor")
  expect_equal(levels(result$group), levels(data$group))
  expect_error(survobrien(), "formula argument")
  expect_error(survobrien(time ~ x, data), "survival object")
  expect_error(survobrien(Surv(time, status) ~ x*z, data), "iteraction")
  expect_error(survobrien(Surv(time, status) ~ group+I(x), data), "No continuous")
  expect_error(survobrien(Surv(time, status) ~ x, data, subset = FALSE), "No \\(non-missing\\) observations")
})

test_that("O'Brien formula expansion does not call the R reference", {
  data <- .obrien_data()
  bridge <- survobrien
  local_mocked_bindings(survobrien = function(...) stop("reference called"), .package = "survival")
  expect_s3_class(bridge(Surv(time, status) ~ log(x+1)+I(z)+group, data), "data.frame")
})
