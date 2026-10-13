.aj_optional_data <- function(counting = FALSE) {
  data <- data.frame(
    time = c(1, 3, 5, 7, 8, 9, 2, 4, 6, 10, 11, 12),
    state = factor(
      c("death", "censor", "recovery", "death", "censor", "death",
        "censor", "death", "recovery", "censor", "recovery", "death"),
      levels = c("censor", "death", "recovery")
    ),
    group = factor(rep(c("a", "b"), each = 6L)),
    id = seq_len(12L)
  )
  if (!counting) return(data)
  first <- data
  first$start <- 0
  first$stop <- data$time / 2
  first$state[] <- "censor"
  data$start <- data$time / 2
  data$stop <- data$time
  rbind(first, data)
}

.aj_optional_numeric <- function(actual, expected, ignore_attr = FALSE, field = "numeric") {
  if (is.null(actual) || is.null(expected)) {
    expect_identical(actual, expected)
    return(invisible(NULL))
  }
  expect_identical(length(actual), length(expected))
  if (!ignore_attr) expect_identical(attributes(actual), attributes(expected))
  actual <- as.numeric(actual)
  expected <- as.numeric(expected)
  expect_identical(is.na(actual), is.na(expected))
  expect_identical(is.nan(actual), is.nan(expected))
  expect_identical(is.infinite(actual), is.infinite(expected))
  expect_identical(actual[is.infinite(expected)], expected[is.infinite(expected)])
  finite <- is.finite(expected)
  expect_true(all(abs(actual[finite] - expected[finite]) <=
    64 * .Machine$double.eps * pmax(1, abs(expected[finite]))),
    info = paste(field, "maximum difference", max(c(0, abs(actual[finite] - expected[finite])))))
}

.aj_optional_compare <- function(actual, expected) {
  expect_identical(names(actual), setdiff(names(expected), "call"))
  fields <- as.list(actual)
  expect_identical(names(fields), names(actual))
  expect_identical(length(actual), length(fields))
  expect_identical(dim(actual), dim(expected))
  for (name in c(
    "time", "n.risk", "n.event", "n.censor", "n.enter", "pstate",
    "n.transition", "cumhaz", "std.err", "std.chaz", "std.auc", "lower", "upper"
  )) {
    .aj_optional_numeric(fields[[name]], expected[[name]])
  }
  for (name in c("states", "type", "conf.type", "conf.int", "logse", "t0")) {
    expect_equal(fields[[name]], expected[[name]], tolerance = 1e-12)
  }
  for (name in c("n", "n.id", "strata", "p0")) {
    .aj_optional_numeric(unname(fields[[name]]), unname(expected[[name]]))
  }
  expect_equal(fields$transitions, expected$transitions)
  expect_identical(actual$pstate, fields$pstate)
  expect_identical(actual[["pstate"]], fields$pstate)
  expect_identical(actual$conf.type, fields$conf.type)
  expect_identical(actual$conf.int, fields$conf.int)
  expect_identical(actual$logse, fields$logse)
  expect_identical(actual$std_err, fields$std.err)
  expect_true(is.matrix(fields$pstate))
  if (is.null(expected$std.err)) {
    expect_false("logse" %in% names(fields))
  } else {
    expect_identical(fields$logse, FALSE)
    expect_identical(dim(fields$std.err), dim(expected$std.err))
  }
}

.aj_optional_compare_frame <- function(actual, expected) {
  if ("strata" %in% names(expected)) {
    expected$strata <- sub("^group=", "", as.character(expected$strata))
  }
  expect_identical(names(actual), names(expected))
  expect_identical(nrow(actual), nrow(expected))
  for (name in names(expected)) {
    if (name %in% c("strata", "state")) {
      expect_identical(as.character(actual[[name]]), as.character(expected[[name]]))
    } else {
      .aj_optional_numeric(actual[[name]], expected[[name]], ignore_attr = TRUE, field = name)
    }
  }
}

.aj_optional_reference_frame <- function(fit) {
  # Stock summary(data.frame = TRUE) tries to bind NULL confidence columns
  # whenever std.err exists. Assemble its actual components for CI-off fits.
  states <- length(fit$states)
  result <- data.frame(
    time = rep(fit$time, states),
    n.risk = as.numeric(fit$n.risk),
    n.event = as.numeric(fit$n.event),
    n.censor = as.numeric(fit$n.censor),
    pstate = as.numeric(fit$pstate)
  )
  for (name in c("std.err", "lower", "upper")) {
    if (!is.null(fit[[name]])) result[[name]] <- as.numeric(fit[[name]])
  }
  if (!is.null(fit$strata)) result$strata <- rep(fit$strata, states)
  result$state <- rep(fit$states, each = length(fit$time))
  result
}

test_that("AJ optional uncertainty preserves components and public accessors", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  specs <- list(
    no_se = list(se.fit = FALSE),
    no_conf_type = list(conf.type = "none"),
    no_conf_int = list(conf.int = FALSE),
    with_ci = list()
  )
  for (counting in c(FALSE, TRUE)) for (grouped in c(FALSE, TRUE)) {
    data <- .aj_optional_data(counting)
    response <- if (counting) "Surv(start, stop, state)" else "Surv(time, state)"
    formula <- as.formula(paste(response, "~", if (grouped) "group" else "1"))
    reference_formula <- as.formula(paste("survival::", response, "~",
                                         if (grouped) "group" else "1", sep = ""))
    baseline <- survfit(formula, data = data, id = data$id)
    baseline_censors <- as.data.frame(summary(baseline))$n.censor
    for (options in specs) {
      actual <- do.call(survfit, c(list(formula = formula, data = data,
                                      id = data$id), options))
      expected <- do.call(reference_survfit, c(list(formula = reference_formula,
        data = data, id = data$id), options))
      .aj_optional_compare(actual, expected)
      .aj_optional_compare_frame(
        as.data.frame(actual),
        .aj_optional_reference_frame(summary(expected, censored = TRUE))
      )
      .aj_optional_compare_frame(
        as.data.frame(summary(actual, censored = TRUE)),
        .aj_optional_reference_frame(summary(expected, censored = TRUE))
      )
      ordinary_summary <- as.data.frame(summary(actual))
      reference_summary <- .aj_optional_reference_frame(summary(expected))
      # Stock grouped default summaries flatten censor matrices and recycle the
      # counts across states. All counts agree at each row and requested time.
      compared <- if (grouped) setdiff(names(reference_summary), "n.censor") else
        names(reference_summary)
      .aj_optional_compare_frame(ordinary_summary[compared], reference_summary[compared])
      expect_identical(ordinary_summary$n.censor, baseline_censors)
      .aj_optional_compare_frame(
        as.data.frame(summary(actual, times = c(2, 4, 6), extend = TRUE)),
        .aj_optional_reference_frame(summary(expected, times = c(2, 4, 6), extend = TRUE))
      )
      actual_frame <- model.frame(actual)
      expect_identical(nrow(actual_frame), nrow(model.frame(expected)))
      # The bridge keeps a tabular response rather than a Surv-valued column.
      expect_identical(names(actual_frame), c(
        if (counting) c("start", "stop") else "time", "status",
        if (grouped) "group", "X.id."
      ))
      expect_equal(actual_frame$status, as.integer(data$state) - 1L)
      expect_equal(actual_frame$X.id., data$id)
      if (grouped) expect_identical(as.character(actual_frame$group), as.character(data$group))
      if (counting) {
        expect_equal(actual_frame$start, data$start)
        expect_equal(actual_frame$stop, data$stop)
      } else {
        expect_equal(actual_frame$time, data$time)
      }
      selected <- actual[, "death"]
      selected_reference <- expected[, "death"]
      # R retains all state count columns; compare its selected public state
      # values without changing that established difference in this adapter.
      for (name in c("n.censor", "n.enter")) {
        if (!is.null(selected_reference[[name]])) {
          selected_reference[[name]] <- expected[[name]][, "death", drop = FALSE]
        }
      }
      .aj_optional_compare(selected, selected_reference)
      expect_identical(ncol(selected$pstate), 1L)
      if (!is.null(selected$std.err)) expect_identical(ncol(selected$std.err), 1L)
    }
  }
})

test_that("AJ start times that empty a curve remain rejected", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  data <- .aj_optional_data()
  for (grouped in c(FALSE, TRUE)) for (se in c(FALSE, TRUE)) {
    formula <- if (grouped) Surv(time, state) ~ group else Surv(time, state) ~ 1
    reference_formula <- if (grouped) survival::Surv(time, state) ~ group else
      survival::Surv(time, state) ~ 1
    start <- if (grouped) 10 else 13
    expect_error(survfit(formula, data = data, id = data$id, start.time = start,
                         se.fit = se), "removed all.*at least one curve")
    expect_error(reference_survfit(reference_formula, data = data, id = data$id,
                                  start.time = start, se.fit = se),
                 "removed all.*at least one curve")
  }
})

test_that("ordinary KM confidence omission remains unchanged", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- .aj_optional_data()
  data$status <- as.integer(data$state != "censor")
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  for (options in list(list(), list(se.fit = FALSE), list(conf.type = "none"),
                       list(conf.int = FALSE))) {
    actual <- do.call(survfit, c(list(formula = Surv(time, status) ~ 1,
                                    data = data), options))
    expected <- do.call(reference_survfit, c(list(
      formula = survival::Surv(time, status) ~ 1, data = data), options))
    for (name in c("time", "surv", "lower", "upper")) {
      expect_equal(actual[[name]], expected[[name]], tolerance = 1e-12)
    }
  }
})

test_that("present zero AJ standard errors retain logse for one-state selections", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  data <- .aj_optional_data()
  data$time[] <- 1
  data$state <- factor(rep("death", nrow(data)), levels = c("censor", "death"))
  for (grouped in c(FALSE, TRUE)) for (se in c(FALSE, TRUE)) {
    formula <- if (grouped) Surv(time, state) ~ group else Surv(time, state) ~ 1
    reference_formula <- if (grouped) survival::Surv(time, state) ~ group else
      survival::Surv(time, state) ~ 1
    actual <- survfit(formula, data = data, id = data$id, se.fit = se)
    expected <- reference_survfit(reference_formula, data = data, id = data$id, se.fit = se)
    .aj_optional_compare(actual, expected)
    selected <- actual[, "death"]
    selected_reference <- expected[, "death"]
    expect_identical(dim(selected$pstate), dim(selected_reference$pstate))
    .aj_optional_numeric(selected$pstate, selected_reference$pstate)
    if (se) {
      expect_true(all(selected$std.err == 0))
      expect_identical(dim(selected$std.err), dim(selected_reference$std.err))
      expect_identical(selected$logse, FALSE)
      expect_true("logse" %in% names(selected))
    } else {
      expect_null(selected$std.err)
      expect_false("logse" %in% names(selected))
    }
  }
})
