.fit_control_warning_capture <- function(expr) {
  messages <- character()
  output <- capture.output(
    error_output <- capture.output(
      value <- withCallingHandlers(
        force(expr),
        warning = function(condition) {
          messages <<- c(messages, conditionMessage(condition))
          invokeRestart("muffleWarning")
        }
      ),
      type = "message"
    )
  )
  list(value = value, messages = messages, output = c(output, error_output))
}

.fit_control_diagnostic_data <- function() {
  data.frame(
    start = c(0, 0, 0, 0, 1, 0, 1, 2, 0, 1),
    time = c(1, 2, 2, 3, 4, 4, 5, 6, 7, 8),
    status = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 1),
    x = c(0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.9, 0.6, 1.2),
    set = rep(1:5, each = 2),
    case = c(1, 0, 0, 1, 1, 0, 0, 1, 1, 0)
  )
}

.skip_fit_control_python <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )
}

test_that("formula fit diagnostics signal R warnings on each call", {
  .skip_fit_control_python()
  data <- .fit_control_diagnostic_data()
  fitters <- list(
    list(
      bridge = function(maxit) coxph(
        Surv(time, status) ~ x, data = data,
        control = coxph.control(iter.max = maxit)
      ),
      reference = function(maxit) survival::coxph(
        survival::Surv(time, status) ~ x, data = data,
        control = survival::coxph.control(iter.max = maxit)
      )
    ),
    list(
      bridge = function(maxit) survreg(
        Surv(time, status) ~ x, data = data, dist = "weibull",
        control = survreg.control(maxiter = maxit)
      ),
      reference = function(maxit) survival::survreg(
        survival::Surv(time, status) ~ x, data = data, dist = "weibull",
        control = survival::survreg.control(maxiter = maxit)
      )
    ),
    list(
      bridge = function(maxit) clogit(
        case ~ x + strata(set), data = data, method = "exact",
        control = coxph.control(iter.max = maxit)
      ),
      # clogit delegates to this Cox fit. Calling it directly also avoids
      # resolving its generated coxph call to the bridge in the test scope.
      reference = function(maxit) survival::coxph(
        survival::Surv(rep(1, nrow(data)), case) ~ x + survival::strata(set),
        data = data, method = "exact",
        control = survival::coxph.control(iter.max = maxit)
      )
    )
  )
  for (fitter in fitters) {
    for (maxit in c(0L, 1L, 2L, 40L)) {
      reference <- .fit_control_warning_capture(fitter$reference(maxit))
      expect_identical(
        reference$messages,
        if (maxit == 2L) "Ran out of iterations and did not converge" else character()
      )
      for (repeat_call in seq_len(2L)) {
        actual <- .fit_control_warning_capture(fitter$bridge(maxit))
        expect_identical(actual$messages, reference$messages)
        expect_identical(actual$output, character())
        expect_s3_class(actual$value, "survival_py_model")
      }
    }
  }
})

test_that("fit warnings obey R warning controls and preserve errors", {
  .skip_fit_control_python()
  data <- .fit_control_diagnostic_data()
  suppressed <- .fit_control_warning_capture(suppressWarnings(coxph(
    Surv(time, status) ~ x, data = data, control = coxph.control(iter.max = 2)
  )))
  expect_identical(suppressed$messages, character())
  expect_identical(suppressed$output, character())
  expect_s3_class(suppressed$value, "survival_py_coxph")

  previous_warn <- getOption("warn")
  tryCatch({
    options(warn = 2)
    expect_error(
      coxph(Surv(time, status) ~ x, data = data, control = coxph.control(iter.max = 2)),
      "Ran out of iterations and did not converge"
    )
  }, finally = options(warn = previous_warn))
  expect_error(
    coxph(Surv(time, status) ~ x, data = data, weights = rep(-1, nrow(data))),
    "weights"
  )
})

test_that("default AFT convergence agrees with an explicit R control object", {
  .skip_fit_control_python()
  data <- data.frame(
    time = 1:6,
    status = c(1, 1, 0, 1, 0, 1),
    x = c(0.2, 0.4, 0.1, 0.8, 1.0, 1.2)
  )
  actual <- survreg(Surv(time, status) ~ x, data = data, dist = "weibull")
  explicit <- survreg(
    Surv(time, status) ~ x, data = data, dist = "weibull",
    control = survival::survreg.control()
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ x, data = data, dist = "weibull"
  )
  expect_identical(coef(actual), coef(explicit))
  expect_identical(vcov(actual), vcov(explicit))
  expect_identical(as.numeric(logLik(actual)), as.numeric(logLik(explicit)))
  expect_equal(unname(coef(actual)), unname(coef(reference)), tolerance = 1e-8)
  expect_equal(unname(vcov(actual)), unname(vcov(reference)), tolerance = 1e-8)
  expect_equal(as.numeric(logLik(actual)), as.numeric(logLik(reference)), tolerance = 1e-8)
})

test_that("low-level fit diagnostics match native R without duplicate warnings", {
  .skip_fit_control_python()
  data <- .fit_control_diagnostic_data()
  fitters <- list(
    list(bridge = coxph.fit, reference = survival::coxph.fit, counting = FALSE, method = "efron"),
    list(bridge = agreg.fit, reference = survival::agreg.fit, counting = TRUE, method = "efron"),
    list(bridge = agexact.fit, reference = survival::agexact.fit, counting = TRUE, method = "exact")
  )
  for (fitter in fitters) {
    response <- if (fitter$counting) {
      survival::Surv(data$start, data$time, data$status)
    } else {
      survival::Surv(data$time, data$status)
    }
    for (maxit in c(0L, 1L, 2L)) {
      arguments <- list(
        x = matrix(data$x, ncol = 1L), y = response, strata = NULL,
        offset = rep(0, nrow(data)), init = 0,
        control = survival::coxph.control(iter.max = maxit),
        weights = rep(1, nrow(data)), method = fitter$method,
        rownames = as.character(seq_len(nrow(data)))
      )
      actual <- .fit_control_warning_capture(do.call(fitter$bridge, arguments))
      reference <- .fit_control_warning_capture(do.call(fitter$reference, arguments))
      expect_identical(actual$messages, reference$messages)
      expect_identical(actual$output, character())
      expect_identical(length(actual$messages), if (maxit > 1L) 1L else 0L)
    }
  }

  # The low-level AFT path already signals an R warning itself, and must
  # remain separate from the captured Python formula-fit warning path.
  for (maxit in c(0L, 1L, 2L)) {
    arguments <- list(
      x = cbind(1, data$x), y = survival::Surv(data$time, data$status),
      weights = rep(1, nrow(data)), offset = rep(0, nrow(data)),
      init = c(0, 0, 0), controlvals = survival::survreg.control(maxiter = maxit),
      dist = "gaussian", scale = 0, nstrat = 1, strata = rep(1L, nrow(data))
    )
    actual <- .fit_control_warning_capture(do.call(survreg.fit, arguments))
    reference <- .fit_control_warning_capture(do.call(survival::survreg.fit, arguments))
    expect_identical(actual$messages, reference$messages)
    expect_identical(actual$output, character())
    expect_identical(length(actual$messages), if (maxit > 1L) 1L else 0L)
  }
})

test_that("no-event Cox diagnostics distinguish formula and low-level fits", {
  .skip_fit_control_python()
  data <- .fit_control_diagnostic_data()
  data$status <- 0L
  for (counting in c(FALSE, TRUE)) {
    actual <- .fit_control_warning_capture(if (counting) {
      coxph(Surv(start, time, status) ~ x, data = data, control = coxph.control(iter.max = 2))
    } else {
      coxph(Surv(time, status) ~ x, data = data, control = coxph.control(iter.max = 2))
    })
    reference <- .fit_control_warning_capture(if (counting) {
      survival::coxph(
        survival::Surv(start, time, status) ~ x, data = data,
        control = survival::coxph.control(iter.max = 2)
      )
    } else {
      survival::coxph(
        survival::Surv(time, status) ~ x, data = data,
        control = survival::coxph.control(iter.max = 2)
      )
    })
    expect_identical(actual$messages, character())
    expect_identical(actual$messages, reference$messages)
    expect_identical(actual$output, character())
  }

  for (maxit in c(0L, 1L, 2L, 20L)) {
    arguments <- list(
      x = matrix(data$x, ncol = 1L),
      y = survival::Surv(data$time, data$status), strata = NULL,
      offset = rep(0, nrow(data)), init = 0,
      control = survival::coxph.control(iter.max = maxit),
      weights = rep(1, nrow(data)), method = "efron",
      rownames = as.character(seq_len(nrow(data)))
    )
    actual <- .fit_control_warning_capture(do.call(coxph.fit, arguments))
    reference <- .fit_control_warning_capture(do.call(survival::coxph.fit, arguments))
    expect_identical(actual$messages, reference$messages)
    expect_identical(length(actual$messages), if (maxit > 1L) 1L else 0L)
    expect_identical(actual$output, character())

    arguments$y <- survival::Surv(data$start, data$time, data$status)
    expect_error(do.call(agreg.fit, arguments), "Can't fit a Cox model with 0 failures")
    expect_error(do.call(survival::agreg.fit, arguments), "Can't fit a Cox model with 0 failures")
  }
})
