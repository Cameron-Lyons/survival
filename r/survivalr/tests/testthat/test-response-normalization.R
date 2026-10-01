.capture_response <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = warnings)
}

test_that("model frames normalize status before omission and preserve censoring", {
  d <- data.frame(time = 1:8, status = rep(c(1, 2), 4), x = rep(c(1, NA), 4))
  for (action in list(na.omit, na.exclude, na.pass)) {
    own <- model.frame(Surv(time, status) ~ x, d, na.action = action)
    expected <- model.frame(survival::Surv(time, status) ~ x, d, na.action = action)
    expect_equal(unname(unclass(model.response(own))), unname(unclass(model.response(expected))))
    expect_equal(attr(own, "na.action"), attr(expected, "na.action"))
    fit <- survfit(Surv(time, status) ~ 1, d, weights = x, na.action = na.omit)
    expect_equal(sum(as.numeric(fit$n_event)), 0)
    expect_equal(as.numeric(fit$surv), rep(1, 4))
  }
})

test_that("native model-frame Surv uses shared normalization for every response type", {
  d <- data.frame(time = 1:8, start = c(0, 2, 4, 0, 0, 0, 0, 0),
                   end = c(NA, NA, NA, 5, NA, NA, NA, 9),
                   status = rep(c(1, 2), 4), invalid = c(0, 1, 8, 1, 0, 1, 0, 1),
                   interval = c(0, 1, 2, 3, 0, 1, 2, 3),
                   flag = c(TRUE, NA, FALSE, TRUE, FALSE, NA, TRUE, FALSE),
                   integer_time = c(1L, NA_integer_, 3:8),
                   integer_status = c(1L, NA_integer_, 0L, 1L, 0L, 1L, 0L, 1L), x = 1:8)
  d$state <- factor(rep(c("c", "event"), 4), levels = c("c", "other", "event"))
  expressions <- c("Surv(time, status)", "Surv(time, invalid)", "Surv(time, flag)",
                   "Surv(start, time, status)", "Surv(time, status, type='left')",
                   "Surv(time, end, interval, type='interval')", "Surv(time, end, type='interval2')",
                   "Surv(time, state)", "Surv(start, time, state)", "Surv(time)",
                   "Surv(integer_time, integer_status)")
  for (response in expressions) for (action in list(na.omit, na.exclude, na.pass)) {
    own <- .capture_response(model.frame(as.formula(paste(response, "~ x")), d, na.action = action))
    expected <- .capture_response(model.frame(as.formula(paste0("survival::", response, "~ x")), d,
                                               na.action = action))
    expect_equal(own$warnings, expected$warnings)
    actual_y <- model.response(own$value)
    expected_y <- model.response(expected$value)
    expect_equal(unname(unclass(actual_y)), unname(unclass(expected_y)))
    expect_equal(attr(actual_y, "type"), attr(expected_y, "type"))
    expect_equal(attr(actual_y, "states"), attr(expected_y, "states"))
    expect_equal(attr(own$value, "na.action"), attr(expected$value, "na.action"))
  }
})

test_that("shared Surv construction retains scalar inputs and R warnings", {
  for (args in list(list(5), list(5, 1), list(0, 5, 1), list(time = 5, event = 1), list(as.difftime(5, units = "days"), 1),
                    list(5, factor("event", levels = c("c", "event"))))) {
    actual <- do.call(Surv, args)
    expected <- do.call(survival::Surv, args)
    expect_equal(unname(as.matrix(actual)), unname(as.matrix(expected)), ignore_attr = TRUE)
  }
  expect_error(Surv(as.Date("2020-01-01"), 1), "Time variable is not numeric")
  expect_warning(Surv(1:3, c(1, 9, 0)), "Invalid status")
  expect_warning(model.frame(Surv(start, time, event) ~ 1,
                             data.frame(start = 2, time = 1, event = factor("a", levels = c("c", "a")))),
                  "Stop time")
})

test_that("formula fitters omit invalid normalized responses with aligned weights", {
  set.seed(819)
  d <- data.frame(time = rexp(80) + .2, status = rbinom(80, 1, .7),
                   x = rnorm(80), w = runif(80, .5, 2))
  d$status[c(4, 19)] <- 8
  d$x[c(7, 22)] <- NA
  for (family in c("coxph", "survreg")) for (action in list(na.omit, na.exclude)) {
    own <- get(family, asNamespace("survivalr"))
    ref <- get(family, asNamespace("survival"))
    fit <- .capture_response(own(Surv(time, status) ~ x, d, weights = w, na.action = action))
    expected <- .capture_response(ref(survival::Surv(time, status) ~ x, d, weights = w, na.action = action))
    expect_equal(fit$warnings, expected$warnings)
    expect_equal(unname(coef(fit$value)), unname(coef(expected$value)))
    expect_equal(unname(vcov(fit$value)), unname(vcov(expected$value)))
    expect_equal(as.numeric(predict(fit$value, type = "lp")), as.numeric(predict(expected$value, type = "lp")))
  }
})


test_that("AFT na.pass refuses unknown statuses instead of treating them as censored", {
  d <- data.frame(time = 1:8, status = c(1, NA, 0, 1, 0, 1, 0, 1), x = rep(0:1, 4))
  expect_error(survreg(Surv(time, status) ~ x, d, na.action = na.pass), "missing")
  expect_error(survival::survreg(survival::Surv(time, status) ~ x, d, na.action = na.pass), "Invalid survival times")
})
