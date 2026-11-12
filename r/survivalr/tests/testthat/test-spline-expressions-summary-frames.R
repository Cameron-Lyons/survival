.composed_basis_data <- function() {
  data <- survival::lung[seq_len(80L), c("time", "status", "age", "sex")]
  data$x <- data$age
  data$z <- data$sex
  data
}

test_that("composed spline expressions retain fitted bases across the R bridge", {
  ns <- splines::ns
  bs <- splines::bs
  nsk <- survival::nsk
  data <- .composed_basis_data()
  newdata <- data.frame(x = c(42, 55, 70, 78), z = c(1, 2, 1, 2))
  terms <- c("ns(log(x), df = 3)", "bs(x / 10, df = 3)",
    "poly(log(x), degree = 2)", "nsk(x + z, df = 3)",
    "ns(log(x), df = 3) * factor(z)")
  for (kind in c("coxph", "survreg")) for (term in terms) {
    formula <- as.formula(paste("Surv(time, status) ~", term))
    actual <- get(kind, asNamespace("survivalr"))(formula, data, x = TRUE, model = TRUE)
    expected <- get(kind, asNamespace("survival"))(formula, data, x = TRUE, model = TRUE)
    info <- paste(kind, term)
    expect_equal(coef(actual), coef(expected), tolerance = 2e-7, info = info)
    expect_equal(vcov(actual), vcov(expected), tolerance = 2e-7, info = info)
    expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-11, info = info)
    for (fit in list(actual, unserialize(serialize(actual, NULL)))) {
      expect_equal(model.matrix(fit, newdata), model.matrix(expected, newdata),
        tolerance = 2e-11, info = info)
      expect_equal(predict(fit, newdata, type = "lp", se.fit = TRUE),
        predict(expected, newdata, type = "lp", se.fit = TRUE), tolerance = 2e-7, info = info)
    }
  }
})

test_that("spline expressions are evaluated before subsets and omission", {
  ns <- splines::ns
  data <- .composed_basis_data()
  data$x[c(6L, 31L)] <- NA_real_
  rows <- c(80:41, 2L, 7L, 2L, 20:30)
  newdata <- data.frame(x = c(42, 55, NA, 70), z = c(1, 2, 1, 2))
  for (kind in c("coxph", "survreg")) {
    formula <- Surv(time, status) ~ ns(log(x), df = 3) + z
    actual <- get(kind, asNamespace("survivalr"))(formula, data, subset = rows,
      na.action = na.exclude, x = TRUE, model = TRUE)
    expected <- get(kind, asNamespace("survival"))(formula, data, subset = rows,
      na.action = na.exclude, x = TRUE, model = TRUE)
    expect_equal(coef(actual), coef(expected), tolerance = 2e-7, info = kind)
    expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-11, info = kind)
    expect_equal(predict(actual, newdata, type = "lp", na.action = na.exclude),
      predict(expected, newdata, type = "lp", na.action = na.exclude),
      tolerance = 2e-7, info = kind)
  }
})

.summary_frame_data <- function() {
  data.frame(time = c(1, 1, 2, 3, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8, 9),
    status = c(1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1),
    event = factor(c("a", "censor", "b", "a", "censor", "b", "censor", "censor",
      "b", "censor", "a", "censor", "b", "a", "censor", "b"),
      levels = c("censor", "a", "b")),
    group = rep(c("a", "b"), each = 8),
    x = c(2, 0, 1, 3, 1, 0, 2, 3, 0, 2, 1, 0, 3, 1, 2, 3), id = seq_len(16L))
}

test_that("Cox data-frame summaries match stock prediction and state margins", {
  data <- .summary_frame_data()
  newdata <- data.frame(x = c(0, 2), tag = factor(c("b", "a"), levels = c("a", "unused", "b")))
  for (kind in c("ordinary", "stratified", "multistate")) {
    formula <- switch(kind,
      ordinary = Surv(time, status) ~ x,
      stratified = Surv(time, status) ~ x + strata(group),
      multistate = Surv(time, event) ~ x)
    options <- if (kind == "multistate") list(id = data$id) else list()
    actual_model <- do.call(survivalr::coxph, c(list(formula, data), options))
    expected_model <- do.call(survival::coxph, c(list(formula, data), options))
    curve_options <- if (kind == "multistate") list(se.fit = FALSE) else list()
    actual <- do.call(survivalr::survfit, c(list(actual_model, newdata = newdata), curve_options))
    expected <- do.call(survival::survfit, c(list(expected_model, newdata = newdata), curve_options))
    for (arguments in list(list(), list(censored = TRUE), list(times = c(2, 4, 7), scale = 2),
      list(times = c(7, 2, 2, 0, 10), extend = TRUE, dosum = FALSE))) {
      a <- do.call(summary, c(list(actual, data.frame = TRUE), arguments))
      e <- do.call(summary, c(list(expected, data.frame = TRUE), arguments))
      expect_s3_class(a, "data.frame")
      expect_identical(names(a), names(e), info = kind)
      for (name in names(e)) expect_equal(a[[name]], e[[name]], tolerance = 1e-9,
        info = paste(kind, name))
    }
    expect_identical(summary(actual, data.frame = TRUE),
      do.call(summary, list(actual, data.frame = TRUE)))
    if (kind == "multistate") expect_s3_class(summary(actual), "summary.survfitms")
  }
})

test_that("Cox frame metadata and validation survive the R boundary", {
  data <- .summary_frame_data()
  actual_model <- survivalr::coxph(Surv(time, status) ~ x, data)
  expected_model <- survival::coxph(Surv(time, status) ~ x, data)
  newdata <- data.frame(x = c(0, 2), time = c(99, 88), tag = factor(c("b", "a")))
  actual <- survivalr::survfit(actual_model, newdata = newdata)
  expected <- survival::survfit(expected_model, newdata = newdata)
  a <- summary(actual, times = c(2, 4), data.frame = TRUE)
  e <- summary(expected, times = c(2, 4), data.frame = TRUE)
  expect_identical(names(a), names(e))
  for (name in names(e)) expect_equal(a[[name]], e[[name]], tolerance = 1e-9, info = name)
  empty <- summary(actual, times = 20, data.frame = TRUE)
  expect_identical(names(empty), names(a))
  expect_equal(nrow(empty), 0L)
  expect_identical(levels(empty$tag_), levels(newdata$tag))
  expect_error(summary(actual, data.frame = "yes"), "data.frame must be TRUE/FALSE", fixed = TRUE)
  expect_error(summary(actual, times = "2", data.frame = TRUE),
    "times must be a numeric vector", fixed = TRUE)
  expect_error(summary(actual, times = c(4, 2), data.frame = TRUE, dosum = TRUE),
    "dosum=TRUE requires the times to be increasing", fixed = TRUE)
})

test_that("prediction names only collide with components present on the curve", {
  cases <- list(
    list(name = "strata", factor = TRUE),
    list(name = "state", factor = FALSE),
    list(name = "std.err", factor = TRUE, options = list(se.fit = FALSE)),
    list(name = "lower", factor = TRUE, options = list(conf.type = "none")),
    list(name = "upper", factor = TRUE, options = list(conf.type = "none")))
  for (case in cases) {
    data <- .summary_frame_data()
    data[[case$name]] <- if (case$factor) factor(ifelse(data$x %% 2, "b", "a")) else data$x
    newdata <- data.frame(value = if (case$factor) factor(c("b", "a"),
      levels = c("a", "unused", "b")) else c(0, 2), tag = c("one", "two"))
    names(newdata)[1L] <- case$name
    formula <- as.formula(paste0("Surv(time, status) ~ `", case$name, "`"))
    actual_model <- survivalr::coxph(formula, data)
    expected_model <- survival::coxph(formula, data)
    actual <- do.call(survivalr::survfit, c(list(actual_model, newdata = newdata), case$options))
    expected <- do.call(survival::survfit, c(list(expected_model, newdata = newdata), case$options))
    a <- summary(actual, times = c(2, 4), data.frame = TRUE)
    e <- summary(expected, times = c(2, 4), data.frame = TRUE)
    expect_identical(names(a), names(e), info = case$name)
    for (name in names(e)) expect_equal(a[[name]], e[[name]], tolerance = 1e-9,
      info = paste(case$name, name))
    expect_identical(a[[case$name]], e[[case$name]], info = case$name)
    empty <- summary(actual, times = 20, data.frame = TRUE)
    expect_identical(empty[[case$name]], newdata[[case$name]][FALSE], info = case$name)
  }
})

test_that("multistate frames preserve state labels beside colliding prediction names", {
  data <- .summary_frame_data()
  data$state <- data$x
  newdata <- data.frame(state = c(0, 2), time = c(101, 202), state_ = c(31, 43),
    tag = factor(c("b", "a"), levels = c("a", "unused", "b")))
  actual <- survivalr::survfit(
    survivalr::coxph(Surv(time, event) ~ state, data, id = id),
    newdata = newdata, se.fit = FALSE)
  expected <- survival::survfit(
    survival::coxph(Surv(time, event) ~ state, data, id = id),
    newdata = newdata, se.fit = FALSE)
  for (fit in list(actual, unserialize(serialize(actual, NULL)))) {
    a <- summary(fit, times = c(2, 4), data.frame = TRUE)
    e <- summary(expected, times = c(2, 4), data.frame = TRUE)
    expect_identical(names(a), names(e))
    for (column in seq_along(e)) expect_equal(a[[column]], e[[column]], tolerance = 1e-9,
      info = paste(column, names(e)[column]))
    expect_equal(sum(names(a) == "state"), 2L)
    expect_equal(sum(names(a) == "time"), 2L)
  }
})
