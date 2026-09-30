.omitted_prediction_compare <- function(actual, expected, info) {
  if (is.list(expected)) {
    expect_named(actual, names(expected), info = info)
    for (name in names(expected)) .omitted_prediction_compare(actual[[name]], expected[[name]], paste(info, name))
    return(invisible(NULL))
  }
  expect_identical(dim(actual), dim(expected), info = info)
  expect_identical(names(actual), names(expected), info = info)
  expect_identical(dimnames(actual), dimnames(expected), info = info)
  expect_identical(is.na(actual), is.na(expected), info = info)
  expect_identical(is.nan(actual), is.nan(expected), info = info)
  expect_equal(as.numeric(actual), as.numeric(expected), tolerance = 3e-7, info = info)
}

test_that("missing kinds and omitted prediction shapes match independent stock calculations", {
  setup <- .omitted_prediction_setup()
  for (name in names(setup$specs)) {
    actual <- .omitted_prediction_fit(name, setup, "survivalr")
    stock <- .omitted_prediction_fit(name, setup, "survival")
    for (case in .omitted_prediction_cases(name)) {
      expected <- .omitted_prediction_expected(stock, setup, case)
      result <- .omitted_prediction_capture(function() .omitted_prediction_call(actual, setup, case))
      expect_identical(result$warnings, expected$warnings, info = case$name)
      .omitted_prediction_compare(result$value, expected$value, case$name)
    }
  }
})

test_that("missing numeric column buffers preserve NA and NaN through mutation and collection", {
  probe <- reticulate::py_run_string(paste(
    "def check_missing_buffers(frame):",
    "    import gc, numpy as np",
    "    gc.collect()",
    "    values = frame['real']",
    "    assert values.dtype == np.dtype('float64')",
    "    assert not values.flags.writeable",
    "    np.testing.assert_array_equal(values[:3], [1., np.nan, 3.])",
    "    assert int(values.view(np.uint64)[1]) & 0xffffffff == 1954",
    "    assert frame['integer'] == [1, None, 3, 4]",
    "    assert np.isnan(frame['real'][3])",
    "    assert int(frame['real'].view(np.uint64)[3]) & 0xffffffff != 1954",
    "    return True", sep = "\n"
  ), local = TRUE)
  data <- data.frame(real = c(1, NA_real_, 3, NaN), integer = c(1L, NA_integer_, 3L, 4L))
  converted <- .as_python_data(data)
  data$real[] <- 0; data$integer[] <- 0L
  rm(data)
  for (i in seq_len(3L)) {
    invisible(gc())
    overwrite <- rep(i, 100000L)
    expect_true(probe$check_missing_buffers(converted))
    rm(overwrite)
  }
})

test_that("missing integer input retains its model-frame source type", {
  d <- survival::ovarian; d$rx <- as.integer(d$rx); d$rx[1L] <- NA_integer_
  for (kind in c("coxph", "survreg")) {
    args <- list(Surv(futime,fustat) ~ age + rx, d, model = TRUE)
    fit <- do.call(get(kind, asNamespace("survivalr")), args)
    stock <- do.call(get(kind, asNamespace("survival")), args)
    expect_identical(model.frame(fit)$rx, model.frame(stock)$rx)
  }
})

test_that("formula-generated NaNs remain distinct from source and omitted NA values", {
  d <- survival::ovarian
  for (kind in c("coxph", "survreg")) for (term in c("log(age)", "sqrt(age)", "I(age / rx)")) {
    args <- list(as.formula(paste("Surv(futime,fustat) ~", term, "+ rx")), d)
    fit <- do.call(get(kind, asNamespace("survivalr")), args)
    stock <- do.call(get(kind, asNamespace("survival")), args)
    nd <- d[3:6, ]; nd$age <- c(NA_real_, NaN, -1, 2)
    if (startsWith(term, "I(")) {nd$age[3L] <- 0; nd$rx <- c(0, 0, 0, 1)}
    for (action in c("na.pass", "na.omit", "na.exclude")) for (type in c("lp", "terms")) {
      options <- list(newdata = nd, type = type, se.fit = TRUE, na.action = action)
      actual <- suppressWarnings(do.call(predict, c(list(fit), options)))
      expected <- suppressWarnings(do.call(predict, c(list(stock), options)))
      .omitted_prediction_compare(actual, expected, paste(kind, term, action, type))
    }
  }
})
