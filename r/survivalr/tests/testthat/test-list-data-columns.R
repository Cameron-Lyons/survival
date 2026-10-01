test_that("list columns preserve missing kinds, factor levels and vector shapes", {
  probe <- reticulate::py_run_string(paste(
    "def check_list(data):",
    "    import numpy as np",
    "    assert isinstance(data, dict)",
    "    for name, size in [('empty', 0), ('single', 1), ('real', 3)]:",
    "        assert data[name].shape == (size,)",
    "        assert data[name].dtype == np.dtype('float64')",
    "    assert data['integer'].dtype == np.dtype('int32')",
    "    assert data['integer_missing'] == [1, None, 3]",
    "    assert data['logical'] == [True, None, False]",
    "    assert data['characters'] == ['a', None, 'c']",
    "    assert list(data['factor']) == ['b', None, 'a']",
    "    assert data['factor'].categories == ('c', 'b', 'a')",
    "    assert list(data['ordered']) == ['b', 'a', 'b']",
    "    assert data['ordered'].categories == ('c', 'b', 'a')",
    "    assert data['date'] == [0., None, 2.]",
    "    assert data['classed'] == [1., None, 3.]",
    "    assert data['null'] is None",
    "    assert data['nested'] == {'a': [1, 2]}",
    "    assert data['matrix'].shape == (3, 2)",
    "    np.testing.assert_array_equal(data['matrix'], [[1, 4], [2, 5], [3, 6]])",
    "    np.testing.assert_array_equal(data['real'], [1.5, np.nan, np.nan])",
    "    assert int(data['real'].view(np.uint64)[1]) & 0xffffffff == 1954",
    "    assert int(data['real'].view(np.uint64)[2]) & 0xffffffff != 1954",
    "    return True",
    sep = "\n"
  ), local = TRUE)
  data <- list(empty = numeric(), single = 2.5, real = c(1.5, NA_real_, NaN),
    integer = 1:3, integer_missing = c(1L, NA_integer_, 3L),
    logical = c(TRUE, NA, FALSE), characters = c("a", NA, "c"),
    factor = factor(c("b", NA, "a"), levels = c("c", "b", "a")),
    ordered = ordered(c("b", "a", "b"), levels = c("c", "b", "a")),
    date = as.Date(c(0, NA, 2), origin = "1970-01-01"), classed = I(c(1, NA, 3)),
    null = NULL, nested = list(a = 1:2), matrix = matrix(1:6, 3))
  expect_true(probe$check_list(.as_python_data(data)))
})

test_that("list numeric views survive mutation and collection", {
  probe <- reticulate::py_run_string(paste(
    "def check_list_views(data):",
    "    import gc",
    "    import numpy as np",
    "    gc.collect()",
    "    np.testing.assert_array_equal(data['integer'], np.arange(1, 100001, dtype=np.int32))",
    "    np.testing.assert_array_equal(data['real'], np.arange(1, 100001, dtype=float) / 4)",
    "    assert not data['integer'].flags.writeable and not data['real'].flags.writeable",
    "    return True",
    sep = "\n"
  ), local = TRUE)
  data <- list(integer = seq_len(100000L), real = seq_len(100000L) / 4)
  converted <- .as_python_data(data)
  data$integer[] <- 0L; data$real[] <- 0
  rm(data)
  for (i in 1:3) {
    invisible(gc())
    overwrite <- rep(i, 200000L)
    expect_true(probe$check_list_views(converted))
    rm(overwrite)
  }
})

test_that("list model matrices match stock under every missing-data option", {
  original <- options(na.action = "na.omit")
  on.exit(options(original), add = TRUE)
  setup <- .matrix_na_setup()
  for (kind in c("coxph", "survreg")) for (rhs in setup$formulas) {
    d <- setup$data
    fit <- function(namespace) get(kind, asNamespace(namespace))(
      as.formula(paste("Surv(futime,fustat)~", rhs)), as.list(d), x = TRUE, model = TRUE)
    actual <- .logical_matrix_capture(function() fit("survivalr"))
    expected <- .logical_matrix_capture(function() fit("survival"))
    expect_identical(gsub("[[:space:]]+", "", actual$warnings),
                     gsub("[[:space:]]+", "", expected$warnings))
    for (action in c("na.omit", "na.exclude", "na.pass", "na.fail")) {
      options(na.action = action)
      for (input in setup$inputs) {
        nd <- .matrix_na_newdata(d, input)
        if (!is.null(nd)) nd <- as.list(nd)
        call <- function(fit) if (is.null(nd)) model.matrix(fit) else model.matrix(fit, nd)
        stock <- .logical_matrix_capture(function() call(expected$value))
        result <- .logical_matrix_capture(function() call(actual$value))
        info <- paste(kind, rhs, action, input)
        if (is.list(stock$value) && !is.null(stock$value$error)) {
          expect_match(result$value$error, "missing values", info = info)
        } else {
          .logical_matrix_compare(result$value, stock$value, info)
          expect_identical(is.na(result$value), is.na(stock$value), info = info)
          expect_identical(is.nan(result$value), is.nan(stock$value), info = info)
          expect_identical(sub(" in (log|sqrt)\\(.*\\)$", "", result$warnings), stock$warnings, info = info)
        }
      }
    }
    options(na.action = "na.omit")
  }
})

test_that("list model fits and predictions retain logical and response omissions", {
  d <- .matrix_na_setup()$data
  for (kind in c("coxph", "survreg")) for (rhs in c("age + flag", "age * flag", "age + strata(g) + flag")) {
    for (variant in c("complete", "logical", "numeric", "status", "strata")) for (action in c("na.omit", "na.exclude")) {
      train <- d
      if (variant == "logical") train$flag[c(2, 7)] <- NA
      if (variant == "numeric") {train$age[2] <- NA_real_; train$age[7] <- NaN}
      if (variant == "status") train$fustat[c(2, 7)] <- NA_integer_
      if (variant == "strata") train$g[c(2, 7)] <- NA
      fit <- function(namespace) get(kind, asNamespace(namespace))(
        as.formula(paste("Surv(futime,fustat)~", rhs)), as.list(train),
        x = TRUE, model = TRUE, na.action = action)
      actual <- .logical_matrix_capture(function() fit("survivalr"))
      expected <- .logical_matrix_capture(function() fit("survival"))
      info <- paste(kind, rhs, variant, action)
      expect_equal(coef(actual$value), coef(expected$value), tolerance = 3e-7, info = info)
      expect_equal(vcov(actual$value), vcov(expected$value), tolerance = 3e-7, info = info)
      .logical_matrix_compare(model.matrix(actual$value), model.matrix(expected$value), info)
      for (missing in c(FALSE, TRUE)) {
        nd <- as.list(d[c(7, 3, 7, 1), ])
        if (missing) nd$flag[2] <- NA
        for (new_action in c("na.omit", "na.exclude", "na.pass")) {
          for (type in if (kind == "coxph") c("lp", "terms") else c("link", "quantile", "terms")) {
            args <- list(newdata = nd, type = type, se.fit = TRUE, na.action = new_action)
            result <- do.call(predict, c(list(actual$value), args))
            stock <- do.call(predict, c(list(expected$value), args))
            expect_named(result, names(stock), info = info)
            for (field in names(stock)) {
              expect_identical(dim(result[[field]]), dim(stock[[field]]), info = info)
              expect_identical(names(result[[field]]), names(stock[[field]]), info = info)
              if (kind == "survreg" && rhs == "age + strata(g) + flag" && type == "terms") {
                # Stock labels the flag contribution strata(g) here. Keep the
                # actual ordinary term name, and compare every value and row.
                expect_identical(colnames(result[[field]]), c("age", "flag"), info = info)
                expect_identical(colnames(stock[[field]]), c("age", "strata(g)"), info = info)
                expect_identical(rownames(result[[field]]), rownames(stock[[field]]), info = info)
              } else {
                expect_identical(dimnames(result[[field]]), dimnames(stock[[field]]), info = info)
              }
              expect_equal(as.numeric(result[[field]]), as.numeric(stock[[field]]), tolerance = 3e-7, info = info)
            }
          }
        }
      }
    }
  }
})

test_that("list inputs preserve missing logical groups across formula APIs", {
  d <- .matrix_na_setup()$data
  d$flag[c(2, 7)] <- NA
  data <- as.list(d)
  form <- survival::Surv(futime, fustat) ~ flag
  curve <- survfit(form, data)
  stock_curve <- get("survfit.formula", asNamespace("survival"))(form, data)
  frame <- as.data.frame(curve)
  expect_equal(frame$time, stock_curve$time)
  expect_equal(frame$n.event, stock_curve$n.event)
  expect_equal(frame$n.risk, stock_curve$n.risk)
  expect_equal(frame$surv, stock_curve$surv)
  test <- survdiff(form, data)
  stock_test <- survival::survdiff(form, data)
  expect_equal(as.numeric(test$chisq), stock_test$chisq)
  concord <- concordance(form, data)
  stock_concord <- get("concordance.formula", asNamespace("survival"))(form, data)
  expect_equal(as.numeric(concord$concordance), stock_concord$concordance)
  expect_equal(as.numeric(concord$var), stock_concord$var)
  expect_equal(as.numeric(concord$count), as.numeric(stock_concord$count))
  expect_error(survfit(form, data, na.action = na.fail), "missing")
  expect_error(survdiff(form, data, na.action = na.fail), "missing")
  expect_error(concordance(form, data, na.action = na.fail), "missing")
})

test_that("variable-free list frames have zero rows while data frames retain their rows", {
  d <- .matrix_na_setup()$data
  for (kind in c("coxph", "survreg")) for (rhs in c("1", "sqrt(z) - sqrt(z)")) {
    form <- as.formula(paste("Surv(futime,fustat)~", rhs))
    actual <- get(kind, asNamespace("survivalr"))(form, as.list(d), x = TRUE)
    stock <- get(kind, asNamespace("survival"))(form, as.list(d), x = TRUE)
    for (nd in list(list(), as.list(d[1:4, ]), d[1:4, ], d[FALSE, ])) {
      # A removed transform is still a model-frame variable and is required.
      if (rhs != "1" && !length(nd)) next
      .logical_matrix_compare(model.matrix(actual, nd), model.matrix(stock, nd), paste(kind, rhs))
      types <- if (kind == "coxph") c("lp", "risk") else c("link", "response")
      for (type in types) expect_equal(predict(actual, newdata = nd, type = type),
        predict(stock, newdata = nd, type = type), tolerance = 3e-7)
    }
    if (kind == "coxph") for (nd in list(as.list(d[1:4, ]), d[1:4, ])) {
      # Expected counts evaluate the response even with no predictor variables.
      # Stock predict tries multiplying by NULL coefficients and fails; its
      # baseline curve independently determines these null-model predictions.
      baseline <- survival::basehaz(stock, centered = FALSE)
      expected <- c(0, baseline$hazard)[findInterval(nd$futime, baseline$time) + 1L]
      for (type in c("expected", "survival")) {
        expect_error(predict(stock, newdata = nd, type = type), "requires numeric/complex")
        expect_equal(as.numeric(predict(actual, newdata = nd, type = type)),
          if (type == "expected") expected else exp(-expected), tolerance = 3e-7)
      }
    }
  }
})
