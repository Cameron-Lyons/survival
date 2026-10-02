test_that("numeric bridge columns keep their length, type and row metadata", {
  probe <- reticulate::py_run_string(paste(
    "def inspect_numeric(frame):",
    "    return dict(rows=frame.nrow, labels=frame.row_names, columns={",
    "        name: dict(shape=list(value.shape), dtype=str(value.dtype), values=value.tolist())",
    "        for name, value in frame.items()})",
    sep = "\n"
  ), local = TRUE)
  for (n in c(0L, 1L, 3L)) {
    data <- data.frame(real = seq_len(n) / 4, integer = seq_len(n))
    if (n == 3L) data$real <- c(-Inf, .25, Inf)
    row.names(data) <- if (n) paste0("patient / ", seq_len(n)) else character()
    result <- probe$inspect_numeric(.as_python_data(data))
    expect_identical(result$rows, n)
    expect_identical(unlist(result$labels, use.names = FALSE),
                     if (n) row.names(data) else NULL)
    for (name in names(data)) {
      column <- result$columns[[name]]
      expect_identical(unlist(column$shape), n)
      expect_identical(column$dtype, if (name == "real") "float64" else "int32")
      expect_equal(unlist(column$values), if (n) data[[name]] else NULL)
    }
  }
  result <- probe$inspect_numeric(.as_python_data(data.frame(x = 1:3)))
  expect_null(result$labels)
  expect_identical(result$rows, 3L)
})

test_that("bulk conversion retains missing, categorical and classed column semantics", {
  probe <- reticulate::py_run_string(paste(
    "def check_columns(frame):",
    "    import numpy as np",
    "    assert isinstance(frame['complete'], np.ndarray)",
    "    assert frame['real_missing'] == [1.5, None, None]",
    "    assert frame['integer_missing'] == [1, None, 3]",
    "    assert frame['logical'] == [True, False, None]",
    "    assert frame['characters'] == ['a', None, 'c']",
    "    assert list(frame['factor']) == ['b', None, 'a']",
    "    assert frame['factor'].categories == ('c', 'b', 'a')",
    "    assert list(frame['ordered']) == ['b', 'a', 'b']",
    "    assert frame['ordered'].categories == ('c', 'b', 'a')",
    "    assert isinstance(frame['date'], list) and frame['date'] == [0., 1., 2.]",
    "    assert isinstance(frame['classed'], list) and frame['classed'] == [1., 2., 3.]",
    "    assert isinstance(frame['matrix'], list) and frame['matrix'] == [1, 2, 3, 4, 5, 6]",
    "    return True",
    sep = "\n"
  ), local = TRUE)
  data <- data.frame(
    complete = c(1.5, 2.5, 3.5), real_missing = c(1.5, NA_real_, NaN),
    integer_missing = c(1L, NA_integer_, 3L), logical = c(TRUE, FALSE, NA),
    characters = c("a", NA, "c"),
    factor = factor(c("b", NA, "a"), levels = c("c", "b", "a")),
    ordered = ordered(c("b", "a", "b"), levels = c("c", "b", "a")),
    date = as.Date(0:2, origin = "1970-01-01"), classed = I(c(1, 2, 3))
  )
  data$matrix <- matrix(1:6, nrow = 3L)
  expect_true(probe$check_columns(.as_python_data(data)))
})

test_that("numeric views survive input mutation and R and Python garbage collection", {
  probe <- reticulate::py_run_string(paste(
    "def check_owned_columns(frame):",
    "    import gc",
    "    import numpy as np",
    "    gc.collect()",
    "    assert np.array_equal(frame['integer'], np.arange(1, 100001, dtype=np.int32))",
    "    assert np.array_equal(frame['real'], np.arange(1, 100001, dtype=float) / 4)",
    "    assert not frame['integer'].flags.writeable",
    "    assert not frame['real'].flags.writeable",
    "    return True",
    sep = "\n"
  ), local = TRUE)
  data <- data.frame(integer = seq_len(100000L), real = seq_len(100000L) / 4)
  converted <- .as_python_data(data)
  data$integer[] <- 0L
  data$real[] <- 0
  rm(data)
  for (i in seq_len(3L)) {
    invisible(gc())
    overwrite <- rep(i, 200000L)
    expect_true(probe$check_owned_columns(converted))
    rm(overwrite)
  }
})

test_that("mixed numeric predictors retain stock values, errors and prediction row labels", {
  d <- survival::ovarian
  d$integer <- as.integer(d$rx)
  d$logical <- d$rx == 2
  d$group <- factor(rep(c("b", "a"), length.out = nrow(d)), levels = c("unused", "b", "a"))
  row.names(d) <- paste0("patient / ", seq_len(nrow(d)))
  selected <- c(12L, 1L, 12L, setdiff(seq_len(nrow(d)), c(12L, 1L)))
  for (family in c("coxph", "survreg")) {
    own <- get(family, asNamespace("survivalr"))
    stock <- get(family, asNamespace("survival"))
    for (term in c("age + integer", "age * group + integer", "log(age) + integer + strata(group)")) {
      formula <- as.formula(paste("Surv(futime, fustat) ~", term))
      fit <- own(formula, d, subset = selected, x = TRUE)
      reference <- stock(formula, d, subset = selected, x = TRUE)
      expect_equal(coef(fit), coef(reference), tolerance = 3e-7)
      expect_equal(vcov(fit), vcov(reference), tolerance = 3e-7)
      matrix <- model.matrix(fit)
      expected_matrix <- model.matrix(reference)
      expect_identical(dim(matrix), dim(expected_matrix))
      expect_identical(colnames(matrix), colnames(expected_matrix))
      expect_equal(as.numeric(matrix), as.numeric(expected_matrix), tolerance = 3e-7)
      for (rows in list(3L, c(3L, 4L, 5L, 6L))) for (missing in c(FALSE, TRUE)) {
        nd <- d[rows, , drop = FALSE]
        if (missing) nd$integer[1L] <- NA_integer_
        for (action in c("na.pass", "na.omit", "na.exclude")) {
          if (missing && nrow(nd) == 1L && action != "na.pass") next
          for (type in if (family == "coxph") c("lp", "terms") else c("link", "quantile", "terms")) {
            args <- list(newdata = nd, type = type, se.fit = TRUE, na.action = action)
            actual <- do.call(predict, c(list(fit), args))
            expected <- do.call(predict, c(list(reference), args))
            info <- paste(family, term, length(rows), missing, action, type)
            expect_named(actual, names(expected), info = info)
            for (field in names(expected)) {
              expect_identical(dim(actual[[field]]), dim(expected[[field]]), info = info)
              expect_identical(names(actual[[field]]), names(expected[[field]]), info = info)
              expect_identical(dimnames(actual[[field]]), dimnames(expected[[field]]), info = info)
              expect_equal(as.numeric(actual[[field]]), as.numeric(expected[[field]]),
                           tolerance = 3e-7, info = info)
            }
          }
        }
      }
    }
  }
})
