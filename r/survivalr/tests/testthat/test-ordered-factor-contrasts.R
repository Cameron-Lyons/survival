.ordered_factor_data <- function(unused = FALSE) {
  i <- seq_len(80L)
  d <- data.frame(time = 10 + i * 37 %% 97 + i / 7,
    status = as.integer((i * 17 + 3) %% 11 > 2), x = sin(i * .63))
  d$g <- ordered(rep(c("z", "m", "a"), length.out = nrow(d)),
    levels = c("z", "m", "a", if (unused) "unused"))
  d
}

.ordered_factor_compare <- function(actual, expected, newdata) {
  expect_identical(names(coef(actual)), names(coef(expected)))
  expect_equal(coef(actual), coef(expected), tolerance = 3e-7)
  expect_equal(vcov(actual), vcov(expected), tolerance = 3e-7)
  expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 3e-7)
  expect_identical(attr(model.matrix(actual), "contrasts"),
    attr(model.matrix(expected), "contrasts"))
  for (type in c("lp", "terms")) {
    expect_equal(predict(actual, newdata, type = type, na.action = na.exclude),
      predict(expected, newdata, type = type, na.action = na.exclude), tolerance = 3e-7)
  }
}

test_that("raw factor conversion preserves ordering and explicit contrasts", {
  probe <- reticulate::py_run_string(paste(
    "def check_ordered_columns(data):",
    "    import numpy as np",
    "    from survival.r._formula import _expression_values",
    "    assert data['g'].ordered",
    "    assert not data['h'].ordered",
    "    assert data['g'].categories == ('z', 'm', 'a')",
    "    assert list(_expression_values(data, \"g > 'm'\", 4)) == [False, False, True, None]",
    "    assert list(_expression_values(data, \"factor(g) > 'm'\", 4)) == [False, False, True, None]",
    "    np.testing.assert_array_equal(data['custom'].contrast['data'], [[1, 0], [0, 1], [-1, -1]])",
    "    return True",
    sep = "\n"
  ), local = TRUE)
  g <- ordered(c("z", "m", "a", NA), levels = c("z", "m", "a"))
  custom <- g
  contrasts(custom) <- contr.sum(3)
  expect_true(probe$check_ordered_columns(.as_python_data(list(g = g,
    h = factor(as.character(g), levels = levels(g)), custom = custom))))
})

test_that("ordered default designs match stock fits, interactions and predictions", {
  for (kind in c("coxph", "survreg")) for (rhs in c("x + g", "x * g", "x:g", "g - 1",
    "x + I(g)", "x + identity(g)", "x + as.factor(g)", "x + factor(g)", "x + I(g >= 'm')")) {
    d <- .ordered_factor_data()
    form <- as.formula(paste("Surv(time, status) ~", rhs))
    fit <- function(namespace) get(kind, asNamespace(namespace))(form, d, x = TRUE, model = TRUE)
    actual <- fit("survivalr")
    expected <- fit("survival")
    nd <- d[c(8L, 3L, 8L, 1L, 5L), ]
    nd$g <- ordered(c("a", "z", "m", NA, "a"), levels = levels(d$g))
    .ordered_factor_compare(actual, expected, nd)
  }
})

test_that("factor constructors and unused levels are evaluated before row selection", {
  rows <- c(8L, 3L, 8L, 1L, 5L, 2L, 10L, 4L, 11L, 6L, 9L, 13L, 16L, 15L, 23L, 32L, 41L)
  for (kind in c("coxph", "survreg")) for (wrapper in c("g", "I(g)", "identity(g)",
    "as.factor(g)", "factor(g)")) {
    d <- .ordered_factor_data(unused = TRUE)
    d$x[[5L]] <- NA_real_
    form <- as.formula(paste("Surv(time, status) ~ x +", wrapper))
    fit <- function(namespace) get(kind, asNamespace(namespace))(form, d,
      subset = rows, na.action = na.exclude, x = TRUE, model = TRUE)
    actual <- fit("survivalr")
    expected <- fit("survival")
    nd <- d[c(3L, 1L, 2L), ]
    .ordered_factor_compare(actual, expected, nd)
  }
})

test_that("explicit factor contrasts and configured defaults override polynomial coding", {
  previous <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(previous), add = TRUE)
  for (kind in c("coxph", "survreg")) for (coding in c("matrix", "name", "configured", "treatment", "mixed"))
    for (wrapper in c("identity(g)", "factor(g)")) {
    d <- .ordered_factor_data()
    if (coding == "mixed") {
      options(contrasts = c("contr.treatment", "contr.sum"))
      contrasts(d$g) <- "contr.helmert"
    } else if (coding %in% c("configured", "treatment")) {
      options(contrasts = c("contr.treatment", if (coding == "configured") "contr.sum" else "contr.treatment"))
    } else {
      options(contrasts = c("contr.treatment", "contr.poly"))
      contrasts(d$g) <- if (coding == "matrix") contr.sum(3) else "contr.sum"
    }
    form <- as.formula(paste("Surv(time, status) ~ x +", wrapper))
    fit <- function(namespace) get(kind, asNamespace(namespace))(form, d, x = TRUE, model = TRUE)
    actual <- fit("survivalr")
    expected <- fit("survival")
    nd <- d[c(3L, 1L, 2L), ]
    .ordered_factor_compare(actual, expected, nd)
    # R serialization retains both fitted coding and source class metadata.
    restored <- unserialize(serialize(actual, NULL))
    .ordered_factor_compare(restored, expected, nd)
  }
})

test_that("restored factor columns retain explicit contrast attributes", {
  d <- .ordered_factor_data()
  contrasts(d$g) <- contr.sum(3)
  restored <- .restore_r_column_classes(data.frame(g = as.character(d$g)), d)
  expect_identical(restored$g, d$g)
  split <- survSplit(Surv(time, status) ~ ., data = d, cut = c(30, 60),
    episode = "episode", id = "split_id")
  stock <- survival::survSplit(Surv(time, status) ~ ., data = d, cut = c(30, 60),
    episode = "episode", id = "split_id")
  expect_identical(attr(split$g, "contrasts"), attr(stock$g, "contrasts"))
  expect_s3_class(split$g, "ordered")
})
