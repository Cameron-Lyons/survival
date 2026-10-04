.typed_formula_data <- function() {
  d <- survival::ovarian
  d$x <- (d$age - 60) / 10
  d$z <- rep(c(-2, .5, 1, 2, -1), length.out = nrow(d))
  d$a <- rep(c(TRUE, FALSE, NA), length.out = nrow(d))
  d$b <- rep(rep(c(TRUE, FALSE, NA), each = 3), length.out = nrow(d))
  d$g <- factor(rep(c("b", "a", "c", "a", "b"), length.out = nrow(d)),
                levels = c("c", "b", "a"))
  row.names(d) <- paste0("patient / ", seq_len(nrow(d)))
  d
}

test_that("logical expressions retain R matrix types, contrasts and omitted rows", {
  d <- .typed_formula_data()
  terms <- c("x + I(a & b)", "x + I(a | b)", "x + I(!a)",
             "x + I(!a | b & a)", "x + I(!x^2 > 1)",
             "x + identity(I(a & b))", "x + I(identity(I(g)))",
             "x + as.numeric(a & b)", "x + offset(a | b)",
             "x + offset(as.numeric(a | b))")
  for (kind in c("coxph", "survreg")) for (term in terms) {
    for (action in list(na.omit, na.exclude)) {
      info <- paste(kind, term, deparse(substitute(action)))
      actual <- suppressWarnings(do.call(get(kind, asNamespace("survivalr")),
        list(formula = as.formula(paste("Surv(futime, fustat) ~", term)),
             data = d, na.action = action, model = TRUE, x = TRUE)))
      expected <- suppressWarnings(do.call(get(kind, asNamespace("survival")),
        list(formula = as.formula(paste("survival::Surv(futime, fustat) ~", term)),
             data = d, na.action = action, model = TRUE, x = TRUE)))
      expect_equal(coef(actual), coef(expected), tolerance = 2e-8, info = info)
      expect_equal(vcov(actual), vcov(expected), tolerance = 2e-8, info = info)
      expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-12, info = info)
      actual_frame <- model.frame(actual)
      expected_frame <- model.frame(expected)
      expect_identical(row.names(actual_frame), row.names(expected_frame), info = info)
      # Public bridge frames expose flattened response and raw source columns.
      # Recovered logical rows may still have missing values in these sources.
      source_rows <- match(row.names(expected_frame), row.names(d))
      for (name in intersect(c("x", "a", "b", "g"), names(actual_frame))) {
        expect_identical(typeof(actual_frame[[name]]), typeof(d[[name]]), info = info)
        expect_equal(as.vector(actual_frame[[name]]), as.vector(d[[name]][source_rows]), info = info)
        expect_identical(levels(actual_frame[[name]]), levels(d[[name]]), info = info)
      }
      expect_equal(predict(actual, type = "lp"), predict(expected, type = "lp"),
                   tolerance = 2e-8, info = info)
    }
  }
})

test_that("typed logical offsets survive subset, prediction and serialization", {
  d <- .typed_formula_data()
  rows <- c(7L, 3L, 7L, 1L, 5L, 2L, 10L, 8L, 4L, 11L, 6L, 9L, 13L, 16L)
  newdata <- d[c(7L, 3L, 7L, 1L, 5L, 2L), ]
  newdata$a <- c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE)
  newdata$b <- c(FALSE, TRUE, TRUE, FALSE, TRUE, TRUE)
  for (kind in c("coxph", "survreg")) {
    term <- "x + I(a | b) + offset(x > 0)"
    actual <- suppressWarnings(do.call(get(kind, asNamespace("survivalr")),
      list(formula = as.formula(paste("Surv(futime, fustat) ~", term)),
           data = d, subset = rows, na.action = na.exclude, model = TRUE, x = TRUE)))
    expected <- suppressWarnings(do.call(get(kind, asNamespace("survival")),
      list(formula = as.formula(paste("survival::Surv(futime, fustat) ~", term)),
           data = d, subset = rows, na.action = na.exclude, model = TRUE, x = TRUE)))
    expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-12)
    expected_lp <- predict(expected, newdata = newdata, type = "lp")
    # Stock predict.survreg drops the new-data offset; derive it from its own frame.
    if (kind == "survreg") {
      expected_frame <- model.frame(delete.response(terms(expected)), newdata,
                                    na.action = na.pass, xlev = expected$xlevels)
      expected_lp <- expected_lp + model.offset(expected_frame)
    }
    expect_equal(predict(actual, newdata = newdata, type = "lp"), expected_lp,
                 tolerance = 2e-8)
    restored <- unserialize(serialize(actual, NULL))
    expect_equal(model.matrix(restored, newdata), model.matrix(actual, newdata))
    expect_equal(predict(restored, newdata = newdata, type = "lp"), expected_lp,
                 tolerance = 2e-8)
  }
})

test_that("all-missing logical sources retain their declared type across the bridge", {
  d <- .typed_formula_data()
  d$a[] <- NA
  for (kind in c("coxph", "survreg")) {
    actual <- do.call(get(kind, asNamespace("survivalr")),
      list(formula = Surv(futime, fustat) ~ x + offset(TRUE | a),
           data = d, model = TRUE))
    expected <- do.call(get(kind, asNamespace("survival")),
      list(formula = survival::Surv(futime, fustat) ~ x + offset(TRUE | a),
           data = d, model = TRUE))
    frame <- model.frame(actual)
    expect_type(frame$a, "logical")
    expect_identical(frame$a, rep(NA, nrow(d)))
    expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-12)
    expect_equal(predict(actual, type = "lp"), predict(expected, type = "lp"), tolerance = 2e-8)
    empty <- model.matrix(actual, d[FALSE, ])
    expect_identical(dim(empty), c(0L, ncol(model.matrix(expected))))
    expect_identical(attr(empty, "contrasts"), attr(model.matrix(expected), "contrasts"))
  }
})

test_that("recovered numeric sources preserve NaN, NA and integer storage", {
  d <- .typed_formula_data()
  d$z[1:2] <- c(NaN, NA_real_)
  d$i <- seq_len(nrow(d))
  d$i[1:2] <- NA_integer_
  for (kind in c("coxph", "survreg")) {
    actual <- do.call(get(kind, asNamespace("survivalr")),
      list(formula = Surv(futime, fustat) ~ x + offset(z^0) + offset(i^0),
           data = d, model = TRUE))
    frame <- model.frame(actual)
    expect_identical(frame$z, d$z)
    expect_identical(is.nan(frame$z), is.nan(d$z))
    expect_identical(frame$i, d$i)
    expect_identical(frame$fustat, d$fustat)
  }
})

test_that("integer expression overflow follows stock missing-row actions", {
  d <- .typed_formula_data()
  d$i <- rep(c(-2L, 0L, 1L, 2L, NA_integer_, 2147483647L), length.out = nrow(d))
  for (kind in c("coxph", "survreg")) for (action in list(na.omit, na.exclude)) {
    expect_warning(actual <- do.call(get(kind, asNamespace("survivalr")),
      list(formula = Surv(futime, fustat) ~ x + I(i + 1L),
           data = d, na.action = action, model = TRUE, x = TRUE)),
      "NAs produced by integer overflow")
    expect_warning(expected <- do.call(get(kind, asNamespace("survival")),
      list(formula = survival::Surv(futime, fustat) ~ x + I(i + 1L),
           data = d, na.action = action, model = TRUE, x = TRUE)),
      "NAs produced by integer overflow")
    expect_equal(coef(actual), coef(expected), tolerance = 2e-8)
    expect_equal(vcov(actual), vcov(expected), tolerance = 2e-8)
    expect_equal(model.matrix(actual), model.matrix(expected), tolerance = 2e-12)
    expect_equal(predict(actual, type = "lp"), predict(expected, type = "lp"),
                 tolerance = 2e-8)
    frame <- model.frame(actual)
    expected_rows <- row.names(model.frame(expected))
    expect_identical(row.names(frame), expected_rows)
    expect_identical(frame$i, d$i[match(expected_rows, row.names(d))])
  }
})
