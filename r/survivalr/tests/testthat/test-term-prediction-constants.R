.constant_test_data <- function() {
  i <- seq_len(60L)
  data.frame(time = 10 + (i * 37) %% 97 + i / 7,
    status = as.integer((i * 17 + 3) %% 11 > 2),
    x = sin(i * .63), z = cos(i * .43), g = rep(c("a", "b"), 30L))
}

.term_constant <- function(value) {
  attr(if (is.list(value)) value$fit else value, "constant")
}

test_that("Cox fitted values ignore prediction types and keep no term constant", {
  d <- .constant_test_data()
  actual <- survivalr::coxph(Surv(time, status) ~ x, d)
  expected <- survival::coxph(Surv(time, status) ~ x, d)
  for (type in c("lp", "risk", "terms")) for (se in c(FALSE, TRUE)) {
    a <- fitted(actual, type = type, se.fit = se)
    b <- fitted(expected, type = type, se.fit = se)
    expect_equal(a, b, tolerance = 3e-7)
    expect_equal(a, predict(actual, type = "lp"), tolerance = 3e-7)
    expect_null(.term_constant(a))
  }
})

test_that("Cox terms preserve constants for every reference unless rows are restored", {
  d <- .constant_test_data()
  for (stratified in c(FALSE, TRUE)) {
    form <- if (stratified) Surv(time, status) ~ x + strata(g) else Surv(time, status) ~ x
    actual <- survivalr::coxph(form, d)
    expected <- survival::coxph(form, d)
    for (missing in c(FALSE, TRUE)) for (action in c("na.omit", "na.exclude", "na.pass"))
      for (reference in c("sample", "zero", "strata")) for (se in c(FALSE, TRUE)) {
      nd <- d[1:4, ]
      if (missing) nd$x[[2L]] <- NA_real_
      a <- predict(actual, nd, type = "terms", na.action = get(action),
        reference = reference, se.fit = se)
      b <- predict(expected, nd, type = "terms", na.action = get(action),
        reference = reference, se.fit = se)
      expect_equal(a, b, tolerance = 3e-7)
      expect_identical(is.null(.term_constant(a)), missing && action == "na.exclude")
      expect_equal(.term_constant(a), .term_constant(b), tolerance = 3e-7)
    }
  }
})

test_that("training omissions and grouped Cox terms follow stock attribute removal", {
  d <- .constant_test_data()
  d$x[[2L]] <- NA_real_
  for (action in c("na.omit", "na.exclude")) for (se in c(FALSE, TRUE)) {
    actual <- survivalr::coxph(Surv(time, status) ~ x + z, d, na.action = get(action))
    expected <- survival::coxph(Surv(time, status) ~ x + z, d, na.action = get(action))
    a <- predict(actual, type = "terms", se.fit = se)
    b <- predict(expected, type = "terms", se.fit = se)
    expect_equal(a, b, tolerance = 3e-7)
    expect_identical(is.null(.term_constant(a)), action == "na.exclude")
    nd <- d[c(1L, 3L, 4L, 5L), ]
    a <- predict(actual, nd, type = "terms", collapse = c("b", "a", "b", "a"), se.fit = se)
    b <- predict(expected, nd, type = "terms", collapse = c("b", "a", "b", "a"), se.fit = se)
    expect_equal(a, b, tolerance = 3e-7)
    expect_null(.term_constant(a))
    expect_null(.term_constant(b))
  }
})

test_that("AFT terms and sparse-only frailty terms have no Cox constant attribute", {
  d <- .constant_test_data()
  actual <- survivalr::survreg(Surv(time, status) ~ x + z, d)
  expected <- survival::survreg(Surv(time, status) ~ x + z, d)
  for (missing in c(FALSE, TRUE)) for (action in c("na.omit", "na.exclude", "na.pass"))
    for (se in c(FALSE, TRUE)) {
    nd <- d[1:4, ]
    if (missing) nd$x[[2L]] <- NA_real_
    a <- predict(actual, nd, type = "terms", na.action = get(action), se.fit = se)
    b <- predict(expected, nd, type = "terms", na.action = get(action), se.fit = se)
    expect_equal(a, b, tolerance = 3e-7)
    expect_null(.term_constant(a))
    expect_null(.term_constant(b))
  }
  form <- Surv(time, status) ~ frailty(g, sparse = TRUE, theta = .1)
  actual <- survivalr::coxph(form, d)
  expected <- survival::coxph(form, d)
  for (se in c(FALSE, TRUE)) for (new in c(FALSE, TRUE)) {
    a <- if (new) predict(actual, d[1:4, ], type = "terms", se.fit = se) else
      predict(actual, type = "terms", se.fit = se)
    b <- if (new) predict(expected, d[1:4, ], type = "terms", se.fit = se) else
      predict(expected, type = "terms", se.fit = se)
    expect_null(.term_constant(a))
    expect_null(.term_constant(b))
  }
})

test_that("new populations with mixed sparse frailty retain regular term constants", {
  d <- .constant_test_data()
  form <- Surv(time, status) ~ x + frailty(g, sparse = TRUE, theta = .1)
  actual <- survivalr::coxph(form, d)
  expected <- survival::coxph(form, d)
  for (se in c(FALSE, TRUE)) {
    a <- predict(actual, d[1:4, ], type = "terms", se.fit = se)
    b <- predict(expected, d[1:4, ], type = "terms", se.fit = se)
    expect_type(.term_constant(a), "double")
    expect_equal(.term_constant(a), .term_constant(b), tolerance = 3e-7)
  }
})
