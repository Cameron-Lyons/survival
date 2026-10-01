.tt_penalty_compare <- function(actual, expected, saved = FALSE) {
  expect_identical(as.character(names(coef(actual))), as.character(names(coef(expected))))
  expect_equal(as.numeric(coef(actual)), as.numeric(coef(expected)), tolerance = 4e-7)
  if (!is.null(expected$var)) expect_equal(vcov(actual), vcov(expected), tolerance = 4e-7)
  expect_equal(as.numeric(actual$loglik), expected$loglik, tolerance = 4e-7)
  expect_equal(as.numeric(actual$df), expected$df, tolerance = 4e-7)
  expect_equal(as.numeric(actual$frail), as.numeric(expected$frail), tolerance = 4e-7)
  expect_equal(as.numeric(actual$fvar), as.numeric(expected$fvar), tolerance = 4e-7)
  expect_equal(unname(fitted(actual)), unname(fitted(expected)), tolerance = 4e-7)
  expect_equal(unname(model.matrix(actual)), unname(expected$x), ignore_attr = TRUE)
  expect_identical(attr(model.matrix(actual), "assign"), attr(expected$x, "assign"))
  expect_equal(summary(actual)$coefficients, summary(expected)$coefficients, tolerance = 4e-7)
  expect_equal(summary(actual)$print2, summary(expected)$print2)
  expanded <- data.frame(t = expected$y[,1], event = expected$y[,2],
                         group = expected$strata, lp = expected$linear.predictors)
  reference <- survival::coxph(survival::Surv(t,event) ~ offset(lp) + strata(group),
    expanded, weights = expected$weights, ties = expected$method,
    control = survival::coxph.control(iter.max = 0))
  residual <- residuals(actual)
  if (!is.null(expected$na.action)) residual <- residual[-as.integer(expected$na.action)]
  expect_equal(unname(residual), unname(residuals(reference)), tolerance = 4e-7)
  if (saved) {
    restored <- unserialize(serialize(actual, NULL))
    expect_equal(summary(restored)$coefficients, summary(actual)$coefficients, tolerance = 4e-7)
    expect_identical(summary(restored)$print2, summary(actual)$print2)
    expect_equal(model.matrix(restored), model.matrix(actual))
    expect_equal(residuals(restored), residuals(actual))
  }
}

test_that("sparse numeric and dense factor penalties preserve all frailty families", {
  d <- survival::ovarian
  d$x <- (d$age - 60)/10
  for (distribution in c("gamma", "gaussian", "t")) for (sparse in c(TRUE, FALSE)) {
    for (constructor in list(survival::frailty, survivalr::frailty)) {
      fun <- function(x,t,...) constructor(x, distribution = distribution,
                                           sparse = sparse, theta = .4)
      for (method in c("efron", "breslow")) {
        actual <- coxph(Surv(futime,fustat) ~ x + tt(rx), d, tt = fun, ties = method,
                         x = TRUE, robust = FALSE)
        expected <- survival::coxph(survival::Surv(futime,fustat) ~ x + tt(rx), d,
                                    tt = fun, ties = method, x = TRUE, robust = FALSE)
        .tt_penalty_compare(actual, expected, saved = method == "efron")
      }
    }
  }
})

test_that("transformed frailties retain searched and df controller histories", {
  d <- survival::ovarian
  d$x <- (d$age - 60)/10
  for (distribution in c("gamma", "gaussian")) for (sparse in c(TRUE, FALSE)) {
    for (method in c("df", "search")) {
      fun <- function(x,t,...) {
        options <- list(x = x, distribution = distribution, sparse = sparse)
        if (method == "df") options$df <- .7
        do.call(survival::frailty, options)
      }
      control <- survival::coxph.control(eps = 1e-10, iter.max = 50, outer.max = 50)
      actual <- coxph(Surv(futime,fustat) ~ x + tt(rx), d, tt = fun, x = TRUE,
                       robust = FALSE, control = control)
      expected <- survival::coxph(survival::Surv(futime,fustat) ~ x + tt(rx), d,
                                  tt = fun, x = TRUE, robust = FALSE, control = control)
      .tt_penalty_compare(actual, expected, saved = TRUE)
      expect_equal(as.numeric(actual$history[[1]]$theta), as.numeric(expected$history[[1]]$theta),
                   tolerance = 4e-7)
    }
  }
})

test_that("sparse-only fits and original group values survive storage", {
  d <- survival::ovarian
  d$x <- (d$age - 60)/10
  d$g <- factor(d$rx, levels = c(2, 1, 9))
  for (kind in c("sparse_only", "factor", "raw_codes", "subset", "missing")) {
    data <- d
    if (kind == "missing") data$x[c(2,9)] <- NA
    form <- if (kind == "sparse_only") Surv(futime,fustat) ~ tt(rx)
            else if (kind == "factor") Surv(futime,fustat) ~ x + tt(g)
            else Surv(futime,fustat) ~ x + tt(rx)
    fun <- function(x,t,...) {
      value <- survival::frailty(x, sparse = TRUE, theta = .4)
      if (kind == "raw_codes") value[] <- ifelse(value == 1, 41, 73)
      value
    }
    options <- list(formula = form, data = data, tt = fun, x = TRUE, robust = FALSE,
                    na.action = if (kind == "missing") na.exclude else na.omit)
    if (kind == "subset") options$subset <- seq_len(nrow(data)) %% 4 != 0
    .tt_penalty_compare(do.call(coxph, options), do.call(survival::coxph, options), saved = TRUE)
  }
})

test_that("numeric penalty vectors and sparse formatters keep the R call contract", {
  d <- survival::ovarian
  transforms <- list(
    function(x,t,...) {
      value <- survival::ridge(x * log(t), theta = 2)
      dim(value) <- NULL
      value
    },
    function(x,t,...) {
      value <- survival::frailty(x, sparse = TRUE, theta = .4)
      formatter <- attr(value, "printfun")
      attr(value, "printfun") <- function(coef,var,var2,df,history) {
        stopifnot(missing(var2))
        formatter(coef,var,,df,history)
      }
      value
    })
  for (fun in transforms) {
    actual <- coxph(Surv(futime,fustat) ~ age + tt(rx), d, tt = fun, x = TRUE)
    expected <- survival::coxph(survival::Surv(futime,fustat) ~ age + tt(rx), d, tt = fun, x = TRUE)
    .tt_penalty_compare(actual, expected, saved = TRUE)
  }
})

test_that("R frailty constructors encode their own numeric and logical labels", {
  for (x in list(c(2,1,2,NA_real_), c(TRUE,FALSE,TRUE,NA),
                 factor(c("a","b","a",NA), levels = c("b","a","unused")))) {
    for (sparse in c(TRUE,FALSE)) {
      actual <- survivalr::frailty(x,sparse = sparse,theta = .4)
      expected <- survival::frailty(x,sparse = sparse,theta = .4)
      expect_identical(class(actual),class(expected))
      if (sparse) expect_equal(as.numeric(actual),as.numeric(expected))
      else {
        expect_identical(as.character(actual),as.character(expected))
        expect_identical(levels(actual),levels(expected))
      }
    }
  }
})
