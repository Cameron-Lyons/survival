.aft_strata_data <- function() {
  d <- survival::lung[1:90, c("time", "status", "age", "sex")]
  d$g <- factor(rep(c("a", "b", "c"), 30), levels = c("a", "b", "c", "never"))
  d$w <- rep(c(.7, 1, 1.4), 30)
  d$id <- rep(seq_len(30), each = 3)
  d
}

test_that("AFT retains empty interior scale strata like R", {
  for (drop in c("a", "b")) for (action in list(na.omit, na.exclude)) {
    for (robust in c(FALSE, TRUE)) for (via in c("subset", "missing")) {
      d <- .aft_strata_data()
      selected <- seq_len(nrow(d))
      if (via == "subset") selected <- which(d$g != drop) else d$age[d$g == drop] <- NA
      form <- survival::Surv(time, status) ~ age + sex + strata(g)
      actual <- survreg(form, d, subset = selected, weights = w, na.action = action,
                         robust = robust, cluster = id, x = TRUE, model = TRUE, score = TRUE)
      expected <- survival::survreg(form, d, subset = selected, weights = w, na.action = action,
                                     robust = robust, cluster = id, x = TRUE, model = TRUE, score = TRUE)
      expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 2e-7)
      expect_equal(vcov(actual), vcov(expected), tolerance = 2e-7)
      expect_equal(as.numeric(actual$scale), unname(expected$scale), tolerance = 2e-7)
      expect_equal(as.numeric(predict(actual, type = "lp")), as.numeric(predict(expected, type = "lp")), tolerance = 2e-7)
      expect_equal(unname(residuals(actual, type = "dfbeta")), unname(residuals(expected, type = "dfbeta")), tolerance = 2e-7)
      new <- d[c(1, 2, 3), ]
      new$age <- c(50, 60, 70)
      expect_equal(predict(actual, new, type = "quantile", p = c(.25, .75), se.fit = TRUE),
                   predict(expected, new, type = "quantile", p = c(.25, .75), se.fit = TRUE),
                   tolerance = 2e-7, ignore_attr = TRUE)
      expect_equal(rownames(summary(actual)$table), rownames(summary(expected)$table))
    }
  }
})

test_that("trailing scales and penalized empty strata preserve estimable results", {
  d <- .aft_strata_data()
  for (drop in c("a", "b", "c")) for (penalized in c(FALSE, TRUE)) {
    form <- if (penalized) survival::Surv(time, status) ~ ridge(age, theta = 2) + sex + strata(g)
            else survival::Surv(time, status) ~ age + sex + strata(g)
    fit <- survreg(form, d, subset = g != drop, weights = w)
    # ridge() is evaluated before subset; preserve its original scaling.
    selected <- which(d$g != drop)
    compact_data <- d
    replacement <- setdiff(c("a", "b", "c"), drop)[1L]
    compact_data$g[compact_data$g == drop] <- replacement
    compact <- survival::survreg(form, compact_data, subset = selected, weights = w)
    expect_equal(unname(coef(fit)), unname(coef(compact)), tolerance = 2e-7)
    expect_identical(as.character(unlist(fit$strata_levels)), c("a", "b", "c"))
    used <- which(c("a", "b", "c") != drop)
    keep <- c(seq_along(coef(fit)), length(coef(fit)) + used)
    expect_equal(unname(vcov(fit)[keep, keep]), unname(vcov(compact)), tolerance = 2e-7)
    expect_equal(unname(fit$scale[used]), unname(compact$scale), tolerance = 2e-7)
    empty <- length(coef(fit)) + match(drop, c("a", "b", "c"))
    expect_equal(unname(vcov(fit)[empty, ]), rep(0, nrow(vcov(fit))))
    new <- d[d$g != drop, ][1:4, ]
    expect_equal(as.numeric(predict(fit, new, type = "quantile", p = .75)),
                 as.numeric(predict(compact, new, type = "quantile", p = .75)), tolerance = 2e-7)
    expect_error(survreg(form, d, subset = g != drop, scale = 1), "multiple strata")
  }
})

test_that("multiple strata terms recombine observed groups after row removal", {
  d <- .aft_strata_data()
  for (form in list(survival::Surv(time, status) ~ age + strata(g) + strata(sex),
                    survival::Surv(time, status) ~ age + strata(g, sex))) {
    fit <- survreg(form, d, subset = g != "b")
    expected <- survival::survreg(form, d, subset = g != "b")
    expect_equal(unname(coef(fit)), unname(coef(expected)), tolerance = 2e-7)
    expect_equal(as.numeric(fit$scale), unname(expected$scale), tolerance = 2e-7)
    expect_equal(vcov(fit), vcov(expected), tolerance = 2e-7)
  }
})


test_that("AFT strata interactions retain empty design columns", {
  d <- .aft_strata_data()
  for (drop in c("a", "b")) for (dist in c("weibull", "gaussian", "lognormal")) {
    form <- survival::Surv(time, status) ~ age * strata(g)
    fit <- survreg(form, d, subset = g != drop, weights = w, dist = dist, x = TRUE)
    expected <- survival::survreg(form, d, subset = g != drop, weights = w, dist = dist, x = TRUE)
    expect_equal(names(coef(fit)), names(coef(expected)))
    expect_equal(unname(coef(fit)), unname(coef(expected)), tolerance = 2e-7)
    expect_equal(vcov(fit), vcov(expected), tolerance = 2e-7)
    expect_equal(as.numeric(fit$scale), unname(expected$scale), tolerance = 2e-7)
    expect_equal(unname(model.matrix(fit)), unname(model.matrix(expected)),
                 tolerance = 2e-7, ignore_attr = TRUE)
    expect_equal(as.numeric(predict(fit, type = "lp")), as.numeric(predict(expected, type = "lp")), tolerance = 2e-7)
  }
})
