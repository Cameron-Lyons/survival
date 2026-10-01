.subset_test_data <- function() {
  i <- seq_len(36)
  data.frame(time = 2+(i*13)%%31, status = as.integer(i%%4 != 0), x = sin(i),
             z = cos(i/3), g = factor(rep(c("c", "a", "b"), 12)),
             w = .5+i%%4, o = sin(i/5)/3, id = rep(1:9, 4),
             row.names = paste0("row", i))
}

test_that("R model subsets use one-based indices and evaluate data expressions", {
  data <- .subset_test_data()
  selected <- c(31, 25, 2, 7, 19, 8, 12, 23, 4, 5, 29, 30)
  for (family in c("coxph", "survreg")) {
    own <- get(family, asNamespace("survivalr"))
    reference <- get(family, asNamespace("survival"))
    for (subset in list(selected, as.integer(selected), -c(2, 5, 9), c(TRUE, FALSE),
                         rownames(data)[selected], selected + .4)) {
      fit <- own(Surv(time, status) ~ x + offset(o), data, subset = subset, weights = w)
      expected <- reference(survival::Surv(time, status) ~ x + offset(o), data, subset = subset, weights = w)
      expect_equal(unname(coef(fit)), unname(coef(expected)), tolerance = 1e-8)
      expect_equal(unname(vcov(fit)), unname(vcov(expected)), tolerance = 1e-8)
    }
    fit <- own(Surv(time, status) ~ x, data, subset = x > 0)
    expected <- reference(survival::Surv(time, status) ~ x, data, subset = x > 0)
    expect_equal(unname(coef(fit)), unname(coef(expected)), tolerance = 1e-8)
  }
})

test_that("missing subset rows reach omission and exclusion in the shared model frame", {
  data <- .subset_test_data()
  data$x[c(7, 12)] <- NA
  for (family in c("coxph", "survreg")) {
    own <- get(family, asNamespace("survivalr"))
    reference <- get(family, asNamespace("survival"))
    form <- if (family == "coxph") survival::Surv(time, status) ~ x + strata(g) + offset(o)
            else survival::Surv(time, status) ~ x + offset(o)
    for (action in list(na.omit, na.exclude)) for (subset in list(
      c(31, NA, 2, 7, 19, 8, 12, 23, 4, 5, 29, 30),
      c(TRUE, NA, FALSE), c("missing", paste0("row", 1:31)))) {
      fit <- own(form, data,
                  subset = subset, weights = w, na.action = action)
      expected <- reference(form, data,
                            subset = subset, weights = w, na.action = action)
      expect_equal(unname(coef(fit)), unname(coef(expected)), tolerance = 1e-8)
      expect_equal(unname(vcov(fit)), unname(vcov(expected)), tolerance = 1e-8)
      expect_equal(as.numeric(predict(fit, type = "lp")), as.numeric(predict(expected, type = "lp")),
                    tolerance = 1e-8)
    }
    expect_error(own(Surv(time, status) ~ x, data, subset = c(1, NA, 4), na.action = na.fail), "missing")
    expect_error(own(Surv(time, status) ~ x, data, subset = FALSE), "no rows|observations")
    expect_error(own(Surv(time, status) ~ x, data, subset = 40), "bounds")
    expect_error(own(Surv(time, status) ~ x, data, subset = c(-1, 3)), "negative")
  }
})

test_that("formula environments and data masking govern subset evaluation", {
  data <- .subset_test_data()
  form <- local({ threshold <- .2; survival::Surv(time, status) ~ x })
  for (family in c("coxph", "survreg")) {
    own <- get(family, asNamespace("survivalr"))
    reference <- get(family, asNamespace("survival"))
    actual <- own(form, data, subset = x > threshold)
    expected <- reference(form, data, subset = x > threshold)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-8)
    data$threshold <- -.2
    actual <- own(form, data, subset = x > threshold)
    expected <- reference(form, data, subset = x > threshold)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-8)
  }
})

test_that("subset selection preserves survival status coding", {
  data <- .subset_test_data()
  data$status <- data$status + 1
  selected <- which(data$status == 1)
  fit <- survfit(Surv(time, status) ~ 1, data, subset = selected)
  reference <- survival::survfit(survival::Surv(time, status) ~ 1, data, subset = selected)
  expect_equal(as.numeric(fit$n_event), reference$n.event)
  expect_equal(as.numeric(fit$surv), reference$surv)
  expect_equal(sum(as.numeric(fit$n_event)), 0)
})

test_that("curves, tests, concordance and redistribution share R selection", {
  data <- .subset_test_data()
  for (selected in list(c(31, 2, 7, 19, 8, 12, 23, 4, 5, 29, 30),
                        c(TRUE, FALSE), -c(2, 5, 9),
                        c("row31", "missing", "row2", "row7", "row19", "row8", "row12", "row23"))) {
    fit <- survfit(Surv(time, status) ~ g, data, subset = selected, weights = w)
    expected <- get("survfit.formula", asNamespace("survival"))(
      survival::Surv(time, status) ~ g, data, subset = selected, weights = w)
    frame <- as.data.frame(fit)
    expect_equal(frame$time, expected$time)
    expect_equal(frame$n.event, expected$n.event)
    expect_equal(frame$surv, expected$surv)
    actual <- survdiff(Surv(time, status) ~ g, data, subset = selected)
    expected <- survival::survdiff(survival::Surv(time, status) ~ g, data, subset = selected)
    expect_equal(as.numeric(actual$chisq), expected$chisq)
    for (response in list(Surv(data$time, data$status), survival::Surv(data$time, data$status))) {
      direct_selected <- if (is.character(selected)) match(selected, rownames(data)) else selected
      direct <- survdiff(response, group = data$g, subset = direct_selected)
      expect_equal(as.numeric(direct$chisq), expected$chisq)
    }
    actual <- concordance(Surv(time, status) ~ x, data, subset = selected, weights = w)
    expected <- get("concordance.formula", asNamespace("survival"))(
      survival::Surv(time, status) ~ x, data, subset = selected, weights = w)
    expect_equal(as.numeric(actual$concordance), expected$concordance)
    expect_equal(as.numeric(actual$var), expected$var)
    legacy <- suppressWarnings(survConcordance(Surv(time, status) ~ x, data,
                                               subset = selected, weights = w))
    legacy_expected <- suppressWarnings(survival::survConcordance(
      survival::Surv(time, status) ~ x, data, subset = selected, weights = w))
    for (field in c("concordance", "stats", "n", "std.err")) {
      expect_equal(legacy[[field]], legacy_expected[[field]])
    }
    expect_equal(as.numeric(rttright(Surv(time, status) ~ 1, data,
                                    subset = selected, weights = w)),
                 as.numeric(survival::rttright(survival::Surv(time, status) ~ 1, data,
                                              subset = selected, weights = w)))
  }
})

test_that("direct responses, formula strings and scalar selections convert once", {
  data <- .subset_test_data()
  for (selected in list(3L, c(31, 2, 7, 7, 19), c(TRUE, FALSE), c(31, NA, 2, 7))) {
    native <- survival::Surv(data$time, data$status)
    response <- Surv(data$time, data$status)
    expected <- get("survfit.formula", asNamespace("survival"))(
      survival::Surv(time, status) ~ 1, data, subset = selected)
    for (fit in list(survfit(native, subset = selected), survfit(response, subset = selected),
                     survfit("Surv(time, status) ~ 1", data, subset = selected))) {
      expect_equal(as.numeric(fit$time), expected$time)
      expect_equal(as.numeric(fit$surv), expected$surv)
    }
    actual <- pyears(time = data$time, event = data$status, subset = selected, scale = 1)
    expected <- survival::pyears(survival::Surv(time, status) ~ 1, data,
                                  subset = selected, scale = 1)
    expect_equal(as.numeric(actual$pyears), as.numeric(expected$pyears))
    expect_equal(as.numeric(actual$event), as.numeric(expected$event))
  }
})

test_that("condensed counting-process data applies expressions and missing selections", {
  data <- data.frame(start = rep(0:2, 4), stop = rep(1:3, 4),
                     status = rep(c(0, 0, 1), 4), id = rep(1:4, each = 3),
                     x = rep(c(0, 1, 0, 1), each = 3))
  for (selected in list(c(1, 2, 3, 7, 8, 9), c(1, NA, 2, 3, 7, 8, 9))) {
    actual <- survcondense(Surv(start, stop, status) ~ x, data, subset = selected,
                           id = id, na.action = na.omit)
    expected <- survival::survcondense(Surv(start, stop, status) ~ x, data,
                                       subset = selected, id = id, na.action = na.omit)
    expect_equal(actual, expected, ignore_attr = TRUE)
  }
})
