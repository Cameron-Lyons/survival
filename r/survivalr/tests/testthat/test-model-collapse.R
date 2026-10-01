.collapse_data <- function() {
  d <- survival::ovarian
  d$x <- (d$age - 60)/10
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)),
                 levels = c("c", "b", "a", "unused"))
  d$w <- rep(c(.7, 1, 1.4), length.out = nrow(d))
  d$o <- (seq_len(nrow(d)) %% 5 - 2)/20
  d
}

.collapse_groups <- function(n) list(
  numeric = rep(c(20, 3, 11), length.out = n),
  text = rep(c("20", "3", "11"), length.out = n),
  factor = factor(rep(c("a", "b", "c"), length.out = n),
                  levels = c("c", "b", "a", "unused")),
  missing = rep(c("a", NA, "b"), length.out = n),
  nan_na = rep(c(NaN, NA_real_, 2, 1), length.out = n),
  logical = rep(c(TRUE, FALSE, NA), length.out = n)
)

.collapse_compare <- function(actual, expected, group) {
  expect_identical(dim(actual), dim(expected))
  labels <- if (is.matrix(expected)) rownames(expected) else names(expected)
  # Some stock methods discard row names when dropping a one-column matrix.
  # The group labels themselves remain independently available from base rowsum.
  if (is.null(labels)) labels <- rownames(suppressWarnings(rowsum(seq_along(group), group)))
  expect_identical(if (is.matrix(actual)) rownames(actual) else names(actual),
                   labels)
  expect_equal(unname(actual), unname(expected), ignore_attr = TRUE, tolerance = 2e-7)
}

test_that("Cox and AFT residual groups retain factor order and missing labels", {
  specs <- list(plain = "x + rx", weighted = "x + rx + strata(resid.ds) + offset(o)",
                ridge = "x + ridge(rx, theta = 2)", sparse = "x + frailty(cl, sparse = TRUE, theta = .4)",
                excluded = "x + rx", aft = "x + rx", aft_ridge = "x + ridge(rx, theta = 2)",
                aft_excluded = "x + rx")
  for (name in names(specs)) {
    d <- .collapse_data()
    excluded <- grepl("excluded", name)
    if (excluded) d$x[c(2, 9)] <- NA
    aft <- startsWith(name, "aft")
    options <- list(formula = as.formula(paste("Surv(futime,fustat) ~", specs[[name]])),
                    data = d, x = TRUE, na.action = if (excluded) na.exclude else na.omit)
    if (!aft) options$robust <- FALSE
    if (name == "weighted") options$weights <- d$w
    actual <- do.call(if (aft) survreg else coxph, options)
    expected <- do.call(if (aft) survival::survreg else survival::coxph, options)
    dense <- expected
    if (name == "sparse") {
      dense$x <- expected$x[, "x", drop = FALSE]
      dense$terms <- terms(Surv(futime, fustat) ~ x)
      dense$assign <- list(x = 1L)
      dense$pterms <- c(x = 0)
      class(dense) <- "coxph"
    }
    types <- if (aft) c("response", "deviance", "dfbeta", "dfbetas", "working", "ldcase",
                        "ldresp", "ldshape", "matrix") else
      c("martingale", "deviance", "score", "dfbeta", "dfbetas", "partial")
    for (group in .collapse_groups(nrow(d))) for (type in types) {
      reference <- suppressWarnings(if ((excluded || is.logical(group)) && !aft && type == "deviance") {
        rr <- drop(rowsum(residuals(expected, type = "martingale"), group))
        events <- drop(rowsum(naresid(expected$na.action, expected$y[, ncol(expected$y)]), group))
        sign(rr) * sqrt(-2*(rr + ifelse(events == 0, 0, events*log(events - rr))))
      } else if (excluded || is.logical(group) || (name == "sparse" && type == "partial")) {
        summed <- rowsum(residuals(if (name == "sparse") dense else expected, type = type), group)
        if (type == "partial") summed else drop(summed)
      } else residuals(expected, type = type, collapse = group))
      .collapse_compare(suppressWarnings(residuals(actual, type = type, collapse = group)), reference, group)
    }
  }
})

test_that("grouped Cox predictions retain names and combine errors in quadrature", {
  d <- .collapse_data()
  for (name in c("plain", "ridge", "sparse", "excluded")) {
    data <- d
    if (name == "excluded") data$x[c(2, 9)] <- NA
    rhs <- switch(name, ridge = "x + ridge(rx, theta = 2)",
                  sparse = "x + frailty(cl, sparse = TRUE, theta = .4)", "x + rx")
    options <- list(formula = as.formula(paste("Surv(futime,fustat) ~", rhs)),
                    data = data, robust = FALSE, x = TRUE,
                    na.action = if (name == "excluded") na.exclude else na.omit)
    actual <- do.call(coxph, options)
    expected <- do.call(survival::coxph, options)
    dense <- expected
    if (name == "sparse") {
      dense$x <- expected$x[, "x", drop = FALSE]
      dense$terms <- terms(Surv(futime, fustat) ~ x)
      dense$assign <- list(x = 1L)
      dense$pterms <- c(x = 0)
      class(dense) <- "coxph"
    }
    for (group in .collapse_groups(nrow(d))) for (type in c("lp", "risk", "expected", "survival", "terms")) {
      source <- if (name == "sparse" && type == "terms") dense else expected
      value <- predict(source, type = if (type == "survival") "expected" else type, se.fit = TRUE)
      if (name == "sparse" && type == "terms") {
        index <- as.integer(factor(expected$x[, 2L]))
        value$fit <- cbind(value$fit, expected$frail[index])
        value$se.fit <- cbind(value$se.fit, sqrt(expected$fvar[index]))
      }
      if (type == "survival") {
        value$fit <- exp(-value$fit)
        value$se.fit <- value$se.fit * value$fit
      }
      reference <- suppressWarnings(list(fit = drop(rowsum(value$fit, group)),
                                          se.fit = drop(sqrt(rowsum(value$se.fit^2, group)))))
      result <- suppressWarnings(predict(actual, type = type, se.fit = TRUE, collapse = group))
      .collapse_compare(result$fit, reference$fit, group)
      .collapse_compare(result$se.fit, reference$se.fit, group)
    }
  }
})

test_that("new-data collapse contributes to missing-data handling", {
  d <- .collapse_data()
  actual <- coxph(Surv(futime,fustat) ~ x + rx, d)
  expected <- survival::coxph(Surv(futime,fustat) ~ x + rx, d)
  nd <- d[seq_len(8), ]; nd$x[2] <- NA; nd$rx[7] <- NA
  group <- factor(c("a", "b", NA, "c", "b", "a", "c", "a"), levels = c("c", "b", "a", "unused"))
  for (action in c("na.pass", "na.omit", "na.exclude")) for (type in c("lp", "risk", "expected", "survival", "terms")) {
    reference <- suppressWarnings(if (action == "na.exclude") {
      value <- predict(expected, nd, type = type, se.fit = TRUE, na.action = action)
      pad <- function(x) { if (is.matrix(x)) x[is.na(group), ] <- NA_real_ else x[is.na(group)] <- NA_real_; x }
      list(fit = drop(rowsum(pad(value$fit), group)),
           se.fit = drop(sqrt(rowsum(pad(value$se.fit)^2, group))))
    } else predict(expected, nd, type = type, se.fit = TRUE, collapse = group, na.action = action))
    result <- suppressWarnings(predict(actual, nd, type = type, se.fit = TRUE, collapse = group,
                                       na.action = action))
    labels_group <- if (action == "na.omit") group[complete.cases(nd[, c("x", "rx")], group)] else group
    .collapse_compare(result$fit, reference$fit, labels_group)
    .collapse_compare(result$se.fit, reference$se.fit, labels_group)
  }
  expect_error(predict(actual, nd, collapse = group, na.action = "na.fail"), "missing")
  clean <- d[seq_len(8), ]
  expect_error(predict(actual, clean, collapse = group, na.action = "na.fail"), "missing")
})

test_that("collapse warnings are R conditions and saved factors retain order", {
  d <- .collapse_data()
  fit <- coxph(Surv(futime,fustat) ~ x + rx, d)
  group <- .collapse_groups(nrow(d))$missing
  expect_warning(residuals(fit, collapse = group), "missing values for 'group'")
  expect_warning(predict(fit, collapse = group), "missing values for 'group'")
  aft <- survreg(Surv(futime,fustat) ~ x + rx, d)
  expect_warning(residuals(aft, collapse = group), "missing values for 'group'")
  for (key in c("cluster", "id")) {
    options <- list(formula = Surv(futime,fustat) ~ x + rx, data = d, subset = seq_len(nrow(d)) <= 20)
    options[[key]] <- d$cl
    saved <- unserialize(serialize(do.call(coxph, options), NULL))
    result <- residuals(saved, type = "dfbeta", collapse = TRUE)
    expect_identical(rownames(result), c("c", "b", "a"))
    expect_equal(result, residuals(saved, type = "dfbeta", collapse = d$cl[seq_len(20)]))
  }
  expect_error(residuals(fit, collapse = TRUE), "no cluster or id")
  for (type in c("schoenfeld", "scaledsch")) {
    expect_equal(residuals(fit, type = type, collapse = c(1, 2)), residuals(fit, type = type))
    expect_equal(residuals(fit, type = type, collapse = TRUE), residuals(fit, type = type))
  }
})
