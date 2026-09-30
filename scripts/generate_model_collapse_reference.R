#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/model_collapse_reference.json"
d <- ovarian
d$x <- (d$age - 60)/10
d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a", "unused"))
d$w <- rep(c(.7, 1, 1.4), length.out = nrow(d))
d$o <- (seq_len(nrow(d)) %% 5 - 2)/20
specs <- list(
  plain = list(rhs = "x + rx"),
  weighted = list(rhs = "x + rx + strata(resid.ds) + offset(o)", weights = TRUE),
  ridge = list(rhs = "x + ridge(rx, theta = 2)"),
  sparse = list(rhs = "x + frailty(cl, sparse = TRUE, theta = 0.4)"),
  cluster = list(rhs = "x + rx + cluster(cl)"),
  id = list(rhs = "x + rx", id = TRUE),
  subset = list(rhs = "x + rx", subset = TRUE),
  excluded = list(rhs = "x + rx", excluded = TRUE),
  aft = list(rhs = "x + rx", model = "survreg"),
  aft_ridge = list(rhs = "x + ridge(rx, theta = 2)", model = "survreg"),
  aft_excluded = list(rhs = "x + rx", model = "survreg", excluded = TRUE)
)
capture <- function(fun) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(result = result, warnings = I(warnings))
}
encode <- function(value) {
  if (is.list(value) && !is.null(value$error)) return(value)
  if (is.list(value)) return(list(fit = encode(value$fit), se_fit = encode(value$se.fit)))
  labels <- if (is.matrix(value)) rownames(value) else names(value)
  list(values = if (is.matrix(value)) unname(value) else I(unname(value)),
       names = if (is.null(labels)) NULL else I(labels),
       columns = if (is.matrix(value) && !is.null(colnames(value))) I(colnames(value)) else NULL)
}
group_values <- function(group) {
  lapply(if (is.factor(group)) as.character(group) else group, function(value) {
    if (is.numeric(value) && is.nan(value)) list(nan = TRUE) else value
  })
}
cases <- list()
for (name in names(specs)) {
  spec <- specs[[name]]
  model <- if (is.null(spec$model)) "coxph" else spec$model
  data <- d
  if (isTRUE(spec$excluded)) data$x[c(2, 9)] <- NA
  formula <- as.formula(paste("Surv(futime, fustat) ~", spec$rhs))
  call <- list(formula = formula, data = data, x = TRUE,
    na.action = if (isTRUE(spec$excluded)) na.exclude else na.omit)
  if (model == "coxph" && name != "cluster") call$robust <- FALSE
  if (isTRUE(spec$weights)) call$weights <- data$w
  if (isTRUE(spec$id)) call$id <- data$cl
  if (isTRUE(spec$subset)) call$subset <- seq_len(nrow(data)) %% 3 != 0
  fit <- do.call(if (model == "coxph") coxph else survreg, call)
  # Build the dense view from the already fitted sparse model. Its coefficients,
  # covariance and martingale residuals retain the fitted frailty contribution.
  # This avoids predict.coxph.penal's duplicate collapse and terms reconstruction.
  dense <- fit
  if (name == "sparse") {
    dense$x <- fit$x[, "x", drop = FALSE]
    dense$terms <- terms(Surv(futime, fustat) ~ x)
    dense$assign <- list(x = 1L)
    dense$pterms <- c(x = 0)
    class(dense) <- "coxph"
  }
  n <- if (isTRUE(spec$excluded)) nrow(data) else nrow(fit$x)
  groups <- list(numeric = rep(c(20, 3, 11), length.out = n),
    text = rep(c("20", "3", "11"), length.out = n),
    factor = factor(rep(c("a", "b", "c"), length.out = n), levels = c("c", "b", "a", "unused")),
    missing = rep(c("a", NA, "b"), length.out = n),
    nan_na = rep(c(NaN, NA_real_, 2, 1), length.out = n),
    logical = rep(c(TRUE, FALSE, NA), length.out = n))
  methods <- if (model == "coxph") c("martingale", "deviance", "score", "dfbeta", "dfbetas", "partial") else
    c("response", "deviance", "dfbeta", "dfbetas", "working", "ldcase", "ldresp", "ldshape", "matrix")
  for (group_name in names(groups)) for (method in methods) {
    group <- groups[[group_name]]
    raw <- capture(function() residuals(fit, type = method, collapse = group))
    expected <- raw; reference <- "unmodified stock residuals"
    if (name == "sparse" && method == "partial") {
      expected <- capture(function() rowsum(residuals(dense, type = method), group))
      reference <- "R dense-term partial residuals from the fitted sparse model, then rowsum"
    }
    if (is.list(raw$result) && !is.null(raw$result$error) && (isTRUE(spec$excluded) || is.logical(group))) {
      expected <- capture(function() {
        if (method == "deviance" && model == "coxph") {
          rr <- drop(rowsum(residuals(fit, type = "martingale"), group))
          status <- drop(rowsum(naresid(fit$na.action, fit$y[, ncol(fit$y)]), group))
          sign(rr) * sqrt(-2*(rr + ifelse(status == 0, 0, status * log(status - rr))))
        } else {
          summed <- rowsum(residuals(if (name == "sparse" && method == "partial") dense else fit,
                                    type = method), group)
          if (method == "partial") summed else drop(summed)
        }
      })
      reference <- "rowsum of R residuals, padding excluded rows first; deviance uses collapsed martingale and event counts"
    }
    cases[[length(cases) + 1L]] <- list(name = paste(name, method, group_name, sep = "/"),
      fit = name, kind = "residuals", type = method, collapse = group_values(group),
      levels = if (is.factor(group)) I(levels(group)) else NULL,
      reference = reference, raw = list(result = encode(raw$result), warnings = raw$warnings),
      expected = list(result = encode(expected$result), warnings = expected$warnings))
  }
  if (model == "coxph") for (group_name in names(groups)) for (method in c("lp", "risk", "expected", "survival", "terms")) {
    group <- groups[[group_name]]
    raw <- capture(function() predict(fit, type = method, collapse = group, se.fit = TRUE))
    expected <- raw; reference <- "unmodified stock predict"
    if (name == "sparse" || (name == "ridge" && method == "survival") ||
        (is.list(raw$result) && !is.null(raw$result$error) && isTRUE(spec$excluded))) {
      expected <- capture(function() {
        values <- predict(if (name == "sparse" && method == "terms") dense else fit,
          type = if (method == "survival") "expected" else method, se.fit = TRUE)
        if (name == "sparse" && method == "terms") {
          index <- as.integer(factor(fit$x[, 2L]))
          values$fit <- cbind(values$fit, fit$frail[index])
          values$se.fit <- cbind(values$se.fit, sqrt(fit$fvar[index]))
          colnames(values$fit) <- colnames(values$se.fit) <- names(fit$pterms)
        }
        if (method == "survival") {
          values$fit <- exp(-values$fit)
          values$se.fit <- values$se.fit * values$fit
        }
        list(fit = drop(rowsum(values$fit, group)), se.fit = drop(sqrt(rowsum(values$se.fit^2, group))))
      })
      reference <- if (name == "sparse")
        "R uncollapsed predictions; terms add fitted frailty and fvar; survival derives from expected; one rowsum"
      else if (method == "survival" && name == "ridge")
        "R expected predictions transformed to survival before grouping, with delta-method errors"
      else "R predictions padded before grouping; errors combined in quadrature"
    }
    cases[[length(cases) + 1L]] <- list(name = paste(name, "predict", method, group_name, sep = "/"),
      fit = name, kind = "predict", type = method, collapse = group_values(group),
      levels = if (is.factor(group)) I(levels(group)) else NULL,
      reference = reference, raw = list(result = encode(raw$result), warnings = raw$warnings),
      expected = list(result = encode(expected$result), warnings = expected$warnings))
  }
  if (name == "plain") {
    nd <- data[seq_len(8), ]; nd$x[2] <- NA; nd$rx[7] <- NA
    group <- factor(c("a", "b", NA, "c", "b", "a", "c", "a"), levels = c("c", "b", "a", "unused"))
    for (action in c("na.pass", "na.omit", "na.exclude", "na.fail")) for (method in c("lp", "risk", "terms", "expected", "survival")) {
      raw <- capture(function() predict(fit, newdata = nd, type = method, collapse = group,
        na.action = action, se.fit = TRUE))
      expected <- raw; reference <- "unmodified stock newdata predict"
      if (action == "na.exclude" && !is.null(raw$result$error)) {
        expected <- capture(function() {
          values <- predict(fit, newdata = nd, type = method, na.action = action, se.fit = TRUE)
          pad <- function(x) { if (is.matrix(x)) x[is.na(group), ] <- NA_real_ else x[is.na(group)] <- NA_real_; x }
          list(fit = drop(rowsum(pad(values$fit), group)),
               se.fit = drop(sqrt(rowsum(pad(values$se.fit)^2, group))))
        })
        reference <- "R newdata predictions padded for covariate and collapse missingness before grouping"
      }
      cases[[length(cases) + 1L]] <- list(name = paste("newdata", method, action, sep = "/"),
        fit = name, kind = "predict", type = method, collapse = group_values(group), levels = I(levels(group)),
        newdata = nd, na_action = action, reference = reference,
        raw = list(result = encode(raw$result), warnings = raw$warnings),
        expected = list(result = encode(expected$result), warnings = expected$warnings))
    }
  }
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()), survival = as.character(packageVersion("survival"))),
  data = d, levels = I(levels(d$cl)), specs = specs, cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "model collapse references written\n")
