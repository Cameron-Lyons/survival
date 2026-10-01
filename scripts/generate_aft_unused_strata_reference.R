#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/aft_unused_strata_reference.json"
# Stock survreg uses max(strata), then assigns all factor levels to the scales.
# Correct only the empty-stratum failures: retain the full scale count in fits
# and methods, and use observed scales for penalized effective sample size.
# The installed R likelihood and optimization code stays unchanged.
source <- deparse(survival::survreg)
stopifnot(sum(grepl("nstrata <- max(strata)", source, fixed = TRUE)) == 1L)
corrected <- eval(parse(text = sub("nstrata <- max(strata)",
  "nstrata <- nlevels(strata.keep)", source, fixed = TRUE)), envir = asNamespace("survival"))
penal_source <- paste(deparse(survival:::survpenal.fit), collapse = "\n")
stopifnot(grepl("temp <- mean(exp(fit0$coef[-1]))", penal_source, fixed = TRUE),
          grepl("solve(matrix(fit0$var, 1 + \n        nstrat2))", penal_source, fixed = TRUE))
penal_source <- sub("temp <- mean(exp(fit0$coef[-1]))",
  "temp <- mean(exp(fit0$coef[if (nstrat2 > 0) 1 + sort(unique(strata)) else 2]))",
  penal_source, fixed = TRUE)
penal_source <- sub("solve(matrix(fit0$var, 1 + \n        nstrat2))",
  "solve(matrix(fit0$var, 1 + nstrat2)[c(1, if (nstrat2 > 0) 1 + sort(unique(strata))), c(1, if (nstrat2 > 0) 1 + sort(unique(strata))), drop = FALSE])",
  penal_source, fixed = TRUE)
reference_env <- new.env(parent = asNamespace("survival"))
reference_env$survpenal.fit <- eval(parse(text = penal_source), envir = reference_env)
environment(corrected) <- reference_env
predict_source <- deparse(survival:::predict.survreg)
stopifnot(sum(grepl("nstrata <- max(strata)", predict_source, fixed = TRUE)) == 1L)
corrected_predict <- eval(parse(text = sub("nstrata <- max(strata)",
  "nstrata <- length(object$scale)", predict_source, fixed = TRUE)), envir = asNamespace("survival"))
residual_source <- deparse(survival:::residuals.survreg)
stopifnot(sum(grepl("nstrata <- max(strata)", residual_source, fixed = TRUE)) == 1L)
reference_env$residuals.survreg <- eval(parse(text = sub("nstrata <- max(strata)",
  "nstrata <- length(object$scale)", residual_source, fixed = TRUE)), envir = reference_env)
base <- lung[1:90, c("time", "status", "age", "sex")]
base$g <- factor(rep(c("a", "b", "c"), length.out = nrow(base)), levels = c("a", "b", "c", "never"))
base$w <- rep(c(.7, 1, 1.4, 1.1), length.out = nrow(base))
base$o <- (seq_len(nrow(base)) %% 5 - 2) / 20
base$id <- (seq_len(nrow(base)) - 1L) %/% 3L
new <- data.frame(age = c(50, 60, 70), sex = c(1, 2, 1), g = factor(c("a", "b", "c"), levels = levels(base$g)), o = c(0, 0, 0))
scenarios <- list(interior_subset = list(drop = "b", via = "subset"),
  leading_subset = list(drop = "a", via = "subset"),
  trailing_subset = list(drop = "c", via = "subset"),
  middle_only = list(drop = c("a", "c"), via = "subset"),
  response_omit = list(drop = "b", via = "time"),
  predictor_exclude = list(drop = "b", via = "age", action = "na.exclude"),
  weight_omit = list(drop = "c", via = "w"),
  subset_then_omit = list(drop = "b", via = "subset", extra = TRUE))
frames <- list()
cases <- list()
for (scenario in names(scenarios)) for (kind in c("plain", "robust", "ridge")) {
  for (dist in c("weibull", "gaussian", "lognormal")) {
    cat(scenario, kind, dist, "\n")
    spec <- scenarios[[scenario]]
    d <- base
    drop <- d$g %in% spec$drop
    selected <- NULL
    if (spec$via == "subset") selected <- which(!drop) else d[[spec$via]][drop] <- NA
    if (isTRUE(spec$extra)) d$age[seq_len(nrow(d)) %% 13 == 0] <- NA
    frames[[scenario]] <- d
    action <- if (is.null(spec$action)) "na.omit" else spec$action
    formula <- paste("Surv(time, status) ~", if (kind == "ridge") "ridge(age, theta=2)" else "age",
                     if (kind == "ridge") "+ sex + strata(g)" else "+ sex + offset(o) + strata(g)")
    fit_args <- list(formula = as.formula(formula), data = d, subset = selected, weights = d$w,
                     na.action = get(action), dist = dist, x = TRUE, y = TRUE, model = TRUE, score = TRUE)
    if (kind == "robust") fit_args$cluster <- d$id
    stock <- tryCatch(do.call(survival::survreg, fit_args), error = function(e) list(error = conditionMessage(e)))
    if (!is.null(stock$error)) {
      stopifnot(grepl("names.*attribute|exactly singular", stock$error))
    }
    fit <- if (is.null(stock$error)) stock else do.call(corrected, fit_args)
    prediction <- corrected_predict(fit, new, type = "quantile", p = c(.25, .75), se.fit = TRUE)
    cases[[length(cases) + 1L]] <- list(name = paste(scenario, kind, dist, sep = "/"),
      formula = formula, frame = scenario,
      subset = if (is.null(selected)) NULL else I(selected - 1L), action = action, dist = dist,
      robust = kind == "robust", stock_error = stock$error,
      coefficients = I(unname(coef(fit))), scale = I(unname(fit$scale)), scale_names = I(names(fit$scale)),
      variance = unname(vcov(fit)), variance_names = I(colnames(vcov(fit))), naive = unname(fit$naive.var),
      loglik = I(fit$loglik), df = I(fit$df), df_residual = fit$df.residual, score = I(fit$score),
      omitted = I(as.integer(fit$na.action)), lp = I(unname(predict(fit, type = "lp"))),
      quantile = unname(prediction$fit), quantile_se = unname(prediction$se.fit),
      dfbeta = unname(reference_env$residuals.survreg(fit, type = "dfbeta", weighted = TRUE)),
      var2 = unname(fit$var2))
  }
}
interactions <- list()
for (drop in c("a", "b")) for (dist in c("weibull", "gaussian", "lognormal")) {
  selected <- which(base$g != drop)
  fit <- survreg(Surv(time, status) ~ age * strata(g), base, subset = selected,
                 weights = w, dist = dist, x = TRUE)
  interactions[[length(interactions) + 1L]] <- list(name = paste(drop, dist, sep = "/"),
    subset = I(selected - 1L), dist = dist, coefficients = I(unname(coef(fit))),
    names = I(names(coef(fit))), variance = unname(vcov(fit)), scale = I(unname(fit$scale)),
    x = unname(fit$x), lp = I(unname(predict(fit, type = "lp"))))
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  correction = "After stock errors: retain all factor levels; penalized initial effective sample size uses only observed scale strata for the mean scale and covariance inverse. Predictions and residuals use length(object$scale), including during robust fitting."),
  interactions = interactions, base = base, levels = I(levels(base$g)), frames = frames, newdata = new, cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases) + length(interactions), "AFT unused-strata cases written\n")
