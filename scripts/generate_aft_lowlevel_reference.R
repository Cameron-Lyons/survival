#!/usr/bin/env Rscript
# Raw survreg.fit outputs; responses are already on the fitting scale.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/aft_lowlevel_reference.json"
cases <- list()
time <- c(1.2, 2.5, .9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9)
status <- c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)
x <- cbind(Intercept = 1, age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
           group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0))
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
named <- function(x) if (is.null(x)) NULL else I(x)
add <- function(name, dist = "extreme", design = x, y = cbind(time, status),
                weights = NULL, offset = NULL, init = NULL, controlvals = list(),
                scale = 0, nstrat = 1, strata = NULL,
                parms = if (dist == "t") 4 else NULL, distribution = dist) {
    warnings <- character()
    result <- withCallingHandlers(tryCatch(survreg.fit(design, y, weights, offset, init,
        do.call(survreg.control, controlvals), distribution, scale, nstrat, strata, parms),
        error = function(e) list(error = conditionMessage(e))),
        warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
    expected <- if (!is.null(result$error)) result else list(
        coefficients = I(unname(result$coefficients)), icoef = I(unname(result$icoef)),
        var = rows(result$var), loglik = I(result$loglik), iter = result$iter,
        linear_predictors = I(result$linear.predictors), df = result$df, score = I(result$score),
        coefficient_names = named(names(result$coefficients)), icoef_names = named(names(result$icoef)),
        variance_names = named(rownames(result$var)))
    cases[[length(cases) + 1L]] <<- list(name = name, x = rows(design), y = rows(y),
        column_names = named(colnames(design)),
        arguments = list(dist = dist, weights = named(weights), offset = named(offset),
            init = named(init), controlvals = controlvals, scale = scale, nstrat = nstrat,
            strata = named(strata), parms = named(parms)), expected = expected, warnings = I(warnings))
}
for (dist in c("extreme", "logistic", "gaussian", "t")) {
    add(paste0(dist, "_default"), dist)
    add(paste0(dist, "_weighted_offset"), dist, weights = rep(c(.5, 1, 2), 4), offset = seq(.1, 1.2, length.out = 12))
    add(paste0(dist, "_fixed"), dist, scale = 1.5)
    add(paste0(dist, "_strata"), dist, nstrat = 2, strata = rep(c(1, 2), 6))
    add(paste0(dist, "_initial"), dist, init = c(2, 0, 0, .1))
    add(paste0(dist, "_partial_initial"), dist, init = c(2, 0, 0))
    add(paste0(dist, "_zero_iterations"), dist, controlvals = list(iter.max = 0))
    add(paste0(dist, "_one_iteration"), dist, controlvals = list(iter.max = 1))
    add(paste0(dist, "_two_iterations"), dist, controlvals = list(iter.max = 2))
    add(paste0(dist, "_mean_only"), dist, design = x[, 1, drop = FALSE])
    add(paste0(dist, "_mean_fixed"), dist, design = x[, 1, drop = FALSE], scale = 1)
    add(paste0(dist, "_no_intercept"), dist, design = x[, 2:3])
    add(paste0(dist, "_alias"), dist, design = cbind(x, duplicate = x[, 2]))
    add(paste0(dist, "_binary"), dist, design = x[, c(1, 3)])
    interval_status <- rep(c(1, 0, 2, 3), 3)
    add(paste0(dist, "_interval"), dist, y = cbind(time, time + .4, interval_status))
}
add("unnamed", design = unname(x))
add("t_parameters", "t", parms = 8)
add("t_missing_parameters", "t", parms = NULL)
add("gaussian_ignored_parameters", "gaussian", parms = 8)
add("unused_scale_stratum", nstrat = 3, strata = rep(c(1, 2), 6))
add("one_stratum_ignores_codes", strata = c(NA, rep(-5, 11)))
add("bad_init", init = c(1, 2))
add("bad_weights", weights = c(0, rep(1, 11)))
add("bad_scale", scale = -1)
add("fixed_strata", scale = 1, nstrat = 2, strata = rep(c(1, 2), 6))
add("missing_strata", nstrat = 2)
add("unknown_distribution", dist = "unknown")
add("derived_distribution", dist = "weibull")
custom <- survreg.distributions$gaussian
custom$name <- "Custom Gaussian"
custom$deviance <- custom$quantile <- NULL
custom$trans <- function(x) stop("bare fitter must not transform")
custom$scale <- 100
add("custom_minimal", dist = "custom", distribution = custom)
custom$init <- NULL
add("custom_explicit_mean_initial", dist = "custom_no_init", distribution = custom,
    design = x[, 1, drop = FALSE], init = c(2, .1))
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
    cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
