#!/usr/bin/env Rscript
# Run from the repository root; Python and Rust consume the fixture without R.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/survreg_partial_init_r_reference.json"
i <- 0:15
data <- data.frame(
    time = 1.1 + ((i * 7) %% 17) * 0.22 + i * 0.03,
    status = rep(c(1, 0, 2, 3), 4),
    x = ((i * 5) %% 13 - 6) * 0.19,
    z = ((i * 3) %% 11 - 5) * 0.17,
    weight = 0.6 + (i %% 5) * 0.27,
    off = ((i %% 4) - 1.5) * 0.09,
    group = rep(c("a", "b"), 8)
)
data$time2 <- data$time + 0.15 + (i %% 3) * 0.2
data$u <- data$x + 2

# survreg.fit drops the native convergence/rank flag from its public return.
# Observe its local native result on exit without changing the fitted model.
suppressMessages(invisible(trace("survreg.fit", where = asNamespace("survival"),
    print = FALSE, exit = quote({
        if (exists("fit", inherits = FALSE))
            base::assign("native_result", fit, envir = .GlobalEnv)
    }))))
cases <- list()
errors <- list()

run_fit <- function(arguments) {
    native_result <<- NULL
    warnings <- character()
    fit <- withCallingHandlers(do.call(survreg, arguments), warning = function(condition) {
        warnings <<- c(warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
    })
    list(fit = fit, native_flag = native_result$flag, warnings = warnings)
}

add_case <- function(distribution, design = "covariates", mode = "estimated",
                     maxiter = 0, partial = TRUE) {
    rhs <- switch(design,
        covariates = "x + z + offset(off)",
        no_intercept = "u + z - 1 + offset(off)",
        one_column = "u - 1 + offset(off)",
        mean_only = "1 + offset(off)")
    nscale <- if (mode == "fixed") 0L else if (mode == "stratified") 2L else 1L
    initial <- switch(design, covariates = c(1.2, 0.1, -0.2),
                      no_intercept = c(0.6, -0.2), one_column = 0.6, mean_only = 1.2)
    if (!partial && nscale > 0) initial <- c(initial, log(c(0.9, 1.4)[seq_len(nscale)]))
    if (mode == "stratified") rhs <- paste(rhs, "+ strata(group)")
    formula <- paste("Surv(time, time2, status, type='interval') ~", rhs)
    arguments <- list(
        formula = as.formula(formula), data = data, weights = data$weight,
        dist = distribution, init = initial, score = TRUE, x = TRUE,
        control = survreg.control(maxiter = maxiter, rel.tolerance = 1e-10, toler.chol = 1e-10)
    )
    if (distribution == "t") arguments$parms <- 5
    result <- run_fit(arguments)
    fit <- result$fit
    initial_log_scales <- if (nscale == 0) numeric() else if (partial)
        unname(fit$icoef[-1L]) else tail(initial, nscale)
    full_initial <- if (partial) c(initial, initial_log_scales) else initial
    # R's partial route must be exactly the same fit as supplying these scales.
    if (partial && nscale > 0) {
        full_arguments <- arguments
        full_arguments$init <- full_initial
        full <- run_fit(full_arguments)
        for (name in c("coefficients", "scale", "var", "score", "loglik", "linear.predictors", "iter"))
            stopifnot(identical(fit[[name]], full$fit[[name]]))
        stopifnot(identical(result$native_flag, full$native_flag))
    }
    cases[[length(cases) + 1L]] <<- list(
        id = paste(distribution, design, mode, if (partial) "location" else "full", maxiter, sep = "_"),
        formula = formula, distribution = distribution, design_kind = design,
        mode = mode, partial = partial && nscale > 0,
        design = unname(fit$x), strata = if (nscale == 2L) as.integer(factor(data$group)) else NULL,
        scale = if (nscale > 0) 0 else unname(fit$scale), initial = initial,
        initial_scales = exp(initial_log_scales), full_initial = full_initial,
        maxiter = maxiter, eps = 1e-10, tolerance = 1e-10,
        coefficients = unname(coef(fit)), scales = unname(fit$scale),
        variance = unname(fit$var), loglik = unname(fit$loglik[[2L]]),
        score = unname(fit$score), linear_predictors = unname(fit$linear.predictors),
        iterations = unname(fit$iter), native_flag = unname(result$native_flag),
        warnings = result$warnings
    )
}

kernels <- c("extreme", "logistic", "gaussian", "weibull", "lognormal", "loglogistic", "t")
for (distribution in kernels) {
    for (mode in c("estimated", "stratified")) {
        for (design in c("covariates", "no_intercept")) {
            for (maxiter in c(0, 200)) add_case(distribution, design, mode, maxiter)
        }
        add_case(distribution, "mean_only", mode, partial = FALSE)
        # R 3.8-11's mean-only partial-start branch refers to an absent fit0.
        rhs <- if (mode == "stratified") "1 + offset(off) + strata(group)" else "1 + offset(off)"
        arguments <- list(formula = as.formula(paste(
            "Surv(time, time2, status, type='interval') ~", rhs)),
            data = data, weights = data$weight, dist = distribution, init = 1.2,
            control = survreg.control(maxiter = 0))
        if (distribution == "t") arguments$parms <- 5
        message <- tryCatch({ run_fit(arguments); stop("expected R mean-only error") },
                            error = function(condition) conditionMessage(condition))
        stopifnot(identical(message, "object 'fit0' not found"))
        errors[[length(errors) + 1L]] <- list(distribution = distribution, mode = mode, message = message)
    }
}
for (distribution in c("gaussian", "weibull", "t")) {
    for (mode in c("estimated", "stratified")) {
        for (maxiter in c(0, 200)) add_case(distribution, "one_column", mode, maxiter)
    }
}
for (distribution in c("exponential", "rayleigh")) {
    for (maxiter in c(0, 200)) add_case(distribution, "covariates", "fixed", maxiter)
}
suppressMessages(invisible(untrace("survreg.fit", where = asNamespace("survival"))))
fixture <- list(
    provenance = list(
        generator = "scripts/generate_survreg_partial_init_reference.R",
        r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
        note = paste("Location-only numeric init retains location coefficients and appends scales from R's",
                     "20-iteration intercept-only fit using weights, offsets, censoring, and scale strata.",
                     "Numeric init bypasses covariate rescaling. Every accepted partial reference is checked",
                     "against its expanded full init. Native flags are observed without modifying R fits.",
                     "Mean-only partial starts fail in R 3.8-11; full-start controls remain valid.")
    ),
    data = as.list(data), cases = cases, mean_only_partial_errors = errors
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null")
cat(sprintf("Wrote %d fitted cases and %d R mean-only errors to %s\n", length(cases), length(errors), destination))
