#!/usr/bin/env Rscript
# Run from the repository root; Python consumes the committed fixture without R.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/survreg_initial_r_reference.json"
i <- 0:15
mixed <- data.frame(
    time = 1.1 + ((i * 7) %% 17) * 0.22 + i * 0.03,
    status = rep(c(1, 0, 2, 3), 4),
    x = ((i * 5) %% 13 - 6) * 0.19,
    z = ((i * 3) %% 11 - 5) * 0.17,
    weight = 0.6 + (i %% 5) * 0.27,
    off = ((i %% 4) - 1.5) * 0.09,
    group = rep(c("a", "b"), 8)
)
mixed$time2 <- mixed$time + 0.15 + (i %% 3) * 0.2
mixed$u <- mixed$x + 2
mixed$zero <- 0
mixed$one <- 1
mixed$binary <- i %% 2
events <- mixed
events$status <- 1
events$time2 <- events$time
covariate_units <- events
covariate_units$x <- events$x * 1e8 + 1e10
covariate_units$z <- events$z * 1e-4 - 0.02
response_units <- events
response_units$time <- events$time * 7 + 10
response_units$time2 <- response_units$time
response_units$off <- events$off * 7
duplicate <- mixed
duplicate$z <- duplicate$x
datasets <- list(mixed = mixed, events = events,
                 covariate_units = covariate_units, response_units = response_units,
                 duplicate = duplicate)
cases <- list()

add_case <- function(id, distribution, mode = "fixed", rhs = "x + z + offset(off)",
                     dataset = "mixed", maxiter = 0, initial = NULL, fixed_scale = 1.1) {
    data <- datasets[[dataset]]
    if (mode == "stratified") rhs <- paste(rhs, "+ strata(group)")
    formula <- paste("Surv(time, time2, status, type='interval') ~", rhs)
    scale <- if (mode == "fixed") fixed_scale else 0
    if (distribution == "exponential") scale <- 1
    if (distribution == "rayleigh") scale <- 0.5
    arguments <- list(
        formula = as.formula(formula), data = data, weights = data$weight,
        dist = distribution, scale = scale, score = TRUE, x = TRUE,
        control = survreg.control(maxiter = maxiter, rel.tolerance = 1e-10, toler.chol = 1e-10)
    )
    if (distribution %in% c("exponential", "rayleigh")) arguments$scale <- NULL
    if (!is.null(initial)) arguments$init <- initial
    if (distribution == "t") arguments$parms <- 5
    warnings <- character()
    fit <- withCallingHandlers(do.call(survreg, arguments), warning = function(condition) {
        warnings <<- c(warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
    })
    design <- fit$x
    center <- rep(0, ncol(design))
    stdev <- rep(1, ncol(design))
    if (is.null(initial) && ncol(design) > 1 && all(design[, 1] == 1)) {
        binary <- apply(design, 2, function(column) all(column == 0 | column == 1))
        center <- ifelse(binary, 0, colMeans(design))
        stdev <- ifelse(binary, 1, apply(design, 2, sd))
    }
    # R leaves the score in its internally rescaled coordinates, unlike coef/V.
    score_original <- fit$score
    for (column in seq_len(ncol(design))) {
        score_original[column] <- stdev[column] * fit$score[column] + center[column] * fit$score[1]
    }
    cases[[length(cases) + 1L]] <<- list(
        id = id, dataset = dataset, formula = formula, distribution = distribution,
        scale = scale, initial = initial, maxiter = maxiter, eps = 1e-10, tolerance = 1e-10,
        coefficients = unname(coef(fit)), scales = unname(fit$scale),
        variance = unname(fit$var), loglik = unname(fit$loglik[[2L]]),
        score = unname(fit$score), score_original = unname(score_original),
        design_center = unname(center), design_scale = unname(stdev),
        linear_predictors = unname(fit$linear.predictors), iterations = unname(fit$iter),
        null_coefficients = unname(fit$icoef), warnings = warnings
    )
}

kernels <- c("extreme", "logistic", "gaussian", "weibull", "lognormal", "loglogistic", "t")
for (distribution in kernels) {
    for (mode in c("fixed", "estimated", "stratified")) {
        for (iterations in c(0, 200)) {
            add_case(paste(distribution, mode, iterations, sep = "_"), distribution,
                     mode = mode, maxiter = iterations)
        }
    }
}
for (distribution in c("exponential", "rayleigh")) {
    for (iterations in c(0, 200)) {
        add_case(paste(distribution, "fixed", iterations, sep = "_"), distribution,
                 maxiter = iterations)
    }
}
for (distribution in c("extreme", "gaussian", "t")) {
    for (iterations in c(0, 200)) {
        add_case(paste(distribution, "intercept_only", iterations, sep = "_"), distribution,
                 mode = "estimated", rhs = "1 + offset(off)", maxiter = iterations)
        add_case(paste(distribution, "no_intercept", iterations, sep = "_"), distribution,
                 rhs = "u + z - 1 + offset(off)", maxiter = iterations,
                 dataset = if (distribution == "gaussian") "events" else "mixed")
    }
}
for (distribution in kernels) {
    add_case(paste0(distribution, "_mean_only_stratified_0"), distribution,
             mode = "stratified", rhs = "1 + offset(off)")
}
for (distribution in c("gaussian", "weibull")) {
    for (mode in c("fixed", "estimated")) {
        for (iterations in c(0, 200)) {
            add_case(paste(distribution, "duplicate", mode, iterations, sep = "_"),
                     distribution, mode = mode, dataset = "duplicate", maxiter = iterations)
        }
    }
}
for (mode in c("fixed", "estimated")) {
    for (iterations in c(0, 200)) {
        add_case(paste("gaussian_binary_alias", mode, iterations, sep = "_"), "gaussian",
                 mode = mode, dataset = "events", rhs = "x + zero + one + binary + offset(off)",
                 maxiter = iterations)
    }
}
for (distribution in c(kernels, "exponential", "rayleigh")) {
    fixed <- distribution %in% c("exponential", "rayleigh")
    initial <- c(1.2, 0.1, -0.2, if (!fixed) log(1.3))
    add_case(paste0(distribution, "_explicit_full"), distribution,
             mode = if (fixed) "fixed" else "estimated", dataset = "events", initial = initial)
}
for (iterations in c(0, 200)) {
    add_case(paste0("gaussian_wls_", iterations), "gaussian", dataset = "events", maxiter = iterations)
    add_case(paste0("gaussian_wls_no_intercept_", iterations), "gaussian", dataset = "events",
             rhs = "x + z - 1 + offset(off)", maxiter = iterations)
    add_case(paste0("gaussian_covariate_units_", iterations), "gaussian",
             dataset = "covariate_units", maxiter = iterations)
    add_case(paste0("gaussian_response_units_", iterations), "gaussian",
             dataset = "response_units", fixed_scale = 7 * 1.1, maxiter = iterations)
}

wls <- lm(I(time - off) ~ x + z, data = events, weights = weight)
wls_no_intercept <- lm(I(time - off) ~ x + z - 1, data = events, weights = weight)
fixture <- list(
    provenance = list(
        generator = "scripts/generate_survreg_initial_reference.R",
        r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
        note = paste("Omitted init uses R distribution starts, intercept-only scale fits, GLIM, and design rescaling.",
                     "R's returned score and its conversion to original design coordinates are both retained.")
    ),
    datasets = lapply(datasets, as.list),
    gaussian_wls = list(coefficients = unname(coef(wls)),
                        no_intercept_coefficients = unname(coef(wls_no_intercept))),
    cases = cases
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null")
cat(sprintf("Wrote %d R initialization cases to %s\n", length(cases), destination))
