#!/usr/bin/env Rscript
# Run from the repository root; tests consume this committed fixture without R.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/survreg_rank_r_reference.json"
cases <- list()

encode_number <- function(value) {
    if (is.na(value)) NA_real_ else if (is.infinite(value)) as.character(value) else unname(value)
}
encode_rows <- function(value) {
    lapply(seq_len(nrow(value)), function(row) lapply(unname(value[row, ]), encode_number))
}

add_case <- function(id, data, rhs, distribution = "gaussian", scale = 1,
                     initial, maxiter = 0, tolerance = 1e-10, eps = 1e-9) {
    formula <- paste("Surv(time, time2, status, type='interval') ~", rhs)
    weights <- if ("weight" %in% names(data)) data$weight else rep(1, nrow(data))
    fit <- suppressWarnings(survreg(
        as.formula(formula), data = data, dist = distribution, scale = scale,
        init = initial, weights = weights, score = TRUE,
        control = survreg.control(maxiter = maxiter, rel.tolerance = eps,
                                  toler.chol = tolerance)
    ))
    summary_table <- suppressWarnings(summary(fit)$table)
    reduced_variance <- vcov(fit, complete = FALSE)
    training_terms <- tryCatch(predict(fit, type = "terms", se.fit = TRUE),
                               error = function(error) list(error = conditionMessage(error)))
    cases[[length(cases) + 1L]] <<- list(
        id = id, formula = formula, data = as.list(data), distribution = distribution,
        fixed_scale = scale, initial = initial, maxiter = maxiter,
        tolerance = tolerance, eps = eps, weights = weights,
        coefficients = lapply(unname(coef(fit)), encode_number),
        scales = unname(fit$scale), variance = unname(fit$var),
        reduced_variance = if (length(reduced_variance)) unname(reduced_variance) else list(),
        loglik = unname(fit$loglik[[2L]]), score = unname(fit$score),
        linear_predictors = unname(fit$linear.predictors),
        iterations = unname(fit$iter), df = unname(fit$df),
        summary = if (is.null(summary_table)) NULL else encode_rows(summary_table),
        confidence = encode_rows(confint(fit)),
        training_prediction = unname(predict(fit, type = "lp")),
        newdata_prediction = lapply(unname(predict(fit, newdata = data, type = "lp")), encode_number),
        terms = if (is.null(training_terms$fit)) NULL else encode_rows(training_terms$fit),
        terms_se = if (is.null(training_terms$se.fit)) NULL else encode_rows(training_terms$se.fit),
        terms_error = training_terms$error,
        response_residuals = unname(residuals(fit, type = "response")),
        dfbeta = encode_rows(as.matrix(residuals(fit, type = "dfbeta"))),
        dfbetas = encode_rows(as.matrix(residuals(fit, type = "dfbetas")))
    )
}

base <- data.frame(time = 1:6, time2 = 1:6, status = rep(1, 6),
                   x = -2:3, z = -2:3, zero = rep(0, 6))
for (iterations in c(0, 1, 20)) {
    add_case(paste0("duplicate_fixed_iter", iterations), base, "x + z",
             initial = c(0, 0, 0), maxiter = iterations)
    add_case(paste0("duplicate_nonzero_alias_iter", iterations), base, "x + z",
             initial = c(0, 0, 2), maxiter = iterations)
}
add_case("duplicate_reversed_order", base, "z + x", initial = c(0, 0, 2), maxiter = 20)
add_case("duplicate_no_intercept", base, "x + z - 1", initial = c(0, 2), maxiter = 20)
add_case("zero_column_prescribed", base, "x + zero", initial = c(2, 0.2, 7))
add_case("zero_column_fitted", base, "x + zero", initial = c(2, 0.2, 7), maxiter = 20)
add_case("zero_column_first", base, "zero + x - 1", initial = c(7, 0.2), maxiter = 20)

near <- base
near$z <- near$x + rep(c(0.001, -0.001), 3)
add_case("near_dependent_retained", near, "x + z", initial = c(0, 0, 0), tolerance = 1e-12)
add_case("near_dependent_aliased", near, "x + z", initial = c(0, 0, 0), tolerance = 1e-5)
small <- data.frame(time = 1:3, time2 = 1:3, status = rep(1, 3))
add_case("absolute_tolerance_retained", small, "1", scale = 1e6,
         initial = 2, tolerance = 1e-14)
add_case("absolute_tolerance_zero_rank", small, "1", scale = 1e6,
         initial = 2, tolerance = 1e-10)

mixed <- data.frame(
    time = c(1.3, 2.1, 1.8, 3.4, 2.9, 4.5, 3.8, 5.2),
    status = rep(1, 8), x = c(-1.2, -0.5, 0.3, 1.1, -0.8, 0.6, 1.4, 0.2),
    group = rep(c("a", "b"), 4),
    weight = c(1, 0.75, 1.5, 2, 0.5, 1.25, 1, 1.75),
    off = c(0.1, -0.1, 0, 0.2, -0.2, 0.1, 0.05, -0.05)
)
mixed$time2 <- mixed$time
mixed$z <- mixed$x
for (iterations in c(0, 100)) {
    suffix <- if (iterations == 0) "prescribed" else "fitted"
    add_case(paste0("duplicate_weighted_fixed_", suffix), mixed, "x + z + offset(off)",
             scale = 1.3, initial = c(2, 0.2, 0.3), maxiter = iterations)
    add_case(paste0("duplicate_estimated_scale_", suffix), mixed, "x + z + offset(off)",
             scale = 0, initial = c(2, 0.2, 0.3, log(1.2)), maxiter = iterations)
    add_case(paste0("duplicate_stratified_scale_", suffix), mixed,
             "x + z + strata(group) + offset(off)", scale = 0,
             initial = c(2, 0.2, 0.3, log(1.2), log(0.9)), maxiter = iterations)
}

censored <- mixed
censored$status <- c(1, 0, 2, 3, 1, 0, 3, 1)
censored$time2[censored$status == 3] <- censored$time[censored$status == 3] + 0.7
for (distribution in c("logistic", "weibull", "loglogistic")) {
    initial_location <- if (distribution == "logistic") 2 else 0.8
    for (iterations in c(0, 100)) {
        add_case(paste(distribution, "censored", iterations, sep = "_"), censored,
                 "x + z + offset(off)", distribution = distribution,
                 scale = 1.2, initial = c(initial_location, 0.1, 0.2), maxiter = iterations)
    }
}

full_rank <- mixed
full_rank$z <- c(0.3, -0.8, 1.2, -0.3, 0.6, 1.1, -0.2, 0.5)
for (iterations in c(0, 100)) {
    add_case(paste0("full_rank_fixed_", iterations), full_rank, "x + z + offset(off)",
             scale = 1.3, initial = c(2.5, 0.2, -0.1), maxiter = iterations)
    add_case(paste0("full_rank_estimated_", iterations), full_rank, "x + z + offset(off)",
             scale = 0, initial = c(2.5, 0.2, -0.1, log(1.2)), maxiter = iterations)
}

set.seed(2048)
n <- 90L
g <- factor(rep(c("a", "b", "c"), length.out = n), levels = c("a", "b", "c"))
h <- factor(rep(c("u", "v"), length.out = n), levels = c("u", "v"))
x <- seq(-1.7, 1.9, length.out = n) +
    rep(c(-0.13, 0.07, 0.19, -0.05, 0.11), length.out = n)
eta <- 0.25 * x + c(a = -0.2, b = 0.15, c = 0.35)[g] + c(u = -0.1, v = 0.1)[h]
event_time <- rexp(n, rate = exp(eta) / 9)
censor_time <- rexp(n, rate = 1 / 14)
interaction <- data.frame(
    time = pmax(pmin(event_time, censor_time), 0.01),
    status = as.integer(event_time <= censor_time), x = x, g = g
)
interaction$time2 <- interaction$time
add_case("weibull_singular_interaction_convergence", interaction, "x + g:x - 1",
         distribution = "weibull", scale = 0, initial = NULL, maxiter = 150, eps = 1e-10)

fixture <- list(
    provenance = list(
        generator = "scripts/generate_survreg_rank_reference.R",
        r_version = R.version.string,
        survival_version = as.character(packageVersion("survival")),
        reference = "survival::survreg, vcov, predict, residuals, summary, confint",
        note = "NA coefficients are JSON null; infinite summary statistics are strings."
    ),
    cases = cases
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null")
cat(sprintf("Wrote %d R rank/covariance cases to %s\n", length(cases), destination))
