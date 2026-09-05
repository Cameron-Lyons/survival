#!/usr/bin/env Rscript
# Run from the repository root; the small committed fixture needs no R at test time.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/survreg_line_search_r_reference.json"
i <- 0:15
data <- data.frame(
    time = 1.1 + ((i * 7) %% 17) * 0.22 + i * 0.03,
    status = rep(c(1, 0, 2, 3), 4),
    x = ((i * 5) %% 13 - 6) * 0.19,
    z = ((i * 3) %% 11 - 5) * 0.17,
    weight = 0.6 + (i %% 5) * 0.27,
    off = ((i %% 4) - 1.5) * 0.09
)
data$time2 <- data$time + 0.15 + (i %% 3) * 0.2
formula <- "Surv(time, time2, status, type='interval') ~ x + z + offset(off)"
specifications <- list(
    list(distribution = "extreme", initial = c(6, -1, 0.7)),
    list(distribution = "logistic", initial = c(-2.5, 1.2, -0.8)),
    list(distribution = "weibull", initial = c(6, -1, 0.7)),
    list(distribution = "loglogistic", initial = c(6, -1, 0.7)),
    list(distribution = "t", initial = c(1, 0.2, -0.1))
)
cases <- lapply(specifications, function(specification) {
    arguments <- list(
        formula = as.formula(formula), data = data, weights = data$weight,
        dist = specification$distribution, scale = 0.9, init = specification$initial,
        score = TRUE,
        control = survreg.control(maxiter = 150, rel.tolerance = 1e-12, toler.chol = 1e-10)
    )
    if (specification$distribution == "t") arguments$parms <- 5
    fit <- do.call(survreg, arguments)
    list(
        distribution = specification$distribution, initial = specification$initial,
        coefficients = unname(coef(fit)), variance = unname(fit$var),
        score = unname(fit$score), loglik = unname(fit$loglik[[2L]]),
        linear_predictors = unname(fit$linear.predictors), scales = unname(fit$scale)
    )
})
fixture <- list(
    provenance = list(
        generator = "scripts/generate_survreg_line_search_reference.R",
        r_version = R.version.string,
        survival_version = as.character(packageVersion("survival")),
        note = paste("Converged states from starts that reject the full native Newton step.",
                     "Iteration counts are deliberately omitted: R and Rust backtrack differently.")
    ),
    formula = formula, data = as.list(data), cases = cases
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(sprintf("Wrote %d R line-search reference cases to %s\n", length(cases), destination))
