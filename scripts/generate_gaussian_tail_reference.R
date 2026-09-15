#!/usr/bin/env Rscript
# Run from the repository root. Tests consume the committed JSON without R.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/gaussian_tail_r_reference.json"

z <- c(-38.467405617144344, -38.4, -38.25, -38, -37.5, -30, -20, -12, -10, -8, -6, -3, -1,
       -1e-8, 0, 1e-8, 1, 3, 6, 8, 10, 20, 38)
probabilities <- c(.Machine$double.xmin * .Machine$double.eps,
                   1e-300, 1e-100, 1e-30, 1e-12, 0.001,
                   0.024249999999999997, 0.02425, 0.024250000000000004, 0.025,
                   0.1, 0.25, 0.4, 0.5 - 2^-54, 0.5, 0.5 + 2^-53,
                   0.6, 0.75, 0.9, 0.975,
                   1 - 1e-12, 1 - 2^-53)

distributions <- list()
quantiles <- list()
predictions <- list()
fits <- list()
for (distribution in c("gaussian", "lognormal")) {
    for (profile in c("standard", "location_scale")) {
        location <- if (profile == "standard") 0 else 0.4
        scale <- if (profile == "standard") 1 else 1.25
        x <- location + scale * z
        if (distribution == "lognormal") x <- exp(x)
        distributions[[length(distributions) + 1L]] <- list(
            id = paste(distribution, profile, sep = "_"),
            distribution = distribution, mean = location, scale = scale,
            x = x, z = z,
            cdf = psurvreg(x, location, scale, distribution),
            density = dsurvreg(x, location, scale, distribution)
        )
        quantiles[[length(quantiles) + 1L]] <- list(
            id = paste(distribution, profile, sep = "_"),
            distribution = distribution, mean = location, scale = scale,
            probabilities = probabilities,
            values = qsurvreg(probabilities, location, scale, distribution)
        )
    }

    prediction_data <- data.frame(x = c(-1, 0, 2), offset = c(0.2, 0, -0.1))
    prediction_location <- 0.3 - 0.4 * prediction_data$x + prediction_data$offset
    prediction_values <- t(vapply(prediction_location, function(location) {
        qsurvreg(probabilities, location, 0.7, distribution)
    }, numeric(length(probabilities))))
    predictions[[length(predictions) + 1L]] <- list(
        id = distribution, distribution = distribution,
        covariates = unname(cbind(1, prediction_data$x)),
        coefficients = c(0.3, -0.4), scale = 0.7,
        offsets = prediction_data$offset, probabilities = probabilities,
        predictions = prediction_values
    )

    tail_cases <- list(
        exact_positive = c(20, 20, 1),
        exact_negative = c(-20, -20, 1),
        right_6 = c(6, 6, 0), right_10 = c(10, 10, 0),
        right_20 = c(20, 20, 0),
        left_6 = c(-6, -6, 2), left_10 = c(-10, -10, 2),
        left_20 = c(-20, -20, 2),
        interval_6 = c(6, 6.25, 3), interval_10 = c(10, 10.25, 3),
        interval_20 = c(20, 20.25, 3),
        interval_negative_6 = c(-6.25, -6, 3),
        interval_negative_10 = c(-10.25, -10, 3),
        interval_negative_20 = c(-20.25, -20, 3)
    )
    for (scale_mode in c("fixed", "estimated")) {
        # Positive Gaussian observations retain the prescribed standardized tails
        # without depending on the separate signed-time validation contract.
        location <- if (distribution == "gaussian") 40 else
            if (scale_mode == "fixed") 0 else 0.4
        scale <- if (scale_mode == "fixed") 1 else 1.25
        offsets <- if (scale_mode == "fixed") rep(0, 4) else c(0, 0.1, -0.2, 0.3)
        weights <- if (scale_mode == "fixed") rep(1, 4) else c(1, 0.75, 1.5, 0.8)
        for (case_name in names(tail_cases)) {
            tail <- tail_cases[[case_name]]
            lower <- location + offsets + scale * c(-1, 0, 1, tail[[1L]])
            upper <- location + offsets + scale * c(-1, 0, 1, tail[[2L]])
            if (distribution == "lognormal") {
                lower <- exp(lower)
                upper <- exp(upper)
            }
            data <- data.frame(lower, upper, status = c(1, 1, 1, tail[[3L]]),
                               offsets, weights)
            initial <- if (scale_mode == "fixed") location else c(location, log(scale))
            model <- survreg(
                Surv(lower, upper, status, type = "interval") ~ 1 + offset(offsets),
                data = data, dist = distribution, weights = weights,
                init = initial, scale = if (scale_mode == "fixed") scale else 0,
                score = TRUE, control = survreg.control(maxiter = 0)
            )
            fits[[length(fits) + 1L]] <- list(
                id = paste(distribution, scale_mode, case_name, sep = "_"),
                distribution = distribution, time = lower, time2 = upper,
                status = data$status, offsets = offsets, weights = weights,
                initial = initial, fixed_scale = if (scale_mode == "fixed") scale else 0,
                coefficients = unname(coef(model)), scale = unname(model$scale),
                loglik = unname(model$loglik[[2L]]), score = unname(model$score),
                variance = unname(model$var)
            )
        }
    }
}

confidence_level <- 1 - 2^-53
critical <- -qnorm((1 - confidence_level) / 2)
summary_fit <- survreg(Surv(c(8, 9, 10), c(1, 1, 1)) ~ 1,
                      dist = "gaussian", init = 9, scale = 1,
                      control = survreg.control(maxiter = 0))
summary_row <- unname(summary(summary_fit)$table[1L, ])
cox_data <- data.frame(time = 1:8, status = c(1, 1, 0, 1, 1, 0, 1, 1),
                       x = c(0.2, 1.1, 0.3, -0.2, 0.7, 1.3, -0.5, 0.9))
cox_fit <- coxph(Surv(time, status) ~ x, data = cox_data,
                 init = 0, control = coxph.control(iter.max = 0))
cox_curve <- survfit(cox_fit, newdata = data.frame(x = 0.4), conf.type = "none")
confidence <- list(
    level = confidence_level, critical = critical,
    # Compute the critical value from R's lower tail: forming the upper-tail
    # probability as (1 + level) / 2 first would round it to exactly one.
    reference = "R qnorm((1-level)/2), with symmetric normal intervals",
    summary = summary_row,
    coefficient_bounds = 9 + c(-1, 1) * critical * summary_row[[2L]],
    probability = c(0.2, 0.5, 0.8), standard_error = c(0.01, 0.02, 0.01),
    probability_lower = c(0.2, 0.5, 0.8) - critical * c(0.01, 0.02, 0.01),
    probability_upper = c(0.2, 0.5, 0.8) + critical * c(0.01, 0.02, 0.01),
    cox = list(
        data = as.list(cox_data), newdata = list(x = 0.4),
        time = cox_curve$time, survival = cox_curve$surv,
        lower = cox_curve$surv * exp(-critical * cox_curve$std.err),
        upper = pmin(1, cox_curve$surv * exp(critical * cox_curve$std.err))
    )
)

fixture <- list(
    provenance = list(
        generator = "scripts/generate_gaussian_tail_reference.R",
        r_version = R.version.string,
        survival_version = as.character(packageVersion("survival")),
        reference = "survival::psurvreg, dsurvreg, qsurvreg, and survreg(maxiter=0)",
        derivative_reference = "survreg score and inverse observed information at explicit initial parameters"
    ),
    distributions = distributions, quantiles = quantiles,
    predictions = predictions, fits = fits, confidence = confidence
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(sprintf("Wrote %d probability profiles, %d quantile profiles, and %d prescribed fits to %s\n",
            length(distributions), length(quantiles), length(fits), destination))
