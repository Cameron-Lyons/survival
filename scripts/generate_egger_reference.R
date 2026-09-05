#!/usr/bin/env Rscript
# Run from the repository root; Python tests consume the fixture without R.
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
destination <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/egger_r_reference.json"
cases <- list()

egger_reference <- function(effects, std_errors) {
    fit <- lm(I(effects / std_errors) ~ I(1 / std_errors))
    intercept <- coef(summary(fit))[1L, ]
    probability <- 2 * pt(-abs(intercept[[3L]]), df.residual(fit))
    stopifnot(isTRUE(all.equal(probability, intercept[[4L]], tolerance = 1e-14)))
    c(intercept = intercept[[1L]], se = intercept[[2L]],
      t = intercept[[3L]], p = probability)
}

add_case <- function(id, effects, std_errors) {
    reference <- egger_reference(effects, std_errors)
    negative <- egger_reference(-effects, std_errors)
    stopifnot(isTRUE(all.equal(negative, reference * c(-1, 1, -1, 1), tolerance = 1e-10)))
    for (factor in c(1e-8, 1, 1e8)) {
        scaled <- egger_reference(effects * factor, std_errors * factor)
        stopifnot(isTRUE(all.equal(scaled, reference, tolerance = 1e-10)))
    }
    cases[[length(cases) + 1L]] <<- list(
        id = id, effects = effects, std_errors = std_errors,
        df = length(effects) - 2L,
        egger_intercept = unname(reference[[1L]]), egger_se = unname(reference[[2L]]),
        egger_t = unname(reference[[3L]]), egger_p = unname(reference[[4L]])
    )
}

add_case("three_studies_negative_probability", c(1.01, 0.49, 1.01 / 3), c(1, 0.5, 1 / 3))
add_case("three_studies_false_significance", c(1, 0.5, 0.4), c(1, 0.5, 1 / 3))

degrees_freedom <- c(1L, 2L, 3L, 5L, 10L, 20L, 30L, 31L, 100L)
target_statistics <- c(0, 0.75, 1.5, 2.5, -3.5, 0.25, 2, 2, 10)
for (index in seq_along(degrees_freedom)) {
    df <- degrees_freedom[[index]]
    i <- seq_len(df + 2L)
    precision <- 0.7 + i / length(i) + ((i * 7L) %% 11L) * 0.09
    residual <- residuals(lm(I(sin(i * 1.7) + cos(i * 0.4)) ~ precision))
    residual <- residual / sqrt(sum(residual^2) / df)
    se_intercept <- coef(summary(lm(residual ~ precision)))[1L, 2L]
    standardized <- target_statistics[[index]] * se_intercept + 0.35 * precision + residual
    add_case(paste0("constructed_df", df), standardized / precision, 1 / precision)
}

fixture <- list(
    provenance = list(
        generator = "scripts/generate_egger_reference.R",
        r_version = R.version.string, stats_version = as.character(packageVersion("stats")),
        reference = "stats::lm standardized effect on precision; 2*stats::pt(-abs(t), n-2)",
        note = "Constructed residuals are orthogonal to the intercept and precision; df100 has t approximately 10."
    ),
    cases = cases
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
write_json(fixture, destination, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(sprintf("Wrote %d R Egger reference cases to %s\n", length(cases), destination))
