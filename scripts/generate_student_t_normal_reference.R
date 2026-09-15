#!/usr/bin/env Rscript
# Rscript scripts/generate_student_t_normal_reference.R [output.json]
# Direct negative tails exercise approximation boundaries without 1-CDF
# cancellation. R is only required for regeneration, not for Python tests.
suppressPackageStartupMessages(library(jsonlite))
stopifnot(getRversion() == "4.6.1")
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/student_t_normal_reference.json"

degrees <- c(999.999, 1000, 1000.001, 2000, 3000,
             9999.999, 10000, 10000.001, 30000, 99999.999,
             100000, 100000.001, 1e6, 1e8)
magnitude <- sort(as.vector(outer(c(1, 2, 4, 8.9, 9), c(-1e-6, 0, 1e-6), `+`)))
probability <- c(.001, .025, .1, .25, .75, .975, .999)
cases <- lapply(degrees, function(df) {
    list(df = df, x = -rev(magnitude),
         cdf = pt(-rev(magnitude), df),
         log_cdf = pt(-rev(magnitude), df, log.p = TRUE),
         p = probability, quantile = qt(probability, df))
})
reference <- list(
    metadata = list(
        generator = "scripts/generate_student_t_normal_reference.R",
        r_version = R.version.string,
        stats_version = as.character(packageVersion("stats")),
        methods = list(cdf = "stats::pt(x, df)",
                       log_cdf = "stats::pt(x, df, log.p=TRUE)",
                       quantile = "stats::qt(p, df)"),
        scope = paste("Local regression grid around magnitude 1, 2, 4, 8.9, 9",
                      "and df 1e3/1e4/1e5 approximation boundaries, with ordinary",
                      "quantile checks; this is not an exhaustive accuracy bound.")),
    cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE)
cat(length(cases), "Student-t normal-approximation cases written to", output, "\n")
