#!/usr/bin/env Rscript
# Independent stock-survival confidence transforms and missing-value edges.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/confidence_band_reference.json"
stock <- get("survfit_confint", asNamespace("survival"))
sources <- list(
    regular = list(p = c(.99, .8, .5, .1, .01), se = c(0, .04, .1, .2, .3)),
    endpoints = list(p = c(1, 0, 1, 0), se = c(0, 0, .1, .1)),
    missing = list(p = c(.8, NA_real_, NaN, .2), se = c(.1, .2, 0, NA_real_)),
    nonfinite = list(p = c(-.1, 1.1, Inf, -Inf, .5), se = c(.1, .1, .1, .1, Inf)),
    scalar = list(p = .5, se = .1),
    empty = list(p = numeric(0), se = numeric(0)))
encode <- function(values) lapply(values, function(value) {
    if (is.nan(value)) return("NaN")
    if (is.na(value)) return("NA")
    if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
    unname(value)
})
cases <- list()
for (source in names(sources)) {
    values <- sources[[source]]
    for (type in c("plain", "log", "log-log", "logit", "arcsin")) {
        for (logse in c(FALSE, TRUE)) for (level in c(.8, .95)) {
            for (lower in c("absent", "equal", "widened")) for (ulimit in c(FALSE, TRUE)) {
                selow <- switch(lower, absent = NULL, equal = values$se,
                    widened = ifelse(seq_along(values$se) %% 2L == 0L, 0, values$se * 1.5))
                arguments <- list(p = values$p, se = values$se, logse = logse,
                    conf.type = type, conf.int = level, ulimit = ulimit)
                if (!is.null(selow)) arguments$selow <- selow
                warnings <- character()
                expected <- withCallingHandlers(do.call(stock, arguments), warning = function(w) {
                    warnings <<- c(warnings, conditionMessage(w))
                    invokeRestart("muffleWarning")
                })
                cases[[length(cases) + 1L]] <- list(
                    name = paste(source, type, logse, level, lower, ulimit, sep = "/"),
                    p = encode(values$p), se = encode(values$se), logse = logse,
                    conf_type = type, conf_int = level, selow = if (is.null(selow)) NULL else encode(selow),
                    ulimit = ulimit, warnings = I(warnings),
                    expected = lapply(expected, encode))
            }
        }
    }
}
write_json(list(r_version = R.version.string,
    survival_version = as.character(packageVersion("survival")),
    generator = "scripts/generate_confidence_band_reference.R", cases = cases),
    output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
cat(length(cases), "confidence-band references written\n")
