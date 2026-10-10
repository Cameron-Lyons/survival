#!/usr/bin/env Rscript
# Rscript scripts/generate_rsurvreg_count_reference.R [output.json]
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/rsurvreg_count_reference.json"

# runif() decides the number of draws before rsurvreg() calls qsurvreg().
# Values in a vector of length other than one do not participate in coercion.
counts <- list(
    integer0 = 0L, integer1 = 1L, integer3 = 3L, numeric3 = 3,
    fractional = 2.5, fractional_small = .75, negative_fraction_small = -.75,
    negative_integer = -1L, negative_fraction = -1.5,
    length2 = c(4, 7), ignored_invalid = c(-1, NA_real_, Inf),
    empty = numeric(), null = NULL, na_integer = NA_integer_, na_real = NA_real_,
    nan = NaN, inf = Inf, negative_inf = -Inf,
    true = TRUE, false = FALSE, bool_vector = c(TRUE, FALSE, NA),
    string3 = "3", stringfraction = "2.5", stringhex = "0x1.8p1",
    stringwhitespace = " +2.5 ", bad_string = "bad", bad_string_separator = "1_0",
    strings = c("bad", "also_bad"), list1 = list(3), list2 = list(-1, NA), list0 = list(),
    matrix = matrix(1:4, 2), matrix1 = matrix(2.5), matrix0 = matrix(numeric(), 0, 3),
    factor = factor("3", levels = c("3", "5")),
    factor_missing = factor(NA, levels = c("3", "5")),
    factor_vector = factor(c("5", "3", NA), levels = c("3", "5")),
    complex = 3 + 2i, complex_fraction = 2.5 + 0i, complex_vector = c(1, 2) + 1i
)

record <- function(n, method, mean = 0, scale = 1, distribution = "weibull") {
    set.seed(123)
    before <- .Random.seed
    warnings <- character()
    error <- NULL
    value <- withCallingHandlers(tryCatch(
        if (method == "runif") stats::runif(n)
        else survival::rsurvreg(n, mean, scale, distribution),
        error = function(e) { error <<- conditionMessage(e); NULL }),
        warning = function(w) {
            warnings <<- c(warnings, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
    list(values = if (is.null(error)) as.list(value) else NULL, error = error,
         length = if (is.null(error)) length(value) else NULL, dim = dim(value),
         warnings = as.list(warnings), rng_changed = !identical(before, .Random.seed),
         next_uniform = stats::runif(1))
}

cases <- lapply(names(counts), function(name) list(
    name = name, expression = deparse(counts[[name]]),
    runif = record(counts[[name]], "runif"),
    rsurvreg = record(counts[[name]], "rsurvreg")))

# Install a local registered family whose quantile records the probability batch.
# This isolated R process never writes or changes the installed package.
namespace <- asNamespace("survival")
distributions <- survival::survreg.distributions
custom <- distributions$gaussian
custom$name <- "Count callback oracle"
calls <- list()
custom$quantile <- function(p, parms) {
    calls[[length(calls) + 1L]] <<- as.list(p)
    p
}
distributions$count_oracle <- custom
unlockBinding("survreg.distributions", namespace)
assign("survreg.distributions", distributions, namespace)
lockBinding("survreg.distributions", namespace)
for (i in seq_along(cases)) {
    calls <- list()
    cases[[i]]$custom <- record(counts[[i]], "rsurvreg", distribution = "count_oracle")
    cases[[i]]$quantile_calls <- calls
}

arguments <- list(
    zero_mean3 = list(n = 0L, mean = 1:3, scale = 1),
    zero_scale3 = list(n = 0L, mean = 0, scale = 1:3),
    zero_both_vectors = list(n = 0L, mean = 1:3, scale = c(.5, 2)),
    zero_meanempty = list(n = 0L, mean = numeric(), scale = 1),
    two_mean3 = list(n = 2L, mean = 1:3, scale = 1),
    three_mean2 = list(n = 3L, mean = c(0, 1), scale = 1),
    three_scale2 = list(n = 3L, mean = 0, scale = c(1, 2)),
    three_meanempty = list(n = 3L, mean = numeric(), scale = 1)
)
recycling <- lapply(names(arguments), function(name) {
    spec <- arguments[[name]]
    built_in <- do.call(record, c(spec, list(method = "rsurvreg")))
    calls <<- list()
    callback <- do.call(record, c(spec, list(method = "rsurvreg", distribution = "count_oracle")))
    list(name = name, n = spec$n, mean = as.list(spec$mean), scale = as.list(spec$scale),
         rsurvreg = built_in, custom = callback, quantile_calls = calls)
})

reference <- list(metadata = list(generator = "scripts/generate_rsurvreg_count_reference.R",
    r_version = as.character(getRversion()),
    survival_version = as.character(packageVersion("survival")), seed = 123L),
    cases = cases, recycling = recycling)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null")
cat(length(cases), "count and", length(recycling), "recycling cases written to", output, "\n")
