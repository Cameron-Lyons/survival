#!/usr/bin/env Rscript
# Independent stock-survival near-tie and complete-fit references at large scales.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/aeq_large_time_reference.json"
cases <- list()
numbers <- function(values) lapply(unname(values), function(value) {
    if (is.na(value)) return(NULL)
    if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
    value
})
add <- function(name, time, start = NULL, tolerance = NULL, fit = TRUE) {
    status <- rep(c(1L, 0L, 1L), length.out = length(time))
    response <- if (is.null(start)) Surv(time, status) else Surv(start, time, status)
    normalized <- tryCatch({
        if (is.null(tolerance)) aeqSurv(response) else aeqSurv(response, tolerance)
    }, error = function(error) list(error = conditionMessage(error)))
    expected <- if (is.list(normalized)) normalized else list(
        time = I(unname(normalized[, ncol(normalized) - 1L])),
        start = if (is.null(start)) NULL else I(unname(normalized[, 1L])))
    curve <- NULL
    if (fit && is.null(tolerance) && !is.list(normalized)) {
        fitted <- survfit(response ~ 1)
        curve <- lapply(c("time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
                          "std.err", "std.chaz", "lower", "upper"),
                        function(field) numbers(fitted[[field]]))
        names(curve) <- c("time", "n_risk", "n_event", "n_censor", "surv", "cumhaz",
                          "std_err", "std_chaz", "lower", "upper")
    }
    cases[[length(cases) + 1L]] <<- list(name = name, time = I(time),
        start = if (is.null(start)) NULL else I(start), status = I(status),
        tolerance = tolerance, expected = expected, fit = curve)
}
add("huge_distinct", c(1e308, 1.1e308, 1.2e308))
add("huge_mixed_sign", c(-1e308, 0, 1e308))
add("huge_near_tie", c(1e308, 1e308 * (1 + 1e-12), 1.2e308))
add("huge_negative_near_tie", c(-1.2e308, -1e308, -1e308 * (1 - 1e-12)))
add("largest_distinct", .Machine$double.xmax * c(.5, .7, 1))
add("largest_near_tie", .Machine$double.xmax * c(1 - 1e-12, 1, 1))
add("repeated_huge", c(1e308, 1e308, 1.2e308, 1.2e308, 1.4e308))
add("ordinary_near_tie", c(1, 1 + 1e-12, 2))
add("relative_scale", c(1e9, 1e9 + 1, 1e9 + 20))
add("subnormal_default", c(0, 5e-324, 1e-323))
add("subnormal_zero_tolerance", c(0, 5e-324, 1e-323), tolerance = 0)
add("subnormal_custom_tolerance", c(0, 5e-324, 1e-323), tolerance = 5e-324)
for (tolerance in c(-1, 0, 1e-12, .09, .1, 1)) {
    add(paste0("huge_tolerance_", tolerance), c(1e308, 1.1e308, 1.2e308),
        tolerance = tolerance)
}
for (tolerance in c(.5, 2/3, .7)) {
    add(paste0("ordinary_boundary_", tolerance), c(1, 2), tolerance = tolerance)
}
add("counting_huge_distinct", c(1e308, 1.1e308, 1.2e308), start = c(0, 0, 0))
add("counting_huge_delayed", c(1e308, 1.1e308, 1.2e308), start = c(0, 1e308, 1.1e308))
add("counting_huge_mixed_sign", c(1e308, 1.1e308, 1.2e308),
    start = c(-1e308, -5e307, 0))
add("counting_huge_near_tie", c(1e308, 1e308 * (1 + 1e-12), 1.2e308),
    start = c(0, 1e307, 2e307))
add("counting_huge_zero_interval", c(1e308 * (1 + 1e-12), 1.2e308),
    start = c(1e308, 1.1e308), fit = FALSE)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(list(r_version = as.character(getRversion()),
    survival_version = as.character(packageVersion("survival")), cases = cases), output,
    # 17 significant digits retain finite xmax instead of rounding it to Inf.
    auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null", na = "null")
