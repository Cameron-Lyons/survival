#!/usr/bin/env Rscript
# Independent ordinary survfitcox margin/subscript references from stock R.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/cox_survfit_subset_reference.json"
numbers <- function(x) lapply(as.vector(x), function(value) if (is.na(value)) "NaN" else value)
array_spec <- function(x) if (is.null(x)) NULL else list(
    shape = I(if (is.null(dim(x))) length(x) else dim(x)), values = numbers(x))
data <- data.frame(time = 1:12, status = rep(c(1, 1, 0), 4), x = rep(c(0, 1, 2), 4),
    z = c(0, 1, .5, -1, 2, .7, 1, 0, -1, .5, 2, 1), g = factor(rep(c("a", "b"), each = 6)))
tiny_data <- data.frame(time = 1:8, status = rep(1, 8), x = rep(c(0, 1), 4),
    g = factor(c(rep("a", 7), "b")))
newdata <- data.frame(x = c(-1, 0, 2), z = c(1, 0, -1),
    category = factor(c("y", "x", "y"), levels = c("x", "unused", "y")),
    row.names = c("two", "one", "three"))
fit <- coxph(Surv(time, status) ~ x + z + strata(g), data,
    init = c(.25, -.125), iter.max = 0)
sources <- list(stratified = survfit(fit, newdata = newdata),
    plain = survfit(coxph(Surv(time, status) ~ x + z, data, init = c(.25, -.125), iter.max = 0),
        newdata = newdata), single = survfit(fit, newdata = newdata[1, , drop = FALSE]),
    tiny = survfit(coxph(Surv(time, status) ~ x + strata(g), tiny_data, init = .25, iter.max = 0),
        newdata = newdata),
    narrow = survfit(coxph(Surv(time, status) ~ x + strata(g), data, init = .25, iter.max = 0),
        newdata = newdata["x"]))
snapshot <- function(x) {
    fields <- lapply(c("time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
        "std.err", "std.chaz", "lower", "upper"), function(name) array_spec(x[[name]]))
    names(fields) <- c("time", "n_risk", "n_event", "n_censor", "surv", "cumhaz",
        "std_err", "std_chaz", "lower", "upper")
    list(n = I(x$n), strata_names = if (is.null(x$strata)) NULL else I(names(x$strata)),
        strata_sizes = if (is.null(x$strata)) NULL else I(unname(x$strata)),
        dim = as.list(dim(x)), fields = fields,
        colnames = if (is.matrix(x$surv) && !is.null(colnames(x$surv))) I(colnames(x$surv)) else NULL,
        newdata_kind = if (is.null(x$newdata)) NULL else if (is.data.frame(x$newdata)) "frame" else "vector",
        newdata = if (is.null(x$newdata)) NULL else if (is.data.frame(x$newdata))
            lapply(x$newdata, function(column) I(as.character(column))) else I(x$newdata),
        newdata_rownames = if (is.null(rownames(x$newdata))) NULL else I(rownames(x$newdata)))
}
derived <- function(x) {
    if (!length(x$time) || is.matrix(x$surv) && !ncol(x$surv) ||
        !is.null(x$newdata) && !is.data.frame(x$newdata)) return(NULL)
    summary_snapshot <- function(value) {
        fields <- lapply(c("time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
            "std.err", "std.chaz", "lower", "upper"), function(name) array_spec(value[[name]]))
        names(fields) <- c("time", "n_risk", "n_event", "n_censor", "surv", "cumhaz",
            "std_err", "std_chaz", "lower", "upper")
        list(fields = fields, strata = if (is.null(value$strata)) NULL else I(as.character(value$strata)))
    }
    quantiles <- quantile(x, probs = .5)
    list(initial = snapshot(survfit0(x)),
        summary = summary_snapshot(summary(x, censored = TRUE)),
        at_times = summary_snapshot(summary(x, times = c(0, 4, 8, 12), extend = TRUE)),
        quantile = lapply(quantiles[c("quantile", "lower", "upper")], numbers))
}
cases <- list()
add <- function(source, name, strata = NULL, data = NULL, drop = TRUE, curves = NULL) {
    x <- sources[[source]]
    if (!is.null(curves)) value <- x[curves + 1L, drop = drop]
    else if (source == "plain") value <- x[data + 1L, drop = drop]
    else if (source == "single") value <- x[strata + 1L, drop = drop]
    else {
        i <- if (is.null(strata)) seq_along(x$strata) else if (is.character(strata)) strata else strata + 1L
        j <- if (is.null(data)) seq_len(ncol(x$surv)) else if (is.character(data)) data else data + 1L
        value <- x[i, j, drop = drop]
    }
    cases[[length(cases)+1L]] <<- list(name = paste(source, name, sep = "/"), source = source,
        strata = if (is.null(strata)) NULL else I(strata), data = if (is.null(data)) NULL else I(data),
        curves = if (is.null(curves)) NULL else I(curves), drop = drop,
        expected = snapshot(value), derived = derived(value))
}
add("stratified", "reordered", c(1L, 0L), c(2L, 1L, 0L), FALSE)
for (drop in c(FALSE, TRUE)) {
    add("stratified", paste0("single_both_", drop), 1L, 1L, drop)
    add("stratified", paste0("single_stratum_", drop), 1L, drop = drop)
    add("stratified", paste0("single_data_", drop), data = 1L, drop = drop)
    add("plain", paste0("single_data_", drop), data = 1L, drop = drop)
    add("single", paste0("single_stratum_", drop), strata = 1L, drop = drop)
}
add("stratified", "repeated", c(1L, 0L, 1L), c(2L, 0L, 2L), FALSE)
add("stratified", "named_repeated", c("b", "a", "b"), c(2L, 0L), FALSE)
add("stratified", "empty_strata", integer(), drop = FALSE)
add("stratified", "empty_data", data = integer(), drop = FALSE)
add("plain", "repeated_data", data = c(2L, 0L, 2L), drop = FALSE)
add("plain", "empty_data", data = integer(), drop = FALSE)
add("single", "repeated_strata", strata = c(1L, 0L, 1L), drop = FALSE)
add("single", "empty_strata", strata = integer(), drop = FALSE)
add("stratified", "empty_strata_drop", integer(), drop = TRUE)
add("single", "empty_strata_drop", strata = integer(), drop = TRUE)
add("plain", "empty_data_drop", data = integer(), drop = TRUE)
add("narrow", "reordered", c(1L, 0L), c(2L, 0L), FALSE)
add("narrow", "single_data", data = 1L, drop = FALSE)
add("narrow", "empty_data", data = integer(), drop = FALSE)
add("narrow", "strata_only", strata = 1L, drop = FALSE)
for (selection in list(c(1L, 0L), 3L, 0:5, 5:0, c(3L, 0L, 3L)))
    add("stratified", paste0("linear_", paste(selection, collapse = "_")), curves = selection)
add("tiny", "single_time_keeps_matrix", 1L, 1L, TRUE)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(list(metadata = list(generator = "scripts/generate_cox_survfit_subset_reference.R",
    r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival"))),
    data = lapply(data, function(column) I(if (is.factor(column)) as.character(column) else column)),
    tiny_data = lapply(tiny_data, function(column) I(if (is.factor(column)) as.character(column) else column)),
    newdata = lapply(newdata, function(column) I(if (is.factor(column)) as.character(column) else column)),
    newdata_rownames = I(rownames(newdata)), category_levels = I(levels(newdata$category)),
    sources = lapply(sources, snapshot), cases = cases), output,
    auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null", na = "null")
cat(length(cases), "ordinary Cox subset cases written to", output, "\n")
