#!/usr/bin/env Rscript
# Independent stock-R aggregate.survfit callback, ordering and sum references.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/aggregate_fun_reference.json"

encode <- function(values) lapply(unname(as.vector(values)), function(value) {
    if (is.na(value)) return("NaN")
    if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
    value
})
array_spec <- function(values) list(shape = I(dim(values)), values = encode(values))
surv <- matrix(c(.91, .82, .73, .64, .55, .46, .37, .28, .19, .12, .08, .04), 3, 4)
pstate <- array(c(.91, .82, .73, .64, .55, .46, .37, .28, .19, .12, .08, .04,
                 .09, .18, .27, .36, .45, .54, .63, .72, .81, .88, .92, .96),
                c(3, 4, 2))
inputs <- list(ordinary = list(surv = array_spec(surv), pstate = NULL),
               multistate = list(surv = NULL, pstate = array_spec(pstate)),
               both = list(surv = array_spec(surv), pstate = array_spec(pstate)))
missing_surv <- surv
missing_surv[2, 1] <- NA_real_
missing_surv[3, 3] <- NaN
missing_pstate <- pstate
missing_pstate[2, 1, 1] <- NA_real_
missing_pstate[3, 3, 2] <- NaN
inputs$missing <- list(surv = array_spec(missing_surv), pstate = array_spec(missing_pstate))
inputs$extreme <- list(surv = array_spec(matrix(c(1e308, 1e-308, -1e308,
    1e-308, 1e308, -1e308, Inf, 2, -Inf, 3, 5, 7), 3, 4)), pstate = NULL)
inputs$sum_precision <- list(surv = array_spec(matrix(c(1e16, 1, -1e16, 0,
    1e308, 1e308, -1e308, 0, Inf, 1, 2, 3), 3, 4, byrow = TRUE)), pstate = NULL)
inputs$nonfinite <- list(surv = array_spec(matrix(c(Inf, -Inf, 2, 0,
    Inf, 1, 2, 3, -Inf, 1, 2, 3), 3, 4, byrow = TRUE)), pstate = NULL)
inputs$extended_precision <- list(surv = array_spec(matrix(c(1e300, 1, -1e300, 0,
    1e308, 1e308, 1e308, -1e308, 1e-308, 1e-308, -1e-308, 0), 3, 4, byrow = TRUE)),
    pstate = NULL)
inputs$maximum <- list(surv = array_spec(matrix(.Machine$double.xmax *
    c(1, 1, 1, 1, 1, 0, -1, 0, 1, 1, -1, -1), 3, 4, byrow = TRUE)), pstate = NULL)

decode <- function(spec) if (is.null(spec)) NULL else array(
    vapply(spec$values, function(value) as.numeric(value), numeric(1)), spec$shape)
common <- list(n = 8L, time = c(1, 2, 4), n.risk = c(8, 6, 3),
    n.event = c(1, 2, 1), n.censor = c(1, 1, 2), type = "right",
    strata = c(first = 1L, second = 2L), start.time = 0.5,
    std.err = matrix(.1, 3, 4), std.cumhaz = matrix(.2, 3, 4),
    lower = matrix(.01, 3, 4), upper = matrix(.99, 3, 4),
    conf.int = .95, conf.type = "log", logse = TRUE,
    cumhaz = matrix(.3, 3, 4), newdata = data.frame(x = 1:4))
curve <- function(input) {
    fields <- common
    fields$surv <- decode(input$surv)
    fields$pstate <- decode(input$pstate)
    if (!is.null(fields$pstate)) {
        fields$states <- c("(s0)", "absorbed")
        fields$p0 <- matrix(c(.8, .2), 1)
        fields$n.id <- 8L
        fields$n.transition <- matrix(1:3, 3)
        fields$transitions <- matrix(c(0L, 1L, 0L, 0L), 2)
        fields$type <- "mright"
        class(fields) <- c("survfitcoxms", "survfitms", "survfit")
    } else class(fields) <- c("survfitcox", "survfit")
    fields
}
by_sources <- list(none = NULL, bare = c("b", "a", "b", "a"),
    compound = list(g = c("b", "a", "b", "a"), h = c(2, 1, 1, 2)),
    constant = rep("same", 4),
    declared = factor(c("b", "a", "b", "a"), levels = c("b", "a")))
by_spec <- function(by) {
    if (is.null(by)) return(NULL)
    if (is.list(by)) return(list(kind = "mapping", columns = lapply(by, I)))
    list(kind = if (is.factor(by)) "factor" else "vector",
         values = I(as.character(by)), levels = if (is.factor(by)) I(levels(by)) else NULL)
}
functions <- list(sum = sum, affine = function(z) z[1] + 2*z[length(z)] + length(z) + sum(z)/8,
    span = function(z) max(z) - min(z), last = function(z) z[length(z)],
    nanmean = function(z) mean(z, na.rm = TRUE),
    boolean = function(z) TRUE, string = function(z) "1",
    vector = function(z) c(1, 2), list = function(z) list(1),
    required = function(z, scale) scale*sum(z))

# The Python API retains a singleton data axis; normalize only that documented
# shape difference, while recording stock R dimensions separately.
snapshot <- function(fit) {
    n_groups <- if (is.null(fit$newdata)) 1L else nrow(fit$newdata)
    record <- function(values, states = NULL) {
        if (is.null(values)) return(NULL)
        shape <- if (is.null(states)) c(length(fit$time), n_groups) else
            c(length(fit$time), n_groups, states)
        list(shape = I(shape), values = encode(values),
             r_dim = if (is.null(dim(values))) NULL else I(dim(values)))
    }
    list(surv = record(fit$surv), pstate = record(fit$pstate, length(fit$states)),
         newdata = if (is.null(fit$newdata)) NULL else lapply(fit$newdata, I),
         preserved = I(setdiff(names(fit), c("surv", "pstate", "newdata"))),
         removed = I(setdiff(names(common), names(fit))))
}
cases <- list()
add <- function(input_name, by_name, fun, dots = list(), named = FALSE, default = FALSE) {
    calls <- list()
    base_fun <- functions[[fun]]
    callback <- function(z, ...) {
        calls[[length(calls) + 1L]] <<- list(values = encode(z), dots = list(...))
        base_fun(z, ...)
    }
    x <- curve(inputs[[input_name]])
    value <- tryCatch({
        options <- list(x = x, by = by_sources[[by_name]])
        if (!default) options$FUN <- if (named) get(fun) else callback
        fit <- do.call(aggregate, c(options, dots))
        snapshot(fit)
    }, error = function(error) list(error = conditionMessage(error)))
    cases[[length(cases) + 1L]] <<- list(
        name = paste(input_name, by_name, fun,
            if (default) "default" else if (named) "name" else "callback", sep = "/"),
        input = input_name, by = by_spec(by_sources[[by_name]]), fun = fun,
        named = named, default = default, dots = dots, calls = calls, expected = value)
}
for (input in c("ordinary", "multistate", "both")) for (by in names(by_sources)) {
    for (fun in c("sum", "affine", "span", "last")) add(input, by, fun)
    add(input, by, "sum", named = TRUE)
}
for (by in c("none", "bare", "compound")) {
    for (fun in c("sum", "affine", "nanmean")) add("missing", by, fun,
        dots = list(na.rm = TRUE, ignored = "unused"))
    add("missing", by, "sum", dots = list(na.rm = TRUE), named = TRUE)
}
for (by in c("none", "bare")) for (fun in c("boolean", "string", "vector", "list"))
    add("both", by, fun)
add("ordinary", "bare", "required", dots = list(scale = 3))
for (by in c("none", "bare")) add("extreme", by, "sum", named = TRUE)
for (input in c("sum_precision", "nonfinite", "extended_precision", "maximum")) for (by in c("none", "bare"))
    for (fun in c("sum", "mean")) add(input, by, fun, named = TRUE)
for (input in names(inputs)) for (by in c("none", "constant", "bare"))
    add(input, by, "mean", named = TRUE, default = TRUE)

# Empty-axis behavior is retained as evidence for stock apply's probing quirks.
# The port intentionally keeps its existing well-shaped empty array behavior.
empty_cases <- list()
for (margin in c("surv", "pstate")) {
    shapes <- if (margin == "surv") list(c(0, 4), c(3, 0)) else
        list(c(0, 4, 2), c(3, 4, 0), c(3, 0, 2))
    for (shape in shapes) for (grouped in c(FALSE, TRUE)) {
        x <- common
        x[[margin]] <- array(numeric(prod(shape)), shape)
        x$states <- if (margin == "pstate") seq_len(shape[3]) else NULL
        class(x) <- if (margin == "pstate") c("survfitms", "survfit") else "survfit"
        calls <- list()
        callback <- function(z) { calls[[length(calls)+1L]] <<- encode(z); sum(z) }
        value <- tryCatch({
            fit <- aggregate(x, by = if (grouped) rep(c("b", "a"), length.out = shape[2]) else NULL,
                             FUN = callback)
            list(shape = if (is.null(dim(fit[[margin]]))) NULL else I(dim(fit[[margin]])),
                 values = encode(fit[[margin]]))
        }, error = function(error) list(error = conditionMessage(error)))
        empty_cases[[length(empty_cases)+1L]] <- list(margin = margin, shape = I(shape),
            grouped = grouped, calls = calls, expected = value)
    }
}
actual_data <- data.frame(time = 1:16, status = rep(c(1, 1, 0, 1), 4),
    x = c(0, 1, 3, 2, 1, 4, 2, 3, 0, 2, 1, 3, 4, 1, 0, 2),
    g = factor(rep(c("a", "b"), 8)))
actual_fit <- survfit(coxph(Surv(time, status) ~ x + strata(g), actual_data),
    newdata = data.frame(x = c(-1, 0, 1, 2)))
actual_empty <- list(time = actual_fit[integer(), , drop = FALSE],
                    data = actual_fit[, integer(), drop = FALSE])
actual_empty_evidence <- lapply(names(actual_empty), function(axis) {
    x <- actual_empty[[axis]]
    calls <- list()
    callback <- function(z) { calls[[length(calls)+1L]] <<- encode(z); sum(z) }
    value <- tryCatch({
        fit <- aggregate(x, FUN = callback)
        list(shape = if (is.null(dim(fit$surv))) NULL else I(dim(fit$surv)),
             values = encode(fit$surv))
    }, error = function(error) list(error = conditionMessage(error)))
    list(axis = axis, source = "survfit.coxph subset with drop=FALSE",
         input_shape = I(dim(x$surv)), calls = calls, expected = value)
})
grouping_inputs <- list(missing_multi = c("b", "a", NA, "b"),
    missing_single = c("b", NA, "b", "b"),
    unused_bare = factor(c("b", "a", "b", "a"), levels = c("b", "unused", "a")),
    unused_named = list(g = factor(c("b", "a", "b", "a"), levels = c("b", "unused", "a"))))
grouping_evidence <- lapply(names(grouping_inputs), function(name) {
    calls <- list()
    callback <- function(z) { calls[[length(calls)+1L]] <<- encode(z); sum(z) }
    value <- tryCatch({
        fit <- aggregate(curve(inputs$ordinary), by = grouping_inputs[[name]], FUN = callback)
        list(surv = array_spec(fit$surv), newdata = lapply(fit$newdata, I))
    }, error = function(error) list(error = conditionMessage(error)))
    list(name = name, calls = calls, expected = value)
})
late_calls <- list()
late_vector <- function(z) {
    late_calls[[length(late_calls)+1L]] <<- encode(z)
    if (length(late_calls) == 1L) 1 else c(1, 2)
}
late_fit <- aggregate(curve(inputs$ordinary), FUN = late_vector)
late_vector_evidence <- list(calls = late_calls, expected = array_spec(late_fit$surv))
numeric_returns <- list(integer = 2L, double = 2, vector_one = c(2), matrix_one = matrix(2, 1, 1),
    complex = 1+2i, boolean = TRUE, string = "1", list_one = list(1), vector_two = c(1, 2))
numeric_return_evidence <- lapply(names(numeric_returns), function(name) {
    calls <- list()
    callback <- function(z) { calls[[length(calls)+1L]] <<- encode(z); numeric_returns[[name]] }
    value <- tryCatch({
        fit <- aggregate(curve(inputs$ordinary), FUN = callback)
        list(type = typeof(fit$surv), values = if (is.complex(fit$surv))
            I(as.character(fit$surv)) else encode(fit$surv))
    }, error = function(error) list(error = conditionMessage(error)))
    list(name = name, calls = calls, expected = value)
})
default_mean_row <- .Machine$double.xmax * c(1, 1, -1, -1, 0)
default_mean_row[5] <- 1
default_mean_evidence <- list()
for (margin in c("ordinary", "both")) for (by_name in c("none", "constant", "grouped")) {
    values <- rep(default_mean_row, if (by_name == "grouped") 2L else 1L)
    x <- curve(inputs$ordinary)
    x$time <- 1
    x$surv <- matrix(values, 1)
    if (margin == "both") {
        x$pstate <- array(rep(values, 2), c(1, length(values), 2))
        x$states <- c("first", "second")
        class(x) <- c("survfitcoxms", "survfitms", "survfit")
    }
    by <- switch(by_name, none = NULL, constant = rep("same", length(values)),
        grouped = rep(c("b", "a"), each = 5))
    for (default in c(TRUE, FALSE)) {
        options <- list(x = x, by = by)
        if (!default) options$FUN <- mean
        fit <- do.call(aggregate, options)
        groups <- if (is.null(fit$newdata)) 1L else nrow(fit$newdata)
        default_mean_evidence[[length(default_mean_evidence)+1L]] <- list(
            name = paste(margin, by_name, if (default) "default" else "mean", sep = "/"),
            margin = margin, default = default, by = by_spec(by),
            surv = array_spec(x$surv), pstate = if (is.null(x$pstate)) NULL else array_spec(x$pstate),
            expected = list(surv = list(shape = I(c(1, groups)), values = encode(fit$surv)),
                pstate = if (is.null(fit$pstate)) NULL else
                    list(shape = I(c(1, groups, 2)), values = encode(fit$pstate)),
                newdata = if (is.null(fit$newdata)) NULL else lapply(fit$newdata, I)))
    }
}
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(list(metadata = list(generator = "scripts/generate_aggregate_fun_reference.R",
    r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
    long_double_mantissa = .Machine$longdouble.digits),
    inputs = inputs, common = list(n = I(common$n), time = I(common$time), n_risk = I(common$n.risk),
        n_event = I(common$n.event), n_censor = I(common$n.censor), strata = as.list(common$strata),
    start_time = common$start.time), cases = cases, empty_axis_evidence = empty_cases,
    actual_empty_axis_evidence = actual_empty_evidence, grouping_evidence = grouping_evidence,
    late_vector_evidence = late_vector_evidence, numeric_return_evidence = numeric_return_evidence,
    default_mean_evidence = default_mean_evidence),
    output, auto_unbox = TRUE, digits = 17, pretty = TRUE, null = "null", na = "null")
cat(length(cases), "aggregate callback/sum cases written to", output, "\n")
