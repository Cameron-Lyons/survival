#!/usr/bin/env Rscript
# Rscript scripts/generate_penalty_constructor_reference.R [output.json]
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/penalty_constructor_reference.json"

number <- function(x) {
    if (is.nan(x)) "NaN" else if (is.na(x)) NULL
    else if (is.infinite(x)) if (x > 0) "Inf" else "-Inf" else unname(x)
}
numbers <- function(x) unname(lapply(x, number))
rows <- function(x) {
    if (is.null(x)) return(NULL)
    lapply(seq_len(nrow(x)), function(i) numbers(x[i, ]))
}
plain <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.matrix(x)) return(rows(x))
    if (is.list(x)) return(lapply(x, plain))
    if (is.numeric(x)) return(if (length(x) == 1L) number(x) else numbers(x))
    if (length(x) == 1L) return(unname(x))
    as.list(unname(x))
}
input <- function(x, name) list(
    name = name, type = typeof(x), matrix = is.matrix(x),
    shape = as.list(if (is.matrix(x)) dim(x) else length(x)),
    values = if (is.matrix(x)) rows(x) else if (is.numeric(x)) numbers(x) else as.list(x),
    columns = if (is.null(colnames(x))) NULL else as.list(colnames(x)),
    levels = if (is.factor(x)) as.list(levels(x)) else NULL)

capture <- function(fun) {
    warnings <- character()
    error <- NULL
    value <- withCallingHandlers(tryCatch(fun(), error = function(e) {
        error <<- conditionMessage(e)
        NULL
    }), warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    list(value = value, error = error, warnings = as.list(warnings))
}

constructors <- list()
add <- function(name, kind, values, options = list(), constructor = kind) {
    symbols <- names(values)
    env <- list2env(values, parent = globalenv())
    call <- as.call(c(list(as.name(constructor)), lapply(symbols, as.name), options))
    got <- capture(function() eval(call, env))
    expected <- list(error = got$error, warnings = got$warnings)
    if (is.null(got$error)) {
        value <- got$value
        groups <- if (kind == "frailty") factor(values[[1]]) else NULL
        lev <- if (kind == "frailty") levels(groups) else NULL
        codes <- if (kind == "frailty") as.numeric(value) else NULL
        sparse <- attr(value, "sparse")
        basis <- if (kind == "ridge") unclass(value)
                 else if (isTRUE(sparse)) matrix(codes, ncol = 1L)
                 else attr(value, "contrasts")[as.integer(value), , drop = FALSE]
        cparm <- attr(value, "cparm")
        # R's $name selects the first entry when options contain duplicate names.
        effective <- if (length(cparm)) cparm[!duplicated(names(cparm))] else cparm
        entries <- lapply(seq_along(cparm), function(i)
            list(name = names(cparm)[i], value = plain(cparm[[i]])))
        coef <- seq_len(if (kind == "ridge") ncol(basis) else length(lev)) / 10
        pparm <- attr(value, "pparm")
        penalty_call <- c(list(coef, .4, 5L), if (!is.null(pparm)) list(pparm))
        penalty <- capture(function() do.call(attr(value, "pfun"), penalty_call))
        initial <- capture(function() attr(value, "cfun")(cparm, 0L, NULL))
        expected <- c(expected, list(
            basis = rows(basis), shape = as.list(dim(basis)),
            levels = if (is.null(lev)) NULL else as.list(lev),
            codes = if (is.null(codes)) NULL else numbers(codes),
            class = as.list(class(value)),
            varname = if (is.null(attr(value, "varname"))) NULL else as.list(attr(value, "varname")),
            pparm = if (is.null(pparm)) NULL else numbers(pparm),
            cparm = plain(effective), cparm_entries = entries,
            diag = attr(value, "diag"), sparse = sparse,
            method = if (kind == "ridge") if (is.null(options$theta)) "df" else "fixed"
                     else get("method", environment(attr(value, "pfun")), inherits = FALSE),
            cargs = if (is.null(attr(value, "cargs"))) NULL else as.list(attr(value, "cargs")),
            pfun_probe = list(coef = numbers(coef), theta = .4, ndead = 5L,
                expected = plain(penalty$value), error = penalty$error),
            initial_control = list(expected = plain(initial$value), error = initial$error)))
    }
    constructors[[length(constructors) + 1L]] <<- list(name = name, kind = kind,
        constructor = constructor, inputs = unname(Map(input, values, symbols)), options = plain(options),
        expression = paste(deparse(call), collapse = " "), expected = expected)
}

x <- c(-1, -.5, 0, .5, 1, 2)
z <- c(3, 1, 4, 2, 6, 5)
m <- cbind(a = x, b = z)
add("ridge/default", "ridge", list(x = x))
add("ridge/fixed", "ridge", list(x = x), list(theta = .4))
add("ridge/unscaled", "ridge", list(x = x), list(theta = .4, scale = FALSE))
add("ridge/zero", "ridge", list(x = x), list(theta = 0))
add("ridge/df", "ridge", list(x = x, z = z), list(df = .7, eps = .002))
add("ridge/two-vectors", "ridge", list(x = x, z = z), list(theta = .4))
add("ridge/matrix", "ridge", list(m = m), list(theta = .4))
add("ridge/matrix-and-vector", "ridge", list(m = m, z = z), list(theta = .4))
add("ridge/recycling", "ridge", list(x = x, z = c(2, 7)), list(theta = .4))
add("ridge/recycling-warning", "ridge", list(x = x[1:5], z = c(2, 7)), list(theta = .4))
add("ridge/missing", "ridge", list(x = c(1, NA, 3, 4, NaN, 6)))
add("ridge/constant", "ridge", list(x = rep(3, 5)))
add("ridge/singleton", "ridge", list(x = 3))
add("ridge/all-missing", "ridge", list(x = c(NA_real_, NA_real_)))
add("ridge/empty-vector", "ridge", list(x = numeric()))
add("ridge/empty-arguments", "ridge", list())
add("ridge/infinity", "ridge", list(x = c(1, 2, Inf)))
add("ridge/extreme", "ridge", list(x = c(1e308, -1e308, 1)))
add("ridge/large-repeated", "ridge", list(x = c(1e308, 1e308)))
add("ridge/subnormal", "ridge", list(x = c(5e-324, 1e-323, 1.5e-323)))
add("ridge/cancellation-variance", "ridge", list(x = c(1e16, 1, 1e16 + 2)))
add("ridge/theta-and-df", "ridge", list(x = x), list(theta = .4, df = 1))
add("ridge/negative-theta", "ridge", list(x = x), list(theta = -1))

g <- c("b", "a", "b", "c", "a", NA)
f <- factor(g, levels = c("unused", "c", "b", "a", "also-unused"))
for (distribution in c("gamma", "gaussian", "t")) {
    for (sparse in c(FALSE, TRUE)) add(
        paste("frailty", distribution, if (sparse) "sparse" else "dense", sep = "/"),
        "frailty", list(x = f), list(distribution = distribution, sparse = sparse, theta = .4))
    add(paste0("frailty/", distribution, "/default"), "frailty", list(x = g),
        list(distribution = distribution))
    add(paste0("frailty/", distribution, "/df"), "frailty", list(x = g),
        list(distribution = distribution, df = 1.1))
    add(paste0("frailty/", distribution, "/aic"), "frailty", list(x = g),
        list(distribution = distribution, method = "aic", eps = .003, caic = TRUE))
    add(paste0("frailty/", distribution, "/init"), "frailty", list(x = g),
        list(distribution = distribution, init = c(.2, 2)))
    add(paste0("frailty/", distribution, "/aic-init1"), "frailty", list(x = g),
        list(distribution = distribution, method = "aic", init = .2))
    add(paste0("frailty/", distribution, "/alias"), "frailty", list(x = f),
        list(sparse = TRUE, theta = .4), paste0("frailty.", distribution))
}
add("frailty/numeric", "frailty", list(x = c(10, 2, 10, 1, NA)), list(theta = .4))
add("frailty/logical", "frailty", list(x = c(TRUE, FALSE, TRUE, NA)), list(theta = .4))
add("frailty/five-default", "frailty", list(x = 1:5))
add("frailty/six-default", "frailty", list(x = 1:6))
add("frailty/distribution-prefix", "frailty", list(x = g), list(distribution = "gaus"))
add("frailty/distribution-unknown", "frailty", list(x = g), list(distribution = "bad"))
add("frailty/distribution-case", "frailty", list(x = g), list(distribution = "Gamma"))
add("frailty/theta-and-df", "frailty", list(x = g), list(theta = .4, df = 1))
add("frailty/df-method-without-df", "frailty", list(x = g), list(method = "df"))
add("frailty/fixed-without-theta", "frailty", list(x = g), list(method = "fixed"))
add("frailty/theta-nonfixed", "frailty", list(x = g), list(theta = .4, method = "em"))
add("frailty/df-nondf", "frailty", list(x = g), list(df = 1, method = "em"))
add("frailty/method-prefix", "frailty", list(x = g), list(method = "e"))
add("frailty/method-unknown", "frailty", list(x = g), list(method = "bad"))
add("frailty/gaussian-dfzero", "frailty", list(x = g), list(distribution = "gaussian", df = 0))
add("frailty/t-dfzero", "frailty", list(x = g), list(distribution = "t", df = 0))
add("frailty/t-invalid-df", "frailty", list(x = g), list(distribution = "t", tdf = 2))
add("frailty/one-dense", "frailty", list(x = rep("a", 3)))
add("frailty/one-sparse", "frailty", list(x = rep("a", 3)), list(sparse = TRUE))
add("frailty/missing-dense", "frailty", list(x = rep(NA_character_, 3)))
add("frailty/missing-sparse", "frailty", list(x = rep(NA_character_, 3)), list(sparse = TRUE))
add("frailty/empty-dense", "frailty", list(x = character()))
add("frailty/empty-sparse", "frailty", list(x = character()), list(sparse = TRUE))

# Complete unweighted fits check that the constructor basis, scaling, names,
# and native penalty configuration remain connected through coxph(tt=).
data <- ovarian
data$x <- (data$age - 60) / 10
data$g <- factor(data$rx, levels = c(2, 1, 9))
fit_specs <- list(
    list(name = "tt/ridge-scaled", kind = "ridge", transform = "single", options = list(theta = .4)),
    list(name = "tt/ridge-multiple", kind = "ridge", transform = "multiple", options = list(theta = .4)),
    list(name = "tt/ridge-unscaled", kind = "ridge", transform = "single", options = list(theta = .4, scale = FALSE)),
    list(name = "tt/ridge-df", kind = "ridge", transform = "single", options = list(df = .7, eps = .001)),
    list(name = "tt/gamma-sparse", kind = "frailty", options = list(distribution = "gamma", sparse = TRUE, theta = .4)),
    list(name = "tt/gamma-dense", kind = "frailty", options = list(distribution = "gamma", sparse = FALSE, theta = .4)),
    list(name = "tt/gaussian-sparse", kind = "frailty", options = list(distribution = "gaussian", sparse = TRUE, theta = .4)),
    list(name = "tt/t-dense", kind = "frailty", options = list(distribution = "t", sparse = FALSE, theta = .4)),
    list(name = "tt/gamma-aic-init1", kind = "frailty", options = list(distribution = "gamma", sparse = TRUE, method = "aic", init = .2)),
    list(name = "ordinary/gamma-aic-init1", kind = "frailty", transform = "ordinary", options = list(distribution = "gamma", sparse = TRUE, method = "aic", init = .2)),
    list(name = "ordinary/gamma-em-init-vector", kind = "frailty", transform = "ordinary", options = list(distribution = "gamma", sparse = TRUE, init = c(.2, 2))),
    list(name = "ordinary/gaussian-reml-init-vector", kind = "frailty", transform = "ordinary", options = list(distribution = "gaussian", sparse = TRUE, init = c(.2, 2))),
    list(name = "ordinary/ridge-subset", kind = "ridge", transform = "ordinary", options = list(theta = .4), subset = which(seq_len(nrow(data)) %% 4 != 0))
)
fits <- lapply(fit_specs, function(spec) {
    fun <- function(x, t, ...) {
        if (spec$kind == "frailty") return(do.call(frailty, c(list(x = x), spec$options)))
        if (spec$transform == "multiple") return(ridge(x * log(t), x * sqrt(t), theta = .4))
        if (isFALSE(spec$options$scale)) return(ridge(x * log(t), theta = .4, scale = FALSE))
        if (!is.null(spec$options$df)) return(ridge(x * log(t), df = .7, eps = .001))
        ridge(x * log(t), theta = .4)
    }
    formula <- if (identical(spec$transform, "ordinary") && spec$kind == "frailty") {
                   penalty <- as.call(c(list(as.name("frailty"), as.name("g")), spec$options))
                   paste("Surv(futime, fustat) ~ x +", paste(deparse(penalty, width.cutoff = 500L), collapse = " "))
               }
               else if (identical(spec$transform, "ordinary")) "Surv(futime, fustat) ~ ridge(x, theta=.4)"
               else if (spec$kind == "frailty") "Surv(futime, fustat) ~ x + tt(g)"
               else "Surv(futime, fustat) ~ tt(x)"
    formula <- paste(deparse(as.formula(formula), width.cutoff = 500L), collapse = " ")
    call <- list(formula = as.formula(formula), data = data, ties = "efron", robust = FALSE,
                 x = TRUE, y = TRUE, control = coxph.control(eps = 1e-10, iter.max = 100, outer.max = 50))
    if (!identical(spec$transform, "ordinary")) call$tt <- fun
    if (!is.null(spec$subset)) call$subset <- spec$subset
    got <- capture(function() do.call(coxph, call))
    expected <- list(error = got$error, warnings = got$warnings)
    if (is.null(got$error)) {
        fit <- got$value
        summary <- summary(fit)
        expected <- c(expected, list(
            coef = numbers(coef(fit)), coefficient_names = as.list(names(coef(fit))),
            variance = rows(fit$var), var2 = rows(fit$var2), loglik = numbers(fit$loglik),
            lp = numbers(fit$linear.predictors), df = numbers(fit$df),
            frail = if (is.null(fit$frail)) NULL else numbers(fit$frail),
            fvar = if (is.null(fit$fvar)) NULL else numbers(fit$fvar),
            x = rows(fit$x), matrix_names = as.list(colnames(fit$x)),
            matrix_assign = as.list(attr(fit$x, "assign")),
            y = rows(unclass(fit$y)), pterms = as.list(fit$pterms),
            summary = rows(summary$coefficients), summary_names = as.list(rownames(summary$coefficients)),
            summary_columns = as.list(colnames(summary$coefficients)), print2 = plain(summary$print2),
            history = plain(fit$history), penalty = numbers(fit$penalty)))
    }
    columns <- if (spec$kind == "frailty") list(list(source = "g", transform = "identity"))
               else if (spec$transform == "multiple") list(
                   list(source = "x", transform = "log_time"),
                   list(source = "x", transform = "sqrt_time"))
               else list(list(source = "x", transform = if (spec$transform == "ordinary") "identity" else "log_time"))
    c(spec, list(formula = formula, columns = columns,
        subset_zero_based = if (is.null(spec$subset)) NULL else as.list(spec$subset - 1L),
        expected = expected))
})

reference <- list(metadata = list(generator = "scripts/generate_penalty_constructor_reference.R",
    r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
    numbers = "NA is null; NaN, Inf and -Inf are strings. Raw varname attributes and cparm entry order are retained."),
    constructors = constructors, data = data, group_levels = as.list(levels(data$g)), fits = fits)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null", dataframe = "columns")
cat(length(constructors), "constructors and", length(fits), "complete fits written to", output, "\n")
