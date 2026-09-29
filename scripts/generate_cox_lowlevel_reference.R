#!/usr/bin/env Rscript
# Bare fit components, warnings and shapes from the exported survival fitters.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/cox_lowlevel_reference.json"
cases <- list()
time <- c(4, 2, 7, 3, 5, 2, 8, 6, 4, 9, 6, 10)
event <- c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0)
start <- c(1, 0, 3, 0, 2, 0, 5, 1, 0, 4, 3, 2)
x <- cbind(age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
           group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0))
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
add <- function(name, fitter, design = x, counting = fitter != "coxph", times = time,
                events = event, starts = start, strata = NULL, offset = NULL,
                init = NULL, weights = NULL, method = "efron", resid = TRUE,
                nocenter = NULL, control = list(), rownames = NULL) {
    response <- if (counting) Surv(starts, times, events) else Surv(times, events)
    warnings <- character()
    result <- withCallingHandlers(tryCatch(do.call(get(paste0(fitter, ".fit")), list(
        x = design, y = response, strata = strata, offset = offset, init = init,
        control = do.call(coxph.control, control), weights = weights, method = method,
        rownames = rownames, resid = resid, nocenter = nocenter)),
        error = function(e) list(error = conditionMessage(e))),
        warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
    expected <- if (!is.null(result$error)) result else list(
        coefficients = if (is.null(result$coefficients)) NULL else I(unname(result$coefficients)),
        var = if (is.null(result$var)) NULL else rows(result$var),
        loglik = I(result$loglik), score = result$score, iter = result$iter,
        linear_predictors = if (is.matrix(result$linear.predictors)) rows(result$linear.predictors) else I(result$linear.predictors),
        residuals = if (is.null(result$residuals)) NULL else I(unname(result$residuals)),
        means = if (is.null(result$means)) NULL else I(result$means),
        first = if (is.null(result$first)) NULL else I(result$first),
        info = if (is.null(result$info)) NULL else as.list(result$info),
        method = result$method, classes = I(result$class %||% character()),
        coefficient_names = if (is.null(names(result$coefficients))) NULL else I(names(result$coefficients)),
        row_names = if (is.null(names(result$residuals))) NULL else I(names(result$residuals)))
    cases[[length(cases) + 1L]] <<- list(name = name, fitter = fitter,
        x = rows(design), column_names = if (is.null(colnames(design))) NULL else I(colnames(design)),
        time = I(times), event = I(events), start = if (counting) I(starts) else NULL,
        arguments = list(strata = if (is.null(strata)) NULL else I(strata),
            offset = if (is.null(offset)) NULL else I(offset), init = if (is.null(init)) NULL else I(init),
            weights = if (is.null(weights)) NULL else I(weights), method = method, resid = resid,
            nocenter = if (is.null(nocenter)) NULL else I(nocenter),
            control = control, rownames = if (is.null(rownames)) NULL else I(rownames)),
        expected = expected, warnings = I(warnings))
}
`%||%` <- function(x, y) if (is.null(x)) y else x
for (fitter in c("coxph", "agreg", "agexact")) {
    add(paste0(fitter, "_default"), fitter)
    add(paste0(fitter, "_breslow"), fitter, method = "breslow")
    add(paste0(fitter, "_exact_label"), fitter, method = "exact")
    add(paste0(fitter, "_offset_strata"), fitter,
        strata = rep(c(2, 1, 1), 4), offset = seq(1.2, 2.3, length.out = 12),
        rownames = paste0("row", seq_len(12)))
    add(paste0(fitter, "_nocenter"), fitter, nocenter = c(-1, 0, 1))
    add(paste0(fitter, "_no_residuals"), fitter, resid = FALSE)
    add(paste0(fitter, "_initial"), fitter, init = c(.2, -.3), control = list(iter.max = 0))
    add(paste0(fitter, "_one_iteration"), fitter, control = list(iter.max = 1))
    add(paste0(fitter, "_nonconvergence"), fitter, control = list(iter.max = 2))
    add(paste0(fitter, "_null"), fitter, design = matrix(numeric(), 12, 0), offset = rep(.2, 12))
    add(paste0(fitter, "_null_no_residuals"), fitter, design = matrix(numeric(), 12, 0), resid = FALSE)
    add(paste0(fitter, "_alias"), fitter, design = cbind(x, duplicate = x[, 1], constant = 1))
    add(paste0(fitter, "_alias_no_iterations"), fitter, design = cbind(x, duplicate = x[, 1]), control = list(iter.max = 0))
    add(paste0(fitter, "_no_events"), fitter, events = rep(0, 12))
    add(paste0(fitter, "_wrong_init"), fitter, init = 1)
    add(paste0(fitter, "_weighted"), fitter, weights = rep(c(.5, 1, 2), 4))
}
add("agexact_right", "agexact", counting = FALSE)
add("agexact_right_null", "agexact", counting = FALSE, design = matrix(numeric(), 12, 0))
add("coxph_near_ties", "coxph", times = time + seq_along(time) * 1e-10)
add("agreg_near_ties", "agreg", times = time + seq_along(time) * 1e-10)
add("agreg_ignored_rows", "agreg", starts = c(3.9, 0, 6.5, 0, 2, 1.9, 5, 5.9, 0, 4, 3, 9.9))
add("coxph_empty_init", "coxph", init = numeric())
add("agreg_empty_init", "agreg", init = numeric())
add("coxph_null_init", "coxph", design = matrix(numeric(), 12, 0), init = c(1, 2))
add("agreg_null_init", "agreg", design = matrix(numeric(), 12, 0), init = 1)
add("coxph_no_events_large_offset", "coxph", events = rep(0, 12), offset = rep(600, 12))
add("agexact_null_no_events", "agexact", events = rep(0, 12), design = matrix(numeric(), 12, 0))
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
    cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
