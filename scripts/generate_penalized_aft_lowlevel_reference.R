#!/usr/bin/env Rscript
# Stock survival only: exclude callback-density interval/sparse cases with known C bugs.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/penalized_aft_lowlevel_reference.json"
cases <- list()
time <- c(1.2, 2.5, .9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9)
status <- c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)
x <- cbind(Intercept = 1, age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
           group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0))
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
vec <- function(x) if (is.null(x)) NULL else I(unname(x))
add <- function(name, dist = "extreme", design = x, y = cbind(time, status),
                weights = NULL, offset = NULL, init = NULL, controlvals = list(),
                scale = 0, nstrat = 1, strata = NULL, parms = if (dist == "t") 4 else NULL,
                specs = list(list(kind = "ridge", theta = 1)), pcols = list(2:3),
                assign = list(Intercept = 1, ridge = 2:3), spline_x = NULL) {
    pattr <- Map(function(spec, columns) {
        kind <- spec$kind; spec$kind <- NULL
        value <- if (kind == "ridge") do.call(ridge, c(list(design[, columns, drop = FALSE]), spec))
        else if (kind == "pspline") do.call(pspline, c(list(spline_x), spec))
        else { d <- spec$distribution; spec$distribution <- NULL
            do.call(frailty, c(list(rep(1:3, 4), distribution = d), spec)) }
        attrs <- attributes(value)
        # Prepared Python penalties carry names through the design, not varname.
        attrs$varname <- NULL
        attrs
    }, specs, pcols)
    warnings <- character()
    result <- withCallingHandlers(tryCatch(survpenal.fit(design, y, weights, offset, init,
        do.call(survreg.control, controlvals), dist, scale, nstrat, strata, pcols, pattr, assign, parms),
        error = function(e) list(error = conditionMessage(e))),
        warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
    expected <- if (!is.null(result$error)) result else list(
        coefficients = vec(result$coefficients), icoef = vec(result$icoef),
        var = rows(result$var), var2 = rows(result$var2), loglik = vec(result$loglik),
        iter = vec(result$iter), linear_predictors = vec(result$linear.predictors),
        df = vec(result$df), penalty = vec(result$penalty), score = vec(result$score),
        frail = vec(result$frail), fvar = vec(result$fvar),
        coefficient_names = vec(names(result$coefficients)), pterms = as.list(result$pterms),
        assign2 = lapply(result$assign2, function(x) I(x - 1L)),
        history = lapply(result$history, function(h) list(theta=h$theta, done=h$done,
            history=if (is.null(h$history)) list() else rows(h$history),
            columns=vec(colnames(h$history)), c_loglik=h$c.loglik, half=h$half)))
    if (is.null(result$error)) {
        full <- result
        nvar <- length(result$coefficients) - if (scale > 0) 0 else nstrat
        full$coefficients <- result$coefficients[seq_len(nvar)]
        full$scale <- if (scale > 0) scale else exp(tail(result$coefficients, nstrat))
        if (length(full$scale) > 1) names(full$scale) <- as.character(seq_len(nstrat))
        full$idf <- 1 + if (scale > 0) 0 else nstrat
        class(full) <- c("survreg.penal", "survreg")
        expected$print <- lapply(c(FALSE, TRUE), function(terms)
            I(sub("[[:space:]]+$", "", capture.output(print(full, terms=terms)))))
    }
    cases[[length(cases) + 1L]] <<- list(name = name, x = rows(design), y = rows(y),
        column_names = vec(colnames(design)), specs = specs, pcols = lapply(pcols, function(x) I(x-1L)),
        assign = lapply(assign, function(x) I(x-1L)), spline_x = vec(spline_x),
        arguments = list(dist = dist, weights = vec(weights), offset = vec(offset), init = vec(init),
            controlvals = controlvals, scale = scale, nstrat = nstrat, strata = vec(strata), parms = vec(parms)),
        expected = expected, warnings = I(warnings))
}
for (dist in c("extreme", "logistic", "gaussian", "t")) {
    add(paste0(dist, "_ridge"), dist)
    add(paste0(dist, "_weighted_offset"), dist, weights = rep(c(.5, 1, 2), 4), offset = seq(.1, 1.2, length.out=12))
    add(paste0(dist, "_fixed"), dist, scale=1.5)
    add(paste0(dist, "_strata"), dist, nstrat=2, strata=rep(c(1, 2), 6))
    add(paste0(dist, "_df"), dist, specs=list(list(kind="ridge", df=1)))
    add(paste0(dist, "_zero_iterations"), dist, controlvals=list(iter.max=0))
    add(paste0(dist, "_one_iteration"), dist, controlvals=list(iter.max=1))
    add(paste0(dist, "_initial"), dist, init=c(2, 0, 0, .1))
    if (dist != "t") add(paste0(dist, "_interval"), dist, y=cbind(time, time+.4, rep(c(1, 0, 2, 3), 3)))
}
add("unscaled_ridge", "gaussian", specs=list(list(kind="ridge", theta=2, scale=FALSE)))
add("outer_limit", "gaussian", specs=list(list(kind="ridge", df=1)), controlvals=list(outer.max=1))
add("no_intercept", "gaussian", design=x[,-1], pcols=list(1:2), assign=list(ridge=1:2))
for (family in c("gamma", "gaussian", "t")) {
    spec <- list(kind="frailty", distribution=family, sparse=TRUE)
    if (family == "t") spec$df <- 1
    add(paste0(family, "_search"), "gaussian", design=cbind(x[,1:2], frailty=rep(1:3,4)),
        specs=list(spec), pcols=list(3), assign=list(Intercept=1, age=2, frailty=3))
}
add("alias", "gaussian", design=cbind(x, duplicate=x[,2]),
    assign=list(Intercept=1, ridge=2:3, duplicate=4))
add("reordered_penalties", "gaussian", specs=list(list(kind="ridge", theta=2), list(kind="ridge", theta=1)),
    pcols=list(3,2), assign=list(Intercept=1, age=2, group=3))
add("reordered_terms", "gaussian", assign=list(ridge=2:3, Intercept=1))
add("unused_stratum", "gaussian", nstrat=3, strata=rep(c(1,2),6))
for (dist in c("extreme", "gaussian")) for (family in c("gamma", "gaussian", "t")) {
    spec <- list(kind="frailty", distribution=family, theta=.4, sparse=TRUE)
    design <- cbind(x[,1:2], frailty=rep(1:3,4))
    add(paste(dist, family, "sparse", sep="_"), dist, design=design, specs=list(spec), pcols=list(3),
        assign=list(Intercept=1, age=2, frailty=3), offset=rep(.2,12))
    spec$sparse <- FALSE
    design <- cbind(x[,1:2], diag(3)[rep(1:3,4),])
    colnames(design)[3:5] <- paste0("frail",1:3)
    add(paste(dist, family, "dense", sep="_"), dist, design=design, specs=list(spec), pcols=list(3:5),
        assign=list(Intercept=1, age=2, frailty=3:5))
}
add("zero_frailty", "gaussian", design=cbind(x[,1:2], frailty=rep(1:3,4)),
    specs=list(list(kind="frailty", distribution="gaussian", theta=0, sparse=TRUE)),
    pcols=list(3), assign=list(Intercept=1, age=2, frailty=3))
for (method in c("fixed", "df")) {
    z <- seq(.1, 1.2, length.out=12)
    spec <- if (method == "fixed") list(kind="pspline", theta=.4, nterm=4) else list(kind="pspline", df=2, nterm=4)
    basis <- do.call(pspline, c(list(z), spec[-1]))
    design <- cbind(Intercept=1, basis)
    colnames(design)[-1] <- paste0("spline",seq_len(ncol(basis)))
    add(paste0("spline_",method), "gaussian", design=design, specs=list(spec),
        pcols=list(2:ncol(design)), assign=list(Intercept=1, spline=2:ncol(design)), spline_x=z)
}
write_json(list(r_version=R.version.string, survival_version=as.character(packageVersion("survival")),
    cases=cases), output, auto_unbox=TRUE, pretty=TRUE, digits=NA, na="null", null="null")
