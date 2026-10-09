#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/aft_rank_deficient_reference.json"

# Preserve stock likelihood, Newton/Fisher steps, Cholesky rank handling and
# methods. Correct only survreg.fit's division by sd=0 for nonbinary constants.
fit_source <- paste(deparse(survival:::survreg.fit), collapse = "\n")
needle <- "function(z) all(z == 0 | z == 1)"
stopifnot(length(gregexpr(needle, fit_source, fixed = TRUE)[[1L]]) == 1L,
          grepl(needle, fit_source, fixed = TRUE))
fit_source <- sub(needle,
  "function(z) all(z == 0 | z == 1) || all(z == z[1L])", fit_source, fixed = TRUE)
reference_env <- new.env(parent = asNamespace("survival"))
reference_env$survreg.fit <- eval(parse(text = fit_source), envir = reference_env)
corrected <- survival::survreg
environment(corrected) <- reference_env

i <- seq_len(80L)
d <- data.frame(time = 10 + (i * 37) %% 97 + i / 7,
  status = as.integer((i * 17 + 3) %% 11 > 2),
  w = rep(c(.7, 1, 1.4, 1.1), length.out = length(i)),
  o = (i %% 5 - 2) / 20, id = (i - 1L) %/% 3L,
  s = factor(ifelse(i %% 2 == 0, "even", "odd")))
d$g <- ordered(c("z", "m", "a")[i %% 3L + 1L], levels = c("z", "m", "a", "unused"))
contrasts(d$g) <- "contr.helmert"
h <- contr.helmert(4L)[as.integer(d$g), , drop = FALSE]
d$h1 <- h[, 1L]
d$h2 <- h[, 2L]
d$constant <- -.1
d$affine <- 3 + 2 * d$h1 - .5 * d$h2

capture <- function(fun, arguments) {
  caught <- character()
  value <- withCallingHandlers(tryCatch(do.call(fun, arguments),
    error = function(e) list(error = conditionMessage(e))),
    warning = function(w) { caught <<- c(caught, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(value = value, warnings = I(caught))
}
rows <- c(1L, 2L, 3L, 5L, 6L, 7L, 8L, 9L, 16L, 30L, 48L, 80L)
pack <- function(fit, complete = TRUE) {
  value <- list(coefficients = I(unname(coef(fit))), var = unname(vcov(fit)),
    naive = unname(fit$naive.var), scale = I(unname(fit$scale)), loglik = I(fit$loglik),
    df = fit$df, df_residual = fit$df.residual, iter = fit$iter,
    score = I(fit$score), lp = I(unname(fit$linear.predictors[rows])))
  if (!complete) return(value)
  lp <- suppressWarnings(predict(fit, type = "lp", se.fit = TRUE))
  quantile <- suppressWarnings(predict(fit, type = "quantile", p = c(.25, .75), se.fit = TRUE))
  c(value, list(x = unname(fit$x[rows, , drop = FALSE]), column_names = I(colnames(fit$x)),
    coefficient_names = I(names(coef(fit))), lp_se = I(unname(lp$se.fit[rows])),
    quantile = unname(quantile$fit[rows, , drop = FALSE]), quantile_se = unname(quantile$se.fit[rows, , drop = FALSE]),
    terms = unname(suppressWarnings(predict(fit, type = "terms", se.fit = TRUE)$fit[rows, , drop = FALSE])),
    terms_se = unname(suppressWarnings(predict(fit, type = "terms", se.fit = TRUE)$se.fit[rows, , drop = FALSE])),
    new_lp = I(unname(suppressWarnings(predict(fit, d[c(3L, 1L, 2L), ], type = "lp")))),
    dfbeta = unname(suppressWarnings(residuals(fit, type = "dfbeta", weighted = TRUE)[rows, , drop = FALSE]))))
}
cases <- list()
for (design in c("helmert_unused", "constant", "affine"))
  for (mode in c("plain", "cluster", "stratified"))
  for (dist in c("weibull", "lognormal", "gaussian", "exponential")) {
    if (mode == "stratified" && dist == "exponential") next
    cat(design, mode, dist, "\n")
    rhs <- switch(design, helmert_unused = "identity(g)",
      constant = "h1 + h2 + constant", affine = "h1 + h2 + affine")
    extra <- if (mode == "cluster") " + offset(o)" else if (mode == "stratified") " + strata(s)" else ""
    form <- as.formula(paste("Surv(time, status) ~", rhs, extra))
    arguments <- list(formula = form, data = d, dist = dist, x = TRUE, model = TRUE, score = TRUE)
    if (mode == "cluster") { arguments$weights <- d$w; arguments$cluster <- d$id }
    stock <- capture(survival::survreg, arguments)
    valid_stock <- is.null(stock$value$error) && all(is.finite(stock$value$loglik)) && !length(stock$warnings)
    fit <- if (valid_stock) stock else capture(corrected, arguments)
    if (!is.null(fit$value$error) || length(fit$warnings)) {
      print(fit$value$error)
      print(fit$warnings)
      stop("reference fit failed")
    }
    stopifnot(anyNA(coef(fit$value)))
    reduced_args <- arguments
    reduced_args$formula <- as.formula(paste("Surv(time, status) ~ h1 + h2", extra))
    reduced <- capture(survival::survreg, reduced_args)
    stopifnot(is.null(reduced$value$error), !length(reduced$warnings), !anyNA(coef(reduced$value)))
    keep <- which(c(!is.na(coef(fit$value)), rep(TRUE, length(fit$value$scale) * (dist != "exponential"))))
    stopifnot(isTRUE(all.equal(fit$value$linear.predictors, reduced$value$linear.predictors, tolerance = 1e-7)),
      isTRUE(all.equal(unname(fit$value$var[keep, keep]), unname(reduced$value$var), tolerance = 1e-7)))
    # Starting at the converged identified model bypasses R's bad scaling and
    # independently exercises its native solver with the entire redundant design.
    initialized_args <- arguments
    initialized_args$init <- c(coef(reduced$value), 0,
      if (dist != "exponential") log(reduced$value$scale))
    initialized <- capture(survival::survreg, initialized_args)
    stopifnot(is.null(initialized$value$error), !length(initialized$warnings),
      isTRUE(all.equal(initialized$value$linear.predictors, fit$value$linear.predictors, tolerance = 1e-7)))
    cases[[length(cases) + 1L]] <- list(name = paste(design, mode, dist, sep = "/"),
      formula = paste(deparse(form), collapse = ""), mode = mode, dist = dist,
      stock_error = stock$value$error, stock_warnings = stock$warnings, corrected = !valid_stock,
      expected = pack(fit$value), init = I(unname(initialized_args$init)),
      initialized = pack(initialized$value, FALSE), reduced = pack(reduced$value, FALSE))
  }
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")),
  correction = "Only when stock fails: leave constant design columns unscaled to avoid division by zero. All R likelihood, optimizer, generalized inverse and prediction code is unchanged.",
  sources = I(c("R/survreg.fit.R", "R/survreg.R", "src/survreg6.c", "src/cholesky3.c", "R/predict.survreg.R"))),
  data = d, factor_levels = I(levels(d$g)), sample_rows = I(rows - 1L), cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "rank-deficient AFT cases written\n")
