#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/cox_time_transform_reference.json"
# Stock 3.8-12 counts main-effect terms rather than distinct tt variables. This
# fails for interaction-only transforms and mismatches lists with mixed orders.
# Correct only that variable count; the R expansion and fitter are unchanged.
source <- deparse(survival::coxph)
stopifnot(sum(grepl("ntrans <- length(timetrans$terms)", source, fixed = TRUE)) == 1L)
corrected <- eval(parse(text = sub("ntrans <- length(timetrans$terms)",
  "ntrans <- length(timetrans$vars)", source, fixed = TRUE)), envir = asNamespace("survival"))
d <- lung[1:60, c("time", "status", "age", "sex")]
d$time <- floor(d$time / 50) + 1
d$status <- as.integer(d$status == 2)
d$x <- (d$age - 60) / 10
d$z <- sin(seq_len(nrow(d)))
d$start <- pmin(seq_len(nrow(d)) %% 3, d$time - .5)
d$g <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
d$h <- factor(rep("same", nrow(d)), levels = c("same", "unused"))
d$w <- rep(c(.7, 1, 1.4), length.out = nrow(d))
d$o <- (seq_len(nrow(d)) %% 5 - 2) / 20
d$id <- (seq_len(nrow(d)) - 1L) %/% 3L
transforms <- list(
  vector = function(x,t,...) x * log(t),
  unnamed = function(x,t,...) cbind(x * log(t), x * sqrt(t)),
  named = function(x,t,...) cbind(log = x * log(t), root = x * sqrt(t)),
  one = function(x,t,...) cbind(log = x * log(t)),
  boolean = function(x,t,...) x > median(x),
  factor = function(x,t,...) factor(x > median(x), levels = c(TRUE, FALSE, "unused")),
  ordered = function(x,t,...) ordered(ifelse(x < -.5,"low",ifelse(x > .5,"high","middle")),
                                     levels=c("low","middle","high")),
  factor_input = function(x,t,...) cbind(log = as.integer(x) * log(t), root = as.integer(x) * sqrt(t)),
  constant_input = function(x,t,...) as.integer(x) * log(t),
  root = function(x,t,...) x * sqrt(t))
specs <- list(
  vector = list(rhs = "x + tt(x)", transform = "vector"),
  unnamed = list(rhs = "x + tt(x)", transform = "unnamed"),
  named = list(rhs = "x + tt(x)", transform = "named"),
  one = list(rhs = "x + tt(x)", transform = "one"),
  boolean = list(rhs = "x + tt(x)", transform = "boolean"),
  factor = list(rhs = "x + tt(x)", transform = "factor"),
  ordered = list(rhs = "x + tt(x)", transform = "ordered"),
  matrix_interaction = list(rhs = "tt(x) + tt(x):z", transform = "named"),
  factor_interaction = list(rhs = "tt(x) + tt(x):z", transform = "factor"),
  matrix_only_interaction = list(rhs = "x:tt(x)", transform = "named"),
  factor_only_interaction = list(rhs = "tt(x):g", transform = "factor"),
  multiple = list(rhs = "tt(x) + tt(z)", transform = c("named", "root")),
  mixed_order = list(rhs = "tt(x):z + tt(z)", transform = c("named", "root")),
  factor_input = list(rhs = "x + tt(g)", transform = "factor_input"),
  constant_input = list(rhs = "x + tt(h)", transform = "constant_input"),
  default_factor = list(rhs = "x + tt(g)", transform = NULL),
  weighted = list(rhs = "x + tt(x) + offset(o)", transform = "named", weighted = TRUE),
  stratified = list(rhs = "x + tt(x) + strata(g)", transform = "named", weighted = TRUE),
  robust = list(rhs = "x + tt(x) + strata(g)", transform = "named", cluster = TRUE),
  counting = list(rhs = "x + tt(x) + strata(g)", transform = "named", counting = TRUE, weighted = TRUE),
  counting_id = list(rhs = "x + tt(x) + strata(g)", transform = "named", counting = TRUE, id = TRUE),
  subset = list(rhs = "x + tt(x) + strata(g)", transform = "named", subset = TRUE),
  missing = list(rhs = "x + tt(x) + strata(g)", transform = "named", missing = TRUE))
cases <- list()
for (name in names(specs)) for (method in c("efron", "breslow")) {
  cat(name, method, "\n")
  spec <- specs[[name]]
  data <- d
  # Stock Ccoxcount2 misindexes stratum boundaries while walking tied deaths.
  # Keep counting-process stop times distinct here so these are stock fits;
  # tied right-censored cases still exercise the complete transform path.
  if (isTRUE(spec$counting)) data$time <- data$time + seq_len(nrow(data)) / 1000
  if (isTRUE(spec$missing)) data$x[c(2, 9, 25)] <- NA
  formula <- paste(if (isTRUE(spec$counting)) "Surv(start, time, status)" else "Surv(time, status)",
                   "~", spec$rhs)
  functions <- unname(transforms[spec$transform])
  call <- list(formula = as.formula(formula), data = data, ties = method, x = TRUE, y = TRUE,
               control = coxph.control(eps = 1e-10, iter.max = 50),
               na.action = if (isTRUE(spec$missing)) na.exclude else na.omit,
               robust = isTRUE(spec$cluster) || isTRUE(spec$id))
  if (length(functions)) call$tt <- functions
  if (isTRUE(spec$weighted)) call$weights <- data$w
  if (isTRUE(spec$cluster)) call$cluster <- data$id
  if (isTRUE(spec$id)) call$id <- data$id
  if (isTRUE(spec$subset)) call$subset <- seq_len(nrow(data)) %% 4 != 0
  stock <- tryCatch(do.call(survival::coxph, call), error = function(e) list(error = conditionMessage(e)))
  fit <- if (is.null(stock$error)) stock else do.call(corrected, call)
  detail <- tryCatch(survival::coxph.detail(fit), error = function(e) list(error = conditionMessage(e)))
  cases[[length(cases) + 1L]] <- c(list(name = paste(name, method, sep = "/"), formula = formula,
    method = method, stock_error = stock$error,
    transform = if (is.null(spec$transform)) NULL else I(spec$transform),
    coefficients = I(unname(coef(fit))), names = I(names(coef(fit))), variance = unname(vcov(fit)),
    naive = unname(fit$naive.var), loglik = I(fit$loglik), x = unname(fit$x), y = unname(unclass(fit$y)),
    means = I(unname(fit$means)), n = fit$n, nevent = fit$nevent,
    assign = lapply(fit$assign, function(x) I(x - 1L)),
    martingale = I(unname(residuals(fit))),
    detail_error = detail$error,
    detail = if (!is.null(detail$error)) NULL else
      list(time = I(detail$time), nrisk = I(detail$nrisk), nevent = I(detail$nevent),
           hazard = I(detail$hazard), score = unname(detail$score), means = unname(detail$means))),
    spec[c("weighted", "cluster", "counting", "id", "subset", "missing")])
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  correction = "Only after stock errors, ntrans counts timetrans$vars rather than main-effect timetrans$terms; R risk-set expansion and numerical kernels are unchanged."),
  data = d, levels = list(g = I(levels(d$g)), h = I(levels(d$h))), cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "Cox time-transform references written\n")
