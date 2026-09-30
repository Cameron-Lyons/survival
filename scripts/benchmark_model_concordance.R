#!/usr/bin/env Rscript
# Complete fitted-model calls, with setup and explicit garbage collection excluded.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 30L, repeats > 0L)
baseline <- NULL
if (length(args) > 2L) {
  # Optional source containing the previous method and its response helper.
  env <- new.env(parent = asNamespace("survivalr"))
  sys.source(args[[3L]], envir = env)
  baseline <- env$concordance.survival_py_model
}
set.seed(914)
data <- data.frame(x = rnorm(n), z = runif(n), w = runif(n, .5, 2),
                   g = factor(rep(1:5, length.out = n)), id = rep(1:100, length.out = n))
data$time <- rexp(n, exp(data$x/3 + data$z/2))
data$status <- as.integer(runif(n) > .3)
first <- coxph(Surv(time, status) ~ x, data)
second <- coxph(Surv(time, status) ~ x + z, data)
weighted <- coxph(Surv(time, status) ~ x + z, data, weights = w)
grouped <- coxph(Surv(time, status) ~ x + z + strata(g) + cluster(id), data)
ref_first <- survival::coxph(Surv(time, status) ~ x, data)
ref_second <- survival::coxph(Surv(time, status) ~ x + z, data)
ref_weighted <- survival::coxph(Surv(time, status) ~ x + z, data, weights = w)
ref_grouped <- survival::coxph(Surv(time, status) ~ x + z + strata(g) + cluster(id), data)
reference <- get("concordance.coxph", asNamespace("survival"))
cases <- list(
  single = list(stock_R = function() reference(ref_second), shared = function() concordance(second)),
  weighted = list(stock_R = function() reference(ref_weighted), shared = function() concordance(weighted)),
  newdata = list(stock_R = function() reference(ref_weighted, newdata = data),
                 shared = function() concordance(weighted, newdata = data)),
  joint = list(stock_R = function() reference(ref_first, ref_second),
               shared = function() concordance(first, second)),
  grouped = list(stock_R = function() reference(ref_grouped), shared = function() concordance(grouped)))
if (!is.null(baseline)) {
  cases$single$previous <- function() baseline(second)
  cases$weighted$previous <- function() baseline(weighted)
  cases$newdata$previous <- function() baseline(weighted, newdata = data)
  cases$joint$previous <- function() baseline(first, second)
  # The previous method discards groups; its grouped result is incorrect.
}
canonical <- function(result) {
  result$call <- NULL
  result$concordance <- unname(result$concordance)
  if (length(result$concordance) > 1L) {
    rownames(result$count) <- NULL
    result$cvar <- as.numeric(result$cvar)
  }
  result
}
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  expected <- canonical(calls$stock_R())
  for (call in calls) {
    comparison <- all.equal(canonical(call()), expected, tolerance = 1e-9)
    if (!isTRUE(comparison)) stop(name, ": ", paste(comparison, collapse = "; "))
    for (i in 1:3) call()
  }
  timings <- lapply(calls, function(call) numeric())
  for (i in seq_len(repeats)) {
    order <- if (i %% 2) names(calls) else rev(names(calls))
    for (kind in order) {
      gc(FALSE)
      start <- proc.time()[["elapsed"]]
      value <- calls[[kind]]()
      timings[[kind]] <- c(timings[[kind]], 1000*(proc.time()[["elapsed"]]-start))
      rm(value)
    }
  }
  results[[name]] <- lapply(timings, function(values) list(
    median_ms = median(values), range_ms = range(values), samples_ms = I(values)))
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L,
  r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
  scope = "Complete R-facing calls; fitting, input setup and explicit garbage collection excluded",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
