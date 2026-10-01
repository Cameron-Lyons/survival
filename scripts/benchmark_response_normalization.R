#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
package_root <- if (length(args) > 2L) args[[3L]] else "r/survivalr"
stopifnot(n > 30L, repeats > 0L)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all(package_root, quiet = TRUE))
set.seed(916)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7), x = rnorm(n), w = runif(n, .5, 2))
d$x[seq_len(n) %% 17 == 0] <- NA
interval <- transform(d, status = rep(0:3, length.out = n), end = time + runif(n, .01, 1))
interval$end[interval$status != 3] <- NA
cases <- list(
  response = list(shared = function() Surv(d$time, d$status),
                  stock_R = function() survival::Surv(d$time, d$status)),
  model_frame = list(shared = function() model.frame(Surv(time, status) ~ x, d, na.action = na.omit),
                     stock_R = function() model.frame(survival::Surv(time, status) ~ x, d, na.action = na.omit)),
  cox = list(shared = function() coxph(Surv(time, status) ~ x, d, weights = w),
             stock_R = function() survival::coxph(survival::Surv(time, status) ~ x, d, weights = w)),
  interval_aft = list(shared = function() survreg(Surv(time, end, status, type='interval') ~ x, interval, weights = w),
                      stock_R = function() survival::survreg(survival::Surv(time, end, status, type='interval') ~ x, interval, weights = w)))
canonical <- function(value, name) {
  if (name == "response") return(unname(as.matrix(value)))
  if (name == "model_frame") return(unname(as.matrix(model.response(value))))
  unname(coef(value))
}
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  expected <- canonical(calls$stock_R(), name)
  for (call in calls) {
    stopifnot(isTRUE(all.equal(canonical(call(), name), expected, check.attributes = FALSE, tolerance = 1e-8)))
    for (i in 1:3) call()
  }
  timings <- lapply(calls, function(call) numeric())
  for (i in seq_len(repeats)) for (kind in if (i %% 2) names(calls) else rev(names(calls))) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    value <- calls[[kind]]()
    timings[[kind]] <- c(timings[[kind]], 1000*(proc.time()[["elapsed"]]-start))
    rm(value)
  }
  results[[name]] <- lapply(timings, function(values) list(
    median_ms = median(values), range_ms = range(values), samples_ms = I(values)))
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete calls including response preparation and fitting; setup and explicit GC excluded",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
