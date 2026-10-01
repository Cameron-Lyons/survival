#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
package_root <- if (length(args) > 2L) args[[3L]] else "r/survivalr"
stopifnot(n > 30L, repeats > 0L)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all(package_root, quiet = TRUE))
set.seed(917)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7), x = rnorm(n),
                w = runif(n, .5, 2), g = factor(rep(c("a", "b", "c"), length.out = n)))
d$x[seq_len(n) %% 17 == 0] <- NA
cases <- list(
  aft = list(shared = function() survreg(Surv(time, status) ~ x + strata(g), d, weights = w),
             stock_R = function() survival::survreg(survival::Surv(time, status) ~ x + strata(g), d, weights = w)),
  aft_empty = list(shared = function() survreg(Surv(time, status) ~ x + strata(g), d, subset = g != "b", weights = w),
                   stock_R = function() survival::survreg(survival::Surv(time, status) ~ x + strata(g), d, subset = g != "b", weights = w)),
  ridge = list(shared = function() survreg(Surv(time, status) ~ ridge(x, theta = 2) + strata(g), d, weights = w),
               stock_R = function() survival::survreg(survival::Surv(time, status) ~ ridge(x, theta = 2) + strata(g), d, weights = w)),
  ridge_empty = list(shared = function() survreg(Surv(time, status) ~ ridge(x, theta = 2) + strata(g), d, subset = g != "b", weights = w)),
  cox_empty = list(shared = function() coxph(Surv(time, status) ~ x + strata(g), d, subset = g != "b", weights = w),
                   stock_R = function() survival::coxph(survival::Surv(time, status) ~ x + strata(g), d, subset = g != "b", weights = w)))
# R's penalized empty-strata fit fails during inversion. Recode only removed
# rows to check estimable coefficients without altering the penalty's scaling.
compact <- d
compact$g[compact$g == "b"] <- "a"
selected <- which(d$g != "b")
compact_ridge <- survival::survreg(survival::Surv(time, status) ~ ridge(x, theta = 2) + strata(g),
                                  compact, subset = selected, weights = w)
canonical <- function(fit) {
  variance <- vcov(fit)
  used <- diag(variance) != 0
  list(coefficients = unname(coef(fit)), variance = unname(variance[used, used, drop = FALSE]))
}
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  expected <- canonical(if (name == "ridge_empty") compact_ridge else calls$stock_R())
  for (call in calls) {
    stopifnot(isTRUE(all.equal(canonical(call()), expected, check.attributes = FALSE, tolerance = 1e-7)))
    for (i in 1:3) call()
  }
  timings <- lapply(calls, function(call) numeric())
  for (i in seq_len(repeats)) for (kind in if (i %% 2) names(calls) else rev(names(calls))) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    value <- calls[[kind]]()
    timings[[kind]] <- c(timings[[kind]], 1000 * (proc.time()[["elapsed"]] - start))
    rm(value)
  }
  results[[name]] <- lapply(timings, function(values) list(
    median_ms = median(values), range_ms = range(values), samples_ms = I(values)))
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete weighted formula calls; input construction and explicit GC excluded",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
