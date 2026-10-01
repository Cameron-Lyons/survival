#!/usr/bin/env Rscript
# Complete R-facing calls with a logical subset, compared against stock R.
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
package_root <- if (length(args) > 2L) args[[3L]] else "r/survivalr"
stopifnot(n >= 30L, repeats > 0L)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all(package_root, quiet = TRUE))
set.seed(915)
data <- data.frame(x = rnorm(n), z = runif(n), w = runif(n, .5, 2))
data$time <- rexp(n, exp(data$x/3 + data$z/2))
data$status <- as.integer(runif(n) > .3)
selected <- data$z > .25
reference_curve <- get("survfit.formula", asNamespace("survival"))
reference_concordance <- get("concordance.formula", asNamespace("survival"))
cases <- list(
  cox = list(shared = function() coxph(Surv(time, status) ~ x + z, data,
                                      subset = selected, weights = w),
             stock_R = function() survival::coxph(survival::Surv(time, status) ~ x + z, data,
                                                   subset = selected, weights = w)),
  aft = list(shared = function() survreg(Surv(time, status) ~ x + z, data,
                                        subset = selected, weights = w),
             stock_R = function() survival::survreg(survival::Surv(time, status) ~ x + z, data,
                                                     subset = selected, weights = w)),
  curve = list(shared = function() survfit(Surv(time, status) ~ 1, data,
                                          subset = selected, weights = w),
               stock_R = function() reference_curve(survival::Surv(time, status) ~ 1, data,
                                                       subset = selected, weights = w)),
  concordance = list(shared = function() concordance(Surv(time, status) ~ x, data,
                                                    subset = selected, weights = w),
                     stock_R = function() reference_concordance(survival::Surv(time, status) ~ x, data,
                                                                   subset = selected, weights = w)))
canonical <- function(result, name) {
  if (name %in% c("cox", "aft")) return(unname(coef(result)))
  if (name == "curve") return(as.numeric(result$surv))
  as.numeric(result$concordance)
}
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  expected <- canonical(calls$stock_R(), name)
  for (call in calls) {
    stopifnot(isTRUE(all.equal(canonical(call(), name), expected, tolerance = 1e-8)))
    for (i in 1:3) call()
  }
  timings <- lapply(calls, function(call) numeric())
  for (i in seq_len(repeats)) {
    for (kind in if (i %% 2) names(calls) else rev(names(calls))) {
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
cat(jsonlite::toJSON(list(rows = n, selected = sum(selected), repeats = repeats, warmups = 3L,
  r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
  scope = "Complete R-facing subset and fit calls; input setup and explicit garbage collection excluded",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
