#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
package_root <- if (length(args) > 2L) args[[3L]] else "installed"
stopifnot(n > 30L, repeats > 0L)
suppressPackageStartupMessages(library(survival))
if (package_root == "installed") {
  suppressPackageStartupMessages(library(survivalr))
} else if (package_root != "stock") {
  suppressPackageStartupMessages(pkgload::load_all(package_root, quiet = TRUE))
}
set.seed(919)
data <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7), x = rnorm(n),
                   w = runif(n, .5, 2), group = factor(rep(c("a", "b"), length.out = n)))
cases <- list(
  cox = list(shared = function() coxph(Surv(time, status) ~ x + strata(group), data, weights = w),
             stock = function() survival::coxph(survival::Surv(time, status) ~ x + strata(group), data, weights = w)),
  aft = list(shared = function() survreg(Surv(time, status) ~ x + strata(group), data, weights = w),
             stock = function() survival::survreg(survival::Surv(time, status) ~ x + strata(group), data, weights = w)),
  km = list(shared = function() survfit(Surv(time, status) ~ group, data, weights = w),
            stock = function() survival::survfit(survival::Surv(time, status) ~ group, data, weights = w)))
measure <- function(call) {
  for (i in 1:3) call()
  times <- numeric(repeats)
  for (i in seq_len(repeats)) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    result <- call()
    times[i] <- 1000 * (proc.time()[["elapsed"]] - start)
    rm(result)
  }
  list(median_ms = median(times), range_ms = range(times), samples_ms = I(times))
}
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  shared_call <- if (package_root == "stock") calls$stock else calls$shared
  fit <- shared_call()
  expected <- calls$stock()
  if (name == "km") {
    actual_summary <- summary(fit, times = c(.2, .5, 1), extend = TRUE)
    expected_summary <- summary(expected, times = c(.2, .5, 1), extend = TRUE)
    stopifnot(isTRUE(all.equal(actual_summary$surv, expected_summary$surv, tolerance = 1e-7)))
  } else {
    stopifnot(isTRUE(all.equal(unname(coef(fit)), unname(coef(expected)), tolerance = 1e-7)))
    stopifnot(isTRUE(all.equal(unname(vcov(fit)), unname(vcov(expected)), tolerance = 1e-7)))
  }
  results[[name]] <- list(fit = measure(shared_call))
  if (name != "km") {
    results[[name]]$coef_1000 <- measure(function() { for (i in 1:1000) coef(fit); NULL })
    if (!is.null(attr(fit, "survival_state", exact = TRUE))) {
      payload <- serialize(fit, NULL)
      results[[name]]$serialize <- measure(function() serialize(fit, NULL))
      results[[name]]$restore_and_coef <- measure(function() coef(unserialize(payload)))
      results[[name]]$serialized_bytes <- length(payload)
      stopifnot(identical(coef(unserialize(payload)), coef(fit)))
    }
  }
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  reticulate = as.character(packageVersion("reticulate")), package_root = package_root,
  scope = "Complete weighted formula calls, 1000 coef calls, and complete serialization/restoration; explicit GC excluded",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
