#!/usr/bin/env Rscript
# Complete Yates setup calls; model fitting and prediction are outside timing.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 5000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 100L, repeats >= 1L)
set.seed(20260929)
data <- data.frame(time = seq_len(n), status = as.integer(seq_len(n) %% 3 != 0),
                   start = runif(n, 0, 1), weight = runif(n, .5, 2))
for (j in seq_len(40L)) data[[paste0("x", j)]] <- rnorm(n)
results <- list()
for (case in c("right_1", "right_40", "counting_10")) {
  counting <- case == "counting_10"
  p <- switch(case, right_1 = 1L, right_40 = 40L, counting_10 = 10L)
  formula <- reformulate(paste0("x", seq_len(p)),
    response = if (counting) "Surv(start,time,status)" else "Surv(time,status)")
  fit <- survival::coxph(formula, data, weights = weight, model = TRUE, x = TRUE, y = TRUE)
  calls <- list(R_survival = function() survival::yates_setup(fit, "survival"),
                Rust_bridge = function() survivalr::yates_setup(fit, "survival"))
  actual <- attr(calls$Rust_bridge(), "survivalr_prediction")$baseline
  expected <- survival::survfit(fit, censor = FALSE)
  actual$call <- expected$call <- NULL
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-10)),
            isTRUE(all.equal(calls$Rust_bridge()$predict(c(-1, 0, 1)),
                             calls$R_survival()$predict(c(-1, 0, 1)), tolerance = 1e-10)))
  for (i in seq_len(3L)) for (call in calls) invisible(call())
  samples <- lapply(calls, function(call) numeric(repeats))
  for (i in seq_len(repeats)) {
    order <- if (i %% 2L) names(calls) else rev(names(calls))
    for (implementation in order) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- calls[[implementation]]()
      samples[[implementation]][i] <- 1000 * (proc.time()[["elapsed"]] - started)
      rm(value)
    }
  }
  results[[case]] <- list(covariates = p, counting = counting,
    baseline_times = length(actual$time), timings = lapply(samples, function(x) {
      list(median_ms = median(x), samples_ms = I(x))
    }))
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")), observations = n,
  repeats = repeats, warmup_calls = 3L, results = results),
  auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
