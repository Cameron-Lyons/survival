#!/usr/bin/env Rscript
# Complete R prediction calls, including the Python/Rust boundary. Fit and
# setup are outside timing; numerical equality is checked before measurement.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 5000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 100L, repeats >= 1L)
set.seed(20260929)
data <- data.frame(time = seq_len(n), status = as.integer(seq_len(n) %% 3 != 0),
                   x = rnorm(n))
data$status[n] <- 0L
fit <- survival::coxph(survival::Surv(time, status) ~ x, data)
reference <- survival::yates_setup(fit, "survival")$predict
prepared <- survivalr::yates_setup(fit, "survival")$predict
results <- list()
for (neta in c(1L, 1000L)) {
  eta <- seq(-1, 1, length.out = neta)
  calls <- if (neta == 1L) 100L else 1L
  comparison <- all.equal(prepared(eta), reference(eta), tolerance = 1e-11)
  if (!isTRUE(comparison)) stop(paste(comparison, collapse = "\n"))
  samples <- list()
  for (implementation in c("R_survival", "Rust_bridge")) {
    call <- if (implementation == "R_survival") reference else prepared
    elapsed <- numeric(repeats)
    for (i in seq_len(repeats)) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      for (j in seq_len(calls)) value <- call(eta)
      elapsed[i] <- 1000 * (proc.time()[["elapsed"]] - started) / calls
      rm(value)
    }
    samples[[implementation]] <- list(median_ms = median(elapsed), samples_ms = I(elapsed))
  }
  results[[length(results) + 1L]] <- list(predictors = neta, calls_per_sample = calls, timings = samples)
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
                         survival_version = as.character(packageVersion("survival")),
                         observations = n, baseline_times = length(survival::survfit(fit, censor = FALSE)$time),
                         repeats = repeats, results = results),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
