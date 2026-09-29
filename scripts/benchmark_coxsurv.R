#!/usr/bin/env Rscript
# Complete R calls, including the Python/Rust boundary. Requires jsonlite,
# pkgload and the built Python extension selected by RETICULATE_PYTHON.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 1000L, repeats >= 1L)
set.seed(20260929)
p <- 4L
x <- matrix(rnorm(n * p), n, p)
time <- seq_len(n)
status <- as.integer(time %% 3L != 0L)
status[(n - 3L):n] <- 0L
strata <- factor(rep(1:4, length.out = n))
base <- list(ctype = 2L, stype = 2L, varmat = diag(.01, p),
             y = cbind(time, status), x = x,
             wt = rep(c(.75, 1, 1.25), length.out = n), risk = exp(.2 * x[, 1]), strata = strata)
x2 <- matrix(rnorm(8L * p), 8L, p)
ordinary <- c(base, list(x2 = x2, risk2 = exp(.2 * x2[, 1])))
intervals <- min(1000L, n %/% 8L)
boundaries <- seq(0, n, length.out = intervals + 1L)
changing <- matrix(rnorm(intervals * p), intervals, p)
individual <- c(base, list(x2 = changing, risk2 = exp(.2 * changing[, 1]),
                           y2 = cbind(head(boundaries, -1L), tail(boundaries, -1L)),
                           strata2 = rep(1:4, length.out = intervals), id2 = rep("one", intervals)))
results <- list()
for (name in c("ordinary", "individual")) for (se in c(FALSE, TRUE)) {
  arguments <- c(if (name == "ordinary") ordinary else individual, list(se.fit = se))
  reference <- do.call(survival::coxsurv.fit, arguments)
  actual <- do.call(survivalr::coxsurv.fit, arguments)
  comparison <- all.equal(actual, reference, tolerance = 1e-10)
  if (!isTRUE(comparison)) stop(paste(comparison, collapse = "\n"))
  samples <- list()
  for (implementation in c("R_survival", "Rust_bridge")) {
    call <- if (implementation == "R_survival") survival::coxsurv.fit else survivalr::coxsurv.fit
    elapsed <- numeric(repeats)
    for (i in seq_len(repeats)) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- do.call(call, arguments)
      elapsed[i] <- 1000 * (proc.time()[["elapsed"]] - started)
      rm(value)
    }
    samples[[implementation]] <- list(median_ms = median(elapsed), samples_ms = I(elapsed))
  }
  results[[length(results) + 1L]] <- list(workload = name, se_fit = se,
                                       time_rows = length(actual$time), timings = samples)
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
                         survival_version = as.character(packageVersion("survival")),
                         n = n, columns = p, strata = 4L, new_rows = 8L,
                         intervals = intervals, repeats = repeats, results = results),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
