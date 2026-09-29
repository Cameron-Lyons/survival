#!/usr/bin/env Rscript
# Complete prepared-matrix R calls, including conversion and result assembly.
# Optionally pass an older bridge.R as the third argument for a before/after run.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 100L, repeats >= 1L)
implementations <- list(R_survival = survival::survreg.fit, Rust_bridge = survivalr::survreg.fit)
if (length(args) > 2L) {
  previous <- new.env(parent = asNamespace("survivalr"))
  sys.source(args[[3L]], envir = previous)
  implementations$Previous_bridge <- previous$survreg.fit
}
set.seed(20260929)
p <- 6L
x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
colnames(x) <- c("Intercept", paste0("x", seq_len(p - 1L)))
eta <- drop(x %*% seq(.1, .6, length.out = p))
offset <- sin(seq_len(n)) / 10
y <- cbind(eta + offset + rnorm(n), as.integer(seq_len(n) %% 4L != 0L))
custom <- survival::survreg.distributions$gaussian
custom$name <- "Benchmark Gaussian callback"
results <- list()
for (kind in c("gaussian", "logistic", "custom")) {
  arguments <- list(x = x, y = y, weights = rep(c(.75, 1, 1.25), length.out = n),
                    offset = offset, init = NULL, controlvals = survival::survreg.control(),
                    dist = if (kind == "custom") custom else kind)
  reference <- do.call(implementations$R_survival, arguments)
  samples <- list()
  for (name in names(implementations)) {
    call <- implementations[[name]]
    # Scores can be nearly zero at convergence; compare with an absolute
    # tolerance instead of dividing rounding noise by those small values.
    comparison <- all.equal(do.call(call, arguments), reference, tolerance = 2e-7, scale = 1)
    if (!isTRUE(comparison)) stop(paste(kind, name, paste(comparison, collapse = "\n")))
    elapsed <- numeric(repeats)
    for (i in seq_len(repeats)) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- do.call(call, arguments)
      elapsed[i] <- 1000 * (proc.time()[["elapsed"]] - started)
      rm(value)
    }
    samples[[name]] <- list(median_ms = median(elapsed), samples_ms = I(elapsed))
  }
  results[[length(results) + 1L]] <- list(distribution = kind, timings = samples)
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
                         survival_version = as.character(packageVersion("survival")),
                         n = n, columns = p, repeats = repeats, results = results),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
