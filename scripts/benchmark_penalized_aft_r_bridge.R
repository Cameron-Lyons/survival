#!/usr/bin/env Rscript
# Complete R matrix fits, including custom penalty calls and result conversion.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 100L, repeats >= 1L)
implementations <- list(R_survival = survival::survpenal.fit, Rust_bridge = survivalr::survpenal.fit)
set.seed(20260929)
p <- 6L
x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
colnames(x) <- c("Intercept", paste0("x", seq_len(p - 1L)))
eta <- drop(x %*% seq(.1, .6, length.out = p))
y <- cbind(eta + rnorm(n), as.integer(seq_len(n) %% 4L != 0L))
results <- list()
for (kind in c("fixed_ridge", "ridge_search", "spline_search")) {
  if (kind == "spline_search") {
    term <- survival::pspline(x[, 2], df = 3, nterm = 6)
    design <- cbind(Intercept = 1, term)
    colnames(design)[-1L] <- paste0("spline", seq_len(ncol(term)))
  } else {
    design <- x
    term <- if (kind == "fixed_ridge") survival::ridge(x[, -1L], theta = 1)
            else survival::ridge(x[, -1L], df = 3)
  }
  columns <- 2:ncol(design)
  arguments <- list(x = design, y = y, weights = NULL, offset = NULL, init = NULL,
                    controlvals = survival::survreg.control(), dist = "gaussian",
                    pcols = list(columns), pattr = list(attributes(term)),
                    assign = list(Intercept = 1L, penalty = columns))
  reference <- do.call(implementations$R_survival, arguments)
  samples <- list()
  for (name in names(implementations)) {
    call <- implementations[[name]]
    comparison <- all.equal(do.call(call, arguments), reference, tolerance = 4e-7, scale = 1)
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
  results[[length(results) + 1L]] <- list(penalty = kind, columns = ncol(design), timings = samples)
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
                         survival_version = as.character(packageVersion("survival")),
                         n = n, repeats = repeats, results = results),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
