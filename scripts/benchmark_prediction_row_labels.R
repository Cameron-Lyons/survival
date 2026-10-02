#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "current"
include_transport <- length(args) > 3L && args[[4L]] == "transport"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(719)
data <- data.frame(time = rexp(2000), status = rbinom(2000, 1, .7))
newdata <- data.frame(row = seq_len(n))
for (j in seq_len(16)) {
  name <- paste0("x", j)
  data[[name]] <- rnorm(nrow(data))
  newdata[[name]] <- rnorm(n)
}
formula <- as.formula(paste("Surv(time,status)~", paste(paste0("x", seq_len(16)), collapse = "+")))
measure <- function(fun) {
  for (i in seq_len(3)) invisible(fun())
  samples <- vapply(seq_len(repeats), function(i) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(fun())
    1000 * (proc.time()[["elapsed"]] - start)
  }, numeric(1))
  list(median_ms = median(samples), range_ms = range(samples), samples_ms = I(samples))
}
transport <- if (include_transport && mode != "stock") {
  convert <- getFromNamespace(".as_python_data", "survivalr")
  list(columns = ncol(newdata), results = measure(function() convert(newdata)))
} else NULL
workloads <- list(cox_lp = list(kind = "coxph", type = "lp"),
                  cox_terms = list(kind = "coxph", type = "terms", terms = c("x9", "x1", "x9")),
                  aft_quantile = list(kind = "survreg", type = "quantile", p = .5),
                  aft_terms = list(kind = "survreg", type = "terms", terms = c("x9", "x1", "x9")))
results <- lapply(workloads, function(workload) {
  kind <- workload$kind
  workload$kind <- NULL
  reference <- do.call(get(kind, envir = asNamespace("survival")), list(formula, data))
  fit <- if (mode == "stock") reference else
    do.call(get(kind, envir = asNamespace("survivalr")), list(formula, data))
  call <- function(object) do.call(predict, c(list(object, newdata = newdata, se.fit = TRUE), workload))
  actual <- call(fit)
  expected <- call(reference)
  for (name in c("fit", "se.fit")) {
    stopifnot(identical(dim(actual[[name]]), dim(expected[[name]])),
              identical(colnames(actual[[name]]), colnames(expected[[name]])),
              isTRUE(all.equal(as.numeric(actual[[name]]), as.numeric(expected[[name]]), tolerance = 2e-7)))
    if (mode != "baseline") {
      stopifnot(identical(names(actual[[name]]), names(expected[[name]])),
                identical(dimnames(actual[[name]]), dimnames(expected[[name]])))
    }
  }
  measure(function() call(fit))
})
cat(jsonlite::toJSON(list(rows = n, columns = 16L, training_rows = 2000L,
  repeats = repeats, warmups = 3L, mode = mode,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  reticulate = if (mode != "stock") as.character(packageVersion("reticulate")) else NULL,
  transport = transport,
  scope = "Complete public new-data predictions with errors, including bridge conversion, formula design, calculation, output materialization and row/column metadata; fitting, input setup and explicit GC excluded. Baseline numerical outputs/shapes checked; current complete names also checked against stock R.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
