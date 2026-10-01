#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 10000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "installed"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(731)
d <- data.frame(time = rexp(n), status = rbinom(n, 1, .7), group = seq_len(n) %% 100)
for (j in seq_len(16)) d[[paste0("x", j)]] <- rnorm(n)
formula <- as.formula(paste("Surv(time,status) ~", paste(paste0("x", seq_len(16)), collapse = "+"),
  "+ frailty(group, sparse=TRUE, theta=.4)"))
reference <- survival::coxph(formula, d, x = TRUE)
fit <- if (mode == "stock") reference else survivalr::coxph(formula, d, x = TRUE)
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
stopifnot(isTRUE(all.equal(unname(model.matrix(fit)), unname(reference$x), check.attributes = FALSE)))
results <- list(stored_sparse = measure(function() model.matrix(fit)))
if (mode != "baseline") {
  nd <- d[seq_len(n %/% 2L), ]; nd$group <- 1000 + nd$group
  stopifnot(isTRUE(all.equal(unname(model.matrix(fit, data = nd)),
    unname(model.matrix(reference, nd)), check.attributes = FALSE)))
  results$new_sparse <- measure(function() model.matrix(fit, data = nd))
}
cat(jsonlite::toJSON(list(rows = n, dense_columns = 16L, repeats = repeats, warmups = 3L,
  mode = mode, r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete public model.matrix calls on a prefit model, including R/Python conversion and metadata; fitting and explicit GC excluded. Baseline new-data path omitted because it drops the sparse column.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
