#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "current"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(732)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7),
                g = factor(seq_len(n) %% 5L), h = factor(seq_len(n) %% 2L))
for (j in seq_len(16)) d[[paste0("x", j)]] <- rnorm(n)
formula <- as.formula(paste("Surv(time,status) ~", paste(paste0("x", seq_len(16)), collapse = "+"),
                            "+ strata(g, h)"))
reference <- survival::survreg(formula, d, x = TRUE)
fit <- if (mode == "stock") reference else survivalr::survreg(formula, d, x = TRUE)
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
check <- function(actual, expected) {
  stopifnot(identical(dim(actual), dim(expected)), identical(dimnames(actual), dimnames(expected)),
            identical(attr(actual, "assign"), as.integer(attr(expected, "assign"))),
            identical(attr(actual, "contrasts"), attr(expected, "contrasts")),
            isTRUE(all.equal(as.numeric(actual), as.numeric(expected))))
}
check(model.matrix(fit), reference$x)
nd <- d[seq_len(n %/% 2L), ]
check(model.matrix(fit, nd), model.matrix(reference, nd))
results <- list(stored = measure(function() model.matrix(fit)),
                new_with_groups = measure(function() model.matrix(fit, nd)))
nd[c("g", "h")] <- NULL
if (mode == "predecessor") {
  message <- tryCatch({model.matrix(fit, nd); NULL}, error=function(e) conditionMessage(e))
  stopifnot(!is.null(message), grepl("column 'g' not found", message, fixed=TRUE))
  results$new_without_groups <- list(error = "Required otherwise unused strata columns")
} else {
  check(model.matrix(fit, nd), model.matrix(reference, nd))
  results$new_without_groups <- measure(function() model.matrix(fit, nd))
}
cat(jsonlite::toJSON(list(rows = n, new_rows = n %/% 2L, columns = 17L, scales = 10L,
  repeats = repeats, warmups = 3L, mode = mode, r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")),
  scope = "Complete public model.matrix calls on a prefit AFT model, including R/Python conversion, matrix construction and metadata; fitting, input setup and explicit GC excluded. Predecessor rejects new data without standalone strata columns.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
