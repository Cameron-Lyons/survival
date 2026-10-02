#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "current"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(739)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7),
  flag = seq_len(n) %% 3L == 0L, g = factor(seq_len(n) %% 5L))
for (j in seq_len(16)) d[[paste0("x", j)]] <- rnorm(n)
formula <- as.formula(paste("Surv(time,status)~",
  paste(paste0("x", seq_len(16)), collapse = "+"), "+flag+strata(g)"))
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
    identical(as.integer(attr(actual, "assign")), as.integer(attr(expected, "assign"))),
    identical(attr(actual, "contrasts"), attr(expected, "contrasts")),
    identical(attr(actual, "strata"), attr(expected, "strata")),
    identical(is.na(actual), is.na(expected)), identical(is.nan(actual), is.nan(expected)),
    isTRUE(all.equal(as.numeric(actual), as.numeric(expected))))
}
results <- list()
for (kind in c("coxph", "survreg")) {
  options(na.action = "na.omit")
  reference <- get(kind, asNamespace("survival"))(formula, d, x = TRUE)
  fit <- if (mode == "stock") reference else get(kind, asNamespace("survivalr"))(formula, d, x = TRUE)
  nd <- d[seq_len(n %/% 2L), ]
  numeric_missing <- logical_missing <- nd
  numeric_missing$x1[seq(1, nrow(nd), 59L)] <- NA_real_
  logical_missing$flag[seq(1, nrow(nd), 59L)] <- NA
  inputs <- list(stored = NULL, complete_frame = nd, complete_list = as.list(nd),
    numeric_missing_list = as.list(numeric_missing), logical_missing_omit = as.list(logical_missing),
    logical_missing_pass = as.list(logical_missing))
  record <- list()
  for (workload in names(inputs)) {
    options(na.action = if (workload == "logical_missing_pass") "na.pass" else "na.omit")
    new <- inputs[[workload]]
    call <- function(object) if (workload == "stored") model.matrix(object) else model.matrix(object, new)
    actual <- call(fit); expected <- call(reference)
    if (mode == "predecessor" && startsWith(workload, "logical_missing")) {
      stopifnot(!identical(is.na(actual), is.na(expected)))
      record[[workload]] <- list(incompatible = "Lost logical NA values", actual_rows = nrow(actual),
        expected_rows = nrow(expected), actual_missing = sum(is.na(actual)), expected_missing = sum(is.na(expected)))
    } else {
      check(actual, expected)
      record[[workload]] <- measure(function() call(fit))
    }
  }
  results[[kind]] <- record
}
cat(jsonlite::toJSON(list(rows = n, new_rows = n %/% 2L, numeric_columns = 16L,
  logical_columns = 1L, repeats = repeats, warmups = 3L, mode = mode,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete public matrix calls including input conversion, omission, construction and metadata; fitting, input setup, option changes and explicit GC excluded. Incorrect predecessor logical-missing calls are verified but not timed.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
