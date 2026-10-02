#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "current"
suppressPackageStartupMessages(library(survival))
set.seed(741)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7),
  flag = seq_len(n) %% 3L == 0L, g = factor(seq_len(n) %% 5L),
  o = runif(n, 1, 3))
for (j in seq_len(16)) d[[paste0("x", j)]] <- runif(n, 1, 3)
formula <- as.formula(paste("Surv(time,status)~",
  paste(c(paste0("log(x", seq_len(8), ")"), paste0("sqrt(x", 9:16, ")")), collapse = "+"),
  "+flag+strata(g)+offset(log(o))"))
capture <- function(fun) {
  messages <- character()
  value <- withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, sub(" in (log|sqrt)\\(.*\\)$", "", conditionMessage(w)))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = messages)
}
measure <- function(fun) {
  for (i in seq_len(3)) invisible(capture(fun))
  samples <- vapply(seq_len(repeats), function(i) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(capture(fun))
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
nd <- d[seq_len(n %/% 2L), ]
missing <- overlap <- nd
indices <- seq(1, nrow(nd), 59L)
missing$x1[indices] <- NA_real_
overlap$x1[indices] <- -1
overlap$x2[indices] <- NA_real_
inputs <- list(stored = NULL, complete_frame = nd, complete_list = as.list(nd),
  missing_source_frame = missing, missing_source_list = as.list(missing),
  overlap_domain = overlap)
call <- function(object, workload) {
  if (workload == "stored") model.matrix(object) else model.matrix(object, inputs[[workload]])
}
# Capture stock outputs before loading the bridge's registered S3 methods.
references <- expected <- list()
for (kind in c("coxph", "survreg")) {
  options(na.action = "na.omit")
  references[[kind]] <- get(kind, asNamespace("survival"))(formula, d, x = TRUE, model = TRUE)
  expected[[kind]] <- lapply(setNames(names(inputs), names(inputs)), function(workload) {
    options(na.action = "na.omit")
    capture(function() call(references[[kind]], workload))
  })
}
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
results <- list()
for (kind in c("coxph", "survreg")) {
  options(na.action = "na.omit")
  fit <- if (mode == "stock") references[[kind]] else get(kind, asNamespace("survivalr"))(
    formula, d, x = TRUE, model = TRUE)
  record <- list()
  for (workload in names(inputs)) {
    options(na.action = "na.omit")
    actual <- capture(function() call(fit, workload))
    reference <- expected[[kind]][[workload]]
    check(actual$value, reference$value)
    if (mode == "predecessor" && workload == "overlap_domain") {
      stopifnot(length(actual$warnings) == 0L, identical(reference$warnings, "NaNs produced"))
      record[[workload]] <- list(incompatible = "Domain warning lost before omission; not timed",
        actual_rows = nrow(actual$value), expected_rows = nrow(reference$value))
    } else {
      stopifnot(identical(actual$warnings, reference$warnings))
      record[[workload]] <- measure(function() call(fit, workload))
    }
  }
  results[[kind]] <- record
}
cat(jsonlite::toJSON(list(rows = n, new_rows = n %/% 2L, transformed_columns = 16L,
  logical_columns = 1L, strata = 5L, repeats = repeats, warmups = 3L, mode = mode,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete public matrix calls including conversion, formula evaluation, omission, construction, metadata and warning capture; fitting, input setup, option changes and explicit GC excluded. Incorrect predecessor warning calls are verified but not timed.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
