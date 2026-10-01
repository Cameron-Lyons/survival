#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "installed"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(719)
d <- data.frame(time = rexp(n), status = rbinom(n, 1, .7))
for (j in seq_len(16)) d[[paste0("x", j)]] <- rnorm(n)
formula <- as.formula(paste("Surv(time,status) ~", paste(paste0("x", seq_len(16)), collapse = "+")))
reference <- survival::coxph(formula, d, x = TRUE)
fit <- if (mode == "stock") reference else survivalr::coxph(formula, d, x = TRUE)
group <- seq_len(n) %% 257L + 1L
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
calls <- list(
  lp_se = function(object) predict(object, type = "lp", se.fit = TRUE, collapse = group),
  terms_se = function(object) predict(object, type = "terms", se.fit = TRUE, collapse = group),
  dfbeta = function(object) residuals(object, type = "dfbeta", collapse = group)
)
results <- lapply(calls, function(fun) {
  actual <- fun(fit); expected <- fun(reference)
  strip <- function(x) if (is.list(x)) lapply(x, strip) else unname(x)
  stopifnot(isTRUE(all.equal(strip(actual), strip(expected), tolerance = 2e-7, check.attributes = FALSE)))
  measure(function() fun(fit))
})
cat(jsonlite::toJSON(list(rows = n, columns = 16L, groups = 257L, repeats = repeats, warmups = 3L,
  mode = mode, r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete public calls on a prefit model, including R/Python conversion, label encoding, predictions or residuals, aggregation, errors and result metadata; fitting and explicit GC excluded. Numeric groups give the predecessor the same group order.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
