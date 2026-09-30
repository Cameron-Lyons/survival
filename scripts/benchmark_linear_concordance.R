#!/usr/bin/env Rscript
# Complete concordance calls; external model fitting and input setup are excluded.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 50000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n >= 30L, repeats > 0L)
set.seed(913)
data <- data.frame(x = rnorm(n), z = runif(n), w = runif(n, .5, 2))
data$y <- 2 + data$x/3 + data$z/2 + rnorm(n)
first <- lm(y ~ x, data)
second <- lm(y ~ x + z, data)
weighted <- lm(y ~ x + z, data, weights = w)
reference <- get("concordance.lm", asNamespace("survival"))
cases <- list(
  single = list(stock_R = function() reference(second), shared = function() concordance(second)),
  weighted = list(stock_R = function() reference(weighted), shared = function() concordance(weighted)),
  newdata = list(stock_R = function() reference(weighted, newdata = data),
                 shared = function() concordance(weighted, newdata = data)),
  joint = list(stock_R = function() reference(first, second),
               shared = function() concordance(first, second)))
results <- list()
for (name in names(cases)) {
  calls <- cases[[name]]
  expected <- calls$stock_R()
  actual <- calls$shared()
  actual$call <- expected$call
  comparison <- all.equal(actual, expected, tolerance = 1e-10)
  if (!isTRUE(comparison)) stop(name, ": ", paste(comparison, collapse = "; "))
  for (call in calls) for (i in 1:3) call()
  timings <- lapply(calls, function(call) numeric())
  for (i in seq_len(repeats)) {
    order <- if (i %% 2) names(calls) else rev(names(calls))
    for (kind in order) {
      gc(FALSE)
      start <- proc.time()[["elapsed"]]
      value <- calls[[kind]]()
      timings[[kind]] <- c(timings[[kind]], 1000 * (proc.time()[["elapsed"]] - start))
      rm(value)
    }
  }
  results[[name]] <- lapply(timings, function(values) list(
    median_ms = median(values), range_ms = range(values), samples_ms = I(values)))
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L,
  r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
  scope = "Complete R-facing calls; model fitting and input construction excluded", results = results),
  auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
