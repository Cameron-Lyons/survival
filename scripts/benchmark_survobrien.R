#!/usr/bin/env Rscript
# Complete O'Brien expansion calls, with stock R's documented strata typos fixed.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 2000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
stopifnot(n > 0L, repeats > 0L)
set.seed(712)
i <- seq_len(n)
d <- data.frame(time = runif(n, 1, 1000), status = as.integer(i %% 3 == 0),
                x = round(rnorm(n), 1), z = rnorm(n), group = factor(i %% 100))
counting <- transform(d, start = i, time = i + runif(n, 1, 6))
reference <- survival::survobrien
body <- paste(deparse(body(reference), width.cutoff = 500L), collapse = "\n")
fixes <- c(
  "y[, 2] >= temp[x, 1] & strata.keep == temp[x, 2]" = "y[, 1] >= temp[x, 1] & strata.keep == temp[x, 2]",
  "!strata.keep == temp[x, 2]" = "strata.keep == temp[x, 2]",
  "names(m)[stemp$vars]" = "stemp$vars")
for (from in names(fixes)) {
  stopifnot(grepl(from, body, fixed = TRUE))
  body <- sub(from, fixes[[from]], body, fixed = TRUE)
}
body(reference) <- str2lang(body)
cases <- list(
  right = list(formula = Surv(time, status) ~ x+z, data = d),
  stratified = list(formula = Surv(time, status) ~ x+z+strata(group), data = d),
  counting = list(formula = Surv(start, time, status) ~ x+z, data = counting),
  custom = list(formula = Surv(time, status) ~ x+z, data = d, transform = function(x) x-mean(x))
)
results <- list()
for (kind in names(cases)) {
  call_args <- cases[[kind]]
  calls <- list(R_survival = function() do.call(reference, call_args),
                R_Rust = function() do.call(survivalr::survobrien, call_args))
  expected <- calls$R_survival()
  actual <- calls$R_Rust()
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-12)))
  expanded_rows <- nrow(actual)
  risk_sets <- length(unique(actual$.strata.))
  rm(actual, expected)
  for (warmup in seq_len(2L)) for (call in calls) invisible(call())
  elapsed <- lapply(calls, function(call) numeric(repeats))
  for (sample in seq_len(repeats)) {
    order <- if (sample %% 2L) names(calls) else rev(names(calls))
    for (name in order) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- calls[[name]]()
      elapsed[[name]][sample] <- 1000 * (proc.time()[["elapsed"]] - started)
      rm(value)
    }
  }
  results[[kind]] <- list(expanded_rows = expanded_rows, risk_sets = risk_sets,
    timing = lapply(elapsed, function(values) list(median_ms = median(values), samples_ms = I(values))))
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")), observations = n,
  continuous_columns = 2L, repeats = repeats, results = results),
  auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
