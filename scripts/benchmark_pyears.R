#!/usr/bin/env Rscript
# Complete calls, including formula preparation, rate conversion and result assembly.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[1L]) else 100000L
repeats <- if (length(args) > 1L) as.integer(args[2L]) else 7L
stopifnot(n > 0L, repeats > 0L)
set.seed(915)
d <- data.frame(time = runif(n, 1, 1000), event = sample(0:1, n, replace = TRUE),
  start = runif(n, 0, 1), age = runif(n, 40, 80)*365.25,
  year = as.Date("2000-01-01") + sample(0:730, n, replace = TRUE),
  sex = sample(1:2, n, replace = TRUE), group = factor(seq_len(n) %% 10L))
cases <- list(
  fixed = list(formula = Surv(time, event) ~ group + sex, data = d),
  counting = list(formula = Surv(start, time, event) ~ group + sex, data = d),
  tcut = list(formula = Surv(time, event) ~ group + tcut(age, c(0, 50, 60, 70, 100)*365.25), data = d),
  rate = list(formula = Surv(time, event) ~ group, data = d, ratetable = survival::survexp.us)
)
results <- list()
for (kind in names(cases)) {
  call_args <- cases[[kind]]
  calls <- list(R_survival = function() do.call(survival::pyears, call_args),
    R_Rust = function() do.call(survivalr::pyears, call_args))
  expected <- calls$R_survival()
  actual <- calls$R_Rust()
  fields <- intersect(c("pyears", "n", "event", "expected", "offtable"), names(expected))
  stopifnot(isTRUE(all.equal(actual[fields], expected[fields], tolerance = 1e-12)))
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
  results[[kind]] <- lapply(elapsed, function(value) list(median_ms = median(value), samples_ms = I(value)))
}
cat(jsonlite::toJSON(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
  observations = n, repeats = repeats, results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
