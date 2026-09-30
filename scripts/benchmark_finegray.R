#!/usr/bin/env Rscript
# Complete R formula calls, including model-frame and conversion costs.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[1L]) else 5000L
repeats <- if (length(args) > 1L) as.integer(args[2L]) else 7L
stopifnot(n > 10L, repeats > 0L)
set.seed(812)
d <- data.frame(time = sample(seq_len(500L), n, replace = TRUE),
  event = factor(sample(c("censor", "a", "b"), n, replace = TRUE), levels = c("censor", "a", "b")),
  x = rnorm(n), group = factor(seq_len(n) %% 100L))
subjects <- seq_len(n %/% 2L)
counting <- d[rep(subjects, each = 2L), ]
counting$id <- rep(subjects, each = 2L)
counting$start <- as.vector(rbind(0, d$time[subjects]/2))
counting$time <- as.vector(rbind(d$time[subjects]/2, d$time[subjects]))
counting$event[seq.int(1L, nrow(counting), by = 2L)] <- "censor"
cases <- list(
  right = list(formula = Surv(time, event) ~ x, data = d),
  strata = list(formula = Surv(time, event) ~ x + strata(group), data = d),
  counting = list(formula = Surv(start, time, event) ~ x, data = counting, id = counting$id),
  formula = list(formula = Surv(time, event) ~ poly(x, 3) + group, data = d)
)
result <- list()
for (kind in names(cases)) {
  call_args <- cases[[kind]]
  calls <- list(R_survival = function() do.call(survival::finegray, call_args),
    R_Rust = function() do.call(survivalr::finegray, call_args))
  expected <- calls$R_survival()
  actual <- calls$R_Rust()
  stopifnot(isTRUE(all.equal(actual, expected, tolerance = 1e-12)))
  rows <- nrow(actual)
  rm(actual, expected)
  for (warmup in seq_len(2L)) for (call in calls) invisible(call())
  elapsed <- lapply(calls, function(call) numeric(repeats))
  for (sample in seq_len(repeats)) {
    order <- if (sample %% 2L) names(calls) else rev(names(calls))
    for (name in order) {
      gc(FALSE)
      start <- proc.time()[["elapsed"]]
      value <- calls[[name]]()
      elapsed[[name]][sample] <- 1000 * (proc.time()[["elapsed"]] - start)
      rm(value)
    }
  }
  result[[kind]] <- list(input_rows = nrow(call_args$data), expanded_rows = rows, timings = lapply(elapsed, function(value) {
    list(median_ms = median(value), samples_ms = I(value))
  }))
}
cat(jsonlite::toJSON(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
  n = n, repeats = repeats, results = result), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
