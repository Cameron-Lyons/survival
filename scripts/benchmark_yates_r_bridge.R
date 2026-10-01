#!/usr/bin/env Rscript
# Complete R-facing Yates calls, including formula preparation and conversion.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 5000L
nsim <- if (length(args) > 1L) as.integer(args[[2L]]) else 200L
repeats <- if (length(args) > 2L) as.integer(args[[3L]]) else 7L
stopifnot(n > 0L, nsim >= 2L, repeats > 0L)
set.seed(123)
d <- data.frame(y = rbinom(180, 1, .5), time = rep(1:30, 6),
                status = rbinom(180, 1, .75), a = factor(rep(c("a", "b", "c"), 60)),
                b = factor(rep(c("x", "y"), 90)), z = rnorm(180))
fits <- list(
  linear = lm(time ~ a*b+z, d),
  response = glm(y ~ a*b+z, family = binomial(), data = d),
  risk = survival::coxph(survival::Surv(time, status) ~ a*b+z, d)
)
fits$survival <- fits$risk
population <- d[rep(seq_len(nrow(d)), length.out = n), c("b", "z")]
reference <- function(fit, ...) {
  # survival::yates evaluates this name in its caller's frame.
  yates_setup <- if (inherits(fit, "coxph")) survival:::yates_setup.coxph else if (inherits(fit, "glm")) survival:::yates_setup.glm else survival:::yates_setup.default
  survival::yates(fit, ...)
}
results <- list()
for (kind in names(fits)) {
  fit <- fits[[kind]]
  calls <- list(
    R_survival = function() {
      set.seed(123)
      reference(fit, "a", population = population, predict = kind, nsim = nsim)
    },
    R_Rust = function() {
      set.seed(123)
      survivalr::yates(fit, "a", population = population, predict = kind, nsim = nsim)
    }
  )
  expected <- calls$R_survival()
  state <- .Random.seed
  actual <- calls$R_Rust()
  stopifnot(identical(.Random.seed, state))
  for (field in c("estimate", "mvar", "test")) {
    stopifnot(isTRUE(all.equal(actual[[field]], expected[[field]], tolerance = 3e-8)))
  }
  if (kind == "survival") {
    stopifnot(isTRUE(all.equal(actual$summary$surv,
      unname(expected$summary$surv[-1L, , drop = FALSE]), tolerance = 3e-8)))
  }
  # R's JIT can compile a closure after its first call; warm every path three times.
  for (warmup in seq_len(2L)) for (call in calls) invisible(call())
  elapsed <- lapply(calls, function(call) numeric(repeats))
  for (i in seq_len(repeats)) {
    order <- if (i %% 2L) names(calls) else rev(names(calls))
    for (name in order) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- calls[[name]]()
      elapsed[[name]][i] <- 1000 * (proc.time()[["elapsed"]] - started)
      rm(value)
    }
  }
  results[[kind]] <- lapply(elapsed, function(values) {
    list(median_ms = median(values), samples_ms = I(values))
  })
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
  survival_version = as.character(packageVersion("survival")), population_rows = n,
  levels = 3L, nsim = nsim, repeats = repeats, results = results),
  auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
