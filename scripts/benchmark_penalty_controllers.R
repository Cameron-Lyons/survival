#!/usr/bin/env Rscript
# Compare complete shared AFT fits changing only the penalty controller callback.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
pkgload::load_all("r/survivalr", quiet = TRUE)
n <- 3000L
i <- seq_len(n)
age <- sin(i / 17)
z <- cos(i / 23)
group <- (i %% 40L) + 1L
y <- cbind(2 + .2 * age + .3 * sin(group) + sin(i * 1.7), as.numeric(i %% 5L != 0L))
results <- list()
for (kind in c("ridge", "gaussian")) {
  terms <- if (kind == "ridge") list(
    shared = survivalr::ridge(cbind(age, z), df = 1.3),
    reference = survival::ridge(cbind(age, z), df = 1.3)
  ) else list(
    shared = survivalr::frailty.gaussian(group, sparse = TRUE),
    reference = survival::frailty.gaussian(group, sparse = TRUE)
  )
  reference_controller <- attr(terms$reference, "cfun")
  terms$reference <- terms$shared
  attr(terms$reference, "cfun") <- reference_controller
  x <- cbind(Intercept = 1, terms$shared)
  fit <- function(term) survivalr::survpenal.fit(
    x, y, weights = NULL, offset = NULL, init = NULL,
    controlvals = survival::survreg.control(), dist = "gaussian",
    pcols = list(2:ncol(x)), pattr = list(attributes(term)),
    assign = list(Intercept = 1, term = 2:ncol(x)))
  stopifnot(isTRUE(all.equal(fit(terms$shared), fit(terms$reference), tolerance = 1e-6)))
  for (warmup in 1:2) for (term in terms) invisible(fit(term))
  samples <- list(shared = numeric(), reference = numeric())
  for (sample in 1:7) for (name in if (sample %% 2L) names(terms) else rev(names(terms))) {
    gc()
    samples[[name]] <- c(samples[[name]], system.time(invisible(fit(terms[[name]])))[["elapsed"]] * 1000)
  }
  results[[kind]] <- lapply(samples, function(x) list(median_ms = median(x), samples_ms = I(x)))
}
cat(toJSON(list(rows = n, results = results), auto_unbox = TRUE, pretty = TRUE), "\n")
