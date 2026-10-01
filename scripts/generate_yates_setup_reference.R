#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/yates_setup_reference.json"
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
pack <- function(x) if (is.matrix(x)) rows(x) else I(as.numeric(x))
d <- data.frame(time = c(1, 2, 2, 3, 4, 4, 5, 6), status = c(1, 1, 1, 0, 1, 1, 0, 1),
                start = c(0, .3, 0, 1, 1.5, .5, 2, 3),
                x = c(.2, -.5, .7, 1.2, -.3, .5, -.8, .1), wt = c(1, .5, 1.2, .7, 1.1, 1, .8, 1))
cases <- list()
for (kind in c("right", "counting", "null", "single", "empty")) for (ties in c("breslow", "efron")) {
  data <- d
  formula <- if (kind == "counting") "Surv(start, time, status) ~ x" else "Surv(time, status) ~ x"
  if (kind %in% c("null", "single", "empty")) formula <- "Surv(time, status) ~ 1"
  if (kind == "single") { data <- d[1:3, ]; data$time <- 2; data$status <- c(1, 1, 0) }
  if (kind == "empty") data$status <- 0
  fit <- coxph(as.formula(formula), data, weights = wt, ties = ties)
  baseline <- survfit(fit, censor = FALSE)
  for (horizon in list(NULL, 0, 2.5, Inf)) for (nlevel in c(1L, 3L)) {
    setup <- suppressWarnings(yates_setup(fit, predict = "survival", options = list(rmean = horizon)))
    eta <- c(-1, 0, .6)[seq_len(nlevel)]
    predicted <- setup$predict(eta)
    variance <- matrix(.0001 * seq_along(predicted), nrow = nlevel)
    raw <- setup$summary(predicted, variance)
    survival <- t(predicted[, -c(1L, 2L), drop = FALSE])
    std <- t(sqrt(variance[, -c(1L, 2L), drop = FALSE]))
    cumhaz <- -log(survival)
    z <- -qnorm((1 - baseline$conf.int) / 2)
    expected <- list(surv = survival, cumhaz = cumhaz, std_err = std / survival,
                     lower = exp(-(cumhaz + z * std)), upper = exp(-(cumhaz - z * std)))
    cases[[length(cases) + 1L]] <- list(
      name = paste(kind, ties, if (is.null(horizon)) "default" else horizon, nlevel, sep = "/"),
      formula = formula, ties = ties, data = lapply(data, I),
      rmean = if (is.null(horizon)) NULL else if (is.infinite(horizon)) "Inf" else horizon,
      eta = I(eta), prediction = rows(predicted), variance = rows(variance),
      time = I(baseline$time), baseline_cumhaz = I(as.numeric(baseline$cumhaz)),
      raw_summary = lapply(unclass(raw)[c("surv", "cumhaz", "std.err", "lower", "upper")], pack),
      corrected_summary = lapply(expected, rows))
  }
}
links <- list(identity = gaussian(), log = poisson(), logit = binomial(),
              probit = binomial("probit"), cloglog = binomial("cloglog"),
              cauchit = binomial("cauchit"), inverse = Gamma("inverse"))
eta <- c(-3, -1, -.1, .1, 1, 3)
glm <- lapply(names(links), function(name) {
  fit <- structure(list(family = links[[name]]), class = "glm")
  inverse <- yates_setup(fit, predict = "response")
  list(link = name, eta = I(eta), expected = I(inverse(eta, NULL)))
})
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
                cases = cases, glm = glm), output,
           auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null")
