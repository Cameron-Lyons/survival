#!/usr/bin/env Rscript
library(survival)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/cox_individual_reference.json"
data <- data.frame(
  start = c(0, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 3),
  stop = c(1, 2, 2, 4, 4, 6, 1, 2, 3, 4, 5, 6),
  event = c(1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0),
  x = c(.2, -.5, .7, -.2, .1, .4, -.6, .3, -.1, .8, -.3, .6),
  z = c(-.5, .3, .1, .2, -.4, .7, -.2, .6, -.8, .5, .4, -.1),
  g = rep(c("a", "b"), each = 6),
  weight = c(1, .5, 1.5, 2, 1, .75, 1.25, 1, 2, .5, 1.5, 1),
  offset = c(.1, 0, -.1, .2, 0, .1, -.2, 0, .1, 0, -.1, .2)
)
newdata <- data.frame(
  start = c(0, 1, 0, 2, 4, 3), stop = c(2, 3, 4, 6, 6, 6), event = 0,
  x = c(.1, -.2, .3, .4, -.5, .6), z = c(-.2, .1, .4, -.3, .2, .5),
  g = c("b", "a", "a", "a", "b", "b"),
  offset = c(.1, -.1, .2, 0, .1, -.2), id = c(20, -3, 7, 20, 7, -3)
)
formula <- Surv(start, stop, event) ~ x + z + strata(g) + offset(offset)
fit <- coxph(formula, data, weights = weight, ties = "efron", robust = FALSE)
cases <- list()
for (stype in 1:2) for (ctype in 1:2) for (se in c(FALSE, TRUE)) {
  curve <- survfit(fit, newdata = newdata, id = id, stype = stype, ctype = ctype, se.fit = se)
  fields <- c("n", "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz")
  if (se) fields <- c(fields, "std.err", "lower", "upper")
  expected <- lapply(unclass(curve)[fields], function(x) I(as.numeric(x)))
  expected$strata <- as.list(curve$strata)
  cases[[length(cases) + 1L]] <- list(stype = stype, ctype = ctype, se_fit = se, expected = expected)
}
columns <- function(x) lapply(x, I)
write_json(list(r_version = R.version.string,
                survival_version = as.character(packageVersion("survival")),
                data = columns(data), newdata = columns(newdata), cases = cases),
           output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null")
