#!/usr/bin/env Rscript
library(survival)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/curve_labels_reference.json"
matrix_values <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
table_values <- function(x) list(rows = I(rownames(x)), columns = I(colnames(x)), values = matrix_values(x))
cases <- list()
for (label in c("censor", "lost", "not yet", "gone (A)")) {
  for (counting in c(FALSE, TRUE)) {
    data <- data.frame(time = 1:6, start = c(0, .5, 1, 0, 3, 2), id = 1:6,
                       event = factor(c(label, "ill", "death", label, "ill", "death"),
                                      levels = c(label, "ill", "death")),
                       group = c("b", "b", "a", "a", "b", "a"), x = c(.2, .8, .3, .7, .1, .9))
    response <- if (counting) "Surv(start,time,event)" else "Surv(time,event)"
    curve <- survfit(as.formula(paste(response, "~group")), data = data, id = id)
    check <- survcheck(as.formula(paste(response, "~1")), data = data, id = id)
    cox <- coxph(as.formula(paste(response, "~x")), data = data, id = id, iter.max = 0)
    prediction <- survfit(cox, newdata = data.frame(x = c(.25, .75)))
    cases[[length(cases) + 1L]] <- list(
      label = label, counting = counting,
      time = I(data$time), start = I(data$start), id = I(data$id),
      event = I(as.character(data$event)), levels = I(levels(data$event)),
      group = I(data$group), x = I(data$x),
      curve = list(time = I(curve$time), pstate = matrix_values(curve$pstate),
                   transitions = table_values(curve$transitions)),
      check = table_values(check$transitions),
      cox = table_values(cox$transitions),
      prediction = table_values(prediction$transitions)
    )
  }
}
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
                cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null")
