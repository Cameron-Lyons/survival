#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/response_normalization_reference.json"
base <- data.frame(time = 1:8, status = rep(c(1, 2), 4), x = rep(c(1, NA), 4),
                   start = rep(0, 8), end = 2:9, w = rep(1, 8))
specs <- list(
  censored_only = list(data = base, formula = "Surv(time, status) ~ x"),
  offset_omission = list(data = base, formula = "Surv(time, status) ~ offset(x)"),
  invalid_status = list(data = transform(base, status = c(0, 1, 8, 1, 0, 1, 0, 1), x = 1:8),
                        formula = "Surv(time, status) ~ x"),
  mixed_codes = list(data = transform(base, status = c(0, 1, 2, 2, 1, 2, 1, 2), x = 1:8),
                     formula = "Surv(time, status) ~ x"),
  logical = list(data = transform(base, status = c(TRUE, NA, FALSE, TRUE, FALSE, NA, TRUE, FALSE), x = 1:8),
                 formula = "Surv(time, status) ~ x"),
  left = list(data = base, formula = "Surv(time, status, type='left') ~ x"),
  backwards = list(data = transform(base, start = c(0, 2, 4, 0, 0, 0, 0, 0), x = 1:8),
                    formula = "Surv(start, time, status) ~ x"),
  arithmetic = list(data = transform(base, x = c(0, 1, 2, 3, 0, 1, 2, 3)),
                    formula = "Surv((time*x)/x, status) ~ x"),
  interval = list(data = transform(base, status = c(0, 1, 2, 3, 0, 1, 2, 3),
                                    end = c(NA, NA, NA, 5, NA, NA, NA, 9), x = 1:8),
                  formula = "Surv(time, end, status, type='interval') ~ x"),
  interval_rhs = list(data = transform(base, status = c(0, 1, 2, 3, 0, 1, 2, 3),
                                        end = c(NA, NA, NA, 5, NA, NA, NA, 9), x = 1:8),
                      formula = "Surv(time, end, status, type='interval') ~ end"),
  interval_invalid = list(data = transform(base, status = c(0, 1, 2, 3, 8, 3, 2, 1),
                                            end = c(NA, NA, NA, 2, 6, NA, 8, NA), x = 1:8),
                          formula = "Surv(time, end, status, type='interval') ~ x"),
  interval2 = list(data = transform(base, time = c(1, NA, 3, 4, 5, NA, NA, 8),
                                     end = c(NA, 2, 3, 5, 4, NA, 7, 9), x = 1:8),
                   formula = "Surv(time, end, type='interval2') ~ x"),
  interval2_rhs = list(data = transform(base, time = c(1, NA, 3, 4, 5, NA, NA, 8),
                                         end = c(NA, 2, 3, 5, 4, NA, 7, 9), x = 1:8),
                       formula = "Surv(time, end, type='interval2') ~ time"),
  weighted = list(data = transform(base, x = 1:8, w = rep(c(1, NA), 4)),
                   formula = "Surv(time, status) ~ x"),
  multistate = list(data = transform(base, status = factor(rep(c('c', 'event'), 4),
                                                          levels = c('c', 'other', 'event'))),
                     formula = "Surv(time, status) ~ x"))
cases <- list()
selections <- list(all = NULL, reverse = 8:1, repeated = c(1, 4, 3, 5, 8, 1))
for (name in names(specs)) {
  spec <- specs[[name]]
  d <- spec$data
  for (selection in names(selections)) for (action in c("na.omit", "na.exclude", "na.pass", "na.fail")) {
    selected <- selections[[selection]]
    warnings <- character()
    mf <- tryCatch(withCallingHandlers(
      model.frame(as.formula(spec$formula), d, subset = selected, na.action = get(action), weights = w),
      warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
      error = function(e) list(error = conditionMessage(e)))
    result <- if (!is.null(mf$error)) mf else {
      y <- model.response(mf)
      list(response = unname(unclass(y)), type = attr(y, "type"),
           states = I(as.character(attr(y, "states"))), omitted = I(as.integer(attr(mf, "na.action"))),
           weights = I(model.weights(mf)))
    }
    cases[[length(cases) + 1L]] <- list(name = paste(name, selection, action, sep = "/"),
      formula = spec$formula, data = d, status_levels = if (is.factor(d$status)) I(levels(d$status)) else NULL,
      subset = if (is.null(selected)) NULL else I(selected - 1L), action = action,
      warnings = I(warnings), expected = result)
  }
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()), survival = as.character(packageVersion("survival"))),
                          cases = cases), output, auto_unbox = TRUE, pretty = TRUE,
                     digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "response normalization cases written\n")
