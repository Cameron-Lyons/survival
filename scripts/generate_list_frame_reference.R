#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-logical-matrices.R")
source("r/survivalr/tests/testthat/helper-aft-matrix-inputs.R")
source("r/survivalr/tests/testthat/helper-matrix-na-action.R")
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/list_frame_reference.json"
setup <- .matrix_na_setup()
d <- setup$data
cases <- list()
for (kind in c("coxph", "survreg")) for (rhs in c("1", "sqrt(z) - sqrt(z)", "strata(g)", "offset(off)")) {
  options(na.action = "na.omit")
  fit <- .logical_matrix_capture(function() get(kind, asNamespace("survival"))(
    as.formula(paste("Surv(futime,fustat)~", rhs)), as.list(d), x = TRUE))
  if (is.list(fit$value) && !is.null(fit$value$error)) stop(fit$value$error)
  nd <- d[c(7, 3, 7, 1), ]
  row.names(nd) <- NULL
  missing <- nd; missing$z[2] <- NA_real_; missing$g[3] <- NA; missing$off[4] <- NA_real_
  inputs <- list(list_complete = as.list(nd), list_missing = as.list(missing), list_empty = list(),
    frame_complete = nd, frame_missing = missing, frame_zero_columns = nd[FALSE], frame_empty = nd[FALSE, ])
  for (action in c("na.omit", "na.exclude", "na.pass", "na.fail")) {
    options(na.action = action)
    expected <- lapply(inputs, function(new) {
      result <- .logical_matrix_capture(function() model.matrix(fit$value, new))
      list(value = .matrix_na_encode(result$value), warnings = I(result$warnings))
    })
    cases[[length(cases) + 1L]] <- list(name = paste(kind, rhs, action, sep = "/"),
      kind = kind, rhs = rhs, action = action, fit_warnings = I(fit$warnings), expected = expected)
  }
}
options(na.action = "na.omit")
json <- jsonlite::toJSON(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival"))), data = lapply(d, function(x) if (is.factor(x)) as.character(x) else x),
  levels = list(g = levels(d$g), h = levels(d$h)), cases = cases),
  auto_unbox = TRUE, digits = 17, na = "null", null = "null")
writeLines(json, output, useBytes = TRUE)
cat(length(cases), "stock list/data-frame cases written\n")
