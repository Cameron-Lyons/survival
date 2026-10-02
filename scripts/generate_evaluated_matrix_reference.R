#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
for (file in c("helper-logical-matrices.R", "helper-aft-matrix-inputs.R",
    "helper-matrix-na-action.R", "helper-evaluated-matrices.R")) {
  source(file.path("r/survivalr/tests/testthat", file))
}
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/evaluated_matrix_reference.json"
setup <- .evaluated_matrix_setup()
cases <- list()
for (kind in c("coxph", "survreg")) for (rhs in setup$formulas) {
  if (kind == "survreg" && grepl("frailty", rhs)) next
  for (named in c(FALSE, TRUE)) for (cached in c(FALSE, TRUE)) {
    options(na.action = "na.omit")
    d <- .aft_matrix_input_data(setup, "complete", named)
    fit <- .logical_matrix_capture(function() get(kind, asNamespace("survival"))(
      as.formula(paste("Surv(futime,fustat)~", rhs)), d, x = cached, model = TRUE))
    if (is.list(fit$value) && !is.null(fit$value$error)) stop(fit$value$error)
    inputs <- lapply(setNames(setup$inputs, setup$inputs), function(input)
      .evaluated_matrix_newdata(model.frame(fit$value), input))
    expected <- lapply(setNames(c("na.omit", "na.exclude", "na.pass", "na.fail"),
      c("na.omit", "na.exclude", "na.pass", "na.fail")), function(action) {
      options(na.action = action)
      lapply(inputs, function(input) {
        result <- .logical_matrix_capture(function() model.matrix(fit$value, input))
        list(value = .matrix_na_encode(result$value), warnings = I(result$warnings))
      })
    })
    references <- list()
    actions <- lapply(expected, function(value) {
      match <- which(vapply(references, identical, TRUE, value))
      if (length(match)) return(match[[1L]])
      references[[length(references) + 1L]] <<- value
      length(references)
    })
    cases[[length(cases) + 1L]] <- list(name = paste(kind, rhs, named, cached, sep = "/"),
      kind = kind, rhs = rhs, named = named, cached = cached, fit_warnings = I(fit$warnings),
      inputs = lapply(inputs, .evaluated_matrix_encode_frame), actions = actions, references = references)
  }
}
options(na.action = "na.omit")
serialize <- function(value) jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17, na = "null", null = "null")
header <- serialize(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival"))),
  data = lapply(setup$data, function(x) if (is.factor(x)) as.character(x) else x),
  levels = list(g = levels(setup$data$g), h = levels(setup$data$h))))
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
  vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),
    if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes = TRUE)
cat(length(cases), "stock evaluated-frame cases written\n")
