#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-logical-matrices.R")
source("r/survivalr/tests/testthat/helper-aft-matrix-inputs.R")
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/aft_matrix_input_reference.json"
setup <- .aft_matrix_input_setup()
cases <- list()
for (rhs in setup$formulas) for (variant in c("complete", "missing", "subset"))
  for (named in c(FALSE, TRUE)) for (cached in c(FALSE, TRUE)) {
    data <- .aft_matrix_input_data(setup, variant, named)
    fit <- .logical_matrix_capture(function() .aft_matrix_input_fit(rhs, data, variant, cached, "survival"))
    outputs <- list()
    if (is.list(fit$value) && !is.null(fit$value$error)) stop(fit$value$error)
    for (input in setup$inputs) {
      nd <- .aft_matrix_input_newdata(data, input)
      result <- .logical_matrix_capture(function() if (is.null(nd)) model.matrix(fit$value) else model.matrix(fit$value, nd))
      outputs[[input]] <- list(value = .logical_matrix_encode(result$value), warnings = I(result$warnings))
    }
    cases[[length(cases)+1L]] <- list(name = paste(rhs,variant,named,cached,sep="/"),
      rhs = rhs, variant = variant, named = named, cached = cached,
      fit_warnings = I(fit$warnings), expected = outputs)
  }
serialize <- function(value) jsonlite::toJSON(value, auto_unbox=TRUE, digits=17, na="null", null="null")
header <- serialize(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival"))),
  data=lapply(setup$data,function(x) if(is.factor(x)) as.character(x) else x),
  levels=list(g=levels(setup$data$g),h=levels(setup$data$h))))
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
  vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),if(i<length(cases)) "," else ""),""),
  "]}"),output,useBytes=TRUE)
cat(length(cases), "stock AFT input cases written\n")
