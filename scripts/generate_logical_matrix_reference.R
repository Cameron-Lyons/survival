#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-logical-matrices.R")
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/logical_matrix_reference.json"
setup <- .logical_matrix_setup()
cases <- list()
for (kind in c("coxph", "survreg")) for (rhs in setup$formulas)
  for (variant in c("mixed", "false", "true", "missing", "subset")) for (named in c(FALSE, TRUE)) {
    data <- .logical_matrix_data(setup, variant, named)
    fit <- .logical_matrix_capture(function() .logical_matrix_fit(kind, rhs, data, variant, "survival"))
    outputs <- list()
    if (is.list(fit$value) && !is.null(fit$value$error)) outputs$error <- fit$value$error else {
      outputs$coefficients <- .logical_matrix_encode(coef(fit$value))
      for (input in c("stored", "complete", "partial", "all_missing", "empty", "single")) {
        nd <- .logical_matrix_newdata(data, input)
        result <- .logical_matrix_capture(function() if (is.null(nd)) model.matrix(fit$value) else model.matrix(fit$value, nd))
        outputs[[input]] <- list(value = .logical_matrix_encode(result$value), warnings = I(result$warnings))
      }
    }
    cases[[length(cases)+1L]] <- list(name = paste(kind,rhs,variant,named,sep="/"),
      kind = kind, rhs = rhs, variant = variant, named = named,
      fit_warnings = I(fit$warnings), expected = outputs)
  }
serialize <- function(value) jsonlite::toJSON(value, auto_unbox=TRUE, digits=17, na="null", null="null")
header <- serialize(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival"))),
  data=lapply(setup$data,function(x) if(is.factor(x)) as.character(x) else x), levels=levels(setup$data$g)))
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
  vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),if(i<length(cases)) "," else ""),""),
  "]}"),output,useBytes=TRUE)
cat(length(cases),"stock logical/model-matrix references written\n")
