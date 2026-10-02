#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-logical-matrices.R")
source("r/survivalr/tests/testthat/helper-aft-matrix-inputs.R")
source("r/survivalr/tests/testthat/helper-matrix-na-action.R")
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1L]] else "python/tests/fixtures/matrix_na_action_reference.json"
setup <- .matrix_na_setup()
cases <- list()
for (kind in c("coxph","survreg")) for(rhs in setup$formulas) for(named in c(FALSE,TRUE)) {
  d <- .aft_matrix_input_data(setup,"complete",named)
  fit <- .logical_matrix_capture(function() get(kind,asNamespace("survival"))(
    as.formula(paste("Surv(futime,fustat)~",rhs)),d,x=TRUE,model=TRUE))
  if(is.list(fit$value) && !is.null(fit$value$error)) stop(fit$value$error)
  for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
    options(na.action=action)
    outputs <- list()
    for(input in setup$inputs) {
      nd <- .matrix_na_newdata(d,input)
      result <- .logical_matrix_capture(function() if(is.null(nd)) model.matrix(fit$value) else model.matrix(fit$value,nd))
      outputs[[input]] <- list(value=.matrix_na_encode(result$value),warnings=I(result$warnings))
    }
    cases[[length(cases)+1L]] <- list(name=paste(kind,rhs,named,action,sep="/"),
      kind=kind,rhs=rhs,named=named,action=action,fit_warnings=I(fit$warnings),expected=outputs)
  }
  options(na.action="na.omit")
}
serialize <- function(value) jsonlite::toJSON(value,auto_unbox=TRUE,digits=17,na="null",null="null")
header <- serialize(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival"))),
  data=lapply(setup$data,function(x)if(is.factor(x))as.character(x)else x),levels=list(g=levels(setup$data$g),h=levels(setup$data$h))))
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
  vapply(seq_along(cases),function(i)paste0(serialize(cases[[i]]),if(i<length(cases))","else""),""),"]}"),output,useBytes=TRUE)
cat(length(cases),"stock model-matrix NA-action cases written\n")
