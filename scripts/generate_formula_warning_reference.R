#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-logical-matrices.R")
source("r/survivalr/tests/testthat/helper-aft-matrix-inputs.R")
source("r/survivalr/tests/testthat/helper-matrix-na-action.R")
source("r/survivalr/tests/testthat/helper-formula-warnings.R")
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args))args[[1]]else"python/tests/fixtures/formula_warning_reference.json"
setup <- .formula_warning_setup(); cases <- fit_cases <- list()
for(kind in c("coxph","survreg"))for(rhs in setup$formulas)for(named in c(FALSE,TRUE)) {
  d <- .aft_matrix_input_data(setup,"complete",named)
  fit <- .logical_matrix_capture(function()get(kind,asNamespace("survival"))(
    as.formula(paste("Surv(futime,fustat)~",rhs)),d,x=TRUE,model=TRUE))
  if(is.list(fit$value)&&!is.null(fit$value$error))stop(fit$value$error)
  for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
    options(na.action=action)
    expected <- lapply(setNames(setup$inputs,setup$inputs),function(input) {
      result <- .logical_matrix_capture(function()model.matrix(fit$value,.formula_warning_newdata(d,input)))
      list(value=.matrix_na_encode(result$value),warnings=I(result$warnings))
    })
    cases[[length(cases)+1L]] <- list(name=paste(kind,rhs,named,action,sep="/"),kind=kind,rhs=rhs,
      named=named,action=action,fit_warnings=I(fit$warnings),expected=expected)
  }
  options(na.action="na.omit")
}
for(kind in c("coxph","survreg"))for(rhs in c("age + log(z)","age + sqrt(z)","age + sqrt(z) - sqrt(z)")) {
  for(variant in c("covariate","weights","subset","pass_unused")) {
    if(variant=="pass_unused"&&rhs!="age + sqrt(z) - sqrt(z)")next
    training <- .formula_warning_training(setup$data,variant)
    for(action in if(variant=="pass_unused")"na.pass"else c("na.omit","na.exclude","na.fail")) {
      result <- .logical_matrix_capture(function()do.call(get(kind,asNamespace("survival")),
        c(list(as.formula(paste("Surv(futime,fustat)~",rhs)),training$data,na.action=action),training$args)))
      value <- if(is.list(result$value)&&!is.null(result$value$error)).matrix_na_encode(result$value) else list(
        coef=.logical_matrix_encode(coef(result$value)),var=.logical_matrix_encode(vcov(result$value)),
        matrix=.matrix_na_encode(model.matrix(result$value)))
      fit_cases[[length(fit_cases)+1L]] <- list(name=paste(kind,rhs,variant,action,sep="/"),kind=kind,rhs=rhs,
        variant=variant,action=action,value=value,warnings=I(result$warnings))
    }
  }
}
options(na.action="na.omit")
serialize <- function(value)jsonlite::toJSON(value,auto_unbox=TRUE,digits=17,na="null",null="null")
header <- serialize(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival"))),
  data=lapply(setup$data,function(x)if(is.factor(x))as.character(x)else x),levels=list(g=levels(setup$data$g),h=levels(setup$data$h))))
writeLines(c(substring(header,1,nchar(header)-1),',"cases":[',
  vapply(seq_along(cases),function(i)paste0(serialize(cases[[i]]),if(i<length(cases))","else""),""),'],"fit_cases":[',
  vapply(seq_along(fit_cases),function(i)paste0(serialize(fit_cases[[i]]),if(i<length(fit_cases))","else""),""),"]}"),output,useBytes=TRUE)
cat(length(cases),"matrix option cases and",length(fit_cases),"training calls written\n")
