#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/survfit_frame_endpoint_reference.json"
stock_fit <- getFromNamespace("survfit.formula", "survival")
stock_summary <- getFromNamespace("summary.survfit", "survival")

encode <- function(values) {
  lapply(unname(values), function(value) {
    if (is.na(value)) return(NULL)
    if (is.numeric(value) && !is.finite(value)) return(if (value>0) "Inf" else "-Inf")
    value
  })
}
inputs <- list(
  finite=c(1,2,2,3,4,5),
  positive=c(1,2,2,3,4,Inf),
  negative=c(-Inf,2,2,3,4,5)
)
status <- c(1,0,1,0,1,1)
weights <- c(.5,1,2,.5,1,2)
cases <- list()
for (input in names(inputs)) for (grouped in c(FALSE,TRUE))
  for (robust in c(FALSE,TRUE)) for (stype in c(1,2)) {
    data <- list(time=inputs[[input]],status=status)
    if (grouped) data$group <- rep(c("a","b"),3)
    formula <- as.formula(if (grouped) "Surv(time,status) ~ group" else "Surv(time,status) ~ 1",
      env=asNamespace("survival"))
    arguments <- list(formula=formula,data=data,timefix=FALSE,stype=stype,
      ctype=stype,id=1:6,robust=robust)
    if (robust) arguments <- c(arguments,list(weights=weights,influence=3))
    fit <- do.call(stock_fit,arguments)
    frame <- stock_summary(fit,censored=TRUE,data.frame=TRUE,rmean="none")
    if (!is.null(frame$strata)) frame$strata <- as.character(frame$strata)
    cases[[length(cases)+1L]] <- list(
      name=paste(input,grouped,robust,stype,sep="/"),
      time=encode(data$time),status=encode(data$status),group=data$group,
      weights=if (robust) encode(weights) else NULL,robust=robust,stype=stype,
      logse=fit$logse,frame=lapply(frame,encode)
    )
  }
write_json(list(
  provenance=list(R=as.character(getRversion()),survival=as.character(packageVersion("survival")),
    reference="stock summary.survfit(censored=TRUE, data.frame=TRUE, rmean='none')"),
  cases=cases
),output,auto_unbox=TRUE,pretty=TRUE,digits=17,na="null",null="null")
