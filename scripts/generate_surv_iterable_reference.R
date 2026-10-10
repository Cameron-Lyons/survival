#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/surv_iterable_reference.json"
times <- c(1,2,NA,4,Inf,6)
binary <- c(0,1,NA,1,0,1)
codes12 <- c(1,2,NA,2,1,2)
logical <- c(FALSE,TRUE,NA,TRUE,FALSE,NA)
states <- factor(c("censor","event",NA,"NA","event","censor"),
  levels=c("censor","NA","unused","event"))
numeric_states <- factor(c(10,2,NA,30,2,10),levels=c(30,10,2,99))
spec <- function(name,arguments,options=list(),constructor="Surv",unused=integer())
  list(name=name,arguments=arguments,options=options,constructor=constructor,unused=unused)
cases <- list(
  spec("right_binary",list(times,binary)),
  spec("right_codes12",list(times,codes12)),
  spec("right_logical",list(times,logical)),
  spec("right_all_missing",list(times,rep(NA,6))),
  spec("right_invalid",list(times,c(0,1,NA,8,2,0))),
  spec("right_mixed_codes",list(times,c(0,1,NA,2,1,2))),
  spec("right_origin",list(times,binary),list(origin=2)),
  spec("left_codes12",list(times,codes12),list(type="left")),
  spec("counting_binary",list(c(0,1,0,2,3,4),times,binary)),
  spec("counting_logical",list(c(0,1,0,2,3,4),times,logical)),
  spec("counting_backwards",list(c(1,1,0,5,3,4),times,codes12)),
  spec("counting_origin",list(c(0,1,0,2,3,4),times,binary),list(origin=.5)),
  spec("interval",list(times,c(2,4,5,6,9,8),c(0,1,2,3,NA,3)),list(type="interval")),
  spec("interval_invalid",list(times,c(2,4,5,2,9,8),c(0,1,2,3,8,3)),list(type="interval")),
  spec("interval_unused_right",list(times,c("ignored","right"),binary),
    list(type="interval"),unused=2L),
  spec("interval2",list(c(1,NA,3,4,-Inf,NA),c(NA,2,3,5,6,NA)),list(type="interval2")),
  spec("mright",list(times,states)),
  spec("mcounting",list(c(0,1,0,2,3,4),times,states)),
  spec("numeric_mright",list(times,numeric_states)),
  spec("numeric_mcounting",list(c(0,1,0,2,3,4),times,numeric_states)),
  spec("timeline_binary",list(times,binary),constructor="Surv2"),
  spec("timeline_logical",list(times,logical),list(repeated=TRUE),"Surv2"),
  spec("timeline_factor",list(times,states),list(repeated="first"),"Surv2"),
  spec("numeric_timeline",list(times,numeric_states),list(repeated=TRUE),"Surv2"),
  spec("empty_right",list(numeric(),logical())),
  spec("time_only",list(times)),
  spec("empty_factor",list(numeric(),factor(character(),levels=c("censor","event","unused")))),
  spec("empty_timeline",list(numeric(),numeric()),constructor="Surv2"),
  spec("short_event",list(1:3,c(0,1))),
  spec("long_event",list(1:3,c(0,1,0,1))),
  spec("short_counting_event",list(0:2,1:3,c(0,1))),
  spec("short_timeline_event",list(1:3,c(0,1)),constructor="Surv2"),
  spec("invalid_character_event",list(1:3,c("a","b","a"))))
encode <- function(value) {
  if (is.factor(value)) return(list(values=I(as.character(value)),levels=I(levels(value)),kind="factor"))
  values <- lapply(value,function(item) {
    if (is.numeric(item) && is.nan(item)) return("NaN")
    if (is.na(item)) return(NULL)
    if (is.numeric(item) && !is.finite(item)) return(if (item>0) "Inf" else "-Inf")
    item
  })
  list(values=values,levels=NULL,kind=typeof(value))
}
`%||%` <- function(x,y) if (is.null(x)) y else x
snapshot <- function(value) {
  matrix <- unname(as.matrix(value))
  rows <- lapply(seq_len(nrow(matrix)),function(row) {
    lapply(seq_len(ncol(matrix)),function(column) {
      item <- matrix[row,column]
      if (is.na(item)) return(NULL)
      if (!is.finite(item)) return(if (item>0) "Inf" else "-Inf")
      item
    })
  })
  list(type=attr(value,"type"),states=I(attr(value,"states") %||% character()),
    clabel=attr(value,"clabel"),repeated=attr(value,"repeated"),
    ncol=ncol(matrix),matrix=rows)
}
results <- lapply(cases,function(case) {
  warnings <- character()
  result <- withCallingHandlers(
    tryCatch(list(response=snapshot(do.call(case$constructor,c(case$arguments,case$options)))),
      error=function(error) list(error=conditionMessage(error))),
    warning=function(warning) {
      warnings <<- c(warnings,conditionMessage(warning))
      invokeRestart("muffleWarning")
    })
  c(list(name=case$name,constructor=case$constructor,arguments=lapply(case$arguments,encode),
    options=case$options,unused=I(case$unused-1L),warnings=I(warnings)),result)
})
fit_times <- c(1,2,2,4,5,6)
fit_status <- c(1,2,1,2,1,2)
fit <- survfit(Surv(fit_times,fit_status) ~ 1,conf.type="none")
fit_reference <- list(time=I(fit_times),event=I(fit_status),
  result=list(n=I(fit$n),time=I(fit$time),n_risk=I(fit$n.risk),
    n_event=I(fit$n.event),n_censor=I(fit$n.censor),surv=I(fit$surv)))
reference <- list(metadata=list(generator="scripts/generate_surv_iterable_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
  cases=results,kaplan_meier=fit_reference)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(results),"Surv/Surv2 input cases written to",output,"\n")
