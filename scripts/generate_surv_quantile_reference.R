#!/usr/bin/env Rscript
# Rscript scripts/generate_surv_quantile_reference.R [output.json]
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/surv_quantile_reference.json"
probs <- c(.75, 0, .5, .25, 1, .5)
responses <- list(
    right=Surv(c(1,2,2,3,4,6,7,8), c(1,0,1,1,0,1,0,1)),
    counting=Surv(c(0,0,1,1,2,2,3,3), c(1,2,3,4,5,6,7,8), c(1,0,1,1,0,1,0,1)),
    left=Surv(c(1,2,2,3,4,6,7,8), c(0,1,1,0,1,1,0,1), type="left"),
    interval=Surv(c(1,1,2,3,4,5,6,7), c(1,2,3,3,Inf,6,8,7), type="interval2"),
    all_censored=Surv(c(1,3,4), c(0,0,0)),
    singleton=Surv(3),
    negative=Surv(c(-3,-2,-1,0,1,2),c(1,1,0,1,0,1)),
    near_ties=Surv(c(1,1+1e-12,2,2+1e-12,3,4),c(1,1,0,1,1,1)),
    missing=Surv(c(1,2,3,NA,4),c(1,0,1,1,NA)))
cases <- lapply(names(responses), function(name) {
    y <- responses[[name]]
    list(name=name, type=attr(y,"type"), response=unname(as.matrix(y)),
         quantile=quantile(y, probs, na.rm=TRUE),
         median=median(y, na.rm=TRUE),
         plain=quantile(y, probs, na.rm=TRUE, conf.int=FALSE, scale=2))
})
# Confidence bands can be non-monotone; approx() sorts their probabilities.
# Also cover duplicate levels, NAs, and ties created by adding tolerance.
findq_cases <- list(
    list(name="nonmonotone", x=0:6, y=c(0,.05,.3,.15,.5,.6,.8),
         p=c(.1,.2,.3,.4,.5,.6,.8,.9), tol=sqrt(.Machine$double.eps)),
    list(name="duplicate_nonmonotone", x=0:8, y=c(0,.3,.6,.3,NA,.4,.6,.8,.9),
         p=c(0,.3,.4,.6,.7,.9,1), tol=sqrt(.Machine$double.eps)),
    list(name="rounded_ties", x=c(0,10,20,30,40), y=c(0,.5,1,.5+.Machine$double.eps/2,.8),
         p=c(0,.5,1), tol=.5),
    list(name="missing_monotone", x=0:7, y=c(0,.2,NA,.2,.4,.6,.6,NA),
         p=c(0,.1,.2,.5,.6,.7), tol=sqrt(.Machine$double.eps)),
    list(name="end_flat", x=0:7, y=c(0,.2,.2,.5,.5,.7,.7,.7),
         p=c(.2,.5,.7,1), tol=sqrt(.Machine$double.eps)),
    list(name="all_censored", x=0:4, y=rep(0,5), p=c(0,.5,1), tol=sqrt(.Machine$double.eps)))
findq_cases <- lapply(findq_cases, function(case) {
    case$expected <- do.call(survival:::findq, case[c("x","y","p","tol")])
    case
})
data <- data.frame(time=1:18, status=rep(c(1,1,0),6), x=rep(c(-1,0,1),6), g=rep(0:1,9))
cox <- coxph(Surv(time,status)~x+strata(g),data)
curves <- survfit(cox,newdata=data.frame(x=c(-.5,.7)),start.time=2)
km <- survfit(Surv(time,status)~g,data)
fit_cases <- list(
    km=list(quantile=unname(quantile(km,probs,conf.int=FALSE)),median=unname(median(km))),
    cox=list(quantile=unname(matrix(quantile(curves,probs,conf.int=FALSE),ncol=length(probs))),
             median=unname(matrix(median(curves),ncol=1))))
reference <- list(metadata=list(generator="scripts/generate_surv_quantile_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    probs=probs,cases=cases,findq=findq_cases,data=data,fit_cases=fit_cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",dataframe="columns")
cat(length(cases),"response and",length(findq_cases),"curve-quantile cases written to",output,"\n")
