#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/surv_vector_reference.json"
time <- c(1,2,1,NA,NA,0,-0)
status <- c(1,0,1,1,1,0,0)
states <- factor(c("a","censor","a",NA,NA,"b","b"), levels=c("censor","b","a","unused"))
responses <- suppressWarnings(list(
    right=Surv(time,status),left=Surv(time,status,type="left"),
    counting=Surv(c(0,1,0,1,1,-2,-2),time+3,status),
    interval=Surv(c(1,2,1,NA,NA,0,-0),c(3,4,3,1,1,2,2),c(3,2,3,0,0,1,1),type="interval"),
    mright=Surv(time,states),mcounting=Surv(c(0,1,0,1,1,-2,-2),time+3,states)))
snapshot <- function(x) list(type=attr(x,"type"), states=attr(x,"states"), clabel=attr(x,"clabel"), matrix=unname(as.matrix(x)))
operations <- list(
    list(name="rep_twice",method="rep",args=list(times=2)),
    list(name="rep_counts",method="rep",args=list(times=c(1,0,2,0,1,1,0))),
    list(name="rep_each_times",method="rep",args=list(each=2,times=2)),
    list(name="rep_each_counts",method="rep",args=list(each=2,times=rep(c(0,1),7))),
    list(name="rep_length",method="rep",args=list(each=2,length.out=5,times=NA)),
    list(name="rep_fractional",method="rep",args=list(each=1.8,times=2.8)),
    list(name="rep_zero",method="rep",args=list(times=0)),
    list(name="reverse",method="rev",args=list()),
    list(name="unique",method="unique",args=list()),
    list(name="unique_last",method="unique",args=list(fromLast=TRUE)),
    list(name="duplicated",method="duplicated",args=list()),
    list(name="duplicated_last",method="duplicated",args=list(fromLast=TRUE)),
    list(name="transpose",method="t",args=list()),
    list(name="levels",method="levels",args=list()),
    list(name="character",method="as.character",args=list()),
    list(name="format",method="format",args=list()),
    list(name="head",method="head",args=list(n=2)),
    list(name="head_negative",method="head",args=list(n=-2)),
    list(name="head_fractional",method="head",args=list(n=-1.8)),
    list(name="tail",method="tail",args=list(n=2)),
    list(name="tail_negative",method="tail",args=list(n=-2)),
    list(name="tail_fractional",method="tail",args=list(n=-1.8)),
    list(name="head_empty",method="head",args=list(n=0)),
    list(name="tail_empty",method="tail",args=list(n=-20)))
cases <- lapply(names(responses),function(name) {
    x <- responses[[name]]
    results <- lapply(operations,function(operation) {
        value <- do.call(operation$method,c(list(x),operation$args))
        if(inherits(value,"Surv")) snapshot(value)
        else if(is.null(value)) NULL
        else if(is.matrix(value)) unname(value)
        else I(value)
    })
    names(results) <- vapply(operations,`[[`,"", "name")
    list(name=name,response=snapshot(x),results=results,concat=snapshot(c(x[c(3,1)],x[c(7,2)])))
})
reference <- list(metadata=list(generator="scripts/generate_surv_vector_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    operations=operations,cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases)*length(operations),"Surv vector operations written to",output,"\n")
