#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/population_summary_reference.json"
make_case <- function(name,time,surv,nrisk,times=NULL,scale=1,omitted=FALSE) {
    surv <- as.matrix(surv)
    nrisk <- as.matrix(nrisk)
    fit <- structure(list(time=time,surv=surv,n.risk=nrisk,method="cohort"),class="survexp")
    result <- if (omitted) summary(fit,scale=scale) else summary(fit,times=times,scale=scale)
    rows <- function(x) matrix(as.numeric(x),nrow=length(result$time),ncol=ncol(surv))
    number <- function(x) if (is.infinite(x)) as.character(x) else x
    list(name=name,time=I(time),surv=surv,n_risk=nrisk,times=lapply(times,number),
         scale=number(scale),omitted=omitted,
         expected=list(time=lapply(result$time,number),surv=rows(result$surv),n_risk=rows(result$n.risk)))
}
native_cases <- list(
    make_case("all",c(1,3,5),c(.9,.8,.6),c(3,2,1),omitted=TRUE),
    make_case("requested",c(1,3,5),c(.9,.8,.6),c(3,2,1),c(9,0,2,2,1,5,-1,NA,Inf,-Inf),2),
    make_case("matrix",c(1,3,5),cbind(c(.9,.8,.6),c(.8,.7,.5)),cbind(c(3,2,1),c(8,5,2)),c(5,0,2,2,3)),
    make_case("singleton",3,.7,4,c(0,1,3,4)),
    make_case("repeated_source",c(1,1,1,3,3,5),c(.99,.95,.9,.85,.8,.6),6:1,c(0,1,2,3,4,5)),
    make_case("negative_source",c(-3,-1,2),c(.9,.8,.6),c(3,2,1),c(-4,-3,-2,-1,0,1,2,3)),
    make_case("empty_request",c(1,3,5),c(.9,.8,.6),c(3,2,1),numeric()),
    make_case("missing_values",c(1,3,5),c(.9,NA,.6),c(3,NA,1),c(0,1,2,3,4,5)),
    make_case("negative_scale",c(1,3,5),c(.9,.8,.6),c(3,2,1),scale=-2,omitted=TRUE),
    make_case("zero_scale",c(0,3,5),c(1,.8,.6),c(3,2,1),scale=0,omitted=TRUE),
    make_case("infinite_scale",c(1,3,5),c(.9,.8,.6),c(3,2,1),scale=Inf,omitted=TRUE),
    make_case("missing_scale",c(1,3,5),c(.9,.8,.6),c(3,2,1),scale=NA_real_,omitted=TRUE))
data <- data.frame(time=c(100,200,400,800,NA),status=c(1,0,1,0,1),
    age=c(40,50,60,70,80)*365.25,sex=c(1,2,1,2,1),
    year=as.Date(rep("2000-01-01",5)),grp=c("a","b","a","b","a"))
fit_cases <- lapply(c("ederer","hakulinen","conditional"), function(method) {
    fit <- survexp(Surv(time,status)~grp,data,rmap=list(age=age,sex=sex,year=year),
                   method=method,times=c(50,200,400,800),na.action=na.omit)
    result <- summary(fit,times=c(0,100,100,200,750,900),scale=10)
    list(method=method,time=result$time,surv=unname(result$surv),n_risk=unname(result$n.risk),
         labels=colnames(result$surv),na_action=as.integer(result$na.action))
})
tables <- lapply(list(us=survexp.us,usr=survexp.usr,mn=survexp.mn),function(table) {
    list(text=paste0(paste(capture.output(summary(table)),collapse="\n"),"\n"),
         dims=dim(table),dimid=names(dimnames(table)),types=attr(table,"type"),
         dimnames=unname(dimnames(table)))
})
base <- data.frame(id=1:3, stop=c(5,7,4), status=c(1,0,1))
merged <- tmerge(base,base,id=id,tstop=stop,death=event(stop,status))
updates <- data.frame(id=c(1,1,2,2,3,4,1,2),time=c(-1,2,0,7,6,2,2,8),value=seq_len(8))
merged <- tmerge(merged,updates,id=id,dose=tdc(time,value))
tmerge_case <- list(base=base,updates=updates,terms=rownames(attr(merged,"tcount")),
                    columns=colnames(attr(merged,"tcount")),counts=unname(attr(merged,"tcount")))
reference <- list(metadata=list(generator="scripts/generate_population_summary_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    native_cases=native_cases,data=data,fit_cases=fit_cases,tables=tables,tmerge=tmerge_case)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(native_cases),"curve summary cases and population/data summaries written to",output,"\n")
