#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/survfit_print_reference.json"

d <- data.frame(time=c(1,2,2,3,4,5,6,6,8),status=c(1,1,0,1,0,1,0,1,0),
                group=factor(c("a","a","a","b","a","b","b","b","a")),
                start=c(0,0,1,0,1,2,0,3,1),weights=c(1,2,.5,1,2,1,.5,1,1))
d$endpoint <- factor(c("disease","death","censor","disease","censor",
                        "death","censor","disease","censor"),
                     levels=c("censor","disease","death"))
z <- data.frame(time=(seq_len(40)*7)%%29+1,status=as.integer(seq_len(40)%%3!=0),
                age=(seq_len(40)*3)%%11,group=factor(seq_len(40)%%2))
z$endpoint <- factor(ifelse(z$status==0,"censor",ifelse(seq_len(40)%%4==0,"a","b")),
                     levels=c("censor","a","b"))
data <- list(small=d,cox=z,interval=data.frame(time=c(1,2,3,4,5),time2=c(1,3,NA,6,5)),
             complete=data.frame(time=1:4,status=rep(1,4)),
             censored=data.frame(time=1:4,status=rep(0,4)))
definitions <- list()
fits <- list()
fit <- function(name,formula,data="small",arguments=list(),kind="km",curve=list(),origin=FALSE) {
  d <- get("data",envir=.GlobalEnv)[[data]]
  model <- do.call(if(kind=="km") survfit else coxph,c(list(formula=as.formula(formula),data=d),arguments))
  if(kind=="cox") model <- do.call(survfit,c(list(formula=model),curve))
  if(origin) model <- survfit0(model)
  definitions[[name]] <<- list(formula=formula,data=data,arguments=arguments,kind=kind,curve=curve,origin=origin)
  model$call <- NULL
  fits[[name]] <<- model
}
fit("km","Surv(time,status)~1")
fit("groups","Surv(time,status)~group")
fit("weighted","Surv(time,status)~group",arguments=list(weights=I(d$weights)))
fit("delayed","Surv(start,time,status)~group")
fit("ids","Surv(start,time,status)~group",arguments=list(id=I(seq_len(nrow(d)))))
fit("conditional","Surv(time,status)~1",arguments=list(start.time=2))
fit("no_se","Surv(time,status)~1",arguments=list(se.fit=FALSE))
fit("ci90","Surv(time,status)~group",arguments=list(conf.int=.90,conf.type="plain"))
fit("interval","Surv(time,time2,type='interval2')~1",data="interval")
fit("complete","Surv(time,status)~1",data="complete")
fit("censored","Surv(time,status)~1",data="censored")
fit("cox","Surv(time,status)~age",data="cox",kind="cox",curve=list(newdata=list(age=I(c(2,7)))))
fit("cox_groups","Surv(time,status)~age+strata(group)",data="cox",kind="cox",
    curve=list(newdata=list(age=I(c(2,7)))))
fit("cox_conditional","Surv(time,status)~age",data="cox",kind="cox",
    curve=list(newdata=list(age=I(c(2,7))),start.time=5.5))
fit("aj","Surv(time,endpoint)~1")
fit("aj_groups","Surv(time,endpoint)~group")
fit("aj_nose","Surv(time,endpoint)~group",arguments=list(se.fit=FALSE))
fit("aj_conditional","Surv(time,endpoint)~1",arguments=list(start.time=2))
fit("coxms","Surv(time,endpoint)~age",data="cox",kind="cox",arguments=list(id=I(seq_len(nrow(z)))),
    curve=list(newdata=list(age=I(c(2,7))),se.fit=FALSE))
fit("km_origin","Surv(time,status)~1",origin=TRUE)
fit("aj_origin","Surv(time,endpoint)~group",origin=TRUE)

capture <- function(x,arguments,width) {
  saved <- options(width=width,digits=7,survfit.rmean=NULL,survfit.print.rmean=NULL)
  on.exit(options(saved))
  f <- getS3method("print",if(inherits(x,"survfitms")) "survfitms" else "survfit")
  environment(f) <- env <- new.env(parent=environment(f))
  table <- NULL
  env$print <- function(x,...) {
    table <<- list(values=unname(x),columns=I(colnames(x)),
                   rows=if(is.null(rownames(x))) NULL else I(as.vector(rownames(x))))
    base::print(x,...)
  }
  lines <- capture.output(do.call(f,c(list(x=x),arguments)))
  list(table=table,lines=I(sub("[[:blank:]]+$","",lines)))
}
cases <- list()
add <- function(name,key,arguments=list(),width=80) {
  cases[[length(cases)+1L]] <<- list(name=name,fit=key,arguments=arguments,width=width,
                                    expected=capture(fits[[key]],arguments,width))
}
for(key in names(fits)) add(key,key)
for(key in c("km","groups","weighted","delayed","ids","interval","cox","cox_groups","censored")) {
  add(paste0(key,"_mean"),key,list(rmean="common"))
}
for(key in c("groups","delayed","aj_groups","cox","coxms")) {
  add(paste0(key,"_individual"),key,list(rmean="individual"))
  add(paste0(key,"_cutoff"),key,list(rmean=5,scale=2))
}
add("legacy_mean","km",list(print.rmean=TRUE))
add("explicit_none","km",list(print.rmean=TRUE,rmean="none"))
add("aj_none","aj",list(rmean="none"))
add("coxms_none","coxms",list(rmean="none"))
add("partial_mean","groups",list(rmean="ind"))
for(digits in c(1,2,5,10)) add(paste0("precision",digits),"groups",list(digits=digits,rmean="common"))
add("narrow","groups",list(rmean="common"),width=30)
add("aj_narrow","aj_groups",list(digits=3),width=22)
add("time_scale","groups",list(scale=365.25,rmean="common"))
add("scientific_scale","groups",list(scale=1e10,rmean="common"))
add("wide_precision","groups",list(digits=10,rmean="common"),width=45)
add("aj_zero_cutoff","aj",list(rmean=0))
add("aj_early_cutoff","aj_groups",list(rmean=.5))
add("coxms_early_cutoff","coxms",list(rmean=.5))
add("cox_conditional_early_cutoff","cox_conditional",list(rmean=5.5))

encode <- function(x) {
  lapply(x,function(column) if(is.factor(column)) list(values=I(as.character(column)),levels=I(levels(column))) else I(column))
}
write_json(list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),
                data=lapply(data,encode),fits=definitions,cases=cases),output,
           auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"curve report cases written to",output,"\n")
