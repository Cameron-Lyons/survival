#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/detailed_report_reference.json"

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
fits <- list(
  km=survfit(Surv(time,status)~1,d),
  groups=survfit(Surv(time,status)~group,d),
  weighted=survfit(Surv(time,status)~group,d,weights=weights),
  delayed=survfit(Surv(start,time,status)~group,d,entry=TRUE,id=seq_len(nrow(d))),
  ci90=survfit(Surv(time,status)~1,d,conf.int=.9),
  no_se=survfit(Surv(time,status)~1,d,se.fit=FALSE),
  no_ci=survfit(Surv(time,status)~1,d,conf.type="none"),
  cox=survfit(coxph(Surv(time,status)~age,z),newdata=data.frame(age=c(2,7))),
  cox_one=survfit(coxph(Surv(time,status)~age,z),newdata=data.frame(age=2)),
  cox_groups=survfit(coxph(Surv(time,status)~age+strata(group),z),newdata=data.frame(age=c(2,7))),
  cox_conditional=survfit(coxph(Surv(time,status)~age,z),newdata=data.frame(age=2),start.time=5.5),
  aj=survfit(Surv(time,endpoint)~1,d),
  aj_groups=survfit(Surv(time,endpoint)~group,d),
  aj_conditional=survfit(Surv(time,endpoint)~1,d,start.time=2),
  coxms=survfit(coxph(Surv(time,endpoint)~age,z,id=seq_len(nrow(z))),newdata=data.frame(age=c(2,7))),
  coxms_groups=survfit(coxph(Surv(time,endpoint)~age+strata(group),z,id=seq_len(nrow(z))),newdata=data.frame(age=c(2,7))),
  interval=survfit(Surv(c(1,2,3,4,5),c(1,3,Inf,6,5),type="interval2")~1),
  censored=survfit(Surv(1:4,rep(0,4))~1),
  empty_group=survfit(Surv(c(1,2,3,5),c(1,0,0,0))~factor(c('a','a','b','b'))))
fits$aj_one <- fits$aj[,2]

p <- data.frame(time=c(100,300,500,800,1200,1500),age=c(50,60,70,80,55,65)*365.25,
                sex=c(1,2,1,2,1,2),year=as.Date("2000-01-01"))
expected <- list(
  single=survexp(~1,p,rmap=list(age=age,sex=sex,year=year),times=c(100,300,600,1000)),
  groups=survexp(~sex,p,rmap=list(age=age,sex=sex,year=year),times=c(100,300,600,1000)),
  one_time=survexp(~sex,p,rmap=list(age=age,sex=sex,year=year),times=100),
  cox=survexp(~sex,lung,ratetable=coxph(Surv(time,status)~age+sex,lung),times=c(100,300,600,1000)))
expected$missing <- expected$groups
expected$missing$surv[1,] <- NA
expected$missing$surv[2,1] <- NA
expected$missing$n.risk[3,1] <- NA
expected$single_missing <- expected$single
expected$single_missing$surv[2] <- NA
expected$all_missing <- expected$single
expected$all_missing$surv[] <- NA
expected$one_column <- expected$single
expected$one_column$surv <- matrix(expected$single$surv,ncol=1,dimnames=list(NULL,"reference"))
expected$one_column$n.risk <- matrix(expected$single$n.risk,ncol=1)

encode <- function(x) if(is.null(x)) NULL else list(values=I(as.numeric(x)),dim=if(is.null(dim(x))) NULL else I(dim(x)))
snapshot <- function(x,method) {
  fields <- c("time","n.risk","n.event","n.censor","n.enter","surv","pstate","std.err","lower","upper")
  values <- lapply(fields,function(name) encode(x[[name]]));names(values)<-gsub("\\.","_",fields)
  list(fields=values,states=if(is.null(x$states)) NULL else I(x$states),
       strata=if(is.null(x$strata)) NULL else I(as.character(x$strata)),
       strata_levels=if(is.null(x$strata)) NULL else I(levels(x$strata)),
       names=if(is.null(colnames(x$surv))) NULL else I(colnames(x$surv)),
       type=x$type,start_time=x$start.time,conf_int=x$conf.int,method=x$method,
       coxms=inherits(x,"summary.survfitms") && length(dim(x$pstate))==3)
}
capture <- function(x,method,arguments,width) {
  saved<-options(width=width,digits=7);on.exit(options(saved))
  x$call<-NULL;x$na.action<-NULL;x$summ<-NULL
  f<-get(paste0("print.",method),asNamespace("survival"))
  environment(f)<-env<-new.env(parent=environment(f))
  tables<-list()
  env$print<-function(x,...) {
    values<-if(is.matrix(x)) unname(x) else matrix(unname(x),nrow=1)
    columns<-if(is.matrix(x)) colnames(x) else names(x)
    tables[[length(tables)+1L]]<<-list(values=values,columns=I(columns))
    base::print(x,...)
  }
  error<-NULL
  lines<-capture.output(tryCatch(do.call(f,c(list(x=x),arguments)),error=function(e) error<<-conditionMessage(e)))
  list(tables=tables,lines=I(sub("[[:blank:]]+$","",lines)),error=error)
}
cases<-list()
add<-function(name,x,method,arguments=list(),width=80) {
  cases[[length(cases)+1L]]<<-list(name=name,input=snapshot(x,method),method=method,
                                  arguments=arguments,width=width,expected=capture(x,method,arguments,width))
}
for(key in names(fits)) {
  x<-summary(fits[[key]])
  method<-if(inherits(x,"summary.survfitms")) "summary.survfitms" else "summary.survfit"
  add(key,x,method)
}
for(key in c("km","groups","delayed","cox","aj_groups","coxms")) {
  x<-summary(fits[[key]],times=c(0,2,4,8,30),extend=TRUE)
  method<-if(inherits(x,"summary.survfitms")) "summary.survfitms" else "summary.survfit"
  add(paste0(key,"_times"),x,method)
}
for(key in c("groups","aj_groups","cox_groups","coxms_groups")) {
  x<-summary(fits[[key]],times=2)
  method<-if(inherits(x,"summary.survfitms")) "summary.survfitms" else "summary.survfit"
  add(paste0(key,"_one_row"),x,method)
}
add("censor_rows",summary(fits$censored,censored=TRUE),"summary.survfit")
add("all_rows",summary(fits$delayed,censored=TRUE),"summary.survfit")
add("scaled",summary(fits$groups,scale=365.25),"summary.survfit")
add("narrow",summary(fits$groups),"summary.survfit",list(digits=5),width=35)
add("narrow_vector",summary(fits$groups,times=2),"summary.survfit",list(digits=5),width=35)
add("aj_precision",summary(fits$aj),"summary.survfitms",list(digits=7))
add("conditional_before_start",summary(fits$cox_conditional,times=c(0,5.5,10)),"summary.survfit")
add("aj_before_start",summary(fits$aj_conditional,times=c(0,1,2,6)),"summary.survfitms")
for(key in names(expected)) {
  add(paste0("expected_",key),expected[[key]],"survexp")
  add(paste0("expected_summary_",key),summary(expected[[key]]),"summary.survexp")
}
add("expected_naprint",expected$missing,"survexp",list(naprint=TRUE))
add("expected_scaled",expected$groups,"survexp",list(scale=365.25,digits=6),width=40)
add("expected_requested",summary(expected$single,times=c(0,50,200,700,1200)),"summary.survexp")
add("expected_empty",summary(expected$single,times=2000),"summary.survexp")

write_json(list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),cases=cases),
           output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"detailed report cases written to",output,"\n")
