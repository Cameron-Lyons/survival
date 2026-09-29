#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/survfit_plot_reference.json"

d <- data.frame(time=c(1,2,2,3,4,5,6,6,8), status=c(1,1,0,1,0,1,0,1,0),
                group=factor(c("a","a","a","b","a","b","b","b","a")),
                start=c(0,0,1,0,1,2,0,3,1), weights=c(1,2,.5,1,2,1,.5,1,1))
d$endpoint <- factor(c("disease","death","censor","disease","censor",
                        "death","censor","disease","censor"),
                     levels=c("censor","disease","death"))
z <- data.frame(time=(seq_len(40)*7)%%29+1, status=as.integer(seq_len(40)%%3!=0),
                age=(seq_len(40)*3)%%11, group=factor(seq_len(40)%%2))
z$endpoint <- factor(ifelse(z$status==0,"censor",ifelse(seq_len(40)%%4==0,"a","b")),
                     levels=c("censor","a","b"))
fits <- list(
  km=survfit(Surv(time,status)~1,d),
  groups=survfit(Surv(time,status)~group,d),
  weighted=survfit(Surv(time,status)~1,d,weights=weights),
  delayed=survfit(Surv(start,time,status)~group,d),
  conditional=survfit(Surv(time,status)~1,d,start.time=2),
  cox=survfit(coxph(Surv(time,status)~age+strata(group),z),newdata=data.frame(age=c(2,7))),
  aj=survfit(Surv(time,endpoint)~1,d),
  aj_groups=survfit(Surv(time,endpoint)~group,d),
  coxms=survfit(coxph(Surv(time,endpoint)~age,z,id=seq_len(nrow(z))),newdata=data.frame(age=c(2,7))),
  interval=survfit(Surv(c(1,2,3,4,5),c(1,3,Inf,6,5),type="interval2")~1),
  no_se=survfit(Surv(time,status)~1,d,se.fit=FALSE),
  terminal=survfit(Surv(c(1,2,3,4),c(1,1,1,1))~1))

numeric_fields <- c("time","surv","cumhaz","pstate","p0","n","n.risk","n.event",
                    "n.censor","std.err","std.chaz","lower","upper")
snapshot <- function(fit) {
  keep <- function(x) if(is.null(x)) NULL else I(x)
  fields <- lapply(numeric_fields,function(name) {
    value <- fit[[name]]
    if(is.null(value)) return(NULL)
    list(values=I(as.numeric(value)),dim=keep(dim(value)))
  })
  names(fields) <- numeric_fields
  list(fields=fields,kind=if(inherits(fit,"survfitcoxms")) "coxms" else
       if(inherits(fit,"survfitms")) "aj" else if(inherits(fit,"survfitcox")) "cox" else "km",
       strata=if(is.null(fit$strata)) NULL else as.list(fit$strata),
       states=keep(fit$states),hazard_names=keep(tail(dimnames(fit$cumhaz),1)[[1]]),
       colnames=keep(colnames(fit$surv)),type=fit$type,t0=fit$t0,logse=fit$logse,
       conf_type=fit$conf.type,conf_int=fit$conf.int)
}

# Run the actual R plotting method and record each graphics call. A null PDF
# supplies the device state used by the method without producing an artifact.
capture <- function(fit,options,method="plot") {
  calls <- list()
  save_xy <- function(kind,x,y=NULL,...) {
    xy <- xy.coords(x,y)
    calls[[length(calls)+1L]] <<- list(kind=kind,x=I(xy$x),y=I(xy$y))
    invisible(NULL)
  }
  f <- get(paste0(method,".survfit"),asNamespace("survival"))
  environment(f) <- env <- new.env(parent=environment(f))
  env$lines <- function(x,y=NULL,...) save_xy("line",x,y,...)
  env$points <- function(x,y=NULL,...) save_xy("point",x,y,...)
  env$segments <- function(x0,y0,x1,y1,...) {
    calls[[length(calls)+1L]] <<- list(kind="segment",x0=I(x0),y0=I(y0),x1=I(x1),y1=I(y1))
  }
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off())
  if(method!="plot") graphics::plot(c(0,max(fit$time)),c(0,1),type="n")
  endpoint <- do.call(f,c(list(x=fit),options))
  list(calls=calls,endpoint=endpoint,xlog=par("xlog"),ylog=par("ylog"))
}
cases <- list()
add <- function(name,fit,options=list(),method="plot") {
  cases[[length(cases)+1L]] <<- list(name=name,fit=fit,options=options,method=method,
                                     expected=capture(fits[[fit]],options,method))
}
for(fit in names(fits)) add(paste0(fit,"_default"),fit,list(mark.time=TRUE))
for(fun in c("event","pct","cumhaz","cloglog","log","logpct","identity")) {
  add(paste0("km_",fun),"km",list(fun=fun,conf.int=TRUE,mark.time=TRUE))
}
for(type in c("plain","log","log-log","logit","arcsin")) {
  add(paste0("ci_",type),"km",list(conf.int=.8,conf.type=type))
}
add("ci_only","km",list(conf.int="only"))
add("ci_none","km",list(conf.int="none"))
add("ci_bars","km",list(conf.times=c(1.5,2,4.5,6),conf.offset=0,conf.cap=0))
add("requested_marks","km",list(mark.time=c(-1,0,1.5,2,6,8,10),conf.int=FALSE))
add("truncated","groups",list(xmax=4.5,mark.time=TRUE,conf.int=TRUE))
add("truncated_tied","km",list(xmax=2,mark.time=TRUE))
add("cox_hazard","cox",list(cumhaz=TRUE,conf.int=TRUE))
add("aj_ci","aj",list(conf.int=TRUE))
add("aj_all_states","aj",list(noplot="",conf.int=FALSE))
add("aj_cumprob","aj",list(cumprob=c(3,2),conf.int=FALSE))
add("aj_hazards","aj_groups",list(cumhaz=TRUE,conf.int=TRUE))
add("aj_selected_hazard","aj",list(cumhaz=2,conf.int=TRUE))
add("aj_pct","aj",list(fun="pct",conf.int=TRUE))
add("aj_event","aj",list(fun="event",conf.int=TRUE))
add("coxms_cumprob","coxms",list(cumprob=c(3,2),conf.int=FALSE))
add("coxms_selected_hazards","coxms",list(cumhaz=c(2,1),conf.int=FALSE))
add("added_lines","groups",list(conf.int=TRUE,mark.time=TRUE),"lines")
add("event_points","km",list(),"points")
add("all_points","km",list(censor=TRUE),"points")
add("grouped_points","groups",list(pch=c(1,2),col=c(1,2)),"points")
add("terminal_log","terminal",list(log=TRUE,conf.int=FALSE,mark.time=c(1,2,3,4)))
add("terminal_log_ci","terminal",list(log=TRUE,conf.int=TRUE))

reference <- list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),
                  fits=lapply(fits,snapshot),cases=cases)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"graphics cases written to",output,"\n")
