#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/response_plot_reference.json"

responses <- list(
  right=Surv(c(1,2,2,3,5,6,8),c(1,1,0,1,0,1,0)),
  complete=Surv(c(1,2,3,4)),
  censored=Surv(c(1,2,3,4),c(0,0,0,0)),
  counting=Surv(c(0,0,1,1,2,3,2),c(1,2,2,3,5,6,8),c(1,1,0,1,0,1,0)),
  left=Surv(c(1,2,3,4,5),c(0,1,1,0,1),type="left"),
  interval=Surv(c(1,2,3,4,5),c(1,3,Inf,6,5),type="interval2"),
  missing=Surv(c(1,NA,2,3,4),c(1,1,0,1,0)),
  multistate=Surv(c(1,2,2,3,5,6,8),
                  factor(c("a","b","censor","a","censor","b","censor"),
                         levels=c("censor","a","b"))))
d <- data.frame(time=c(100,300,500,800,1200,1500),age=c(50,60,70,80,55,65)*365.25,
                sex=c(1,2,1,2,1,2),year=as.Date("2000-01-01"))
rates <- function(formula,...) survexp(formula,d,rmap=list(age=age,sex=sex,year=year),...)
times <- c(1,100,300,600,1000,1500)
expected <- list(
  single=rates(~1,times=times),
  groups=rates(~sex,times=times),
  zero=rates(~1,times=c(0,times)),
  hakulinen=rates(time~sex,times=times,method="hakulinen"),
  conditional=rates(time~sex,times=times,method="conditional"),
  cox=survexp(~sex,lung,ratetable=coxph(Surv(time,status)~age+sex,lung),times=c(1,100,300,600,1000)))

keep <- function(x) if(is.null(x)) NULL else I(x)
snapshot_response <- function(x) list(values=unclass(x),type=attr(x,"type"),states=keep(attr(x,"states")))
snapshot_expected <- function(x) list(time=I(x$time),surv=if(is.matrix(x$surv)) x$surv else I(x$surv),
                                      n_risk=if(is.matrix(x$n.risk)) x$n.risk else I(x$n.risk),
                                      names=keep(colnames(x$surv)),method=x$method)
capture <- function(object,options,method) {
  calls <- list()
  previous <- base::options(plot.survfit=NULL)
  on.exit(base::options(previous))
  original <- getS3method(method,"survfit")
  f <- original
  environment(f) <- env <- new.env(parent=environment(f))
  save_xy <- function(kind,x,y=NULL,...) {
    xy <- xy.coords(x,y)
    calls[[length(calls)+1L]] <<- list(kind=kind,x=I(xy$x),y=I(xy$y))
  }
  env$lines <- function(x,y=NULL,...) save_xy("line",x,y,...)
  env$points <- function(x,y=NULL,...) save_xy("point",x,y,...)
  # Keep actual method bodies: plot.Surv fits the response and lines.survexp
  # supplies its straight-line default before dispatching to this method.
  registerS3method(method,"survfit",f,envir=asNamespace("graphics"))
  on.exit(registerS3method(method,"survfit",original,envir=asNamespace("graphics")),add=TRUE)
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(),add=TRUE)
  if(method!="plot") graphics::plot(c(0,2000),c(0,1),type="n")
  if(inherits(object,"Surv")) {
    wrapper <- getS3method(method,"Surv")
    environment(wrapper) <- wrapper_env <- new.env(parent=environment(wrapper))
    wrapper_env$plot <- f
    do.call(wrapper,c(list(x=object),options))
  } else do.call(get(method,asNamespace("graphics")),c(list(x=object),options))
  list(calls=calls,xlog=par("xlog"),ylog=par("ylog"))
}
cases <- list()
add <- function(name,source,key,options=list(),method="plot") {
  object <- if(source=="response") responses[[key]] else expected[[key]]
  cases[[length(cases)+1L]] <<- list(name=name,source=source,key=key,options=options,
                                    method=method,expected=capture(object,options,method))
}
for(key in names(responses)) add(paste0("raw_",key),"response",key,list(mark.time=TRUE))
add("raw_event","response","right",list(fun="event",conf.int=FALSE))
add("raw_hazard","response","counting",list(fun="cumhaz"))
add("raw_interval_truncated","response","interval",list(xmax=4))
for(key in names(expected)) {
  add(paste0("expected_",key),"expected",key)
  add(paste0("overlay_",key),"expected",key,method="lines")
}
add("overlay_steps","expected","groups",list(type="s"),"lines")
add("overlay_event","expected","groups",list(fun="event"),"lines")
add("overlay_hazard","expected","single",list(fun="cumhaz"),"lines")
add("overlay_pct","expected","single",list(fun="pct"),"lines")
add("overlay_cutoff","expected","groups",list(xmax=450),"lines")
add("expected_marks","expected","single",list(mark.time=c(0,50,200,1000)))
add("expected_no_censors","expected","single",list(mark.time=TRUE))
add("expected_points","expected","groups",list(censor=TRUE,col=1:2),"points")
add("expected_no_event_points","expected","single",method="points")

write_json(list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),
                responses=lapply(responses,snapshot_response),expected=lapply(expected,snapshot_expected),
                cases=cases),output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"response/expected graphics cases written to",output,"\n")
