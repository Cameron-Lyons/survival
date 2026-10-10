#!/usr/bin/env Rscript
# Independent stock-R nondimensional selectors and stale aggregation uncertainty.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args))args[[1L]]else 'python/tests/fixtures/cox_survfit_single_reference.json'
reference <- local({
d <- data.frame(time=1:12, status=rep(c(1,1,0),4), x=rep(c(0,1,2),4),
  z=c(0,1,.5,-1,2,.7,1,0,-1,.5,2,1), g=factor(rep(c("a","b"),each=6)))
nd <- data.frame(x=c(-1,0,2),z=c(1,0,-1),tag=factor(c("b","a","b"),levels=c("a","unused","b")),
  row.names=c("two","one","three"))
f <- coxph(Surv(time,status)~x+z,d,init=c(.25,-.125),iter.max=0)
fs <- coxph(Surv(time,status)~x+z+strata(g),d,init=c(.25,-.125),iter.max=0)
sources <- list(default=survfit(f), plain_one=survfit(f,newdata=nd[1,,drop=FALSE]),
  selected_one=survfit(fs,newdata=nd)[1,1], aggregated=aggregate(survfit(f,newdata=nd)),
  starts_late=survfit(f,newdata=nd[1,,drop=FALSE],start.time=4.5))
snapshot <- function(x) list(fields=names(x),dim=as.list(dim(x)),newdata=x$newdata,
  start.time=x$start.time,time=x$time,surv=x$surv,n=x$n,strata=x$strata,
  logse=x$logse,cumhaz=x$cumhaz,std.err=x$std.err)
selectors <- list(missing=quote(x[]),null=quote(x[NULL]),one=quote(x[1]),
  repeated=quote(x[c(1,1)]),empty=quote(x[integer()]),zero=quote(x[0]),
  negative=quote(x[-1]),true=quote(x[TRUE]),false=quote(x[FALSE]),
  repeat_true=quote(x[c(TRUE,TRUE)]),one_string=quote(x["1"]),
  bad_name=quote(x["two"]),fraction=quote(x[1.5]),missing_value=quote(x[NA]),
  matrix_ones=quote(x[matrix(1,2,2)]),one_list=quote(x[list(1)]),
  extra_dimension=quote(x[1,1]),drop_false=quote(x[1,drop=FALSE]),
  drop_null=quote(x[1,drop=NULL]))
capture <- function(expr) tryCatch(list(value=force(expr)),error=function(e)list(error=conditionMessage(e)))
cases <- list()
for (source in names(sources)) for (selector in names(selectors)) {
  x <- sources[[source]]
  selected <- capture(eval(selectors[[selector]]))
  if (!is.null(selected$value)) {
    y <- selected$value
    selected <- list(value=snapshot(y),serialized=snapshot(unserialize(serialize(y,NULL))),
      initial=capture(snapshot(survfit0(y))),
      summary=capture(summary(y,censored=TRUE,data.frame=TRUE)),
      at_times=capture(summary(y,times=c(0,4,8,12),extend=TRUE,data.frame=TRUE)),
      quantile=capture(quantile(y,probs=.5)),
      again=capture(snapshot(y[c(1,1)])))
  }
  cases[[length(cases)+1L]] <- list(name=paste(source,selector,sep="/"),
    source=source,selector=selector,expected=selected)
}
list(sources=lapply(sources,snapshot),cases=cases)
})
std_chaz <- local({
d <- data.frame(time=1:12,status=rep(c(1,1,0),4),x=rep(c(0,1,2),4),
  z=c(0,1,.5,-1,2,.7,1,0,-1,.5,2,1),g=factor(rep(c("a","b"),each=6)))
f <- coxph(Surv(time,status)~x+z+strata(g),d,init=c(.25,-.125),iter.max=0)
nd <- data.frame(x=c(-1,0,2,3),z=c(1,0,-1,.5))
x <- survfit(f,newdata=nd)
snapshot <- function(y) list(dim=dim(y),surv_dim=dim(y$surv),surv=y$surv,
  std.chaz_dim=dim(y$std.chaz),std.chaz=y$std.chaz,cumhaz=y$cumhaz,
  std.err=y$std.err,newdata=y$newdata)
cases <- list(original=snapshot(x))
for (by_name in c("none","constant","two_groups")) {
  by <- switch(by_name,none=NULL,constant=rep("same",4),two_groups=c("b","a","b","a"))
  y <- aggregate(x,by=by,FUN=mean)
  cases[[by_name]] <- list(aggregate=snapshot(y),identical_uncertainty=identical(y$std.chaz,x$std.chaz),
    selected_first=snapshot(if(is.matrix(y$surv))y[1,1]else y[1]),
    selected_second=snapshot(if(is.matrix(y$surv))y[1,2]else y[2]),
    corrected=list(std.chaz=NULL,note="Discard unchanged original uncertainty columns after aggregating curves"))
}
list(cases=cases)
})
start_time <- local({
d <- data.frame(time=1:12,status=rep(c(1,1,0),4),x=rep(c(0,1,2),4),
  z=c(0,1,.5,-1,2,.7,1,0,-1,.5,2,1),g=factor(rep(c("a","b"),each=6)))
nd <- data.frame(x=c(-1,0,2),z=c(1,0,-1))
form <- Surv(time,status)~x+z+strata(g)
x <- survival::survfit(survival::coxph(form,d,init=c(.25,-.125),iter.max=0),newdata=nd,start.time=4.5)
selections <- list(missing=function(x)x[],null=function(x)x[NULL],both_missing=function(x)x[,],
  full_margins=function(x)x[1:2,1:3,drop=FALSE],reorder_margins=function(x)x[2:1,3:1,drop=FALSE],
  repeat_margins=function(x)x[c(2,1,2),c(3,1),drop=FALSE],
  single_margin=function(x)x[1,,drop=FALSE],data_margin=function(x)x[,1,drop=FALSE],
  single_both=function(x)x[1,1],linear_full=function(x)x[1:6],
  linear_reorder=function(x)x[6:1],linear_repeat=function(x)x[c(4,1,4)],
  linear_single=function(x)x[2])
lapply(selections,function(select) {
  y <- select(x)
  list(start.time=y$start.time,dim=as.list(dim(y)),newdata=y$newdata,time=y$time,surv=y$surv)
})
})
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(list(metadata=list(generator='scripts/generate_cox_survfit_single_reference.R',
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion('survival'))),
  single=reference,start_time=start_time,std_chaz=std_chaz),output,
  auto_unbox=TRUE,digits=17,pretty=TRUE,null='null',na='string')
cat(length(reference$cases),'single-curve selector cases,',length(start_time),
  'start.time cases and 3 stale std.chaz aggregation cases written to',output,'\n')
