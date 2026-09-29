#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/aareg_plot_reference.json"

d <- data.frame(time=(seq_len(60)*7)%%13+1, status=as.integer(seq_len(60)%%3!=0),
                x=(seq_len(60)*3)%%11, z=seq_len(60)%%2,
                group=seq_len(60)%%9, start=0)
d$start <- pmax(0,d$time-3)
fits <- list(
  ovarian=aareg(Surv(futime,fustat)~age+ecog.ps,ovarian),
  robust=aareg(Surv(futime,fustat)~age+ecog.ps,ovarian,dfbeta=TRUE),
  clustered=aareg(Surv(futime,fustat)~age+ecog.ps+cluster(rx),ovarian),
  weighted=aareg(Surv(futime,fustat)~age+ecog.ps,ovarian,weights=rep(c(.5,1,2),length.out=26)),
  tapered=aareg(Surv(futime,fustat)~age+ecog.ps,ovarian,taper=c(1,.7,.3)),
  ties=aareg(Surv(time,status)~x+z,d),
  ties_robust=aareg(Surv(time,status)~x+z,d,dfbeta=TRUE),
  ties_clustered=aareg(Surv(time,status)~x+z+cluster(group),d),
  delayed=aareg(Surv(start,time,status)~x+z,d,nmin=3),
  delayed_robust=aareg(Surv(start,time,status)~x+z+cluster(group),d,nmin=3))
fits$single <- fits$ovarian[2]
fits$single_robust <- fits$robust[2]
fits$zero <- fits$ovarian; fits$zero$times <- fits$zero$times-min(fits$zero$times)
fits$negative <- fits$ovarian; fits$negative$times <- fits$negative$times-100
fits$zero_robust <- fits$robust; fits$zero_robust$times <- fits$zero_robust$times-min(fits$zero_robust$times)

snapshot <- function(fit) {
  list(time=I(fit$times),coefficient=fit$coefficient,names=I(names(fit$test.statistic)),
       dfbeta=if(is.null(fit$dfbeta)) NULL else list(values=I(as.numeric(fit$dfbeta)),dim=dim(fit$dfbeta)))
}
capture <- function(fit,options,method) {
  calls <- list()
  f <- get(paste0(method,".aareg"),asNamespace("survival"))
  environment(f) <- env <- new.env(parent=environment(f))
  env$matplot <- env$matlines <- function(x,y,...,ylab=NULL,type="p") {
    calls[[length(calls)+1L]] <<- list(x=I(x),y=as.matrix(y),ylab=ylab,type=type)
  }
  do.call(f,c(list(x=fit),options))
  calls
}
cases <- list()
add <- function(name,fit,options=list(),method="plot",var=NULL) {
  object <- fits[[fit]]
  if(!is.null(var)) object <- object[var]
  cases[[length(cases)+1L]] <<- list(name=name,fit=fit,options=options,method=method,var=var,
                                    expected=capture(object,options,method))
}
for(name in setdiff(names(fits),"zero_robust")) add(name,name)
for(name in c("ovarian","robust","single","single_robust","ties","ties_robust","delayed_robust")) {
  add(paste0(name,"_no_se"),name,list(se=FALSE))
  add(paste0(name,"_truncated"),name,list(maxtime=median(fits[[name]]$times)))
}
add("selected_terms","ovarian",var=c(3,2))
add("selected_robust","robust",var=3)
add("repeated_terms","ties_robust",var=c(2,2))
add("cut_between_events","ovarian",list(maxtime=300))
add("cut_after_events","ovarian",list(maxtime=2000))
add("lines_default","ovarian",method="lines")
add("lines_bands","ties_robust",list(se=TRUE),"lines")
add("lines_selected","robust",list(se=TRUE,maxtime=400),"lines",var=2)
add("straight_lines","single",list(type="l"))

# Record upstream failures separately, rather than treating them as the
# numerical contract: dropped dimensions and the zero-origin robust bands.
failures <- list()
for(item in list(list(name="first_event",fit="ovarian",options=list(maxtime=59)),
                list(name="before_first",fit="ovarian",options=list(maxtime=1)),
                list(name="zero_robust",fit="zero_robust",options=list()))) {
  item$error <- tryCatch({capture(fits[[item$fit]],item$options,"plot"); NULL},error=conditionMessage)
  failures[[length(failures)+1L]] <- item
}
write_json(list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),
                fits=lapply(fits,snapshot),cases=cases,upstream_failures=failures),
           output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"Aalen graphics cases written to",output,"\n")
