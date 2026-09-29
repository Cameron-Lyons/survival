#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/cox_zph_plot_reference.json"

fit <- coxph(Surv(time,status)~age+sex,lung)
fits <- lapply(c("km","identity","rank","log"), function(transform) cox.zph(fit,transform))
names(fits) <- c("km","identity","rank","log")
fits$custom <- cox.zph(fit,function(time) sqrt(time))
fits$reverse <- cox.zph(fit,function(time) -time)
fits$stratified <- cox.zph(coxph(Surv(time,status)~age:strata(sex),lung))
fits$penalized <- cox.zph(coxph(Surv(time,status)~pspline(age,df=4)+sex,lung))
fits$counting <- cox.zph(coxph(Surv(start,stop,event)~age+surgery,heart))
fits$weighted <- cox.zph(coxph(Surv(time,status)~age+sex,lung,
                              weights=rep(c(.5,1,2),length.out=nrow(lung))))
z <- data.frame(time=(seq_len(80)*7)%%29+1,
                status=factor(ifelse(seq_len(80)%%3==0,"censor",
                  ifelse(seq_len(80)%%4==0,"a","b")),levels=c("censor","a","b")),
                age=(seq_len(80)*3)%%11)
fits$multistate <- cox.zph(coxph(Surv(time,status)~age,z,id=seq_len(nrow(z))))
fits$missing <- fits$identity
fits$missing$y[seq(1,nrow(fits$missing$y),by=3),1] <- NA
fits$singular <- fits$missing
fits$singular$y[,2] <- NA
fits$singular$y[1,2] <- 1

snapshot <- function(x) list(x=I(x$x),time=I(x$time),y=x$y,variance=x$var,
                             transform=x$transform,names=I(colnames(x$y)))
capture <- function(x,options) {
  plots <- list()
  warnings <- character()
  f <- get("plot.cox.zph",asNamespace("survival"))
  environment(f) <- env <- new.env(parent=environment(f))
  env$plot <- function(x,y,...,log="",ylab="") {
    plots[[length(plots)+1L]] <<- list(xlim=I(x),ylim=I(y),log=log,ylab=ylab,
                                      lines=list(),points=list(),ticks=NULL,labels=NULL)
  }
  env$lines <- function(x,y,...) {
    i <- length(plots)
    plots[[i]]$lines[[length(plots[[i]]$lines)+1L]] <<- list(x=I(x),y=I(as.numeric(y)))
  }
  env$points <- function(x,y,...) {
    i <- length(plots)
    plots[[i]]$points <<- list(x=I(x),y=I(as.numeric(y)))
  }
  env$axis <- function(side,at,labels,...) {
    i <- length(plots)
    plots[[i]]$ticks <<- I(at)
    plots[[i]]$labels <<- I(labels)
  }
  withCallingHandlers(do.call(f,c(list(x=x),options)),warning=function(w) {
    warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
  })
  list(plots=plots,warnings=I(warnings))
}
cases <- list()
add <- function(name,fit,options=list()) {
  cases[[length(cases)+1L]] <<- list(name=name,fit=fit,options=options,
                                    expected=capture(fits[[fit]],options))
}
for(name in names(fits)) add(name,name)
for(df in c(2,3,6,8)) add(paste0("df",df),"km",list(df=df,nsmo=17))
for(nsmo in c(2,5,100)) add(paste0("grid",nsmo),"identity",list(nsmo=nsmo))
for(name in c("km","identity","log","stratified","multistate")) {
  add(paste0(name,"_hr"),name,list(hr=TRUE))
}
add("no_se","km",list(se=FALSE))
add("no_residuals","km",list(resid=FALSE))
add("no_se_or_residuals","log",list(se=FALSE,resid=FALSE,hr=TRUE))
add("named_selection","km",list(var=c("sex","age")))
add("indexed_selection","km",list(var=2,df=3,nsmo=11))
add("repeated_selection","km",list(var=c(2,2)))
add("missing_no_se","missing",list(se=FALSE,df=6))
reference <- list(metadata=list(R=R.version.string,survival=as.character(packageVersion("survival"))),
                  fits=lapply(fits,snapshot),cases=cases)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"diagnostic cases written to",output,"\n")
