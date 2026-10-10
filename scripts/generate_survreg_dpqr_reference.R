# Independent stock survival DPQR vector arithmetic and callback references.
# Run from the repository root; an optional argument selects the fixture directory.
arguments <- commandArgs(trailingOnly=TRUE)
output_directory <- if (length(arguments)) arguments[[1L]] else 'python/tests/fixtures'
dir.create(output_directory,recursive=TRUE,showWarnings=FALSE)
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
number <- function(x) if (is.nan(x)) 'NaN' else if (is.na(x)) 'NA' else if (is.infinite(x)) if (x>0) 'Inf' else '-Inf' else unname(x)
numbers <- function(x) unname(lapply(x, number))
events <- list()
event <- function(kind, ...) events[[length(events)+1L]] <<- c(list(kind=kind),list(...))
record <- function(method, query, mean, scale, distribution, parms=NULL) {
 set.seed(123); before <- .Random.seed; events <<- list(); error <- NULL
 call <- list(query,mean,scale,distribution)
 if (!is.null(parms)) call$parms <- parms
 result <- withCallingHandlers(tryCatch(do.call(get(method,asNamespace('survival')),call),
   error=function(e) {error<<-conditionMessage(e);event('error',message=error);NULL}),
   warning=function(w) {event('warning',message=conditionMessage(w),call=paste(deparse(conditionCall(w)),collapse=' '));invokeRestart('muffleWarning')})
 list(values=if(is.null(error)) numbers(result) else NULL,length=if(is.null(error))length(result) else NULL,
      error=error,events=events,rng_changed=!identical(before,.Random.seed),next_uniform=runif(1))
}
builtins <- list()
for (family in c('gaussian','weibull','logistic','t','extreme','lognormal')) {
 parms <- if (family=='t') 5 else NULL
 for (method in c('dsurvreg','psurvreg','qsurvreg','rsurvreg')) {
  for (nq in 0:5) for (nm in 0:5) for(ns in 0:5) {
   query <- if(method=='rsurvreg') nq else seq_len(nq)/7
   mean <- (seq_len(nm)-1)/3
   scale <- .5+seq_len(ns)/4
   builtins[[length(builtins)+1L]] <- list(name=paste(family,method,nq,nm,ns,sep='/'),distribution=family,
    method=method,query=if(method=='rsurvreg') query else numbers(query),mean=numbers(mean),scale=numbers(scale),
    expected=record(method,query,mean,scale,family,parms))
  }
 }
}
exceptions <- list()
vecs <- list(na=NA_real_,nan=NaN,inf=Inf,negative_inf=-Inf,zero=0,negative=-1,
 mixed=c(NA,NaN,Inf,-Inf,0,-1),empty=numeric())
for (family in c('gaussian','weibull','logistic','t','extreme','lognormal')) {
 parms <- if(family=='t')5 else NULL
 for (method in c('dsurvreg','psurvreg','qsurvreg','rsurvreg')) {
  for (argument in c('query','mean','scale')) for (name in names(vecs)) {
   query <- if(method=='rsurvreg')3L else c(.2,.5,.8)
   mean <- 0; scale <- 1
   if(argument=='query' && method=='rsurvreg')next
   if(argument=='query')query<-vecs[[name]]
   if(argument=='mean')mean<-vecs[[name]]
   if(argument=='scale')scale<-vecs[[name]]
   exceptions[[length(exceptions)+1L]] <- list(name=paste(family,method,argument,name,sep='/'),distribution=family,
    method=method,query=if(method=='rsurvreg')query else numbers(query),mean=numbers(mean),scale=numbers(scale),
    expected=record(method,query,mean,scale,family,parms))
  }
 }
}
write_json(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion('survival'))),
 builtins=builtins,exceptions=exceptions),file.path(output_directory, 'survreg_dpqr_reference.json'),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')

namespace <- asNamespace('survival')
distributions <- survival::survreg.distributions
callback <- distributions$gaussian
callback$name <- 'DPQR callback oracle'
callback$density <- function(z,parms) {event('density',values=numbers(z));cbind(pnorm(z),pnorm(-z),dnorm(z),-z,z*z-1)}
callback$quantile <- function(p,parms) {event('quantile',values=numbers(p));qnorm(p)}
derived <- list(name='DPQR transform oracle',dist='audit_base',
 trans=function(x) {event('transform',values=numbers(x));log(x)},
 dtrans=function(x) {event('derivative',values=numbers(x));1/x},
 itrans=function(x) {event('inverse',values=numbers(x));exp(x)})
distributions$audit_base <- callback; distributions$audit_derived<-derived
unlockBinding('survreg.distributions',namespace);assign('survreg.distributions',distributions,namespace);lockBinding('survreg.distributions',namespace)
callbacks<-list()
for(family in c('audit_base','audit_derived'))for(method in c('dsurvreg','psurvreg','qsurvreg','rsurvreg')) {
 specs<-list(ordinary=list(q=c(.2,.5,.8),m=0,s=1),query0=list(q=numeric(),m=0,s=1),
    mean0=list(q=c(.2,.5,.8),m=numeric(),s=1),scale0=list(q=c(.2,.5,.8),m=0,s=numeric()),
    meanlong=list(q=c(.2,.5),m=1:3,s=1),scalelong=list(q=c(.2,.5),m=0,s=1:3),
    three_lengths=list(q=c(.2,.5),m=1:3,s=1:5),mixed_query=list(q=c(NA,NaN,Inf,-Inf,0,-1),m=0,s=1),
    mean_nonfinite=list(q=c(.2,.5,.8),m=c(NA,NaN,Inf,-Inf,0),s=1),
    scale_nonfinite=list(q=c(.2,.5,.8),m=0,s=c(NA,NaN,Inf,-Inf,0,-1)),
    invalid_probability=list(q=c(-.1,.2,1.1),m=c(0,1),s=c(1,2,3,4)),
    negative_transform=list(q=c(-1,.5),m=1:3,s=1:5))
 for(name in names(specs)) {
  spec<-specs[[name]];query<-if(method=='rsurvreg')length(spec$q)else spec$q
  callbacks[[length(callbacks)+1L]]<-list(name=paste(family,method,name,sep='/'),distribution=family,method=method,
    query=if(method=='rsurvreg')query else numbers(query),mean=numbers(spec$m),scale=numbers(spec$s),
    expected=record(method,query,spec$m,spec$s,family))
 }
}
write_json(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion('survival'))),cases=callbacks),
 file.path(output_directory, 'survreg_dpqr_callback_reference.json'),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
cat(length(builtins),'length cases;',length(exceptions),'exception cases;',length(callbacks),'callback cases\n')

setup <- function(mode) {
 callback$density <- function(z,parms) {
  event('density',values=numbers(z))
  if(mode=='density_error')stop('density stopped')
  if(mode=='callback_warnings')warning('density warning')
  y<-cbind(pnorm(z),pnorm(-z),dnorm(z),-z,z*z-1)
  if(mode=='density_unused_invalid')y[,c(2,4,5)]<-NaN
  if(mode=='density_negative')y[,3]<- -y[,3]
  if(mode=='density_nan')y[,1:3]<-NaN
  if(mode=='density_short')y<-y[1,,drop=FALSE]
  y
 }
 callback$quantile <- function(p,parms) {
  if(mode=='quantile_error_unforced') {event('quantile_unforced');stop('quantile stopped')}
  if(mode=='quantile_constant_unforced') {event('quantile_unforced');return(.5)}
  event('quantile',values=numbers(p))
  if(mode=='quantile_error')stop('quantile stopped')
  if(mode=='callback_warnings')warning('quantile warning')
  x<-qnorm(p)
  if(mode=='quantile_short')x<-x[1]
  x
 }
 derived$trans <- function(x) {
  event('transform',values=numbers(x))
  if(mode=='transform_error')stop('transform stopped')
  if(mode=='callback_warnings')warning('transform warning')
  z<-log(x)
  if(mode=='transform_long')z<-c(z,.7)
  z
 }
 derived$dtrans <- function(x) {
  event('derivative',values=numbers(x))
  if(mode=='derivative_error')stop('derivative stopped')
  if(mode=='callback_warnings')warning('derivative warning')
  1/x
 }
 derived$itrans <- function(x) {
  event('inverse',values=numbers(x))
  if(mode=='inverse_error')stop('inverse stopped')
  if(mode=='callback_warnings')warning('inverse warning')
  z<-exp(x)
  if(mode=='inverse_short')z<-z[1]
  z
 }
 distributions$audit_base <- callback;distributions$audit_derived<-derived
 unlockBinding('survreg.distributions',namespace);assign('survreg.distributions',distributions,namespace);lockBinding('survreg.distributions',namespace)
}
result<-list()
for(mode in c('derivative_error','transform_error','density_error','quantile_error','inverse_error',
 'callback_warnings','density_unused_invalid','density_negative','density_nan','density_short',
 'quantile_short','transform_long','inverse_short','quantile_error_unforced','quantile_constant_unforced')) {
 setup(mode)
 for(family in c('audit_base','audit_derived'))for(method in c('dsurvreg','psurvreg','qsurvreg','rsurvreg')) {
  query<-if(method=='rsurvreg')2L else c(.2,.5)
  result[[length(result)+1L]]<-list(name=paste(mode,family,method,sep='/'),expected=record(method,query,1:3,1:5,family))
 }
}
for(method in c('dsurvreg','psurvreg','qsurvreg','rsurvreg')) {
 query<-if(method=='rsurvreg')3L else .5
 result[[length(result)+1L]]<-list(name=paste('unknown_distribution',method,sep='/'),expected=record(method,query,0,1,'missing'))
}
write_json(list(cases=result),file.path(output_directory, 'survreg_dpqr_callback_edges.json'),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
cat(length(result),'callback edge cases\n')

result<-list()
setup('ordinary')
for(family in c('gaussian','weibull','logistic','t','extreme','lognormal','audit_base','audit_derived'))
 for(method in c('dsurvreg','psurvreg','qsurvreg'))for(empty in c('mean','scale')) {
  query<-c(-.1,.5,1.1,NA,NaN,Inf,-Inf)
  mean<-if(empty=='mean')numeric() else 0
  scale<-if(empty=='scale')numeric() else 1
  result[[length(result)+1L]]<-list(name=paste(family,method,empty,sep='/'),
   expected=record(method,query,mean,scale,family,if(family=='t')5 else NULL))
 }
write_json(list(cases=result),file.path(output_directory, 'survreg_dpqr_empty_reference.json'),auto_unbox=TRUE,pretty=TRUE,digits=17,null='null')
cat(length(result),'empty-result warning order cases\n')
