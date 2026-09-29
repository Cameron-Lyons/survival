#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else 'python/tests/fixtures/aft_aalen_report_reference.json'
options(digits=7,width=80)
encode <- function(x) if(is.null(x)) NULL else list(rows=I(rownames(x)),columns=I(colnames(x)),values=unname(x))
models <- list();datasets <- list();fits <- list();cases <- list()
add_fit <- function(name,kind,formula,data,arguments=list()) {
  matched <- which(vapply(datasets,function(x) identical(x,data),logical(1)))
  key <- if(length(matched)) names(datasets)[matched[1]] else paste0('data',length(datasets)+1L)
  datasets[[key]] <<- data
  models[[name]] <<- list(kind=kind,formula=formula,data=key,arguments=arguments)
  fits[[name]] <<- do.call(get(kind),c(list(formula=as.formula(formula),data=data),arguments))
}
f <- 'Surv(time,status)~age+sex'
for(dist in c('weibull','exponential','rayleigh','lognormal','loglogistic','gaussian','logistic','t'))
  add_fit(dist,'survreg',f,lung,if(dist=='t') list(dist=dist,parms=6) else list(dist=dist))
add_fit('fixed','survreg',f,lung,list(scale=1.25))
add_fit('null','survreg','Surv(time,status)~1',lung)
add_fit('stratified','survreg','Surv(time,status)~age+strata(sex)+sex',lung)
add_fit('robust','survreg','Surv(time,status)~age+sex+cluster(inst)',lung)
add_fit('missing','survreg','Surv(time,status)~age+ph.ecog+wt.loss',lung)
add_fit('aliased','survreg','Surv(time,status)~age+I(age * 2)+sex',lung)
add_fit('ridge','survreg','Surv(time,status)~ridge(age, sex, theta = 1)',lung)
add_fit('spline','survreg','Surv(time,status)~pspline(age, df = 3)+sex',lung)
add_fit('penal_scales','survreg','Surv(time,status)~pspline(age, df = 3)+strata(sex)',lung)
add_fit('penal_fixed','survreg','Surv(time,status)~ridge(age, sex, theta = 1)',lung,list(scale=1))
add_fit('interval','survreg','Surv(durable,durable>0,type="left")~age+quant',tobin,list(dist='gaussian'))
ovarian$grp <- seq_len(nrow(ovarian))%%7
a <- 'Surv(futime,fustat)~age+ecog.ps'
add_fit('aalen','aareg',a,ovarian)
add_fit('aalen_robust','aareg',a,ovarian,list(dfbeta=TRUE))
add_fit('aalen_cluster','aareg','Surv(futime,fustat)~age+ecog.ps+cluster(grp)',ovarian)
add_fit('aalen_nrisk','aareg',a,ovarian,list(test='nrisk'))
add_fit('aalen_weighted','aareg',a,ovarian,list(weights=rep(c(.5,1,2),length.out=26)))
add_fit('aalen_missing','aareg','Surv(time,status)~age+ph.ecog+wt.loss',lung)
i <- seq_len(60)
d <- data.frame(time=(i*7)%%13+1,status=as.integer(i%%3!=0),x=(i*3)%%11,z=i%%2,group=i%%9)
d$start <- pmax(0,d$time-3)
add_fit('aalen_ties','aareg','Surv(time,status)~x+z',d,list(dfbeta=TRUE))
add_fit('aalen_delayed','aareg','Surv(start,time,status)~x+z',d,list(dfbeta=TRUE,nmin=3))
add_fit('aalen_ties_cluster','aareg','Surv(time,status)~x+z+cluster(group)',d)

snapshot <- function(x,kind) {
  if(kind=='survreg') return(list(
    model_type=kind,table=encode(x$table),location_coefficients=I(x$coefficients),
    var=unname(x$var),scales=I(x$scale),scale_names=if(is.null(names(x$scale))) list() else I(names(x$scale)),
    fixed_scale=nrow(x$var)==length(x$coefficients),df=I(x$df),idf=x$idf,n=x$n,
    loglik=I(x$loglik),chi=x$chi,chi_df=sum(x$df)-x$idf,iter=I(x$iter),
    robust=x$robust,parms=paste0(x$parms,collapse=''),correlation=encode(x$correlation),
    omit=if(is.null(x$na.action)) NULL else I(as.integer(x$na.action))))
  list(model_type=kind,table=encode(x$table),test=x$test,test_statistic=I(x$test.statistic),
       test_var=unname(x$test.var),test_var2=if(is.null(x$test.var2)) NULL else unname(x$test.var2),
       chisq=as.numeric(x$chisq),df=nrow(x$table)-1,
       p=as.numeric(pchisq(x$chisq,nrow(x$table)-1,lower.tail=FALSE)),n=I(x$n))
}
add <- function(name,fit_name,method='direct',arguments=list(),summary_arguments=list(),width=80) {
  fit <- fits[[fit_name]]; kind <- models[[fit_name]]$kind
  original_lines <- NULL; adjustment <- NULL
  requested <- if(method=='direct') arguments else summary_arguments
  recompute <- !is.null(requested$maxtime) || (!is.null(requested$test) && requested$test!=fit$test)
  if(kind=='aareg' && grepl('cluster(',models[[fit_name]]$formula,fixed=TRUE) && !recompute) {
    # aareg.R computes test.dfbeta in event-sorted order, but its final rowsum
    # uses the original cluster order. Refit sorted input to align those rows.
    raw <- if(method=='summary') do.call(summary,c(list(fit),summary_arguments)) else fit
    raw$call <- NULL
    saved <- options(width=width)
    original_lines <- I(sub('[[:blank:]]+$','',capture.output(do.call(print,c(list(raw),arguments)))))
    options(saved)
    spec <- models[[fit_name]]; data <- datasets[[spec$data]]
    response <- model.response(model.frame(as.formula(spec$formula),data))
    ord <- order(response[,ncol(response)-1],-response[,ncol(response)])
    fit <- do.call(aareg,c(list(formula=as.formula(spec$formula),data=data[ord,]),spec$arguments))
    adjustment <- 'Refit event-sorted input to correct R aareg cluster-order covariance.'
  }
  x <- if(method=='summary') do.call(summary,c(list(fit),summary_arguments)) else fit
  input <- if(method=='summary') snapshot(x,kind) else NULL
  x$call <- NULL
  saved <- options(width=width)
  lines <- capture.output(do.call(print,c(list(x),arguments)))
  options(saved)
  if(kind=='survreg' && method=='summary') {
    stopifnot(identical(lines[1:3],c('','Call:','NULL')))
    lines <- lines[-(1:3)]
  }
  cases[[length(cases)+1L]] <<- list(name=name,fit=fit_name,method=method,arguments=arguments,
                                    summary_arguments=summary_arguments,width=width,input=input,
                                    lines=I(sub('[[:blank:]]+$','',lines)),original_lines=original_lines,adjustment=adjustment)
}
for(name in names(fits)) {add(name,name);add(paste0(name,'_summary'),name,'summary')}
for(name in c('weibull','fixed','stratified','robust','ridge','spline','penal_scales','penal_fixed')) {
  add(paste0(name,'_narrow'),name,arguments=list(digits=5),width=40)
  add(paste0(name,'_correlation'),name,'summary',list(digits=4,signif.stars=TRUE),list(correlation=TRUE),width=55)
}
add('single_correlation','null','summary',summary_arguments=list(correlation=TRUE))
for(name in c('aalen','aalen_robust','aalen_cluster','aalen_nrisk')) {
  add(paste0(name,'_cutoff'),name,arguments=list(maxtime=400,scale=100))
  add(paste0(name,'_reweighted'),name,'summary',summary_arguments=list(test='nrisk',scale=365.25),width=45)
}
for(name in c('aalen_ties','aalen_delayed','aalen_ties_cluster')) {
  add(paste0(name,'_cutoff'),name,'summary',summary_arguments=list(maxtime=7),width=50)
  add(paste0(name,'_reweighted'),name,'summary',summary_arguments=list(test='nrisk',maxtime=7))
}
add('test_label_override','aalen',arguments=list(test='nrisk'))
add('cutoff_after_last','aalen_robust','summary',summary_arguments=list(maxtime=2000))
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion('survival')),
                datasets=lapply(datasets,as.list),models=models,cases=cases),output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
