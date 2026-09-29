#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else 'python/tests/fixtures/cox_report_reference.json'
options(digits=7, width=80)
encode <- function(x) if(is.null(x)) NULL else list(rows=I(rownames(x)),columns=I(colnames(x)),values=unname(x))
capture <- function(x, arguments=list(), width=80) {
  saved <- options(width=width, digits=7); on.exit(options(saved))
  x$call <- NULL
  I(sub('[[:blank:]]+$', '', capture.output(do.call(print,c(list(x),arguments)))))
}
snapshot <- function(x, level=.95) {
  list(model_type=if(inherits(x,'summary.coxph.penal')) 'coxph.penal' else 'coxph',
       n=x$n,nevent=x$nevent,omit=if(is.null(x$na.action)) NULL else I(as.integer(x$na.action)),
       loglik=if(length(x$loglik)==2) x$loglik[2] else x$loglik,
       null_loglik=if(length(x$loglik)==2) x$loglik[1] else NULL,
       df=if(inherits(x,'summary.coxph.penal')) I(x$df) else x$logtest['df'],
       iter=if(is.null(x$iter)) NULL else I(x$iter), print2=if(is.null(x$print2)) list() else I(x$print2),
       coefficients=encode(x$coefficients), conf_int=encode(x$conf.int),conf_level=level,
       logtest=x$logtest,waldtest=x$waldtest,sctest=x$sctest,robscore=x$robscore,
       concordance=x$concordance,used_robust=x$used.robust,cmap=encode(x$cmap),states=if(is.null(x$states)) NULL else I(x$states))
}
d <- lung
# Deterministic data keep all multistate coefficients finite.
i <- seq_len(80)
m <- data.frame(time=(i*7)%%29+1,status=as.integer(i%%3!=0),age=(i*3)%%11,
                sex=as.integer(i%%5>2), id=i, endpoint=factor(ifelse(i%%3==0,'censor',ifelse(i%%4==0,'a','b')),
                                             levels=c('censor','a','b')))
many <- m
many$endpoint <- factor(ifelse(m$status==0,'censor',paste0('event',(i*7)%%13%%6+1)), levels=c('censor',paste0('event',1:6)))
models <- list(); cases <- list(); datasets <- list()
add_model <- function(name,formula,data,arguments=list()) {
  form <- if(length(formula)>1) lapply(formula,as.formula) else as.formula(formula)
  fit <- do.call(coxph,c(list(formula=form,data=data),arguments))
  matched <- which(vapply(datasets,function(x) identical(x,data),logical(1)))
  key <- if(length(matched)) names(datasets)[matched[1]] else paste0("data",length(datasets)+1L)
  datasets[[key]] <<- data
  models[[name]] <<- list(formula=I(formula),data=key,arguments=arguments,
                          endpoint_levels=if(is.factor(data$endpoint)) I(levels(data$endpoint)) else NULL)
  fit
}
fits <- list(
  many_states=add_model('many_states',c('Surv(time,endpoint)~age+sex','1:2+1:3+1:4+1:5+1:6+1:7~(age+sex)/common'),many,list(id=many$id)),
  frailty_only=add_model('frailty_only','Surv(time,status)~frailty(inst, theta = 0.5)',d[!is.na(d$inst),]),
  ordinary=add_model('ordinary','Surv(time,status)~age+sex',d),
  robust=add_model('robust','Surv(time,status)~age+sex+cluster(inst)',d),
  stratified=add_model('stratified','Surv(time,status)~age+strata(sex)',d),
  missing=add_model('missing','Surv(time,status)~age+ph.ecog+wt.loss',d),
  aliased=add_model('aliased','Surv(time,status)~age+I(age * 2)+sex',d),
  null=add_model('null','Surv(time,status)~1',d),
  null_missing=add_model('null_missing','Surv(time,status)~offset(wt.loss/100)',d),
  ridge=add_model('ridge','Surv(time,status)~ridge(age,theta=1)+sex',d),
  spline=add_model('spline','Surv(time,status)~pspline(age, df = 4)+sex',d),
  frailty=add_model('frailty',"Surv(time,status)~age+sex+frailty(inst, theta = 0.5)",d[!is.na(d$inst),]),
  gaussian=add_model('gaussian',"Surv(time,status)~age+sex+frailty(inst, distribution = \"gaussian\", theta = 0.5, sparse = FALSE)",d[!is.na(d$inst),]),
  terms=add_model('terms','Surv(time,status)~ridge(age,theta=1)+factor(ph.ecog)',d[!is.na(d$ph.ecog),]),
  multistate=add_model('multistate','Surv(time,endpoint)~age+sex',m,list(id=m$id)),
  shared=add_model('shared',c('Surv(time,endpoint)~age+sex','1:2+1:3~age/common'),m,list(id=m$id)),
  proportional=add_model('proportional',c('Surv(time,endpoint)~age+sex','1:2+1:3~1/shared'),m,list(id=m$id)))
add <- function(name, fit_name, method='direct', arguments=list(), summary_arguments=list(), width=80) {
  fit <- fits[[fit_name]]
  x <- if(method=='summary') do.call(summary,c(list(fit),summary_arguments)) else fit
  input <- if(method=='summary' && !inherits(x,'coxph.null')) snapshot(x,if(is.null(summary_arguments$conf.int)) .95 else summary_arguments$conf.int) else NULL
  cases[[length(cases)+1L]] <<- list(name=name,fit=fit_name,method=method,arguments=arguments,
                                    summary_arguments=summary_arguments,width=width,input=input,
                                    lines=capture(x,arguments,width))
}
for(name in names(fits)) {
  add(name,name)
  add(paste0(name,'_summary'),name,'summary')
}
add('multistate_stars','multistate',arguments=list(signif.stars=TRUE))
add('summary_ci975','ordinary','summary',summary_arguments=list(conf.int=.975))
add('ordinary_stars','ordinary',arguments=list(signif.stars=TRUE))
add('ordinary_precision','ordinary',arguments=list(digits=7),width=40)
add('robust_narrow','robust','summary',list(digits=6),width=35)
add('summary_no_stars','ordinary','summary',list(signif.stars=FALSE))
add('summary_no_ci','ordinary','summary',summary_arguments=list(conf.int=FALSE))
add('summary_scaled_ci90','ordinary','summary',summary_arguments=list(conf.int=.9,scale=10))
add('spline_short','spline',arguments=list(maxlabel=12,digits=5),width=55)
add('spline_summary_short','spline','summary',list(maxlabel=12,digits=5),width=55)
add('term_tests','terms',arguments=list(terms=TRUE))
add('term_tests_summary','terms','summary',summary_arguments=list(terms=TRUE))
for(name in c('multistate','shared','proportional')) {
  add(paste0(name,'_expanded'),name,'summary',list(expand=TRUE))
  add(paste0(name,'_narrow'),name,arguments=list(digits=5),width=40)
  add(paste0(name,'_expanded_narrow'),name,'summary',list(expand=TRUE,digits=5),width=40)
}
# Isolate number-formatting edge cases from numerical solver precision.
matrices <- list()
a <- cbind(coef=c(.0123456,-3.98765,0,NA,1e-12,1e3),se=c(.0234567,2.65432,0,NA,2e-13,10),
           z=c(1.23456,-2.45678,0,NA,5,100),p=c(.217,.014,1,NA,2e-7,0))
rownames(a) <- c('a','long coefficient','zero','aliased','tiny','large')
for(digits in c(1,3,4,7,12)) for(width in c(35,80)) for(title in c('', '1:2, 1:3')) for(stars in c(FALSE,TRUE)) {
  b <- a
  if(nchar(title)) names(dimnames(b)) <- c(title,'')
  saved <- options(width=width)
  lines <- capture.output(printCoefmat(b,digits=digits,signif.stars=stars,has.Pvalue=TRUE,P.values=TRUE))
  options(saved)
  matrices[[length(matrices)+1L]] <- list(table=encode(a),stars=stars,digits=digits,width=width,title=if(nchar(title)) title else NULL,
                                         lines=I(sub('[[:blank:]]+$','',lines)))
}
numeric_matrices <- list()
for(title in c('1:2','a long transition title')) for(w in 25:45) {
  b <- matrix(c(1.234,2.5,3,4,5,6,7,8),2,dimnames=list(c('short','a long name'),c('coef','exp(coef)','lower .95','upper .95')))
  names(dimnames(b)) <- c(title,'')
  saved <- options(width=w)
  lines <- capture.output(print(b,digits=5))
  options(saved)
  numeric_matrices[[length(numeric_matrices)+1L]] <- list(table=encode(b),digits=5,width=w,title=title,lines=I(sub('[[:blank:]]+$','',lines)))
}
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion('survival')),
                datasets=lapply(datasets,as.list),models=models,cases=cases,matrices=matrices,numeric_matrices=numeric_matrices),output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null')
