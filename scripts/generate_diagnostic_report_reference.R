#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else 'python/tests/fixtures/diagnostic_report_reference.json'
options(digits=7,width=80)
labels <- function(x) if(is.null(x)) NULL else I(x)
encode <- function(x) if(is.null(x)) NULL else list(rows=labels(rownames(x)),columns=labels(colnames(x)),values=unname(x))
datasets <- list(lung=as.list(lung))
cases <- list(); objects <- list(); specs <- list()
add_object <- function(name,kind,formula=NULL,data='lung',arguments=list(),x=NULL) {
  if(is.null(x)) x<-suppressWarnings(do.call(get(kind),c(list(as.formula(formula),data=as.data.frame(datasets[[data]])),arguments)))
  objects[[name]] <<- x
  specs[[name]] <<- list(kind=kind,formula=formula,data=data,arguments=arguments)
}
d <- lung;d$status<-as.integer(d$status==2);d$id<-seq_len(nrow(d));d$subcoh<-as.integer(d$id%%3==0)
sizes <- c("1"=sum(d$sex==1),"2"=sum(d$sex==2))
d <- d[d$status==1 | d$subcoh==1,]
datasets$cohort <- as.list(d)
for(method in c('Prentice','SelfPrentice','LinYing','I.Borgan','II.Borgan')) {
  stratified <- grepl('Borgan',method)
  arguments <- list(subcoh=d$subcoh,id=d$id,method=method,cohort.size=if(stratified) sizes else 228)
  if(stratified) arguments$stratum<-d$sex
  add_object(method,'cch',if(stratified) 'Surv(time,status)~age' else 'Surv(time,status)~age+sex','cohort',arguments)
}
add_object('Prentice_single','cch','Surv(time,status)~age','cohort',list(subcoh=d$subcoh,id=d$id,method='Prentice',cohort.size=228))
add_object('LinYing_robust','cch','Surv(time,status)~age+sex','cohort',list(subcoh=d$subcoh,id=d$id,method='LinYing',robust=TRUE,cohort.size=228))
fit1<-coxph(Surv(time,status)~age+sex,lung)
fit2<-coxph(Surv(time,status)~ph.ecog,lung)
add_object('zph','cox_zph',x=cox.zph(fit1))
add_object('zph_identity','cox_zph',x=cox.zph(fit1,transform='identity',global=FALSE))
add_object('zph_spline','cox_zph',x=cox.zph(coxph(Surv(time,status)~pspline(age,df=4)+sex,lung)))
add_object('conditional','clogit','status~sex+strata(group)','matched',x={
  datasets$matched <- as.list(data.frame(status=as.integer((1:60)%%3==0),sex=as.integer((1:60)%%7>2),group=rep(1:20,each=3)))
  clogit(status~sex+strata(group),as.data.frame(datasets$matched))
})
for(item in list(c('concordance','Surv(time,status)~age'),c('kept_strata','Surv(time,status)~age+strata(sex)'),c('omitted','Surv(time,status)~wt.loss'))) {
  add_object(item[1],'concordance',item[2])
}
add_object('reverse','concordance','Surv(time,status)~age',arguments=list(reverse=TRUE))
add_object('weights','concordance','Surv(time,status)~age',arguments=list(weights=rep(c(.5,1,2),length.out=nrow(lung))))
y<-Surv(lung$time,lung$status)
add_object('multiple','concordancefit',x=concordancefit(y,cbind(age=lung$age,sex=lung$sex)))
add_object('multiple_no_variance','concordancefit',x={result<-objects$multiple;result$var<-NULL;result})
add_object('no_variance','concordancefit',x=concordancefit(y,lung$age,std.err=FALSE))
add_object('multiple_none_comparable','concordancefit',x=concordancefit(Surv(1:3,c(0,0,0)),cbind(first=1:3,second=3:1)))
add_object('none_comparable','concordancefit',x=concordancefit(Surv(1:3,c(0,0,0)),1:3))
for(item in list(c('legacy','Surv(time,status)~age'),c('legacy_strata','Surv(time,status)~age+strata(sex)'),c('legacy_omitted','Surv(time,status)~wt.loss'))) {
  add_object(item[1],'survConcordance',item[2])
}
for(item in list(c('logrank','Surv(time,status)~sex'),c('logrank_strata','Surv(time,status)~sex+strata(inst)'),c('logrank_missing','Surv(time,status)~ph.ecog'))) {
  add_object(item[1],'survdiff',item[2])
}
add_object('rho','survdiff','Surv(time,status)~sex',arguments=list(rho=1))
datasets$expected <- as.list(data.frame(time=1:8,status=c(1,0,1,1,0,0,1,0),expected=c(.9,.8,.7,.6,.5,.4,.3,.2)))
add_object('one_sample','survdiff','Surv(time,status)~offset(expected)','expected')
datasets$no_expected <- list(time=c(1,2,3),status=c(0,1,1),group=c('a','b','b'))
add_object('zero_expected','survdiff','Surv(time,status)~group','no_expected')
snapshot <- function(x,method) {
  omit <- if(is.null(x$na.action)) NULL else I(as.integer(x$na.action))
  if(method=='cox.zph') return(list(table=encode(x$table),transform=x$transform))
  if(method=='summary.cch') return(list(method=x$method,stratified=x$stratified,cohort_size=I(as.numeric(x$cohort.size)),subcohort_size=I(as.numeric(x$subcohort.size)),stratum_names=if(x$stratified) I(names(x$subcohort.size)) else NULL,table=encode(x$coefficients)))
  if(method=='concordance') return(list(concordance=I(x$concordance),variance=unname(x$var),count=if(is.matrix(x$count)) encode(x$count) else list(values=I(x$count),columns=I(names(x$count))),n=x$n,omit=omit))
  if(method=='survConcordance') return(list(concordance=x$concordance,std_err=x$std.err,stats=if(is.matrix(x$stats)) encode(x$stats) else list(values=I(x$stats),columns=I(names(x$stats))),n=x$n,omit=omit))
  if(method=='survdiff') return(list(n=I(as.numeric(x$n)),groups=labels(names(x$n)),obs=unname(x$obs),exp=unname(x$exp),variance=unname(x$var),chisq=x$chisq,pvalue=x$pvalue,omit=omit))
  NULL
}
add <- function(name,object,method,arguments=list(),width=80) {
  x<-objects[[object]]
  if(method=='summary.cch') x<-summary(x)
  input<-snapshot(x,method)
  x$call<-NULL;x$userCall<-NULL
  saved<-options(width=width)
  f<-get(paste0('print.',method),asNamespace('survival'))
  lines<-capture.output(do.call(f,c(list(x),arguments)))
  options(saved)
  if(method %in% c('cch','summary.cch')) lines<-lines[lines!='Call: NULL']
  cases[[length(cases)+1L]]<<-list(name=name,object=object,method=method,arguments=arguments,width=width,input=input,lines=I(sub('[[:blank:]]+$','',lines)))
}
for(name in names(objects)) {
  kind<-specs[[name]]$kind
  method<-switch(kind,cox_zph='cox.zph',concordancefit='concordance',kind)
  # print.clogit calls NextMethod and needs the generic dispatch context.
  if(method=='clogit') method<-'coxph'
  add(name,name,method)
  if(method=='cch') add(paste0(name,'_summary'),name,'summary.cch')
}
add('zph_stars','zph','cox.zph',list(digits=5,signif.stars=TRUE),width=35)
add('spline_narrow','zph_spline','cox.zph',list(digits=6),width=30)
add('borgan_narrow','I.Borgan','summary.cch',list(digits=8),width=35)
add('concordance_precision','concordance','concordance',list(digits=7),width=35)
add('concordance_matrix_precision','multiple','concordance',list(digits=2),width=40)
add('count_precision','weights','concordance',list(digits=2),width=35)
add('legacy_narrow','legacy_strata','survConcordance',width=35)
add('logrank_precision','logrank_strata','survdiff',list(digits=5),width=40)
add('one_sample_precision','one_sample','survdiff',list(digits=6),width=25)
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion('survival')),datasets=datasets,objects=specs,cases=cases),output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na='string')
