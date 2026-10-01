#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/cox_time_transform_penalty_reference.json"
d <- ovarian
d$x <- (d$age-60)/10
d$g <- factor(d$rx,levels=c(2,1,9))
d$w <- rep(c(.7,1,1.4),length.out=nrow(d))
d$o <- (seq_len(nrow(d))%%5-2)/20
d$start <- pmin(seq_len(nrow(d))%%3,d$futime/2)
cases <- list()
specs <- list()
for (distribution in c("gamma","gaussian","t")) for (sparse in c(TRUE,FALSE)) {
  for (method in c("efron","breslow")) specs[[length(specs)+1L]] <- list(
    kind="fixed", distribution=distribution, sparse=sparse, method=method)
}
for (distribution in c("gamma","gaussian")) for (sparse in c(TRUE,FALSE)) {
  for (kind in c("df","search")) specs[[length(specs)+1L]] <- list(
    kind=kind, distribution=distribution,sparse=sparse,method="efron")
}
for (kind in c("sparse_only","weighted","counting","factor","subset","missing","raw_codes")) {
  specs[[length(specs)+1L]] <- list(kind=kind,distribution="gamma",sparse=TRUE,method="efron")
}
for (kind in c("spline","spline_fixed","spline_degree","spline_combined","spline_unpenalized")) {
  specs[[length(specs)+1L]] <- list(kind=kind,distribution=NULL,sparse=FALSE,method="efron")
}
for (spec in specs) {
  kind<-spec$kind;distribution<-spec$distribution;sparse<-spec$sparse
  data<-d
  if(kind=="missing")data$x[c(2,9)]<-NA
  if(kind=="counting")data$futime<-data$futime+seq_len(nrow(data))/1000
  formula<-paste(if(kind=="counting")"Surv(start,futime,fustat)" else "Surv(futime,fustat)",
    "~",if(kind=="sparse_only")"" else "x +",
    if(grepl("spline",kind))"tt(x)" else if(kind=="factor")"tt(g)" else "tt(rx)",
    if(kind%in%c("weighted","counting"))"+ strata(resid.ds) + offset(o)" else "")
  fun<-function(x,t,...) {
    if(kind=="spline")return(pspline(x*log(t),df=3))
    if(kind=="spline_fixed")return(pspline(x*log(t),theta=.4))
    if(kind=="spline_degree")return(pspline(x*log(t),df=3,degree=2))
    if(kind=="spline_combined")return(pspline(x*log(t),df=2,nterm=6,
      combine=c(1,1,2,2,3,3,4,4)))
    if(kind=="spline_unpenalized")return(pspline(x*log(t),df=2,nterm=3,degree=1,penalty=FALSE))
    options<-list(x=x,distribution=distribution,sparse=sparse)
    if(kind=="df")options$df<-.7 else if(kind!="search")options$theta<-.4
    value<-do.call(frailty,options)
    if(kind=="raw_codes")value[]<-ifelse(value==1,41,73)
    value
  }
  precise<-kind%in%c("weighted","counting")
  call<-list(formula=as.formula(formula),data=data,tt=fun,ties=spec$method,robust=FALSE,x=TRUE,
    control=coxph.control(eps=if(precise)1e-14 else 1e-10,
      toler.chol=if(precise)1e-15 else .Machine$double.eps^.75,iter.max=100,outer.max=50),
    na.action=if(kind=="missing")na.exclude else na.omit)
  if(kind%in%c("weighted","counting"))call$weights<-data$w
  if(kind=="subset")call$subset<-seq_len(nrow(data))%%4!=0
  fit<-do.call(coxph,call)
  raw_variance<-fit$var;raw_df<-fit$df;raw_fvar<-fit$fvar
  raw_summary<-summary(fit)$coefficients
  information_reference<-NULL
  if(kind%in%c("weighted","counting")) {
    # coxfit5b omits the event weight from the sparse indicator diagonal.
    # Form that diagonal and the dense/cross blocks from the independent
    # weighted risk-set covariance, then use the documented sparse block
    # approximation and gamma penalty's second derivatives.
    codes<-match(fit$x[,2],sort(unique(fit$x[,2])))
    groups<-max(codes)
    basis<-cbind(diag(groups)[codes,,drop=FALSE],fit$x[,1])
    information<-matrix(0,ncol(basis),ncol(basis))
    for(group in unique(fit$strata)) {
      rows<-which(fit$strata==group)
      risk<-fit$weights[rows]*exp(fit$linear.predictors[rows])
      probability<-risk/sum(risk)
      mean<-colSums(probability*basis[rows,,drop=FALSE])
      weight<-sum(fit$weights[rows]*fit$y[rows,2])
      information<-information+weight*(crossprod(basis[rows,,drop=FALSE],
        probability*basis[rows,,drop=FALSE])-tcrossprod(mean))
    }
    penalized<-information
    likelihood<-function(delta) {
      predictor<-fit$linear.predictors+c(basis%*%delta)
      sum(vapply(unique(fit$strata),function(group) {
        rows<-which(fit$strata==group)
        event<-fit$weights[rows]*fit$y[rows,2]
        sum(event*predictor[rows])-sum(event)*log(sum(fit$weights[rows]*exp(predictor[rows])))
      },numeric(1)))
    }
    step<-1e-4;zero<-numeric(ncol(basis));center<-likelihood(zero)
    numerical<-vapply(seq_len(ncol(basis)),function(j) {
      delta<-zero;delta[j]<-step
      -(likelihood(delta)-2*center+likelihood(-delta))/step^2
    },numeric(1))
    stopifnot(max(abs(numerical-diag(information)))<1e-5)
    penalized[seq_len(groups),seq_len(groups)]<-diag(diag(information)[seq_len(groups)]+
      fit$coxlist1$second)
    inverse<-solve(penalized)
    dense<-groups+1L
    fit$var<-inverse[dense,dense,drop=FALSE]
    fit$fvar<-diag(inverse)[seq_len(groups)]
    fit$var2<-fit$var-crossprod(inverse[seq_len(groups),dense,drop=FALSE],
      fit$coxlist1$second*inverse[seq_len(groups),dense,drop=FALSE])
    fit$df<-c(fit$var2/fit$var,groups-sum(fit$fvar*fit$coxlist1$second))
    information_reference<-"Independent weighted risk-set covariance with the sparse frailty diagonal approximation and stock gamma penalty derivatives; stock coxfit5b omits the event weight in that diagonal."
  }
  # The stock penalized cleanup residuals do not satisfy the risk-set sum
  # identity here. Use the ordinary R Cox kernel at its exact fitted LP.
  expanded<-data.frame(t=fit$y[,1],event=fit$y[,2],group=fit$strata,lp=fit$linear.predictors)
  residual_fit<-coxph(Surv(t,event)~offset(lp)+strata(group),expanded,weights=fit$weights,
    ties=spec$method,control=coxph.control(iter.max=0))
  names<-paste(kind,distribution,if(sparse)"sparse" else "dense",spec$method,sep="/")
  cat(names,"\n")
  tab<-summary(fit)$coefficients
  summary_columns<-if(is.null(colnames(tab)))NULL else I(colnames(tab))
  cases[[length(cases)+1L]]<-c(spec,list(name=names,formula=formula,
    coefficients=I(if(is.null(coef(fit)))numeric() else unname(coef(fit))),
    coefficient_names=I(if(is.null(names(coef(fit))))character() else names(coef(fit))),
    variance=if(is.null(fit$var))I(list()) else unname(fit$var),
    raw_variance=if(is.null(raw_variance))I(list()) else unname(raw_variance),
    raw_df=if(is.null(raw_df))NULL else I(raw_df),raw_fvar=if(is.null(raw_fvar))NULL else I(raw_fvar),
    raw_summary=unname(raw_summary),information_reference=information_reference,
    vcov_error=tryCatch({vcov(fit);NULL},error=function(e)conditionMessage(e)),
    loglik=I(fit$loglik),df=if(is.null(fit$df))NULL else I(fit$df),
    frail=if(is.null(fit$frail))NULL else I(fit$frail),fvar=if(is.null(fit$fvar))NULL else I(fit$fvar),
    lp=I(unname(fit$linear.predictors)),x=unname(fit$x),assign=lapply(fit$assign,function(x)I(x-1L)),
    matrix_assign=I(attr(fit$x,"assign")),
    y=unname(unclass(fit$y)),pterms=if(is.null(fit$pterms))NULL else I(fit$pterms),
    summary=unname(tab),summary_columns=summary_columns,summary_names=I(rownames(tab)),
    print2=summary(fit)$print2,martingale=I(unname(residuals(residual_fit))),
    raw_martingale=I(unname(residuals(fit)))))
}
jsonlite::write_json(list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival")),
  residual_reference="Stock fits, with weighted sparse covariance/df/summary corrections explicitly marked per case. Martingale residuals use the ordinary R Cox kernel at the same fitted linear predictors, without optimization. Raw stock covariance, df, fvar, summaries and residuals are retained."),
  data=d,levels=I(levels(d$g)),cases=cases),output,auto_unbox=TRUE,pretty=TRUE,digits=17,
  na="null",null="null",dataframe="columns")
cat(length(cases),"Cox time-transform penalty references written\n")
