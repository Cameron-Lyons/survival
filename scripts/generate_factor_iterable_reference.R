#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/factor_iterable_reference.json"
encode <- function(value) {
  values <- lapply(if (is.factor(value)) as.character(value) else value,function(item) {
    if (is.na(item)) return(NULL)
    if (is.numeric(item) && !is.finite(item)) return(if (item>0) "Inf" else "-Inf")
    item
  })
  list(values=values,levels=if (is.factor(value)) I(levels(value)) else NULL,
    kind=if (is.factor(value)) "factor" else typeof(value))
}
factors <- list(
  character=c("b","a","b",NA,"c","a"),
  numeric=c(30,10,30,NA,20,10),
  logical=c(TRUE,FALSE,TRUE,NA,FALSE,TRUE),
  zero=c(-0,0,2,NA,1,2),
  infinite=c(Inf,-Inf,2,NA,2,Inf),
  missing=rep(NA,4),empty=character(),
  declared=factor(c("b","a","b",NA,"c","a"),levels=c("c","unused","b","a")))
factor_cases <- lapply(names(factors),function(name) {
  value <- factors[[name]]
  expected <- as.factor(value)
  list(name=name,input=encode(value),codes=I(as.integer(expected)-1L),levels=I(levels(expected)))
})
surv <- matrix(c(.9,.8,.3,.2,.5,.4),nrow=2)
pstate <- array(c(.9,.8,.3,.2,.5,.4,.1,.2,.7,.8,.5,.6),c(2,3,2))
by_sources <- list(character=c("b","a","b"),numeric=c(2,1,2),
  declared=factor(c("b","a","b"),levels=c("b","a")),
  compound=list(g=c("b","a","b"),h=c(1,1,2)))
aggregate_cases <- list()
for (margin in c("surv","pstate")) for (name in names(by_sources)) {
  by <- by_sources[[name]]
  common <- list(n=3,time=c(1,2),n.risk=c(3,2),n.event=c(1,1),n.censor=c(0,0))
  curves <- if (margin=="surv") structure(c(common,list(surv=surv,type="right")),class="survfit") else
    structure(c(common,list(pstate=pstate,states=c("s0","s1"),type="mright")),
      class=c("survfitms","survfit"))
  for (fun in c("mean","median","min","max")) {
    fit <- aggregate(curves,by=by,FUN=get(fun))
    value <- fit[[margin]]
    # Explicit row-major nesting avoids jsonlite's array dimension convention.
    rows <- if (margin=="surv") lapply(seq_len(nrow(value)),function(row) I(value[row,])) else
      lapply(seq_len(dim(value)[1]),function(row) {
        lapply(seq_len(dim(value)[2]),function(column) I(value[row,column,]))
      })
    aggregate_cases[[length(aggregate_cases)+1L]] <- list(
      name=paste(margin,name,fun,sep="/"),margin=margin,fun=fun,
      named=is.list(by),by=if (is.list(by)) lapply(by,encode) else list(encode(by)),
      result=list(values=rows,newdata=lapply(fit$newdata,I)))
  }
}
times <- c(1,2,2,4,5,6,7,8)
event <- c(1,1,0,1,1,0,1,1)
predictor <- c(3,1,2,4,5,0,1,2)
weights <- c(1,2,.5,1,1.5,2,1,.75)
cluster_sources <- list(character=c("b","a","c","b","a","c","b","a"),
  numeric=c(30,10,20,30,10,20,30,10),
  declared=factor(c("b","a","c","b","a","c","b","a"),
    levels=c("c","unused","b","a")),
  numeric_declared=factor(c(30,10,20,30,10,20,30,10),levels=c(30,99,20,10)))
snapshot_concordance <- function(value) list(concordance=unname(value$concordance),
  count=as.list(value$count),n=value$n,var=unname(value$var),cvar=unname(value$cvar),
  dfbeta=I(unname(value$dfbeta)),dfbeta_labels=I(names(value$dfbeta)),
  influence=unname(value$influence),ranks=lapply(value$ranks,I))
concordance_cases <- list()
for (name in names(cluster_sources)) for (weighted in c(FALSE,TRUE)) {
  cluster <- cluster_sources[[name]]
  fit <- concordancefit(Surv(times,event),predictor,cluster=cluster,
    weights=if (weighted) weights else NULL,influence=3,ranks=TRUE)
  concordance_cases[[length(concordance_cases)+1L]] <- list(
    name=paste(name,weighted,sep="/"),cluster=encode(cluster),weighted=weighted,
    result=snapshot_concordance(fit))
}
strata <- list(character=cluster_sources$character,numeric=cluster_sources$numeric,
  declared=cluster_sources$declared)
legacy_cases <- lapply(names(strata),function(name) {
  groups <- strata[[name]]
  fit <- suppressWarnings(survConcordance.fit(Surv(times,event),predictor,strata=groups))
  list(name=name,strata=encode(groups),result=lapply(seq_len(nrow(fit)),function(row) as.list(fit[row,])),
    names=I(rownames(fit)))
})
reference <- list(metadata=list(generator="scripts/generate_factor_iterable_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
  factors=factor_cases,aggregate=aggregate_cases,concordance=concordance_cases,legacy=legacy_cases,
  data=list(time=I(times),event=I(event),x=I(predictor),weights=I(weights)),
  curves=list(surv=lapply(seq_len(nrow(surv)),function(row) I(surv[row,])),
    pstate=lapply(seq_len(dim(pstate)[1]),function(row) {
      lapply(seq_len(dim(pstate)[2]),function(column) I(pstate[row,column,]))
    })))
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(factor_cases),"factor,",length(aggregate_cases),"aggregate,",
  length(concordance_cases),"concordance and",length(legacy_cases),"legacy cases written to",output,"\n")
