#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/km_lowlevel_reference.json"
cases <- list()
time <- c(1,1,2,3,3,4,5,6)
status <- c(1,0,1,1,0,1,0,0)
x <- factor(rep(c("b","a"),4), levels=c("b","a"))
y <- Surv(time,status)
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i,])))
vec <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.numeric(x) && any(is.infinite(x))) x <- as.character(x)
    I(unname(x))
}
influence <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.matrix(x)) x <- list(x)
    lapply(x, function(m) if (is.null(m)) NULL else list(values=rows(m), cluster=I(rownames(m))))
}
add <- function(name, group=x, response=y, ...) {
    values <- list(...)
    warnings <- character()
    # Force defaults before filtering. R's bare fitter otherwise first evaluates
    # its lazy weights after subsetting x, and can access missing id/cluster.
    result <- withCallingHandlers(tryCatch(do.call(survfitKM,modifyList(list(x=group,y=response,weights=rep(1,length(group)),id=NULL,cluster=NULL),values,keep.null=TRUE)),
        error=function(e) list(error=conditionMessage(e))),
        warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")})
    expected <- if (!is.null(result$error)) result else list(
        n=vec(result$n), time=vec(result$time), n_risk=vec(result$n.risk), n_event=vec(result$n.event),
        n_censor=vec(result$n.censor), n_enter=vec(result$n.enter), n_id=vec(result$n.id),
        surv=vec(result$surv), cumhaz=vec(result$cumhaz), std_err=vec(result$std.err),
        std_chaz=vec(result$std.chaz), lower=vec(result$lower), upper=vec(result$upper),
        counts=if(is.null(result$counts)) NULL else rows(result$counts),
        strata=if(is.null(result$strata)) NULL else as.list(result$strata),
        type=result$type, t0=result$t0, logse=result$logse, conf_int=result$conf.int,
        conf_type=result$conf.type, conf_lower=result$conf.lower,
        influence_surv=influence(result$influence.surv), influence_chaz=influence(result$influence.chaz))
    vector_args <- c("weights","id","cluster")
    for (name_ in intersect(vector_args,names(values))) values[[name_]] <- I(values[[name_]])
    names(values) <- gsub("\\.","_",names(values))
    cases[[length(cases)+1L]] <<- list(name=name, group=vec(as.character(group)), levels=vec(levels(group)),
        response=rows(response), response_type=attr(response,"type"), arguments=values,
        expected=expected,warnings=I(warnings))
}
for (stype in 1:2) for (ctype in 1:2) {
    tag <- paste(stype,ctype,sep="_")
    add(paste0("right_",tag),stype=stype,ctype=ctype)
    add(paste0("weighted_",tag),stype=stype,ctype=ctype,weights=c(.5,1,2,1.5,.5,1,2,1.5))
    add(paste0("cluster_",tag),stype=stype,ctype=ctype,cluster=c("z","a","z","b","a","b","z","a"),influence=TRUE)
    add(paste0("id_",tag),stype=stype,ctype=ctype,id=rep(c("a","b","c","d"),2),influence=TRUE)
}
for (conf in c("log","log-log","plain","none","logit","arcsin"))
    for (lower in c("usual","peto","modified")) add(paste("conf",conf,lower,sep="_"),conf.type=conf,conf.lower=lower)
for (level in 0:3) add(paste0("influence_",level),group=factor(rep("one",8)),influence=level)
add("unused_levels",group=factor(x,levels=c("empty1","b","empty2","a","empty3")))
add("single_observed_level",group=factor(rep("b",8),levels=c("a","b","c")))
add("empty_after_start",group=factor(c(rep("early",4),rep("late",4))),start.time=4)
add("negative_times",response=Surv(time-4,status))
add("near_ties",response=Surv(c(1,1+1e-10,2,3,3+1e-10,4,5,6),status))
add("start_time",start.time=2.5,cluster=seq_len(8),influence=3)
add("time0_ignored",time0="unused")
add("empty_cluster",cluster=character())
add("empty_id",id=integer())
add("no_se",se.fit=FALSE,influence=TRUE,cluster=seq_len(8))
add("no_interval",conf.int=FALSE)
add("custom_confidence",conf.int=.8,conf.type="plain")
add("robust_false",robust=FALSE,influence=3,cluster=seq_len(8))
add("old_type_override",type="fh2",stype=99,ctype=99)
add("zero_weights",weights=c(0,1,1,0,1,1,1,1))
add("near_unit_weights",weights=rep(1+1e-10,8))
add("unit_integer_weights",weights=rep(2,8))
add("all_censored",response=Surv(time,rep(0,8)))
add("all_events",response=Surv(time,rep(1,8)))
counting <- Surv(c(0,2,0,2,1,3,0,4),c(2,5,2,6,3,7,4,8),c(0,1,1,0,0,1,0,1))
ids <- rep(c("z","a","b","c"),each=2)
for (stype in 1:2) for (ctype in 1:2) {
    tag <- paste(stype,ctype,sep="_")
    add(paste0("counting_",tag),response=counting,stype=stype,ctype=ctype,id=ids)
    add(paste0("entry_",tag),response=counting,stype=stype,ctype=ctype,id=ids,entry=TRUE,influence=TRUE)
    add(paste0("counting_weighted_",tag),response=counting,stype=stype,ctype=ctype,weights=rep(c(.5,1.5),4))
}
add("entry_single",group=factor(rep("one",8)),response=counting,id=ids,entry=TRUE,influence=TRUE)
add("entry_no_id",response=counting,entry=TRUE)
add("entry_right",id=ids,entry=TRUE)
add("counting_explicit_robust",response=counting,robust=TRUE)
add("bad_type",type="unknown")
add("bad_stype",stype=3)
add("bad_ctype",ctype=3)
add("bad_conf",conf.type="unknown")
add("bad_influence",influence=4)
add("all_removed",start.time=100)
add("wrong_response",response=Surv(time,status,type="left"))
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion("survival")),cases=cases),
    output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na="null",null="null")
