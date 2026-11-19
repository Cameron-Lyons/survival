#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/aj_mean_infinite_reference.json"

encode <- function(value) {
    if (is.null(value)) return(NULL)
    if (is.matrix(value)) return(lapply(seq_len(nrow(value)), function(i) encode(value[i,])))
    if (is.list(value)) return(lapply(value, encode))
    lapply(unname(value), function(item) {
        if (is.na(item)) return(NULL)
        if (is.numeric(item) && !is.finite(item)) return(if (item > 0) "Inf" else "-Inf")
        item
    })
}
influences <- function(value) {
    if (is.null(value)) return(NULL)
    if (!is.list(value)) value <- list(value)
    lapply(unname(value), function(curve) lapply(seq_len(dim(curve)[1]), function(i)
        lapply(seq_len(dim(curve)[2]), function(t) encode(curve[i,t,]))))
}
snapshot <- function(fit) {
    fields <- c("time", "n.risk", "n.event", "n.censor", "n.enter", "n.transition",
                "pstate", "cumhaz", "std.err", "std.chaz", "std.auc")
    result <- setNames(lapply(fields, function(field) encode(fit[[field]])),
                       gsub("\\.", "_", fields))
    result$n <- encode(fit$n)
    result$n_id <- encode(fit$n.id)
    result$states <- encode(fit$states)
    result$p0 <- if (is.matrix(fit$p0)) encode(fit$p0) else list(encode(fit$p0))
    result$t0 <- encode(fit$t0)
    result["strata"] <- list(encode(fit$strata))
    result["influence_pstate"] <- list(influences(fit$influence.pstate))
    result
}
capture_table <- function(fit, rmean) {
    tryCatch({
        value <- summary(fit, rmean=rmean)
        list(values=encode(value$table), columns=encode(colnames(value$table)),
             rows=encode(rownames(value$table)), ends=encode(value$rmean.endtime))
    }, error=function(e) list(error=conditionMessage(e)))
}
input <- function(time, state=c(1L,2L,0L), start=NULL, weights=NULL, strata=NULL,
                  id=NULL, istate=NULL, cluster=NULL) {
    list(time=time, state=state, states=c("a","b"), start=start, weights=weights,
         strata=strata, id=id, istate=istate,
         istate_levels=if (is.null(istate)) NULL else c("a","b"), cluster=cluster)
}
inputs <- list(
    finite=input(c(1,2,3)),
    weighted_groups=input(c(1,2,3,4,1,2,3,5), c(1L,2L,0L,0L,2L,1L,0L,0L),
        weights=c(.5,1,2,1,.75,2,.5,1.5), strata=c(1L,1L,1L,1L,2L,2L,2L,2L),
        cluster=c(1L,1L,2L,3L,4L,5L,5L,6L)),
    estimated_initial=input(c(1,2,3,4,5), c(0L,0L,2L,0L,0L),
        weights=c(.5,1,2,1.5,1), istate=c("a","b","a","b","a")),
    counting=input(c(2,4,3,5), c(1L,2L,0L,1L), start=c(0,1,0,2), id=1:4,
        weights=c(.5,1,2,1.5)),
    positive_censor=input(c(1,2,Inf)),
    positive_event=input(c(1,2,Inf), c(1L,0L,2L)),
    positive_only=input(rep(Inf,3)),
    negative_event=input(c(-Inf,1,2)),
    negative_only=input(rep(-Inf,3)),
    counting_negative_entry=input(c(1,2,Inf), start=c(-Inf,0,1), id=1:3),
    mixed_groups=input(c(1,2,Inf,3,1,2,3,Inf), c(1L,2L,0L,1L,2L,1L,0L,0L),
        weights=c(.5,1,2,1,.75,2,.5,1.5), strata=c(1L,1L,1L,1L,2L,2L,2L,2L))
)
cases <- list()
for (name in names(inputs)) for (influence in c(FALSE,TRUE)) for (time0 in c(FALSE,TRUE)) {
    source <- inputs[[name]]
    columns <- source[intersect(names(source),
        c("time","state","start","weights","strata","id","istate","cluster"))]
    frame <- as.data.frame(columns[!vapply(columns,is.null,logical(1))])
    frame$event <- factor(c("censor","a","b")[source$state+1L],
                          levels=c("censor","a","b"))
    response <- if (is.null(source$start)) "Surv(time,event)" else "Surv(start,time,event)"
    formula <- as.formula(paste(response,"~",if(is.null(source$strata)) "1" else "strata"))
    arguments <- list(formula=formula, data=frame, influence=influence, time0=time0,
                      timefix=FALSE, entry=!is.null(source$start), conf.type="plain")
    for (arg in c("weights","id","cluster"))
        if (!is.null(source[[arg]])) arguments[[arg]] <- source[[arg]]
    if (!is.null(source$istate)) arguments$istate <- factor(source$istate,levels=source$istate_levels)
    fit <- do.call(survival::survfit,arguments)
    tables <- lapply(list("none","common","individual",1.5,2,2.5,4,Inf),function(rmean) {
        list(rmean=if(is.character(rmean)) rmean else encode(rmean), expected=capture_table(fit,rmean))
    })
    cases[[length(cases)+1L]] <- list(
        name=paste(name,influence,time0,sep="_"), input=lapply(source,encode),
        options=list(influence=influence,time0=time0,timefix=FALSE,
                     entry=!is.null(source$start),conf_type="plain"),
        fit=snapshot(fit), tables=tables)
}
write_json(list(metadata=list(r_version=R.version.string,
    survival_version=as.character(packageVersion("survival")),
    generator="scripts/generate_aj_mean_infinite_reference.R"), cases=cases),
    output, auto_unbox=TRUE, pretty=TRUE, digits=NA, null="null", na="null")
