#!/usr/bin/env Rscript
# Independent stock-R tables for streaming restricted-mean calculations.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/survmean_reference.json"
data <- data.frame(time=c(1,1,2,4,7,10,2,3,5,6,8,12),
    status=c(1,0,1,0,1,0,0,1,1,0,1,1), group=rep(c("a","b"),each=6),
    weight=c(.5,1,2,1.5,.5,1,2,.75,1.25,1,.5,2))
counting <- transform(data, start=pmax(0,time-2))
fits <- list(
    one=survfit(Surv(time,status)~1,data),
    groups=survfit(Surv(time,status)~group,data),
    weighted=survfit(Surv(time,status)~group,data,weights=weight),
    no_limits=survfit(Surv(time,status)~group,data,conf.type="none"),
    all_censored=survfit(Surv(time,rep(0,nrow(data)))~1,data),
    all_events=survfit(Surv(time,rep(1,nrow(data)))~1,data),
    counting=survfit(Surv(start,time,status)~group,counting,weights=weight),
    conditional=survfit(Surv(time,status)~group,data,start.time=2.5),
    negative=survfit(Surv(time-5,status)~group,data))
# Directly exercise the midpoint/tolerance and missing confidence-band rules
# on a valid stock fit; the numerical table function accepts these bands.
plateau <- fits$one
plateau$surv <- c(1,.75,.5,.5,.5-1e-10,.25,.1,0,0,0)
plateau$lower <- c(NA,.5,.5,.49,.4,.3,.2,.1,0,0)
plateau$upper <- c(1,1,1,.75,.5+1e-10,.5,.5,.4,NA,.1)
stopifnot(length(plateau$time)==length(plateau$surv))
fits$plateau <- plateau
encode <- function(x) if (is.null(x)) NULL else I(unname(x))
sources <- list()
cases <- list()
stock_mean <- get("survmean",asNamespace("survival"))
for (name in names(fits)) for (zero in c(FALSE,TRUE)) {
    fit <- if (zero) survfit0(fits[[name]]) else fits[[name]]
    key <- paste(name,if (zero) "time0" else "raw",sep="_")
    fields <- c("time","surv","n.risk","n.event","n","strata","n.id","lower","upper")
    source <- setNames(lapply(fields,function(field) encode(fit[[field]])),
                       gsub("\\.","_",fields))
    source$t0 <- if (is.null(fit$t0)) min(0,fit$time) else fit$t0
    sources[[key]] <- source
    # Include an exact observed cutoff and a cutoff beyond all observations.
    cutoffs <- list("none","common","individual",
        fit$time[ceiling(length(fit$time)/2)],max(fit$time)+2)
    for (scale in c(1,.1,2.5,365.25)) for (cutoff in cutoffs) {
        table <- stock_mean(fit,scale,cutoff)
        matrix <- if (is.matrix(table$matrix)) table$matrix else
            matrix(table$matrix,nrow=1,dimnames=list(NULL,names(table$matrix)))
        expected <- list()
        columns <- c(records="records",n_max=if (is.null(fit$n.id)) "n.max" else "n.id",
            n_start="n.start",events="events",rmean="rmean",se_rmean="se(rmean)",
            median="median",lower="0.95LCL",upper="0.95UCL")
        for (field in names(columns)) expected[field] <- list(
            if (columns[[field]] %in% colnames(matrix)) encode(matrix[,columns[[field]]]) else NULL)
        expected$end_time <- encode(table$end.time)
        cases[[length(cases)+1L]] <- list(name=paste(key,scale,cutoff,sep="_"),
            source=key,scale=scale,rmean=cutoff,expected=expected)
    }
}
write_json(list(r_version=R.version.string,
    survival_version=as.character(packageVersion("survival")),
    generator="scripts/generate_survmean_reference.R",sources=sources,cases=cases),
    output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na="null",null="null")
