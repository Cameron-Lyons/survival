#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/summary_counts_reference.json"

data <- data.frame(
    time=c(1,1,2,3,3,4,5,6,2,3,4,5,6,7,8,9),
    start=c(0,0,.5,1,0,2,1,3,0,1,2,0,3,4,2,6),
    status=c(1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1),
    event=factor(c("a","censor","b","a","censor","b","censor","censor",
                   "b","censor","a","censor","b","a","censor","b"),
                 levels=c("censor","a","b")),
    group=rep(c("a","b"),each=8),
    x=c(2,0,1,3,1,0,2,3,0,2,1,0,3,1,2,3),
    id=1:16
)
weights <- c(.5,1,2,1.5,.5,1,2,1.5,1,.5,1.5,2,1,.5,2,1.5)
fits <- list()
specs <- list()
for (counting in c(FALSE, TRUE)) {
    response <- if (counting) "Surv(start,time,status)" else "Surv(time,status)"
    mresponse <- if (counting) "Surv(start,time,event)" else "Surv(time,event)"
    suffix <- if (counting) "counting" else "right"
    for (grouped in c(FALSE, TRUE)) {
        rhs <- if (grouped) "group" else "1"
        for (weighted in c(FALSE, TRUE)) {
            tag <- paste(suffix, if (grouped) "groups" else "one",
                         if (weighted) "weighted" else "unit", sep="_")
            w <- if (weighted) weights else rep(1,nrow(data))
            for (multi in c(FALSE, TRUE)) {
                kind <- if (multi) "aj" else "km"
                formula <- paste(if (multi) mresponse else response, "~", rhs)
                name <- paste(kind,tag,sep="_")
                fit <- survfit(as.formula(formula),data=data,weights=w,id=id,entry=counting)
                fits[[name]] <- fit
                specs[[name]] <- list(kind=kind,formula=formula,weights=I(w),id="id",entry=counting)
            }
        }
    }
    for (grouped in c(FALSE, TRUE)) {
        formula <- paste(response,"~ x",if (grouped) "+ strata(group)" else "")
        name <- paste("cox",suffix,if (grouped) "groups" else "one",sep="_")
        model <- coxph(as.formula(formula),data=data,weights=weights)
        newdata <- data.frame(x=c(0,2),group=c("a","b"))
        fits[[name]] <- survfit(model,newdata=newdata)
        specs[[name]] <- list(kind="cox",formula=formula,weights=I(weights),
                             newdata=lapply(newdata,I))
    }
    formula <- paste(mresponse,"~ x")
    name <- paste("coxms",suffix,sep="_")
    model <- coxph(as.formula(formula),data=data,weights=weights,id=id)
    newdata <- data.frame(x=c(0,2))
    fits[[name]] <- survfit(model,newdata=newdata)
    specs[[name]] <- list(kind="coxms",formula=formula,weights=I(weights),id="id",
                         newdata=lapply(newdata,I))
}

encode <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.matrix(x)) return(lapply(seq_len(nrow(x)),function(i) I(unname(x[i,]))))
    I(unname(x))
}
# Stock summary.survfitms indexes grouped count matrices as vectors when
# times are absent. Retain that raw result, then accumulate the original
# fitted counts per column with R's own cumsum/diff operations.
grouped_counts <- function(fit, field) {
    sizes <- fit$strata
    ends <- cumsum(sizes)
    pieces <- lapply(seq_along(sizes),function(group) {
        indices <- seq.int(ends[group]-sizes[group]+1,ends[group])
        selected <- which(rowSums(fit$n.event[indices,,drop=FALSE])>0)
        block <- fit[[field]][indices,,drop=FALSE]
        result <- matrix(0,length(selected),ncol(block))
        for (column in seq_len(ncol(block))) {
            cumulative <- cumsum(block[,column])
            result[,column] <- diff(c(0,cumulative[selected]))
        }
        result
    })
    do.call(rbind,pieces)
}
cases <- list()
for (name in names(fits)) {
    for (times in list(c(0,2,4,7,10),c(7,2,2,0,10),NULL,4)) {
        for (dosum in list(NULL,FALSE,TRUE)) {
            for (extend in c(FALSE,TRUE)) {
                arguments <- list(extend=extend,scale=2,rmean="none")
                if (!is.null(times)) arguments$times <- times
                if (!is.null(dosum)) arguments$dosum <- dosum
                value <- tryCatch(do.call(summary,c(list(object=fits[[name]]),arguments)),
                                  error=function(e) list(error=conditionMessage(e)))
                fields <- c("time","n.risk","n.event","n.censor","n.enter","n.transition",
                            "surv","pstate","cumhaz","std.err","std.chaz","lower","upper")
                expected <- if (!is.null(value$error)) value else
                    setNames(lapply(fields,function(field) encode(value[[field]])),
                             gsub("\\.","_",fields))
                raw <- NULL
                note <- NULL
                fit <- fits[[name]]
                if (is.null(times) && !is.null(fit$pstate) && !is.null(fit$strata)) {
                    raw <- expected
                    for (field in c("n.censor","n.enter","n.transition")) {
                        if (!is.null(fit[[field]])) {
                            expected[[gsub("\\.","_",field)]] <- encode(grouped_counts(fit,field))
                        }
                    }
                    note <- "Stock grouped summary drops count-matrix dimensions; expected uses per-column accumulation from the original stock fit."
                }
                tag <- paste(name,length(cases)+1,sep="_")
                cases[[length(cases)+1]] <- list(name=tag,fit=name,
                    times=if (is.null(times)) NULL else I(times),dosum=dosum,
                    extend=extend,expected=expected,raw_expected=raw,reference_note=note)
            }
        }
    }
}
serialize <- function(value) toJSON(value,auto_unbox=TRUE,digits=NA,na="null",null="null")
header <- serialize(list(r_version=R.version.string,survival_version=as.character(packageVersion("survival")),
    data=lapply(data,function(x) I(if (is.factor(x)) as.character(x) else x)),
    event_levels=I(levels(data$event)),fits=specs))
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
    vapply(seq_along(cases),function(i) paste0(serialize(cases[[i]]),
        if (i < length(cases)) "," else ""), ""), "]}"), output,useBytes=TRUE)
