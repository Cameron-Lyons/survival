#!/usr/bin/env Rscript
# Independent references for the exact counting-process risk-set sweep.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else
    "test/r/kernel_references/exact_counting_sweep.json"
i <- 0:59
d <- data.frame(stop=1+(i*17 %% 61)+.01*(i %% 7), status=as.integer(i %% 4 != 0),
    x1=sin(i*1.7)+cos(i*.31), x2=(i %% 9-4)/5+.1*sin(i*2.3),
    offset=.25*sin(i*.4), stratum=c(-5L,3L,19L)[i %% 3+1])
d$start <- d$stop*((i*37+17) %% 101+1)/103
initial <- c(.3,-.2)
cases <- list()
result <- function(fit) list(coefficients=I(unname(fit$coefficients)),
    variance=unname(fit$var), loglik=I(fit$loglik),
    score=unname(fit$score), means=I(unname(fit$means)), iter=fit$iter)
add <- function(name, data, init=initial, fitted=TRUE, reference=data, r_error=NULL,
                removed_common_offset=NULL) {
    form <- Surv(start,stop,status) ~ x1+x2+offset(offset)+strata(stratum)
    control <- coxph.control(iter.max=0,eps=1e-9,toler.chol=1e-10)
    at_initial <- coxph(form,reference,ties="exact",init=init,control=control)
    final <- if (fitted) coxph(form,reference,ties="exact",init=init,
        control=coxph.control(iter.max=50,eps=1e-9,toler.chol=1e-10)) else NULL
    cases[[length(cases)+1L]] <<- list(name=name,
        start=I(data$start), stop=I(data$stop), status=I(data$status),
        x=unname(as.matrix(data[,c("x1","x2")])), offset=I(data$offset),
        strata=I(as.integer(data$stratum)), init=I(init),
        r_error=r_error, removed_common_offset=removed_common_offset,
        initial=result(at_initial), fitted=if (!is.null(final)) result(final) else NULL)
}
single <- d; single$stratum <- 0L
add("delayed_entry",single)
add("delayed_entry_strata",d)
add("permuted_strata",d[order(i*11 %% 60),])
censors <- d
for (row in which(censors$status == 0)) {
    other <- which(censors$status == 1 & censors$stratum == censors$stratum[row])[1]
    censors$stop[row] <- censors$stop[other]
}
censors$start <- censors$stop*((i*37+17) %% 101+1)/103
add("same_time_censors",censors)
growing <- d; growing$start <- -5+i*.01
add("growing_strata",growing)
mixed <- d; mixed$start[mixed$stratum == -5] <- growing$start[mixed$stratum == -5]
add("mixed_growing_and_removal_strata",mixed)
tied <- d
rows <- which(tied$status == 1 & tied$stratum == -5)[1:2]
tied$stop[rows] <- max(tied$stop[rows])
tied$start[rows] <- 0
add("tied_deaths",tied)
for (shift in c(-1000,1000)) {
    shifted <- d; shifted$offset <- shifted$offset+shift
    # R can refuse extreme offsets before fitting. When that happens, removing
    # their common shift leaves the same partial likelihood and information.
    failure <- tryCatch({
        coxph(Surv(start,stop,status) ~ x1+x2+offset(offset)+strata(stratum),
            shifted,ties="exact",init=initial,control=coxph.control(iter.max=0))
        NULL
    },error=function(e) conditionMessage(e))
    if (is.null(failure)) add(paste0("common_offset_",shift),shifted) else
        add(paste0("common_offset_",shift),shifted,reference=d,
            r_error=failure,removed_common_offset=shift)
}
extreme <- data.frame(start=c(0,0,0,2),stop=c(1,3,4,4),status=c(1L,1L,0L,0L),
    x1=c(0,1,-1,100),x2=c(0,0,1,-50),offset=0,stratum=0L)
add("dominant_risk_removed",extreme,c(1,0),FALSE)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(list(survival_version=as.character(packageVersion("survival")),
    r_version=as.character(getRversion()), cases=cases), output,
    auto_unbox=TRUE,digits=NA,pretty=TRUE,null="null")
