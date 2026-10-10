#!/usr/bin/env Rscript
# Stock G-rho tests and a direct risk-set oracle for the delayed-entry extension.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/weighted_logrank_reference.json"

data <- data.frame(
    time=c(1,1,2,3,3,4,5,6,2,3,4,5,6,7,8,9),
    start=c(0,0,.5,1,0,2,1,3,0,1,2,0,3,4,2,6),
    status=c(1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1),
    group=(0:15) %% 3,
    stratum=rep(c(7,9),each=8)
)
encode_matrix <- function(x) lapply(seq_len(nrow(x)),function(i) I(unname(x[i,])))
encode <- function(fit, grouped, nstrata) {
    obs <- matrix(fit$obs,nrow=3,ncol=nstrata)
    exp <- matrix(fit$exp,nrow=3,ncol=nstrata)
    keep <- rowSums(exp)>0
    df <- sum(keep)-1L
    list(n=I(as.integer(fit$n)),obs=encode_matrix(obs),exp=encode_matrix(exp),
         var=encode_matrix(fit$var),chisq=fit$chisq,
         pvalue=pchisq(fit$chisq,df,lower.tail=FALSE),df=df,
         strata=if (grouped) I(as.integer(fit$strata)) else NULL,
         group_codes=I(0:2))
}
counting_oracle <- function(frame,rho,timefix,grouped) {
    y <- Surv(frame$start,frame$time,frame$status)
    if (timefix) y <- aeqSurv(y)
    frame$start <- y[,1]
    frame$time <- y[,2]
    strata <- if (grouped) frame$stratum else rep(1,nrow(frame))
    levels <- sort(unique(strata))
    obs <- exp <- matrix(0,3,length(levels))
    variance <- matrix(0,3,3)
    for (s in seq_along(levels)) {
        d <- frame[strata==levels[s],]
        # Use stock KM to obtain left limits; construct test moments by
        # explicitly selecting risk sets, independent of the Rust sweeps.
        km <- survfit(Surv(start,time,status)~1,data=d,se.fit=FALSE,timefix=FALSE)
        for (t in sort(unique(d$time[d$status==1]))) {
            earlier <- which(km$time<t)
            survival <- if (length(earlier)) km$surv[tail(earlier,1)] else 1
            weight <- survival^rho
            at_risk <- d$start<t & d$time>=t
            deaths <- d$time==t & d$status==1
            risk <- tabulate(d$group[at_risk]+1,nbins=3)
            observed <- tabulate(d$group[deaths]+1,nbins=3)
            n <- sum(risk)
            ndead <- sum(observed)
            probability <- risk/n
            obs[,s] <- obs[,s]+weight*observed
            exp[,s] <- exp[,s]+weight*ndead*probability
            if (n>1) variance <- variance+
                weight^2*ndead*(n-ndead)/(n-1)*
                (diag(probability)-outer(probability,probability))
        }
    }
    keep <- rowSums(exp)>0
    difference <- (rowSums(obs)-rowSums(exp))[keep][-1]
    covariance <- variance[keep,keep,drop=FALSE][-1,-1,drop=FALSE]
    chisq <- drop(crossprod(difference,solve(covariance,difference)))
    fit <- list(n=tabulate(frame$group+1,nbins=3),obs=obs,exp=exp,var=variance,
                chisq=chisq,strata=tabulate(match(strata,levels)))
    encode(fit,grouped,length(levels))
}
cases <- list()
for (near in c(FALSE,TRUE)) {
    frame <- data
    if (near) frame$time[2] <- frame$time[2]+5e-10
    for (counting in c(FALSE,TRUE)) for (grouped in c(FALSE,TRUE))
    for (timefix in c(FALSE,TRUE)) for (rho in c(-1,0,.25,1,2)) {
        formula <- paste("Surv(time,status) ~ group",
                         if (grouped) "+ strata(stratum)" else "")
        expected <- if (counting) counting_oracle(frame,rho,timefix,grouped) else {
            if (timefix) {
                fit <- survdiff(as.formula(formula),frame,rho=rho)
            } else {
                # Stock 3.8-11 forwards explicit timefix into model.frame.
                # Call its unchanged native test on the unrounded response.
                strata <- if (grouped) factor(frame$stratum) else rep(1,nrow(frame))
                native <- survival:::survdiff.fit(Surv(frame$time,frame$status),
                                                 factor(frame$group),strata,rho)
                obs <- matrix(native$observed,nrow=3)
                exp <- matrix(native$expected,nrow=3)
                difference <- (rowSums(obs)-rowSums(exp))[-1]
                chisq <- drop(crossprod(difference,
                    solve(native$var[-1,-1,drop=FALSE],difference)))
                fit <- list(n=tabulate(frame$group+1,nbins=3),obs=obs,exp=exp,
                            var=native$var,chisq=chisq,
                            strata=if (grouped) table(strata) else NULL)
            }
            encode(fit,grouped,if (grouped) 2 else 1)
        }
        name <- paste(if (near) "near" else "ties",if (counting) "counting" else "right",
                      if (grouped) "strata" else "one",timefix,rho,sep="_")
        cases[[length(cases)+1L]] <- list(name=name,near=near,counting=counting,
            grouped=grouped,timefix=timefix,rho=rho,expected=expected,
            oracle=if (counting) "stock KM and direct risk-set moments" else
                   if (timefix) "stock survdiff" else "stock survdiff.fit without rounding")
    }
}
write_json(list(r_version=R.version.string,
    survival_version=as.character(packageVersion("survival")),
    generator="scripts/generate_weighted_logrank_reference.R",
    data=lapply(data,I),cases=cases),output,
    auto_unbox=TRUE,pretty=TRUE,digits=NA,na="null",null="null")
