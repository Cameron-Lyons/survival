#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/strata_interaction_reference.json"
d <- lung[, c("time", "status", "age", "sex", "ph.ecog")]
d <- d[complete.cases(d),]
d <- d[d$ph.ecog < 3,]
d$start <- pmin(d$time/2,seq_len(nrow(d)) %% 11)
d$case_weight <- .5 + (seq_len(nrow(d)) %% 3)
new <- data.frame(age=c(50,60,70,55),sex=c(1,2,1,2),ph.ecog=c(0,1,2,0))
cases <- list()
capture <- function(expr) tryCatch(expr,error=function(e) list(r_error=conditionMessage(e)))
add <- function(name,rhs,kind="coxph",counting=FALSE,weighted=FALSE,ties="efron") {
    formula <- paste(if(counting) "Surv(start,time,status) ~" else "Surv(time,status) ~",rhs)
    kwargs <- list()
    if(weighted) kwargs$weights <- d$case_weight
    if(kind=="coxph") kwargs$ties <- ties
    fit <- if(kind=="coxph") do.call(coxph,c(list(formula=as.formula(formula),data=d,x=TRUE,model=TRUE),
        kwargs)) else
                          survreg(as.formula(formula),d,x=TRUE,model=TRUE)
    prediction <- capture(predict(fit,new,type="lp",se.fit=TRUE))
    newx <- if(kind=="coxph") model.matrix(fit,new) else {
        tt <- delete.response(terms(fit))
        mf <- model.frame(tt,new,xlev=fit$xlevels)
        drop <- survival:::untangle.specials(tt,"strata",1)$terms
        if(length(drop)) tt <- tt[-drop]
        model.matrix(tt,mf,contrasts.arg=fit$contrasts)
    }
    prediction_error <- prediction$r_error
    if(!is.null(prediction_error) && kind=="coxph") {
        # predict.coxph drops columns by term number, which is wrong after
        # a multi-column factor. Use R's correct model.matrix.coxph output.
        center <- rowsum(fit$x,fit$strata) / as.numeric(table(fit$strata))
        centered <- newx - center[match(attr(newx,"strata"),rownames(center)),,drop=FALSE]
        beta <- coef(fit); beta[is.na(beta)] <- 0
        prediction <- list(fit=c(centered %*% beta),
            se.fit=sqrt(rowSums((centered %*% vcov(fit))*centered)))
    }
    result <- list(name=name,kind=kind,formula=formula,options=kwargs,beta=I(unname(coef(fit))),
        beta_names=I(names(coef(fit))),variance=unname(vcov(fit)),x=unname(fit$x),
        assign=I(attr(fit$x,"assign")),loglik=I(fit$loglik),
        lp=I(unname(predict(fit,type="lp"))),prediction=I(unname(prediction$fit)),
        prediction_se=I(unname(prediction$se.fit)),
        prediction_error=prediction_error,newx=unname(newx),
        terms=capture(unname(predict(fit,new,type="terms"))))
    if(kind=="coxph") {
        centered <- sweep(newx,2,fit$means)
        beta <- coef(fit); beta[is.na(beta)] <- 0
        result$terms_from_model_matrix <- unname(sapply(fit$assign,function(cols)
            c(centered[,cols,drop=FALSE] %*% beta[cols])))
        result$martingale <- I(unname(residuals(fit,type="martingale")))
        result$schoenfeld <- unname(as.matrix(residuals(fit,type="schoenfeld")))
        sf <- capture(survfit(fit,newdata=new))
        result$survfit_error <- sf$r_error
        if(!is.null(sf$r_error)) {
            # These saturated factor-by-stratum models equal independent Cox
            # fits within each stratum; use those when R's curve method fails.
            stopifnot(name %in% c("factor_star","strata_first_factor"))
            parts <- lapply(seq_len(nrow(new)),function(i) {
                separate <- coxph(Surv(time,status) ~ factor(ph.ecog),d[d$sex==new$sex[i],])
                survfit(separate,newdata=new[i,,drop=FALSE])
            })
            sf <- lapply(c("time","surv","cumhaz","std.err","lower","upper"),
                function(key) unlist(lapply(parts,function(part) part[[key]]),use.names=FALSE))
            names(sf) <- c("time","surv","cumhaz","std.err","lower","upper")
        }
        result$survfit <- list(time=I(sf$time),surv=unname(sf$surv),
            cumhaz=unname(sf$cumhaz),std_err=unname(sf$std.err),lower=unname(sf$lower),upper=unname(sf$upper))
        zph <- capture(cox.zph(fit))
        result$zph_error <- zph$r_error
        if(is.null(zph$r_error)) result$zph <- list(table=unname(zph$table),y=unname(zph$y),var=unname(zph$var))
    } else result$scale <- I(unname(fit$scale))
    cases[[length(cases)+1L]] <<- result
}
add("numeric_star","age * strata(sex)")
add("strata_first","strata(sex) * age")
add("interaction_only","age:strata(sex)")
add("adjusted_interaction","age + strata(sex):ph.ecog")
add("factor_star","factor(ph.ecog) * strata(sex)")
add("strata_first_factor","strata(sex) * factor(ph.ecog)")
add("strata_factor_interaction_only","strata(sex):factor(ph.ecog)")
add("combined_strata","age * strata(sex, ph.ecog)")
add("two_strata_calls","age * strata(sex) + strata(ph.ecog)")
add("removed_main_strata","age * strata(sex) - strata(sex)")
add("no_intercept","0 + strata(sex) * age")
add("breslow","age * strata(sex)",ties="breslow")
add("counting","age * strata(sex)",counting=TRUE)
add("case_weights","age * strata(sex)",weighted=TRUE)
add("aft_numeric_star","age * strata(sex)","survreg")
add("aft_interaction_only","age:strata(sex)","survreg")
add("aft_factor_star","factor(ph.ecog) * strata(sex)","survreg")
add("aft_strata_first_factor","strata(sex) * factor(ph.ecog)","survreg")
add("aft_two_strata_calls","age * strata(sex) + strata(ph.ecog)","survreg")
reference <- list(metadata=list(generator="scripts/generate_strata_interaction_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    data=d,newdata=new,cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"strata interaction cases written to",output,"\n")
