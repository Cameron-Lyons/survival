#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/partial_prediction_reference.json"
d <- lung[,c("time","status","age","sex")]
d$o <- log(d$age/60)
new <- data.frame(age=c(50,NA,60,70),sex=c(1,2,NA,1),o=c(.1,NA,NA,.4),
    time=c(100,200,300,400),status=c(NA,1,1,0))
rhs <- c(additive="age + sex",factor="age + factor(sex)",
    interaction="age * factor(sex)",transform="sqrt(age) + sex",
    arithmetic="I(age/sex) + sex",offset="age + offset(o)",
    ridge="ridge(age, theta=1) + sex",pspline="pspline(age, df=3) + sex",
    strata="age + strata(sex)",strata_interaction="age * strata(sex)")
capture <- function(expr) tryCatch(expr,error=function(e) list(r_error=conditionMessage(e)))
cases <- list()
for(kind in c("coxph","survreg")) for(name in names(rhs)) {
    formula <- paste("Surv(time,status) ~",rhs[[name]])
    fit <- do.call(kind,list(formula=as.formula(formula),data=d))
    types <- if(kind=="coxph") c("lp","risk","terms","expected","survival") else
        c("lp","response","terms","quantile","uquantile")
    if(name=="offset" && kind=="coxph") types <- c("lp","risk","terms")
    results <- list()
    for(type in types) {
        value <- capture(predict(fit,new,type=type,se.fit=TRUE))
        result <- list(type=type,raw=value)
        if(!is.null(value$r_error)) {
            stopifnot(kind=="coxph",name %in% c("ridge","pspline"),type=="survival")
            # R's penalized method omits survival from match.arg; derive it
            # from that method's expected count and its delta-method error.
            expected <- predict(fit,new,type="expected",se.fit=TRUE)
            value <- list(fit=exp(-expected$fit),se.fit=expected$se.fit*exp(-expected$fit))
        }
        if(is.null(value$r_error)) {
            intended <- value
            if(kind=="coxph" && name %in% c("strata","strata_interaction") &&
                type %in% c("expected","survival")) {
                # R leaves the initial expected count at zero when the
                # stratum is unknown; the port keeps this prediction missing.
                intended$fit[is.na(new$sex)] <- NA
                intended$se.fit[is.na(new$sex)] <- NA
            }
            if(kind=="survreg" && name=="offset" && type!="terms") {
                # Keep the port's documented correction of omitted AFT offsets.
                if(type %in% c("response","quantile")) {
                    intended$fit <- intended$fit * exp(new$o)
                    intended$se.fit <- intended$se.fit * exp(new$o)
                } else intended$fit <- intended$fit + new$o
            }
            result$fit <- if(is.matrix(intended$fit)) unname(intended$fit) else I(unname(intended$fit))
            result$se_fit <- if(is.matrix(intended$se.fit)) unname(intended$se.fit) else I(unname(intended$se.fit))
        }
        results[[length(results)+1L]] <- result
    }
    cases[[length(cases)+1L]] <- list(name=paste(kind,name,sep="_"),kind=kind,
        formula=formula,results=results)
}
reference <- list(metadata=list(generator="scripts/generate_partial_prediction_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    data=d,newdata=new,cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"partial prediction models written to",output,"\n")
