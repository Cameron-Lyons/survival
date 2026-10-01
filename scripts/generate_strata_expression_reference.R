#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/strata_expression_reference.json"
d <- lung[,c("time","status","age","sex","ph.ecog")]
d$g <- d$sex
d$g[seq_len(nrow(d)) %% 17 == 0] <- NA
d$z <- ifelse(seq_len(nrow(d)) %% 19 == 0, -1, d$sex^2)
d$label <- factor(c("b","a","c")[(seq_len(nrow(d))-1) %% 3 + 1], levels=c("c","a","b","unused"))
d$`group label` <- d$sex
new <- data.frame(time=c(100,200,300,400), status=c(1,2,1,2), age=c(50,60,70,55),
    sex=c(1,2,1,2),ph.ecog=c(0,1,2,0),g=c(1,2,1,2),z=c(1,4,1,4),
    label=factor(c("a","b","c","a"),levels=levels(d$label)))
new$`group label` <- new$sex
new_missing <- new
new_missing$g[3] <- NA
new_missing$z[3] <- -1
new_missing$age[3] <- 80
rhs <- c(
    logical="age + strata(sex == 1)",
    arithmetic="age + strata(sex - 1)",
    transformed="age + strata(sqrt(z))",
    identity="age + strata(I(sex == 1))",
    factored="age + strata(factor(sex))",
    character_factor="age + strata(label)",
    refactored="age + strata(factor(label))",
    as_factor="age + strata(sex, as.factor(label))",
    named="age + strata(group = sex)",
    named_factor="age + strata(group = label)",
    short_numeric="age + strata(sex, shortlabel = TRUE)",
    long_factor="age + strata(label, shortlabel = FALSE)",
    joint="age + strata(sex == 1, ph.ecog, sep = ':')",
    named_joint="age + strata(group = sex, level = label, sep = ' / ')",
    nested="age + strata(strata(sex, shortlabel = TRUE), label)",
    quoted="age + strata(`group label`)",
    cut="sex + strata(cut(age, c(30, 60, 90)))",
    cut_missing="sex + strata(cut(age, c(45, 60, 75)))",
    cut_codes="sex + strata(cut(age, c(30, 60, 90), labels = FALSE))",
    na_group="age + strata(g, na.group = TRUE)",
    na_shared_strata="age + strata(g) + strata(g, na.group = TRUE)",
    shared_source="age + g + strata(g, na.group = TRUE)",
    made_na_group="age + strata(sqrt(z), na.group = TRUE)",
    cut_na_group="sex + strata(cut(age, c(45, 60, 75)), na.group = TRUE)",
    na_joint="age + strata(g, label, na.group = TRUE, shortlabel = TRUE)",
    interaction="age * strata(sex == 1)",
    named_interaction="age * strata(group = sex, shortlabel = TRUE)",
    multiple="age + strata(sex == 1) + strata(ph.ecog, shortlabel = TRUE)"
)
cases <- list()
for(kind in c("coxph","survreg")) for(name in names(rhs)) {
    cat(kind,name,"\n")
    formula <- paste("Surv(time, status) ~",rhs[[name]])
    fit <- tryCatch(do.call(kind,list(formula=as.formula(formula),data=d,model=TRUE,x=TRUE,y=TRUE)),
        error=function(e) list(r_error=conditionMessage(e)))
    fit_error <- fit$r_error
    if(!is.null(fit_error)) {
        # Retain the empty trailing scale instead of assigning three names to
        # two scales. Only change the count; use the installed numerical fitter.
        stopifnot(kind=="survreg",name=="shared_source")
        source <- deparse(survival::survreg)
        stopifnot(sum(grepl("nstrata <- max(strata)",source,fixed=TRUE)) == 1L)
        corrected <- eval(parse(text=sub("nstrata <- max(strata)",
            "nstrata <- nlevels(strata.keep)",source,fixed=TRUE)),envir=asNamespace("survival"))
        fit <- corrected(as.formula(formula),d,model=TRUE,x=TRUE,y=TRUE)
    }
    prediction <- predict(fit,new,type="lp",se.fit=TRUE,na.action=na.pass)
    vars <- names(fit$model)[vapply(fit$model,is.factor,logical(1))]
    result <- list(name=paste(kind,name,sep="_"),kind=kind,formula=formula,fit_error=fit_error,
        beta=I(unname(coef(fit))),beta_names=I(names(coef(fit))),variance=unname(vcov(fit)),
        n=nrow(fit$x),omitted=I(as.integer(fit$na.action)),x_head=unname(head(fit$x,8)),
        assign=I(attr(fit$x,"assign")),loglik=I(unname(fit$loglik)),
        lp=I(unname(predict(fit,type="lp"))),prediction=I(unname(prediction$fit)),
        prediction_se=I(unname(prediction$se.fit)),
        terms=unname(predict(fit,new,type="terms",na.action=na.pass)),
        strata_levels=I(if(kind=="coxph") levels(fit$strata) else names(fit$scale)),
        used_strata_levels=I(if(kind=="coxph") levels(droplevels(fit$strata)) else names(fit$scale)),
        frame_labels=lapply(fit$model[vars],function(x) I(as.character(head(x,8)))))
    if(kind=="survreg") {
        result$scale <- I(unname(fit$scale))
        result$quantiles <- unname(predict(fit,new,type="quantile",p=c(.25,.5,.75)))
    } else {
        p <- predict(fit,new,type="expected",se.fit=TRUE)
        result$raw_expected <- I(unname(p$fit))
        result$raw_expected_se <- I(unname(p$se.fit))
        if(name=="shared_source") {
            # g is constant within every stratum and has an aliased coefficient.
            # R's expected prediction propagates that NA; the reduced model is
            # identical and gives the identifiable expected counts and errors.
            reduced <- coxph(Surv(time,status) ~ age + strata(g,na.group=TRUE),
                d[!is.na(d$g),])
            p <- predict(reduced,new,type="expected",se.fit=TRUE)
        }
        result$expected <- I(unname(p$fit))
        result$expected_se <- I(unname(p$se.fit))
    }
    if(name %in% c("na_group","made_na_group","cut_na_group","na_joint","cut_missing")) {
        result$missing_predictions <- lapply(c("na.pass","na.omit","na.exclude","na.fail"),function(action) {
            prediction <- tryCatch(predict(fit,new_missing,type="lp",se.fit=TRUE,
                na.action=get(action)),error=function(e) list(r_error=conditionMessage(e)))
            list(action=action,error=prediction$r_error,fit=I(as.numeric(prediction$fit)),
                se_fit=I(as.numeric(prediction$se.fit)))
        })
    }
    cases[[length(cases)+1L]] <- result
}
subsets <- lapply(c("strata(cut(age, 3))", "strata(cut(age, 4), sex)",
    "strata(g, na.group = TRUE)"),function(rhs) {
    rows <- which(d$age > 50 & d$age < 75 & seq_len(nrow(d)) %% 3 != 0)
    formula <- paste("Surv(time, status) ~ age +",rhs)
    fit <- coxph(as.formula(formula),d,subset=rows,x=TRUE,model=TRUE)
    list(formula=formula,subset=I(rows-1L),beta=I(unname(coef(fit))),variance=unname(vcov(fit)),
        strata_levels=I(levels(droplevels(fit$strata))),strata=I(as.character(fit$strata)),
        omitted=I(as.integer(fit$na.action)),lp=I(unname(predict(fit,type="lp"))))
})
# The grouping-only methods use the same evaluated factors.
grouping <- lapply(c("strata(sex == 1)","strata(g, na.group = TRUE)",
    "strata(group = label, shortlabel = FALSE)","strata(cut(age, c(45, 60, 75)))"),function(rhs) {
    f <- as.formula(paste("Surv(time, status) ~",rhs))
    s <- summary(survfit(f,d),times=c(100,300,600),extend=TRUE)
    test <- survdiff(as.formula(paste("Surv(time, status) ~ sex +",rhs)),d)
    list(formula=deparse(f),surv=I(s$surv),time=I(s$time),n_risk=I(s$n.risk),
        strata=I(as.character(s$strata)),test_chisq=test$chisq,
        test_observed=unname(test$obs),test_expected=unname(test$exp))
})
reference <- list(metadata=list(generator="scripts/generate_strata_expression_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    data=d,newdata=new,missing_newdata=new_missing,label_levels=I(levels(d$label)),
    cases=cases,grouping=grouping,subsets=subsets)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"strata expression cases written to",output,"\n")
