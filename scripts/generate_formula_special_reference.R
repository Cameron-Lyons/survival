#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/formula_special_reference.json"
d <- lung[,c("time","status","age","sex","ph.ecog")]
d$group <- (seq_len(nrow(d))-1) %% 12
d$label <- factor(c("b","a","c")[(seq_len(nrow(d))-1) %% 3 + 1],levels=c("c","a","b"))
d$offs <- log(d$age/60)
d$unused <- ifelse(seq_len(nrow(d)) %% 17 == 0,NA,seq_len(nrow(d)))
d$transformed <- ifelse(seq_len(nrow(d)) %% 19 == 0,-1,2)
new <- d[c(1,2,3,4),]
new$unused[3] <- NA
rhs <- c(
    offset_star="age * offset(offs)",
    offset_first="offset(offs) * age",
    offset_only_interaction="offset(offs):age",
    offset_adjusted="offset(offs):age + sex",
    offset_group="(age + sex) * offset(offs)",
    offset_nested="age/(sex + offset(offs))",
    offset_power="(age + sex + offset(offs))^2",
    offset_removed="age + offset(offs) - offset(offs)",
    offset_negative="age - offset(offs)",
    offset_canceled="age:offset(offs) - age:offset(offs)",
    offset_pair="offset(offs) * offset(sex)",
    offset_transform="age * offset(log(transformed))",
    removed_variable="age + unused - unused",
    removed_group="(age + unused - unused)^2",
    removed_transform="age + log(transformed) - log(transformed)",
    removed_strata="age + strata(sex) - strata(sex)",
    cluster_star="age * cluster(group)",
    cluster_first="cluster(group) * age",
    cluster_factor="factor(sex) * cluster(group)",
    cluster_labels="age * cluster(label)",
    cluster_power="(age + sex + cluster(group))^2",
    cluster_removed_variable="age + unused - unused + cluster(group)",
    cluster_removed_strata="age + strata(sex) - strata(sex) + cluster(group)",
    cluster_offset="age * cluster(group) + offset(offs)",
    cluster_arithmetic="age + cluster(group + sex)",
    cluster_transform="age + cluster(sqrt(group))",
    cluster_made_missing="age + cluster(log(transformed*sex))",
    cluster_logical="age + cluster(group>5)",
    cluster_arithmetic_interaction="age * cluster(group + sex)",
    cluster_transformed_interaction="age * cluster(sqrt(group))",
    cluster_first_arithmetic="cluster(group/2) * age",
    cluster_expression_factor="factor(sex) * cluster(group + sex)",
    cluster_removed_source="age + unused - unused + cluster(group + sex)",
    cluster_missing_source="age + cluster(group+unused)",
    reversed_removal="age * sex - sex:age",
    reversed_duplicate="age:sex + sex:age",
    reversed_factor="factor(sex):ph.ecog + ph.ecog:factor(sex)",
    joint_nesting="(age + sex)/ph.ecog",
    power_product="(age + sex)^2 * ph.ecog",
    implicit_margin="(factor(sex) + factor(ph.ecog)):age",
    nested_no_intercept="(age - 1) + sex",
    nested_double_negative="age - (sex - 1)",
    power_interaction="(age + sex)^2:ph.ecog",
    cluster_implicit_margin="age:sex + age:cluster(group) + cluster(group)"
)
cases <- list()
capture <- function(expr) tryCatch(expr,error=function(e) list(r_error=conditionMessage(e)))
for(kind in c("coxph","survreg")) for(name in names(rhs)) {
    cat(kind,name,"\n")
    formula <- paste("Surv(time,status) ~",rhs[[name]])
    fit <- do.call(kind,list(formula=as.formula(formula),data=d,model=TRUE,x=TRUE,y=TRUE))
    prediction <- capture(predict(fit,new,type="lp",se.fit=TRUE,na.action=na.pass))
    prediction_error <- prediction$r_error
    if(!is.null(prediction_error)) {
        stopifnot(kind=="coxph",length(coef(fit))==0L)
        prediction <- list(fit=predict(fit,new,type="lp",na.action=na.pass),se.fit=rep(0,nrow(new)))
    }
    x <- model.matrix(fit)
    newoffset <- model.offset(model.frame(delete.response(terms(fit)),new,na.action=na.pass))
    intended <- prediction$fit
    if(!is.null(newoffset)) {
        # The port deliberately includes AFT new-data offsets and centers Cox
        # offsets consistently even when there are no fitted coefficients.
        if(kind=="survreg") intended <- intended + newoffset
        else if(!length(coef(fit))) intended <- intended - mean(model.offset(fit$model))
    }
    cases[[length(cases)+1L]] <- list(name=paste(kind,name,sep="_"),kind=kind,formula=formula,
        beta=I(as.numeric(coef(fit))),beta_names=I(as.character(names(coef(fit)))),
        variance=if(length(coef(fit))) unname(vcov(fit)) else matrix(numeric(),0,0),
        x_head=unname(head(x,8)),assign=I(attr(x,"assign")),n=nrow(x),
        omitted=I(as.integer(fit$na.action)),loglik=I(unname(fit$loglik)),
        prediction=I(unname(prediction$fit)),prediction_se=I(unname(prediction$se.fit)),
        prediction_error=prediction_error,
        intended_prediction=I(unname(intended)),
        train_prediction=I(unname(predict(fit,type="lp"))),
        terms=capture(unname(predict(fit,new,type="terms",na.action=na.pass))),
        retained_columns=I(names(fit$model)),
        cluster_variables=I(all.vars(fit$call$cluster)),
        scale=if(kind=="survreg") I(unname(fit$scale)) else NULL)
}
bad <- c("age:cluster(group)","cluster(group):age + sex",
    "age:cluster(group) + age:sex + cluster(group)",
    "age + cluster(group) + cluster(sex)","age + cluster(group) - cluster(group)")
errors <- lapply(bad,function(rhs) {
    formula <- paste("Surv(time,status) ~",rhs)
    list(formula=formula,error=tryCatch({coxph(as.formula(formula),d);stop("expected an error")},
        error=function(e) conditionMessage(e)))
})
curves <- lapply(c("group + sex", "log(transformed*sex)","group>5"),function(expression) {
    formula <- paste0("Surv(time, status) ~ sex + cluster(",expression,")")
    raw <- survfit(as.formula(formula),data=d)
    # In 3.8-12 terms(mf, specials) reuses mf's terms without recognizing
    # cluster(). The intended deprecated branch never runs: the cluster
    # instead becomes a curve grouping variable. Use the explicit argument.
    cluster_values <- eval(str2lang(expression),d)
    fit <- survfit(Surv(time,status) ~ sex,data=d,cluster=cluster_values)
    list(formula=formula,time=I(fit$time),surv=I(fit$surv),cumhaz=I(fit$cumhaz),
        std_err=I(fit$std.err),n_risk=I(fit$n.risk),n_event=I(fit$n.event),
        n_censor=I(fit$n.censor),strata=as.list(fit$strata),
        raw_strata=as.list(raw$strata),raw_std_err_head=I(head(raw$std.err)))
})
reference <- list(metadata=list(generator="scripts/generate_formula_special_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    data=d,newdata=new,label_levels=I(levels(d$label)),cases=cases,errors=errors,curves=curves)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"formula special cases written to",output,"\n")
