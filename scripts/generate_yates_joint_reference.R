#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/yates_joint_reference.json"
d <- expand.grid(a=factor(c("a","b","c")), b=factor(c("x","y")),
                 cc=factor(c("low","high")), replicate=1:4)
d$z <- sin(seq_len(nrow(d))) + d$replicate
d$y <- seq_len(nrow(d)) / 3 + cos(seq_len(nrow(d))) + d$z
d$time <- 1 + (seq_len(nrow(d))*17 %% 53)
d$status <- as.integer(seq_len(nrow(d)) %% 4 != 0)
cases <- list()
add <- function(name,formula,term,levels=NULL,population="data",test="global",
                method="direct",cox=FALSE,predict="linear",rows=seq_len(nrow(d))) {
    data <- d[rows,]
    fit <- if(cox) coxph(as.formula(formula),data) else lm(as.formula(formula),data)
    options <- list(fit=fit,term=term,population=population,test=test,method=method,
                    predict=predict,nsim=100)
    if(!is.null(levels)) options$levels <- levels
    set.seed(123)
    result <- do.call(yates,options)
    cases[[length(cases)+1L]] <<- list(name=name,formula=formula,data=data,term=term,
        levels=levels,population=population,test=test,method=method,cox=cox,predict=predict,
        beta=I(unname(coef(fit))),variance=unname(vcov(fit,complete=FALSE)),
        sigma2=if(cox) NULL else summary(fit)$sigma^2,
        estimate=result$estimate,mvar=unname(result$mvar),cmat=unname(result$cmat),
        sas=unname(result$SAS),tests=unname(result$test),test_names=I(rownames(result$test)))
}
add("two_factors","y ~ a * b + cc","a + b")
add("interaction_spelling","y ~ a * b + cc","a:b")
add("crossed_spelling","y ~ a * b + cc","a*b")
add("reverse_order","y ~ b * a + cc","b + a")
add("three_factors","y ~ a * b * cc","a + b + cc")
add("partial_levels","y ~ a * b + cc","a + b",levels=list(a=c("c","a")))
add("reordered_levels","y ~ b * a + cc","b+a",levels=list(a=c("c","a"),b=c("y","x")))
add("numeric_levels","y ~ a * z + b","a + z",levels=list(z=c(1,3)))
add("two_numeric","y ~ z * replicate + a","z + replicate",levels=list(z=c(1,3),replicate=c(2,4)))
add("single_mapping","y ~ a * b","a",levels=list(a=c("c","a")))
add("factorial","y ~ a * b + cc","a+b",population="factorial")
add("sas","y ~ a * b + cc + z","a+b",population="sas")
add("pairwise","y ~ a * b + cc","a+b",levels=list(a=c("a","b")),test="pairwise")
add("missing_cell","y ~ a * b","a+b",rows=which(!(d$a=="c" & d$b=="y")))
add("sgtt","y ~ a * b + cc","a+b",population="sas",method="sgtt")
add("sgtt_reverse","y ~ b * a + cc","b+a",population="sas",method="sgtt")
add("sgtt_numeric","y ~ a * z + b","a+z",levels=list(z=c(1,3)),population="sas",method="sgtt")
add("cox","Surv(time,status) ~ a * b + z","a+b",cox=TRUE)
add("cox_risk","Surv(time,status) ~ a * b + z","a+b",cox=TRUE,predict="risk",test="pairwise")
reference <- list(metadata=list(generator="scripts/generate_yates_joint_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"joint Yates cases written to",output,"\n")
