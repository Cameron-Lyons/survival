#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/yates_sgtt_reference.json"
data <- expand.grid(a=factor(c("a","b","c")), b=factor(c("x","y")),
                    cc=factor(c("low","high")), replicate=1:4)
data$z <- sin(seq_len(nrow(data))) + data$replicate
data$y <- seq_len(nrow(data)) / 3 + cos(seq_len(nrow(data))) + data$z
cases <- list()
add <- function(name,formula,rows=seq_len(nrow(data)),terms=c("a","b"),weights=NULL) {
    d <- data[rows,]
    fit <- lm(as.formula(formula),d,weights=weights)
    results <- lapply(terms,function(term) {
        result <- yates(fit,term,method="sgtt")
        list(term=term,estimate=result$estimate,mvar=unname(result$mvar),
             cmat=unname(result$cmat),sas=unname(result$SAS),sas_names=colnames(result$SAS),
             sas_row_names=rownames(result$SAS),
             tests=unname(result$test),test_names=I(rownames(result$test)))
    })
    cases[[length(cases)+1L]] <<- list(name=name,formula=formula,data=d,
        beta=I(unname(coef(fit))),variance=unname(vcov(fit,complete=FALSE)),
        sigma2=summary(fit)$sigma^2,results=results)
}
add("additive","y ~ a + b + z")
add("interaction","y ~ a * b")
add("unbalanced","y ~ a * b + z",rows=setdiff(1:nrow(data),c(1,3,5,8,10,18,24)))
add("missing_cell","y ~ a * b",rows=which(!(data$a=="c" & data$b=="y")))
add("numeric_interaction","y ~ a * z + b",terms=c("a","b"))
add("no_intercept_additive","y ~ 0 + a + b")
add("no_intercept_interaction","y ~ 0 + a * b")
add("three_factors","y ~ a * b * cc",terms=c("a","b","cc"))
add("weights","y ~ a * b",weights=1+(seq_len(nrow(data)) %% 3))
reference <- list(metadata=list(generator="scripts/generate_yates_sgtt_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"SGTT model cases written to",output,"\n")
