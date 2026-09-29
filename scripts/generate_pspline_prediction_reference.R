#!/usr/bin/env Rscript
# Rscript scripts/generate_pspline_prediction_reference.R [output.json]
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/pspline_prediction_reference.json"
x <- c(-2, -1, -.4, 0, .3, 1, 2, 3)
newx <- c(-5, -2, NA, -1.25, 0, 3, 6, 0)
options <- list(
    default=list(df=3),
    intercept=list(df=3, intercept=TRUE),
    boundaries=list(df=3, Boundary.knots=c(-1,2)),
    combined=list(df=3, nterm=6, degree=2, combine=c(1,1,2,3,3,3,4)),
    combined_intercept=list(df=3, nterm=4, degree=3, intercept=TRUE,
                            combine=c(1,1,2,2,3,3,4)),
    linear=list(df=2, nterm=5, degree=1),
    quadratic=list(df=3, nterm=7, degree=2),
    fixed=list(theta=.4, nterm=6),
    aic=list(df=0),
    unpenalized=list(df=3, penalty=FALSE))
cases <- lapply(names(options), function(name) {
    object <- do.call(pspline, c(list(x=x), options[[name]]))
    predicted <- predict(object, newx)
    penalized <- do.call(pspline, c(list(x=x), modifyList(options[[name]],list(penalty=TRUE))))
    list(name=name, options=options[[name]],
         basis=matrix(as.numeric(predicted),ncol=ncol(predicted)),
         scalar=matrix(as.numeric(predict(object,0)),nrow=1), ncol=ncol(predicted),
         dmat=matrix(attr(penalized,"pparm"),ncol=ncol(predicted)),
         nterm=attr(predicted,"nterm"), degree=attr(predicted,"degree"),
         intercept=attr(predicted,"intercept"), boundary_knots=attr(predicted,"Boundary.knots"),
         combine=attr(predicted,"combine"),
         omitted_identical=identical(predict(object),object))
})
errors <- list(
    small_basis=tryCatch(predict(pspline(x,df=2,nterm=3),newx), error=conditionMessage),
    constant_boundary=tryCatch(predict(pspline(rep(2,5)),2), error=conditionMessage))
reference <- list(metadata=list(generator="scripts/generate_pspline_prediction_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    x=x,newx=newx,cases=cases,errors=errors)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"P-spline prediction cases written to",output,"\n")
