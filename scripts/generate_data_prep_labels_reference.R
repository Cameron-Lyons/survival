#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/data_prep_labels_reference.json"

column <- function(x) {
    result <- list(
    values=I(if (is.factor(x)) as.character(x) else x),
    kind=if (is.factor(x)) "factor" else typeof(x),
    levels=if (is.factor(x)) I(levels(x)) else NULL,
    ordered=is.ordered(x))
    if (is.numeric(x) && any(is.nan(x))) result$nan <- I(is.nan(x))
    result
}
snapshot <- function(x) {
    response <- Surv(seq_along(x), x)
    list(values=I(as.character(x)), levels=I(levels(x)), ordered=is.ordered(x),
         response=list(type=attr(response,"type"), states=I(attr(response,"states")),
                       matrix=unname(as.matrix(response))))
}
near <- list(
    numeric_character=list(id1=c(1,2,1),id2=c("1","2","1"),y1=c(1,2,5),y2=c(2,3,4)),
    character_numeric=list(id1=c("1","2","1"),id2=c(1,2,1),y1=c(1,2,5),y2=c(2,3,4)),
    logical_character=list(id1=c(TRUE,FALSE,TRUE),id2=c("TRUE","FALSE","TRUE"),
                           y1=c(2,1,4),y2=c(2,2,3)),
    character_logical=list(id1=c("TRUE","FALSE","TRUE"),id2=c(TRUE,FALSE,TRUE),
                           y1=c(2,1,4),y2=c(2,2,3)),
    logical_integer_character=list(id1=c(TRUE,2L),id2=c("1","2"),y1=c(1,1),y2=c(2,2)),
    character_logical_integer=list(id1=c("1","2"),id2=c(TRUE,2L),y1=c(1,1),y2=c(2,2)),
    logical_double_character=list(id1=c(TRUE,2.5),id2=c("1","2.5"),y1=c(1,1),y2=c(2,2)),
    character_logical_double=list(id1=c("1","2.5"),id2=c(TRUE,2.5),y1=c(1,1),y2=c(2,2)),
    logical_nan_character=list(id1=c(TRUE,NaN),id2=c("1","NaN"),y1=c(1,1),y2=c(2,2)),
    character_logical_nan=list(id1=c("1","NaN"),id2=c(TRUE,NaN),y1=c(1,1),y2=c(2,2)),
    logical_na_character=list(id1=c(TRUE,NA),id2=c("TRUE",NA),y1=c(1,1),y2=c(2,2)),
    character_logical_na=list(id1=c("TRUE",NA),id2=c(TRUE,NA),y1=c(1,1),y2=c(2,2)),
    mixed_character_labels=list(id1=c(TRUE,1.5,"a"),id2=c("TRUE","1.5","a"),
                                y1=c(1,1,1),y2=c(2,2,2)),
    character_mixed_labels=list(id1=c("TRUE","1.5","a"),id2=c(TRUE,1.5,"a"),
                                y1=c(1,1,1),y2=c(2,2,2)),
    integer_double_character=list(id1=c(1000000000L,.1),id2=c("1e+09","0.1"),
                                  y1=c(1,1),y2=c(2,2)),
    character_integer_double=list(id1=c("1e+09","0.1"),id2=c(1000000000L,.1),
                                  y1=c(1,1),y2=c(2,2)),
    factor_numeric=list(id1=factor(c(2,1,2),levels=c(2,1,3)),id2=c(1,2,3),
                        y1=c(1,2,5),y2=c(2,3,4)),
    numeric_factor=list(id1=c(1,2,1),id2=factor(c(2,1,1),levels=c(2,1,3)),
                        y1=c(1,2,5),y2=c(2,3,4)),
    factors=list(id1=factor(c("b","a","b"),levels=c("b","a","unused")),
                 id2=factor(c("a","b","b"),levels=c("a","unused","b")),
                 y1=c(1,2,5),y2=c(2,3,4)),
    different_factor_levels=list(id1=factor(c("b","a","b"),levels=c("b","a","unused")),
                 id2=factor(c("a","b","b"),levels=c("a","b")),
                 y1=c(1,2,5),y2=c(2,3,4)),
    decimals=list(id1=c(.1,1e-7,1e20,-0),id2=c("0.1","1e-07","1e+20","0"),
                  y1=c(1,1,1,1),y2=c(2,2,2,2)),
    integer_labels=list(id1=c(1000000000L,12L),id2=c("1000000000","12"),
                        y1=c(1,1),y2=c(2,2)),
    duplicates=list(id1=c(1,2,1,2),id2=c("2","1","1","2"),
                     y1=c(2,2,3,3),y2=c(2,2,2,2)),
    missing_cross_types=list(id1=c(NA,NA,1,2),id2=c(NA,"1","2","2"),
                             y1=c(NA,1,1,2),y2=c(1,2,NA,3)),
    missing_numeric=list(id1=c(NA,NA,1),id2=c(NA,1),y1=c(NA,1,1),y2=c(1,2)),
    numeric_na_nan_to_na=list(id1=c(NA_real_,NaN),id2=NA_real_,y1=c(2,0),y2=1),
    numeric_na_nan_to_nan=list(id1=c(NA_real_,NaN),id2=NaN,y1=c(2,0),y2=1),
    repeated_numeric_missing=list(id1=c(NA_real_,NaN,NA_real_,NaN,1),id2=c(NaN,NA_real_,1),
                                  y1=c(0,0,2,2,1),y2=c(1,1,1)),
    nan_to_character=list(id1=c(NA_real_,NaN),id2=c(NA_character_,"NaN"),y1=c(2,0),y2=c(1,1)),
    character_to_nan=list(id1=c(NA_character_,"NaN"),id2=c(NA_real_,NaN),y1=c(2,0),y2=c(1,1)),
    nan_to_factor=list(id1=c(NA_real_,NaN),id2=factor(c(NA,"NaN"),levels=c("NaN","unused")),y1=c(2,0),y2=c(1,1)),
    factor_to_nan=list(id1=factor(c(NA,"NaN"),levels=c("NaN","unused")),id2=c(NA_real_,NaN),y1=c(2,0),y2=c(1,1)),
    factor_na_sentinels=list(id1=factor(c(NA,NA),levels="NaN"),id2=c(NaN,NA_real_),y1=c(2,0),y2=c(1,1)),
    unmatched_nan=list(id1=NaN,id2=NA_real_,y1=1,y2=1),
    rounded_numeric=list(id1=c(1,1+1e-15,1,1+1e-15),id2="1",
                          y1=c(1,1,3,3),y2=2),
    no_matching_ids=list(id1=c(1,2),id2=c("a","b"),y1=c(1,1),y2=c(2,2)))
near_cases <- list()
for (name in names(near)) for (best in c("after","prior")) for (missing in c(FALSE,TRUE)) {
    input <- near[[name]]
    nomatch <- if (missing) NA_integer_ else -1L
    result <- tryCatch(do.call(neardate,c(input,list(best=best,nomatch=nomatch))),
                       error=function(e) list(error=conditionMessage(e)))
    stock_error <- NULL
    if (name == "different_factor_levels") {
        stock_error <- result$error
        common <- input
        common$id1 <- as.character(input$id1)
        common$id2 <- as.character(input$id2)
        result <- do.call(neardate,c(common,list(best=best,nomatch=nomatch)))
    }
    expected <- if (is.list(result)) result else I(ifelse(result > 0,result-1L,result))
    near_cases[[length(near_cases)+1L]] <- list(
        name=paste(name,best,if(missing) "default" else "sentinel",sep="_"),
        id1=column(input$id1),id2=column(input$id2),y1=I(input$y1),y2=I(input$y2),
        best=best,nomatch=nomatch,stock_error=stock_error,expected=expected)
}

carry <- list(
    declared_order=list(id=c(1,1,2,2),x=factor(c("b",NA,"a",NA),levels=c("b","a","unused"))),
    ordered_factor=list(id=c(1,1,2,2),x=ordered(c("b",NA,"a",NA),levels=c("b","a","unused"))),
    missing_initial=list(id=c(1,1,1,2,2),x=factor(c(NA,"yes",NA,NA,"no"),levels=c("no","yes","unused"))),
    numeric_levels=list(id=c(1,1,1,2),x=factor(c(NA,1,NA,0),levels=c(0,1,2))),
    empty_factor=list(id=numeric(),x=factor(character(),levels=c("censor","event"))),
    time_order=list(id=c(2,1,2,1,2,1),x=factor(c(NA,"yes",NA,NA,"no",NA),levels=c("no","yes")),
                    time=c(NA,2,1,NA,2,1)))
carry_cases <- lapply(names(carry),function(name) {
    input <- carry[[name]]
    list(name=name,id=I(input$id),x=column(input$x),
         time=if(is.null(input$time)) NULL else I(input$time),
         expected=snapshot(do.call(lvcf,input)))
})

stutter <- list(
    numeric=list(id=rep(1,4),x=c(0,1,1,2),censor=0),
    character=list(id=rep(1,4),x=c("censor","b","b","a"),censor="censor"),
    factor_order=list(id=rep(1,4),x=factor(c("censor","b","b","a"),levels=c("censor","b","a")),censor="censor"),
    ordered_factor=list(id=rep(1,4),x=ordered(c("censor","b","b","a"),levels=c("censor","b","a")),censor="censor"),
    censor_existing=list(id=rep(1,4),x=factor(c("b","b","a","a"),levels=c("b","a")),censor="a"),
    missing=list(id=rep(1,5),x=c(NA,"b","b","a",NA),censor=0),
    all_missing=list(id=c(1,1),x=c(NA_real_,NA_real_),censor=0),
    decimal=list(id=rep(1,4),x=c(.25,.25,.5,.5),censor=0),
    no_repeats=list(id=c(1,1),x=c("a","b"),censor=0),
    unused_levels=list(id=rep(1,4),x=factor(c("b","b","a","a"),levels=c("censor","b","a","unused")),censor="censor"),
    single=list(id=c(1,1,1,1,2,2,2),x=c("a","b","a","c","a","a","b"),censor=0,single=TRUE),
    empty=list(id=numeric(),x=numeric(),censor=0))
intended <- list(
    no_repeats=factor(c("a","b"),levels=c("0","a","b")),
    unused_levels=factor(c("b","censor","a","censor"),levels=c("censor","b","a","unused")),
    single=factor(c("a","b","0","c","a","0","b"),levels=c("0","a","b","c")),
    empty=factor(character(),levels="0"))
stutter_cases <- lapply(names(stutter),function(name) {
    input <- stutter[[name]]
    result <- tryCatch(do.call(nostutter,input),error=function(e)e)
    error <- if (inherits(result,"error")) conditionMessage(result) else NULL
    if (!is.null(error)) result <- intended[[name]]
    list(name=name,id=I(input$id),x=column(input$x),censor=input$censor,
         single=isTRUE(input$single),stock_error=error,expected=snapshot(result))
})
reference <- list(metadata=list(generator="scripts/generate_data_prep_labels_reference.R",
    r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
    neardate=near_cases,lvcf=carry_cases,nostutter=stutter_cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(near_cases)+length(carry_cases)+length(stutter_cases),"data-preparation label cases written to",output,"\n")
