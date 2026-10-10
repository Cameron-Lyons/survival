#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/formula_iterator_reference.json"

stock <- function(name) getFromNamespace(name, "survival")
numbers <- function(values) lapply(unname(values), function(value) {
  if (is.na(value)) return(NULL)
  if (is.numeric(value) && !is.finite(value)) return(if (value>0) "Inf" else "-Inf")
  value
})
matrix_rows <- function(value) {
  if (is.null(value)) return(NULL)
  lapply(seq_len(nrow(value)), function(row) numbers(value[row, ]))
}
na_record <- function(value) {
  if (is.null(value)) return(NULL)
  list(rows=numbers(unclass(value)),kind=if (inherits(value,"exclude")) "exclude" else "omit")
}
frame_columns <- function(frame) {
  columns <- list()
  for (name in names(frame)) {
    value <- frame[[name]]
    if (inherits(value,"Surv")) {
      matrix <- unclass(value)
      if (ncol(matrix)==3L) columns$start <- numbers(matrix[,1L])
      columns$time <- numbers(matrix[,ncol(matrix)-1L])
      columns$status <- numbers(matrix[,ncol(matrix)])
    } else columns[[name]] <- if (is.factor(value)) numbers(as.character(value)) else numbers(value)
  }
  columns
}
matrix_snapshot <- function(value) {
  list(data=matrix_rows(value),columns=I(colnames(value)),assign=I(attr(value,"assign")),
       row_names=I(rownames(value)),contrasts=attr(value,"contrasts"))
}
base <- list(time=c(4,2,9,6,5,7,8,1,11,10,3,12),
             status=c(1,0,1,1,0,1,0,1,1,0,1,0),
             x=c(2,-1,1,0,3,2,-2,1,0,2,-1,3),
             g=factor(c("b","a","c","a","b","c","a","b","c","a","b","c"),
                      levels=c("c","a","b","unused")),
             off=c(.1,0,-.1,.2,0,.1,-.2,0,.2,-.1,.1,0),
             wt=c(.5,1,2,1,.5,2,1,1,.5,2,1,.5),
             id=rep(1:6,2))
inputs <- list(complete=base,missing=base,ordered=base)
inputs$missing$x[9L] <- NA_real_
inputs$missing$g[5L] <- NA
inputs$missing$wt[11L] <- NA_real_
inputs$missing$status[7L] <- NA_real_
inputs$ordered$g <- ordered(as.character(base$g),levels=levels(base$g))
inputs$aliased <- base
inputs$aliased$wt <- base$time
encode_input <- function(data) lapply(data,function(value) {
  list(values=if(is.factor(value))numbers(as.character(value))else numbers(value),
       levels=if(is.factor(value))I(levels(value))else NULL,
       ordered=is.ordered(value),kind=if(is.factor(value))"factor"else typeof(value))
})
selections <- list(all=NULL,reordered=c(12L,3L,8L,3L,6L,2L,11L,1L),
                   logical=c(TRUE,FALSE,TRUE,TRUE,FALSE,TRUE,TRUE,FALSE,TRUE,TRUE,FALSE,TRUE))
formulas <- list(frame="Surv(time,status)~x+g",coxph="Surv(time,status)~x+offset(off)",
                 survreg="Surv(time,status)~x+offset(off)",survfit="Surv(time,status)~g",
                 aareg="Surv(time,status)~x",concordance="Surv(time,status)~x")
snapshot <- function(routine,fit) {
  if(routine=="frame")return(list(frame=frame_columns(fit),
    row_names=I(row.names(fit)),na_action=na_record(attr(fit,"na.action")),
    factor=list(values=numbers(as.character(fit$g)),levels=I(levels(fit$g)),ordered=is.ordered(fit$g))))
  action <- na_record(fit$na.action)
  if(routine=="coxph")return(list(coefficients=numbers(coef(fit)),var=matrix_rows(fit$var),
    loglik=numbers(fit$loglik),means=numbers(fit$means),linear_predictors=numbers(fit$linear.predictors),
    residuals=numbers(fit$residuals),predict=numbers(stock("predict.coxph")(fit)),
    expected=numbers(stock("predict.coxph")(fit,type="expected")),
    matrix=matrix_snapshot(stock("model.matrix.coxph")(fit)),frame=frame_columns(fit$model),na_action=action))
  if(routine=="survreg")return(list(coefficients=numbers(coef(fit)),var=matrix_rows(fit$var),
    loglik=numbers(fit$loglik),scale=fit$scale,linear_predictors=numbers(fit$linear.predictors),
    predict=numbers(stock("predict.survreg")(fit)),
    quantile=matrix_rows(stock("predict.survreg")(fit,type="quantile",p=c(.25,.75))),
    matrix=matrix_snapshot(stock("model.matrix.survreg")(fit)),frame=frame_columns(fit$model),na_action=action))
  if(routine=="survfit") {
    frame <- stock("summary.survfit")(fit,censored=TRUE,data.frame=TRUE,rmean="none")
    frame$strata <- as.character(frame$strata)
    return(list(n=numbers(fit$n),frame=lapply(frame,numbers),model=frame_columns(fit$model),na_action=action))
  }
  if(routine=="aareg")return(list(n=numbers(fit$n),times=numbers(fit$times),n_risk=numbers(fit$nrisk),
    coefficient=matrix_rows(fit$coefficient),test_statistic=numbers(fit$test.statistic),
    test_variance=matrix_rows(fit$test.var),time_weights=matrix_rows(fit$tweight),
    frame=frame_columns(fit$model),na_action=action))
  list(concordance=fit$concordance,count=as.list(fit$count),var=fit$var,n=fit$n,na_action=action)
}
cases <- list()
for(input in names(inputs))for(selection in names(selections))for(action in c("omit","exclude"))
  for(routine in names(formulas)) {
    if(input=="aliased" && (selection!="all" || action!="omit" ||
      routine %in% c("aareg","concordance")))next
    if(routine %in% c("aareg","concordance") && selection!="all")next
    formula <- as.formula(formulas[[routine]],env=asNamespace("survival"))
    data <- inputs[[input]]
    arguments <- list(formula=formula,data=data,weights=quote(wt),
      na.action=if(action=="omit")stats::na.omit else stats::na.exclude)
    selected <- selections[[selection]]
    if(!is.null(selected))arguments$subset <- selected
    if(routine=="frame") {
      function_name <- stats::model.frame
    } else {
      function_name <- stock(if(routine=="survfit")"survfit.formula"else
        if(routine=="concordance")"concordance.formula"else routine)
      if(routine!="concordance")arguments$model <- TRUE
      if(routine %in% c("coxph","survreg"))arguments$x <- TRUE
      if(routine=="survfit")arguments$timefix <- FALSE
      if(routine=="aareg")arguments$nmin <- 3L
      if(routine=="concordance")names(arguments)[1L] <- "object"
    }
    warnings <- character()
    result <- withCallingHandlers(tryCatch({
      fit <- do.call(function_name,arguments)
      if(routine!="frame")stopifnot(!any(grepl("^survival_py",class(fit))))
      if(routine=="survfit" && is.null(fit$model)) {
        # Stock ignores model=TRUE for this fitter. The port's retained frame
        # extension is compared with stock model.frame on the identical call.
        fit$model <- do.call(stats::model.frame,
          arguments[intersect(names(arguments),c("formula","data","weights","subset","na.action"))])
      }
      list(result=snapshot(routine,fit))
    },error=function(error)list(error=conditionMessage(error))),warning=function(warning) {
      warnings <<- c(warnings,conditionMessage(warning));invokeRestart("muffleWarning")
    })
    cases[[length(cases)+1L]] <- c(list(name=paste(routine,input,selection,action,sep="/"),
      routine=routine,input=input,formula=formulas[[routine]],subset=if(is.logical(selected))I(selected)
      else if(is.null(selected))NULL else I(selected-1L),na_action=action,warnings=I(warnings)),result)
  }
row_counts <- list()
for(first in c(TRUE,FALSE))for(formula in c("~x","~1","~I(2)"))for(kind in c("list","frame")) {
  data <- if(first)list(unused=1:3,x=4:6)else list(x=4:6,unused=1:3)
  if(kind=="frame")data <- as.data.frame(data)
  frame <- stats::model.frame(as.formula(formula),data)
  row_counts[[length(row_counts)+1L]] <- list(formula=formula,kind=kind,unused_first=first,
    nrow=nrow(frame),row_names=I(row.names(frame)),columns=frame_columns(frame))
}
extra_rows <- list()
for(formula in c("~1","~x"))for(kind in c("list","frame"))for(size in c(0L,1L,4L)) {
  data <- list(unused=1:4,x=5:8)
  if(kind=="frame")data <- as.data.frame(data)
  arguments <- list(formula=as.formula(formula),data=data,na.action=stats::na.pass)
  if(size)arguments$weights <- rep(1,size)
  result <- tryCatch({
    frame <- do.call(stats::model.frame,arguments)
    list(nrow=nrow(frame),row_names=I(row.names(frame)),columns=frame_columns(frame))
  },error=function(error)list(error=conditionMessage(error)))
  extra_rows[[length(extra_rows)+1L]] <- c(list(formula=formula,kind=kind,
    weights=if(size)I(rep(1,size))else NULL),result)
}
prediction_inputs <- list(complete=lapply(base,function(value)value[c(12L,3L,8L,3L,6L,2L,11L,1L)]))
# Keep new-data offsets zero for unmodified stock prediction comparisons:
# stock drops AFT new offsets and adds Cox expected offsets outside exp().
# Those documented numerical differences have dedicated corrected references.
prediction_inputs$complete$off[] <- 0
prediction_inputs$missing <- prediction_inputs$complete
prediction_inputs$missing$x[2L] <- NA_real_
prediction_inputs$missing$time[6L] <- NA_real_
prediction_inputs$missing$status[7L] <- NA_real_
prediction_cases <- list()
prediction_value <- function(value) if(is.matrix(value))matrix_rows(value)else numbers(value)
for(routine in c("coxph","survreg")) {
  formula <- as.formula(formulas[[routine]],env=asNamespace("survival"))
  fit <- stock(routine)(formula,data=base,weights=wt,model=TRUE,x=TRUE)
  stopifnot(!any(grepl("^survival_py",class(fit))))
  types <- if(routine=="coxph")c("lp","risk","terms","expected","survival")else
    c("response","lp","terms","quantile","uquantile")
  for(input in names(prediction_inputs))for(action in c("pass","omit","exclude"))
    for(type in types)for(se in c(FALSE,TRUE)) {
      warnings <- character()
      result <- withCallingHandlers(tryCatch({
        value <- stock(paste0("predict.",routine))(fit,newdata=prediction_inputs[[input]],
          type=type,se.fit=se,p=c(.25,.75),na.action=get(paste0("na.",action),asNamespace("stats")))
        list(result=if(se)list(fit=prediction_value(value$fit),se_fit=prediction_value(value$se.fit))
          else prediction_value(value))
      },error=function(error)list(error=conditionMessage(error))),warning=function(warning) {
        warnings <<- c(warnings,conditionMessage(warning));invokeRestart("muffleWarning")
      })
      prediction_cases[[length(prediction_cases)+1L]] <- c(list(
        name=paste(routine,input,action,type,se,sep="/"),routine=routine,input=input,
        na_action=action,type=type,se_fit=se,p=I(c(.25,.75)),warnings=I(warnings)),result)
    }
  if(routine=="coxph")for(type in c("lp","risk"))for(se in c(FALSE,TRUE)) {
    newdata <- prediction_inputs$complete
    value <- stock("predict.coxph")(fit,newdata=newdata,type=type,se.fit=se,collapse=newdata$x)
    prediction_cases[[length(prediction_cases)+1L]] <- list(
      name=paste(routine,"complete","collapse-x",type,se,sep="/"),routine=routine,input="complete",
      na_action="pass",type=type,se_fit=se,collapse="x",warnings=I(character()),
      result=if(se)list(fit=prediction_value(value$fit),se_fit=prediction_value(value$se.fit))
        else prediction_value(value))
  }
}
direct_inputs <- list(
  survcheck=list(start=c(0,1,0,2,NA),stop=c(1,2,1,3,4),status=c(1,0,0,1,1),id=c(1,1,2,2,3)),
  survdiff=list(time=1:8,status=c(1,1,0,1,0,1,1,0),g=c("A","A",NA,"A","B","B","B","B")))
direct_cases <- list()
for(routine in names(direct_inputs))for(selection in c("all","reordered")) {
  data <- direct_inputs[[routine]]
  selected <- if(selection=="all")seq_along(data[[1L]])else
    if(routine=="survcheck")c(3L,4L,1L,2L,5L)else c(7L,1L,2L,3L,5L,6L)
  formula <- as.formula(if(routine=="survcheck")"Surv(start,stop,status)~1"else
    "Surv(time,status)~g",env=asNamespace("survival"))
  arguments <- list(formula=formula,data=data,subset=selected)
  if(routine=="survcheck")arguments$id <- quote(id)
  fit <- do.call(stock(routine),arguments)
  stopifnot(!any(grepl("^survival_py",class(fit))))
  if(routine=="survcheck") {
    result <- list(states=I(fit$states),istate=I(fit$istate),n=as.list(fit$n),id=numbers(fit$id),
      na_action=if(is.null(fit$na.action))NULL else numbers(unclass(fit$na.action)),
      flag=as.list(fit$flag),
      transitions=list(from_states=I(rownames(fit$transitions)),to_states=I(colnames(fit$transitions)),
        counts=matrix_rows(fit$transitions)),
      events=list(states=I(rownames(fit$events)),count=numbers(as.numeric(colnames(fit$events))),
        subjects=matrix_rows(fit$events)))
    for(name in c("overlap","gap","jump","teleport"))result[name] <- list(
      if(is.null(fit[[name]]))NULL else lapply(fit[[name]],numbers))
  } else result <- list(n=numbers(fit$n),groups=I(sub("^g=","",names(fit$n))),
    obs=numbers(fit$obs),exp=numbers(fit$exp),var=matrix_rows(fit$var),
    chisq=fit$chisq,pvalue=fit$pvalue,df=length(fit$n)-1L,
    na_action=na_record(fit$na.action))
  direct_cases[[length(direct_cases)+1L]] <- list(name=paste(routine,selection,sep="/"),
    routine=routine,subset=I(selected-1L),result=result)
}
special_inputs <- list(typed_double=base,cancelled=base)
special_inputs$typed_double$x <- as.double(1200000000 + base$x*100000000)
special_inputs$cancelled$z <- c(1,NA,3,4,NA,6,7,8,9,10,11,12)
special_cases <- list()
for(input in names(special_inputs))for(selection in c("all","reordered")) {
  if(input=="cancelled" && selection!="all")next
  data <- special_inputs[[input]]
  formula_string <- if(input=="typed_double")"Surv(time,status)~I(x*2L)+offset(off)"else
    "Surv(time,status)~x+z-z"
  formula <- as.formula(formula_string,env=asNamespace("survival"))
  selected <- if(selection=="all")NULL else selections$reordered
  fit <- stock("coxph")(formula,data=data,weights=wt,subset=selected,na.action=stats::na.pass,
    model=FALSE,x=TRUE)
  stopifnot(!any(grepl("^survival_py",class(fit))))
  # A separate stock method rebuilds the unretained model frame, including the
  # formally evaluated z that was removed from the design by +z-z.
  fit$model <- stock("model.frame.coxph")(fit)
  source_frame <- if(input=="typed_double")frame_columns(stats::model.frame(
    as.formula("Surv(time,status)~x+offset(off)",env=asNamespace("survival")),
    data=data,weights=wt,subset=selected,na.action=stats::na.pass))else NULL
  special_cases[[length(special_cases)+1L]] <- list(name=paste(input,selection,sep="/"),
    routine="coxph",input=input,formula=formula_string,subset=if(is.null(selected))NULL else I(selected-1L),
    na_action="pass",source_frame=source_frame,result=snapshot("coxph",fit))
}
selector <- c(TRUE,FALSE,TRUE)
alias_data <- list(time=1:3,status=c(1,0,1),x=c(10,20,30),keep=selector,wt=selector,id=selector)
subset_alias_cases <- list()
for(name in c("frame-source","frame-response","frame-weights","coxph-weights",
             "km-source","km-response","direct-km-group","direct-check-id")) {
  data <- alias_data
  source <- if(grepl("response",name))"status"else if(grepl("weights",name))"wt"else
    if(name=="direct-check-id")"id"else "keep"
  if(source=="status")data$status <- selector
  formula_string <- if(name %in% c("frame-source","km-source","direct-km-group"))
    "Surv(time,status)~keep"else if(grepl("^frame",name))"Surv(time,status)~x"else
    "Surv(time,status)~1"
  arguments <- list(formula=as.formula(formula_string,env=asNamespace("survival")),
    data=data,subset=as.name(source))
  if(grepl("weights",name))arguments$weights <- quote(wt)
  if(name=="direct-check-id")arguments$id <- quote(id)
  routine <- if(grepl("^frame",name))"frame"else if(name=="coxph-weights")"coxph"else
    if(name=="direct-check-id")"survcheck"else "survfit"
  fitter <- if(routine=="frame")stats::model.frame else
    stock(if(routine=="survfit")"survfit.formula"else routine)
  fit <- do.call(fitter,arguments)
  if(routine!="frame")stopifnot(!any(grepl("^survival_py",class(fit))))
  if(routine=="frame")result <- frame_columns(fit)else if(routine=="survfit") {
    fields <- c("n","time","n.risk","n.event","n.censor","surv","std.err","lower","upper","cumhaz")
    result <- lapply(unclass(fit)[fields],numbers)
    names(result) <- gsub("\\.","_",names(result))
    result["strata"] <- list(if(is.null(fit$strata))NULL else as.list(fit$strata))
  } else if(routine=="coxph")result <- list(n=fit$n,nevent=fit$nevent,
    coefficients=numbers(coef(fit)),loglik=numbers(fit$loglik),
    predict=lapply(setNames(c("lp","risk","expected","survival"),c("lp","risk","expected","survival")),
      function(type)numbers(stock("predict.coxph")(fit,type=type))))else {
    result <- list(states=I(fit$states),istate=I(fit$istate),n=as.list(fit$n),id=numbers(fit$id),
      na_action=if(is.null(fit$na.action))NULL else numbers(unclass(fit$na.action)),flag=as.list(fit$flag),
      transitions=list(from_states=I(rownames(fit$transitions)),to_states=I(colnames(fit$transitions)),
        counts=matrix_rows(fit$transitions)),
      events=list(states=I(rownames(fit$events)),count=numbers(as.numeric(colnames(fit$events))),
        subjects=matrix_rows(fit$events)))
    for(problem in c("overlap","gap","jump","teleport"))result[problem] <- list(
      if(is.null(fit[[problem]]))NULL else lapply(fit[[problem]],numbers))
  }
  subset_alias_cases[[length(subset_alias_cases)+1L]] <- list(name=name,routine=routine,
    formula=formula_string,source=source,input=encode_input(data),result=result)
}
reference <- list(provenance=list(generator="scripts/generate_formula_iterator_reference.R",
  R=as.character(getRversion()),survival=as.character(packageVersion("survival")),
  reference="Direct stock survival namespace fitters/methods and stats::model.frame"),
  inputs=lapply(inputs,encode_input),cases=cases,row_count_rules=row_counts,
  extra_row_count_rules=extra_rows,prediction_inputs=lapply(prediction_inputs,encode_input),
  prediction_cases=prediction_cases,direct_inputs=lapply(direct_inputs,encode_input),
  direct_cases=direct_cases,special_inputs=lapply(special_inputs,encode_input),
  special_cases=special_cases,subset_alias_cases=subset_alias_cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,pretty=TRUE,digits=17,na="null",null="null")
cat(length(cases),"formula fit/frame cases and",length(row_counts)+length(extra_rows),
  "row-count controls and",length(prediction_cases),"predictions written to",output,"\n")
