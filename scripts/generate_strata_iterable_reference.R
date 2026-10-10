#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/strata_iterable_reference.json"
sources <- list(
  integers=c(10,2,10,NA,2,30,NA),
  doubles=c(.5,2,.5,NA,-1,2,NA),
  logical=c(TRUE,FALSE,NA,TRUE,FALSE,NA,TRUE),
  character=c("b","a","b",NA,"long","a",NA),
  factor=factor(c("b","a","b",NA,"long","a",NA),
    levels=c("b","a","unused label","long")),
  numeric_factor=factor(c("10","2","10",NA,"30","2",NA),
    levels=c("2","30","10","unused")),
  missing=rep(NA_real_,7)
)
inputs <- lapply(names(sources), function(name) list(name=name,columns=list(sources[[name]])))
inputs <- c(inputs,list(
  list(name="numeric_character",columns=list(sources$integers,sources$character)),
  list(name="logical_factor",columns=list(sources$logical,sources$factor)),
  list(name="factor_numeric",columns=list(sources$factor,sources$integers)),
  list(name="numeric_factor_character",columns=list(sources$numeric_factor,sources$character)),
  list(name="three_columns",columns=list(sources$integers,sources$logical,sources$factor))))
snapshot <- function(value) list(codes=I(unname(as.integer(value)-1L)),
  levels=I(levels(value)),labels=I(unname(as.character(value))),
  counts=I(unname(as.integer(table(value)))))
response <- function(value) list(type=attr(value,"type"),states=I(attr(value,"states")),
  clabel=attr(value,"clabel"),matrix=unname(as.matrix(value)))
column <- function(value) list(values=I(unname(as.vector(value))),
  levels=if (is.factor(value)) I(levels(value)) else NULL,kind=typeof(value))
cases <- list()
for (input in inputs) for (named in c(FALSE,TRUE)) for (na in c(FALSE,TRUE)) {
  for (short in list(NULL,FALSE,TRUE)) {
    # A bare Python list of only None has no numeric/character dtype, so the
    # all-missing positional case specifies its label convention explicitly.
    if (input$name=="missing" && !named && is.null(short)) next
    columns <- input$columns
    names(columns) <- paste0("v",seq_along(columns))
    options <- list(na.group=na,sep=" / ")
    if (!is.null(short)) options$shortlabel <- short
    call_arguments <- if (named) columns else lapply(names(columns),as.name)
    call <- as.call(c(list(quote(survival::strata)),call_arguments,options))
    value <- eval(call,list2env(columns,parent=globalenv()))
    cases[[length(cases)+1L]] <- list(
      name=paste(input$name,if (named) "named" else "positional",na,
        if (is.null(short)) "default" else short,sep="/"),
      named=named,columns=lapply(columns,column),
      options=list(na_group=na,shortlabel=short,sep=" / "),
      expected=snapshot(value),roundtrip=snapshot(strata(value)),
      right_response=response(Surv(seq_along(value),value)),
      counting_response=response(Surv(rep(0,length(value)),seq_along(value),value)))
  }
}
fit_data <- list(time=c(1,8,7,3,4,10,9,2,6,5,11,12,13,15,14,18,16,20),
  status=c(1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1),
  group=c(10,2,30,10,30,2,10,2,30,2,10,30,30,10,2,10,2,30),
  z=c(1,0,2,1,0,2,1,2,0,1,2,0,2,0,1,2,0,1))
d <- as.data.frame(fit_data)
d$g <- strata(d$group,shortlabel=TRUE)
fit <- survreg(Surv(time,status)~g+z,d,dist="gaussian",x=TRUE,model=TRUE)
new <- data.frame(group=c(30,10,2),z=c(1,2,0))
new$g <- strata(new$group,shortlabel=TRUE)
fit_case <- list(data=fit_data,newdata=as.list(new[c("group","z")]),
  formula="Surv(time,status) ~ g + z",dist="gaussian",
  beta_names=I(names(coef(fit))),beta=I(unname(coef(fit))),variance=unname(vcov(fit)),
  design=unname(fit$x),design_names=I(colnames(fit$x)),
  lp=I(unname(predict(fit,type="lp"))),prediction=I(unname(predict(fit,new,type="response"))),
  scale=I(unname(fit$scale)))
km <- survfit(Surv(time,status)~g,d)
fit_case$grouped_curves <- list(n=I(km$n),strata=as.list(km$strata),
  time=I(km$time),n_risk=I(km$n.risk),n_event=I(km$n.event),n_censor=I(km$n.censor),
  surv=I(km$surv))
state_labels <- factor(c("censor","NA",NA,"NA","censor"),
  levels=c("censor","NA","unused state"))
state_label_case <- list(values=I(as.character(state_labels)),levels=I(levels(state_labels)),
  expected=response(Surv(seq_along(state_labels),state_labels)))
reference <- list(metadata=list(generator="scripts/generate_strata_iterable_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
  cases=cases,fit=fit_case,state_label=state_label_case)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"strata input cases and one three-group AFT fit written to",output,"\n")
