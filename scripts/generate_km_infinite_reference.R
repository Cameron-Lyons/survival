#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/km_infinite_reference.json"
stock_fit <- getFromNamespace("survfit.formula", "survival")
stock_summary <- getFromNamespace("summary.survfit", "survival")
stock_quantile <- getFromNamespace("quantile.survfit", "survival")
stock_zero <- getFromNamespace("survfit0", "survival")
stock_aeq <- getFromNamespace("aeqSurv", "survival")
stock_subset <- getFromNamespace("[.survfit", "survival")

encode <- function(value) {
  if (is.null(value)) return(NULL)
  if (is.matrix(value)) return(lapply(seq_len(nrow(value)), function(i) encode(unclass(value)[i,])))
  if (is.list(value)) return(lapply(value, encode))
  lapply(unname(value), function(item) {
    if (is.na(item)) return(NULL)
    if (is.numeric(item) && !is.finite(item)) return(if (item>0) "Inf" else "-Inf")
    if (is.numeric(item) && item==0 && is.infinite(1/item) && 1/item<0) return("-0")
    item
  })
}
fields <- c("n","time","n.risk","n.event","n.censor","n.enter","n.id",
  "surv","cumhaz","std.err","std.chaz","lower","upper","strata","t0",
  "influence.surv","influence.chaz")
snapshot <- function(fit) {
  value <- setNames(lapply(fields, function(field) encode(fit[[field]])),
    gsub("\\.","_",fields))
  value$counts <- if (is.null(fit$counts)) NULL else
    setNames(lapply(seq_len(ncol(fit$counts)),function(i) encode(fit$counts[,i])),
      sub("^n[._]?","n_",colnames(fit$counts)))
  value
}
capture <- function(call, transform=identity) {
  warnings <- character()
  result <- withCallingHandlers(tryCatch(list(value=transform(call())),
    error=function(error) list(error=conditionMessage(error))), warning=function(warning) {
      warnings <<- c(warnings,conditionMessage(warning))
      invokeRestart("muffleWarning")
    })
  c(result,list(warnings=I(warnings)))
}

# Stock aeqSurv computes finite cutpoints, then indexes all endpoints with
# them. Positive infinity becomes the last finite cut; a zero index for
# negative infinity drops a value before R recycles/reconstructs columns.
# Keep those raw outcomes below. The corrected oracle uses stock aeqSurv
# only on finite entries, then inserts its values back into the same rows.
finite_aeq <- function(response) {
  columns <- seq_len(ncol(response)-1L)
  times <- unclass(response)[,columns,drop=FALSE]
  finite <- is.finite(times)
  if (any(finite)) {
    fixed <- stock_aeq(Surv(times[finite], rep(0,sum(finite))))
    times[finite] <- unclass(fixed)[,1]
  }
  response[,columns] <- times
  response
}

inputs <- list(
  finite=list(time=c(1,2,2,3,4,5),status=c(1,0,1,0,1,1)),
  positive=list(time=c(1,2,2,3,Inf,Inf),status=c(1,0,1,0,1,0)),
  negative=list(time=c(-Inf,-Inf,1,2,3,4),status=c(1,0,1,0,1,1)),
  mixed_near=list(time=c(-Inf,1,1+1e-12,2,3,Inf),status=c(1,0,1,0,1,1)),
  positive_only=list(time=rep(Inf,4),status=c(1,0,1,0)),
  negative_only=list(time=rep(-Inf,4),status=c(1,0,1,0)),
  counting=list(start=c(-Inf,-Inf,0,1,2,3),time=c(1,2,2,3,Inf,Inf),status=c(1,0,1,0,1,0)),
  counting_near=list(start=c(-Inf,0,0,1,2,3),time=c(1,1+1e-12,2,3,Inf,Inf),status=c(1,0,1,0,1,0)),
  counting_positive_only=list(start=c(-Inf,0,1,2),time=rep(Inf,4),status=c(1,0,1,0))
)
cases <- list()
summary_count <- quantile_count <- 0L
for (input_name in names(inputs)) {
  input <- inputs[[input_name]]
  rows <- length(input$time)
  response <- if (is.null(input$start)) Surv(input$time,input$status) else
    Surv(input$start,input$time,input$status)
  for (timefix in c(FALSE,TRUE)) for (grouped in c(FALSE,TRUE))
    for (weighted in c(FALSE,TRUE)) for (type in list(c(1,1),c(2,2))) {
      group <- if (grouped) rep(c("a","b"),length.out=rows) else NULL
      weights <- if (weighted) rep(c(.5,1,2),length.out=rows) else NULL
      formula <- as.formula(if (grouped) "response ~ group" else "response ~ 1",
        env=asNamespace("survival"))
      data <- if (is.null(group)) list(response=response) else list(response=response,group=group)
      arguments <- list(formula=formula,data=data,timefix=timefix,
        stype=type[[1]],ctype=type[[2]],id=seq_len(rows),entry=!is.null(input$start))
      if (weighted) arguments <- c(arguments,list(weights=weights,influence=3))
      raw <- capture(function() do.call(stock_fit,arguments),snapshot)
      fixed <- if (timefix) finite_aeq(response) else response
      corrected <- timefix && any(!is.finite(unclass(response)[,-ncol(response)])) &&
        !identical(unclass(fixed),unclass(response))
      arguments$data <- if (is.null(group)) list(response=fixed) else list(response=fixed,group=group)
      arguments$timefix <- FALSE
      fit <- do.call(stock_fit,arguments)
      summaries <- list()
      requests <- list(
        list(times=c(-Inf,0,2,Inf),extend=FALSE),
        list(times=c(-Inf,0,2,Inf),extend=TRUE),
        list(times=c(Inf,2,2,-Inf,0),extend=FALSE),
        list(times=c(Inf,2,2,-Inf,0),extend=TRUE),
        list(times=c(-Inf,-Inf,0,Inf),extend=TRUE,dosum=FALSE),
        list(times=Inf,extend=FALSE,dosum=TRUE),
        list(times=Inf,extend=TRUE,dosum=TRUE),
        list(times=c(0,1,2,4),extend=TRUE,dosum=TRUE),
        list(times=c(0,1,2,4),extend=TRUE,dosum=FALSE)
      )
      for (request in requests) {
        raw_summary <- capture(function() do.call(stock_summary,
          c(list(object=fit,rmean="none"),request)),snapshot)
        value <- capture(function() {
          result <- do.call(stock_summary,c(list(object=fit,rmean="none"),request))
          count_fields <- c("n.event","n.censor","n.enter","surv","cumhaz",
            "std.err","std.chaz","lower","upper")
          missing_counts <- count_fields[vapply(count_fields,function(field)
            is.null(result[[field]]) && !is.null(fit[[field]]),logical(1))]
          if (length(missing_counts)) {
            # Stock unlistsurv loses fitted fields when its first stratum
            # selects no times. Reuse the same stock kernels per stratum and
            # concatenate only those fields, preserving the global time rows.
            groups <- if (is.null(fit$strata)) list(fit) else
              lapply(seq_along(fit$strata),function(i) stock_subset(fit,i))
            pieces <- lapply(groups,function(group) do.call(stock_summary,
              c(list(object=group,rmean="none"),request)))
            for (field in missing_counts) result[[field]] <-
              do.call(c,c(list(numeric()),lapply(pieces,function(piece) piece[[field]])))
          }
          result
        },snapshot)
        summaries[[length(summaries)+1L]] <- list(
          times=encode(request$times),extend=request$extend,dosum=request$dosum,
          expected=value,raw_expected=raw_summary)
        summary_count <- summary_count+1L
      }
      quantiles <- list()
      for (scale in c(1,-2,Inf)) for (confidence in c(FALSE,TRUE)) {
        value <- capture(function() stock_quantile(fit,probs=c(0,.25,.5,.75,1),
          scale=scale,conf.int=confidence),function(value) {
            if (is.list(value)) lapply(value,encode) else list(quantile=encode(value))
          })
        quantiles[[length(quantiles)+1L]] <- list(scale=encode(scale),
          confidence=confidence,expected=value)
        quantile_count <- quantile_count+1L
      }
      tables <- lapply(list("common","none",3),function(rmean) {
        value <- capture(function() stock_summary(fit,rmean=rmean),function(value)
          list(table=encode(value$table),endtime=encode(value$rmean.endtime)))
        list(rmean=rmean,expected=value)
      })
      zero <- stock_zero(fit)
      raw_zero <- snapshot(zero)
      # Stock survfit0 adds a zero influence column to every stratum once any
      # curve needs a time-zero row, even to a curve already starting at t0.
      # Retain the raw result and remove only those unmatched columns.
      sizes <- if (is.null(fit$strata)) length(fit$time) else unname(fit$strata)
      begins <- c(1L,head(cumsum(sizes),-1L)+1L)
      inserted <- fit$time[begins] != fit$t0
      if (!is.null(fit$counts) && any(inserted)) {
        pieces <- lapply(seq_along(sizes),function(curve) {
          rows <- seq.int(begins[curve],length.out=sizes[curve])
          values <- fit$counts[rows,,drop=FALSE]
          if (inserted[curve]) {
            first <- matrix(0,1,ncol(values),dimnames=list(NULL,colnames(values)))
            first[,"nrisk"] <- values[1,"nrisk"]
            values <- rbind(first,values)
          }
          values
        })
        zero$counts <- do.call(rbind,pieces)
      }
      if (!is.null(fit$strata)) {
        if (any(inserted) && any(!inserted)) for (field in c("influence.surv","influence.chaz")) {
          if (!is.null(zero[[field]])) for (curve in which(!inserted))
            zero[[field]][[curve]] <- zero[[field]][[curve]][,-1,drop=FALSE]
        }
      }
      name <- paste(input_name,timefix,grouped,weighted,paste(type,collapse=""),sep="_")
      cases[[length(cases)+1L]] <- list(name=name,input=input_name,timefix=timefix,
        group=encode(group),weights=encode(weights),stype=type[[1]],ctype=type[[2]],
        corrected_infinite_timefix=corrected,raw_stock=raw,fit=snapshot(fit),
        zero=snapshot(zero),raw_zero=raw_zero,summaries=summaries,quantiles=quantiles,tables=tables)
    }
}

aeq_cases <- list()
for (input_name in names(inputs)) {
  input <- inputs[[input_name]]
  response <- if (is.null(input$start)) Surv(input$time,input$status) else
    Surv(input$start,input$time,input$status)
  aeq_cases[[length(aeq_cases)+1L]] <- list(name=input_name,
    raw_stock=capture(function() stock_aeq(response),encode),
    expected=encode(finite_aeq(response)))
}
starts <- list()
for (input_name in c("finite","positive")) for (start in c(-Inf,Inf,NaN)) {
  input <- inputs[[input_name]]
  formula <- as.formula("Surv(time,status) ~ 1",env=asNamespace("survival"))
  value <- capture(function() stock_fit(formula,data=input,start.time=start),snapshot)
  starts[[length(starts)+1L]] <- list(input=input_name,start=encode(start),expected=value)
}
reference <- list(metadata=list(generator="scripts/generate_km_infinite_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival")),
  fit_calls=length(cases),summary_calls=summary_count,quantile_calls=quantile_count,
  reference_note="Finite-time normalization uses stock aeqSurv on finite entries only, preserving infinite endpoints and row alignment. Lost summary fields use direct stock per-stratum kernels; entirely empty results retain numeric-vector shapes. Stock survfit0 counts receive the inserted zero rows, and unmatched influence columns are removed only from curves already at t0. Raw unmodified stock outputs and warnings are retained."),
  inputs=lapply(inputs,function(input) lapply(input,encode)),cases=cases,
  aeq_cases=aeq_cases,start_cases=starts)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
serialize <- function(value) toJSON(value,auto_unbox=TRUE,digits=17,na="null",null="null")
header <- serialize(reference[names(reference) != "cases"])
writeLines(c(substring(header,1L,nchar(header)-1L), ',"cases":[',
  vapply(seq_along(cases),function(i) paste0(serialize(cases[[i]]),
    if (i < length(cases)) "," else ""), ""), "]}"), output,useBytes=TRUE)
cat(length(cases),"fitted cases,",summary_count,"summary and",quantile_count,
  "quantile calls written to",output,"\n")
