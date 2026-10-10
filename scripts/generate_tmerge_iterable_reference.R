#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/tmerge_iterable_reference.json"
encode <- function(value) list(values=I(if (is.factor(value)) as.character(value) else value),
  levels=if (is.factor(value)) I(levels(value)) else NULL,kind=typeof(value))
snapshot <- function(value) {
  retained <- attr(value,"tm.retain")
  counts <- attr(value,"tcount")
  list(columns=lapply(value,encode),tname=retained$tname,
    tevent=retained$tevent$censor %||% list(),tdcvar=I(retained$tdcvar %||% character()),
    tcount=setNames(lapply(seq_len(nrow(counts)),function(row) as.list(counts[row,])),rownames(counts)))
}
`%||%` <- function(x,y) if (is.null(x)) y else x
base <- data.frame(id=c(3L,1L,2L),time=c(12,10,8),status=c(1L,0L,1L),x=c("c","a","b"))
initial <- tmerge(base,base,id=id,death=event(time,status))
long <- data.frame(id=c(1L,3L,1L,2L,3L,2L,9L,1L),
  time=c(-1,0,4,8,12,10,1,4),lab=c(2,3,NA,1,2,3,4,5),
  event=c(0L,1L,1L,1L,1L,1L,1L,0L),
  state=factor(c("wait","a",NA,"b","a","b","wait","b"),levels=c("wait","unused","b","a")))
call_tmerge <- function(data1,data2,operations,id="id",tstart=NULL,tstop=NULL,options=list(),direct=FALSE) {
  source <- function(name) if (direct) data2[[name]] else as.name(name)
  arguments <- lapply(operations,function(operation) {
    values <- list(as.name(operation$kind),source(operation$time))
    if (!is.null(operation$value)) values <- c(values,list(source(operation$value)))
    if (!is.null(operation$init)) values$init <- operation$init
    as.call(values)
  })
  call <- c(list(quote(tmerge),quote(data1),quote(data2),id=source(id)),arguments)
  if (!is.null(tstart)) call$tstart <- if (is.character(tstart)) source(tstart) else tstart
  if (!is.null(tstop)) call$tstop <- if (is.character(tstop)) source(tstop) else tstop
  if (length(options)) call$options <- options
  eval(as.call(call))
}
cases <- list()
add <- function(name,data1,data2,operations,id="id",tstart=NULL,tstop=NULL,
  options=list(),update=FALSE,same=FALSE,direct=FALSE) {
  warnings <- character()
  result <- withCallingHandlers(tryCatch(list(result=snapshot(call_tmerge(
      data1,data2,operations,id,tstart,tstop,options,direct))),
    error=function(error) list(error=conditionMessage(error))),
    warning=function(warning) {
      warnings <<- c(warnings,conditionMessage(warning));invokeRestart("muffleWarning")
    })
  # First-call id must be a name, even when direct update vectors are requested.
  cases[[length(cases)+1L]] <<- c(list(name=name,data1=lapply(data1,encode),
    data2=lapply(data2,encode),operations=operations,id=id,tstart=tstart,tstop=tstop,
    options=options,update=update,same=same,direct=direct,warnings=I(warnings)),result)
}
event_operations <- list(death=list(kind="event",time="time",value="status"))
add("implicit_range",base,base,event_operations,same=TRUE)
add("explicit_range",base,base,event_operations,tstart=1,tstop="time",same=TRUE)
add("renamed_intervals",base,base,event_operations,tstart=2,tstop="time",same=TRUE,
  options=list(tstartname="enter",tstopname="exit",idname="subject"))
factor_base <- transform(base,status=factor(c("a","wait","b"),levels=c("wait","unused","b","a")))
add("factor_censor",factor_base,factor_base,event_operations,same=TRUE)
spans <- data.frame(id=c(1L,1L,2L,2L,3L,3L),start=c(0,4,1,5,2,7),
  time=c(4,10,5,8,7,12),status=c(0L,0L,0L,1L,0L,1L))
add("repeated_initial_ranges",base,spans,event_operations,tstart="start",tstop="time")
operations <- list(lab=list(kind="tdc",time="time",value="lab"),
  visits=list(kind="cumtdc",time="time"),infection=list(kind="event",time="time",value="event"),
  count=list(kind="cumevent",time="time",value="event"))
for (na_rm in c(FALSE,TRUE)) for (delay in c(0,1)) for (direct in c(FALSE,TRUE)) {
  add(paste("all_updates",na_rm,delay,direct,sep="/"),initial,long,operations,
    options=list(na.rm=na_rm,delay=delay),update=TRUE,direct=direct)
}
add("shared_update_values",initial,long,list(
  a=list(kind="tdc",time="time",value="lab",init=0),
  b=list(kind="tdc",time="time",value="lab",init=-1)),update=TRUE,direct=TRUE)
add("factor_event",initial,long,list(state=list(kind="event",time="time",value="state")),update=TRUE)
add("factor_tdc",initial,long,list(state=list(kind="tdc",time="time",value="state")),update=TRUE)
add("cumulative_values",initial,long,list(total=list(kind="cumtdc",time="time",value="lab",init=2)),update=TRUE)
add("explicit_censor",initial,transform(long,event=event>0),
  list(event2=list(kind="event",time="time",value="event")),update=TRUE)
reference <- list(metadata=list(generator="scripts/generate_tmerge_iterable_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
  initial=snapshot(initial),cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases),"complete tmerge cases written to",output,"\n")
