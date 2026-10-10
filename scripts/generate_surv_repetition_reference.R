#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/surv_repetition_reference.json"
states <- factor(c("censor", "b", "a"), levels=c("censor", "b", "a", "unused"))
responses <- list(
  right=Surv(c(1,3,5), c(0,1,0)),
  missing_right=Surv(c(1,NA_real_,NaN), c(NA,1,0)),
  left=Surv(c(1,3,5), c(0,1,0), type="left"),
  counting=Surv(c(0,1,3), c(1,3,5), c(0,1,0)),
  interval=Surv(c(1,2,3), c(2,4,4), c(0,2,3), type="interval"),
  mright=Surv(c(1,3,5), states),
  mcounting=Surv(c(0,1,3), c(1,3,5), states))
operations <- list(
  default=list(), twice=list(times=2),
  each=list(each=3), each_times=list(each=3,times=2),
  counts=list(times=c(0,2,1)), each_counts=list(each=2,times=c(0,1,2,0,3,1)),
  length_prefix=list(length.out=2), length_cycle=list(length.out=8),
  each_length_prefix=list(each=3,length.out=5),
  each_length_cycle=list(each=3,length.out=20),
  length_precedence=list(times="bad",length.out=4),
  fractional=list(times=2.8,each=1.8,length.out=6.8),
  negative_fraction_times=list(times=-.5),
  negative_fraction_each=list(each=-.5),
  negative_fraction_length=list(length.out=-.5),
  negative_fraction_counts=list(times=c(-.5,1,1)),
  each_na=list(each=NA_real_), each_nan=list(each=NaN),
  each_inf=list(each=Inf), each_negative_inf=list(each=-Inf),
  length_na=list(length.out=NA_real_), length_nan=list(length.out=NaN),
  length_inf=list(length.out=Inf), length_negative_inf=list(length.out=-Inf),
  each_vector=list(each=c(2,1)), each_empty=list(each=numeric(0)),
  each_missing_vector=list(each=c(NA,2)),
  length_vector=list(length.out=c(2,9)), length_empty=list(length.out=numeric(0)),
  length_missing_vector=list(length.out=c(NA,9)),
  both_vectors=list(each=c(2,1),length.out=c(5,2)),
  times_character=list(times="2"), each_character=list(each="2"),
  length_character=list(length.out="2"), each_invalid_character=list(each="bad"),
  length_invalid_character=list(length.out="bad"),
  both_invalid_character=list(each="bad",length.out="bad"),
  zero_times=list(times=0), zero_each=list(each=0),
  zero_each_length_zero=list(each=0,length.out=0),
  zero_each_length_one=list(each=0,length.out=1),
  zero_each_counts=list(each=0,times=numeric(0)),
  negative_times=list(times=-1), negative_each=list(each=-1),
  negative_length=list(length.out=-1), missing_times=list(times=NA_real_),
  infinite_times=list(times=Inf), empty_times=list(times=numeric(0)),
  invalid_character_times=list(times="bad"), negative_character_times=list(times="-2"),
  short_counts=list(times=c(1,2)),
  negative_length_vector=list(each=c(-2,1),length.out=c(-1,2)),
  negative_each_vector=list(each=c(-2,1),length.out=c(0,2)),
  zero_each_vector=list(each=c(0,1),length.out=c(1,2)),
  ignored_long_each=list(each=1e12,length.out=1),
  ignored_long_each_zero=list(each=1e12,times=0))
encode <- function(value) {
  if (is.list(value)) return(lapply(value, encode))
  if (is.numeric(value) && length(value)==1L && !is.na(value) && !is.finite(value))
    return(list(number=if (value>0) "Inf" else "-Inf"))
  if (length(value) != 1L) return(I(value))
  value
}
snapshot <- function(x) list(type=attr(x,"type"),states=attr(x,"states"),
  clabel=attr(x,"clabel"),matrix=unname(as.matrix(x)))
cases <- lapply(names(responses), function(name) {
  x <- responses[[name]]
  results <- lapply(names(operations), function(operation) {
    warnings <- character()
    result <- withCallingHandlers(
      tryCatch(list(response=snapshot(do.call(rep,c(list(x),operations[[operation]])))),
        error=function(error) list(error=conditionMessage(error))),
      warning=function(warning) {
        warnings <<- c(warnings,conditionMessage(warning))
        invokeRestart("muffleWarning")
      })
    c(list(name=operation,args=encode(operations[[operation]]),warnings=I(warnings)),result)
  })
  list(name=name,response=snapshot(x),results=results)
})
reference <- list(metadata=list(generator="scripts/generate_surv_repetition_reference.R",
  r_version=as.character(getRversion()),survival_version=as.character(packageVersion("survival"))),
  cases=cases)
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null")
cat(length(cases)*length(operations),"Surv repetition calls written to",output,"\n")
