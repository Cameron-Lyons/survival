#!/usr/bin/env Rscript
# Complete calls: model-frame preparation, baseline construction and cohort output.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[1L]) else 5000L
repeats <- if (length(args) > 1L) as.integer(args[2L]) else 7L
stopifnot(n > 0L, repeats > 0L)
set.seed(318)
train <- data.frame(time=sample(1:1000,2000,TRUE),status=rbinom(2000,1,.7),x=rnorm(2000))
fit <- survival::coxph(survival::Surv(time,status)~x,train,model=TRUE)
d <- data.frame(time=runif(n,50,1000),x=rnorm(n),group=factor(seq_len(n)%%4),weight=runif(n,.5,2))
results <- list()
for (method in c("ederer","hakulinen","conditional")) {
  calls <- list(R_survival=function() survival::survexp(time~group,d,ratetable=fit,weights=weight,method=method),
    R_Rust=function() survivalr::survexp(time~group,d,ratetable=fit,weights=weight,method=method))
  expected <- calls$R_survival(); actual <- calls$R_Rust()
  fields <- c("time","surv","n.risk")
  stopifnot(isTRUE(all.equal(unclass(actual)[fields],unclass(expected)[fields],tolerance=1e-12)))
  rm(actual,expected)
  for(warmup in seq_len(2L)) for(call in calls) invisible(call())
  elapsed <- lapply(calls,function(call) numeric(repeats))
  for(sample in seq_len(repeats)) {
    order <- if(sample%%2L) names(calls) else rev(names(calls))
    for(name in order) {
      gc(FALSE)
      started <- proc.time()[["elapsed"]]
      value <- calls[[name]]()
      elapsed[[name]][sample] <- 1000*(proc.time()[["elapsed"]]-started)
      rm(value)
    }
  }
  results[[method]] <- lapply(elapsed,function(value) list(median_ms=median(value),samples_ms=I(value)))
}
cat(jsonlite::toJSON(list(r_version=R.version.string,survival_version=as.character(packageVersion("survival")),
  observations=n,training_rows=2000,repeats=repeats,results=results),auto_unbox=TRUE,pretty=TRUE,digits=NA),"\n")
