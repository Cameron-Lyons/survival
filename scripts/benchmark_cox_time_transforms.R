#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 500L
repeats <- if (length(args)>1L) as.integer(args[[2L]]) else 7L
mode <- if (length(args)>2L) args[[3L]] else "installed"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(2031)
d <- data.frame(time=round(rexp(n)*30)+1,status=rbinom(n,1,.6),x=rnorm(n),z=rnorm(n),
                g=factor(rep(c("a","b"),length.out=n)),w=runif(n,.7,1.5))
functions <- list(vector=function(x,t,...) x*log(t),
                  matrix=function(x,t,...) cbind(log=x*log(t),root=x*sqrt(t)),
                  factor=function(x,t,...) factor(x>median(x)))
measure <- function(fun) {
  for (i in 1:3) invisible(fun())
  times <- numeric(repeats)
  for (i in seq_len(repeats)) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(fun())
    times[i] <- 1000*(proc.time()[["elapsed"]]-start)
  }
  list(median_ms=median(times),range_ms=range(times),samples_ms=I(times))
}
results <- list()
for (name in if (mode=="baseline") "vector" else names(functions)) {
  callback <- functions[[name]]
  stock <- function() survival::coxph(survival::Surv(time,status)~x+tt(x)+strata(g),
                                      d,tt=callback,weights=w,robust=FALSE,x=TRUE)
  call <- if (mode=="stock") stock else function() survivalr::coxph(
    Surv(time,status)~x+tt(x)+strata(g),d,tt=callback,weights=w,robust=FALSE,x=TRUE)
  a <- call(); b <- stock()
  stopifnot(isTRUE(all.equal(unname(coef(a)),unname(coef(b)),tolerance=1e-7)),
            isTRUE(all.equal(vcov(a),vcov(b),tolerance=1e-7)))
  results[[name]] <- measure(call)
}
cat(jsonlite::toJSON(list(rows=n,repeats=repeats,warmups=3L,mode=mode,
  r=as.character(getRversion()),survival=as.character(packageVersion("survival")),
  scope="Complete weighted stratified Cox formula calls including callback transport, risk-set expansion, fitting, and retained X; explicit GC excluded",
  results=results),auto_unbox=TRUE,pretty=TRUE,digits=NA),"\n")
