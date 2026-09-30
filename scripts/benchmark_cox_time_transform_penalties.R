#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 500L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 7L
mode <- if (length(args) > 2L) args[[3L]] else "installed"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(2032)
d <- data.frame(time = round(rexp(n)*30)+1, status = rbinom(n,1,.6), x = rnorm(n),
                group = rep(seq_len(8), length.out = n),
                stratum = factor(rep(c("a","b","c"), length.out = n)))
callbacks <- list(
  ridge = function(x,t,...) survival::ridge(cbind(x*log(t),x*sqrt(t)),theta = 2),
  spline = function(x,t,...) survival::pspline(x*log(t),theta = .4),
  sparse_frailty = function(x,t,...) survival::frailty(x,sparse = TRUE,theta = .4),
  dense_frailty = function(x,t,...) survival::frailty(x,sparse = FALSE,theta = .4))
measure <- function(fun) {
  for (i in 1:3) invisible(fun())
  times <- numeric(repeats)
  for (i in seq_len(repeats)) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(fun())
    times[i] <- 1000*(proc.time()[["elapsed"]]-start)
  }
  list(median_ms = median(times), range_ms = range(times), samples_ms = I(times))
}
results <- list()
for (name in if (mode == "baseline") c("ridge","spline") else names(callbacks)) {
  callback <- callbacks[[name]]
  formula <- if (grepl("frailty",name)) Surv(time,status) ~ x + tt(group) + strata(stratum)
             else Surv(time,status) ~ x + tt(x) + strata(stratum)
  stock <- function() survival::coxph(formula,d,tt = callback,robust = FALSE,x = TRUE)
  call <- if (mode == "stock") stock else function() survivalr::coxph(
    formula,d,tt = callback,robust = FALSE,x = TRUE)
  actual <- call(); expected <- stock()
  stopifnot(isTRUE(all.equal(unname(coef(actual)),unname(coef(expected)),tolerance = 1e-7)),
            isTRUE(all.equal(vcov(actual),vcov(expected),tolerance = 1e-7)),
            isTRUE(all.equal(as.numeric(actual$df),expected$df,tolerance = 1e-7)))
  results[[name]] <- measure(call)
}
cat(jsonlite::toJSON(list(rows = n, repeats = repeats, warmups = 3L, mode = mode,
  r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete stratified penalized Cox formula calls including callback transport, risk-set expansion, fitting, and retained X; explicit GC excluded",
  results = results),auto_unbox = TRUE,pretty = TRUE,digits = NA),"\n")
