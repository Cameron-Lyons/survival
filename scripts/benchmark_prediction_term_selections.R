#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "installed"
suppressPackageStartupMessages(library(survival))
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
set.seed(719)
data <- data.frame(time = rexp(2000), status = rbinom(2000, 1, .7))
newdata <- data.frame(row = seq_len(n))
for (j in seq_len(16)) {
  data[[paste0("x",j)]] <- rnorm(nrow(data))
  newdata[[paste0("x",j)]] <- rnorm(n)
}
formula <- as.formula(paste("Surv(time,status)~", paste(paste0("x",seq_len(16)),collapse="+")))
measure <- function(fun) {
  for (i in seq_len(3)) invisible(fun())
  samples <- vapply(seq_len(repeats), function(i) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(fun())
    1000 * (proc.time()[["elapsed"]] - start)
  }, numeric(1))
  list(median_ms = median(samples), range_ms = range(samples), samples_ms = I(samples))
}
results <- lapply(c("coxph","survreg"), function(kind) {
  reference <- do.call(get(kind,envir=asNamespace("survival")),list(formula,data))
  fit <- if (mode == "stock") reference else do.call(get(kind,envir=asNamespace("survivalr")),list(formula,data))
  call <- function(object) predict(object, newdata, type="terms", terms=c("x9","x1","x9"), se.fit=TRUE)
  actual <- call(fit); expected <- call(reference)
  for (name in c("fit","se.fit")) {
    stopifnot(identical(dim(actual[[name]]),dim(expected[[name]])),
      identical(colnames(actual[[name]]),colnames(expected[[name]])),
      isTRUE(all.equal(unname(actual[[name]]),unname(expected[[name]]),
                       tolerance=2e-7,check.attributes=FALSE)))
  }
  measure(function() call(fit))
})
names(results) <- c("coxph","survreg")
cat(jsonlite::toJSON(list(rows=n,columns=16L,selected=3L,training_rows=2000L,repeats=repeats,warmups=3L,
  mode=mode,r=as.character(getRversion()),survival=as.character(packageVersion("survival")),
  scope="Complete public new-data term predictions with errors and a repeated named selection, including bridge conversion, formula design, selection, native prediction, materialization and column metadata; fitting, input creation and explicit GC excluded.",
  results=results),auto_unbox=TRUE,pretty=TRUE,digits=NA),"\n")
