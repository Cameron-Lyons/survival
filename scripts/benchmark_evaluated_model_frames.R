#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 20000L
repeats <- if (length(args) > 1L) as.integer(args[[2L]]) else 9L
mode <- if (length(args) > 2L) args[[3L]] else "current"
suppressPackageStartupMessages(library(survival))
set.seed(741)
d <- data.frame(time = rexp(n) + .1, status = rbinom(n, 1, .7),
  flag = seq_len(n) %% 3L == 0L, g = factor(seq_len(n) %% 5L), o = runif(n, 1, 3))
for (j in seq_len(16)) d[[paste0("x", j)]] <- runif(n, 1, 3)
formulas <- c(scalar = "x1", transformed = paste(
  paste(c(paste0("log(x", 1:8, ")"), paste0("sqrt(x", 9:16, ")")), collapse = "+"),
  "+flag+strata(g)+offset(log(o))"), ridge = "ridge(x1, x2, theta = 2)")
capture <- function(fun) {
  messages <- character()
  value <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = messages)
}
check <- function(actual, expected) {
  stopifnot(identical(actual$warnings, expected$warnings))
  a <- actual$value; e <- expected$value
  stopifnot(is.matrix(a), identical(dim(a), dim(e)), identical(dimnames(a), dimnames(e)),
    identical(as.integer(attr(a, "assign")), as.integer(attr(e, "assign"))),
    identical(attr(a, "contrasts"), attr(e, "contrasts")),
    identical(attr(a, "strata"), attr(e, "strata")),
    identical(is.na(a), is.na(e)), identical(is.nan(a), is.nan(e)),
    isTRUE(all.equal(as.numeric(a), as.numeric(e))))
}
measure <- function(fun) {
  for (i in seq_len(3)) invisible(capture(fun))
  samples <- vapply(seq_len(repeats), function(i) {
    gc(FALSE)
    start <- proc.time()[["elapsed"]]
    invisible(capture(fun))
    1000 * (proc.time()[["elapsed"]] - start)
  }, numeric(1))
  list(median_ms = median(samples), range_ms = range(samples), samples_ms = I(samples))
}
matrix_call <- function(object, input) {
  if (is.null(input)) model.matrix(object) else model.matrix(object, input)
}
references <- list()
for (kind in c("coxph", "survreg")) for (name in names(formulas)) {
  options(na.action = "na.omit")
  formula <- as.formula(paste("Surv(time,status)~", formulas[[name]]))
  fit <- get(kind, asNamespace("survival"))(formula, d, x = TRUE, model = TRUE)
  raw <- d[seq_len(n %/% 2L), ]
  prepared <- model.frame(fit)[seq_len(n %/% 2L), , drop = FALSE]
  missing <- prepared
  variable <- names(missing)[2L]
  indices <- seq(1L, nrow(missing), 59L)
  if (is.matrix(missing[[variable]])) missing[[variable]][indices, 1L] <- NA_real_ else
    missing[[variable]][indices] <- NA_real_
  inputs <- list(stored = NULL, raw = raw, evaluated = prepared, evaluated_missing = missing)
  expected <- lapply(inputs, function(input) capture(function() matrix_call(fit, input)))
  references[[paste(kind, name, sep = "/")]] <- list(kind = kind, name = name,
    formula = formula, fit = fit, inputs = inputs, expected = expected)
}
if (mode != "stock") suppressPackageStartupMessages(library(survivalr))
results <- list()
for (key in names(references)) {
  reference <- references[[key]]
  options(na.action = "na.omit")
  fit <- if (mode == "stock") reference$fit else get(reference$kind, asNamespace("survivalr"))(
    reference$formula, d, x = TRUE, model = TRUE)
  results[[key]] <- lapply(setNames(names(reference$inputs), names(reference$inputs)), function(workload) {
    input <- reference$inputs[[workload]]
    call <- function() matrix_call(fit, input)
    actual <- capture(call)
    expected <- reference$expected[[workload]]
    if (mode == "predecessor" && startsWith(workload, "evaluated") && reference$name != "scalar") {
      stopifnot(grepl("not found", actual$value$error), is.matrix(expected$value))
      return(list(incompatible = "Requires absent raw source columns; not timed"))
    }
    if (mode == "predecessor" && workload == "evaluated_missing") {
      stopifnot(reference$name == "scalar", nrow(actual$value) < nrow(expected$value),
        identical(actual$warnings, expected$warnings))
      return(list(incompatible = "Drops rows retained by stock evaluated frames; not timed",
        actual_rows = nrow(actual$value), expected_rows = nrow(expected$value)))
    }
    tryCatch(check(actual, expected), error = function(error)
      stop(paste(key, workload, conditionMessage(error),
        "actual dimensions", paste(dim(actual$value), collapse = "x"),
        "stock dimensions", paste(dim(expected$value), collapse = "x")), call. = FALSE))
    measure(call)
  })
}
cat(jsonlite::toJSON(list(rows = n, new_rows = n %/% 2L, repeats = repeats, warmups = 3L,
  mode = mode, r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  scope = "Complete public matrix calls, including conversion, assembly, metadata and warning/error capture; fitting, input setup, option changes and explicit GC excluded. All timed outputs checked against stock first; incorrect predecessor calls verified but not timed.",
  results = results), auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
