#!/usr/bin/env Rscript
# Independent stock kernels: masks in Python correspond to these R NA cells.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/masked_matrix_formula_reference.json"
stock_pyears <- getFromNamespace("pyears", "survival")
ns <- asNamespace("survival")
numbers <- function(x) lapply(unname(x), function(value) if (is.na(value)) NULL else value)
rows <- function(x) lapply(seq_len(nrow(x)), function(row) numbers(x[row, ]))
na_record <- function(value) {
  if (is.null(value)) return(NULL)
  list(rows = I(as.integer(value)), kind = if (inherits(value, "exclude")) "exclude" else "omit")
}
frame_snapshot <- function(frame) list(
  columns = lapply(frame, function(column) if (is.matrix(column)) rows(column) else numbers(column)),
  row_names = I(row.names(frame)), na_action = na_record(attr(frame, "na.action")))
capture <- function(call, snapshot) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(snapshot(call()), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = I(warnings))
}
base <- list(Y = matrix(c(2,1,5,0,3,2,9,0,4,.5,6,1), ncol = 2L, byrow = TRUE),
             group = c("b","a","b","a","c","b"), wt = c(1,.5,2,1,1.5,.5))
inputs <- list(complete = base, missing_event = base, missing_time = base)
inputs$missing_event$Y[2L,2L] <- NA_real_
inputs$missing_time$Y[4L,1L] <- NA_real_
selections <- list(all = NULL, reordered = c(6L,2L,1L,6L,4L))
actions <- list(omit = na.omit, exclude = na.exclude, fail = na.fail, pass = na.pass)
cases <- list()
for (input in names(inputs)) for (selection in names(selections))
  for (action in names(actions)) for (rhs in c("1", "group")) {
    data <- inputs[[input]]
    formula <- as.formula(paste("Y ~", rhs), env = ns)
    arguments <- list(formula = formula, data = data, weights = quote(wt),
                      na.action = actions[[action]])
    if (!is.null(selections[[selection]])) arguments$subset <- selections[[selection]]
    expected_frame <- capture(function() do.call(stats::model.frame, arguments), frame_snapshot)
    expected_pyears <- capture(function() do.call(stock_pyears,
      c(arguments, list(scale = 1, x = rhs != "1", y = TRUE))), function(fit) {
      stopifnot(inherits(fit, "pyears"), !inherits(fit, "survival_py_pyears"))
      list(pyears = numbers(fit$pyears), event = numbers(fit$event), n = numbers(fit$n),
           offtable = fit$offtable, observations = fit$observations,
           dim = I(as.integer(dim(fit$pyears))), dimnames = lapply(dimnames(fit$pyears), I),
           y = rows(fit$y), x = if (is.null(fit$x)) NULL else
             if (is.matrix(fit$x)) rows(fit$x) else numbers(fit$x),
           model = frame_snapshot(do.call(stock_pyears,
             c(arguments, list(scale = 1, model = TRUE)))$model),
           na_action = na_record(fit$na.action))
    })
    cases[[length(cases)+1L]] <- list(name = paste(input,selection,action,rhs,sep="/"),
      input = input, subset = if (is.null(selections[[selection]])) NULL else I(selections[[selection]]-1L),
      formula = paste("Y ~",rhs), na_action = action,
      frame = expected_frame, pyears = expected_pyears)
  }
# Scalar masked values use the same stock NA vector, independently of matrices.
vector_cases <- list()
for (input in c("complete","missing")) for (action in names(actions)) {
  data <- list(Y = c(2,5,3), wt = c(1,.5,2))
  if (input == "missing") data$Y[2L] <- NA_real_
  formula <- as.formula("Y ~ 1", env = ns)
  arguments <- list(formula=formula,data=data,weights=quote(wt),na.action=actions[[action]])
  vector_cases[[length(vector_cases)+1L]] <- list(name=paste(input,action,sep="/"),
    input=input, na_action=action,
    frame=capture(function()do.call(stats::model.frame,arguments),frame_snapshot))
}
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion("survival")),
  inputs=lapply(inputs,function(data)list(Y=rows(data$Y),group=numbers(data$group),wt=numbers(data$wt))),
  cases=cases,vector_cases=vector_cases),output,pretty=TRUE,auto_unbox=TRUE,digits=NA,
  null="null",na="null")
