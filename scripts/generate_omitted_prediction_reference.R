#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
source("r/survivalr/tests/testthat/helper-omitted-predictions.R")
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/omitted_prediction_reference.json"
setup <- .omitted_prediction_setup()
encode <- function(value) {
  if (is.list(value)) return(lapply(value, encode))
  list(values = I(as.numeric(value)), nan = I(as.logical(is.nan(value))),
       dim = if (is.null(dim(value))) NULL else I(dim(value)),
       has_dimnames = !is.null(dimnames(value)),
       names = if (is.null(names(value))) NULL else I(names(value)),
       rows = if (is.null(rownames(value))) NULL else I(rownames(value)),
       columns = if (is.null(colnames(value))) NULL else I(colnames(value)))
}
cases <- list()
for (name in names(setup$specs)) {
  fit <- .omitted_prediction_fit(name, setup, "survival")
  for (case in .omitted_prediction_cases(name)) {
    result <- .omitted_prediction_expected(fit, setup, case)
    case$raw_error <- result$raw_error
    if (!is.null(case$p)) case$p <- I(case$p)
    case$result <- if (is.list(result$value) && !is.null(result$value$error)) result$value else encode(result$value)
    case$warnings <- I(result$warnings)
    cases[[length(cases) + 1L]] <- case
  }
}
raw_na <- writeBin(NA_real_, raw(), endian = "little")
payload <- readBin(raw_na[1:4], "integer", n = 1L, endian = "little")
serialize <- function(value) jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17, na = "null", null = "null")
header <- serialize(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")), na_payload = payload),
  data = lapply(setup$data, function(x) if(is.factor(x)) as.character(x) else x),
  row_names = row.names(setup$data), new_names = setup$names,
  levels = levels(setup$data$cl), specs = setup$specs))
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
  vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]), if(i < length(cases)) "," else ""), ""),
  "]}"), output, useBytes = TRUE)
cat(length(cases), "stock omitted/missing prediction references written\n")
