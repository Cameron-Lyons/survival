#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
  "python/tests/fixtures/model_matrix_iterator_reference.json"

training <- kidney[seq_len(24), c("time", "status", "age", "id")]
training$group <- 100 + 3 * training$id
categorical <- training
categorical$group <- factor(paste0("g", training$id %% 3),
  levels = c("g2", "g0", "g1", "unused"))

columns <- function(data) lapply(data, function(value) list(
  values = I(if (is.factor(value)) as.character(value) else unname(value)),
  levels = if (is.factor(value)) I(levels(value)) else NULL,
  ordered = is.ordered(value)))
contrast_value <- function(value) {
  if (is.matrix(value)) list(data = unname(value), rows = I(rownames(value)),
    columns = I(colnames(value))) else value
}
matrix_value <- function(value) list(
  data = unname(value), columns = I(colnames(value)),
  assign = I(attr(value, "assign")), row_names = I(rownames(value)),
  strata = if (is.null(attr(value, "strata"))) NULL else
    I(as.character(attr(value, "strata"))),
  contrasts = if (is.null(attr(value, "contrasts"))) NULL else
    lapply(attr(value, "contrasts"), contrast_value))
capture_matrix <- function(fun) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers(matrix_value(fun()),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }), error = function(e) list(error = conditionMessage(e)))
  list(value = result, warnings = I(warnings))
}

specs <- list(
  numeric_sparse = list(source = "numeric", sparse = "TRUE"),
  numeric_dense = list(source = "numeric", sparse = "FALSE"),
  numeric_auto = list(source = "numeric", sparse = NULL),
  categorical_sparse = list(source = "categorical", sparse = "TRUE"),
  categorical_dense = list(source = "categorical", sparse = "FALSE"))
fits <- list()
cases <- list()
for (name in names(specs)) {
  spec <- specs[[name]]
  data <- if (spec$source == "numeric") training else categorical
  formula <- as.formula(paste0("Surv(time,status) ~ age + frailty(group,",
    if (is.null(spec$sparse)) "" else paste0("sparse=", spec$sparse, ","),
    "theta=.4)"))
  fit <- coxph(formula, data, x = TRUE, model = TRUE)
  fits[[name]] <- list(source = spec$source,
    formula = paste(deparse(formula, width.cutoff = 500L), collapse = " "),
    stored = capture_matrix(function() model.matrix(fit)))
  new <- list(age = c(40, 50, 60, 35), group = c(112, 103, 112, 106))
  if (spec$source == "categorical") new$group <- factor(c("g2", "g0", "g2", "g1"),
    levels = c("g1", "g2", "g0", "unused"))
  missing <- new
  missing$age[2L] <- NA_real_
  missing$group[3L] <- NA
  alias <- new
  alias$age <- alias$group
  one <- new
  one$group[] <- new$group[1L]
  variants <- list(complete = new, missing = missing)
  if (spec$source == "numeric") variants$alias <- alias else variants$one_group <- one
  if (name %in% c("numeric_sparse", "categorical_dense")) {
    named <- as.data.frame(missing)
    rownames(named) <- c("patient9", "patient2", "patient9.1", "patient7")
    variants$named_missing <- named
  }
  for (variant in names(variants)) {
    newdata <- variants[[variant]]
    cases[[length(cases) + 1L]] <- list(name = paste(name, variant, sep = "/"),
      fit = name, newdata = columns(newdata),
      row_names = if (is.data.frame(newdata)) I(rownames(newdata)) else NULL,
      shared_age_group = variant == "alias",
      expected = capture_matrix(function() model.matrix(fit, newdata)))
  }
}
write_json(list(provenance = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")),
  reference = "Direct stock model.matrix.coxph, including full row labels and fitted contrast metadata; warnings retained."),
  inputs = list(numeric = columns(training), categorical = columns(categorical)),
  fits = fits, cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null")
cat(length(cases), "stock Cox matrix iterator controls written\n")
