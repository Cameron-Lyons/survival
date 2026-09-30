#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
  "python/tests/fixtures/prediction_term_selection_reference.json"
d <- ovarian
d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
nd <- d[c(3L, 4L, 5L, 6L, 7L, 8L), ]
nd$age[2L] <- NA
nd$rx[4L] <- NA
d$age[c(2L, 9L)] <- NA
specs <- list(
  aft_plain = c("survreg", "age + rx"),
  aft_strata_first = c("survreg", "strata(cl) + age + rx"),
  aft_strata_last = c("survreg", "age + rx + strata(cl)"),
  aft_ridge = c("survreg", "age + ridge(rx, theta = 2)"),
  aft_spline = c("survreg", "pspline(age, df = 3) + rx"),
  cox_plain = c("coxph", "age + rx"),
  cox_strata_first = c("coxph", "strata(cl) + age + rx"),
  cox_strata_last = c("coxph", "age + rx + strata(cl)"),
  cox_ridge = c("coxph", "age + ridge(rx, theta = 2)")
)
selections <- list(
  default = NULL, null = NULL, empty_integer = integer(), empty_logical = logical(),
  empty_character = character(), first = 1, repeated = c(2,1,2), zero = 0,
  zero_and_positive = c(0,2,0,1), negative = -1, repeated_negative = c(-2,0,-2),
  outside_negative = -999, fractional = c(1.9,2.7), negative_fractional = c(-1.9,0),
  true = TRUE, false = FALSE, logical = c(TRUE,FALSE), recycled = TRUE,
  numeric_missing = c(1,NA_real_), logical_missing = c(TRUE,NA),
  missing_numeric_only = NA_real_, missing_logical_only = NA,
  missing_numeric_and_zero = c(NA_real_,0), missing_numeric_invalid = c(NA_real_,999),
  mixed_sign = c(1,-2), negative_missing = c(-1,NA_real_),
  long_logical = c(TRUE,FALSE,FALSE), outside_positive = 999,
  character_missing = c("age",NA_character_), unknown_name = "absent",
  infinite = c(1,Inf), integer_overflow = 2147483648
)
matrix_record <- function(value) {
  list(values = unname(lapply(seq_len(nrow(value)), function(i) as.list(unname(value[i,])))),
       dim = as.list(dim(value)), columns = as.list(colnames(value)),
       constant = attr(value, "constant"))
}
capture <- function(fun) {
  messages <- character()
  result <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  if (is.matrix(result)) result <- matrix_record(result)
  else if (is.list(result) && is.null(result$error)) {
    result <- list(fit = matrix_record(result$fit), se_fit = matrix_record(result$se.fit))
  }
  list(result = result, warnings = as.list(messages))
}
cases <- list()
for (name in names(specs)) {
  spec <- specs[[name]]
  fit <- do.call(get(spec[[1L]]), list(as.formula(paste("Surv(futime,fustat)~", spec[[2L]])),
                                     data = d, x = TRUE, na.action = na.exclude))
  labels <- colnames(predict(fit, type = "terms"))
  intended <- labels
  if (spec[[1L]] == "survreg") {
    tt <- fit$terms
    specials <- attr(tt, "specials")$strata
    if (length(specials)) tt <- survival:::drop.special(tt, specials)
    mm <- model.matrix(fit)
    groups <- survival::attrassign(mm, tt)
    groups$"(Intercept)" <- NULL
    intended <- names(groups)
    stopifnot(length(intended) == length(labels))
  }
  choices <- c(selections, list(named = intended[c(2L,1L)],
                                factor = factor(intended[c(2L,1L)], levels = rev(intended)),
                                factor_unused = factor(intended[1L], levels = c("unused", intended))))
  for (choice in names(choices)) for (se in c(FALSE,TRUE)) {
    selection <- choices[[choice]]
    options <- list(object = fit, newdata = nd, type = "terms", se.fit = se)
    if (choice != "default") options["terms"] <- list(selection)
    raw <- capture(function() do.call(predict, options))
    record <- raw
    correction <- NULL
    if (!identical(labels, intended)) {
      record <- capture(function() {
        all <- predict(fit, newdata = nd, type = "terms", se.fit = se)
        select <- function(value) {
          colnames(value) <- intended
          if (choice == "default" || is.null(selection)) value else value[, selection, drop = FALSE]
        }
        if (se) list(fit = select(all$fit), se.fit = select(all$se.fit)) else select(all)
      })
      correction <- "Stock AFT attrassign uses the full terms labels after removing strata columns; use groups from the fitted model matrix and strata-removed terms, then base matrix indexing."
    }
    cases[[length(cases) + 1L]] <- list(
      name = paste(name, choice, se, sep = "/"), model = name, selection = choice, se_fit = se,
      kind = if (is.factor(selection)) "factor" else typeof(selection),
      terms = if (is.factor(selection)) as.list(as.character(selection)) else lapply(selection, function(value) {
        if (is.numeric(value) && is.infinite(value)) as.character(value) else value
      }),
      levels = if (is.factor(selection)) as.list(levels(selection)) else NULL,
      expected = record, raw = raw, correction = correction
    )
  }
  # AFT ignores terms for other types; Cox validates them before prediction.
  for (type in if (spec[[1L]] == "survreg") c("lp", "quantile") else c("lp", "risk", "expected", "survival")) {
    options <- list(object = fit, newdata = nd, type = type, terms = -1, se.fit = TRUE)
    raw <- capture(function() {
      result <- do.call(predict, options)
      result$fit <- as.matrix(result$fit); result$se.fit <- as.matrix(result$se.fit)
      result
    })
    record <- raw
    correction <- NULL
    if (type == "survival" && inherits(fit, "coxph.penal")) {
      plain <- fit; class(plain) <- "coxph"; options$object <- plain
      record <- capture(function() do.call(predict, options))
      correction <- "The penalized wrapper refuses survival before validating terms; the ordinary Cox method validates the same invalid selector."
    }
    cases[[length(cases) + 1L]] <- list(name = paste(name,type,"ignored",sep="/"), model = name,
      selection = "negative", se_fit = TRUE, type = type, kind = "double", terms = list(-1),
      levels = NULL, expected = record, raw = raw, correction = correction)
  }
}
payload <- list(r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  data = lapply(d, function(value) as.list(if (is.factor(value)) as.character(value) else value)),
  newdata = lapply(nd, function(value) as.list(if (is.factor(value)) as.character(value) else value)),
  levels = as.list(levels(d$cl)), specs = specs, cases = cases)
jsonlite::write_json(payload, output, pretty = TRUE, auto_unbox = TRUE, digits = NA, null = "null", na = "null")
cat(length(cases), "independent prediction calls written to", output, "\n")
