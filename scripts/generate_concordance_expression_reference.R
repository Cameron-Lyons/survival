#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/concordance_expression_reference.json"

base <- data.frame(x = c(-2, -1, 1, 2, 3), z = c(3, 1, 2, 0, 4))
positive <- data.frame(x = c(1, 1, 2, 3, 3, 4), z = c(6, 1, 5, 3, 2, 4))
cases <- list(
  list(name = "logical_explicit", formula = "I(x > 0) ~ x", data = base),
  list(name = "logical_dot", formula = "I(x > 0) ~ .", data = base),
  list(name = "logical_dot_explicit", formula = "I(x > 0) ~ . + x", data = base),
  list(name = "logical_dot_interaction", formula = "I(x > 0) ~ .:x", data = base),
  list(name = "logical_dot_power", formula = "I(x > 0) ~ (. + x)^2", data = base),
  list(name = "factor_expression", formula = "factor(x > 0) ~ x", data = base),
  list(name = "factor_source", formula = "factor(x) ~ x + z",
       data = data.frame(x = c(1, 1, 2, 2, 2), z = base$z)),
  list(name = "shared_weights_cluster", formula = "I(x > 2) ~ z", data = positive,
       weights = "x", cluster = "x"),
  list(name = "formula_cluster", formula = "I(x > 2) ~ z + cluster(x)", data = positive,
       weights = "x"),
  list(name = "subset_missing", formula = "I(x > 2) ~ . + x",
       data = data.frame(x = c(1, NA, 2, 3, 4, 5), z = c(4, 1, 3, NA, 2, 0)),
       weights = "x", cluster = "x", subset = c(1, 3, 4, 5, 6), na_action = "na.exclude"),
  list(name = "quoted_source", formula = "factor(`response source`) ~ `response source` + .",
       data = data.frame("response source" = c("b", "a", "b", "a", "b"),
                         z = base$z, check.names = FALSE)),
  list(name = "internal_name_collision", formula = "I(x > 0) ~ .",
       data = data.frame(x = base$x, "(response)" = 5:1, check.names = FALSE)),
  list(name = "warning_once", formula = "I(log(x) > 0) ~ z",
       data = data.frame(x = c(-1, 1, 2, 3, 4), z = c(2, 4, 1, 5, 3)))
)

records <- lapply(cases, function(case) {
  call <- list(object = as.formula(case$formula), data = case$data, influence = 1, ranks = TRUE)
  for (key in c("weights", "cluster")) {
    if (!is.null(case[[key]])) call[[key]] <- case$data[[case[[key]]]]
  }
  if (!is.null(case$subset)) call$subset <- case$subset
  if (!is.null(case$na_action)) call$na.action <- get(case$na_action)
  warnings <- character()
  fit <- withCallingHandlers(do.call(concordance, call), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  p <- length(fit$concordance)
  counts <- if (is.matrix(fit$count)) unname(fit$count) else matrix(unname(fit$count), nrow = 1)
  ranks <- lapply(seq_len(p), function(j) {
    columns <- fit$ranks[, 4 * (j - 1) + seq_len(4), drop = FALSE]
    names(columns) <- c("time", "rank", "timewt", "casewt")
    lapply(columns, function(x) I(unname(x)))
  })
  list(name = case$name, formula = case$formula,
       data = lapply(case$data, function(x) I(unname(x))),
       weights = case$weights, cluster = case$cluster,
       subset = if (is.null(case$subset)) NULL else I(case$subset - 1L),
       na_action = case$na_action,
       expected = list(concordance = I(unname(fit$concordance)), count = counts,
                       names = if (p == 1) NULL else I(names(fit$concordance)),
                       n = fit$n, var = matrix(fit$var, p, p), cvar = I(unname(fit$cvar)),
                       dfbeta = matrix(fit$dfbeta, ncol = p), ranks = ranks,
                       omitted = I(as.integer(fit$na.action))), warnings = I(warnings))
})
cat(R.version.string, "; survival", as.character(packageVersion("survival")), "\n")
write_json(list(metadata = list(survival = as.character(packageVersion("survival"))),
                cases = records), output, auto_unbox = TRUE, digits = NA,
           na = "null", null = "null", pretty = TRUE)
