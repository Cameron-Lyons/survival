#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/grouped_cox_prediction_reference.json"
d <- ovarian
d$x <- (d$age - 60)/10
d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
specs <- list(plain = "x + rx", ridge = "x + ridge(rx, theta = 2)",
              multi = "ridge(x, rx, theta = 2)",
              sparse = "x + frailty(cl, sparse = TRUE, theta = .4)",
              sparse_only = "frailty(cl, sparse = TRUE, theta = .4)")
capture <- function(fun) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(result = result, warnings = I(warnings))
}
encode <- function(value) {
  if (is.list(value) && !is.null(value$error)) return(value)
  if (is.list(value)) return(list(fit = encode(value$fit), se_fit = encode(value$se.fit)))
  list(values = if (is.matrix(value)) unname(value) else I(unname(value)),
       names = I(if (is.matrix(value)) rownames(value) else names(value)),
       matrix = is.matrix(value), width = if (is.matrix(value)) ncol(value) else NULL)
}
groups <- list(
  numeric = replace(rep(c(-11, 99), length.out = nrow(d)), c(2, 9), 5),
  factor = factor(replace(rep(c("kept-a", "kept-c"), length.out = nrow(d)), c(2, 9), "only-omitted"),
                  levels = c("kept-c", "only-omitted", "kept-a", "unused"))
)
cases <- list()
for (name in names(specs)) {
  data <- d
  if (name == "sparse_only") data$cl[c(2, 9)] <- NA else data$x[c(2, 9)] <- NA
  fit <- coxph(as.formula(paste("Surv(futime,fustat) ~", specs[[name]])), data,
               x = TRUE, robust = FALSE, na.action = na.exclude)
  types <- if (name == "sparse_only") c("lp", "risk", "terms") else c("lp", "risk", "terms", "expected", "survival")
  dense <- fit
  if (name == "sparse") {
    dense$x <- fit$x[, "x", drop = FALSE]
    dense$terms <- terms(Surv(futime,fustat) ~ x)
    dense$assign <- list(x = 1L)
    dense$pterms <- c(x = 0)
    class(dense) <- "coxph"
  }
  for (group_name in names(groups)) for (type in types) for (se in c(FALSE, TRUE)) {
    selections <- if (type == "terms" && !name %in% c("sparse", "sparse_only"))
      list(all = NULL, repeated = if (name == "multi") c(1L, 1L) else c(2L, 1L, 2L), empty = integer()) else list(all = NULL)
    for (selection_name in names(selections)) {
      selection <- selections[selection_name][[1L]]
      group <- groups[[group_name]]
      raw_options <- list(object = fit, type = type, collapse = group, se.fit = se)
      if (!is.null(selection)) raw_options$terms <- selection
      raw <- capture(function() do.call(predict, raw_options))
      expected <- capture(function() {
        options <- list(object = if (name == "sparse" && type == "terms") dense else fit,
                        type = if (type == "survival") "expected" else type, se.fit = se)
        # Select explicitly after the stock full terms call, retaining dimensions
        # for repeated/empty columns and avoiding stock's one-term drop behavior.
        value <- do.call(predict, options)
        if (name == "sparse" && type == "terms") {
          index <- as.integer(factor(fit$x[, 2L]))
          frail <- naresid(fit$na.action, fit$frail[index])
          fvar <- naresid(fit$na.action, sqrt(fit$fvar[index]))
          if (se) { value$fit <- cbind(value$fit, frail); value$se.fit <- cbind(value$se.fit, fvar) }
          else value <- cbind(value, frail)
        }
        if (type == "terms" && !is.null(selection)) {
          pick <- function(x) {
            if (!is.matrix(x)) x <- matrix(x, ncol = 1L)
            x[, selection, drop = FALSE]
          }
          if (se) { value$fit <- pick(value$fit); value$se.fit <- pick(value$se.fit) } else value <- pick(value)
        }
        if (type == "survival") {
          if (se) { value$fit <- exp(-value$fit); value$se.fit <- value$se.fit * value$fit }
          else value <- exp(-value)
        }
        if (se) list(fit = rowsum(value$fit, group), se.fit = sqrt(rowsum(value$se.fit^2, group)))
        else rowsum(value, group)
      })
      cases[[length(cases) + 1L]] <- list(name = paste(name, type, se, group_name, selection_name, sep = "/"),
        model = name, type = type, se_fit = se, terms = if (is.null(selection)) NULL else I(selection),
        collapse = I(as.character(group)), levels = if (is.factor(group)) I(levels(group)) else NULL,
        numeric_groups = is.numeric(group), raw = list(result = encode(raw$result), warnings = raw$warnings),
        expected = list(result = encode(expected$result), warnings = expected$warnings))
    }
  }
  if (name == "plain") {
    nd <- d[seq_len(8), ]; nd$x[c(2, 7)] <- NA
    group <- factor(c("kept-a", "only-omitted", NA, "kept-c", "kept-a", "kept-c", "only-omitted", "kept-a"),
                    levels = c("kept-c", "only-omitted", "kept-a", "unused"))
    for (type in types) for (se in c(FALSE, TRUE)) for (action in c("na.pass", "na.omit", "na.exclude")) {
      raw <- capture(function() predict(fit, nd, type = type, se.fit = se, collapse = group, na.action = action))
      expected <- capture(function() {
        # For omit/exclude, collapse itself contributes to model-frame missingness.
        used <- nd
        if (action != "na.pass") used$x[is.na(group)] <- NA
        value <- predict(fit, used, type = type, se.fit = se, na.action = action)
        labels <- if (action == "na.omit") group[complete.cases(used[, c("x", "rx")])] else group
        if (se) list(fit = rowsum(value$fit, labels), se.fit = sqrt(rowsum(value$se.fit^2, labels)))
        else rowsum(value, labels)
      })
      cases[[length(cases) + 1L]] <- list(name = paste("newdata", type, se, action, sep = "/"), model = name,
        type = type, se_fit = se, terms = NULL, collapse = I(as.character(group)), levels = I(levels(group)),
        newdata = nd, na_action = action, numeric_groups = FALSE,
        raw = list(result = encode(raw$result), warnings = raw$warnings),
        expected = list(result = encode(expected$result), warnings = expected$warnings))
    }
  }
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()), survival = as.character(packageVersion("survival")),
  reference = "uncollapsed stock R predictions padded before base rowsum; sparse terms add fitted frailties; survival uses expected and delta-method errors"),
  data = d, levels = I(levels(d$cl)), specs = specs, cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "grouped Cox prediction references written\n")
