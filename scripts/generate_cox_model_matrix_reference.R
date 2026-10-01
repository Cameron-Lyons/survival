#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/cox_model_matrix_reference.json"
d <- kidney[seq_len(24), ]
d$group <- 100 + 3 * d$id
d$small <- factor(d$id %% 3, levels = c(2, 0, 1))
d$o <- (seq_len(nrow(d)) %% 5 - 2) / 20
specs <- list(
  ridge = list(rhs = "ridge(age, theta = 2)"),
  ridge_matrix = list(rhs = "ridge(age, sex, theta = 2)"),
  spline = list(rhs = "pspline(age, theta = 0.4)"),
  spline_unpenalized = list(rhs = "pspline(age, df = 2, nterm = 3, degree = 1, penalty = FALSE)"),
  spline_interaction = list(rhs = "pspline(age, df = 2, nterm = 3, degree = 1, penalty = FALSE):small"),
  ordinary = list(rhs = "age * small + strata(sex) + offset(o) + cluster(id)"),
  sparse_first = list(rhs = "frailty(group, sparse = TRUE, theta = 0.4) + age + small"),
  sparse_only = list(rhs = "frailty(group, sparse = TRUE, theta = 0.4)"),
  sparse_middle = list(rhs = "age + frailty(group, sparse = TRUE, theta = 0.4) + small"),
  factor_dense = list(rhs = "age + frailty(small, sparse = FALSE, theta = 0.4)"),
  factor_auto = list(rhs = "age + frailty(small, theta = 0.4)")
)
for (family in c("gamma", "gaussian", "t")) for (sparse in c("TRUE", "FALSE", "auto")) {
  name <- paste(family, sparse, sep = "_")
  specs[[name]] <- list(rhs = paste0("age + strata(sex) + frailty.", family,
    "(group, ", if (sparse == "auto") "" else paste0("sparse = ", sparse, ", "),
    "theta = 0.4) + offset(o)"))
}
for (name in c("ridge", "ridge_matrix", "spline", "spline_unpenalized")) {
  specs[[paste0("aft_", name)]] <- list(rhs = paste(specs[[name]]$rhs, "+ strata(sex)"), model = "survreg")
}
matrix_result <- function(fun) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers({
    x <- fun()
    list(data = if (nrow(x)) unname(x) else I(list()), columns = I(colnames(x)),
      assign = I(attr(x, "assign")),
      strata = if (is.null(attr(x, "strata"))) NULL else I(as.character(attr(x, "strata"))))
  }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) list(error = conditionMessage(e)))
  result$warnings <- I(warnings)
  result
}
reconstructed_matrix <- function(fit, nd, formula, model, unpenalized) {
  # The stock Cox method misspells contrasts.arg when removing strata. Use
  # stats' ordinary matrix constructor with the fitted contrasts explicitly.
  Terms <- delete.response(terms(fit))
  if (unpenalized) {
    # makepredictcall.pspline drops df and penalty=FALSE, which can make a
    # valid fitted basis fail (default df=4 with nterm=3). Retain the original
    # constructor options and add the fitted Boundary.knots instead.
    original <- attr(terms(formula), "variables")
    training <- model.frame(formula, d)
    pv <- attr(Terms, "predvars")
    for (i in seq_along(pv)[-1L]) if (is.call(pv[[i]]) && identical(pv[[i]][[1L]], as.name("pspline"))) {
      call <- original[[i + 1L]]
      basis <- training[[i]]
      call$Boundary.knots <- attr(basis, "Boundary.knots")
      pv[[i]] <- call
    }
    attr(Terms, "predvars") <- pv
  }
  mf <- model.frame(Terms, nd, xlev = fit$xlevels)
  x <- stats::model.matrix(Terms, mf, contrasts.arg = fit$contrasts)
  assign <- attr(x, "assign")
  strata <- which(grepl("^strata\\(", attr(Terms, "term.labels")))
  keep <- !assign %in% c(if (model == "coxph") 0L else integer(), strata)
  x <- x[, keep, drop = FALSE]
  attr(x, "assign") <- if (model == "coxph") assign[keep] else
    vapply(assign[keep], function(code) as.integer(code - sum(strata < code)), integer(1))
  if (model == "coxph" && length(strata)) attr(x, "strata") <- mf[[attr(Terms, "term.labels")[strata]]]
  x
}
cases <- list()
for (name in names(specs)) {
  spec <- specs[[name]]
  model <- if (is.null(spec$model)) "coxph" else spec$model
  formula <- as.formula(paste("Surv(time, status) ~", spec$rhs))
  fit <- if (model == "coxph") coxph(formula, d, x = TRUE, robust = FALSE)
    else survreg(formula, d, x = TRUE)
  existing <- d[c(8, 1, 8, 4), ]; existing$age <- c(40, 50, 60, 35)
  missing <- existing
  missing$age[2] <- NA; missing$group[3] <- NA; missing$o[4] <- NA
  unseen <- existing; unseen$group <- c(999, 888, 999, 888)
  one <- existing; one$group <- rep(existing$group[1], nrow(one)); one$small <- factor(rep(2, nrow(one)), levels = levels(d$small))
  switched <- d[seq_len(8), ]; switched$small <- factor(20:27)
  variants <- list(stored = NULL, existing = existing, missing = missing, unseen = unseen,
    one_group = one, empty = d[FALSE, ], auto_switch = switched)
  for (variant in names(variants)) {
    nd <- variants[[variant]]
    raw <- matrix_result(function() if (is.null(nd)) model.matrix(fit) else model.matrix(fit, nd))
    expected <- raw
    reference <- "unmodified stock model.matrix"
    unpenalized <- grepl("penalty = FALSE", spec$rhs, fixed = TRUE)
    dense_strata <- grepl("sparse = FALSE", spec$rhs, fixed = TRUE) && grepl("strata(", spec$rhs, fixed = TRUE)
    if (!is.null(nd) && (unpenalized || (dense_strata && is.null(raw$error)))) {
      expected <- matrix_result(function() reconstructed_matrix(fit, nd, formula, model, unpenalized))
      reference <- if (unpenalized) "R spline basis with original constructor options and fitted knots" else
        "stats::model.matrix with fitted full frailty contrasts and explicit strata removal"
    }
    if (!is.null(nd) && nrow(nd) == 0L && grepl("pspline(", spec$rhs, fixed = TRUE)) {
      expected <- matrix_result(function() {
        x <- fit$x[FALSE, , drop = FALSE]
        attr(x, "assign") <- attr(fit$x, "assign")
        x
      })
      reference <- "Empty basis retaining fitted matrix columns and assignments"
    }
    cases[[length(cases) + 1L]] <- list(name = paste(name, variant, sep = "/"), model = model,
      formula = paste(deparse(formula, width.cutoff = 500L), collapse = " "),
      newdata = nd, new_levels = if (is.null(nd)) NULL else I(levels(nd$small)),
      reference = reference, raw_stock = raw, expected = expected)
  }
}
jsonlite::write_json(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")),
  reference = "Stock model.matrix calls, with explicitly marked fitted-contrast, spline prediction-call, and empty-basis references. Raw stock matrices, errors and warnings retained."),
  data = d, levels = I(levels(d$small)), cases = cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "Cox/AFT model matrix references written\n")
