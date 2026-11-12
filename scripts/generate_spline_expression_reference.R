#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(splines))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/spline_expression_reference.json"
d <- lung[seq_len(80), c("time", "status", "age", "sex")]
d$x <- d$age
d$z <- d$sex
d[["age value"]] <- d$age
missing <- d
missing$x[c(6, 31)] <- NA_real_
new <- data.frame(x = c(40, 55, NA, 70, 80), z = c(1, 2, 1, 2, 1))
new[["age value"]] <- new$x
specs <- list(
  ns_log = list(rhs = "ns(log(x), df = 3)"),
  bs_arithmetic = list(rhs = "bs(x / 10, df = 3)"),
  poly_log = list(rhs = "poly(log(x), degree = 2)"),
  nsk_multiple = list(rhs = "nsk(x + z, df = 3)"),
  poly_raw = list(rhs = "poly(x / z, degree = 2, raw = TRUE)"),
  quoted = list(rhs = "ns(`age value`, df = 3)"),
  interaction = list(rhs = "ns(log(x), df = 3) * factor(z)"),
  scale_subset = list(rhs = "ns(scale(x), df = 3)", subset = c(80:41, 2, 7, 2, 20:30)),
  sqrt_missing = list(rhs = "bs(sqrt(x - 50), df = 3)", missing = TRUE),
  recovered_missing = list(rhs = "ns(x^0 * `age value`, df = 3)", missing = TRUE)
)
matrix_snapshot <- function(x) list(data = unname(x), columns = I(colnames(x)),
                                   assign = I(attr(x, "assign")))
cases <- list()
for (name in names(specs)) for (model in c("coxph", "survreg")) {
  spec <- specs[[name]]
  data <- if (isTRUE(spec$missing)) missing else d
  f <- as.formula(paste("Surv(time, status) ~", spec$rhs))
  warnings <- character()
  value <- withCallingHandlers({
    fit <- do.call(get(model), c(list(formula = f, data = data, x = TRUE,
                   model = TRUE, na.action = na.exclude),
                   if (is.null(spec$subset)) list() else list(subset = spec$subset)))
    pred <- predict(fit, new, type = "lp", se.fit = TRUE, na.action = na.pass)
    list(coefficients = I(unname(coef(fit))), variance = unname(vcov(fit)),
         loglik = I(fit$loglik), matrix = matrix_snapshot(model.matrix(fit)),
         newmatrix = matrix_snapshot(model.matrix(fit, new)),
         predict = I(unname(pred$fit)), se = I(unname(pred$se.fit)),
         omitted = I(as.integer(fit$na.action)))
  }, warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  cases[[length(cases) + 1L]] <- c(list(name = paste(model, name, sep = "/"),
      model = model, formula = paste(deparse(f, width.cutoff = 500L), collapse = " "),
      missing = isTRUE(spec$missing), subset = if (is.null(spec$subset)) NULL else I(spec$subset - 1L),
      warnings = I(warnings)), value)
}
jsonlite::write_json(list(metadata = list(survival = as.character(packageVersion("survival"))),
    data = d, missing_data = missing,
    newdata = new, cases = cases), output, auto_unbox = TRUE, pretty = TRUE,
    digits = 17, na = "null", null = "null", dataframe = "columns")
cat(length(cases), "spline expression references written with", R.version.string, "\n")
