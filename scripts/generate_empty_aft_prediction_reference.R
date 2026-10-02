#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/empty_aft_prediction_reference.json"
d <- ovarian
d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
nd <- d[3:6, ]
row.names(nd) <- c("四", "two / 2", "six:6", "one")
nd$age[] <- NA_real_
encode <- function(value) {
  if (is.list(value)) return(lapply(value, encode))
  list(values = I(as.numeric(value)), dim = if (is.null(dim(value))) NULL else I(dim(value)),
       has_dimnames = !is.null(dimnames(value)),
       names = if (is.null(names(value))) NULL else I(names(value)),
       rows = if (is.null(rownames(value))) NULL else I(rownames(value)),
       columns = if (is.null(colnames(value))) NULL else I(colnames(value)))
}
cases <- list()
for (model in c("aft", "aft_strata", "aft_fixed")) {
  rhs <- if (model == "aft_strata") "age + rx + strata(cl)" else "age + rx"
  fit <- survreg(as.formula(paste("Surv(futime, fustat) ~", rhs)), d,
                scale = if (model == "aft_fixed") 1 else 0)
  for (type in c("response", "link", "quantile", "uquantile")) {
    probabilities <- if (type %in% c("quantile", "uquantile")) list(.5, c(.1,.5,.9)) else list(.5)
    for (p in probabilities) for (action in c("na.omit", "na.exclude")) for (se in c(FALSE, TRUE)) {
      value <- predict(fit, nd, type = type, p = p, na.action = action, se.fit = se)
      cases[[length(cases) + 1L]] <- list(
        name = paste(model, type, length(p), action, se, sep = "/"), model = model,
        type = type, p = I(p), na_action = action, se_fit = se, result = encode(value))
    }
  }
}
serialize <- function(value) jsonlite::toJSON(value, auto_unbox = TRUE, digits = 17,
                                            na = "null", null = "null")
header <- serialize(list(metadata = list(r = as.character(getRversion()),
                                        survival = as.character(packageVersion("survival")),
                                        reference = "stock AFT predictions when all new rows are omitted")))
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
             vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),
                 if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes = TRUE)
cat(length(cases), "empty AFT prediction references written\n")
