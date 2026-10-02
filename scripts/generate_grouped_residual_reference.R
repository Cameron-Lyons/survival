#!/usr/bin/env Rscript
# Stock survival differential references for grouped residual and pseudo preparation.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/grouped_residual_reference.json"

make_data <- function(histories, multistate) {
  groups <- sprintf("clinic%02d", seq_len(7))
  pieces <- lapply(seq_along(groups), function(g) {
    if (histories) {
      d <- data.frame(start = c(0, 1, 0, 2, 0, 3),
                      time = c(1, 5, 2, 6, 3, 5.5),
                      status = c(0, 1, 1, 0, 0, 1),
                      event = c("a", "b", "b", "a", "censor", "a"),
                      subject = sprintf("subject%02d_%d", g, rep(1:3, each = 2)),
                      weight = rep(c(.5, 1.25, 2), each = 2) + g / 20)
    } else {
      d <- data.frame(time = c(1, 2.5, 4, 5, 6, NA), status = c(1, 0, 1, 1, 0, 1),
                      event = c("a", "censor", "b", "a", "censor", "a"),
                      subject = sprintf("subject%02d_%d", g, 1:6),
                      weight = c(.5, 1.25, 2, 1.5, .75, 1) + g / 20)
    }
    d$group <- groups[g]
    d$cluster <- paste0("cluster_", d$subject)
    d
  })
  d <- do.call(rbind, pieces)
  # Original rows alternate strata and reverse within-stratum history order.
  # This keeps first-appearance id order different from factor-level order.
  per_group <- nrow(pieces[[1L]])
  order <- as.vector(t(matrix(seq_len(nrow(d)), nrow = per_group)[per_group:1, 7:1]))
  d <- d[order, , drop = FALSE]
  rownames(d) <- NULL
  d$group <- factor(d$group, levels = c("unused", rev(groups)))
  if (multistate) d$event <- factor(d$event, levels = c("censor", "b", "a", "unused-state"))
  d
}

encode <- function(value) list(values = I(as.numeric(value)), dim = I(dim(value)))
frame <- function(value) lapply(value, function(column) I(unname(column)))
cases <- list()
datasets <- list()
for (histories in c(FALSE, TRUE)) for (multistate in c(FALSE, TRUE)) {
  name <- paste(if (multistate) "aj" else "km", if (histories) "histories" else "omitted", sep = "_")
  d <- make_data(histories, multistate)
  formula <- paste0("Surv(", if (histories) "start," else "", "time,",
                    if (multistate) "event" else "status", ") ~ group")
  options <- list(formula = as.formula(formula), data = d, weights = quote(weight),
                  id = quote(subject), cluster = quote(cluster), timefix = FALSE,
                  na.action = na.omit, model = TRUE)
  # Both counting KM and right-censored AJ retain model rows before the
  # cutoff, with different residual rules. Unconditional AJ histories also
  # exercise multiple transitions and empirical initial-state preparation.
  start_time <- if (xor(histories, multistate)) 1.5 else NULL
  if (!is.null(start_time)) options$start.time <- start_time
  fit <- do.call(survfit, options)
  datasets[[name]] <- list(data = frame(d), formula = formula,
                           levels = I(levels(d$group)), states = if (multistate) I(levels(d$event)) else NULL,
                           start_time = start_time)
  for (type in c("pstate", "cumhaz", "auc")) {
    for (mode in c("unweighted", "weighted", "collapsed")) {
      collapse <- mode == "collapsed"
      weighted <- mode != "unweighted"
      value <- residuals(fit, times = c(2.5, 4.5), type = type,
                         collapse = collapse, weighted = weighted, extra = TRUE)
      long <- residuals(fit, times = c(2.5, 4.5), type = type,
                        collapse = collapse, weighted = weighted, data.frame = TRUE)
      cases[[length(cases) + 1L]] <- list(
        name = paste(name, "residual", type, mode, sep = "/"), dataset = name,
        operation = "residual", type = type, collapse = collapse, weighted = weighted,
        expected = encode(value$resid), id = I(dimnames(value$resid)[[1L]]),
        curve = if (is.null(value$curve)) NULL else I(value$curve),
        columns = if (multistate) I(dimnames(value$resid)[[2L]]) else NULL,
        data_frame = frame(long))
    }
    for (collapse in c(FALSE, TRUE)) {
      value <- pseudo(fit, times = c(2.5, 4.5), type = type, collapse = collapse)
      long <- pseudo(fit, times = c(2.5, 4.5), type = type, collapse = collapse, data.frame = TRUE)
      cases[[length(cases) + 1L]] <- list(
        name = paste(name, "pseudo", type, collapse, sep = "/"), dataset = name,
        operation = "pseudo", type = type, collapse = collapse,
        expected = encode(value), data_frame = frame(long))
    }
  }
}
reference <- list(metadata = list(R = R.version.string,
                  survival = as.character(packageVersion("survival")),
                  provenance = "Unmodified stock residuals.survfit and pseudo; arrays stored in R column order."),
                  times = I(c(2.5, 4.5)), datasets = datasets, cases = cases)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE, na = "null", null = "null")
cat(length(cases), "grouped residual/pseudo cases written to", output, "\n")
