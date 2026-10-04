#!/usr/bin/env Rscript
# Stock survival references for omitted/excluded survival residual rows.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/survfit_missing_residual_reference.json"

frame <- function(value) lapply(value, function(column) I(unname(column)))
encode <- function(value) list(values = I(as.numeric(value)),
                               dim = if (is.null(dim(value))) NULL else I(dim(value)),
                               row_names = if (is.null(dimnames(value))) NULL else I(dimnames(value)[[1L]]))
datasets <- list()
cases <- list()
for (histories in c(FALSE, TRUE)) for (multistate in c(FALSE, TRUE)) {
  if (histories) {
    d <- data.frame(start = c(0,1,0,2,0,1,0,3,0,4),
                    time = c(1,5,2,6,1,4,3,7,NA,8),
                    status = c(0,1,1,0,0,1,0,1,0,1),
                    event = c("a","b","b","censor","a","b","b","a","a","censor"),
                    subject = rep(letters[1:5], each = 2),
                    weight = rep(c(.75,1.25,1.5,.5,2), each = 2),
                    group = rep(c("a","b"), c(4,6)))
    d$weight[3L] <- NA
  } else {
    d <- data.frame(time = c(1,NA,3,4,5,6,7,8),
                    status = c(1,0,1,0,1,0,1,0),
                    event = c("a","censor","b","censor","a","censor","b","censor"),
                    subject = letters[1:8], weight = c(1,1,.75,1.25,1.5,NA,1,2),
                    group = rep(c("a","b"), each = 4))
  }
  if (multistate) d$event <- factor(d$event, levels = c("censor","b","a"))
  # The competing-risk case without ids verifies original observation positions.
  use_id <- histories || !multistate
  name <- paste(if (multistate) "aj" else "km", if (histories) "histories" else "right", sep = "_")
  formula <- paste0("Surv(", if (histories) "start," else "", "time,",
                    if (multistate) "event" else "status", ") ~ group")
  datasets[[name]] <- list(data = frame(d), formula = formula, id = use_id,
                           states = if (multistate) I(levels(d$event)) else NULL)
  for (action in c("na.omit", "na.exclude")) {
    options <- list(formula = as.formula(formula), data = d, weights = quote(weight),
                    na.action = get(action), timefix = FALSE, model = TRUE)
    if (use_id) options$id <- quote(subject)
    fit <- do.call(survfit, options)
    for (collapse in c(FALSE, TRUE)) for (type in c("pstate", "cumhaz", "auc")) {
      query_times <- c(if (type == "auc") 5.5 else 2.5, 6.5)
      for (times in list(query_times[1L], query_times)) {
        residual <- residuals(fit, times = times, type = type, collapse = collapse, extra = TRUE)
        residual_frame <- tryCatch(
          residuals(fit, times = times, type = type, collapse = collapse, data.frame = TRUE),
          error = function(e) e)
        residual_frame_error <- NULL
        if (inherits(residual_frame, "error")) {
          residual_frame_error <- conditionMessage(residual_frame)
          # Grouped AJ single-time tables fail in stock R's col(array) call.
          # Select the same rows from an independent successful multi-time call.
          residual_frame <- residuals(fit, times = query_times, type = type,
                                      collapse = collapse, data.frame = TRUE)
          residual_frame <- residual_frame[residual_frame$time %in% times, , drop = FALSE]
        }
        values <- suppressWarnings(pseudo(fit, times = times, type = type, collapse = collapse))
        pseudo_frame <- tryCatch(
          suppressWarnings(pseudo(fit, times = times, type = type,
                                  collapse = collapse, data.frame = TRUE)),
          error = function(e) e)
        pseudo_frame_error <- NULL
        if (inherits(pseudo_frame, "error")) {
          pseudo_frame_error <- conditionMessage(pseudo_frame)
          pseudo_frame <- suppressWarnings(pseudo(fit, times = query_times, type = type,
                                                  collapse = collapse, data.frame = TRUE))
          pseudo_frame <- pseudo_frame[pseudo_frame$time %in% times, , drop = FALSE]
        }
        cases[[length(cases) + 1L]] <- list(
          name = paste(name, action, collapse, type, length(times), sep = "/"),
          dataset = name, na_action = action, collapse = collapse, type = type, times = I(times),
          na_rows = I(as.integer(fit$na.action)), residual = encode(residual$resid),
          curve = if (is.null(residual$curve)) NULL else I(residual$curve),
          residual_frame = frame(residual_frame), residual_frame_error = residual_frame_error,
          pseudo = encode(values), pseudo_frame = frame(pseudo_frame),
          pseudo_frame_error = pseudo_frame_error)
      }
    }
  }
}
reference <- list(metadata = list(R = as.character(getRversion()),
                  survival = as.character(packageVersion("survival")),
                  provenance = "Unmodified stock residuals.survfit and pseudo; arrays stored in R column order."),
                  datasets = datasets, cases = cases)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE, na = "null", null = "null")
cat(length(cases), "missing residual/pseudo cases written to", output, "\n")
