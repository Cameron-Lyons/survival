#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/pyears_matrix_reference.json"
rows <- function(x) lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
d <- data.frame(time = c(0, 1, 3, 4, 6, 8, 5, 7),
  start = c(0, 0, 1, 2, 3, 5, 5, 8), event = c(2, .5, 3, 0, 1, 4, 1, 0),
  age = c(1, 2, 5, 9, 3, 4, 6, 8), group = rep(c("a", "b"), 4),
  weight = c(1, .5, 2, 0, 1.2, 1, 3, .5))
table <- structure(c(.01, .02, .03), dim = 3L, dimnames = list(age = c("0", "5", "10")),
  dimid = "age", type = 2L, factor = 0L, cutpoints = list(c(0, 5, 10)), class = "ratetable")
cases <- list()
for (rate in c(FALSE, TRUE)) for (expect in if (rate) c("event", "pyears") else "event") {
 for (width in c(1L, 2L)) {
  for (rhs in c("1", "group", "group + tcut(age, c(0, 5, 10, 20))")) {
    for (selection in c("all", "subset_na")) {
      data <- d
      subset <- if (selection == "all") seq_len(nrow(data)) else c(8, 3, 2, 3, 1, 6)
      if (selection == "subset_na") {
        data$time[2] <- NA_real_
        data$event[6] <- NA_real_
        data$age[3] <- NA_real_
      }
      response <- if (width == 1L) "cbind(time)" else
        if (rate) "cbind(start, time)" else "cbind(time, event)"
      formula <- paste(response, "~", rhs)
      input <- list(formula = as.formula(formula), data = data, weights = data$weight,
        subset = subset, na.action = na.exclude, scale = 2, expect = expect, x = TRUE, y = TRUE)
      if (rate) input$ratetable <- table
      out <- do.call(pyears, input)
      model <- do.call(pyears, modifyList(input, list(model = TRUE)))$model
      cases[[length(cases) + 1L]] <- list(name = paste(rate, expect, width, rhs, selection, sep = "/"),
        formula = formula, data = lapply(data, I), subset = I(subset - 1L), ratetable = rate, expect = expect,
        pyears = I(as.numeric(out$pyears)), n = I(as.numeric(out$n)),
        event = if (is.null(out$event)) NULL else I(as.numeric(out$event)),
        expected = if (is.null(out$expected)) NULL else I(as.numeric(out$expected)),
        offtable = out$offtable, observations = out$observations, tcut = out$tcut,
        dim = I(as.integer(dim(out$pyears))), dimnames = lapply(dimnames(out$pyears), I),
        na_action = I(as.integer(out$na.action)), y = rows(out$y),
        model_y = rows(model[[1L]]), model_names = I(names(model)),
        x = if (is.matrix(out$x)) rows(out$x) else I(out$x))
    }
  }
 }
}
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
  cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null")
