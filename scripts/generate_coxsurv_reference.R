#!/usr/bin/env Rscript
library(survival)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/coxsurv_reference.json"
array_values <- function(x) if (is.null(x)) NULL else I(x)
rows <- function(x) lapply(seq_len(nrow(x)), function(i) array_values(as.numeric(x[i, ])))
pack <- function(fit) {
  out <- lapply(fit, function(x) if (is.matrix(x)) rows(x) else array_values(as.numeric(x)))
  out
}
x <- cbind(c(.2, .8, -.3, 1.1, .4, -.7, .6, -.1), c(1, 0, 1, 2, -1, .5, 1.5, -.5))
stop <- c(1, 2, 2, 3, 4, 4, 2.5, 5)
start <- c(0, .5, 0, 1.5, 2, 1, .25, 3.5)
status <- c(1, 1, 1, 0, 1, 1, 1, 0)
weights <- c(1, .5, 2, 1.5, 1, .75, 1.25, .8)
risk <- exp(c(-.2, .3, .1, -.4, .5, .2, -.1, .4))
variance <- matrix(c(.08, .01, .01, .05), 2)
group <- c("a", "b", "b", "a", "a", "a", "b", "b")
levels <- c("b", "a")
cases <- list()
add <- function(name, y, x2, risk2, strata, y2 = NULL, strata2 = NULL, id2 = NULL) {
  for (stype in 1:2) for (ctype in 1:2) for (se in c(FALSE, TRUE)) for (unlist in c(FALSE, TRUE)) {
    arguments <- list(ctype = ctype, stype = stype, se.fit = se, varmat = variance,
                      y = y, x = x, wt = weights, risk = risk, strata = strata,
                      x2 = x2, risk2 = risk2, y2 = y2, strata2 = strata2, id2 = id2,
                      unlist = unlist)
    fit <- do.call(survival::coxsurv.fit, arguments)
    cases[[length(cases) + 1L]] <<- list(
      name = paste(name, stype, ctype, se, unlist, sep = "/"),
      stype = stype, ctype = ctype, se_fit = se, unlist = unlist,
      y = rows(y), x2 = rows(x2), risk2 = array_values(risk2), rownames = array_values(rownames(x2)),
      strata = if (is.null(strata)) NULL else array_values(as.character(strata)),
      levels = if (is.factor(strata)) array_values(base::levels(strata)) else NULL,
      y2 = if (is.null(y2)) NULL else rows(y2), strata2 = array_values(strata2), id2 = array_values(id2),
      expected = if (unlist) pack(fit) else lapply(fit, pack),
      names = if (unlist) array_values(names(fit$strata)) else array_values(names(fit))
    )
  }
}
for (counting in c(FALSE, TRUE)) for (stratified in c(FALSE, TRUE)) for (multiple in c(FALSE, TRUE)) {
  y <- if (counting) cbind(start, stop, status) else cbind(stop, status)
  x2 <- rbind(first = c(.1, .2), second = c(-.2, .6))
  if (!multiple) x2 <- x2[1L, , drop = FALSE]
  add(paste(if (counting) "counting" else "right", stratified, multiple, sep = "/"),
      y, x2, exp(c(.12, -.21))[seq_len(nrow(x2))],
      if (stratified) factor(group, levels = levels) else NULL)
}
add("individual", cbind(start, stop, status),
    rbind(c(.1, .2), c(-.2, .6), c(.3, -.1), c(.4, .5)), exp(c(.12, -.21, .07, .18)),
    factor(group, levels = levels),
    y2 = rbind(c(0, 2.5), c(0, 3), c(2.5, 5), c(3, 5)),
    strata2 = c(1L, 2L, 2L, 1L), id2 = c("second", "first", "second", "first"))
write_json(list(r_version = R.version.string,
                survival_version = as.character(packageVersion("survival")),
                x = rows(x), weights = array_values(weights), risk = array_values(risk), variance = rows(variance),
                cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
