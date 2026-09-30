#!/usr/bin/env Rscript
# Regenerate with R 4.5.3 / survival 3.8-12.
library(survival)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
  "python/tests/fixtures/coxms_subset_reference.json"

data <- do.call(rbind, lapply(1:24, function(id) {
  dead <- id %% 3 == 0
  ill <- id %% 2 == 0
  status <- c(if (ill) "ill" else "censor",
              if (dead) "dead" else if (ill) "censor" else "ill",
              if (id %% 4 == 0) "censor" else "dead")
  k <- if (dead) 2L else 3L
  shift <- (id %% 4) / 4
  data.frame(id = id, start = c(0, 2 + shift, 4 + shift)[seq_len(k)],
             stop = c(2, 4, 6)[seq_len(k)] + shift, event = status[seq_len(k)],
             x = ((id * 7) %% 13 - 6) / 4 + seq_len(k) / 10,
             g = if (id %% 3 == 1) "a" else "b", weight = 0.5 + (id %% 3) / 2)
}))
data$event <- factor(data$event, levels = c("censor", "ill", "dead"))
newdata <- data.frame(x = c(-0.7, 0.2, 1.1))

array_data <- function(x) if (is.null(x)) NULL else
  list(shape = I(if (is.null(dim(x))) length(x) else dim(x)), values = I(as.numeric(x)))
curve_data <- function(x) list(
  time = I(x$time), n = I(x$n), n_id = I(x$n.id), states = I(x$states),
  n_risk = array_data(x$n.risk), n_event = array_data(x$n.event),
  n_censor = array_data(x$n.censor), pstate = array_data(x$pstate),
  p0 = array_data(matrix(x$p0, ncol = length(x$states))),
  strata = if (is.null(x$strata)) NULL else as.list(x$strata))
summary_data <- function(x) list(
  time = I(x$time), states = I(x$states),
  n_risk = array_data(x$n.risk), n_event = array_data(x$n.event),
  n_censor = array_data(x$n.censor), pstate = array_data(x$pstate),
  strata = if (is.null(x$strata)) NULL else I(as.character(x$strata)),
  table = array_data(x$table), table_rows = I(as.character(rownames(x$table))),
  table_columns = I(as.character(colnames(x$table))),
  rmean_endtime = if (is.null(x$rmean.endtime)) NULL else I(x$rmean.endtime))

cases <- list()
for (stratified in c(FALSE, TRUE)) {
  formula <- if (stratified) Surv(start, stop, event) ~ x + strata(g) else
    Surv(start, stop, event) ~ x
  fit <- coxph(formula, data, id = id, weights = weight)
  for (conditional in c(FALSE, TRUE)) for (time0 in c(FALSE, TRUE)) {
    options <- list(formula = fit, newdata = newdata, time0 = time0)
    if (conditional) options <- c(options, list(start.time = 3, p0 = c(.6, .4, 0)))
    source <- do.call(survfit, options)
    for (states in list(3L, c(3L, 1L), c(3L, 2L, 3L))) {
      # Reverse strata and prediction rows as well; one case retains one stratum.
      groups <- if (!stratified) 1L else if (length(states) == 1L) 2L else 2:1
      rows <- if (stratified) unlist(lapply(groups, function(i) {
        seq_len(source$strata[i]) + sum(source$strata[seq_len(i - 1L)])
      })) else seq_along(source$time)
      selected <- source[groups, c(3, 1), states, drop = FALSE]
      raw_censor <- selected$n.censor
      # [.survfitms leaves n.censor in the original state order and does not
      # subset n.id. Keep raw counts for the compatibility record, then align
      # only these metadata fields before asking stock R for summary rows.
      selected$n.censor <- source$n.censor[rows, states, drop = FALSE]
      selected$n.id <- source$n.id[groups]
      summaries <- list()
      for (mode in c("events", "all", "times", "extended", "scaled", "none")) {
        settings <- switch(mode,
          events = list(), all = list(censored = TRUE),
          times = list(times = c(3, 4, 6)),
          extended = list(times = c(if (conditional) 3 else 0, 4, 8), extend = TRUE),
          scaled = list(times = c(3, 4, 6), scale = 2, rmean = 5),
          none = list(rmean = "none"))
        result <- do.call(summary, c(list(object = selected), settings))
        raw_table <- result$table
        raw_summary_censor <- result$n.censor
        # survmean2 recycles nevent in the wrong order for several strata and
        # prediction rows. Compute that column directly from selected counts.
        sizes <- if (is.null(selected$strata)) length(selected$time) else selected$strata
        ends <- cumsum(sizes)
        starts <- c(1L, head(ends, -1L) + 1L)
        if (mode %in% c("events", "none")) {
          # summary.survfitms drops the censor matrix to its first column when
          # joining several strata without requested times. Accumulate each
          # state's counts independently between the selected event rows.
          result$n.censor <- do.call(rbind, lapply(seq_along(sizes), function(g) {
            rows <- starts[g]:ends[g]
            kept <- which(rowSums(selected$n.event[rows, , drop = FALSE]) > 0)
            previous <- c(0L, head(kept, -1L))
            counts <- vapply(seq_along(kept), function(i) {
              colSums(selected$n.censor[rows[(previous[i] + 1L):kept[i]], , drop = FALSE])
            }, numeric(length(states)))
            t(matrix(counts, nrow = length(states)))
          }))
        }
        totals <- vapply(seq_along(states), function(state) {
          vapply(seq_along(sizes), function(g) {
            sum(selected$n.event[starts[g]:ends[g], state])
          }, numeric(1))
        }, numeric(length(sizes)))
        dim(totals) <- c(length(sizes), length(states))
        result$table[, "nevent"] <- unlist(lapply(seq_along(states), function(s) {
          rep(totals[, s], 2)
        }))
        summaries[[mode]] <- list(expected = summary_data(result),
                                  raw_table = array_data(raw_table),
                                  raw_censor = array_data(raw_summary_censor))
      }
      cases[[length(cases) + 1L]] <- list(
        stratified = stratified, conditional = conditional, time0 = time0,
        states = I(states - 1L), groups = I(groups - 1L),
        curve = curve_data(selected), initial = curve_data(survfit0(selected)),
        raw_censor = array_data(raw_censor), summaries = summaries)
    }
  }
}
write_json(list(r_version = R.version.string,
                survival_version = as.character(packageVersion("survival")),
                data = lapply(data, I), newdata = lapply(newdata, I), cases = cases),
           output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
cat(length(cases), "multistate curve subsets written to", output, "\n")
