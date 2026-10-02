#!/usr/bin/env Rscript
# Compact stock-R checks for independent competing-risk uncertainty and its fallbacks.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/aj_variance_reference.json"

number <- function(value) {
    if (is.na(value)) return("NaN")
    if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
    value
}
vector <- function(values) lapply(unname(values), number)
rows <- function(values) {
    if (is.null(values)) return(NULL)
    if (!is.matrix(values)) values <- matrix(values, nrow = 1)
    lapply(seq_len(nrow(values)), function(row) vector(values[row, ]))
}
cases <- list()
add <- function(name, time, state, states = c("a", "b"), weights = rep(1, length(time)),
                istate = NULL, strata = NULL, p0 = NULL, start_time = NULL,
                time0 = FALSE, conf_type = "log") {
    data <- data.frame(time = time,
                       event = factor(c("censor", states)[state + 1],
                                      levels = c("censor", states)),
                       weight = weights)
    if (!is.null(strata)) data$g <- strata
    arguments <- list(formula = if (is.null(strata)) Surv(time, event) ~ 1
                      else Surv(time, event) ~ g,
                      data = data, weights = data$weight,
                      time0 = time0, timefix = FALSE, conf.int = .9, conf.type = conf_type)
    if (!is.null(istate)) arguments$istate <- istate
    if (!is.null(p0)) arguments$p0 <- p0
    if (!is.null(start_time)) arguments$start.time <- start_time
    fit <- do.call(survfit, arguments)
    matrices <- c(n_risk = "n.risk", n_event = "n.event", n_censor = "n.censor",
                  n_transition = "n.transition", pstate = "pstate", cumhaz = "cumhaz",
                  std_err = "std.err", std_chaz = "std.chaz", std_auc = "std.auc",
                  lower = "lower", upper = "upper", p0 = "p0")
    expected <- lapply(matrices, function(component) rows(fit[[component]]))
    expected$time <- I(fit$time)
    expected$n <- I(unname(fit$n))
    expected$n_id <- I(unname(fit$n.id))
    expected$states <- I(fit$states)
    expected$hazard_names <- I(colnames(fit$cumhaz))
    expected$t0 <- fit$t0
    cases[[length(cases) + 1L]] <<- list(name = name,
        input = list(time = I(time), state = I(state), states = I(states),
                     weights = I(weights), istate = if (is.null(istate)) NULL else I(istate),
                     strata = if (is.null(strata)) NULL else I(strata)),
        options = list(p0 = if (is.null(p0)) NULL else I(p0), start_time = start_time,
                       time0 = time0, timefix = FALSE, conf_int = .9, conf_type = conf_type),
        expected = expected)
}

time <- c(1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8)
state <- c(1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0)
weight <- c(.5, 1.75, .25, 2, 1.5, .75, 0, 3, 1, .5, 2.5, .8)
add("weighted_tied_competing_risks", time, state, weights = weight)
add("grouped_weighted_competing_risks", time, state, weights = weight,
    strata = rep(c(7, -3), 6), conf_type = "plain")
add("fixed_initial_distribution", time, state, weights = weight,
    p0 = c(.2, .5, .3), conf_type = "logit")
add("conditional_at_tied_event", time, state, weights = weight, start_time = 2)
add("conditional_at_tied_event_time0", time, state, weights = weight,
    start_time = 2, time0 = TRUE)
add("conditional_between_events_time0", time, state, weights = weight,
    start_time = 2.5, time0 = TRUE, conf_type = "log-log")
add("source_self_events", c(1, 1, 2, 3, 4, 4, 5, 6), c(1, 2, 0, 3, 1, 2, 3, 0),
    states = c("origin", "a", "b"), weights = c(.5, 1.5, 2, 1, .25, 1, 2, .75),
    istate = rep("origin", 8))
add("source_not_first_and_self_events", c(1, 1, 2, 3, 4, 4, 5, 6),
    c(2, 1, 0, 3, 2, 1, 3, 0), states = c("a", "b", "c"),
    weights = c(.5, 1.5, 2, 1, .25, 1, 2, .75), istate = rep("b", 8))
add("complete_absorption", c(1, 2, 2, 3, 4, 4), c(1, 2, 1, 2, 1, 2),
    weights = c(.5, 1, 2, 1.5, 1.25, 2.75))
add("grouped_single_destination_absorption", rep(1:4, 2), c(rep(1, 4), rep(2, 4)),
    weights = rep(c(.375, 3, .25, 1.875), 2), strata = rep(c(7, -3), each = 4),
    conf_type = "log-log")
add("single_destination_zero_weight_competitor", c(1:4, 1), c(rep(1, 4), 2),
    weights = c(.375, 3, .25, 1.875, 0), conf_type = "log-log")
add("zero_weight_event_tail", c(1, 2, 3), c(1, 2, 1), weights = c(1, 0, 0))
add("zero_weight_censor_tail", c(1, 2, 3), c(1, 0, 0), weights = c(1, 0, 0))
add("all_zero_weights", c(1, 2, 3), c(1, 0, 2), weights = c(0, 0, 0))

reference <- list(metadata = list(generator = "scripts/generate_aj_variance_reference.R",
    r_version = as.character(getRversion()),
    survival_version = as.character(packageVersion("survival"))), cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null")
cat(length(cases), "AJ uncertainty cases written to", output, "\n")
