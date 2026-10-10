#!/usr/bin/env Rscript
# Independent stock-survival references for the entry reporting grid.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/aj_entry_reference.json"

data <- data.frame(
    start = c(0,3, 0,2, 1,4, 2, 0,4,6, 0,1.5,2.5, 1,5),
    stop = c(3,8, 2,5, 4,9, 7, 4,6,10, 1.5,2.5,7, 5,8),
    event = factor(c("a","b", "censor","censor", "censor","a", "b",
                     "a","censor","b", "censor","censor","censor", "a","censor"),
                   levels = c("censor","a","b")),
    id = c(1,1, 2,2, 3,3, 4, 5,5,5, 6,6,6, 7,7),
    group = c(rep("one",6),rep("two",9)),
    weight = c(1.5,1.5, .5,.5, 2,2, 1, .75,.75,.75, 2,2,2, 1.25,1.25)
)
encode <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.matrix(x)) return(lapply(seq_len(nrow(x)), function(i) I(unname(x[i,]))))
    I(unname(x))
}
encode_influence <- function(x) {
    if (!is.list(x)) x <- list(x)
    lapply(unname(x), function(curve) lapply(seq_len(dim(curve)[1]), function(i)
        lapply(seq_len(dim(curve)[2]), function(t) I(unname(curve[i,t,])))))
}
fields <- c("time", "n.risk", "n.event", "n.censor", "n.enter", "n.transition",
            "pstate", "cumhaz", "std.err", "std.chaz", "std.auc", "lower", "upper")
cases <- list()
for (grouped in c(FALSE, TRUE)) {
    for (weighted in c(FALSE, TRUE)) {
        for (entry in c(FALSE, TRUE)) {
            for (time0 in c(FALSE, TRUE)) {
                for (conditional in c(FALSE, TRUE)) {
                    for (shuffled in c(FALSE, TRUE)) {
                        rows <- if (shuffled) rev(seq_len(nrow(data))) else seq_len(nrow(data))
                        frame <- data[rows,]
                        formula <- as.formula(paste("Surv(start, stop, event) ~",
                                                    if (grouped) "group" else "1"))
                        arguments <- list(formula = formula, data = frame, id = frame$id,
                                          weights = if (weighted) frame$weight else rep(1,nrow(frame)),
                                          entry = entry, time0 = time0, influence = TRUE)
                        if (conditional) arguments$start.time <- 2
                        fit <- do.call(survival::survfit, arguments)
                        expected <- setNames(lapply(fields, function(field) encode(fit[[field]])),
                                             gsub("\\.", "_", fields))
                        expected$states <- I(fit$states)
                        expected$n <- I(unname(fit$n))
                        expected$n_id <- I(unname(fit$n.id))
                        expected$p0 <- if (is.matrix(fit$p0)) encode(fit$p0) else list(I(unname(fit$p0)))
                        expected$t0 <- fit$t0
                        expected["strata"] <- list(encode(fit$strata))
                        expected$influence_pstate <- encode_influence(fit$influence.pstate)
                        if (is.null(fit$counts)) {
                            expected["counts"] <- list(NULL)
                        } else {
                            ns <- length(fit$states)
                            nh <- ncol(fit$cumhaz)
                            counts <- list(n_risk=encode(fit$counts[,seq_len(ns),drop=FALSE]),
                                n_transition=encode(fit$counts[,ns+seq_len(nh),drop=FALSE]),
                                n_censor=encode(fit$counts[,ns+nh+seq_len(ns),drop=FALSE]),
                                n_enter=if (entry) encode(fit$counts[,2*ns+nh+seq_len(ns),drop=FALSE]) else NULL)
                            expected$counts <- counts
                        }
                        name <- paste(if (grouped) "groups" else "one",
                                      if (weighted) "weighted" else "unit",
                                      if (entry) "entry" else "stops",
                                      if (time0) "time0" else "no_time0",
                                      if (conditional) "conditional" else "full",
                                      if (shuffled) "shuffled" else "ordered", sep = "_")
                        cases[[length(cases)+1L]] <- list(name = name, grouped = grouped,
                            weighted = weighted, entry = entry, time0 = time0,
                            start_time = if (conditional) 2 else NULL,
                            rows = I(rows), expected = expected)
                    }
                }
            }
        }
    }
}
write_json(list(r_version = R.version.string,
    survival_version = as.character(packageVersion("survival")),
    generator = "scripts/generate_aj_entry_reference.R",
    data = lapply(data, function(x) I(if (is.factor(x)) as.character(x) else x)),
    event_levels = I(levels(data$event)), cases = cases),
    output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null", null = "null")
