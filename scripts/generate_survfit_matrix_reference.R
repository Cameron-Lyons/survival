#!/usr/bin/env Rscript
# Rscript scripts/generate_survfit_matrix_reference.R [output.json]
# No random data or changes to the main survival 3.8-11 fixture corpus.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(expm))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else
    "python/tests/fixtures/survfit_matrix_reference.json"

d <- data.frame(time = rep(c(1, 2, 2, 3, 4, 5, 6, 7), 2),
                status = rep(c(1, 0, 1, 1, 0, 1, 0, 1), 2),
                g = rep(c("a", "b"), each = 8),
                x = rep(c(-1, 1, 0, 2, -2, 1, -1, 0), 2))
d$time[9:16] <- d$time[9:16] + .25
frames <- list(d, transform(d, time = time + .5, status = 1 - status),
               transform(d, time = time + 1, status = rev(status)))
make_matrix <- function(curves) {
    m <- matrix(vector("list", 9), 3, 3)
    m[1, 2] <- curves[1]
    m[1, 3] <- curves[2]
    m[2, 3] <- curves[3]
    m
}
km <- lapply(frames, function(d) survfit(Surv(time, status) ~ g, data = d))
cox <- lapply(frames, function(d) {
    fit <- coxph(Surv(time, status) ~ x + strata(g), data = d)
    survfit(fit, newdata = data.frame(x = c(-.5, 1)))
})
clean_curve <- function(x) list(
    n = unname(x$n), time = x$time, n_risk = x$n.risk, n_event = x$n.event,
    n_censor = x$n.censor, surv = unname(x$surv), cumhaz = unname(x$cumhaz),
    strata = as.list(x$strata), type = x$type)
clean_fit <- function(x) list(
    n = unname(x$n), time = x$time, n_risk = unname(x$n.risk),
    n_event = unname(x$n.event), pstate = unname(x$pstate),
    p0 = unname(x$p0), states = as.character(x$states), strata = as.list(x$strata))
cases <- list()
for (kind in c("km", "cox")) {
    curves <- if (kind == "km") km else cox
    m <- make_matrix(curves)
    for (method in c("discrete", "matexp")) {
        for (start in c(0, 2.5)) {
            for (initial in c("vector", "matrix")) {
                p0 <- if (initial == "vector") c(healthy = .8, ill = .2, dead = 0) else {
                    matrix(rep(c(.7, .1, .2, .4, .6, 0), if (kind == "km") 1 else 2),
                           ncol = 3, byrow = TRUE)
                }
                fit <- survfit(m, p0 = p0, method = method, start.time = start)
                cases[[length(cases) + 1]] <- list(
                    name = paste(kind, method, start, initial, sep = "_"),
                    kind = kind, method = method, start = start, initial = initial,
                    expected = clean_fit(fit))
            }
        }
    }
}
reference <- list(
    metadata = list(generator = "scripts/generate_survfit_matrix_reference.R",
                    r_version = as.character(getRversion()),
                    survival_version = as.character(packageVersion("survival")),
                    expm_version = as.character(packageVersion("expm"))),
    data = frames, curves = list(km = lapply(km, clean_curve), cox = lapply(cox, clean_curve)),
    cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE, dataframe = "columns")
cat(length(cases), "survfit.matrix cases written to", output, "\n")
