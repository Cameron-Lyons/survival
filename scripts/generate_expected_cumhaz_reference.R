#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly=TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/expected_cumhaz_reference.json"
number <- function(value) {
    if (is.na(value)) return("NaN")
    if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
    value
}
vector <- function(values) lapply(unname(values), number)
matrix_rows <- function(values) {
    lapply(seq_len(nrow(values)), function(row) vector(values[row, ]))
}

values <- c(1, .5, 0, -0, -1, NaN, NA_real_, Inf, -Inf, 2)
synthetic <- list(
    list(name="vector", surv=vector(values),
         cumhaz=vector(suppressWarnings(-log(values)))),
    list(name="matrix", surv=matrix_rows(matrix(values, ncol=2)),
         cumhaz=matrix_rows(suppressWarnings(-log(matrix(values, ncol=2)))))
)

data <- na.omit(lung[, c("time", "status", "ph.ecog", "age", "sex")])
model <- coxph(Surv(time, status) ~ age + sex, data)
curves <- lapply(c("conditional", "hakulinen"), function(method) {
    fit <- survexp(Surv(time, status) ~ ph.ecog, data, ratetable=model,
                   method=method, times=c(100, 300, 500))
    list(method=method, time=I(fit$time), labels=I(colnames(fit$surv)),
         surv=matrix_rows(fit$surv), cumhaz=matrix_rows(-log(fit$surv)))
})
reference <- list(metadata=list(generator="scripts/generate_expected_cumhaz_reference.R",
    r_version=as.character(getRversion()), survival_version=as.character(packageVersion("survival"))),
    synthetic=synthetic, curves=curves)
dir.create(dirname(output), recursive=TRUE, showWarnings=FALSE)
write_json(reference, output, auto_unbox=TRUE, digits=17, pretty=TRUE,
           na="null", null="null")
cat("Expected-survival cumulative hazards written to", output, "\n")
