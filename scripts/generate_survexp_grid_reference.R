#!/usr/bin/env Rscript
# Exact requested-time selection through the rate-table survexp path.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/survexp_grid_reference.json"

table <- array(c(.01, .035, .08), 3, dimnames = list(age = c("0", "5", "10")))
attr(table, "type") <- 2L
attr(table, "cutpoints") <- list(c(0, 5, 10))
class(table) <- "ratetable"
data <- data.frame(time = c(0, 1.5, 2, 6, 7, 9), age = c(0, 8, 3, 11, 0, 7),
                   g = c("b", "a", "b", "a", "b", "a"))
# Hex strings preserve adjacent doubles and the sign of zero through JSON.
hex <- function(x) I(sprintf("%a", x))
near_two <- 2 + 2 * .Machine$double.eps
requested <- c(-0, 0, 2, 2, near_two, 4, 4, 8, 8)
cases <- list()
add <- function(name, formula = time ~ g, times = requested, method = NULL,
                conditional = FALSE, scale = 2.5) {
    arguments <- list(formula = formula, data = data, ratetable = table,
                      conditional = conditional, scale = scale)
    if (!is.null(times)) arguments$times <- times
    if (!is.null(method)) arguments$method <- method
    fit <- do.call(survexp, arguments)
    rows <- function(x) {
        matrix <- matrix(as.numeric(x), nrow = length(fit$time), ncol = 2)
        lapply(seq_len(nrow(matrix)), function(i) I(matrix[i, ]))
    }
    cases[[length(cases) + 1L]] <<- list(name = name,
        formula = paste(deparse(formula), collapse = ""),
        times = if (is.null(times)) NULL else hex(times),
        method = method, conditional = conditional, scale = scale,
        expected = list(time = hex(fit$time), surv = rows(fit$surv),
                        n_risk = rows(fit$n.risk), method = fit$method,
                        labels = I(if (is.matrix(fit$surv)) colnames(fit$surv)
                                   else names(fit$surv))))
}
add("grouped_cohort_duplicates")
add("grouped_conditional_duplicates", conditional = TRUE)
add("explicit_conditional_duplicates", method = "conditional")
add("explicit_ederer_duplicates", method = "ederer")
add("implicit_followup_grid", times = NULL)
add("singleton_requested_time", times = 4)
# R's no-response caller matches against the original grid even though the
# fit deduplicates it. Use a unique grid here; duplicate grids have a separate
# analytical regression in Python because R can fail with out-of-bounds rows.
add("no_response_unique_grid", formula = ~ g,
    times = c(-0, 2, near_two, 4, 8), scale = 1.25)

reference <- list(metadata = list(generator = "scripts/generate_survexp_grid_reference.R",
    r_version = as.character(getRversion()),
    survival_version = as.character(packageVersion("survival"))),
    table = list(dims = I(dim(table)), dimid = "age",
                 dimnames = list(I(c("0", "5", "10"))),
                 cutpoints = list(I(c(0, 5, 10))), types = I(2),
                 rates = I(as.numeric(table))), data = data, cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = NA, pretty = TRUE,
           na = "null", null = "null", dataframe = "columns")
cat(length(cases), "requested-time cases written to", output, "\n")
