#!/usr/bin/env Rscript
# Independent stock-survival AFT predictions with missing probabilities.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/aft_prediction_probability_reference.json"

i <- 1:32
data <- data.frame(x = cos(i * .43), status = as.integer(i %% 5 != 0),
                   s = rep(c("one", "two"), 16))
data$time <- exp(.9 + .45 * data$x + sin(i * .61) * .5 + (i %% 5) / 10)
newdata <- data.frame(x = c(0, .8, -.4), s = c("one", "two", "one"))
probabilities <- list(mixed = c(.2, NA_real_, .8), missing = NA_real_,
                      invalid = c(NaN, -.1, 1.1, Inf, -Inf))
encode <- function(value) {
    if (is.list(value)) return(lapply(value, encode))
    values <- as.numeric(value)
    encoded <- lapply(values, function(x) {
        if (is.nan(x)) return("NaN")
        if (is.na(x)) return("NA")
        if (is.infinite(x)) return(if (x > 0) "Inf" else "-Inf")
        x
    })
    list(values = encoded, dim = if (is.null(dim(value))) NULL else I(dim(value)))
}
fits <- list()
cases <- list()
queries <- list()
for (family in c("weibull", "lognormal", "loglogistic", "gaussian", "logistic",
                 "t", "exponential", "rayleigh")) {
    for (routine in c("dsurvreg", "psurvreg", "qsurvreg")) {
        value <- get(routine)(c(.2, NA_real_, .8), mean = .3, scale = .8,
                              distribution = family, parms = if (family == "t") 5 else NULL)
        queries[[length(queries) + 1L]] <- list(name = paste(routine, family, sep = "/"),
            routine = routine, distribution = family, parms = if (family == "t") 5 else NULL,
            mean = .3, scale = .8, values = encode(c(.2, NA_real_, .8))$values,
            expected = encode(value))
    }
    modes <- if (family %in% c("exponential", "rayleigh")) "fixed" else
        c("estimated", "fixed", "stratified")
    for (mode in modes) {
        formula <- paste("Surv(time,status) ~ x", if (mode == "stratified") "+ strata(s)" else "")
        scale <- if (mode == "fixed" && !family %in% c("exponential", "rayleigh")) .7 else 0
        parms <- if (family == "t") 5 else NULL
        arguments <- list(formula = as.formula(formula), data = data, dist = family, parms = parms)
        if (!family %in% c("exponential", "rayleigh")) arguments$scale <- scale
        fit <- do.call(survreg, arguments)
        name <- paste(family, mode, sep = "/")
        fits[[name]] <- list(formula = formula, dist = family, scale = scale, parms = parms)
        for (type in c("quantile", "uquantile")) {
            for (kind in names(probabilities)) for (errors in c(FALSE, TRUE)) {
                p <- probabilities[[kind]]
                value <- suppressWarnings(predict(fit, newdata, type = type, p = p, se.fit = errors))
                cases[[length(cases) + 1L]] <- list(
                    name = paste(name, type, kind, errors, sep = "/"), fit = name,
                    type = type, p = encode(p)$values, se_fit = errors, expected = encode(value))
            }
        }
        for (type in c("response", "link", "lp", "linear", "terms")) {
            for (errors in c(FALSE, TRUE)) {
                value <- predict(fit, newdata, type = type, p = "unused", se.fit = errors)
                cases[[length(cases) + 1L]] <- list(
                    name = paste(name, type, "ignored", errors, sep = "/"), fit = name,
                    type = type, p = "unused", se_fit = errors, expected = encode(value))
            }
        }
    }
}
serialize <- function(value) toJSON(value, auto_unbox = TRUE, digits = NA, na = "null", null = "null")
header <- serialize(list(r_version = R.version.string,
    survival_version = as.character(packageVersion("survival")),
    generator = "scripts/generate_aft_prediction_probability_reference.R",
    data = lapply(data, I), newdata = lapply(newdata, I), fits = fits, queries = queries))
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
    vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),
        if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes = TRUE)
cat(length(cases), "AFT prediction and", length(queries), "nullable distribution references written\n")
