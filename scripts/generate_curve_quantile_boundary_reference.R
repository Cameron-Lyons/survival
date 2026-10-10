#!/usr/bin/env Rscript
# Independent stock-survival curve quantiles and probability boundaries.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/curve_quantile_boundary_reference.json"
i <- 1:30
data <- data.frame(time = as.numeric(i %% 11 + 1), status = as.integer(i %% 4 != 0),
    start = as.numeric((i %% 3) / 4), x = cos(i * .47), g = rep(c("a", "b"), 15),
    weight = .5 + (i %% 4) / 2, zero = 0, id = i)
data$lo <- data$time
data$hi <- data$time + .5
data$lo[i %% 4 == 1] <- NA
data$hi[i %% 4 == 2] <- NA
data$hi[i %% 4 == 0] <- data$lo[i %% 4 == 0]
data$event <- factor(ifelse(data$status == 0, "censor", ifelse(i %% 2 == 0, "a", "b")),
                     levels = c("censor", "a", "b"))
newdata <- data.frame(x = c(.2, .7, -.4), g = c("a", "b", "a"))
specs <- list(
    km_right = list(kind = "km", formula = "Surv(time,status)~1"),
    km_groups = list(kind = "km", formula = "Surv(time,status)~g"),
    km_counting = list(kind = "km", formula = "Surv(start,time,status)~g"),
    km_none = list(kind = "km", formula = "Surv(time,status)~g", conf_type = "none"),
    km_plain = list(kind = "km", formula = "Surv(time,status)~g", conf_type = "plain"),
    km_loglog = list(kind = "km", formula = "Surv(time,status)~g", conf_type = "log-log"),
    km_censor = list(kind = "km", formula = "Surv(time,zero)~g"),
    turnbull = list(kind = "km", formula = "Surv(lo,hi,type='interval2')~g"),
    cox_one = list(kind = "cox", formula = "Surv(time,status)~x", nd_rows = I(1L)),
    cox_many = list(kind = "cox", formula = "Surv(time,status)~x", nd_rows = I(1:3)),
    cox_groups = list(kind = "cox", formula = "Surv(time,status)~x+strata(g)", nd_rows = I(1:3)),
    cox_counting = list(kind = "cox", formula = "Surv(start,time,status)~x+strata(g)", nd_rows = I(1:3)),
    cox_cond = list(kind = "cox", formula = "Surv(time,status)~x+strata(g)", nd_rows = I(1:3), start_time = 3),
    aj = list(kind = "aj", formula = "Surv(time,event)~g"))
fits <- lapply(specs, function(spec) {
    if (spec$kind == "cox") {
        model <- coxph(as.formula(spec$formula), data, weights = weight)
        arguments <- list(formula = model, newdata = newdata[spec$nd_rows,,drop = FALSE])
        if (!is.null(spec$start_time)) arguments$start.time <- spec$start_time
    } else {
        arguments <- list(formula = as.formula(spec$formula), data = data, weights = data$weight)
        if (!is.null(spec$conf_type)) arguments$conf.type <- spec$conf_type
        if (spec$kind == "aj") arguments$id <- data$id
    }
    do.call(survfit, arguments)
})
encode_numbers <- function(values) {
    lapply(values, function(value) {
        if (is.nan(value)) return("NaN")
        if (is.na(value)) return("NA")
        if (is.infinite(value)) return(if (value > 0) "Inf" else "-Inf")
        value
    })
}
probability_spec <- function(value) {
    if (is.null(value)) return(list(kind = "null", values = NULL))
    if (is.factor(value)) return(list(kind = "factor", values = I(as.character(value))))
    list(kind = if (is.logical(value)) "logical" else if (is.character(value)) "character" else "numeric",
         values = if (is.numeric(value)) encode_numbers(value) else I(value))
}
encode <- function(value, curves, width) {
    if (is.null(value)) return(NULL)
    matrix_value <- matrix(as.numeric(value), nrow = curves, ncol = width)
    lapply(seq_len(curves), function(row) I(unname(matrix_value[row,])))
}
cases <- list()
record <- function(name, p, tolerance, conf, scale, method = "quantile", tag) {
    fit <- fits[[name]]
    curves <- if (is.null(fit$strata)) 1L else length(fit$strata)
    if (is.matrix(fit$surv)) curves <- curves * ncol(fit$surv)
    arguments <- list(x = fit, scale = scale)
    if (!is.null(tolerance)) arguments$tolerance <- tolerance
    if (method == "quantile") {
        arguments$probs <- p
        # Keep an explicit NULL probability argument instead of deleting it.
        if (is.null(p)) arguments["probs"] <- list(NULL)
        arguments$conf.int <- conf
    }
    value <- tryCatch(suppressWarnings({
        result <- do.call(get(method), arguments)
        if (!is.list(result)) result <- list(quantile = result)
        lapply(result, encode, curves = curves, width = if (method == "median") 1L else length(p))
    }), error = function(error) list(error = conditionMessage(error)))
    cases[[length(cases) + 1L]] <<- list(name = paste(name, tag, length(cases) + 1L, sep = "/"),
        fit = name, method = method, probs = probability_spec(p),
        tolerance = if (is.null(tolerance)) NULL else encode_numbers(tolerance)[[1L]],
        conf_int = conf, scale = scale, expected = value)
}
probabilities <- list(c(0, .25, .5, .75, 1), c(.5, .5 + 1e-9, .5 - 1e-9), numeric(0), 0, 1, .5)
tolerances <- list(NULL, 0, -.01, 1e-16, .125, .5, 1, Inf, -Inf)
invalid <- list(TRUE, c(FALSE, TRUE), logical(0), "0.5", c(".2", ".8"), character(0),
    c(.2, NA_real_, .8), NaN, Inf, -Inf, NULL, -.01, 1.01, factor(c(.2, .8)))
for (name in setdiff(names(fits), "aj")) {
    for (p in probabilities) for (tolerance in tolerances) for (conf in c(FALSE, TRUE)) {
        for (scale in c(1, 2)) record(name, p, tolerance, conf, scale, tag = "numeric")
    }
    for (tolerance in tolerances) for (scale in c(1, 2)) {
        record(name, .5, tolerance, FALSE, scale, method = "median", tag = "median")
    }
    for (p in invalid) record(name, p, NULL, TRUE, 1, tag = "invalid_probability")
    record(name, c(0, .5, 1), NaN, TRUE, 1, tag = "invalid_tolerance")
}
for (p in list(c(0, .5, 1), numeric(0))) {
    record("aj", p, NULL, TRUE, 1, tag = "undefined_multistate")
}
record("aj", .5, NULL, FALSE, 1, method = "median", tag = "undefined_multistate_median")
dtype_inputs <- list(
    numeric = list(numeric(0), c(0, 1)),
    integer = list(integer(0), 0:1),
    logical = list(logical(0), c(FALSE, TRUE)),
    character = list(character(0), c("0", "1")),
    factor = list(factor(character(0), levels = c("0", "1")), factor(c("0", "1"))),
    complex = list(complex(0), as.complex(0:1)),
    date = list(as.Date(character(0)), as.Date(c("1970-01-01", "1970-01-02"))),
    duration = list(as.difftime(numeric(0), units = "secs"),
                    as.difftime(0:1, units = "secs")),
    raw = list(raw(0), as.raw(0:1)),
    missing = list(logical(0), c(NA_real_, NA_real_)))
dtype_cases <- list()
for (kind in names(dtype_inputs)) for (width in seq_len(2)) {
    p <- dtype_inputs[[kind]][[width]]
    value <- tryCatch(suppressWarnings({
        result <- quantile(fits$km_right, p)
        lapply(result, encode, curves = 1L, width = length(p))
    }), error = function(error) list(error = conditionMessage(error)))
    dtype_cases[[paste(kind, if (width == 1L) "empty" else "nonempty", sep = "/")]] <-
        list(r_type = typeof(p), r_class = I(class(p)), expected = value)
}
scale_cases <- list()
scales <- list(one = 1, two = 2, negative_one = -1, negative_two = -2,
    zero = 0, negative_zero = -0, infinity = Inf, negative_infinity = -Inf,
    missing = NA_real_, nan = NaN, true = TRUE, false = FALSE)
encode_scale_result <- function(value, curves, width) {
    matrix_value <- matrix(as.numeric(value), nrow = curves, ncol = width)
    lapply(seq_len(curves), function(row) lapply(matrix_value[row, ], function(number) {
        if (is.na(number)) return(NULL)
        if (is.infinite(number)) return(if (number > 0) "Inf" else "-Inf")
        if (number == 0 && 1 / number < 0) return("-0")
        number
    }))
}
record_scale <- function(name, p, scale_name, conf, method = "quantile") {
    fit <- fits[[name]]
    scale <- scales[[scale_name]]
    curves <- if (is.null(fit$strata)) 1L else length(fit$strata)
    if (is.matrix(fit$surv)) curves <- curves * ncol(fit$surv)
    arguments <- list(x = fit, scale = scale)
    if (method == "quantile") {
        arguments$probs <- p
        arguments$conf.int <- conf
    }
    value <- suppressWarnings(do.call(get(method), arguments))
    if (!is.list(value)) value <- list(quantile = value)
    scale_cases[[length(scale_cases) + 1L]] <<- list(
        name = paste(name, method, scale_name, conf, length(p), sep = "/"),
        fit = name, method = method, probs = encode_numbers(p),
        scale_kind = if (is.logical(scale)) "logical" else "numeric",
        scale = if (scale_name == "negative_zero") "-0" else encode_numbers(scale)[[1L]],
        conf_int = conf,
        expected = lapply(value, encode_scale_result, curves = curves, width = length(p)))
}
for (name in setdiff(names(fits), "aj")) for (scale_name in names(scales)) {
    for (p in list(c(0, .5, 1), c(.25, .5, .75), numeric(0))) {
        for (conf in c(FALSE, TRUE)) record_scale(name, p, scale_name, conf)
    }
    record_scale(name, .5, scale_name, FALSE, method = "median")
}
serialize <- function(value) toJSON(value, auto_unbox = TRUE, digits = NA, na = "null", null = "null")
header <- serialize(list(r_version = R.version.string,
    survival_version = as.character(packageVersion("survival")),
    generator = "scripts/generate_curve_quantile_boundary_reference.R",
    data = lapply(data, function(value) I(if (is.factor(value)) as.character(value) else value)),
    event_levels = I(levels(data$event)), newdata = lapply(newdata, I), fits = specs,
    dtype_cases = dtype_cases, scale_cases = scale_cases))
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
    vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),
        if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes = TRUE)
cat(length(cases) + length(dtype_cases) + length(scale_cases),
    "curve quantile boundary references written\n")
