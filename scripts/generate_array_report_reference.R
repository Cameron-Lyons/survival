#!/usr/bin/env Rscript
# Rebuild raw-response and rate-array print references with R survival.
suppressPackageStartupMessages(library(survival))

suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)

output <- if (length(args)) args[[1]] else "python/tests/fixtures/array_report_reference.json"

options(width = 80, digits = 7)

objects <- list()

specs <- list()

cases <- list()

response <- function(name, arguments, kind = "Surv") {
    objects[[name]] <<- suppressWarnings(do.call(get(kind), arguments))
    encoded <- lapply(arguments, function(x) if (is.factor(x))
        list(values = I(as.character(x)), levels = I(levels(x)))
    else if (length(x) > 1)
        I(x)
    else x)
    specs[[name]] <<- list(kind = kind, arguments = encoded)
}

response("right", list(time = c(1.111111111, 2.22, 3, 4, 5, 100), event = c(1, 0, 1, 0, 1, 0)))

response("counting", list(time = c(0, 1, 0, 3), time2 = c(1, 4, 5, 6), event = c(1, 0, 1, 0)))

response("left", list(time = c(0.001, 10, Inf, NA), event = c(0, 1, 1, NA), type = "left"))

response("interval", list(time = c(1, 2, 3, 4, 5, NA), time2 = c(2, 3, Inf, 4, 8, 5), type = "interval2"))

response("missing", list(time = c(1, NA, 3, 4), event = c(1, 0, NA, 1)))

response("large", list(time = c(1:14, 1000), event = c(rep(1, 14), 0)))

response("states", list(time = 1:5, event = factor(c("none", "病気", "recovered", "病気", "none"),
    levels = c("none", "病気", "recovered"))))

response("escaped", list(time = 1:5, event = factor(c("quote\"d", "line\nnew", "tab\tnew", "slash\\end",
    "日本"), levels = c("quote\"d", "line\nnew", "tab\tnew", "slash\\end", "日本"))))

response("timeline", list(time = c(0, 2, 4, 0, 3), event = c(0, 1, 0, 0, 1)), "Surv2")

response("timeline_states", list(time = c(0, 2, 4, 0, 3), event = factor(c("none", "病気", "recovered",
    "none", "病気"), levels = c("none", "病気", "recovered")), repeated = TRUE), "Surv2")

rate <- function(name, dims, labels, ids, rates) {
    x <- array(rates, dims, dimnames = setNames(labels, ids))
    attr(x, "type") <- rep(1L, length(dims))
    attr(x, "cutpoints") <- rep(list(NULL), length(dims))
    class(x) <- "ratetable"
    objects[[name]] <<- x
    specs[[name]] <<- list(kind = "ratetable", dims = I(dims), dimid = I(ids), dimnames = lapply(labels,
        I), rates = I(as.numeric(x)))
}

rate("one", 3, list(c("first", "second", "third")), "age", 1:3 * 0.000123456789)

rate("matrix", c(2, 3), list(c("first", "longer row"), c("male", "female", "other")), c("age", "sex"),
    1:6 * 0.000123456789)

rate("array", c(2, 3, 2), list(c("r1", "r2"), c("one", "two", "three"), c("a", "b")), c("age", "sex",
    "year"), 1:12 * 0.1234567)

rate("four", c(1, 2, 2, 2), list("row", c("M", "F"), c("one", "two"), c("A", "B")), c("age", "sex",
    "race", "year"), 1:8 * 0.000123456789)

rate("precision", c(3, 2, 2), list(c("x", "y", "z"), c("first", "second"), c("early", "late")),
    c("age", "sex", "year"), c(1e-09, 2e-08, 9.99999999, 0, 1e-12, 3, 1000, 1e-09, 3.14159265, 1,
        10, 1e-05))

rate("unicode", c(2, 2), list(c("a\nb", "日本"), c("男", "女")), c("年齢", "性別"), c(0.1,
    0.00012, 1.234, 12))

rate("labels", c(2, 2), list(c("x", "y"), c("a\"b", "c\\d")), c("row", "column"), c(0.01, 0.002,
    100, 200))

for (name in c("survexp.us", "survexp.usr", "survexp.mn")) {
    objects[[name]] <- get(name)
    specs[[name]] <- list(kind = "builtin", name = name)
}

add <- function(name, object, arguments = list(), width = 80, row_names = NULL) {
    x <- objects[[object]]
    if (!is.null(row_names))
        rownames(x) <- row_names
    old <- options(width = width)
    lines <- capture.output(do.call(print, c(list(x), arguments)))
    options(old)
    cases[[length(cases) + 1L]] <<- list(name = name, object = object, arguments = arguments, width = width,
        row_names = if (is.null(row_names)) NULL else I(row_names), labels = if (inherits(x, c("Surv",
            "Surv2"))) I(as.character(x)) else NULL, lines = I(sub("[[:blank:]]+$", "", lines)))
}

for (name in names(objects)) add(name, name, if (grepl("survexp", name)) list(max = 6) else list())

for (width in c(10, 20, 25, 30, 80)) add(paste0("wrap_", width), "large", list(max = 12), width)

add("right_quoted", "right", list(quote = TRUE), 30)

add("right_aligned", "right", list(right = TRUE), 30)

add("names", "right", list(), 30, paste0("row", 1:6))

add("names_quoted", "right", list(quote = TRUE, max = 3), 30, c("first", "long", "日本", "other",
    "x", "y"))

add("escaped_quoted", "escaped", list(quote = TRUE), 50)

for (maximum in c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)) add(paste0("array_max_", maximum),
    "array", list(max = maximum), 30)

for (maximum in c(0, 1, 2, 3, 4, 5, 6)) add(paste0("matrix_max_", maximum), "matrix", list(max = maximum),
    30)

for (maximum in c(0, 1, 2, 3)) add(paste0("one_max_", maximum), "one", list(max = maximum), 30)

for (maximum in c(0, 1, 4, 5)) add(paste0("response_max_", maximum), "right", list(max = maximum),
    30)

add("precise", "precision", list(digits = 12, max = 7), 35)

add("rough", "precision", list(digits = 2), 40)

add("four_narrow", "four", list(max = 3), 15)

write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
    objects = specs, cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "string",
    null = "null")
