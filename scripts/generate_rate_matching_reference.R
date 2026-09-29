#!/usr/bin/env Rscript
# Rate positions, cutpoints, summaries and validation errors from survival.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/rate_matching_reference.json"
options(digits = 7)
tables <- list()
specs <- list()
cases <- list()
for (name in c("survexp.us", "survexp.usr", "survexp.mn")) {
    tables[[name]] <- get(name)
    specs[[name]] <- list(builtin = name)
}
custom <- function(name, ids, levels, types, cuts) {
    dims <- lengths(levels)
    x <- array(rep(.01, prod(dims)), dims, dimnames = setNames(levels, ids))
    attr(x, "type") <- types
    attr(x, "cutpoints") <- cuts
    class(x) <- "ratetable"
    tables[[name]] <<- x
    specs[[name]] <<- list(dims = I(dims), dimid = I(ids), dimnames = lapply(levels, I),
        types = I(types), cutpoints = lapply(cuts, function(x) if (is.null(x)) NULL else I(x)),
        rates = I(as.numeric(x)))
}
custom("prefix", c("age", "group"), list(c("0", "10"), c("male", "man", "female", "fem")),
       c(2, 1), list(c(0, 10), NULL))
custom("duplicate", "group", list(c("Male", "MALE")), 1, list(NULL))
custom("duplicate_axes", c("group", "group"), list("a", "a"), c(1, 1), list(NULL, NULL))
custom("unicode", "group", list(c("女性", "男性")), 1, list(NULL))
encode <- function(x) {
    if (is.factor(x)) return(list(kind = "factor", values = I(as.character(x)), levels = I(levels(x))))
    if (inherits(x, "Date")) return(list(kind = "date", values = I(as.character(x))))
    if (inherits(x, "POSIXct")) return(list(kind = "datetime", values = I(format(x, format = "%Y-%m-%dT%H:%M:%S%z"))))
    if (inherits(x, "difftime")) return(list(kind = "timedelta", values = I(as.numeric(x, units = "days"))))
    list(kind = "numeric", values = I(x))
}
add <- function(name, table, data) {
    warnings <- character()
    result <- withCallingHandlers(tryCatch(match.ratetable(data, tables[[table]]),
        error = function(e) list(error = conditionMessage(e))),
        warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") })
    expected <- if (!is.null(result$error)) result else list(
        r = lapply(seq_len(nrow(result$R)), function(i) I(as.numeric(result$R[i, ]))),
        cutpoints = lapply(result$cutpoints, function(x) if (is.null(x)) NULL else I(as.numeric(x))),
        summary = if (is.null(result$summ)) NULL else result$summ)
    cases[[length(cases) + 1L]] <<- list(name = name, table = table, data = lapply(data, encode),
        expected = expected, warnings = I(warnings))
}
dates <- as.Date(c("1996-01-01", "1996-07-15", "2002-03-01"))
base <- data.frame(year = dates, sex = c(1, 2, 1), age = c(50, 60.5, 70) * 365.25)
for (table in c("survexp.us", "survexp.mn")) add(table, table, base)
add("race", "survexp.usr", transform(base, race = factor(c("W", "b", "white"))))
add("factors", "survexp.us", transform(base, sex = factor(c("M", "f", "male"), levels = c("f", "M", "male"))))
add("dates_numeric", "survexp.us", transform(base, year = as.numeric(year)))
add("datetime", "survexp.us", transform(base, year = as.POSIXct(year, tz = "UTC")))
add("datetime_zone", "survexp.us", transform(base, year = as.POSIXct(
    c("1996-01-01 23:30:00", "1996-07-01 23:30:00", "1996-01-01 00:30:00"),
    tz = "America/Chicago")))
add("age_duration", "survexp.us", transform(base, age = as.difftime(age, units = "days")))
add("extra", "survexp.us", transform(base, irrelevant = c(NA, NA, NA)))
add("empty", "survexp.us", base[FALSE, ])
add("prefix_unique", "prefix", data.frame(group = factor(c("mal", "fem", "fema", "man")), age = c(2, 3, 12, 13)))
add("prefix_ambiguous", "prefix", data.frame(age = 1, group = factor("m")))
add("unused_ambiguous", "prefix", data.frame(age = 1, group = factor("male", levels = c("male", "m"))))
add("unused_missing", "survexp.us", transform(base, sex = factor(c("male", "female", "male"), levels = c("male", "female", "other"))))
add("factor_continuous", "survexp.us", transform(base, age = factor(age)))
add("date_continuous", "survexp.us", transform(base, age = dates))
add("date_factor", "survexp.us", transform(base, sex = dates))
add("factor_outside", "survexp.us", transform(base, sex = c(0, 1, 2)))
add("factor_fractional", "survexp.us", transform(base, sex = c(1, 1.5, 2)))
add("missing_dimension", "survexp.us", base[, c("age", "sex")])
add("duplicate", "duplicate", data.frame(group = factor("male")))
add("duplicate_axes", "duplicate_axes", data.frame(group = 1))
add("unicode", "unicode", data.frame(group = factor(c("女", "男性", "男"))))
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
    tables = specs, cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = NA,
    na = "null", null = "null")
