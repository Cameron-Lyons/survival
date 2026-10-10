#!/usr/bin/env Rscript
# Independent stock survival values for population boundary ownership tests.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/population_iterable_reference.json"
snapshot <- function(value) {
    if (is.null(value)) return(NULL)
    if (inherits(value, "Surv")) return(list(kind = "surv", values = unname(as.matrix(value)),
                                            surv_type = attr(value, "type")))
    if (inherits(value, "tcut")) return(list(kind = "tcut", values = I(as.numeric(value)),
        cutpoints = I(attr(value, "cutpoints")), levels = I(attr(value, "labels"))))
    if (inherits(value, "Date")) return(list(kind = "date", values = I(as.character(value))))
    if (is.factor(value)) return(list(kind = "factor", values = I(as.character(value)),
        levels = I(levels(value)), codes = I(as.integer(value)), ordered = is.ordered(value)))
    if (is.matrix(value)) return(list(kind = "matrix", values = unname(value)))
    list(kind = "vector", values = I(unname(value)))
}
base <- list(time = c(2, 3, 4, 6, 5, 7, 8, 9), start = c(0, 1, 0, 2, 1, 3, 2, 1),
    status = c(1, 0, 1, 1, 0, 1, 0, 1), agebase = c(1, 3, 7, 9, 4, 6, 8, 2),
    sex = factor(rep(c("male", "female"), 4), levels = c("female", "male")),
    grp = factor(rep(c("b", "a"), 4), levels = c("b", "a", "empty")),
    weight = c(1, 2, .5, 1, 1.5, 2, .5, 1), score = c(0, 1, 2, 1, 2, 0, 1, 2),
    unused = c("never", "read"))
base$Y_event <- cbind(base$time, c(1, 2, 0, 3, 1, 1, 0, 2))
base$Y_interval <- cbind(base$start, base$time)
base$zc <- factor(c("c", "a", "b", "a", "b", "c", "a", "b"), levels = c("c", "a", "b"))
table <- structure(c(.01, .02, .03, .005, .01, .015), dim = c(3L, 2L),
    dimnames = list(age = c("0", "5", "10"), sex = c("male", "female")),
    dimid = c("age", "sex"), type = c(2L, 1L), factor = c(0L, 1L),
    cutpoints = list(c(0, 5, 10), NULL), class = "ratetable")
stopifnot(is.ratetable(table))
training <- data.frame(time = c(2, 3, 4, 6, 7, 9, 10, 12),
    status = c(1, 0, 1, 1, 0, 1, 0, 1), z = c(1, 2, 3, 1, 2, 3, 1, 2))
cox <- coxph(Surv(time, status) ~ z, training)
training$zc <- factor(c("a", "b", "c", "b", "a", "c", "b", "a"), levels = c("a", "b", "c"))
cox_factor <- coxph(Surv(time, status) ~ zc, training)
cases <- list()
add <- function(name, function_name, formula, table_name = "none", options = list(), missing = FALSE,
                weights = FALSE, rmap_kind = "expression") {
    data <- base
    if (missing) {
        data$time[2] <- NA_real_
        data$agebase[3] <- NA_real_
        data$grp[5] <- NA
        data$Y_event[2, 1] <- NA_real_
        data$Y_interval[2, 2] <- NA_real_
    }
    if (table_name == "us") {
        data$age_days <- (data$agebase + 40) * 365.25
        data$entry <- rep(as.Date("2000-01-01"), length(data$time))
    }
    rate_vectors <- NULL
    if (rmap_kind == "vectors") {
        rate_age <- data$agebase + 1
        rate_sex <- data$sex
        rate_vectors <- list(age = snapshot(rate_age), sex = snapshot(rate_sex))
        data <- list(unused = seq_along(rate_age))
    }
    selection <- c(8L, 3L, 2L, 3L, 1L, 5L, 6L)
    arguments <- c(list(formula = as.formula(formula), data = data,
        subset = selection, na.action = na.exclude), options)
    if (identical(weights, "time")) arguments$weights <- data$time
    else if (weights) arguments$weights <- data$weight
    if (table_name == "population") {
        arguments$ratetable <- table
        arguments$rmap <- if (rmap_kind == "vectors") quote(list(age = rate_age, sex = rate_sex)) else
            if (rmap_kind == "response") quote(list(age = time, sex = sex)) else
            quote(list(age = agebase + 1, sex = sex))
    } else if (table_name == "cox") {
        arguments$ratetable <- cox
        arguments$rmap <- quote(list(z = score + 1))
    } else if (table_name == "cox_factor") {
        arguments$ratetable <- cox_factor
        arguments$rmap <- quote(list(zc = zc))
    } else if (table_name == "us") {
        arguments$ratetable <- survexp.us
        arguments$rmap <- quote(list(age = age_days, sex = sex, year = entry))
    }
    captured <- character()
    run <- function(arguments) withCallingHandlers(do.call(function_name, arguments),
        warning = function(w) {captured <<- c(captured, conditionMessage(w)); invokeRestart("muffleWarning")})
    fit <- run(c(arguments, list(x = TRUE, y = TRUE)))
    if (is.list(fit)) {
        retained <- run(c(arguments, list(model = TRUE)))
        expected <- list(x = snapshot(fit$x), y = snapshot(fit$y),
            model = lapply(retained$model, snapshot), n = nrow(retained$model),
            na_action = I(as.integer(fit$na.action)))
        if (function_name == "pyears") {
            expected$n <- NULL
            for (field in c("pyears", "n", "event", "expected"))
                expected[field] <- list(if (is.null(fit[[field]])) NULL else I(as.numeric(fit[[field]])))
            for (field in c("offtable", "observations", "tcut")) expected[[field]] <- fit[[field]]
            expected$dim <- I(as.integer(dim(fit$pyears)))
            expected$dimnames <- lapply(dimnames(fit$pyears), I)
        } else {
            expected$time <- I(fit$time)
            expected$surv <- snapshot(fit$surv)
            expected["n_risk"] <- list(snapshot(fit$n.risk))
            expected["strata"] <- list(if (is.matrix(fit$surv)) I(colnames(fit$surv)) else NULL)
            expected$method <- fit$method
        }
    } else expected <- list(individual = I(unname(fit)))
    cases[[length(cases) + 1L]] <<- list(name = paste0(name, if (missing) "_missing" else "_complete"),
        function_name = function_name, formula = formula, data = lapply(data, snapshot),
        table = table_name, options = options, weights = weights, subset = I(selection - 1L),
        rmap_kind = rmap_kind, rate_vectors = rate_vectors,
        na_action = "exclude", warnings = I(unique(captured)), expected = expected)
}
for (missing in c(FALSE, TRUE)) {
    add("pyears_right", "pyears", "Surv(time, status) ~ grp", options = list(scale = 1),
        missing = missing, weights = TRUE)
    add("pyears_matrix_events", "pyears", "Y_event ~ grp", options = list(scale = 1),
        missing = missing, weights = TRUE)
    add("pyears_matrix_entry", "pyears", "Y_interval ~ grp", "population", list(scale = 1),
        missing = missing, weights = TRUE)
    add("pyears_counting", "pyears", "Surv(start, time, status) ~ grp", "population", list(scale = 1),
        missing = missing, weights = TRUE)
    add("pyears_cut", "pyears", "time ~ cut(agebase, c(0, 5, 10, 20)) + grp", "population",
        list(scale = 1, expect = "pyears"), missing = missing, weights = TRUE)
    add("pyears_tcut", "pyears", "time ~ tcut(agebase, c(0, 5, 10, 20)) + grp",
        options = list(scale = 1), missing = missing, weights = TRUE)
    add("pyears_factor", "pyears", "time ~ factor(grp)", "population", list(scale = 1), missing)
    for (method in c("hakulinen", "conditional", "ederer"))
        add(paste0("survexp_", method), "survexp", "time ~ grp", "population",
            list(times = c(0, 2, 5, 10), method = method), missing)
    add("survexp_response_free", "survexp", "~ grp", "population", list(times = c(0, 2, 5, 10)), missing)
    add("survexp_cox", "survexp", "time ~ grp", "cox", list(times = c(2, 5, 10)), missing, TRUE)
    add("survexp_cox_response_free", "survexp", "~ grp", "cox", list(times = c(2, 5, 10)), missing)
    add("survexp_cox_factor", "survexp", "time ~ grp", "cox_factor", list(times = c(2, 5, 10)), missing)
    add("survexp_external_vectors", "survexp", "~ 1", "population", list(times = c(0, 2, 5, 10)),
        missing, rmap_kind = "vectors")
    add("survexp_direct_vectors", "survexp", "time ~ 1", "us", list(times = c(0, 2, 5, 10)), missing)
    add("pyears_response_alias", "pyears", "time ~ grp", "population", list(scale = 1),
        missing, weights = "time", rmap_kind = "response")
    for (method in c("individual.s", "individual.h"))
        add(paste0("survexp_", method), "survexp", "time ~ 1", "population", list(method = method), missing)
}
zero_rows <- tryCatch(survexp(~1, list(unused = 1:8), ratetable = table,
    rmap = list(age = 2, sex = "male"), times = c(0, 2)), error = conditionMessage)
stopifnot(identical(zero_rows, "Data set has 0 rows"))
scalar_weights <- list()
for (data_frame in c(FALSE, TRUE)) for (count in c(1L, 4L)) {
    data <- if (data_frame) data.frame(unused = 1:4) else list(unused = 1:4)
    weights <- rep(1, count)
    # Stock scalar rate entries reach the native kernel unexpanded at n > 1.
    # Use explicit external vectors there: the independent oracle for broadcast.
    rate_age <- rep(14610, count)
    rate_sex <- rep(1, count)
    rate_year <- rep(as.Date("2000-01-01"), count)
    rate_call <- if (count == 1L) quote(list(age = 14610, sex = 1, year = as.Date("2000-01-01"))) else
        quote(list(age = rate_age, sex = rate_sex, year = rate_year))
    fit <- do.call(survexp, list(formula = ~1, data = data, weights = weights,
        ratetable = survexp.us, rmap = rate_call, times = c(100, 200), x = TRUE, y = TRUE))
    scalar_weights[[length(scalar_weights) + 1L]] <- list(data_frame = data_frame, weights = I(weights),
        rate_call = if (count == 1L) "literal_single_row" else "expanded_external_vectors",
        expected = list(time = I(fit$time), surv = snapshot(fit$surv), n_risk = snapshot(fit$n.risk),
            x = snapshot(fit$x), y = snapshot(fit$y), method = fit$method))
}
reference <- list(metadata = list(generator = "scripts/generate_population_iterable_reference.R",
    r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival"))),
    training = training, cases = cases, scalar_mapping_error = zero_rows,
    scalar_weights = scalar_weights)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE,
    na = "null", null = "null", dataframe = "columns")
cat(length(cases), "population iterator reference cases written to", output, "\n")
