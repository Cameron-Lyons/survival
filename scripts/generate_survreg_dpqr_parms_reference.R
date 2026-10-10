# Independent parameter-query references from stock R survival, never survivalr.
# Run with R_LIBS_USER pointing to the stock survival library. Optional fixture directory.
arguments <- commandArgs(trailingOnly = TRUE)
output_directory <- if (length(arguments)) arguments[[1L]] else "python/tests/fixtures"
dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
output <- file.path(output_directory, "survreg_dpqr_parms_reference.json")
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

number <- function(x) {
  if (is.nan(x)) "NaN" else if (is.na(x)) "NA" else if (is.infinite(x)) {
    if (x > 0) "Inf" else "-Inf"
  } else unname(x)
}
numbers <- function(x) unname(lapply(x, number))
methods <- c("dsurvreg", "psurvreg", "qsurvreg", "rsurvreg")
query_values <- function(method, n) {
  if (method == "rsurvreg") n else if (method == "qsurvreg") {
    c(0, .1, .5, .9, 1)[seq_len(n)]
  } else c(-2, -.1, .3, 1, 2)[seq_len(n)]
}
means <- function(n) c(-.25, .1, .5, .8, -.2)[seq_len(n)]
scales <- function(n) c(.7, 1.2, 2, 3, 1.1)[seq_len(n)]

record <- function(method, query, mean, scale, distribution, mode, parms, names = NULL) {
  set.seed(123)
  before <- .Random.seed
  events <- list()
  error <- NULL
  args <- list(query, mean, scale, distribution)
  if (mode != "omitted") {
    if (mode == "null") args["parms"] <- list(NULL) else {
      if (!is.null(names)) names(parms) <- names
      args["parms"] <- list(parms)
    }
  }
  result <- withCallingHandlers(
    tryCatch(do.call(get(method, asNamespace("survival")), args),
      error = function(e) {
        error <<- conditionMessage(e)
        events[[length(events) + 1L]] <<- list(kind = "error", message = error)
        NULL
      }),
    warning = function(w) {
      events[[length(events) + 1L]] <<- list(kind = "warning", message = conditionMessage(w),
        call = paste(deparse(conditionCall(w)), collapse = " "))
      invokeRestart("muffleWarning")
    })
  list(values = if (is.null(error)) numbers(result) else NULL,
    error = error, events = events, rng_changed = !identical(before, .Random.seed),
    next_uniform = runif(1))
}

cases <- list()
add <- function(name, method, nq, nm, ns, distribution = "t", mode = "numeric",
                parms = numeric(), parm_names = NULL, query_override = NULL) {
  query <- if (is.null(query_override)) query_values(method, nq) else query_override
  mean <- means(nm)
  scale <- scales(ns)
  values <- if (mode == "character") unname(as.list(parms)) else if (mode == "list") {
    unname(lapply(parms, number))
  } else numbers(parms)
  cases[[length(cases) + 1L]] <<- list(name = name, distribution = distribution,
    method = method, query = if (method == "rsurvreg") query else numbers(query),
    mean = numbers(mean), scale = numbers(scale), parms_mode = mode, parms = values,
    parm_names = if (is.null(parm_names)) NULL else unname(as.list(parm_names)),
    expected = record(method, query, mean, scale, distribution, mode, parms, parm_names))
}

# Every query/df/location/scale length from zero through five, with fractional df.
df_values <- c(.5, 1, 2, 3, 8)
for (method in methods) for (nq in 0:5) for (nd in 0:5) for (nm in 0:5) for (ns in 0:5) {
  add(paste("shape", method, nq, nd, nm, ns, sep = "/"), method, nq, nm, ns,
    parms = df_values[seq_len(nd)])
}

# Mix invalid and nonfinite df with valid entries; preserve NA separately from NaN.
patterns <- list(invalid = c(0, -1, .5, 2, 3),
  nonfinite = c(Inf, NA_real_, NaN, -Inf, 4),
  mixed = c(-1, 0, NA_real_, Inf, NaN))
for (pattern in names(patterns)) for (method in methods) for (nq in 0:5) for (nd in 0:5) {
  add(paste(pattern, method, nq, nd, sep = "/"), method, nq, 2, 3,
    parms = patterns[[pattern]][seq_len(nd)])
}
for (df in c(0, -1, 2, Inf, -Inf, NA_real_, NaN)) for (method in methods) for (nq in 0:5) {
  add(paste("scalar", number(df), method, nq, sep = "/"), method, nq, 1, 1, parms = df)
}

# Mathlib's missing-value precedence and warning suppression is separate from arithmetic.
missing_queries <- list(na = NA_real_, nan = NaN,
  na_nan = c(NA_real_, NaN), nan_na = c(NaN, NA_real_),
  mixed = c(NA_real_, NaN, -.1, .5, 1.1))
missing_df <- list(na = NA_real_, nan = NaN, na_nan = c(NA_real_, NaN),
  nan_na = c(NaN, NA_real_), zero = 0, negative = -1,
  mixed = c(0, -1, NA_real_, NaN, Inf), valid_mixed = c(1, NA_real_, 4, NaN))
for (method in methods[methods != "rsurvreg"]) {
  for (query_name in names(missing_queries)) for (df_name in names(missing_df)) {
    add(paste("missing_mathlib", method, query_name, df_name, sep = "/"), method, 1, 1, 1,
      parms = missing_df[[df_name]], query_override = missing_queries[[query_name]])
  }
}

# Missing and NULL fail when the distribution function forces parms, not at lookup.
for (mode in c("omitted", "null")) for (method in methods) for (nq in 0:5) {
  for (nm in c(0, 1, 3)) for (ns in c(0, 1, 5)) {
    add(paste(mode, method, nq, nm, ns, sep = "/"), method, nq, nm, ns, mode = mode)
  }
}

# Names do not change the numeric argument of pt/dt/qt.
for (method in methods) for (nq in 0:5) {
  add(paste("named", method, nq, sep = "/"), method, nq, 2, 3,
    parms = c(1, 4, Inf), parm_names = c("unexpected", "df", "other"))
}

# Named families other than t do not evaluate or validate the supplied parms.
ignored <- list(omitted = NULL, null = NULL, empty = numeric(),
  invalid = c(-1, 0, 2), nonfinite = c(Inf, NA_real_, NaN),
  character = c("not numeric", "bad"), list = list(1, 2))
for (distribution in c("extreme", "gaussian", "logistic", "weibull", "exponential",
                       "rayleigh", "lognormal", "loglogistic")) {
  for (method in methods) for (kind in names(ignored)) {
    mode <- if (kind %in% c("omitted", "null", "character", "list")) kind else "numeric"
    add(paste("ignored", distribution, method, kind, sep = "/"), method, 3, 2, 5,
      distribution, mode, ignored[[kind]])
  }
}

# Keep every density column as separate evidence for warnings from unselected columns.
density_cases <- list()
all_patterns <- c(list(valid = df_values), patterns)
for (pattern in names(all_patterns)) for (nx in 0:5) for (nd in 0:5) {
  x <- c(-2, -.1, .3, 1, 2)[seq_len(nx)]
  df <- all_patterns[[pattern]][seq_len(nd)]
  events <- list()
  result <- withCallingHandlers(survreg.distributions$t$density(x, df), warning = function(w) {
    events[[length(events) + 1L]] <<- list(kind = "warning", message = conditionMessage(w),
      call = paste(deparse(conditionCall(w)), collapse = " "))
    invokeRestart("muffleWarning")
  })
  density_cases[[length(density_cases) + 1L]] <- list(
    name = paste("density", pattern, nx, nd, sep = "/"), x = numbers(x), df = numbers(df),
    shape = unname(as.list(dim(result))),
    columns = unname(lapply(seq_len(ncol(result)), function(i) numbers(result[, i]))),
    events = events)
}

write_json(list(metadata = list(r = as.character(getRversion()),
  survival = as.character(packageVersion("survival")), seed = 123),
  cases = cases, density_cases = density_cases), output,
  auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null")
cat(length(cases), "DPQR parameter cases;", length(density_cases), "five-column density cases\n")
