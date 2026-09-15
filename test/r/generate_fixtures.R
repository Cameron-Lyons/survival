#!/usr/bin/env Rscript
#
# Generate the R reference fixtures consumed by
#   python/tests/test_r_fixtures.py  and  src/tests/r_fixtures.rs
#
# Run from any directory:
#   <renv>/bin/Rscript test/r/generate_fixtures.R [--check]
#
# Every fixture file is a JSON document with the schema described in
# test/r/README.md.  The script is deterministic: no timestamps, and the one
# Monte-Carlo computation (yates with predict = "risk") is seeded.  --check
# regenerates a second copy into a scratch directory and fails unless the two
# are byte-identical; R_FIXTURES_DIR overrides the output directory.

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

# ---------------------------------------------------------------------------
# Paths and metadata
# ---------------------------------------------------------------------------

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
  if (length(file_arg) == 0) {
    return(normalizePath("test/r/generate_fixtures.R", mustWork = TRUE))
  }
  normalizePath(file_arg[1], mustWork = TRUE)
}

script_dir <- dirname(script_path())
# R_FIXTURES_DIR overrides the output directory (used by the --check run).
fixture_dir <- Sys.getenv("R_FIXTURES_DIR", unset = file.path(script_dir, "fixtures"))
dir.create(fixture_dir, showWarnings = FALSE, recursive = TRUE)
self_check <- "--check" %in% commandArgs(trailingOnly = TRUE)

metadata <- list(
  survival_version = as.character(packageVersion("survival")),
  generator = "test/r/generate_fixtures.R",
  encoding = list(
    matrices = "row-major nested arrays",
    non_finite = "NA -> null, NaN -> \"NaN\", Inf -> \"Inf\", -Inf -> \"-Inf\"",
    named_vectors = "JSON objects keyed by name",
    rows = "1-based row indices into the named dataset"
  )
)

cat("survival", metadata$survival_version, "/", R.version.string, "\n")

# ---------------------------------------------------------------------------
# JSON encoding helpers
# ---------------------------------------------------------------------------

# Numeric vector -> JSON array (always an array, even for length 1).
jvec <- function(x) {
  x <- unname(x)
  if (is.factor(x)) {
    return(I(as.character(x)))
  }
  if (is.character(x) || is.logical(x)) {
    return(I(x))
  }
  if (inherits(x, "Date")) {
    return(I(as.numeric(x)))
  }
  x <- as.numeric(x)
  if (length(x) == 0) {
    return(I(numeric(0)))
  }
  # Guard against denormal garbage: none of the recorded quantities is
  # legitimately that small (see clean_survfitms for the known source).
  x[!is.na(x) & abs(x) < 1e-300] <- 0
  if (all(is.finite(x))) {
    return(I(x))
  }
  lapply(x, function(v) {
    if (is.nan(v)) {
      "NaN"
    } else if (is.na(v)) {
      NA
    } else if (is.infinite(v)) {
      if (v > 0) "Inf" else "-Inf"
    } else {
      v
    }
  })
}

# Matrix -> row-major nested arrays.
jmat <- function(x) {
  x <- as.matrix(x)
  if (nrow(x) == 0) {
    return(I(list()))
  }
  lapply(seq_len(nrow(x)), function(i) jvec(x[i, ]))
}

# Named vector -> JSON object.
jnamed <- function(x) {
  if (is.null(names(x))) {
    stop("jnamed requires a named vector")
  }
  out <- lapply(seq_along(x), function(i) {
    v <- jvec(x[[i]])
    if (is.list(v)) v[[1]] else v[[1]]
  })
  names(out) <- names(x)
  out
}

# Matrix with dimnames -> {rownames, colnames, values}.
jmat_named <- function(x) {
  x <- as.matrix(x)
  list(
    rownames = I(if (is.null(rownames(x))) character(0) else rownames(x)),
    colnames = I(if (is.null(colnames(x))) character(0) else colnames(x)),
    values = jmat(x)
  )
}

# 3-d array -> list of matrices along the third dimension.
jarray3 <- function(x) {
  lapply(seq_len(dim(x)[3]), function(k) jmat(x[, , k, drop = TRUE]))
}

# data.frame -> {columns: {name: values}, factors: {name: levels}}
jframe <- function(df) {
  cols <- list()
  factors <- list()
  for (nm in names(df)) {
    col <- df[[nm]]
    if (is.factor(col)) {
      factors[[nm]] <- I(levels(col))
      cols[[nm]] <- I(as.character(col))
    } else if (is.character(col)) {
      cols[[nm]] <- jvec(col)
    } else if (is.logical(col)) {
      cols[[nm]] <- lapply(col, function(v) if (is.na(v)) NA else v)
    } else if (inherits(col, "Date")) {
      cols[[nm]] <- jvec(as.numeric(col))
    } else if (inherits(col, "Surv")) {
      cols[[nm]] <- jmat(unclass(col))
    } else if (is.matrix(col)) {
      cols[[nm]] <- jmat(col)
    } else {
      cols[[nm]] <- jvec(col)
    }
  }
  list(nrow = nrow(df), columns = cols, factors = factors)
}

# Surv object -> matrix plus attributes.
jsurv <- function(s) {
  out <- list(
    type = attr(s, "type"),
    colnames = I(colnames(s)),
    values = jmat(unclass(s))
  )
  if (!is.null(attr(s, "states"))) out$states <- I(attr(s, "states"))
  if (!is.null(attr(s, "inputAttributes"))) {
    out$inputAttributes <- lapply(attr(s, "inputAttributes"), function(a) {
      lapply(a, function(v) if (is.character(v)) I(v) else jvec(v))
    })
  }
  out
}

# ---------------------------------------------------------------------------
# survfit encoding
# ---------------------------------------------------------------------------

survfit_curve_fields <- c(
  "time", "n.risk", "n.event", "n.censor", "n.enter", "surv", "std.err",
  "cumhaz", "std.chaz", "lower", "upper", "pstate", "std.err0", "n.transition"
)

split_survfit <- function(fit) {
  if (is.null(fit$strata)) {
    return(list(list(name = "1", index = seq_along(fit$time))))
  }
  ends <- cumsum(fit$strata)
  starts <- c(1, head(ends, -1) + 1)
  lapply(seq_along(fit$strata), function(i) {
    list(name = names(fit$strata)[i], index = seq(starts[i], ends[i]))
  })
}

# survfitAJ's C code (src/survfitaj.c) zeroes only the first nstate slots of
# the std.chaz accumulator, so when a curve has more transition types than
# states its rows before the first event carry uninitialised memory in the
# remaining columns (values differ between runs and machines).  The value
# the code would produce had it been initialised is 0: no influence has
# accumulated before the first event, and every later row is recomputed
# from the (zeroed) influence matrix.  A row with no event yet is exactly a
# row whose cumulative hazards are all zero.
clean_survfitms <- function(fit) {
  if (inherits(fit, "survfitms") && !is.null(fit$std.chaz) && !is.null(fit$cumhaz)) {
    no_event_yet <- rowSums(as.matrix(fit$cumhaz) != 0) == 0
    if (is.matrix(fit$std.chaz)) {
      fit$std.chaz[no_event_yet, ] <- 0
    } else {
      fit$std.chaz[no_event_yet] <- 0
    }
  }
  fit
}

encode_curve_slice <- function(fit, index) {
  out <- list()
  for (field in survfit_curve_fields) {
    value <- fit[[field]]
    if (is.null(value)) next
    key <- gsub("\\.", "_", field)
    if (is.matrix(value)) {
      out[[key]] <- jmat(value[index, , drop = FALSE])
    } else if (length(value) == length(fit$time)) {
      out[[key]] <- jvec(value[index])
    }
  }
  out
}

# survfit -> {n, strata: {name: size}, curves: [{name, time, n_risk, ...}],
#             conf_type, conf_int, type, states, p0, transitions}
jsurvfit <- function(fit) {
  fit <- clean_survfitms(fit)
  parts <- split_survfit(fit)
  curves <- lapply(parts, function(p) {
    c(list(name = p$name), encode_curve_slice(fit, p$index))
  })
  out <- list(
    n = jvec(fit$n),
    strata = if (is.null(fit$strata)) NULL else jnamed(fit$strata),
    conf_type = fit$conf.type,
    conf_int = fit$conf.int,
    type = fit$type,
    curves = curves
  )
  if (!is.null(fit$states)) out$states <- I(fit$states)
  if (!is.null(fit$p0)) {
    out$p0 <- if (is.matrix(fit$p0)) jmat(fit$p0) else jvec(fit$p0)
  }
  if (!is.null(fit$transitions)) out$transitions <- jmat_named(fit$transitions)
  if (!is.null(fit$t0)) out$t0 <- fit$t0
  if (!is.null(fit$logse)) out$logse <- fit$logse
  out
}

jsurvfit_summary_table <- function(fit, ...) {
  s <- summary(fit, ...)
  tab <- s$table
  if (is.matrix(tab)) {
    jmat_named(tab)
  } else {
    jnamed(tab)
  }
}

jsurvfit_summary_times <- function(fit, times, ...) {
  s <- summary(clean_survfitms(fit), times = times, extend = TRUE, ...)
  out <- list(times = jvec(times))
  for (field in c("time", "n.risk", "n.event", "n.censor", "surv", "std.err",
                  "cumhaz", "std.chaz", "lower", "upper", "pstate")) {
    value <- s[[field]]
    if (is.null(value)) next
    key <- gsub("\\.", "_", field)
    out[[key]] <- if (is.matrix(value)) jmat(value) else jvec(value)
  }
  if (!is.null(s$strata)) out$strata <- I(as.character(s$strata))
  out
}

jquantile <- function(fit, probs = c(0.25, 0.5, 0.75)) {
  q <- quantile(fit, probs = probs, conf.int = TRUE)
  wrap <- function(v) if (is.matrix(v)) jmat(v) else jvec(v)
  list(probs = jvec(probs), quantile = wrap(q$quantile), lower = wrap(q$lower),
       upper = wrap(q$upper))
}

# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

fixtures <- list()
failures <- character(0)

add_case <- function(topic, name, inputs, expected, note = NULL) {
  case <- c(list(name = name, topic = topic), inputs, list(expected = expected))
  if (!is.null(note)) case$note <- note
  fixtures[[topic]][[length(fixtures[[topic]]) + 1]] <<- case
  invisible(NULL)
}

# Evaluate a case body, recording failures instead of aborting the run.
run_case <- function(topic, name, body) {
  result <- tryCatch(
    {
      body()
      TRUE
    },
    error = function(e) {
      msg <- sprintf("%s/%s: %s", topic, name, conditionMessage(e))
      cat("FAILED", msg, "\n")
      failures <<- c(failures, msg)
      FALSE
    }
  )
  invisible(result)
}

# Inputs description helpers
dataset_input <- function(dataset, rows = NULL, formula = NULL, args = list(),
                          factors = NULL) {
  out <- list(dataset = dataset)
  if (!is.null(rows)) out$rows <- I(as.integer(rows))
  if (!is.null(formula)) out$formula <- formula
  if (!is.null(factors)) out$factors <- factors
  out$args <- args
  out
}

# Inline (derived or synthetic) data frames are registered once per name and
# written into the topic document's "data" section; cases refer to them by
# "data_ref" so a frame shared by several cases is only stored once.
inline_data <- list()
register_data <- function(name, df) {
  inline_data[[name]] <<- df
  invisible(df)
}

inline_input <- function(ref, formula = NULL, args = list(), rows = NULL) {
  if (is.null(inline_data[[ref]])) stop(sprintf("inline data %s is not registered", ref))
  out <- list(data_ref = ref)
  if (!is.null(rows)) out$rows <- I(as.integer(rows))
  if (!is.null(formula)) out$formula <- formula
  out$args <- args
  out
}

write_topic <- function(topic) {
  cases <- fixtures[[topic]]
  refs <- unique(unlist(lapply(cases, function(case) c(case$data_ref, case$data_ref2))))
  data_section <- list()
  for (ref in refs) data_section[[ref]] <- jframe(inline_data[[ref]])
  doc <- list(
    topic = topic,
    metadata = metadata,
    n_cases = length(cases),
    data = data_section,
    cases = cases
  )
  path <- file.path(fixture_dir, paste0(topic, ".json"))
  json <- toJSON(doc, digits = NA, auto_unbox = TRUE, na = "null", null = "null",
                 pretty = FALSE)
  writeLines(json, path)
  size <- file.info(path)$size
  cat(sprintf("wrote %-28s %4d cases %8.1f KB\n", basename(path), length(cases),
              size / 1024))
  if (size > 2 * 1024 * 1024) {
    stop(sprintf("%s exceeds 2 MB", path))
  }
}

factor_levels <- function(df, cols) {
  out <- list()
  for (nm in cols) {
    if (is.factor(df[[nm]])) out[[nm]] <- I(levels(df[[nm]]))
  }
  if (length(out) == 0) NULL else out
}

# ---------------------------------------------------------------------------
# Datasets used
# ---------------------------------------------------------------------------

data(infert, package = "datasets")

lung_cc <- which(complete.cases(lung[, c("time", "status", "age", "sex", "ph.ecog",
                                          "ph.karno", "inst")]))
pbc_trial <- which(!is.na(pbc$trt))
colon_death <- which(colon$etype == 2)
flchain_rows <- seq_len(500)
nafld_rows <- seq_len(1000)
mgus2_rows <- seq_len(400)

# Small synthetic sets ---------------------------------------------------------

synthetic_ties <- register_data("synthetic_ties", data.frame(
  time = c(1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6, 7, 8, 9, 9),
  status = c(1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1),
  x = c(0.5, 1.2, -0.3, 0.8, 0.1, 1.5, -1.0, 0.4, 0.9, -0.2, 0.3, 1.1, -0.7, 0.6, 0.2, -0.4),
  g = factor(c("a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"))
))

synthetic_delayed <- register_data("synthetic_delayed", data.frame(
  entry = c(0, 0, 1, 2, 2, 3, 0, 4, 1, 5, 0, 3),
  exit = c(4, 6, 5, 7, 3, 9, 2, 8, 6, 10, 5, 7),
  status = c(1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1),
  x = c(1.0, 0.2, -0.5, 0.7, 1.4, -0.1, 0.3, 0.9, -0.8, 0.5, 1.1, -0.3)
))

synthetic_timefix <- register_data("synthetic_timefix", data.frame(
  time = c(0.1 + 0.2, 0.3, 0.7, 0.1 * 7, 1.0, 1.0 + 1e-13, 2.5, 3.1),
  status = c(1, 1, 1, 0, 1, 1, 0, 1)
))

synthetic_interval <- register_data("synthetic_interval", data.frame(
  left = c(1, 2, NA, 4, 5, 3, 6, NA, 2, 7),
  right = c(3, 4, 2, 6, 5, NA, 8, 5, 3, NA),
  g = rep(c("a", "b"), 5)
))

synthetic_interval_status <- register_data("synthetic_interval_status", data.frame(
  time = c(1, 2, 2, 4, 5, 3, 6, 5, 2, 7),
  time2 = c(3, 4, NA, 6, NA, NA, 8, NA, 3, NA),
  status = c(3, 3, 2, 3, 1, 0, 3, 2, 3, 0)
))

# mgus2 competing-risk data (survival vignette "compete")
mgus2_cr <- within(mgus2, {
  etime <- ifelse(pstat == 0, futime, ptime)
  event <- factor(ifelse(pstat == 0, 2 * death, 1), 0:2,
                  labels = c("censor", "pcm", "death"))
})
mgus2_cr <- register_data("mgus2_cr", mgus2_cr[, c("id", "age", "sex", "etime", "event")])

# Illness-death data derived from myeloid (survival vignette "multi")
myeloid_ms <- local({
  data1 <- tmerge(myeloid[, 1:4], myeloid, id = id, death = event(futime, death),
                  sct = event(txtime), cr = event(crtime),
                  relapse = event(rltime), priorsct = tdc(txtime),
                  priorcr = tdc(crtime))
  temp <- with(data1, cr + 2 * sct + 4 * relapse + 8 * death)
  data1$event <- factor(temp, c(0, 1, 2, 4, 8),
                        c("none", "CR", "SCT", "relapse", "death"))
  data1
})
myeloid_ms <- register_data("myeloid_ms", myeloid_ms[myeloid_ms$id <= 120, ])

# Small explicit multistate history with istate (subjects start in
# different states, some transitions back).
synthetic_mstate <- register_data("synthetic_mstate", data.frame(
  id = c(1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8),
  tstart = c(0, 3, 0, 2, 0, 0, 5, 0, 0, 4, 0, 0),
  tstop = c(3, 8, 2, 9, 6, 5, 7, 4, 4, 10, 7, 9),
  event = factor(c("ill", "dead", "ill", "well", "dead", "ill", "dead", "censor",
                   "ill", "censor", "dead", "censor"),
                 levels = c("censor", "well", "ill", "dead")),
  istate = factor(c("well", "ill", "well", "ill", "well", "ill", "ill", "well",
                    "well", "ill", "well", "ill"),
                  levels = c("well", "ill", "dead")),
  x = c(1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1)
))

# jasa is a data.frame with Date columns; keep the numeric part for fixtures.
jasa_num <- register_data("jasa_num", data.frame(
  futime = jasa$futime,
  fustat = jasa$fustat,
  surgery = jasa$surgery,
  age = jasa$age,
  transplant = jasa$transplant
))

# ---------------------------------------------------------------------------
# datasets.json: checksums of the bundled datasets used by the fixtures
# ---------------------------------------------------------------------------

dataset_summary <- function(df) {
  cols <- lapply(names(df), function(nm) {
    col <- df[[nm]]
    item <- list(name = nm)
    if (is.factor(col)) {
      item$type <- "factor"
      item$levels <- I(levels(col))
      item$codes_sum <- sum(as.integer(col), na.rm = TRUE)
      item$n_missing <- sum(is.na(col))
    } else if (is.character(col)) {
      item$type <- "character"
      item$levels <- I(sort(unique(col[!is.na(col)])))
      item$n_missing <- sum(is.na(col))
    } else if (is.logical(col)) {
      item$type <- "logical"
      item$sum <- sum(col, na.rm = TRUE)
      item$n_missing <- sum(is.na(col))
    } else if (inherits(col, "Date")) {
      item$type <- "date"
      item$sum <- sum(as.numeric(col), na.rm = TRUE)
      item$n_missing <- sum(is.na(col))
    } else {
      item$type <- if (is.integer(col)) "integer" else "numeric"
      item$sum <- sum(as.numeric(col), na.rm = TRUE)
      item$n_missing <- sum(is.na(col))
      item$head <- jvec(head(col, 5))
    }
    item
  })
  list(nrow = nrow(df), ncol = ncol(df), columns = cols)
}

for (ds in c("lung", "aml", "ovarian", "veteran", "pbc", "kidney", "rats", "rats2",
             "cgd", "cgd0", "bladder", "heart", "mgus", "mgus2", "myeloid", "colon",
             "flchain", "nafld1", "transplant", "jasa", "nwtco", "tobin", "stanford2",
             "udca", "solder", "pbcseq", "hoel", "myeloma", "rhDNase", "diabetic",
             "retinopathy", "gbsg", "rotterdam", "logan")) {
  run_case("datasets", ds, function() {
    df <- get(ds)
    add_case("datasets", ds, list(dataset = ds), dataset_summary(df))
  })
}

# ---------------------------------------------------------------------------
# survfit: Kaplan-Meier and friends
# ---------------------------------------------------------------------------

km_expected <- function(fit, rmean = NULL, times = NULL, quantiles = TRUE) {
  out <- list(fit = jsurvfit(fit))
  out$summary_table <- if (is.null(rmean)) jsurvfit_summary_table(fit) else
    jsurvfit_summary_table(fit, rmean = rmean)
  out$summary_std_err <- {
    s <- summary(fit)
    jvec(s$std.err)
  }
  if (!is.null(times)) out$summary_times <- jsurvfit_summary_times(fit, times)
  if (quantiles) out$quantile <- jquantile(fit)
  out
}

# Arguments given as a column name (id = "id") are resolved against the data
# frame before the call; the fixture keeps the column name.
resolve_column_args <- function(args, df, names = c("id", "cluster", "weights", "istate",
                                                     "subcoh", "strata")) {
  for (nm in intersect(names(args), names)) {
    if (is.character(args[[nm]]) && length(args[[nm]]) == 1 && !is.null(df[[args[[nm]]]])) {
      args[[nm]] <- df[[args[[nm]]]]
    }
  }
  args
}

# Resolve a case's data: either a bundled dataset or a registered inline frame.
case_frame <- function(dataset, data_ref, rows) {
  df <- if (is.null(data_ref)) get(dataset) else inline_data[[data_ref]]
  if (!is.null(rows)) df <- df[rows, , drop = FALSE]
  df
}

case_inputs <- function(dataset, data_ref, rows, formula, args, df) {
  if (is.null(data_ref)) {
    dataset_input(dataset, rows, formula, args,
                  factor_levels(df, intersect(all.vars(as.formula(formula)), names(df))))
  } else {
    inline_input(data_ref, formula, args, rows)
  }
}

km_case <- function(name, dataset, formula, rows = NULL, args = list(),
                    data_ref = NULL, rmean = NULL, times = NULL, quantiles = TRUE,
                    note = NULL) {
  run_case("survfit_km", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df),
                   resolve_column_args(args, df))
    fit <- do.call(survfit, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survfit_km", name, inputs, km_expected(fit, rmean, times, quantiles), note)
  })
}

km_case("lung_sex", "lung", "Surv(time, status) ~ sex", rmean = 500,
        times = c(0, 100, 250, 500, 750, 1000, 1100))
km_case("lung_1", "lung", "Surv(time, status) ~ 1", times = c(50, 200, 400, 800))
km_case("aml_x", "aml", "Surv(time, status) ~ x", times = c(0, 10, 20, 30, 50, 100, 200))
km_case("ovarian_rx", "ovarian", "Surv(futime, fustat) ~ rx")
km_case("veteran_celltype", "veteran", "Surv(time, status) ~ celltype")
km_case("kidney_sex", "kidney", "Surv(time, status) ~ sex")
km_case("pbc_trt", "pbc", "Surv(time, status == 2) ~ trt", rows = pbc_trial)
km_case("colon_rx_death", "colon", "Surv(time, status) ~ rx", rows = colon_death)
km_case("flchain_sex_500", "flchain", "Surv(futime, death) ~ sex", rows = flchain_rows,
        note = "first 500 rows of flchain")
km_case("nafld1_male_1000", "nafld1", "Surv(futime, status) ~ male", rows = nafld_rows,
        note = "first 1000 rows of nafld1")
km_case("transplant_abo_ltx", "transplant", "Surv(futime, event == \"ltx\") ~ abo")
km_case("mgus2_sex", "mgus2", "Surv(futime, death) ~ sex")
km_case("myeloid_trt", "myeloid", "Surv(futime, death) ~ trt")
km_case("rats_rx", "rats", "Surv(time, status) ~ rx")
km_case("bladder_rx_enum1", "bladder", "Surv(stop, event) ~ rx", rows = which(bladder$enum == 1))
km_case("jasa_surgery", NULL, "Surv(futime, fustat) ~ surgery", data_ref = "jasa_num")

lung_weights <- rep(c(1, 2, 0.5, 1.5), length.out = nrow(lung))
km_case("lung_weighted", "lung", "Surv(time, status) ~ sex",
        args = list(weights = jvec(lung_weights)), rmean = 400)
km_case("lung_weighted_1", "lung", "Surv(time, status) ~ 1",
        args = list(weights = jvec(lung_weights)))

for (ct in c("log", "log-log", "plain", "logit", "arcsin", "none")) {
  km_case(paste0("lung_conf_", gsub("-", "", ct)), "lung", "Surv(time, status) ~ 1",
          args = list(conf.type = ct), quantiles = ct != "none")
}
km_case("lung_conf_lower_modified", "lung", "Surv(time, status) ~ 1",
        args = list(conf.type = "log", conf.lower = "modified"))
km_case("lung_conf_lower_peto", "lung", "Surv(time, status) ~ 1",
        args = list(conf.type = "log", conf.lower = "peto"))
km_case("lung_conf_int_90", "lung", "Surv(time, status) ~ sex", args = list(conf.int = 0.9))
km_case("lung_se_fit_false", "lung", "Surv(time, status) ~ sex", args = list(se.fit = FALSE),
        quantiles = FALSE)
km_case("lung_type_fh", "lung", "Surv(time, status) ~ sex", args = list(type = "fh"))
km_case("lung_type_fh2", "lung", "Surv(time, status) ~ sex", args = list(type = "fh2"))
km_case("lung_stype2_ctype1", "lung", "Surv(time, status) ~ 1", args = list(stype = 2, ctype = 1))
km_case("lung_stype2_ctype2", "lung", "Surv(time, status) ~ 1", args = list(stype = 2, ctype = 2))
km_case("lung_stype1_ctype2", "lung", "Surv(time, status) ~ 1", args = list(stype = 1, ctype = 2))
km_case("aml_fh_ties", "aml", "Surv(time, status) ~ x", args = list(type = "fh"))
km_case("aml_fh2_ties", "aml", "Surv(time, status) ~ x", args = list(type = "fh2"))
km_case("lung_start_time_100", "lung", "Surv(time, status) ~ sex", args = list(start.time = 100))
km_case("lung_reverse", "lung", "Surv(time, status == 1) ~ sex",
        note = "censoring distribution: R has no reverse= argument, the status is flipped")
km_case("lung_reverse_1", "lung", "Surv(time, status == 1) ~ 1")
km_case("lung_time0", "lung", "Surv(time, status) ~ sex", args = list(time0 = TRUE))

km_case("cgd_counting_treat_id", "cgd", "Surv(tstart, tstop, status) ~ treat",
        args = list(id = "id"))
km_case("cgd_counting_treat_noid", "cgd", "Surv(tstart, tstop, status) ~ treat")
km_case("heart_counting_transplant_id", "heart", "Surv(start, stop, event) ~ transplant",
        args = list(id = "id"))
km_case("cgd_counting_cluster", "cgd", "Surv(tstart, tstop, status) ~ treat",
        args = list(cluster = "id"))
km_case("synthetic_ties", NULL, "Surv(time, status) ~ g", data_ref = "synthetic_ties",
        times = c(0, 1, 2, 4, 6, 9, 10))
km_case("synthetic_ties_1", NULL, "Surv(time, status) ~ 1", data_ref = "synthetic_ties")
km_case("synthetic_delayed_entry", NULL, "Surv(entry, exit, status) ~ 1",
        data_ref = "synthetic_delayed", times = c(0, 2, 4, 6, 8, 10))
km_case("synthetic_timefix_true", NULL, "Surv(time, status) ~ 1", data_ref = "synthetic_timefix",
        args = list(timefix = TRUE))
km_case("synthetic_timefix_false", NULL, "Surv(time, status) ~ 1", data_ref = "synthetic_timefix",
        args = list(timefix = FALSE))
km_case("lung_sex_ph_ecog", "lung", "Surv(time, status) ~ sex + ph.ecog", rows = lung_cc,
        quantiles = FALSE)

# ---------------------------------------------------------------------------
# survfit: Aalen-Johansen multistate
# ---------------------------------------------------------------------------

aj_case <- function(name, dataset, formula, data_ref = NULL, rows = NULL, args = list(),
                    times = NULL, influence = FALSE, note = NULL) {
  run_case("survfit_multistate", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df),
                   resolve_column_args(args, df))
    if (influence) call_args$influence <- TRUE
    fit <- do.call(survfit, call_args)
    expected <- list(fit = jsurvfit(fit))
    if (!is.null(times)) expected$summary_times <- jsurvfit_summary_times(fit, times)
    if (influence) {
      expected$influence_pstate <- jarray3(fit$influence.pstate)
      if (!is.null(fit$influence.chaz)) expected$influence_chaz <- jarray3(fit$influence.chaz)
    }
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survfit_multistate", name, inputs, expected, note)
  })
}

aj_case("mgus2_sex", NULL, "Surv(etime, event) ~ sex", data_ref = "mgus2_cr",
        times = c(0, 50, 100, 200, 300))
aj_case("mgus2_400_1", NULL, "Surv(etime, event) ~ 1", data_ref = "mgus2_cr", rows = mgus2_rows,
        times = c(12, 120, 240), note = "first 400 rows of mgus2")
aj_case("mgus2_400_1_p0", NULL, "Surv(etime, event) ~ 1", data_ref = "mgus2_cr", rows = mgus2_rows,
        args = list(p0 = c(0.9, 0.05, 0.05)))
aj_case("mgus2_400_sex_conf_loglog", NULL, "Surv(etime, event) ~ sex", data_ref = "mgus2_cr",
        rows = mgus2_rows, args = list(conf.type = "log-log"))
aj_case("mgus2_400_id", NULL, "Surv(etime, event) ~ sex", data_ref = "mgus2_cr", rows = mgus2_rows,
        args = list(id = "id"))
aj_case("myeloid_ms_trt", NULL, "Surv(tstart, tstop, event) ~ trt", data_ref = "myeloid_ms",
        args = list(id = "id"), times = c(0, 100, 365, 730, 1500),
        note = "myeloid subjects with id <= 120, expanded with tmerge as in the multi vignette")
aj_case("myeloid_ms_1", NULL, "Surv(tstart, tstop, event) ~ 1", data_ref = "myeloid_ms",
        args = list(id = "id"))
aj_case("transplant_abo", "transplant", "Surv(futime, event) ~ abo",
        times = c(0, 30, 100, 365, 730))
aj_case("transplant_1_p0", "transplant", "Surv(futime, event) ~ 1",
        args = list(p0 = c(0.8, 0.05, 0.1, 0.05)))
aj_case("synthetic_istate", NULL, "Surv(tstart, tstop, event) ~ 1", data_ref = "synthetic_mstate",
        args = list(id = "id", istate = "istate"), influence = TRUE)
aj_case("synthetic_istate_x", NULL, "Surv(tstart, tstop, event) ~ x", data_ref = "synthetic_mstate",
        args = list(id = "id", istate = "istate"))
register_data("synthetic_ties_mstate", within(synthetic_ties, {
  event <- factor(ifelse(status == 1, as.character(g), "censor"), levels = c("censor", "a", "b"))
}))
aj_case("synthetic_ties_influence", NULL, "Surv(time, event) ~ 1", data_ref = "synthetic_ties_mstate",
        influence = TRUE)

# ---------------------------------------------------------------------------
# survfit: interval censoring (Turnbull)
# ---------------------------------------------------------------------------

turnbull_expected <- function(fit) {
  list(
    time = jvec(fit$time),
    n_risk = jvec(fit$n.risk),
    n_event = jvec(fit$n.event),
    n_censor = jvec(fit$n.censor),
    surv = jvec(fit$surv),
    std_err = jvec(fit$std.err),
    lower = jvec(fit$lower),
    upper = jvec(fit$upper),
    n = fit$n
  )
}

interval_case <- function(name, data_ref, formula, grouped = FALSE) {
  run_case("survfit_interval", name, function() {
    df <- inline_data[[data_ref]]
    fit <- survfit(as.formula(formula), data = df)
    expected <- if (grouped) list(fit = jsurvfit(fit)) else turnbull_expected(fit)
    add_case("survfit_interval", name, inline_input(data_ref, formula), expected)
  })
}
interval_case("interval2_synthetic", "synthetic_interval",
              "Surv(left, right, type = \"interval2\") ~ 1")
interval_case("interval_status_synthetic", "synthetic_interval_status",
              "Surv(time, time2, status, type = \"interval\") ~ 1")
interval_case("interval2_synthetic_group", "synthetic_interval",
              "Surv(left, right, type = \"interval2\") ~ g", grouped = TRUE)

# ---------------------------------------------------------------------------
# survdiff
# ---------------------------------------------------------------------------

survdiff_expected <- function(sd) {
  list(
    n = jvec(sd$n),
    obs = if (is.matrix(sd$obs)) jmat(sd$obs) else jvec(sd$obs),
    exp = if (is.matrix(sd$exp)) jmat(sd$exp) else jvec(sd$exp),
    var = if (is.matrix(sd$var)) jmat(sd$var) else jvec(sd$var),
    chisq = sd$chisq,
    pvalue = sd$pvalue,
    df = if (is.matrix(sd$obs)) nrow(sd$obs) - 1 else max(length(sd$obs) - 1, 1),
    strata_names = I(if (is.null(names(sd$n))) character(0) else names(sd$n))
  )
}

survdiff_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL,
                          note = NULL) {
  run_case("survdiff", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    sd <- do.call(survdiff, c(list(formula = as.formula(formula), data = df), args))
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survdiff", name, inputs, survdiff_expected(sd), note)
  })
}

survdiff_case("lung_sex", "lung", "Surv(time, status) ~ sex")
survdiff_case("lung_sex_rho1", "lung", "Surv(time, status) ~ sex", args = list(rho = 1))
survdiff_case("lung_sex_rho05", "lung", "Surv(time, status) ~ sex", args = list(rho = 0.5))
survdiff_case("veteran_celltype", "veteran", "Surv(time, status) ~ celltype")
survdiff_case("veteran_celltype_rho1", "veteran", "Surv(time, status) ~ celltype",
              args = list(rho = 1))
survdiff_case("pbc_trt_strata_sex", "pbc", "Surv(time, status == 2) ~ trt + strata(sex)",
              rows = pbc_trial)
survdiff_case("lung_ph_ecog_strata_sex", "lung", "Surv(time, status) ~ ph.ecog + strata(sex)",
              rows = lung_cc)
survdiff_case("aml_x", "aml", "Surv(time, status) ~ x")
survdiff_case("ovarian_rx", "ovarian", "Surv(futime, fustat) ~ rx")
survdiff_case("kidney_disease", "kidney", "Surv(time, status) ~ disease")
survdiff_case("colon_rx_death", "colon", "Surv(time, status) ~ rx", rows = colon_death)
survdiff_case("synthetic_ties_g", NULL, "Surv(time, status) ~ g", data_ref = "synthetic_ties")

run_case("survdiff", "lung_one_sample_survexp_us", function() {
  df <- lung
  df$year <- as.Date("2000-01-01")
  expect <- survexp(time ~ 1, data = df,
                    rmap = list(age = age * 365.25, sex = sex, year = year),
                    ratetable = survexp.us, cohort = FALSE)
  sd <- survdiff(Surv(time, status) ~ offset(expect), data = df)
  expected <- survdiff_expected(sd)
  expected$expect <- jvec(expect)
  add_case("survdiff", "lung_one_sample_survexp_us",
           list(dataset = "lung", formula = "Surv(time, status) ~ offset(expect)",
                args = list(expect = jvec(expect))),
           expected,
           note = "expect = survexp(time ~ 1, cohort=FALSE, survexp.us, year=2000-01-01)")
})

# ---------------------------------------------------------------------------
# coxph
# ---------------------------------------------------------------------------

# Evaluate `expr`; on error return list(r_error = message) so a case can
# record which aspects R itself cannot produce (e.g. score residuals with
# the exact method).
try_aspect <- function(expr) {
  tryCatch(expr, error = function(e) list(r_error = conditionMessage(e)))
}

jresiduals_cox <- function(fit) {
  out <- list()
  for (type in c("martingale", "deviance", "score", "schoenfeld", "scaledsch",
                 "dfbeta", "dfbetas", "partial")) {
    out[[type]] <- try_aspect({
      r <- residuals(fit, type = type)
      if (type %in% c("schoenfeld", "scaledsch")) {
        list(time = jvec(as.numeric(rownames(as.matrix(r)))), values = jmat(r))
      } else if (is.matrix(r)) {
        jmat(r)
      } else {
        jvec(r)
      }
    })
  }
  out
}

jpredict_cox <- function(fit, newdata = NULL, extra = FALSE) {
  out <- list()
  types <- if (extra) c("lp", "risk", "expected", "terms", "survival") else
    c("lp", "risk", "expected", "terms")
  for (type in types) {
    out[[type]] <- try_aspect({
      p <- if (is.null(newdata)) predict(fit, type = type, se.fit = TRUE) else
        predict(fit, newdata = newdata, type = type, se.fit = TRUE)
      item <- list()
      if (is.matrix(p$fit)) {
        item$fit <- jmat(p$fit)
        item$se_fit <- jmat(p$se.fit)
        item$colnames <- I(colnames(p$fit))
        if (!is.null(attr(p$fit, "constant"))) item$constant <- attr(p$fit, "constant")
      } else {
        item$fit <- jvec(p$fit)
        item$se_fit <- jvec(p$se.fit)
      }
      item
    })
  }
  out$lp_uncentered <- try_aspect({
    p <- if (is.null(newdata)) predict(fit, type = "lp", reference = "zero") else
      predict(fit, newdata = newdata, type = "lp", reference = "zero")
    jvec(p)
  })
  out
}

jbasehaz <- function(fit, centered) {
  bh <- basehaz(fit, centered = centered)
  out <- list(time = jvec(bh$time), hazard = jvec(bh$hazard))
  if (!is.null(bh$strata)) out$strata <- I(as.character(bh$strata))
  out
}

jzph <- function(fit, all = FALSE) {
  out <- list()
  for (transform in c("km", "rank", "identity")) {
    for (terms in c(TRUE, FALSE)) {
      if (!all && !(transform == "km" && terms)) next
      key <- sprintf("%s_%s", transform, if (terms) "terms" else "noterms")
      out[[key]] <- try_aspect({
        z <- cox.zph(fit, transform = transform, terms = terms, global = TRUE)
        list(
          table = jmat_named(z$table),
          time = jvec(z$time),
          x = jvec(z$x),
          y = jmat(z$y),
          y_colnames = I(colnames(z$y)),
          var = jmat(z$var),
          transform = z$transform
        )
      })
    }
  }
  out
}

jdetail <- function(fit) {
  try_aspect({
    d <- coxph.detail(fit)
    out <- list(
      time = jvec(d$time),
      nevent = jvec(d$nevent),
      nrisk = jvec(d$nrisk),
      hazard = jvec(d$hazard),
      varhaz = jvec(d$varhaz),
      wtrisk = jvec(d$wtrisk),
      score = if (is.matrix(d$score)) jmat(d$score) else jvec(d$score),
      means = if (is.matrix(d$means)) jmat(d$means) else jvec(d$means)
    )
    if (length(dim(d$imat)) == 3) {
      out$imat <- jarray3(d$imat)
    } else {
      out$imat <- jvec(d$imat)
    }
    if (!is.null(d$strata)) out$strata <- jvec(as.integer(d$strata))
    out
  })
}

jsummary_cox <- function(fit) {
  try_aspect({
    s <- summary(fit)
    out <- list(
      logtest = jnamed(s$logtest),
      sctest = jnamed(s$sctest),
      waldtest = jnamed(s$waldtest),
      rsq = jnamed(s$rsq),
      concordance = jnamed(s$concordance),
      coefficients = jmat_named(s$coefficients),
      conf_int = jmat_named(s$conf.int),
      n = s$n,
      nevent = s$nevent,
      used_robust = isTRUE(s$used.robust)
    )
    out
  })
}

janova <- function(fit) {
  try_aspect({
    a <- anova(fit)
    list(
      terms = I(rownames(a)),
      loglik = jvec(a$loglik),
      chisq = jvec(a$Chisq),
      df = jvec(a$Df),
      p = jvec(a[["Pr(>|Chi|)"]])
    )
  })
}

jconcordance <- function(cc) {
  out <- list(
    concordance = if (is.null(dim(cc$concordance))) jvec(cc$concordance) else jmat(cc$concordance),
    n = jvec(cc$n),
    count = if (is.matrix(cc$count)) jmat_named(cc$count) else jnamed(cc$count)
  )
  if (!is.null(cc$var)) out$var <- if (is.matrix(cc$var)) jmat(cc$var) else jvec(cc$var)
  if (!is.null(cc$cvar)) out$cvar <- if (is.matrix(cc$cvar)) jmat(cc$cvar) else jvec(cc$cvar)
  if (!is.null(cc$dfbeta)) out$dfbeta <- if (is.matrix(cc$dfbeta)) jmat(cc$dfbeta) else jvec(cc$dfbeta)
  if (!is.null(cc$influence)) out$influence <- jmat(cc$influence)
  if (!is.null(cc$timewt)) out$timewt <- cc$timewt
  out
}

# NB: use fit[["name"]] rather than fit$name for optional components so that
# R's partial matching (fit$x -> fit$xlevels) cannot pick the wrong element.
cox_core_expected <- function(fit) {
  null_model <- is.null(fit[["coefficients"]])
  out <- list(
    coef = if (null_model) setNames(list(), character(0)) else jnamed(fit[["coefficients"]]),
    coef_names = I(if (null_model) character(0) else names(fit[["coefficients"]])),
    var = if (null_model) I(list()) else jmat(fit[["var"]]),
    loglik = jvec(fit[["loglik"]]),
    score = if (is.null(fit[["score"]])) NULL else fit[["score"]],
    iter = jvec(fit[["iter"]]),
    n = fit[["n"]],
    nevent = fit[["nevent"]],
    means = jvec(fit[["means"]]),
    linear_predictors = jvec(fit[["linear.predictors"]]),
    wald_test = if (is.null(fit[["wald.test"]])) NULL else fit[["wald.test"]],
    method = fit[["method"]]
  )
  if (!is.null(fit[["naive.var"]])) out$naive_var <- jmat(fit[["naive.var"]])
  if (!is.null(fit[["var2"]])) out$var2 <- jmat(fit[["var2"]])
  if (!is.null(fit[["df"]])) out$df <- jvec(fit[["df"]])
  if (!is.null(fit[["df2"]])) out$df2 <- jvec(fit[["df2"]])
  if (!is.null(fit[["nocenter"]])) out$nocenter <- jvec(fit[["nocenter"]])
  if (!is.null(fit[["x"]])) out$x <- jmat_named(fit[["x"]])
  out
}

# The coxph fixtures are split over three topics that share case names:
#   coxph             core fit quantities, martingale/deviance residuals,
#                     concordance, summary tests, wald test, anova
#   coxph_predict     basehaz, survfit(fit), survfit(fit, newdata), predict
#   coxph_diagnostics all residual types, cox.zph, coxph.detail
# `extra = TRUE` adds the larger survfit/zph variants to a handful of cases.
cox_core_topic_expected <- function(fit, tt = FALSE) {
  out <- cox_core_expected(fit)
  if (is.null(fit[["coefficients"]])) {
    out$survfit <- try_aspect(jsurvfit(survfit(fit)))
    return(out)
  }
  if (tt) {
    # the per-observation quantities of a tt() fit refer to the expanded
    # (time-varying) data set; only whole-fit quantities are recorded
    out$linear_predictors <- NULL
    out$summary <- jsummary_cox(fit)
    return(out)
  }
  out$residuals <- list(
    martingale = try_aspect(jvec(residuals(fit, type = "martingale"))),
    deviance = try_aspect(jvec(residuals(fit, type = "deviance")))
  )
  out$concordance <- try_aspect(jconcordance(concordance(fit)))
  out$summary <- jsummary_cox(fit)
  out$wtest <- try_aspect({
    w <- coxph.wtest(fit$var, fit$coefficients)
    list(test = w$test, df = w$df, solve = jvec(w$solve))
  })
  out$anova <- janova(fit)
  out
}

cox_predict_expected <- function(fit, newdata = NULL, extra = FALSE) {
  out <- list(coef = jnamed(fit[["coefficients"]]))
  out$basehaz_centered <- try_aspect(jbasehaz(fit, TRUE))
  out$basehaz_uncentered <- try_aspect(jbasehaz(fit, FALSE))
  out$survfit <- try_aspect(jsurvfit(survfit(fit)))
  out$predict <- jpredict_cox(fit, extra = extra)
  if (extra) {
    out$survfit_censor_false <- try_aspect(jsurvfit(survfit(fit, censor = FALSE)))
    out$survfit_stype2 <- try_aspect(jsurvfit(survfit(fit, stype = 2, ctype = 1)))
    out$survfit_ctype2 <- try_aspect(jsurvfit(survfit(fit, stype = 2, ctype = 2)))
  }
  if (!is.null(newdata)) {
    out$newdata <- jframe(newdata)
    out$predict_newdata <- jpredict_cox(fit, newdata, extra = TRUE)
    out$survfit_newdata <- try_aspect(jsurvfit(survfit(fit, newdata = newdata)))
    if (extra) {
      out$survfit_newdata_loglog <- try_aspect(
        jsurvfit(survfit(fit, newdata = newdata, conf.type = "log-log")))
    }
  }
  out
}

cox_diagnostics_expected <- function(fit, extra = FALSE) {
  out <- list(coef = jnamed(fit[["coefficients"]]))
  out$residuals <- jresiduals_cox(fit)
  out$zph <- jzph(fit, all = extra)
  out$detail <- jdetail(fit)
  out
}

cox_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL,
                     newdata = NULL, predict = !is.null(newdata), diagnostics = FALSE,
                     extra = FALSE, note = NULL) {
  run_case("coxph", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df),
                   resolve_column_args(args, df))
    if (is.character(call_args$tt)) call_args$tt <- eval(parse(text = call_args$tt))
    fit <- do.call(coxph, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("coxph", name, inputs, cox_core_topic_expected(fit, tt = !is.null(args$tt)), note)
    if (predict) {
      add_case("coxph_predict", name, inputs, cox_predict_expected(fit, newdata, extra), note)
    }
    if (diagnostics) {
      add_case("coxph_diagnostics", name, inputs, cox_diagnostics_expected(fit, extra), note)
    }
  })
}

lung_newdata <- data.frame(age = c(50, 70), sex = c(1, 2))

cox_case("lung_age_sex_efron", "lung", "Surv(time, status) ~ age + sex", newdata = lung_newdata,
         extra = TRUE, diagnostics = TRUE)
cox_case("lung_age_sex_breslow", "lung", "Surv(time, status) ~ age + sex",
         args = list(ties = "breslow"), newdata = lung_newdata, extra = TRUE, diagnostics = TRUE)
cox_case("lung_age_sex_exact", "lung", "Surv(time, status) ~ age + sex",
         args = list(ties = "exact"), diagnostics = TRUE)
cox_case("lung_age_sex_x_true", "lung", "Surv(time, status) ~ age + sex",
         args = list(x = TRUE))
cox_case("lung_age_sex_nocenter_null", "lung", "Surv(time, status) ~ age + sex",
         args = list(nocenter = NULL), newdata = lung_newdata, diagnostics = TRUE)
cox_case("lung_age_sex_weighted", "lung", "Surv(time, status) ~ age + sex",
         args = list(weights = jvec(lung_weights)), newdata = lung_newdata, diagnostics = TRUE)
cox_case("lung_age_sex_weighted_breslow", "lung", "Surv(time, status) ~ age + sex",
         args = list(weights = jvec(lung_weights), ties = "breslow"))
cox_case("lung_age_offset", "lung", "Surv(time, status) ~ age + offset(sex)",
         newdata = lung_newdata, diagnostics = TRUE)
cox_case("lung_age_sex_cluster_inst", "lung", "Surv(time, status) ~ age + sex + cluster(inst)",
         rows = lung_cc, newdata = lung_newdata, diagnostics = TRUE)
cox_case("lung_age_sex_robust", "lung", "Surv(time, status) ~ age + sex",
         args = list(robust = TRUE))
cox_case("lung_age_strata_sex", "lung", "Surv(time, status) ~ age + ph.ecog + strata(sex)",
         rows = lung_cc, newdata = data.frame(age = c(50, 70), ph.ecog = c(0, 1), sex = c(1, 2)), diagnostics = TRUE)
cox_case("lung_age_sex_init_iter0", "lung", "Surv(time, status) ~ age + sex",
         args = list(init = c(0.02, -0.5), control = coxph.control(iter.max = 0)),
         note = "loglik and score at a fixed beta")
cox_case("lung_age_sex_init", "lung", "Surv(time, status) ~ age + sex",
         args = list(init = c(0.02, -0.5)))
cox_case("lung_age_sex_eps", "lung", "Surv(time, status) ~ age + sex",
         args = list(control = coxph.control(eps = 1e-4, iter.max = 3)))
cox_case("lung_factor_ph_ecog", "lung", "Surv(time, status) ~ age + sex + factor(ph.ecog)",
         rows = lung_cc, newdata = data.frame(age = c(50, 70), sex = c(1, 2), ph.ecog = c(0, 2)))
cox_case("lung_interaction", "lung", "Surv(time, status) ~ age * sex + ph.karno",
         rows = lung_cc)
cox_case("lung_log_transform", "lung", "Surv(time, status) ~ log(age) + sex")
cox_case("veteran_celltype_karno_trt", "veteran",
         "Surv(time, status) ~ karno + celltype + trt",
         newdata = data.frame(karno = c(60, 80), celltype = factor(c("smallcell", "adeno"),
           levels = levels(veteran$celltype)), trt = c(1, 2)), diagnostics = TRUE)
cox_case("veteran_celltype_x_true", "veteran", "Surv(time, status) ~ celltype + karno",
         args = list(x = TRUE))
cox_case("veteran_tt_karno", "veteran", "Surv(time, status) ~ karno + tt(karno)",
         args = list(tt = "function(x, t, ...) x * log(t + 20)"),
         note = "tt = function(x, t, ...) x * log(t + 20)")
cox_case("veteran_tt_strata", "veteran", "Surv(time, status) ~ karno + age + tt(karno) + strata(celltype)",
         args = list(tt = "function(x, t, ...) x * log(t + 20)"),
         note = "tt = function(x, t, ...) x * log(t + 20)")
cox_case("pbc_trial_age_edema_strata_sex", "pbc",
         "Surv(time, status == 2) ~ age + edema + log(bili) + strata(sex)", rows = pbc_trial,
         newdata = data.frame(age = c(45, 60), edema = c(0, 0.5), bili = c(1, 3),
                              sex = factor(c("m", "f"), levels = c("m", "f"))), diagnostics = TRUE)
cox_case("pbc_trial_trt_factor", "pbc", "Surv(time, status == 2) ~ factor(trt) + age + albumin",
         rows = pbc_trial)
cox_case("heart_counting_age_surgery_transplant", "heart",
         "Surv(start, stop, event) ~ age + surgery + transplant",
         newdata = data.frame(age = c(-5, 5), surgery = c(0, 1), transplant = factor(c(0, 1))), diagnostics = TRUE)
cox_case("heart_counting_breslow", "heart",
         "Surv(start, stop, event) ~ age + surgery + transplant", args = list(ties = "breslow"))
cox_case("cgd_counting_treat_cluster_id", "cgd",
         "Surv(tstart, tstop, status) ~ treat + inherit + steroids + cluster(id)",
         newdata = data.frame(treat = factor(c("placebo", "rIFN-g"), levels = levels(cgd$treat)),
                              inherit = factor(c("X-linked", "autosomal"), levels = levels(cgd$inherit)),
                              steroids = c(0, 1)), diagnostics = TRUE)
cox_case("cgd_counting_treat_id_robust", "cgd",
         "Surv(tstart, tstop, status) ~ treat + age", args = list(id = "id", robust = TRUE))
cox_case("cgd_counting_strata_enum", "cgd",
         "Surv(tstart, tstop, status) ~ treat + age + strata(enum)")
cox_case("bladder_wlw", "bladder",
         "Surv(stop, event) ~ rx + size + number + strata(enum) + cluster(id)")
cox_case("bladder_ag", "bladder", "Surv(stop, event) ~ rx + size + number + cluster(id)",
         rows = which(bladder$enum == 1))
cox_case("kidney_age_sex", "kidney", "Surv(time, status) ~ age + sex")
cox_case("kidney_age_sex_disease", "kidney", "Surv(time, status) ~ age + sex + disease",
         newdata = data.frame(age = c(30, 50), sex = c(1, 2),
                              disease = factor(c("GN", "PKD"), levels = levels(kidney$disease))), diagnostics = TRUE)
cox_case("ovarian_age_rx", "ovarian", "Surv(futime, fustat) ~ age + rx", newdata =
           data.frame(age = c(50, 65), rx = c(1, 2)), diagnostics = TRUE)
cox_case("ovarian_age_rx_exact", "ovarian", "Surv(futime, fustat) ~ age + rx",
         args = list(ties = "exact"), diagnostics = TRUE)
cox_case("aml_x", "aml", "Surv(time, status) ~ x", newdata = data.frame(x = factor(
  c("Maintained", "Nonmaintained"), levels = levels(aml$x))), diagnostics = TRUE)
cox_case("aml_x_breslow", "aml", "Surv(time, status) ~ x", args = list(ties = "breslow"))
cox_case("aml_x_exact", "aml", "Surv(time, status) ~ x", args = list(ties = "exact"))
cox_case("colon_death_rx_nodes_extent", "colon",
         "Surv(time, status) ~ rx + nodes + extent + surg", rows = colon_death)
cox_case("flchain_500_age_sex_kappa", "flchain", "Surv(futime, death) ~ age + sex + kappa",
         rows = flchain_rows, note = "first 500 rows of flchain")
cox_case("nafld1_1000_age_male_bmi", "nafld1", "Surv(futime, status) ~ age + male + bmi",
         rows = intersect(nafld_rows, which(!is.na(nafld1$bmi))),
         note = "first 1000 rows of nafld1 with non-missing bmi")
cox_case("mgus2_age_sex_mspike", "mgus2", "Surv(futime, death) ~ age + sex + mspike",
         rows = which(!is.na(mgus2$mspike)))
cox_case("myeloid_trt_sex_flt3", "myeloid", "Surv(futime, death) ~ trt + sex + flt3")
cox_case("transplant_ltx_age_abo", "transplant", "Surv(futime, event == \"ltx\") ~ age + abo")
cox_case("jasa_surgery_age", NULL, "Surv(futime, fustat) ~ surgery + age",
         data_ref = "jasa_num")
cox_case("rats_rx_litter_cluster", "rats", "Surv(time, status) ~ rx + cluster(litter)")
cox_case("synthetic_ties_efron", NULL, "Surv(time, status) ~ x + g", data_ref = "synthetic_ties",
         newdata = data.frame(x = c(0, 1), g = factor(c("a", "b"))), extra = TRUE, diagnostics = TRUE)
cox_case("synthetic_ties_breslow", NULL, "Surv(time, status) ~ x + g", data_ref = "synthetic_ties",
         args = list(ties = "breslow"), newdata = data.frame(x = c(0, 1), g = factor(c("a", "b"))),
         extra = TRUE, diagnostics = TRUE)
cox_case("synthetic_ties_exact", NULL, "Surv(time, status) ~ x + g", data_ref = "synthetic_ties",
         args = list(ties = "exact"), diagnostics = TRUE)
cox_case("synthetic_delayed_x", NULL, "Surv(entry, exit, status) ~ x", data_ref = "synthetic_delayed",
         newdata = data.frame(x = c(0, 1)), extra = TRUE, diagnostics = TRUE)
cox_case("synthetic_delayed_x_breslow", NULL, "Surv(entry, exit, status) ~ x",
         data_ref = "synthetic_delayed", args = list(ties = "breslow"))
cox_case("synthetic_delayed_x_exact", NULL, "Surv(entry, exit, status) ~ x",
         data_ref = "synthetic_delayed", args = list(ties = "exact"))
cox_case("synthetic_timefix_true", NULL, "Surv(time, status) ~ 1", data_ref = "synthetic_timefix",
         note = "null model; loglik only")

# multiple-model anova and nested fits
run_case("coxph", "anova_lung_nested", function() {
  f1 <- coxph(Surv(time, status) ~ age, lung)
  f2 <- coxph(Surv(time, status) ~ age + sex, lung)
  f3 <- coxph(Surv(time, status) ~ age + sex + ph.ecog, lung, subset = lung_cc)
  f1b <- coxph(Surv(time, status) ~ age, lung, subset = lung_cc)
  f2b <- coxph(Surv(time, status) ~ age + sex, lung, subset = lung_cc)
  a <- anova(f1b, f2b, f3)
  add_case("coxph", "anova_lung_nested",
           list(dataset = "lung", rows = I(lung_cc),
                formulas = I(c("Surv(time, status) ~ age", "Surv(time, status) ~ age + sex",
                               "Surv(time, status) ~ age + sex + ph.ecog")), args = list()),
           list(anova = list(loglik = jvec(a$loglik), chisq = jvec(a$Chisq), df = jvec(a$Df),
                             p = jvec(a[["Pr(>|Chi|)"]]))))
})

# ---------------------------------------------------------------------------
# coxph with penalised terms
# ---------------------------------------------------------------------------

penal_expected <- function(fit) {
  out <- cox_core_expected(fit)
  out$penalty <- fit$penalty
  out$pterms <- jvec(fit$pterms)
  out$history <- lapply(fit$history, function(h) {
    item <- list(theta = jvec(h$theta), done = isTRUE(all(h$done)))
    if (!is.null(h$history)) {
      item$history <- if (is.matrix(h$history)) jmat_named(h$history) else jvec(h$history)
    }
    if (!is.null(h$c.loglik)) item$c_loglik <- h$c.loglik
    if (!is.null(h$half)) item$half <- h$half
    item
  })
  if (!is.null(fit[["frail"]])) out$frail <- jvec(fit[["frail"]])
  if (!is.null(fit[["fvar"]])) out$fvar <- jvec(fit[["fvar"]])
  out$residuals <- list(
    martingale = try_aspect(jvec(residuals(fit, type = "martingale"))),
    deviance = try_aspect(jvec(residuals(fit, type = "deviance")))
  )
  out$concordance <- try_aspect(jconcordance(concordance(fit)))
  out$predict_lp <- try_aspect(jvec(predict(fit, type = "lp")))
  out$predict_risk <- try_aspect(jvec(predict(fit, type = "risk")))
  out$survfit <- try_aspect(jsurvfit(survfit(fit)))
  out$basehaz_centered <- try_aspect(jbasehaz(fit, TRUE))
  out
}

penal_case <- function(name, dataset, formula, rows = NULL, args = list(), note = NULL) {
  run_case("coxph_penalized", name, function() {
    df <- case_frame(dataset, NULL, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    fit <- do.call(coxph, call_args)
    inputs <- case_inputs(dataset, NULL, rows, formula, args, df)
    add_case("coxph_penalized", name, inputs, penal_expected(fit), note)
  })
}

penal_case("lung_ridge_age_sex_theta1", "lung", "Surv(time, status) ~ ridge(age, sex, theta = 1)")
penal_case("lung_ridge_age_sex_theta5_scaled", "lung",
           "Surv(time, status) ~ ph.ecog + ridge(age, sex, theta = 5, scale = TRUE)", rows = lung_cc)
penal_case("lung_ridge_df2", "lung", "Surv(time, status) ~ ridge(age, sex, ph.karno, df = 2)",
           rows = lung_cc)
penal_case("lung_pspline_age_df4", "lung", "Surv(time, status) ~ pspline(age, df = 4) + sex")
penal_case("lung_pspline_age_df0_aic", "lung", "Surv(time, status) ~ pspline(age, df = 0) + sex",
           note = "df = 0 chooses the degrees of freedom by AIC")
penal_case("lung_pspline_karno_df3_nterm6", "lung",
           "Surv(time, status) ~ pspline(ph.karno, df = 3, nterm = 6) + sex", rows = lung_cc)
penal_case("kidney_frailty_gamma", "kidney", "Surv(time, status) ~ age + sex + frailty(id)")
penal_case("kidney_frailty_gamma_theta_fixed", "kidney",
           "Surv(time, status) ~ age + sex + frailty(id, theta = 0.5)")
penal_case("kidney_frailty_gaussian", "kidney",
           "Surv(time, status) ~ age + sex + frailty(id, dist = \"gauss\")")
penal_case("kidney_frailty_gaussian_df", "kidney",
           "Surv(time, status) ~ age + sex + frailty(id, dist = \"gauss\", df = 10)")
penal_case("kidney_frailty_t", "kidney",
           "Surv(time, status) ~ age + sex + frailty(id, dist = \"t\")")
penal_case("rats_frailty_gamma_litter", "rats", "Surv(time, status) ~ rx + frailty(litter)")
penal_case("rats_frailty_gaussian_litter", "rats",
           "Surv(time, status) ~ rx + frailty(litter, dist = \"gauss\")")
penal_case("cgd_frailty_gamma_id", "cgd",
           "Surv(tstart, tstop, status) ~ treat + frailty(id)")

# ---------------------------------------------------------------------------
# survreg
# ---------------------------------------------------------------------------

jresiduals_survreg <- function(fit, types = c("response", "deviance", "dfbeta", "dfbetas",
                                              "working", "ldcase", "ldresp", "ldshape",
                                              "matrix")) {
  out <- list()
  for (type in types) {
    out[[type]] <- try_aspect({
      r <- residuals(fit, type = type)
      if (is.matrix(r)) {
        list(values = jmat(r),
             colnames = I(if (is.null(colnames(r))) character(0) else colnames(r)))
      } else {
        jvec(r)
      }
    })
  }
  out
}

jpredict_survreg <- function(fit, newdata = NULL, quantiles = TRUE) {
  out <- list()
  pred <- function(type, ...) {
    if (is.null(newdata)) predict(fit, type = type, se.fit = TRUE, ...) else
      predict(fit, newdata = newdata, type = type, se.fit = TRUE, ...)
  }
  for (type in c("response", "lp", "terms")) {
    out[[type]] <- try_aspect({
      p <- pred(type)
      if (is.matrix(p$fit)) {
        item <- list(fit = jmat(p$fit), se_fit = jmat(p$se.fit), colnames = I(colnames(p$fit)))
        if (!is.null(attr(p$fit, "constant"))) item$constant <- attr(p$fit, "constant")
        item
      } else {
        list(fit = jvec(p$fit), se_fit = jvec(p$se.fit))
      }
    })
  }
  if (!quantiles) return(out)
  probs <- c(0.1, 0.5, 0.9)
  for (type in c("quantile", "uquantile")) {
    out[[type]] <- try_aspect({
      p <- pred(type, p = probs)
      list(p = jvec(probs), fit = jmat(p$fit), se_fit = jmat(p$se.fit))
    })
  }
  out
}

survreg_expected <- function(fit, newdata = NULL, full = TRUE) {
  out <- list(
    coef = jnamed(fit[["coefficients"]]),
    coef_names = I(names(fit[["coefficients"]])),
    icoef = jvec(fit[["icoef"]]),
    scale = jvec(fit[["scale"]]),
    var = jmat(fit[["var"]]),
    loglik = jvec(fit[["loglik"]]),
    iter = fit[["iter"]],
    df = fit[["df"]],
    df_residual = fit[["df.residual"]],
    linear_predictors = jvec(fit[["linear.predictors"]]),
    means = jvec(fit[["means"]]),
    dist = fit[["dist"]],
    n = length(fit[["linear.predictors"]])
  )
  if (!is.null(fit[["naive.var"]])) out$naive_var <- jmat(fit[["naive.var"]])
  if (!is.null(fit[["parms"]])) out$parms <- jvec(fit[["parms"]])
  if (!is.null(fit[["x"]])) out$x <- jmat_named(fit[["x"]])
  if (identical(full, FALSE)) return(out)
  if (identical(full, "light")) {
    out$residuals <- jresiduals_survreg(fit, c("response", "deviance", "working"))
    out$predict <- jpredict_survreg(fit, quantiles = FALSE)
  } else {
    out$residuals <- jresiduals_survreg(fit)
    out$predict <- jpredict_survreg(fit)
  }
  if (!is.null(newdata)) {
    out$newdata <- jframe(newdata)
    out$predict_newdata <- jpredict_survreg(fit, newdata)
  }
  out$summary <- try_aspect({
    s <- summary(fit)
    list(table = jmat_named(s$table), chi = s$chi, df = s$df)
  })
  out$anova <- try_aspect({
    a <- anova(fit)
    list(terms = I(rownames(a)), df = jvec(a$Df), deviance = jvec(a$Deviance),
         resid_df = jvec(a[["Resid. Df"]]), loglik = jvec(a[["-2*LL"]]),
         p = jvec(a[["Pr(>Chi)"]]))
  })
  out$concordance <- try_aspect(jconcordance(concordance(fit)))
  out
}

survreg_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL,
                         newdata = NULL, full = TRUE, note = NULL) {
  run_case("survreg", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    fit <- do.call(survreg, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survreg", name, inputs, survreg_expected(fit, newdata, full), note)
  })
}

for (dist in c("weibull", "exponential", "rayleigh", "gaussian", "logistic", "lognormal",
               "loglogistic", "t")) {
  survreg_case(paste0("lung_age_sex_", dist), "lung", "Surv(time, status) ~ age + sex",
               args = list(dist = dist), newdata = lung_newdata,
               full = if (dist %in% c("weibull", "lognormal", "gaussian")) TRUE else "light")
}
survreg_case("lung_age_sex_t_df8", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "t", parms = 8), full = FALSE)
survreg_case("lung_weibull_scale_fixed", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull", scale = 1), newdata = lung_newdata, full = "light")
survreg_case("lung_weibull_strata_sex", "lung", "Surv(time, status) ~ age + strata(sex) + sex",
             args = list(dist = "weibull"), newdata = lung_newdata, full = "light")
survreg_case("lung_lognormal_strata_sex", "lung", "Surv(time, status) ~ age + ph.ecog + strata(sex)",
             rows = lung_cc, args = list(dist = "lognormal"), full = FALSE)
survreg_case("lung_weibull_weighted", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull", weights = jvec(lung_weights)), newdata = lung_newdata, full = "light")
survreg_case("lung_weibull_robust", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull", robust = TRUE), full = FALSE)
survreg_case("lung_weibull_cluster_inst", "lung", "Surv(time, status) ~ age + sex + cluster(inst)",
             rows = lung_cc, args = list(dist = "weibull"), full = FALSE)
survreg_case("lung_weibull_factor_ph_ecog", "lung",
             "Surv(time, status) ~ age + sex + factor(ph.ecog)", rows = lung_cc,
             args = list(dist = "weibull"),
             newdata = data.frame(age = c(50, 70), sex = c(1, 2), ph.ecog = c(0, 2)), full = "light")
survreg_case("lung_weibull_x_true", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull", x = TRUE), full = FALSE)
survreg_case("lung_weibull_init", "lung", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull", init = c(6.5, 0, -0.5), scale = 0.9), full = FALSE)
survreg_case("tobin_gaussian_left", "tobin",
             "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
             args = list(dist = "gaussian"), newdata = data.frame(age = c(40, 55), quant = c(220, 260)))
survreg_case("tobin_logistic_left", "tobin",
             "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
             args = list(dist = "logistic"), full = FALSE)
survreg_case("ovarian_weibull_ecog_rx", "ovarian", "Surv(futime, fustat) ~ ecog.ps + rx",
             args = list(dist = "weibull"), newdata = data.frame(ecog.ps = c(1, 2), rx = c(1, 2)), full = "light")
survreg_case("ovarian_exponential_ecog_rx", "ovarian", "Surv(futime, fustat) ~ ecog.ps + rx",
             args = list(dist = "exponential"), full = FALSE)
survreg_case("ovarian_loglogistic_age", "ovarian", "Surv(futime, fustat) ~ age",
             args = list(dist = "loglogistic"), full = FALSE)
survreg_case("veteran_weibull_celltype_karno", "veteran",
             "Surv(time, status) ~ celltype + karno", args = list(dist = "weibull"),
             newdata = data.frame(celltype = factor(c("squamous", "large"),
                                                    levels = levels(veteran$celltype)),
                                  karno = c(60, 90)), full = "light")
survreg_case("veteran_lognormal_celltype_karno", "veteran",
             "Surv(time, status) ~ celltype + karno", args = list(dist = "lognormal"), full = FALSE)
survreg_case("pbc_trial_weibull", "pbc", "Surv(time, status == 2) ~ age + edema + log(bili)",
             rows = pbc_trial, args = list(dist = "weibull"), full = FALSE)
survreg_case("kidney_weibull_age_sex", "kidney", "Surv(time, status) ~ age + sex",
             args = list(dist = "weibull"), full = FALSE)
survreg_case("interval2_synthetic_weibull", NULL,
             "Surv(left, right, type = \"interval2\") ~ 1", data_ref = "synthetic_interval",
             args = list(dist = "weibull"))
survreg_case("interval2_synthetic_lognormal_g", NULL,
             "Surv(left, right, type = \"interval2\") ~ g", data_ref = "synthetic_interval",
             args = list(dist = "lognormal"))
survreg_case("interval_status_synthetic_weibull", NULL,
             "Surv(time, time2, status, type = \"interval\") ~ 1",
             data_ref = "synthetic_interval_status", args = list(dist = "weibull"))
survreg_case("synthetic_ties_weibull", NULL, "Surv(time, status) ~ x + g",
             data_ref = "synthetic_ties", args = list(dist = "weibull"))
survreg_case("lung_intercept_only_weibull", "lung", "Surv(time, status) ~ 1",
             args = list(dist = "weibull"), full = FALSE)
# dsurvreg / psurvreg / qsurvreg samples
run_case("survreg", "distribution_functions", function() {
  x <- c(0.5, 1, 2, 5, 10)
  p <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  out <- list()
  for (dist in c("weibull", "exponential", "rayleigh", "gaussian", "logistic", "lognormal",
                 "loglogistic")) {
    for (scale in c(0.5, 1, 2)) {
      key <- sprintf("%s_scale%s", dist, scale)
      out[[key]] <- try_aspect(list(
        mean = 1, scale = scale, x = jvec(x), p = jvec(p),
        d = jvec(dsurvreg(x, mean = 1, scale = scale, distribution = dist)),
        p_ = jvec(psurvreg(x, mean = 1, scale = scale, distribution = dist)),
        q = jvec(qsurvreg(p, mean = 1, scale = scale, distribution = dist))
      ))
    }
  }
  out[["t_df4_scale1"]] <- list(
    mean = 1, scale = 1, x = jvec(x), p = jvec(p),
    d = jvec(dsurvreg(x, mean = 1, scale = 1, distribution = "t", parms = 4)),
    p_ = jvec(psurvreg(x, mean = 1, scale = 1, distribution = "t", parms = 4)),
    q = jvec(qsurvreg(p, mean = 1, scale = 1, distribution = "t", parms = 4))
  )
  add_case("survreg", "distribution_functions", list(args = list()), out)
})

# ---------------------------------------------------------------------------
# concordance
# ---------------------------------------------------------------------------

concordance_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL,
                             note = NULL) {
  run_case("concordance", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(object = as.formula(formula), data = df), resolve_column_args(args, df))
    cc <- do.call(concordance, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("concordance", name, inputs, jconcordance(cc), note)
  })
}

concordance_case("lung_age", "lung", "Surv(time, status) ~ age")
concordance_case("lung_age_reverse", "lung", "Surv(time, status) ~ age", args = list(reverse = TRUE))
concordance_case("lung_age_ph_karno", "lung", "Surv(time, status) ~ age + ph.karno", rows = lung_cc)
concordance_case("lung_age_weighted", "lung", "Surv(time, status) ~ age",
                 args = list(weights = jvec(lung_weights)))
for (tw in c("n", "S", "S/G", "n/G2", "I")) {
  concordance_case(paste0("lung_age_timewt_", gsub("/", "", tw)), "lung",
                   "Surv(time, status) ~ age", args = list(timewt = tw, reverse = TRUE))
}
concordance_case("lung_age_ranks", "lung", "Surv(time, status) ~ age",
                 args = list(ranks = TRUE, reverse = TRUE))
concordance_case("lung_age_influence", "lung", "Surv(time, status) ~ age",
                 args = list(influence = 1, reverse = TRUE))
concordance_case("lung_age_strata_sex", "lung", "Surv(time, status) ~ age + strata(sex)",
                 args = list(reverse = TRUE))
concordance_case("lung_age_cluster_inst", "lung", "Surv(time, status) ~ age + cluster(inst)",
                 rows = lung_cc, args = list(reverse = TRUE))
concordance_case("heart_counting_age", "heart", "Surv(start, stop, event) ~ age",
                 args = list(reverse = TRUE))
concordance_case("cgd_counting_age_cluster", "cgd", "Surv(tstart, tstop, status) ~ age + cluster(id)",
                 args = list(reverse = TRUE))
concordance_case("cgd_counting_age_id", "cgd", "Surv(tstart, tstop, status) ~ age",
                 args = list(cluster = "id", reverse = TRUE))
concordance_case("aml_x_numeric", "aml", "Surv(time, status) ~ as.numeric(x)")
concordance_case("synthetic_ties_x", NULL, "Surv(time, status) ~ x", data_ref = "synthetic_ties",
                 args = list(influence = 2))
concordance_case("synthetic_delayed_x", NULL, "Surv(entry, exit, status) ~ x",
                 data_ref = "synthetic_delayed", args = list(reverse = TRUE, influence = 1))
concordance_case("ovarian_age_influence", "ovarian", "Surv(futime, fustat) ~ age",
                 args = list(influence = 3, reverse = TRUE))
concordance_case("veteran_karno_age_multi", "veteran", "Surv(time, status) ~ karno + age",
                 args = list(reverse = TRUE))
concordance_case("lung_numeric_y_age", "lung", "time ~ age",
                 note = "numeric response, no censoring")

run_case("concordance", "coxph_survreg_fits", function() {
  cfit <- coxph(Surv(time, status) ~ age + sex, lung)
  cfit2 <- coxph(Surv(time, status) ~ age, lung)
  sfit <- survreg(Surv(time, status) ~ age + sex, lung)
  add_case("concordance", "coxph_survreg_fits",
           list(dataset = "lung", formula = "Surv(time, status) ~ age + sex", args = list()),
           list(coxph = jconcordance(concordance(cfit)),
                coxph_timewt_S = jconcordance(concordance(cfit, timewt = "S")),
                survreg = jconcordance(concordance(sfit)),
                both = jconcordance(concordance(cfit, cfit2))),
           note = "both = concordance(fit_age_sex, fit_age)")
})

# ---------------------------------------------------------------------------
# aareg
# ---------------------------------------------------------------------------

aareg_case <- function(name, dataset, formula, rows = NULL, args = list(), dfbeta = FALSE) {
  run_case("aareg", name, function() {
    df <- case_frame(dataset, NULL, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    if (dfbeta) call_args$dfbeta <- TRUE
    fit <- do.call(aareg, call_args)
    expected <- list(
      n = jvec(fit$n),
      times = jvec(fit$times),
      nrisk = jvec(fit$nrisk),
      coefficient = jmat_named(fit$coefficient),
      test_statistic = jnamed(fit$test.statistic),
      test_var = jmat(fit$test.var),
      tweight = if (is.matrix(fit$tweight)) jmat(fit$tweight) else jvec(fit$tweight),
      test = fit$test,
      chisq = tryCatch({
        s <- summary(fit)
        list(table = jmat_named(s$table), chisq = s$chisq)
      }, error = function(e) list(r_error = conditionMessage(e)))
    )
    if (!is.null(fit$test.var2)) expected$test_var2 <- jmat(fit$test.var2)
    if (dfbeta) expected$dfbeta <- jarray3(fit$dfbeta)
    inputs <- case_inputs(dataset, NULL, rows, formula, args, df)
    add_case("aareg", name, inputs, expected)
  })
}

aareg_case("lung_age_sex_ph_ecog", "lung", "Surv(time, status) ~ age + sex + ph.ecog", rows = lung_cc)
aareg_case("lung_age_sex_nrisk", "lung", "Surv(time, status) ~ age + sex", args = list(test = "nrisk"))
aareg_case("ovarian_age_ecog_dfbeta", "ovarian", "Surv(futime, fustat) ~ age + ecog.ps", dfbeta = TRUE)
aareg_case("ovarian_age_rx_dfbeta_nrisk", "ovarian", "Surv(futime, fustat) ~ age + rx",
           args = list(test = "nrisk"), dfbeta = TRUE)
aareg_case("lung_weighted", "lung", "Surv(time, status) ~ age + sex",
           args = list(weights = jvec(lung_weights)))
aareg_case("lung_qrtol_taper", "lung", "Surv(time, status) ~ age + sex",
           args = list(taper = 10, qrtol = 1e-8))
aareg_case("kidney_age_sex", "kidney", "Surv(time, status) ~ age + sex")
aareg_case("veteran_karno_celltype", "veteran", "Surv(time, status) ~ karno + celltype")

# ---------------------------------------------------------------------------
# cch on the nwtco case-cohort subset (from ?cch)
# ---------------------------------------------------------------------------

nwtco_cc <- local({
  subcoh <- nwtco$in.subcohort
  selccoh <- with(nwtco, rel == 1 | subcoh == 1)
  ccoh.data <- nwtco[selccoh, ]
  ccoh.data$subcohort <- subcoh[selccoh]
  ccoh.data$histol <- factor(ccoh.data$histol, labels = c("FH", "UH"))
  ccoh.data$stage <- factor(ccoh.data$stage, labels = c("I", "II", "III", "IV"))
  ccoh.data$age <- ccoh.data$age / 12
  ccoh.data$stratum <- ifelse(ccoh.data$instit == 1, 1, 2)
  ccoh.data[, c("seqno", "edrel", "rel", "stage", "histol", "age", "subcohort", "stratum")]
})
register_data("nwtco_cc", nwtco_cc)

cch_case <- function(name, method, formula = "Surv(edrel, rel) ~ stage + histol + age",
                     stratified = FALSE) {
  run_case("cch", name, function() {
    args <- list(subcoh = "subcohort", id = "seqno", cohort.size = 4028, method = method)
    if (stratified) {
      args$stratum <- "stratum"
      args$cohort.size <- c("1" = 3622, "2" = 406)
    }
    call_args <- list(formula = as.formula(formula), data = nwtco_cc,
                      subcoh = ~subcohort, id = ~seqno, cohort.size = args$cohort.size,
                      method = method)
    if (stratified) call_args$stratum <- ~stratum
    fit <- do.call(cch, call_args)
    coef <- fit$coefficients
    if (is.null(names(coef))) names(coef) <- colnames(fit$var)
    if (is.null(names(coef))) names(coef) <- attr(terms(as.formula(formula)), "term.labels")
    expected <- list(
      coef = jnamed(coef),
      coef_names = I(names(coef)),
      var = jmat(fit$var),
      naive_var = if (is.null(fit$naive.var)) NULL else jmat(fit$naive.var),
      subcohort_size = if (is.null(fit$subcohort.size)) NULL else jvec(fit$subcohort.size),
      cohort_size = jvec(fit$cohort.size),
      n_subcohort = if (is.null(fit$subcohort.size)) NULL else sum(fit$subcohort.size),
      method = fit$method,
      summary = try_aspect({
        sm <- summary(fit)
        list(coefficients = jmat_named(sm$coefficients))
      })
    )
    add_case("cch", name, inline_input("nwtco_cc", formula, args), expected,
             note = "nwtco rows with rel == 1 | in.subcohort == 1; age in years")
  })
}

cch_case("prentice", "Prentice")
cch_case("self_prentice", "SelfPrentice")
cch_case("lin_ying", "LinYing")
cch_case("i_borgan", "I.Borgan", stratified = TRUE)
cch_case("ii_borgan", "II.Borgan", stratified = TRUE)
cch_case("prentice_age_only", "Prentice", formula = "Surv(edrel, rel) ~ age")

# ---------------------------------------------------------------------------
# clogit on infert (datasets package, embedded inline)
# ---------------------------------------------------------------------------

register_data("infert", infert[, c("education", "age", "parity", "induced", "case",
                                   "spontaneous", "stratum", "pooled.stratum")])

clogit_case <- function(name, formula, method) {
  run_case("clogit", name, function() {
    fit <- clogit(as.formula(formula), data = infert, method = method)
    expected <- list(
      coef = jnamed(fit$coefficients),
      coef_names = I(names(fit$coefficients)),
      var = jmat(fit$var),
      loglik = jvec(fit$loglik),
      iter = fit$iter,
      score = fit$score,
      n = fit$n,
      nevent = fit$nevent,
      linear_predictors = jvec(fit$linear.predictors),
      method = fit$method,
      summary = jsummary_cox(fit)
    )
    add_case("clogit", name, inline_input("infert", formula, list(method = method)), expected)
  })
}

for (m in c("exact", "approximate", "efron", "breslow")) {
  clogit_case(paste0("infert_spont_induced_", m),
              "case ~ spontaneous + induced + strata(stratum)", m)
}
clogit_case("infert_spont_induced_pooled_exact", "case ~ spontaneous + induced + strata(pooled.stratum)",
            "exact")
clogit_case("infert_age_spont_efron", "case ~ age + spontaneous + strata(stratum)", "efron")

# ---------------------------------------------------------------------------
# finegray
# ---------------------------------------------------------------------------

finegray_case <- function(name, data_ref, formula, rows = NULL, args = list(), cox_formula = NULL) {
  run_case("finegray", name, function() {
    df <- case_frame(NULL, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    fg <- do.call(finegray, call_args)
    expected <- list(frame = jframe(fg))
    if (!is.null(cox_formula)) {
      cfit <- coxph(as.formula(cox_formula), data = fg, weight = fgwt)
      expected$coxph <- cox_core_expected(cfit)
      expected$cox_formula <- cox_formula
    }
    add_case("finegray", name, inline_input(data_ref, formula, args, rows), expected)
  })
}

finegray_case("mgus2_400_pcm", "mgus2_cr", "Surv(etime, event) ~ .", rows = mgus2_rows,
              args = list(etype = "pcm"),
              cox_formula = "Surv(fgstart, fgstop, fgstatus) ~ age + sex")
finegray_case("mgus2_400_death", "mgus2_cr", "Surv(etime, event) ~ .", rows = mgus2_rows,
              args = list(etype = "death"),
              cox_formula = "Surv(fgstart, fgstop, fgstatus) ~ age + sex")
finegray_case("mgus2_400_pcm_strata_sex", "mgus2_cr", "Surv(etime, event) ~ age + strata(sex)",
              rows = mgus2_rows, args = list(etype = "pcm"),
              cox_formula = "Surv(fgstart, fgstop, fgstatus) ~ age")
finegray_case("synthetic_ties_a", "synthetic_ties_mstate", "Surv(time, event) ~ x",
              args = list(etype = "a"), cox_formula = "Surv(fgstart, fgstop, fgstatus) ~ x")
finegray_case("synthetic_ties_b_prefix", "synthetic_ties_mstate", "Surv(time, event) ~ x",
              args = list(etype = "b", prefix = "fg2"))
finegray_case("synthetic_ties_a_count", "synthetic_ties_mstate", "Surv(time, event) ~ x",
              args = list(etype = "a", count = "ncut"))

# ---------------------------------------------------------------------------
# survobrien
# ---------------------------------------------------------------------------

survobrien_case <- function(name, dataset, formula, rows = NULL, args = list()) {
  run_case("survobrien", name, function() {
    df <- case_frame(dataset, NULL, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), args)
    ob <- do.call(survobrien, call_args)
    expected <- list(frame = jframe(ob))
    vars <- all.vars(as.formula(formula))
    covariates <- setdiff(names(ob), c(vars[1:2], "time", "status", ".id.", ".strata."))
    cox_formula <- sprintf("Surv(time, status) ~ %s + strata(.strata.)",
                           paste(covariates, collapse = " + "))
    cfit <- coxph(as.formula(cox_formula), data = ob)
    expected$cox_formula <- cox_formula
    expected$coxph_coef <- jnamed(cfit$coefficients)
    expected$coxph_loglik <- jvec(cfit$loglik)
    inputs <- case_inputs(dataset, NULL, rows, formula, args, df)
    add_case("survobrien", name, inputs, expected)
  })
}

survobrien_case("ovarian_age_rx", "ovarian", "Surv(futime, fustat) ~ age + factor(rx)")
survobrien_case("ovarian_age_ecog", "ovarian", "Surv(futime, fustat) ~ age + ecog.ps")
survobrien_case("lung_age_ph_karno_subset", "lung", "Surv(time, status) ~ age + ph.karno",
                rows = lung_cc[1:60])

# ---------------------------------------------------------------------------
# yates
# ---------------------------------------------------------------------------

jyates <- function(y) {
  out <- list(
    estimate = jframe(y$estimate),
    test = jmat_named(y$test),
    mvar = jmat(y$mvar)
  )
  if (!is.null(y$cmat)) out$cmat <- jmat_named(y$cmat)
  if (!is.null(y$summary)) out$summary <- jframe(as.data.frame(y$summary))
  out
}

yates_case <- function(name, dataset, formula, term, rows = NULL, args = list(), fun = coxph,
                       fit_args = list()) {
  run_case("yates", name, function() {
    df <- case_frame(dataset, NULL, rows)
    fit <- do.call(fun, c(list(formula = as.formula(formula), data = df), fit_args))
    # predict = "risk" (non-linear scale) uses Monte-Carlo standard errors
    set.seed(20240601)
    y <- do.call(yates, c(list(fit = fit, term = term), args))
    expected <- jyates(y)
    if (fun_name(fun) == "coxph") expected$coef <- jnamed(fit$coefficients)
    inputs <- case_inputs(dataset, NULL, rows, formula, c(list(term = term), args, fit_args), df)
    inputs$fit <- fun_name(fun)
    add_case("yates", name, inputs, expected)
  })
}
fun_name <- function(fun) {
  if (identical(fun, coxph)) "coxph" else if (identical(fun, survreg)) "survreg" else "lm"
}

yates_case("veteran_celltype_linear", "veteran", "Surv(time, status) ~ celltype + karno + trt",
           "celltype")
yates_case("veteran_celltype_pop_data", "veteran", "Surv(time, status) ~ celltype + karno + trt",
           "celltype", args = list(population = "data"))
yates_case("veteran_celltype_pop_factorial", "veteran",
           "Surv(time, status) ~ celltype + factor(trt)", "celltype",
           args = list(population = "factorial"))
yates_case("veteran_celltype_pop_sas", "veteran",
           "Surv(time, status) ~ celltype + factor(trt)", "celltype",
           args = list(population = "sas"))
yates_case("veteran_trt_factor", "veteran", "Surv(time, status) ~ celltype + factor(trt) + karno",
           "factor(trt)")
yates_case("lung_ph_ecog_factor", "lung", "Surv(time, status) ~ factor(ph.ecog) + age + sex",
           "factor(ph.ecog)", rows = lung_cc)
yates_case("lung_ph_ecog_factor_pop_data", "lung", "Surv(time, status) ~ factor(ph.ecog) + age + sex",
           "factor(ph.ecog)", rows = lung_cc, args = list(population = "data"))
yates_case("veteran_celltype_lm", "veteran", "time ~ celltype + karno", "celltype", fun = lm)
yates_case("veteran_celltype_predict_risk", "veteran", "Surv(time, status) ~ celltype + karno",
           "celltype", args = list(predict = "risk"))

# ---------------------------------------------------------------------------
# royston, brier
# ---------------------------------------------------------------------------

run_case("royston_brier", "lung_age_sex", function() {
  fit <- coxph(Surv(time, status) ~ age + sex, lung)
  add_case("royston_brier", "lung_age_sex",
           list(dataset = "lung", formula = "Surv(time, status) ~ age + sex",
                args = list(times = jvec(c(100, 300, 500, 800)))),
           list(royston = jnamed(royston(fit)),
                royston_adjust = jnamed(royston(fit, adjust = TRUE)),
                brier = local({
                  b <- brier(fit, times = c(100, 300, 500, 800))
                  list(times = jvec(b$times), brier = jvec(b$brier), rsquared = jvec(b$rsquared))
                }),
                brier_ties_false = local({
                  b <- brier(fit, times = c(100, 300, 500, 800), ties = FALSE)
                  list(times = jvec(b$times), brier = jvec(b$brier), rsquared = jvec(b$rsquared))
                }),
                brier_default_times = local({
                  b <- brier(fit)
                  list(times = jvec(b$times), brier = jvec(b$brier), rsquared = jvec(b$rsquared))
                })))
})
run_case("royston_brier", "veteran_karno_celltype", function() {
  fit <- coxph(Surv(time, status) ~ karno + celltype, veteran)
  add_case("royston_brier", "veteran_karno_celltype",
           list(dataset = "veteran", formula = "Surv(time, status) ~ karno + celltype",
                args = list(times = jvec(c(30, 90, 180, 365)))),
           list(royston = jnamed(royston(fit)),
                brier = local({
                  b <- brier(fit, times = c(30, 90, 180, 365))
                  list(times = jvec(b$times), brier = jvec(b$brier), rsquared = jvec(b$rsquared))
                })))
})
pbc_trial_df <- pbc[pbc_trial, ]
run_case("royston_brier", "pbc_trial_bili_edema", function() {
  fit <- coxph(Surv(time, status == 2) ~ log(bili) + edema + age, pbc_trial_df)
  add_case("royston_brier", "pbc_trial_bili_edema",
           list(dataset = "pbc", rows = I(pbc_trial),
                formula = "Surv(time, status == 2) ~ log(bili) + edema + age",
                args = list(times = jvec(c(365, 730, 1460)))),
           list(royston = jnamed(royston(fit)),
                brier = local({
                  b <- brier(fit, times = c(365, 730, 1460))
                  list(times = jvec(b$times), brier = jvec(b$brier), rsquared = jvec(b$rsquared))
                })))
})

# ---------------------------------------------------------------------------
# rttright
# ---------------------------------------------------------------------------

rttright_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL) {
  run_case("rttright", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    w <- do.call(rttright, call_args)
    expected <- if (is.matrix(w)) list(weights = jmat(w), times = jvec(as.numeric(colnames(w)))) else
      list(weights = jvec(w))
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("rttright", name, inputs, expected)
  })
}

rttright_case("aml_1", "aml", "Surv(time, status) ~ 1")
rttright_case("aml_x", "aml", "Surv(time, status) ~ x")
rttright_case("aml_1_times", "aml", "Surv(time, status) ~ 1", args = list(times = c(12, 24, 36)))
rttright_case("aml_x_times", "aml", "Surv(time, status) ~ x", args = list(times = c(12, 24, 36)))
rttright_case("lung_sex_times", "lung", "Surv(time, status) ~ sex",
              args = list(times = c(100, 300, 500)))
rttright_case("lung_1_weighted", "lung", "Surv(time, status) ~ 1",
              args = list(weights = jvec(lung_weights), times = c(100, 300, 500)))
rttright_case("synthetic_ties_g", NULL, "Surv(time, status) ~ g", data_ref = "synthetic_ties")
rttright_case("synthetic_ties_mstate", NULL, "Surv(time, event) ~ 1", data_ref = "synthetic_ties_mstate",
              args = list(times = c(2, 5)))

# ---------------------------------------------------------------------------
# pseudo, residuals.survfit, survfit0
# ---------------------------------------------------------------------------

pseudo_case <- function(name, dataset, formula, times, rows = NULL, data_ref = NULL, args = list(),
                        types = c("pstate", "cumhaz", "rmst", "auc", "sojourn", "survival")) {
  run_case("pseudo", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    fit <- do.call(survfit, call_args)
    expected <- list(times = jvec(times))
    for (type in types) {
      expected[[paste0("pseudo_", type)]] <- try_aspect({
        p <- pseudo(fit, times = times, type = type)
        if (length(dim(p)) == 3) jarray3(p) else if (is.matrix(p)) jmat(p) else jvec(p)
      })
      expected[[paste0("residuals_", type)]] <- try_aspect({
        r <- residuals(fit, times = times, type = type)
        if (length(dim(r)) == 3) jarray3(r) else if (is.matrix(r)) jmat(r) else jvec(r)
      })
    }
    expected$survfit0 <- try_aspect(jsurvfit(survfit0(fit)))
    inputs <- case_inputs(dataset, data_ref, rows, formula, c(args, list(times = jvec(times))), df)
    add_case("pseudo", name, inputs, expected)
  })
}

pseudo_case("aml_1", "aml", "Surv(time, status) ~ 1", c(12, 24, 48))
pseudo_case("aml_x", "aml", "Surv(time, status) ~ x", c(12, 24, 48))
pseudo_case("aml_1_single_time", "aml", "Surv(time, status) ~ 1", 24)
pseudo_case("lung_1", "lung", "Surv(time, status) ~ 1", c(100, 300, 500))
pseudo_case("lung_sex", "lung", "Surv(time, status) ~ sex", c(100, 300, 500))
pseudo_case("synthetic_ties_g", NULL, "Surv(time, status) ~ g", c(2, 5, 8),
            data_ref = "synthetic_ties")
pseudo_case("synthetic_delayed", NULL, "Surv(entry, exit, status) ~ 1", c(3, 6, 9),
            data_ref = "synthetic_delayed")
pseudo_case("synthetic_ties_mstate", NULL, "Surv(time, event) ~ 1", c(2, 5, 8),
            data_ref = "synthetic_ties_mstate")
pseudo_case("mgus2_150_mstate", NULL, "Surv(etime, event) ~ 1", c(60, 120, 240),
            data_ref = "mgus2_cr", rows = seq_len(150), types = c("pstate", "cumhaz", "sojourn"))
pseudo_case("synthetic_istate", NULL, "Surv(tstart, tstop, event) ~ 1", c(3, 6, 9),
            data_ref = "synthetic_mstate", args = list(id = "id", istate = "istate"))
pseudo_case("cgd_counting_id", "cgd", "Surv(tstart, tstop, status) ~ 1", c(100, 250),
            args = list(id = "id"))

# ---------------------------------------------------------------------------
# survcheck
# ---------------------------------------------------------------------------

jsurvcheck <- function(sc) {
  list(
    states = I(sc$states),
    transitions = jmat_named(unclass(sc$transitions)),
    events = jmat_named(unclass(sc$events)),
    flag = jnamed(sc$flag),
    istate = I(as.character(sc$istate)),
    n = jnamed(sc$n),
    overlap = if (is.null(sc$overlap)) NULL else lapply(sc$overlap, function(v) jvec(v)),
    gap = if (is.null(sc$gap)) NULL else lapply(sc$gap, function(v) jvec(v)),
    teleport = if (is.null(sc$teleport)) NULL else lapply(sc$teleport, function(v) jvec(v)),
    jump = if (is.null(sc$jump)) NULL else lapply(sc$jump, function(v) jvec(v)),
    duplicate = if (is.null(sc$duplicate)) NULL else lapply(sc$duplicate, function(v) jvec(v))
  )
}

survcheck_case <- function(name, data_ref, formula, args = list(), rows = NULL, dataset = NULL) {
  run_case("survcheck", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    sc <- do.call(survcheck, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survcheck", name, inputs, jsurvcheck(sc))
  })
}

register_data("synthetic_overlap", data.frame(
  id = c(1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6),
  tstart = c(0, 5, 4, 0, 6, 0, 3, 0, 2, 0, 4, 0),
  tstop = c(5, 10, 12, 4, 9, 3, 7, 2, 6, 4, 4, 3),
  event = factor(c("b", "c", "censor", "b", "c", "censor", "c", "b", "censor", "b", "c", "b"),
                 levels = c("censor", "b", "c")),
  istate = c("a", "b", "b", "a", "b", "a", "a", "a", "b", "a", "b", "a")
))
survcheck_case("myeloid_ms_trt", "myeloid_ms", "Surv(tstart, tstop, event) ~ trt", args = list(id = "id"))
survcheck_case("myeloid_ms_1", "myeloid_ms", "Surv(tstart, tstop, event) ~ 1", args = list(id = "id"))
survcheck_case("synthetic_istate", "synthetic_mstate", "Surv(tstart, tstop, event) ~ 1",
               args = list(id = "id", istate = "istate"))
survcheck_case("synthetic_overlap_gaps", "synthetic_overlap", "Surv(tstart, tstop, event) ~ 1",
               args = list(id = "id", istate = "istate"))
survcheck_case("synthetic_overlap_no_istate", "synthetic_overlap", "Surv(tstart, tstop, event) ~ 1",
               args = list(id = "id"))
survcheck_case("cgd_counting", NULL, "Surv(tstart, tstop, status) ~ 1", args = list(id = "id"),
               dataset = "cgd")
survcheck_case("heart_counting", NULL, "Surv(start, stop, event) ~ 1", args = list(id = "id"),
               dataset = "heart")

# ---------------------------------------------------------------------------
# survSplit, survcondense
# ---------------------------------------------------------------------------

survsplit_case <- function(name, dataset, formula, rows = NULL, args = list(), data_ref = NULL) {
  run_case("survSplit", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), args)
    out <- do.call(survSplit, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survSplit", name, inputs, list(frame = jframe(out)))
  })
}

survsplit_case("lung_50_cut", "lung", "Surv(time, status) ~ .", rows = 1:50,
               args = list(cut = c(100, 300, 500), episode = "epi"))
survsplit_case("lung_50_cut_start_end_event", "lung", "Surv(time, status) ~ age + sex", rows = 1:50,
               args = list(cut = c(200, 400), start = "tstart", end = "tstop", event = "died",
                           episode = "period", id = "subject"))
survsplit_case("cgd_counting_cut", "cgd", "Surv(tstart, tstop, status) ~ treat + age",
               rows = which(cgd$id <= 20), args = list(cut = c(100, 200, 300), episode = "epi"))
survsplit_case("synthetic_delayed_cut", NULL, "Surv(entry, exit, status) ~ x",
               data_ref = "synthetic_delayed", args = list(cut = c(3, 6), episode = "epi"))
survsplit_case("aml_zero_cut", "aml", "Surv(time, status) ~ x", args = list(cut = c(0, 10, 20), zero = 0))
survsplit_case("synthetic_mstate_cut", NULL, "Surv(tstart, tstop, event) ~ x",
               data_ref = "synthetic_mstate", args = list(cut = c(4, 8), episode = "epi"))

survcondense_case <- function(name, data_ref, formula, args = list(), build = NULL) {
  run_case("survcondense", name, function() {
    df <- inline_data[[data_ref]]
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    out <- do.call(survcondense, call_args)
    add_case("survcondense", name, inline_input(data_ref, formula, args), list(frame = jframe(out)))
  })
}

register_data("lung_split", local({
  d <- survSplit(Surv(time, status) ~ age + sex + inst, lung[1:30, ], cut = c(100, 200, 300, 500),
                 episode = "epi", id = "subject")
  d
}))
survcondense_case("lung_split_age_sex", "lung_split", "Surv(tstart, time, status) ~ age + sex",
                  args = list(id = "subject"))
survcondense_case("lung_split_age_sex_epi", "lung_split", "Surv(tstart, time, status) ~ age + sex + epi",
                  args = list(id = "subject"))
survcondense_case("lung_split_start_end", "lung_split", "Surv(tstart, time, status) ~ age",
                  args = list(id = "subject", start = "t1", end = "t2", event = "died"))
register_data("cgd_split", local({
  d <- cgd[cgd$id <= 10, c("id", "tstart", "tstop", "status", "treat", "age", "enum")]
  survSplit(Surv(tstart, tstop, status) ~ treat + age + id, d, cut = c(50, 150, 250, 350),
            episode = "epi")
}))
survcondense_case("cgd_split_treat_age", "cgd_split", "Surv(tstart, tstop, status) ~ treat + age",
                  args = list(id = "id"))

# ---------------------------------------------------------------------------
# tmerge (cgd0 vignette example) and tdc/cumtdc/event/cumevent on a small set
# ---------------------------------------------------------------------------

run_case("tmerge", "cgd0_vignette", function() {
  newcgd <- tmerge(data1 = cgd0[, 1:13], data2 = cgd0, id = id, tstop = futime)
  steps <- list(after_base = jframe(newcgd))
  for (k in 1:7) {
    newcgd <- eval(parse(text = sprintf(
      "tmerge(newcgd, cgd0, id = id, infect = event(etime%d))", k)))
  }
  steps$after_events <- jframe(newcgd)
  newcgd <- tmerge(newcgd, newcgd, id, enum = cumtdc(tstart))
  steps$final <- jframe(newcgd)
  steps$tcount <- jmat_named(attr(newcgd, "tcount"))
  steps$matches_cgd <- isTRUE(all.equal(newcgd$tstart, cgd$tstart)) &&
    isTRUE(all.equal(newcgd$tstop, cgd$tstop)) && isTRUE(all.equal(newcgd$infect, cgd$status))
  add_case("tmerge", "cgd0_vignette",
           list(dataset = "cgd0", args = list(),
                steps = I(c("tmerge(cgd0[,1:13], cgd0, id=id, tstop=futime)",
                            "infect = event(etime1) ... event(etime7)",
                            "enum = cumtdc(tstart)"))),
           steps)
})

register_data("tmerge_base", data.frame(id = 1:4, futime = c(10, 20, 15, 30),
                                        death = c(1, 0, 1, 1), sex = c(1, 2, 1, 2)))
register_data("tmerge_long", data.frame(
  id = c(1, 1, 2, 2, 2, 3, 4, 4, 4, 4),
  time = c(2, 5, 3, 8, 21, 4, 5, 10, 15, 25),
  lab = c(1.1, 1.5, 0.9, 1.2, 1.8, 2.0, 0.5, 0.7, 1.4, 1.9),
  infection = c(1, 0, 1, 1, 0, 1, 0, 1, 1, 1)
))

run_case("tmerge", "synthetic_tdc_cumtdc_event_cumevent", function() {
  base <- inline_data[["tmerge_base"]]
  long <- inline_data[["tmerge_long"]]
  d1 <- tmerge(base, base, id = id, death = event(futime, death))
  d2 <- tmerge(d1, long, id = id, lab = tdc(time, lab))
  d3 <- tmerge(d2, long, id = id, nlab = cumtdc(time))
  d4 <- tmerge(d3, long, id = id, infect = event(time, infection))
  d5 <- tmerge(d4, long, id = id, ninfect = cumevent(time, infection))
  d6 <- tmerge(d1, long, id = id, lab = tdc(time, lab, init = 0.5))
  d7 <- tmerge(d1, long, id = id, lab = tdc(time, lab), options = list(tdcstart = -1))
  add_case("tmerge", "synthetic_tdc_cumtdc_event_cumevent",
           list(data_ref = "tmerge_base", data_ref2 = "tmerge_long", args = list()),
           list(step1_death_event = jframe(d1),
                step2_lab_tdc = jframe(d2),
                step3_nlab_cumtdc = jframe(d3),
                step4_infect_event = jframe(d4),
                step5_ninfect_cumevent = jframe(d5),
                tdc_init = jframe(d6),
                tdc_tdcstart = jframe(d7),
                tcount_final = jmat_named(attr(d5, "tcount"))))
})

run_case("tmerge", "pbcseq_20_vignette", function() {
  pbc1 <- pbc[pbc$id <= 20, c("id", "time", "status", "trt", "age", "sex")]
  seq1 <- pbcseq[pbcseq$id <= 20, c("id", "day", "bili", "albumin", "protime", "edema")]
  register_data("pbc_20", pbc1)
  register_data("pbcseq_20", seq1)
  pbc2 <- tmerge(pbc1, pbc1, id = id, death = event(time, status == 2))
  pbc2 <- tmerge(pbc2, seq1, id = id, bili = tdc(day, bili), albumin = tdc(day, albumin),
                 protime = tdc(day, protime), edema = tdc(day, edema))
  add_case("tmerge", "pbcseq_20_vignette",
           list(data_ref = "pbc_20", data_ref2 = "pbcseq_20", args = list()),
           list(frame = jframe(pbc2), tcount = jmat_named(attr(pbc2, "tcount"))))
})

# ---------------------------------------------------------------------------
# neardate
# ---------------------------------------------------------------------------

run_case("neardate", "doc_example", function() {
  id1 <- c(1, 1, 2, 2, 2, 3, 4, 4, 5)
  y1 <- c(10, 20, 5, 15, 25, 30, 4, 12, 7)
  id2 <- c(1, 1, 1, 2, 2, 3, 4, 4, 4, 6)
  y2 <- c(8, 12, 22, 4, 26, 30, 3, 11, 13, 1)
  expected <- list()
  for (best in c("after", "prior")) {
    expected[[best]] <- jvec(neardate(id1, id2, y1, y2, best = best))
    expected[[paste0(best, "_nomatch0")]] <- jvec(neardate(id1, id2, y1, y2, best = best, nomatch = 0))
  }
  expected$after_dates <- jvec(neardate(id1, id2, as.Date(y1, origin = "2000-01-01"),
                                        as.Date(y2, origin = "2000-01-01")))
  add_case("neardate", "doc_example",
           list(args = list(id1 = jvec(id1), y1 = jvec(y1), id2 = jvec(id2), y2 = jvec(y2))),
           expected)
})

# ---------------------------------------------------------------------------
# tcut + pyears, survexp, ratetableDate
# ---------------------------------------------------------------------------

jpyears <- function(p) {
  wrap <- function(v) if (is.array(v) && length(dim(v)) == 1) jvec(v) else if (is.matrix(v)) jmat(v) else
    if (is.array(v)) jarray3(v) else jvec(v)
  out <- list(
    pyears = wrap(p$pyears),
    n = wrap(p$n),
    offtable = p$offtable,
    observations = p$observations,
    tcut = p$tcut
  )
  if (!is.null(p$event)) out$event <- wrap(p$event)
  if (!is.null(p$expected)) out$expected <- wrap(p$expected)
  if (!is.null(dimnames(p$pyears))) {
    out$dimnames <- lapply(dimnames(p$pyears), function(v) I(v))
    out$dim <- jvec(dim(p$pyears))
  }
  if (!is.null(p$data)) out$data <- jframe(p$data)
  out
}

register_data("hearta", local({
  h <- by(heart, heart$id, function(x) x[x$stop == max(x$stop), ])
  h <- do.call("rbind", h)
  rownames(h) <- NULL
  h
}))

pyears_case <- function(name, data_ref, formula, args = list(), dataset = NULL, rows = NULL,
                        note = NULL) {
  run_case("pyears", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    if (!is.null(call_args$rmap)) call_args$rmap <- str2lang(call_args$rmap)
    if (!is.null(call_args$ratetable) && is.character(call_args$ratetable)) {
      call_args$ratetable <- get(call_args$ratetable)
    }
    p <- do.call(pyears, call_args)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("pyears", name, inputs, jpyears(p), note)
  })
}

pyears_case("hearta_age_surgery", "hearta",
            "Surv(stop / 365.25, event) ~ cut(age + 48, c(0, 50, 60, 70, 100)) + surgery",
            args = list(scale = 1))
pyears_case("hearta_age_surgery_data_frame", "hearta",
            "Surv(stop / 365.25, event) ~ cut(age + 48, c(0, 50, 60, 70, 100)) + surgery",
            args = list(scale = 1, data.frame = TRUE))
pyears_case("hearta_surgery_scale365", "hearta", "Surv(stop, event) ~ surgery",
            args = list(scale = 365.25))
pyears_case("hearta_no_event", "hearta", "stop ~ surgery", args = list(scale = 365.25))

register_data("lung_py", local({
  d <- lung[lung_cc, c("time", "status", "age", "sex", "ph.ecog")]
  d$entry <- as.Date("1996-01-01") + seq(0, by = 30, length.out = nrow(d))
  d$agedays <- d$age * 365.25
  d
}))
pyears_case("lung_tcut_age_survexp_us", "lung_py",
            "Surv(time, status) ~ tcut(agedays, c(0, 50, 60, 70, 100) * 365.25, labels = c(\"<50\", \"50-60\", \"60-70\", \"70+\")) + sex",
            args = list(ratetable = "survexp.us",
                        rmap = "list(age = agedays, sex = sex, year = entry)", scale = 365.25),
            note = "rmap = list(age = agedays, sex = sex, year = entry)")
pyears_case("lung_tcut_age_year_survexp_us", "lung_py",
            "Surv(time, status) ~ tcut(agedays, c(0, 60, 70, 100) * 365.25) + tcut(entry, as.Date(c(\"1995-01-01\", \"1997-01-01\", \"1999-01-01\")))",
            args = list(ratetable = "survexp.us",
                        rmap = "list(age = agedays, sex = sex, year = entry)", scale = 365.25),
            note = "tcut on both age and calendar time")
pyears_case("lung_tcut_age_sex_no_ratetable", "lung_py",
            "Surv(time, status) ~ tcut(agedays, c(0, 50, 60, 70, 100) * 365.25) + sex",
            args = list(scale = 365.25))
pyears_case("lung_tcut_weighted", "lung_py",
            "Surv(time, status) ~ tcut(agedays, c(0, 60, 70, 100) * 365.25)",
            args = list(scale = 365.25, weights = jvec(rep(c(1, 2), length.out = length(lung_cc)))))

run_case("pyears", "tcut_basis", function() {
  x <- c(10, 25, 40, 55, 70)
  tc <- tcut(x, c(0, 20, 50, 100))
  tc2 <- tcut(x, c(0, 20, 50, 100), labels = c("young", "mid", "old"))
  tc3 <- tcut(x, 3)
  add_case("pyears", "tcut_basis",
           list(args = list(x = jvec(x), breaks = jvec(c(0, 20, 50, 100)))),
           list(values = jvec(as.numeric(unclass(tc))), labels = I(attr(tc, "labels")),
                cutpoints = jvec(attr(tc, "cutpoints")),
                labelled = I(attr(tc2, "labels")),
                scalar_breaks_cutpoints = jvec(attr(tc3, "cutpoints")),
                scalar_breaks_values = jvec(as.numeric(unclass(tc3)))),
           note = "a tcut object keeps the raw values; cutpoints/labels are attributes")
})

# survexp -------------------------------------------------------------------

jsurvexp <- function(sx) {
  out <- list(time = jvec(sx$time), n = jvec(sx$n))
  if (is.matrix(sx$surv)) {
    out$surv <- jmat(sx$surv)
    out$strata_names <- I(if (is.null(colnames(sx$surv))) character(0) else colnames(sx$surv))
  } else {
    out$surv <- jvec(sx$surv)
  }
  if (!is.null(sx$n.risk)) out$n_risk <- if (is.matrix(sx$n.risk)) jmat(sx$n.risk) else jvec(sx$n.risk)
  if (!is.null(sx$method)) out$method <- sx$method
  out
}

survexp_case <- function(name, data_ref, formula, args, dataset = NULL, rows = NULL, individual = FALSE,
                         note = NULL) {
  run_case("survexp", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    if (!is.null(call_args$rmap)) call_args$rmap <- str2lang(call_args$rmap)
    if (is.character(call_args$ratetable)) {
      call_args$ratetable <- get(call_args$ratetable)
    }
    sx <- do.call(survexp, call_args)
    expected <- if (individual) list(surv = jvec(sx)) else jsurvexp(sx)
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survexp", name, inputs, expected, note)
  })
}

survexp_times <- c(0, 182.5, 365, 730, 1095)
survexp_case("lung_ederer_us", "lung_py", "~ 1",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times)),
             note = "method = ederer (default) with survexp.us")
survexp_case("lung_ederer_us_by_sex", "lung_py", "~ sex",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times)))
survexp_case("lung_hakulinen_us", "lung_py", "Surv(time, status) ~ 1",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times), method = "hakulinen"))
survexp_case("lung_hakulinen_us_by_sex", "lung_py", "Surv(time, status) ~ sex",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times), method = "hakulinen"))
survexp_case("lung_conditional_us", "lung_py", "Surv(time, status) ~ 1",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times), method = "conditional"))
survexp_case("lung_conditional_us_by_sex", "lung_py", "Surv(time, status) ~ sex",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times), method = "conditional"))
survexp_case("lung_individual_us", "lung_py", "time ~ 1",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)",
                  cohort = FALSE), individual = TRUE,
             note = "cohort = FALSE returns one expected survival per subject")
survexp_case("lung_ederer_usr", "lung_py", "~ 1",
             list(ratetable = "survexp.usr",
                  rmap = "list(age = agedays, sex = sex, year = entry, race = \"white\")",
                  times = jvec(survexp_times)))
survexp_case("lung_ederer_mn", "lung_py", "~ 1",
             list(ratetable = "survexp.mn", rmap = "list(age = agedays, sex = sex, year = entry)",
                  times = jvec(survexp_times)))
survexp_case("lung_ederer_us_default_times", "lung_py", "Surv(time, status) ~ 1",
             list(ratetable = "survexp.us", rmap = "list(age = agedays, sex = sex, year = entry)"),
             note = "times default to the unique follow-up times of the response")

run_case("survexp", "lung_coxph_ratetable", function() {
  df <- lung[lung_cc, ]
  fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, df)
  sx <- survexp(~ sex, data = df, ratetable = fit, times = survexp_times)
  sx2 <- survexp(~ 1, data = df, ratetable = fit, times = survexp_times)
  sx3 <- survexp(time ~ 1, data = df, ratetable = fit, cohort = FALSE)
  add_case("survexp", "lung_coxph_ratetable",
           list(dataset = "lung", rows = I(lung_cc), formula = "~ sex",
                args = list(ratetable = "coxph(Surv(time, status) ~ age + sex + ph.ecog)",
                            times = jvec(survexp_times))),
           list(by_sex = jsurvexp(sx), overall = jsurvexp(sx2), individual = jvec(sx3),
                coef = jnamed(fit$coefficients)))
})

run_case("survexp", "ratetableDate", function() {
  d <- as.Date(c("1960-01-01", "1990-06-15", "2000-02-29", "2020-12-31"))
  add_case("survexp", "ratetableDate",
           list(args = list(dates = I(format(d)), numeric = jvec(as.numeric(d)))),
           list(from_date = jvec(ratetableDate(d)),
                from_numeric = jvec(ratetableDate(as.numeric(d))),
                from_posixct = jvec(ratetableDate(as.POSIXct(d, tz = "UTC"))),
                from_character_years = jvec(ratetableDate(c(1960, 1990.5)))))
})

run_case("survexp", "survexp_us_table", function() {
  rt <- survexp.us
  dn <- dimnames(rt)
  add_case("survexp", "survexp_us_table", list(args = list()),
           list(dim = jvec(dim(rt)), dimnames = lapply(dn, function(v) I(v)),
                type = jvec(attr(rt, "type")),
                cutpoints = lapply(attr(rt, "cutpoints"), function(v) if (is.null(v)) NULL else jvec(v)),
                sample = list(
                  age0_male_1990 = rt[1, 1, "1990"],
                  age50_female_2000 = rt["50", "female", "2000"],
                  age100_male_1940 = rt["100", "male", "1940"],
                  sum = sum(rt)
                ),
                summary = list(
                  usr_dim = jvec(dim(survexp.usr)),
                  usr_dimnames = lapply(dimnames(survexp.usr), function(v) I(v)),
                  usr_sum = sum(survexp.usr),
                  mn_dim = jvec(dim(survexp.mn)),
                  mn_dimnames = lapply(dimnames(survexp.mn), function(v) I(v)),
                  mn_sum = sum(survexp.mn)
                )))
})

# ---------------------------------------------------------------------------
# small utilities: cipoisson, bounded links, nsk, pspline basis, aeqSurv,
# Surv types, statefig
# ---------------------------------------------------------------------------

run_case("utilities", "cipoisson", function() {
  k <- c(0, 1, 5, 10, 25, 100)
  time <- c(1, 2, 3, 0.5, 10, 100)
  out <- list()
  for (method in c("exact", "anscombe")) {
    ci <- cipoisson(k, time = time, p = 0.95, method = method)
    out[[method]] <- jmat(ci)
    ci90 <- cipoisson(k, time = time, p = 0.90, method = method)
    out[[paste0(method, "_p90")]] <- jmat(ci90)
  }
  out$scalar_k5 <- jvec(cipoisson(5))
  out$scalar_k0_time2 <- jvec(cipoisson(0, time = 2))
  add_case("utilities", "cipoisson", list(args = list(k = jvec(k), time = jvec(time))), out)
})

run_case("utilities", "bounded_links", function() {
  x <- c(0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1)
  add_case("utilities", "bounded_links", list(args = list(x = jvec(x), edge = 0.05)),
           list(blogit = jvec(blogit()$linkfun(x)),
                bprobit = jvec(bprobit()$linkfun(x)),
                bcloglog = jvec(bcloglog()$linkfun(x)),
                blog = jvec(blog()$linkfun(x)),
                blogit_edge01 = jvec(blogit(0.1)$linkfun(x)),
                blog_edge001 = jvec(blog(0.01)$linkfun(x)),
                blogit_linkinv = jvec(blogit()$linkinv(c(-3, 0, 3))),
                bprobit_linkinv = jvec(bprobit()$linkinv(c(-2, 0, 2))),
                bcloglog_linkinv = jvec(bcloglog()$linkinv(c(-2, 0, 1))),
                blog_linkinv = jvec(blog()$linkinv(c(-2, -1, 0)))))
})

run_case("utilities", "nsk_basis", function() {
  x <- c(20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
  b1 <- nsk(x, df = 4)
  b2 <- nsk(x, knots = c(35, 50, 65), Boundary.knots = c(20, 80))
  b3 <- nsk(x, df = 3, intercept = TRUE)
  b4 <- nsk(x, df = 4, b = 0.1)
  enc <- function(b) list(values = jmat(unclass(b)), knots = jvec(attr(b, "knots")),
                          boundary_knots = jvec(attr(b, "Boundary.knots")),
                          intercept = attr(b, "intercept"))
  add_case("utilities", "nsk_basis", list(args = list(x = jvec(x))),
           list(df4 = enc(b1), knots_35_50_65 = enc(b2), df3_intercept = enc(b3), df4_b01 = enc(b4),
                lung_age_df3 = enc(nsk(lung$age, df = 3))))
})

run_case("utilities", "pspline_basis", function() {
  x <- lung$age
  p1 <- pspline(x, df = 4)
  p2 <- pspline(x, df = 4, nterm = 8)
  p3 <- pspline(x, degree = 2, nterm = 6, df = 3)
  p4 <- pspline(x, df = 0)
  p5 <- pspline(x, theta = 0.5)
  enc <- function(p) list(values = jmat(unclass(p)), nterm = attr(p, "nterm"),
                          degree = attr(p, "degree"), knots = jvec(attr(p, "knots")),
                          boundary_knots = jvec(attr(p, "Boundary.knots")),
                          intercept = attr(p, "intercept"),
                          pparm = jvec(attr(p, "pparm")), eps = attr(p, "eps"),
                          cparm = lapply(attr(p, "cparm"), function(v) if (is.numeric(v)) jvec(v) else v),
                          df = attr(p, "df"),
                          pfun_theta_half = local({
                            pf <- attr(p, "pfun")(rep(1, ncol(p)), 0.5, 0, attr(p, "pparm"))
                            list(penalty = pf$penalty, first = jvec(pf$first),
                                 second = jvec(pf$second))
                          }))
  add_case("utilities", "pspline_basis", list(dataset = "lung", args = list(x = "age")),
           list(df4 = enc(p1), df4_nterm8 = enc(p2), degree2_nterm6_df3 = enc(p3), df0 = enc(p4),
                theta05 = enc(p5)))
})

run_case("utilities", "aeqSurv", function() {
  s <- Surv(synthetic_timefix$time, synthetic_timefix$status)
  s2 <- Surv(c(1, 1 + 1e-14, 2, 2 + 1e-9, 3), c(1, 1, 0, 1, 1))
  s3 <- Surv(c(0, 0 + 1e-14, 1, 1.5), c(1, 1 + 1e-14, 2, 2.5), c(1, 1, 0, 1))
  add_case("utilities", "aeqSurv",
           list(data_ref = "synthetic_timefix", args = list(
             time2 = jvec(c(1, 1 + 1e-14, 2, 2 + 1e-9, 3)), status2 = jvec(c(1, 1, 0, 1, 1)),
             start3 = jvec(c(0, 0 + 1e-14, 1, 1.5)), stop3 = jvec(c(1, 1 + 1e-14, 2, 2.5)),
             status3 = jvec(c(1, 1, 0, 1)))),
           list(synthetic_timefix = jsurv(aeqSurv(s)),
                right_1e9 = jsurv(aeqSurv(s2)),
                right_1e9_tol_1e8 = jsurv(aeqSurv(s2, tolerance = 1e-8)),
                counting = jsurv(aeqSurv(s3))))
})

run_case("utilities", "surv_types", function() {
  out <- list(
    right = jsurv(Surv(c(1, 2, 3, 4), c(1, 0, 1, 1))),
    right_12 = jsurv(Surv(c(1, 2, 3, 4), c(2, 1, 2, 2))),
    right_logical = jsurv(Surv(c(1, 2, 3, 4), c(TRUE, FALSE, TRUE, TRUE))),
    left = jsurv(Surv(c(1, 2, 3, 4), c(1, 0, 1, 1), type = "left")),
    interval = jsurv(Surv(c(1, 2, 3, 4, 5), c(2, NA, 6, NA, 7), c(3, 0, 3, 1, 2), type = "interval")),
    interval2 = jsurv(Surv(c(1, NA, 3, 4, 5), c(2, 3, NA, 4, 8), type = "interval2")),
    counting = jsurv(Surv(c(0, 1, 2, 0), c(3, 4, 5, 2), c(1, 0, 1, 1))),
    counting_type = jsurv(Surv(c(0, 1, 2, 0), c(3, 4, 5, 2), c(1, 0, 1, 1), type = "counting")),
    mstate = jsurv(Surv(c(1, 2, 3, 4), factor(c("a", "censor", "b", "a"),
                                             levels = c("censor", "a", "b")), type = "mstate")),
    mstate_factor = jsurv(Surv(c(1, 2, 3, 4), factor(c("a", "censor", "b", "a"),
                                                    levels = c("censor", "a", "b")))),
    mcounting = jsurv(Surv(c(0, 1, 2, 0), c(3, 4, 5, 2),
                           factor(c("a", "censor", "b", "a"), levels = c("censor", "a", "b")))),
    origin = jsurv(Surv(c(3, 4, 5), c(1, 0, 1), origin = 2)),
    is_na = jvec(is.na(Surv(c(1, NA, 3), c(1, 0, NA)))),
    format_right = I(format(Surv(c(1, 2, 3), c(1, 0, 1)))),
    format_counting = I(format(Surv(c(0, 1), c(3, 4), c(1, 0)))),
    format_interval2 = I(format(Surv(c(1, NA, 3), c(2, 3, NA), type = "interval2"))),
    format_mstate = I(format(Surv(c(1, 2), factor(c("a", "censor"), levels = c("censor", "a")))))
  )
  add_case("utilities", "surv_types", list(args = list()), out)
})

run_case("utilities", "statefig", function() {
  connect <- matrix(0, 3, 3, dimnames = list(c("A", "B", "C"), c("A", "B", "C")))
  connect[1, 2] <- connect[2, 3] <- connect[1, 3] <- 1
  pdf(NULL)
  s1 <- statefig(c(1, 2), connect)
  s2 <- statefig(matrix(c(1, 2), ncol = 1), connect)
  connect4 <- matrix(0, 4, 4, dimnames = list(c("A", "B", "C", "D"), c("A", "B", "C", "D")))
  connect4[1, 2] <- connect4[1, 3] <- connect4[2, 4] <- connect4[3, 4] <- 1
  s3 <- statefig(c(1, 2, 1), connect4)
  s4 <- statefig(matrix(c(1, 2, 1), ncol = 1), connect4)
  dev.off()
  add_case("utilities", "statefig",
           list(args = list(connect3 = jmat(connect), connect4 = jmat(connect4))),
           list(layout_1_2 = jmat(s1), layout_1_2_column = jmat(s2),
                layout_1_2_1 = jmat(s3), layout_1_2_1_column = jmat(s4)))
})

# ---------------------------------------------------------------------------
# km: aggregate.survfit (population averages of survfit(coxfit, newdata))
#
# The consumer receives the curves of the survfit object as R stores them
# (surv: time x data, pstate: time x data x state) rather than refitting the
# Cox model, so the case records that input matrix next to the aggregate.
# The file is km-aggregate_survfit.json.
# ---------------------------------------------------------------------------

aggregate_topic <- "km-aggregate_survfit"

# by (a vector, or a named list of vectors) and FUN as recorded in the case.
aggregate_case <- function(name, csurv, by = NULL, fun = NULL, note = NULL) {
  run_case(aggregate_topic, name, function() {
    agg <- if (is.null(fun)) aggregate(csurv, by = by) else aggregate(csurv, by = by, FUN = fun)
    args <- list()
    if (!is.null(csurv$surv)) args$surv <- jmat(csurv$surv)
    if (!is.null(csurv$pstate)) args$pstate <- jarray3(csurv$pstate)
    if (!is.null(by)) {
      args$by <- if (is.list(by)) lapply(by, jvec) else list(jvec(by))
      if (is.list(by) && !is.null(names(by))) args$by_names <- I(names(by))
    }
    args$fun <- if (is.null(fun)) "mean" else fun
    expected <- list()
    if (!is.null(agg$surv)) expected$surv <- if (is.matrix(agg$surv)) jmat(agg$surv) else jvec(agg$surv)
    if (!is.null(agg$pstate)) {
      expected$pstate <- if (length(dim(agg$pstate)) == 3) jarray3(agg$pstate) else jmat(agg$pstate)
    }
    # `$<- NULL` would drop the entry; keep it so the case records "no newdata"
    expected["newdata"] <- list(if (is.null(agg$newdata)) NULL else jframe(agg$newdata))
    add_case(aggregate_topic, name, list(args = args), expected, note)
  })
}

lung_cfit <- coxph(Surv(time, status) ~ age + sex, data = lung)
lung_population <- data.frame(age = c(50, 60, 70, 55, 65, 75), sex = c(1, 1, 2, 2, 1, 2))
lung_csurv <- survfit(lung_cfit, newdata = lung_population)
aggregate_case("lung_mean", lung_csurv)
aggregate_case("lung_constant_by", lung_csurv, by = rep(2, 6),
               note = "every column in one group: R drops back to the no-by case")
aggregate_case("lung_by_sex", lung_csurv, by = lung_population$sex)
aggregate_case("lung_by_sex_median", lung_csurv, by = lung_population$sex, fun = "median")
aggregate_case("lung_by_sex_grp_max", lung_csurv,
               by = list(sex = lung_population$sex, grp = c(1, 2, 3, 1, 2, 3)), fun = "max",
               note = "two grouping vectors: groups ordered with the first varying fastest")
lung_sfit <- coxph(Surv(time, status) ~ age + strata(sex), data = lung)
lung_scsurv <- survfit(lung_sfit, newdata = data.frame(age = c(45, 55, 65, 75)))
aggregate_case("lung_strata_by", lung_scsurv, by = c(1, 1, 2, 2),
               note = "stratified fit: the rows of both strata are stacked")
mgus2_mfit <- coxph(Surv(etime, event) ~ age + sex, data = mgus2_cr[mgus2_rows, ], id = id)
mgus2_msurv <- survfit(mgus2_mfit, newdata = data.frame(age = c(60, 70, 80, 65),
                                                        sex = c("F", "M", "F", "M")))
aggregate_case("mgus2_pstate_mean", mgus2_msurv, note = "multi-state fit on the first 400 rows of mgus2")
aggregate_case("mgus2_pstate_by", mgus2_msurv, by = c(1, 2, 1, 2),
               note = "multi-state fit on the first 400 rows of mgus2")

# ---------------------------------------------------------------------------
# survreg-extra (key: survreg): families, censoring types and options the
# survreg topic leaves light -- extreme/t/logistic with left censoring, the
# fixed-scale and strata paths with every residual and prediction type, an
# offset, and the iteration limits (iter.max = 0 returns the starting values).
# ---------------------------------------------------------------------------

survreg_extra_case <- function(name, dataset, formula, rows = NULL, args = list(),
                               data_ref = NULL, newdata = NULL, full = TRUE, note = NULL) {
  run_case("survreg-extra", name, function() {
    df <- case_frame(dataset, data_ref, rows)
    call_args <- c(list(formula = as.formula(formula), data = df), resolve_column_args(args, df))
    fit <- suppressWarnings(do.call(survreg, call_args))
    inputs <- case_inputs(dataset, data_ref, rows, formula, args, df)
    add_case("survreg-extra", name, inputs, survreg_expected(fit, newdata, full), note)
  })
}

tobin_newdata <- data.frame(age = c(40, 55), quant = c(220, 260))
survreg_extra_case("tobin_extreme_left", "tobin",
                   "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
                   args = list(dist = "extreme"), newdata = tobin_newdata)
survreg_extra_case("tobin_t_left", "tobin",
                   "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
                   args = list(dist = "t"), newdata = tobin_newdata)
survreg_extra_case("tobin_t_df6_left", "tobin",
                   "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
                   args = list(dist = "t", parms = 6), newdata = tobin_newdata)
survreg_extra_case("tobin_logistic_left_full", "tobin",
                   "Surv(durable, durable > 0, type = \"left\") ~ age + quant",
                   args = list(dist = "logistic"), newdata = tobin_newdata)
survreg_extra_case("ovarian_rayleigh_age", "ovarian", "Surv(futime, fustat) ~ age",
                   args = list(dist = "rayleigh"), newdata = data.frame(age = c(50, 65)))
survreg_extra_case("ovarian_exponential_ecog_rx_full", "ovarian",
                   "Surv(futime, fustat) ~ ecog.ps + rx", args = list(dist = "exponential"),
                   newdata = data.frame(ecog.ps = c(1, 2), rx = c(1, 2)))
survreg_extra_case("ovarian_loglogistic_age_full", "ovarian", "Surv(futime, fustat) ~ age",
                   args = list(dist = "loglogistic"), newdata = data.frame(age = c(50, 65)))
survreg_extra_case("lung_lognormal_strata_sex_full", "lung",
                   "Surv(time, status) ~ age + ph.ecog + strata(sex)", rows = lung_cc,
                   args = list(dist = "lognormal"),
                   newdata = data.frame(age = c(50, 70), ph.ecog = c(0, 2), sex = c(1, 2)))
survreg_extra_case("lung_weibull_offset_sex", "lung", "Surv(time, status) ~ age + offset(sex)",
                   args = list(dist = "weibull"), newdata = lung_newdata)
survreg_extra_case("lung_weibull_weighted_strata_sex", "lung",
                   "Surv(time, status) ~ age + strata(sex)",
                   args = list(dist = "weibull", weights = jvec(lung_weights)),
                   newdata = lung_newdata)
survreg_extra_case("lung_weibull_iter_max_2", "lung", "Surv(time, status) ~ age + sex",
                   args = list(dist = "weibull", control = list(iter.max = 2)), full = FALSE,
                   note = "stops before convergence (R warns)")
survreg_extra_case("lung_weibull_iter_max_0", "lung", "Surv(time, status) ~ age + sex",
                   args = list(dist = "weibull", control = list(iter.max = 0)), full = FALSE,
                   note = "returns the starting values of survreg.fit")
survreg_extra_case("lung_gaussian_iter_max_0", "lung", "Surv(time, status) ~ age + sex",
                   args = list(dist = "gaussian", control = list(iter.max = 0)), full = FALSE,
                   note = "returns the starting values of survreg.fit")
survreg_extra_case("lung_weibull_init_all", "lung", "Surv(time, status) ~ age + sex",
                   args = list(dist = "weibull", init = c(6.5, 0, -0.5, -0.2)), full = FALSE,
                   note = "starting values for the coefficients and log(scale)")
survreg_extra_case("interval2_synthetic_gaussian_g", NULL,
                   "Surv(left, right, type = \"interval2\") ~ g", data_ref = "synthetic_interval",
                   args = list(dist = "gaussian"), full = "light")

# ---------------------------------------------------------------------------
# dataprep: tmerge last-value-carried-forward (tmerge2's `k--`) -- a tdc
# applied to subjects that already have several intervals, with update
# times that miss some interval starts (na.rm dropping NA covariate rows,
# or an event split from an earlier argument).  Written to
# dataprep-tmerge.json.
# ---------------------------------------------------------------------------

register_data("tmerge_lvcf_base", data.frame(id = 1:3, futime = c(15, 12, 20),
                                             death = c(1, 0, 1)))
register_data("tmerge_lvcf_long", data.frame(
  id = c(1, 1, 1, 1, 2, 2, 3, 3, 3),
  time = c(-1, 5, 7, 10, 3, 6, 8, 12, 15),
  x = c(7, NA, 8, NA, 4, NA, 1, NA, 2),
  visit = c(0, 1, 0, 1, 1, 1, 1, 1, 0)
))

run_case("dataprep-tmerge", "lvcf_after_event_split", function() {
  base <- inline_data[["tmerge_lvcf_base"]]
  long <- inline_data[["tmerge_lvcf_long"]]
  d1 <- tmerge(base, base, id = id, death = event(futime, death))
  d2 <- tmerge(d1, long, id = id, visit = event(time, visit))
  d3 <- tmerge(d2, long, id = id, x = tdc(time, x))
  d4 <- tmerge(d3, long, id = id, nx = cumtdc(time, x))
  d5 <- tmerge(d4, long, id = id, seen = tdc(time))
  d6 <- tmerge(d2, long, id = id, x = tdc(time, x), options = list(na.rm = FALSE))
  d7 <- tmerge(d2, long, id = id, x = tdc(time, x, 0))
  add_case("dataprep-tmerge", "lvcf_after_event_split",
           list(data_ref = "tmerge_lvcf_base", data_ref2 = "tmerge_lvcf_long", args = list()),
           list(step2_visit_event = jframe(d2),
                step3_x_tdc = jframe(d3),
                step4_nx_cumtdc = jframe(d4),
                step5_seen_tdc = jframe(d5),
                x_tdc_na_kept = jframe(d6),
                x_tdc_init_0 = jframe(d7),
                tcount_step5 = jmat_named(attr(d5, "tcount"))),
           note = "x has NA values on rows that split the intervals, so the tdc must carry values across interval starts without an update")
})

run_case("dataprep-tmerge", "pbcseq_20_chol_na", function() {
  pbc1 <- pbc[pbc$id <= 20, c("id", "time", "status", "trt", "age", "sex")]
  seq2 <- pbcseq[pbcseq$id <= 20, c("id", "day", "bili", "chol", "ascites", "hepato")]
  register_data("pbc_20", pbc1)
  register_data("pbcseq_20_chol", seq2)
  pbc2 <- tmerge(pbc1, pbc1, id = id, death = event(time, status == 2))
  pbc3 <- tmerge(pbc2, seq2, id = id, bili = tdc(day, bili), chol = tdc(day, chol),
                 ascites = tdc(day, ascites), hepato = tdc(day, hepato))
  pbc4 <- tmerge(pbc2, seq2, id = id, chol = tdc(day, chol), options = list(na.rm = FALSE))
  add_case("dataprep-tmerge", "pbcseq_20_chol_na",
           list(data_ref = "pbc_20", data_ref2 = "pbcseq_20_chol", args = list()),
           list(frame = jframe(pbc3), tcount = jmat_named(attr(pbc3, "tcount")),
                chol_na_kept = jframe(pbc4)),
           note = "chol, ascites and hepato have NA rows in pbcseq; na.rm drops them so their values carry forward over the bili split")
})

# ---------------------------------------------------------------------------
# validation: extra cases for survobrien, yates, anova, survcheck and the
# survfit summaries (topic file validation-extra.json)
# ---------------------------------------------------------------------------

run_case("validation-extra", "survobrien_cgd_counting", function() {
  df <- cgd[cgd$id <= 40, ]
  ob <- survobrien(Surv(tstart, tstop, status) ~ age + height, data = df)
  cfit <- coxph(Surv(start, stop, status) ~ age + height + strata(.strata.), data = ob)
  add_case("validation-extra", "survobrien_cgd_counting",
           list(dataset = "cgd", rows = I(which(cgd$id <= 40)),
                formula = "Surv(tstart, tstop, status) ~ age + height", args = list()),
           list(frame = jframe(ob),
                cox_formula = "Surv(start, stop, status) ~ age + height + strata(.strata.)",
                coxph_coef = jnamed(cfit$coefficients)),
           note = "counting-process risk sets: start < t <= stop")
})

run_case("validation-extra", "yates_veteran_celltype_pairwise", function() {
  fit <- coxph(Surv(time, status) ~ celltype + karno, veteran)
  y <- yates(fit, "celltype", test = "pairwise")
  add_case("validation-extra", "yates_veteran_celltype_pairwise",
           list(dataset = "veteran", formula = "Surv(time, status) ~ celltype + karno",
                factors = factor_levels(veteran, "celltype"),
                args = list(term = "celltype", test = "pairwise"), fit = "coxph"),
           jyates(y))
})

run_case("validation-extra", "anova_lung_model_list", function() {
  df <- lung[lung_cc, ]
  fit1 <- coxph(Surv(time, status) ~ age, df)
  fit2 <- coxph(Surv(time, status) ~ age + sex, df)
  fit3 <- coxph(Surv(time, status) ~ age + sex + ph.ecog, df)
  a <- anova(fit1, fit2, fit3)
  add_case("validation-extra", "anova_lung_model_list",
           list(dataset = "lung", rows = I(lung_cc),
                formulas = I(c("Surv(time, status) ~ age", "Surv(time, status) ~ age + sex",
                               "Surv(time, status) ~ age + sex + ph.ecog")),
                args = list()),
           list(loglik = jvec(a$loglik), chisq = jvec(a$Chisq), df = jvec(a$Df),
                p = jvec(a[["Pr(>|Chi|)"]])),
           note = "anova.coxphlist: absolute differences between the fits")
})

register_data("synthetic_right_dupid", data.frame(
  id = c(1, 1, 2, 3, 3, 4),
  time = c(5, 8, 3, 2, 6, 4),
  status = c(0, 1, 1, 0, 1, 0)
))
run_case("validation-extra", "survcheck_right_censored_dupid", function() {
  df <- inline_data[["synthetic_right_dupid"]]
  sc <- survcheck(Surv(time, status) ~ 1, data = df, id = id)
  add_case("validation-extra", "survcheck_right_censored_dupid",
           inline_input("synthetic_right_dupid", "Surv(time, status) ~ 1", list(id = "id")),
           jsurvcheck(sc),
           note = "right-censored rows of one id all start at 0: R flags them as overlaps")
})

run_case("validation-extra", "survfit_lung_sex_rmean_individual", function() {
  fit <- survfit(Surv(time, status) ~ sex, lung)
  add_case("validation-extra", "survfit_lung_sex_rmean_individual",
           dataset_input("lung", NULL, "Surv(time, status) ~ sex", list()),
           list(summary_table_individual = jsurvfit_summary_table(fit, rmean = "individual"),
                summary_table_none = jsurvfit_summary_table(fit, rmean = "none"),
                summary_table_scale = jsurvfit_summary_table(fit, rmean = 365.25, scale = 365.25),
                quantile_scale = jquantile(fit),
                quantile_probs = local({
                  q <- quantile(fit, probs = c(0.1, 0.3, 0.9), conf.int = TRUE)
                  list(probs = jvec(c(0.1, 0.3, 0.9)), quantile = jmat(q$quantile),
                       lower = jmat(q$lower), upper = jmat(q$upper))
                })))
})

# Turnbull with a jump point that ends the EM with zero mass: survfitKM
# keeps the zero-weight pseudo-observation row (n.event = 0).
register_data("synthetic_interval_dead_jump", data.frame(
  time = c(7, 12, 12, 12, 18, 24, 30),
  time2 = c(NA, NA, NA, NA, NA, 27, 33),
  status = c(1, 0, 2, 2, 2, 3, 3)
))
run_case("validation-extra", "turnbull_dead_jump", function() {
  df <- inline_data[["synthetic_interval_dead_jump"]]
  fit <- survfit(Surv(time, time2, status, type = "interval") ~ 1, data = df)
  add_case("validation-extra", "turnbull_dead_jump",
           inline_input("synthetic_interval_dead_jump",
                        "Surv(time, time2, status, type = \"interval\") ~ 1"),
           turnbull_expected(fit),
           note = "the jump at 15 (between the right-censored 12 and the left-censored 18) ends with zero EM mass")
})

# Two larger curves (a fixed draw of exponential times with random status
# codes); curve "a" has a dead jump at 24.55 in the middle of the curve.
register_data("synthetic_interval_dead_jump_groups", data.frame(
  time = c(4, 13.2, 5.7, 0.8, 9.5, 29.3, 6.3, 8.2, 23.8, 14.3, 26.9, 48.2, 1.9, 1.1,
           25.1, 6.2, 9.5, 12.4, 24.9, 7.4, 97.3, 13.3, 99.9, 4.5, 24.2, 14.4, 26.2,
           5.6, 4.6, 25.7, 5.7, 30.2, 5, 3.5, 17.6, 7.4, 0.3, 3.5, 3.9, 19, 7.8, 6.6,
           3.9, 14.2, 15.2, 4.4, 41.7, 12, 0.3, 68.5, 1.6, 16.5, 11.3, 2.9, 5.5, 1.3,
           14.5, 9.1, 3.3, 84.6),
  time2 = c(NA, NA, 7, NA, NA, NA, 12.3, NA, NA, NA, NA, 53.3, NA, 3.1, NA, 13.8, NA,
            17.1, NA, NA, NA, NA, NA, NA, NA, NA, 35, 9.5, NA, 33.4, NA, NA, 13.1, NA,
            NA, NA, NA, NA, NA, NA, 9.3, 12.4, NA, NA, NA, NA, NA, NA, NA, NA, NA, 26.2,
            NA, NA, NA, NA, 21.3, NA, NA, NA),
  status = c(1, 0, 3, 1, 1, 1, 3, 0, 1, 0, 1, 3, 2, 3, 0, 3, 0, 3, 2, 1, 1, 2, 0, 0, 1,
             2, 3, 3, 1, 3, 0, 0, 3, 0, 0, 0, 1, 2, 0, 1, 3, 3, 0, 2, 1, 0, 0, 1, 0, 0,
             2, 3, 1, 2, 1, 1, 3, 0, 2, 0),
  g = rep(c("a", "b"), each = 30)
))
run_case("validation-extra", "turnbull_dead_jump_groups", function() {
  df <- inline_data[["synthetic_interval_dead_jump_groups"]]
  fit <- survfit(Surv(time, time2, status, type = "interval") ~ g, data = df)
  add_case("validation-extra", "turnbull_dead_jump_groups",
           inline_input("synthetic_interval_dead_jump_groups",
                        "Surv(time, time2, status, type = \"interval\") ~ g"),
           list(fit = jsurvfit(fit)),
           note = "curve a has a dead jump at 24.55 (n.event = 0) before later events")
})

# ---------------------------------------------------------------------------
# Write everything
# ---------------------------------------------------------------------------

for (topic in names(fixtures)) write_topic(topic)

if (length(failures) > 0) {
  cat("\n", length(failures), "case(s) failed:\n")
  cat(paste0("  ", failures), sep = "\n")
  quit(status = 1)
}

# --check: run the generator a second time into a scratch directory and
# require byte-identical output, which catches uninitialised memory leaking
# into a fixture (the values change between runs) and unseeded randomness.
if (self_check) {
  check_dir <- tempfile("fixtures-check-")
  status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(script_path()),
                    env = paste0("R_FIXTURES_DIR=", shQuote(check_dir)),
                    stdout = FALSE, stderr = FALSE)
  if (status != 0) stop("the --check regeneration failed")
  files <- sort(list.files(fixture_dir, pattern = "\\.json$"))
  check_files <- sort(list.files(check_dir, pattern = "\\.json$"))
  if (!identical(files, check_files)) stop("the --check regeneration wrote different files")
  differing <- files[unname(tools::md5sum(file.path(fixture_dir, files))) !=
                       unname(tools::md5sum(file.path(check_dir, files)))]
  unlink(check_dir, recursive = TRUE)
  if (length(differing) > 0) {
    stop("regeneration is not reproducible: ", paste(differing, collapse = ", "))
  }
  cat("check: two runs are byte-identical\n")
}
cat("done\n")
