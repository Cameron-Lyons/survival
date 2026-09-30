#!/usr/bin/env Rscript
# Stock R numerical oracle; correct only the documented delayed-entry row index.
suppressPackageStartupMessages(library(survival))
reference <- survival::finegray
code <- paste(deparse(body(reference), width.cutoff = 500L), collapse = "\n")
stopifnot(grepl("Y[first, 1]", code, fixed = TRUE))
body(reference) <- str2lang(sub("Y[first, 1]", "Y[index[first], 1]", code, fixed = TRUE))
cases <- list()
for (seed in seq_len(60L)) {
  set.seed(seed + 2700L)
  history <- seed %% 3L == 0L
  counting <- seed %% 3L != 1L
  n <- 24L
  subjects <- if (history) n / 2L else n
  ids <- rep(seq_len(subjects), each = if (history) 2L else 1L)
  groups <- rep(sample(c(-17L, 80L, 402L), subjects, replace = TRUE), each = if (history) 2L else 1L)
  entry <- sample(0:4, subjects, replace = TRUE)
  end <- entry + sample(3:16, subjects, replace = TRUE)
  status <- sample(0:2, subjects, replace = TRUE)
  status[2:3] <- c(1L, 2L)
  # Exercise entry after the earliest censoring.
  entry[1L] <- 0
  end[1L] <- 1
  status[1L] <- 0L
  if (history) {
    middle <- (entry + end)/2
    start <- as.vector(rbind(entry, middle))
    time <- as.vector(rbind(middle, end))
    status <- as.vector(rbind(0L, status))
  } else { start <- entry; time <- end }
  d <- data.frame(start = start, time = time,
    status = factor(status, levels = 0:2, labels = c("censor", "a", "b")),
    group = factor(groups), id = ids, weight = runif(n, 0.5, 3))
  if (seed %% 7L == 0L) {
    d$start <- d$start - 7
    d$time <- d$time - 7
  }
  d <- d[sample(seq_len(n)), ]
  d$rowid <- seq_len(n)
  stratified <- seed %% 2L == 0L
  formula <- if (counting) Surv(start, time, status) ~ rowid else Surv(time, status) ~ rowid
  if (stratified) formula <- update(formula, . ~ . + strata(group))
  args <- list(formula = formula, data = d, count = "added", etype = if (seed %% 4L == 0L) "b" else "a")
  if (counting) args$id <- d$id
  if (seed %% 5L != 0L) args$weights <- d$weight
  result <- do.call(reference, args)
  expected <- if (any(!is.finite(result$fgwt))) list(error = "censoring probability is zero") else {
    list(row = I(result$rowid), start = I(result$fgstart), end = I(result$fgstop), wt = I(result$fgwt), add = I(result$added))
  }
  cases[[seed]] <- list(name = paste0("case_", seed),
    time = I(d$time), status = I(as.integer(d$status) - 1L), event_type = if (args$etype == "b") 2L else 1L,
    start = if (counting) I(d$start) else NULL, id = if (counting) I(d$id) else NULL,
    strata = if (stratified) I(as.integer(as.character(d$group))) else NULL,
    weights = if (is.null(args$weights)) NULL else I(args$weights), expected = expected)
}
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args)) args[[1L]] else "python/tests/fixtures/finegray_prepared.json"
writeLines(jsonlite::toJSON(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
  reference_correction = "Y[first, 1] -> Y[index[first], 1]", cases = cases),
  auto_unbox = TRUE, null = "null", pretty = TRUE, digits = NA), path)
