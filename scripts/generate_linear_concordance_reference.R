#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/linear_concordance_reference.json"
i <- seq_len(30)
d <- data.frame(x = ((i * 7) %% 31) / 10 - 1, z = sin(i / 2),
                a = factor(rep(c("a", "b", "c"), 10)), o = cos(i) / 2)
d$alias <- 2 * d$x
d$y <- 4 + .8 * d$x + .5 * d$z + as.numeric(d$a) / 3 + cos(i * 1.3)
w <- rep(c(1, 2, .5, 3, 1), 6)
clean <- function(result) {
  fields <- c("concordance", "count", "n", "var", "cvar", "dfbeta", "influence", "ranks")
  value <- setNames(lapply(fields, function(name) result[[name]]), fields)
  for (name in setdiff(names(value), "ranks")) {
    item <- value[[name]]
    value[name] <- list(if (length(dim(item)) == 3) {
      lapply(seq_len(dim(item)[3]), function(j) unname(item[, , j]))
    } else if (is.null(item)) NULL else if (is.null(dim(item))) I(unname(item)) else unname(item))
  }
  if (!is.null(value$ranks$fit)) {
    value$ranks$fit <- paste0("fit", match(value$ranks$fit, unique(value$ranks$fit)))
  }
  value
}
cases <- list()
add <- function(name, formulas = "y ~ x + z", weighted = FALSE, zero = FALSE,
                new = FALSE, family = NULL, influence = 3L, clustered = FALSE,
                bounds = FALSE, timefix = TRUE) {
  data <- d
  if (!is.null(family)) {
    data$y <- if (family$family == "binomial") as.numeric((i * 11) %% 17 < 8) else (i * 13) %% 7
  }
  weights <- if (weighted) w else NULL
  if (zero) weights[seq(1, nrow(data), by = 5)] <- 0
  fits <- lapply(formulas, function(formula) {
    if (is.null(family)) lm(as.formula(formula), data, weights = weights) else
      glm(as.formula(formula), data = data, weights = weights, family = family)
  })
  newdata <- if (new) data[c(29, 4, 17, 3, 26, 8, 13, 22, 2, 7, 11, 19), ] else NULL
  options <- list(influence = influence, ranks = TRUE, timefix = timefix)
  if (new) options$newdata <- newdata
  cluster <- if (clustered) rep(c("b", "a", "c"), length.out = if (new) nrow(newdata) else nrow(data)) else NULL
  if (!is.null(cluster)) options$cluster <- cluster
  if (bounds) { options$ymin <- 3; options$ymax <- 5 }
  result <- do.call(concordance, c(fits, options))
  raw <- clean(result)
  # R applies case weights twice to the joint sandwich. Use the already
  # weighted, optionally clustered single-model influence vectors exactly once.
  if (length(fits) > 1 && weighted && !new) {
    columns <- lapply(fits, function(fit) {
      args <- options; args$influence <- 1L
      do.call(concordance, c(list(fit), args))$dfbeta
    })
    result$var <- crossprod(do.call(cbind, columns))
  }
  rebuilt <- lapply(fits, function(fit) {
    beta <- coef(fit); beta[is.na(beta)] <- 0
    value <- drop(model.matrix(fit) %*% beta)
    offset <- model.offset(model.frame(fit))
    if (!is.null(offset)) value <- value + offset
    fit$linear.predictors <- value
    fit
  })
  reconstructed <- do.call(concordance, c(rebuilt, options))
  if (length(fits) > 1 && weighted && !new) {
    columns <- lapply(rebuilt, function(fit) {
      args <- options; args$influence <- 1L
      do.call(concordance, c(list(fit), args))$dfbeta
    })
    reconstructed$var <- crossprod(do.call(cbind, columns))
  }
  cases[[length(cases) + 1L]] <<- list(
    name = name, data = data, newdata = newdata,
    models = lapply(seq_along(fits), function(j) list(
      formula = formulas[j], coefficients = I(unname(coef(fits[[j]]))),
      weights = if (is.null(weights)) NULL else I(weights),
      linear_predictors = I(unname(if (is.null(fits[[j]]$linear.predictors)) fits[[j]]$fitted.values else fits[[j]]$linear.predictors)),
      glm = !is.null(family))),
    options = list(influence = influence, ranks = TRUE, timefix = timefix,
                   cluster = if (is.null(cluster)) NULL else I(cluster),
                   ymin = if (bounds) 3 else NULL, ymax = if (bounds) 5 else NULL),
    raw = raw, expected = clean(result), reconstructed = clean(reconstructed))
}
for (new in c(FALSE, TRUE)) {
  suffix <- if (new) "_newdata" else "_training"
  add(paste0("ordinary", suffix), new = new)
  add(paste0("factor_interaction", suffix), "y ~ a * x + z", new = new)
  add(paste0("no_intercept", suffix), "y ~ 0 + a + x", new = new)
  add(paste0("offset", suffix), "y ~ x + offset(o)", new = new)
  add(paste0("transformed", suffix), "log(y) ~ x + I(z * z)", new = new)
  add(paste0("intercept_only", suffix), "y ~ 1", new = new)
  add(paste0("weights", suffix), weighted = TRUE, new = new)
  add(paste0("zero_weights", suffix), weighted = TRUE, zero = TRUE, new = new)
  add(paste0("cluster", suffix), weighted = TRUE, clustered = TRUE, new = new)
  add(paste0("poisson", suffix), "y ~ x + z + offset(o)", family = poisson(), new = new)
  add(paste0("binomial", suffix), "y ~ a + x", family = binomial("probit"), new = new)
  for (influence in 0:3) add(paste0("joint", influence, suffix), c("y ~ x", "y ~ x + z"),
                           new = new, influence = influence)
  add(paste0("joint_weights", suffix), c("y ~ x", "y ~ x + z"),
      weighted = TRUE, new = new, influence = 1L)
}
add("bounds", new = TRUE, bounds = TRUE)
add("no_timefix", new = TRUE, timefix = FALSE)
add("aliased", "y ~ x + z + alias")
reference <- list(metadata = list(generator = "scripts/generate_linear_concordance_reference.R",
  r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival")),
  correction = "Joint weighted variance is crossprod of the single-fit dfbeta; raw R results are retained."),
  cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null", dataframe = "columns")
cat(length(cases), "linear-model concordance cases written to", output, "\n")
