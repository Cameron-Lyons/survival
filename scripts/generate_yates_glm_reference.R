#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/yates_glm_reference.json"
d <- expand.grid(a = factor(c("a", "b", "c")), b = factor(c("x", "y")), replicate = 1:12)
i <- seq_len(nrow(d))
d$z <- sin(i) / 3 + d$replicate / 12
d$alias <- 2 * d$z
d$y <- 3 + as.numeric(d$a) / 3 + as.numeric(d$b) / 4 + d$z / 5 + cos(i) / 3
cases <- list()
add <- function(name, family = binomial(), formula = "y ~ a * b + z", term = "a",
                population = "data", predict = "response", test = "global", levels = NULL,
                weighted = FALSE, nsim = 73L, seed = 123L, method = "direct") {
  data <- d
  if (family$family %in% c("binomial", "quasibinomial")) {
    data$y <- as.integer(((i * 11) %% 17) < (4 + 2 * as.numeric(d$a) + as.numeric(d$b)))
  } else if (family$family %in% c("poisson", "quasipoisson")) {
    data$y <- 2 + (i * 7) %% 9 + as.numeric(d$a)
  }
  weights <- if (weighted) rep(c(1, 3, 2, 1), length.out = nrow(data)) else NULL
  design <- model.matrix(as.formula(formula), data)
  start <- if ("(Intercept)" %in% colnames(design)) {
    c(family$linkfun(mean(data$y)), rep(0, ncol(design) - 1L))
  } else NULL
  fit <- glm(as.formula(formula), family = family, data = data, weights = weights,
             start = start, control = glm.control(maxit = 100))
  arguments <- list(fit = fit, term = term, population = population, predict = predict,
                    test = test, nsim = nsim, method = method)
  if (!is.null(levels)) arguments$levels <- levels
  set.seed(seed)
  result <- do.call(yates, arguments)
  cases[[length(cases) + 1L]] <<- list(
    name = name, family = family$family, link = family$link, formula = formula, data = data,
    beta = I(unname(coef(fit))), variance = unname(vcov(fit, complete = FALSE)),
    weights = if (is.null(weights)) NULL else I(weights),
    term = term, population = population, predict = predict, test = test, levels = levels,
    nsim = nsim, seed = seed, method = method,
    estimate = result$estimate, mvar = unname(result$mvar), cmat = unname(result$cmat),
    tests = unname(result$test), test_names = I(rownames(result$test)), sas = unname(result$SAS))
}
for (link in c("logit", "probit", "cloglog", "cauchit")) add(paste0("binomial_", link), binomial(link))
for (link in c("log", "identity", "sqrt")) add(paste0("poisson_", link), poisson(link))
for (link in c("identity", "log", "inverse")) add(paste0("gaussian_", link), gaussian(link))
for (link in c("inverse", "log", "identity")) add(paste0("gamma_", link), Gamma(link))
add("quasibinomial", quasibinomial())
add("quasipoisson", quasipoisson())
add("weighted_response", weighted = TRUE)
add("weighted_link", weighted = TRUE, predict = "link")
add("weighted_linear", weighted = TRUE, predict = "linear")
add("sas_population", population = "sas")
add("factorial_population", population = "factorial", formula = "y ~ a * b")
add("explicit_population", population = data.frame(b = c("x", "y", "x"), z = c(.2, .5, .8)))
add("pairwise", test = "pairwise", nsim = 37L, seed = 42L)
add("joint_pairwise", term = "a + b", test = "pairwise")
add("continuous_levels", term = "z", levels = c(.1, .9), predict = "resp")
add("reordered_levels", levels = c("c", "a"))
add("aliased_coefficient", formula = "y ~ a * b + z + alias")
add("no_intercept", formula = "y ~ 0 + a + b + z")
add("offset_response", family = poisson(), formula = "y ~ a + b + offset(z)")
add("offset_linear", family = poisson(), formula = "y ~ a + b + offset(z)", predict = "linear")
add("sgtt_link", population = "sas", predict = "link", method = "sgtt")
reference <- list(metadata = list(generator = "scripts/generate_yates_glm_reference.R",
  r_version = as.character(getRversion()), survival_version = as.character(packageVersion("survival"))),
  cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE,
           na = "null", null = "null", dataframe = "columns")
cat(length(cases), "GLM Yates cases written to", output, "\n")
