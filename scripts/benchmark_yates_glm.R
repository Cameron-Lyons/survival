#!/usr/bin/env Rscript
# Complete marginal-mean calls on an already fitted external GLM and population.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(pkgload::load_all("r/survivalr", quiet = TRUE))
args <- commandArgs(trailingOnly = TRUE)
n <- if (length(args)) as.integer(args[[1L]]) else 5000L
nsim <- if (length(args) > 1L) as.integer(args[[2L]]) else 200L
repeats <- if (length(args) > 2L) as.integer(args[[3L]]) else 7L
stopifnot(n > 0L, nsim >= 2L, repeats > 0L)
seed <- 123L
set.seed(seed)
d <- data.frame(y = rbinom(300, 1, .5), a = factor(rep(c("a", "b", "c"), 100)),
                 b = factor(rep(c("x", "y"), 150)), z = rnorm(300))
fit <- glm(y ~ a * b + z, family = binomial(), data = d)
population <- d[rep(seq_len(nrow(d)), length.out = n), c("b", "z")]
as_columns <- function(frame) lapply(frame, function(column) {
  as.list(if (is.factor(column)) as.character(column) else column)
})
python <- reticulate::py_run_string(
  "def inverse(eta):\n    import numpy as np\n    return 1 / (1 + np.exp(-eta))", local = TRUE,
  convert = FALSE)
adapter <- survivalr:::.python_attr("YatesModel")(
  "y ~ a * b + z", as_columns(d), as.list(unname(coef(fit))), unname(vcov(fit)),
  family = list(linkinv = python$inverse))
population_py <- reticulate::r_to_py(as_columns(population), convert = FALSE)
calls <- list(
  R_survival = function() {
    set.seed(seed)
    survival::yates(fit, "a", population = population, predict = "response", nsim = nsim)
  },
  Python_Rust = function() survivalr:::.python_attr("yates")(
    adapter, "a", population = population_py, predict = "response", nsim = nsim,
    options = list(seed = seed)))
reference <- calls$R_survival()
actual <- calls$Python_Rust()
estimate <- survivalr:::.result_field(actual, "estimate")
for (field in c("pmm", "std")) {
  observed <- survivalr:::.as_numeric_vector(survivalr:::.result_field(estimate, field))
  stopifnot(isTRUE(all.equal(observed, reference$estimate[[field]], tolerance = 3e-8)))
}
variance <- survivalr:::.as_numeric_matrix(survivalr:::.result_field(actual, "mvar"))
stopifnot(isTRUE(all.equal(variance, unname(reference$mvar), tolerance = 3e-8)))
contrast <- survivalr:::.result_field(actual, "test")[[1L]]
chisq <- survivalr:::.result_field(contrast, "chisq")
df <- survivalr:::.result_field(contrast, "df")
stopifnot(isTRUE(all.equal(as.numeric(chisq), unname(reference$test[1, "chisq"]), tolerance = 3e-8)),
          as.numeric(df) == reference$test[1, "df"])
results <- list()
for (name in names(calls)) {
  elapsed <- numeric(repeats)
  for (i in seq_len(repeats)) {
    gc(FALSE)
    started <- proc.time()[["elapsed"]]
    value <- calls[[name]]()
    elapsed[i] <- 1000 * (proc.time()[["elapsed"]] - started)
    rm(value)
  }
  results[[name]] <- list(median_ms = median(elapsed), samples_ms = I(elapsed))
}
cat(jsonlite::toJSON(list(r_version = R.version.string,
                         survival_version = as.character(packageVersion("survival")),
                         population_rows = n, levels = 3L, coefficients = length(coef(fit)),
                         nsim = nsim, repeats = repeats, results = results),
                    auto_unbox = TRUE, pretty = TRUE, digits = NA), "\n")
