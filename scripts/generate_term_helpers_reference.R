#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/term_helpers_reference.json"
specials <- c("strata", "cluster", "offset", "foo", "survival::strata")
metadata <- function(tt) {
  factors <- attr(tt, "factors")
  variables <- vapply(as.list(attr(tt, "variables"))[-1L],
                     function(x) paste(deparse(x), collapse = ""), "")
  list(term_labels = I(attr(tt, "term.labels")), variables = I(variables),
       factors = if (is.matrix(factors)) lapply(seq_along(variables), function(i) {
         I(as.integer(factors[i, ]))
       }) else NULL, order = I(attr(tt, "order")), response = attr(tt, "response"),
       specials = lapply(attr(tt, "specials"), function(x) if (is.null(x)) NULL else I(x)))
}
formulas <- c(
  "~ 1", "y ~ 1", "~ 0", "y ~ x + strata(g)", "~ strata(g) + x",
  "y ~ strata(g) + strata(h) + x:strata(g)",
  "y ~ (x + strata(g) + cluster(id))^2", "y ~ strata(g):cluster(id)",
  "y ~ x + strata(g) - strata(g)", "y ~ x - strata(g)",
  "y ~ x:strata(g) - x:strata(g)", "y ~ x + offset(z)",
  "y ~ x * offset(z)", "y ~ x:offset(z)", "y ~ strata(g, h) + x:strata(g, h)",
  "y ~ x + foo(g)", "y ~ x:foo(g) + foo(h)", "y ~ foo(strata(g)) + x",
  "y ~ survival::strata(g) * x", "y ~ x + log(z)",
  "Surv(time, status) ~ strata(g) * x + cluster(id)", "strata(g) ~ x",
  "y ~ strata(g):strata(h):x", "y ~ strata(g) / (x + z)"
)
special_cases <- lapply(formulas, function(formula) {
  tt <- terms(as.formula(formula), specials = specials)
  queries <- list()
  for (special in c(specials, "absent")) for (order in list(1L, 2L, c(1L, 2L, 3L), integer())) {
    found <- tryCatch(untangle.specials(tt, special, order), error = identity)
    queries[[length(queries) + 1L]] <- list(special = special, order = I(order),
                                          expected = if (inherits(found, "error")) NULL else lapply(found, I),
                                          error = if (inherits(found, "error")) conditionMessage(found) else NULL)
  }
  list(formula = formula, metadata = metadata(tt), queries = queries)
})
d <- data.frame(y = 1:12, x = c(1, 4, 2, 8, 3, 7, 5, 9, 6, 11, 10, 12),
                z = seq(.1, 1.2, by = .1), g = factor(rep(c("b", "a", "c"), 4)),
                h = factor(rep(c("yes", "no"), each = 6)))
matrix_formulas <- c("~1", "~0", "~x + g", "~0 + g", "~x*g + z", "~g:h", "~0 + g:h",
                     "~g/h", "~strata(g) + x:strata(g)", "~x + offset(z)",
                     "~splines::ns(x, 3)", "~(x + z + g)^2")
assign_cases <- list()
for (formula in matrix_formulas) {
  tt <- terms(as.formula(formula), specials = specials, data = d)
  x <- model.matrix(tt, d)
  original <- attr(x, "assign")
  positions <- seq_along(original)
  for (name in c("original", "reverse", "interleaved")) {
    selected <- switch(name, original = positions, reverse = rev(positions),
                       interleaved = c(positions[positions %% 2L == 1L], positions[positions %% 2L == 0L]))
    value <- x[, selected, drop = FALSE]
    attr(value, "assign") <- original[selected]
    assign_cases[[length(assign_cases) + 1L]] <- list(
      name = paste(formula, name), term_labels = I(attr(tt, "term.labels")),
      assign = I(attr(value, "assign")), expected = lapply(attrassign(value, tt), I)
    )
  }
}
fit_cases <- list()
for (kind in c("coxph", "survreg")) for (rhs in c(
    "age + strata(sex) + ph.ecog", "strata(sex) + age", "age + strata(sex)",
    "factor(ph.ecog) + strata(sex)", "age + cluster(inst) + strata(sex)",
    "age + strata(sex) + ph.ecog:age")) {
  formula <- paste("Surv(time, status) ~", rhs)
  fit <- do.call(kind, list(formula = as.formula(formula), data = lung))
  matrix <- model.matrix(fit)
  fit_cases[[length(fit_cases) + 1L]] <- list(
    kind = kind, formula = formula, labels = I(labels(fit)),
    assign = I(attr(matrix, "assign")), expected = lapply(attrassign(matrix, terms(fit)), I))
}
write_json(list(r_version = R.version.string,
                survival_version = as.character(packageVersion("survival")),
                special_cases = special_cases, assign_cases = assign_cases, fit_cases = fit_cases),
           output, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null")
