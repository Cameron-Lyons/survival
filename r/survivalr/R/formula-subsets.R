# R evaluates a subset against the data and formula environment, then applies
# ordinary R row indexing before the shared missing-data handling.
.eval_formula_subset <- function(expr, missing_arg, formula, data, env) {
  if (isTRUE(missing_arg)) return(NULL)
  if (is.list(formula) && !inherits(formula, "formula")) formula <- formula[[1L]]
  if (inherits(formula, "formula") && !is.null(environment(formula))) env <- environment(formula)
  value <- if (is.null(data)) eval(expr, env) else eval(expr, data, env)
  n <- if (inherits(formula, "survival_py_surv")) {
    reticulate::py_len(formula)
  } else if (inherits(formula, "Surv")) {
    NROW(formula)
  } else NULL
  .as_python_formula_subset(value, data, n)
}

.as_python_formula_subset <- function(value, data = NULL, n = NULL) {
  if (is.null(value)) return(NULL)
  if (is.null(n)) {
    n <- if (is.data.frame(data) || is.matrix(data)) {
      nrow(data)
    } else if (is.list(data) && length(data)) {
      NROW(data[[1L]])
    } else {
      NROW(data)
    }
  }
  labels <- if (is.data.frame(data) || is.matrix(data)) {
    rownames(data)
  } else if (is.list(data) && length(data)) {
    names(data[[1L]])
  } else {
    NULL
  }
  if (is.null(labels)) labels <- as.character(seq_len(n))
  # data.frame resolves character selectors before subsetting matrix columns.
  if (is.character(value)) value <- pmatch(value, labels, duplicates.ok = TRUE)
  selected <- matrix(seq_len(n), ncol = 1L)[value, 1L]
  if (anyNA(selected)) {
    selected[is.na(selected)] <- 0L
    return(.pybridge_attr("_r_subset")(as.list(as.integer(selected) - 1L)))
  }
  as.list(as.integer(selected) - 1L)
}
