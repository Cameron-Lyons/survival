.evaluated_matrix_setup <- function() {
  setup <- .matrix_na_setup()
  setup$formulas <- c("1", "age", "age - 1", "age + flag", "age + I(flag)",
    "age + log(z)", "age + I(log(z) + sqrt(w))", "age + factor(g)", "age * g",
    "g:flag", "g + age:flag", "age + strata(g)", "strata(g) + age + flag",
    "strata(g) + flag - 1", "age + strata(g, h)", "age + strata(g) + strata(h)",
    "age + strata(g) + g", "age + strata(g):flag", "age + strata(g) + strata(g):flag",
    "age + offset(log(off))", "age + log(z) + offset(off)", "age + sqrt(z) - sqrt(z)",
    "ridge(age, theta = 2)", "ridge(age, rx, theta = 2)", "pspline(age, df = 3)",
    "age + frailty(id, sparse = TRUE, theta = 0.4)",
    "age + frailty(id, sparse = FALSE, theta = 0.4)",
    "age + frailty.gaussian(id, sparse = FALSE, theta = 0.4)")
  setup$inputs <- c("complete", "na_numeric", "nan_numeric", "na_logical", "na_factor",
    "na_strata", "na_offset", "na_response", "drop_response", "drop_ignored", "drop_variable",
    "altered_values", "altered_levels", "dropped_levels", "custom_contrasts", "numeric_factor",
    "numeric_logical", "character_factor", "matrix_na", "matrix_nan", "narrow_matrix",
    "wide_matrix", "rename_matrix", "all_missing", "empty", "single_na", "marker", "untagged")
  setup
}

.evaluated_matrix_newdata <- function(model_frame, input) {
  nd <- model_frame[c(7, 3, 7, 1), , drop = FALSE]
  term_marker <- attr(model_frame, "terms")
  response <- names(nd)[1L]
  variables <- setdiff(names(nd), response)
  numeric <- variables[vapply(nd[variables], function(x) is.numeric(x) && is.null(dim(x)), TRUE)]
  logical <- variables[vapply(nd[variables], is.logical, TRUE)]
  factors <- variables[vapply(nd[variables], is.factor, TRUE)]
  ordinary_factors <- factors[!startsWith(factors, "strata(")]
  stratum <- factors[startsWith(factors, "strata(")]
  offsets <- variables[startsWith(variables, "offset(")]
  matrices <- variables[vapply(nd[variables], is.matrix, TRUE)]
  target <- switch(input, na_numeric = numeric, nan_numeric = numeric, na_logical = logical,
    na_factor = ordinary_factors, na_strata = stratum, na_offset = offsets)
  if (length(target)) nd[[target[[1L]]]][2L] <- if (input == "nan_numeric") NaN else NA
  if (input == "na_response") nd[[response]][2L, ] <- NA_real_
  if (input %in% c("drop_response", "drop_ignored")) nd[[response]] <- NULL
  if (input == "drop_ignored") {
    for (name in c(offsets, intersect(variables, "sqrt(z)"))) nd[[name]] <- NULL
  }
  if (input == "drop_variable" && length(variables)) nd[[variables[[1L]]]] <- NULL
  if (input == "altered_values") {
    for (name in numeric) nd[[name]] <- -(seq_len(nrow(nd)) + 10)
    nd$z <- rep(-1, nrow(nd))
  }
  if (input == "altered_levels") {
    for (name in factors) nd[[name]] <- factor(rep("unseen", nrow(nd)), levels = c("unused", "unseen"))
  }
  if (input == "dropped_levels") for (name in factors) nd[[name]] <- droplevels(nd[[name]])
  if (input == "custom_contrasts") {
    for (name in factors) if (nlevels(nd[[name]]) > 1L) contrasts(nd[[name]]) <- "contr.sum"
  }
  if (input == "numeric_factor") for (name in ordinary_factors) nd[[name]] <- as.numeric(nd[[name]])
  if (input == "numeric_logical") for (name in logical) nd[[name]] <- as.numeric(nd[[name]])
  if (input == "character_factor") for (name in ordinary_factors) nd[[name]] <- as.character(nd[[name]])
  if (input %in% c("matrix_na", "matrix_nan")) {
    for (name in matrices) nd[[name]][2L, 1L] <- if (input == "matrix_nan") NaN else NA_real_
  }
  if (input == "narrow_matrix") for (name in matrices) nd[[name]] <- nd[[name]][, 1L, drop = FALSE]
  if (input == "wide_matrix") for (name in matrices) nd[[name]] <- cbind(unclass(nd[[name]]), extra = 0.5)
  if (input == "rename_matrix") {
    for (name in matrices) colnames(nd[[name]]) <- paste0("基 / ", seq_len(ncol(nd[[name]])))
  }
  if (input == "all_missing") for (name in variables) nd[[name]][] <- NA
  if (input == "empty") nd <- nd[FALSE, , drop = FALSE]
  if (input == "single_na") {
    nd <- nd[2L, , drop = FALSE]
    for (name in intersect(variables, names(nd))) nd[[name]][] <- NA
  }
  attr(nd, "terms") <- if (input == "untagged") NULL else if (input == "marker") "present" else term_marker
  nd
}

.evaluated_matrix_encode_numeric <- function(value) {
  result <- .matrix_na_encode(value)
  result$na <- I(which(is.na(value) & !is.nan(value)))
  result$nan <- I(which(is.nan(value)))
  result$positive_infinity <- I(which(value == Inf))
  result$negative_infinity <- I(which(value == -Inf))
  result
}

.evaluated_matrix_encode_column <- function(column) {
  kind <- if (is.matrix(column)) "matrix" else if (is.factor(column)) "factor" else typeof(column)
  value <- if (kind %in% c("factor", "character")) I(as.character(column)) else
    if (kind == "logical") I(column) else .evaluated_matrix_encode_numeric(unclass(column))
  result <- list(kind = kind, value = value)
  if (kind == "factor") result$levels <- I(levels(column))
  if (kind %in% c("logical", "factor")) {
    factor <- if (kind == "logical") factor(column, levels = c(FALSE, TRUE)) else column
    if (nlevels(factor) > 1L) {
      contrast <- contrasts(factor)
      label <- attr(factor, "contrasts")
      if (is.null(label)) label <- getOption("contrasts")[[if (is.ordered(factor)) 2L else 1L]]
      result$contrast <- list(value = .matrix_na_encode(contrast),
        label = if (is.character(label)) label else NULL)
    }
  }
  result
}

.evaluated_matrix_encode_frame <- function(data) {
  list(columns = lapply(data, .evaluated_matrix_encode_column), nrow = nrow(data),
    rows = if (.row_names_info(data, 1L) < 0L) NULL else I(row.names(data)),
    tagged = !is.null(attr(data, "terms")))
}
