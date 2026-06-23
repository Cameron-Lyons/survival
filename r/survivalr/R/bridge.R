.survival_python_module <- local({
  module <- NULL

  function() {
    if (is.null(module)) {
      module <<- reticulate::import("survival.r_api", convert = TRUE)
    }
    module
  }
})

.python_attr <- function(name) {
  reticulate::py_get_attr(.survival_python_module(), name)
}

.survival_data_prep_module <- local({
  module <- NULL

  function() {
    if (is.null(module)) {
      module <<- reticulate::import("survival.data_prep", convert = TRUE)
    }
    module
  }
})

.data_prep_attr <- function(name) {
  reticulate::py_get_attr(.survival_data_prep_module(), name)
}

.compact_null <- function(values) {
  values[!vapply(values, is.null, logical(1))]
}

.as_formula_string <- function(value) {
  if (inherits(value, "formula")) {
    paste(deparse(value, width.cutoff = 500L), collapse = " ")
  } else if (is.character(value)) {
    paste(value, collapse = " ")
  } else {
    value
  }
}

.as_python_vector <- function(value) {
  if (is.factor(value)) {
    value <- as.character(value)
  } else if (inherits(value, "Date")) {
    value <- as.numeric(value)
  } else {
    value <- as.vector(value)
  }
  if (is.atomic(value) && anyNA(value)) {
    return(lapply(value, function(item) {
      if (is.na(item)) {
        reticulate::py_none()
      } else {
        item
      }
    }))
  }
  value
}

.as_python_optional_vector <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  if (is.matrix(value) || is.data.frame(value)) {
    return(value)
  }
  .as_python_vector(value)
}

.as_python_data <- function(data) {
  if (is.null(data)) {
    return(NULL)
  }
  if (is.data.frame(data)) {
    return(lapply(data, .as_python_vector))
  }
  data
}

.as_numeric_vector <- function(value) {
  as.numeric(unlist(value, recursive = TRUE, use.names = FALSE))
}

.as_numeric_matrix <- function(value) {
  if (is.matrix(value)) {
    return(value)
  }
  rows <- lapply(value, .as_numeric_vector)
  n_row <- length(rows)
  n_col <- if (n_row > 0L) length(rows[[1L]]) else 0L
  if (n_row > 0L && any(vapply(rows, length, integer(1)) != n_col)) {
    stop("vcov result must be rectangular")
  }
  matrix(unlist(rows, use.names = FALSE), nrow = n_row, ncol = n_col, byrow = TRUE)
}

.as_coefficient_table <- function(rows) {
  if (length(rows) == 0L) {
    return(matrix(numeric(), nrow = 0L, ncol = 4L))
  }
  coefficient_names <- vapply(rows, function(row) as.character(row[["name"]]), character(1))
  values <- matrix(
    unlist(
      lapply(rows, function(row) {
        as.numeric(c(row[["coef"]], row[["se"]], row[["statistic"]], row[["p"]]))
      }),
      use.names = FALSE
    ),
    nrow = length(rows),
    ncol = 4L,
    byrow = TRUE
  )
  dimnames(values) <- list(coefficient_names, c("coef", "se(coef)", "z", "Pr(>|z|)"))
  values
}

.as_confint_matrix <- function(rows, level) {
  alpha <- 1 - level
  columns <- paste0(format(100 * c(alpha / 2, 1 - alpha / 2), trim = TRUE), " %")
  if (length(rows) == 0L) {
    result <- matrix(numeric(), nrow = 0L, ncol = 2L)
    dimnames(result) <- list(NULL, columns)
    return(result)
  }
  coefficient_names <- vapply(rows, function(row) as.character(row[["name"]]), character(1))
  values <- matrix(
    unlist(
      lapply(rows, function(row) {
        as.numeric(c(row[["lower"]], row[["upper"]]))
      }),
      use.names = FALSE
    ),
    nrow = length(rows),
    ncol = 2L,
    byrow = TRUE
  )
  dimnames(values) <- list(coefficient_names, columns)
  values
}

.as_r_data_frame <- function(x, row.names = NULL, optional = FALSE, ...) {
  columns <- .call_r_api("as_data_frame", x)
  as.data.frame(columns, row.names = row.names, optional = optional, ...)
}

.as_summary_data_frame <- function(x, class_name, ...) {
  result <- as.data.frame(x, ...)
  class(result) <- c(class_name, class(result))
  result
}

.print_tabular_result <- function(x, ...) {
  print(as.data.frame(x), ...)
  invisible(x)
}

.as_model_matrix <- function(result) {
  values <- .as_numeric_matrix(result[["data"]])
  colnames(values) <- as.character(result[["columns"]])
  values
}

.as_numeric_result <- function(value, matrix_result = FALSE, drop_single_column = FALSE,
                               col.names = NULL) {
  if (is.matrix(value)) {
    result <- matrix(
      as.numeric(value),
      nrow = nrow(value),
      ncol = ncol(value),
      dimnames = dimnames(value)
    )
    if (!is.null(col.names) && length(col.names) == ncol(result)) {
      colnames(result) <- col.names
    }
    if (drop_single_column && ncol(result) == 1L) {
      return(as.numeric(result[, 1L]))
    }
    return(result)
  }
  if (is.list(value) && !is.data.frame(value) && length(value) > 0L) {
    rows <- lapply(value, .as_numeric_vector)
    widths <- vapply(rows, length, integer(1))
    if (all(widths == widths[[1L]])) {
      if (matrix_result || widths[[1L]] != 1L) {
        result <- matrix(
          unlist(rows, use.names = FALSE),
          nrow = length(rows),
          ncol = widths[[1L]],
          byrow = TRUE
        )
        if (!is.null(col.names) && length(col.names) == ncol(result)) {
          colnames(result) <- col.names
        }
        if (drop_single_column && ncol(result) == 1L) {
          return(as.numeric(result[, 1L]))
        }
        return(result)
      }
    }
  }
  .as_numeric_vector(value)
}

.as_prediction_curve <- function(value) {
  if (!is.list(value) || length(value) != 2L ||
      !is.list(value[[2L]]) || is.data.frame(value[[2L]])) {
    return(NULL)
  }
  list(
    time = .as_numeric_vector(value[[1L]]),
    fit = .as_numeric_result(value[[2L]], matrix_result = TRUE)
  )
}

.as_prediction_value <- function(value, matrix_result = FALSE, col.names = NULL) {
  curve <- .as_prediction_curve(value)
  if (!is.null(curve)) {
    return(curve)
  }
  .as_numeric_result(value, matrix_result = matrix_result, col.names = col.names)
}

.as_prediction_result <- function(value, matrix_result = FALSE, col.names = NULL) {
  if (inherits(value, "python.builtin.object") &&
      reticulate::py_has_attr(value, "fit") &&
      reticulate::py_has_attr(value, "se_fit")) {
    fit <- .as_prediction_value(value$fit, matrix_result = matrix_result, col.names = col.names)
    se.fit <- .as_prediction_value(value$se_fit, matrix_result = matrix_result, col.names = col.names)
    if (is.list(fit) && all(c("time", "fit") %in% names(fit)) &&
        is.list(se.fit) && all(c("time", "fit") %in% names(se.fit))) {
      return(list(time = fit$time, fit = fit$fit, se.fit = se.fit$fit))
    }
    return(list(fit = fit, se.fit = se.fit))
  }
  .as_prediction_value(value, matrix_result = matrix_result, col.names = col.names)
}

.model_term_names <- function(object) {
  values <- as.character(.call_r_api("coef_names", object))
  if (inherits(object, "survival_py_survreg")) {
    values <- values[values != "(Intercept)"]
  }
  values
}

.predict_matrix_result <- function(type) {
  if (is.null(type)) {
    return(FALSE)
  }
  value <- tolower(gsub("-", "_", trimws(type)))
  startsWith("terms", value) || startsWith("quantile", value) || startsWith("uquantile", value)
}

.predict_column_names <- function(object, type) {
  if (is.null(type)) {
    return(NULL)
  }
  value <- tolower(gsub("-", "_", trimws(type)))
  if (startsWith("terms", value)) {
    return(.model_term_names(object))
  }
  NULL
}

.residual_type_key <- function(type) {
  value <- tolower(gsub("-", "_", trimws(type)))
  aliases <- c(
    mart = "martingale",
    dev = "deviance",
    dfb = "dfbeta",
    sch = "schoenfeld",
    scaled_sch = "scaledsch",
    scaled_schoenfeld = "scaledsch",
    mat = "matrix",
    part = "partial"
  )
  if (value %in% names(aliases)) {
    return(aliases[[value]])
  }
  value
}

.residual_column_names <- function(object, key) {
  if (inherits(object, "survival_py_coxph") &&
      key %in% c("score", "schoenfeld", "scaledsch", "partial")) {
    return(as.character(.call_r_api("coef_names", object)))
  }
  if (inherits(object, "survival_py_survreg") && identical(key, "matrix")) {
    return(c("g", "dg", "ddg", "ds", "dds", "dsg"))
  }
  NULL
}

.as_residual_result <- function(object, value, type) {
  key <- .residual_type_key(type)
  matrix_result <- FALSE
  drop_single_column <- FALSE
  if (inherits(object, "survival_py_coxph") &&
      key %in% c("score", "dfbeta", "dfbetas", "schoenfeld", "scaledsch", "partial")) {
    matrix_result <- TRUE
    drop_single_column <- key != "partial"
  }
  if (inherits(object, "survival_py_survreg") &&
      key %in% c("dfbeta", "dfbetas", "matrix")) {
    matrix_result <- TRUE
  }
  .as_numeric_result(
    value,
    matrix_result = matrix_result,
    drop_single_column = drop_single_column,
    col.names = .residual_column_names(object, key)
  )
}

.as_na_action <- function(value) {
  if (is.null(value) || is.character(value)) {
    return(value)
  }
  if (is.function(value)) {
    if (identical(value, stats::na.omit)) {
      return("omit")
    }
    if (identical(value, stats::na.exclude)) {
      return("exclude")
    }
    if (identical(value, stats::na.pass)) {
      return("pass")
    }
    if (identical(value, stats::na.fail)) {
      return("fail")
    }
  }
  stop("na.action must be a string, NULL, or one of stats::na.omit, na.exclude, na.pass, na.fail")
}

.as_finite_scalar <- function(value, name, positive = FALSE) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
    stop(name, " must be a finite numeric scalar", call. = FALSE)
  }
  if (positive && value <= 0) {
    stop(name, " must be > 0", call. = FALSE)
  }
  as.numeric(value)
}

.as_integer_scalar <- function(value, name, nonnegative = FALSE, positive = FALSE) {
  numeric_value <- .as_finite_scalar(value, name)
  if (numeric_value != floor(numeric_value)) {
    stop(name, " must be an integer", call. = FALSE)
  }
  if (positive && numeric_value <= 0) {
    stop(name, " must be > 0", call. = FALSE)
  }
  if (nonnegative && numeric_value < 0) {
    stop(name, " must be >= 0", call. = FALSE)
  }
  as.integer(numeric_value)
}

.as_logical_scalar <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop(name, " must be TRUE or FALSE", call. = FALSE)
  }
  value
}

.tcut_default_labels <- function(breaks) {
  formatted <- format(breaks)
  paste0(formatted[-length(formatted)], "+ thru ", formatted[-1L])
}

.is_integerish_vector <- function(value) {
  numeric_value <- suppressWarnings(as.numeric(value))
  length(numeric_value) == length(value) &&
    all(is.finite(numeric_value)) &&
    all(numeric_value == floor(numeric_value))
}

.wrap_python <- function(value, classes) {
  if (is.null(value)) {
    return(value)
  }
  class(value) <- unique(c(classes, class(value)))
  value
}

.call_r_api <- function(name, ..., .wrap = character()) {
  result <- do.call(.python_attr(name), .compact_null(list(...)))
  if (length(.wrap) > 0L) {
    return(.wrap_python(result, .wrap))
  }
  result
}

.call_data_prep <- function(name, ...) {
  do.call(.data_prep_attr(name), .compact_null(list(...)))
}

survival_python_config <- function() {
  reticulate::py_config()
}

tcut <- function(x, breaks, labels, scale = 1) {
  x <- as.numeric(x)
  breaks <- as.numeric(breaks)
  scale <- .as_finite_scalar(scale, "scale", positive = TRUE)
  if (length(breaks) < 2L) {
    stop("breaks must have at least 2 elements", call. = FALSE)
  }
  if (any(!is.finite(x)) || any(!is.finite(breaks))) {
    stop("x and breaks must contain only finite values", call. = FALSE)
  }
  if (any(diff(breaks) <= 0)) {
    stop("breaks must be strictly increasing", call. = FALSE)
  }
  labels <- if (missing(labels)) {
    .tcut_default_labels(breaks)
  } else {
    as.character(labels)
  }
  if (length(labels) != length(breaks) - 1L) {
    stop("labels length must equal length(breaks) - 1", call. = FALSE)
  }
  # Validate against the Rust-backed implementation while returning R's tcut shape.
  .call_data_prep("tcut", x, breaks, labels)
  structure(
    x * scale,
    cutpoints = breaks * scale,
    labels = labels,
    class = "tcut"
  )
}

neardate <- function(id1, id2, y1, y2, best = c("after", "prior"), nomatch = NA_integer_) {
  if (missing(best)) {
    best <- "after"
  }
  best <- match.arg(best, c("after", "prior", "closest"))
  if (length(id1) != length(y1)) {
    stop("id1 and y1 must have the same length", call. = FALSE)
  }
  if (length(id2) != length(y2)) {
    stop("id2 and y2 must have the same length", call. = FALSE)
  }
  if (length(nomatch) != 1L) {
    stop("nomatch must be a scalar", call. = FALSE)
  }
  y1 <- as.numeric(y1)
  y2 <- as.numeric(y2)
  if (any(!is.finite(y1)) || any(!is.finite(y2))) {
    stop("y1 and y2 must contain only finite values", call. = FALSE)
  }

  if (.is_integerish_vector(id1) && .is_integerish_vector(id2)) {
    result <- .call_data_prep(
      "neardate",
      as.integer(id1),
      y1,
      as.integer(id2),
      y2,
      best = best
    )
  } else {
    result <- .call_data_prep(
      "neardate_str",
      as.character(id1),
      y1,
      as.character(id2),
      y2,
      best = best
    )
  }

  indices <- result$indices
  matched <- !vapply(indices, is.null, logical(1))
  values <- rep(nomatch, length(indices))
  values[matched] <- as.integer(unlist(indices[matched], use.names = FALSE)) + 1L
  values
}

Surv <- function(time, time2, event, type = NULL, origin = 0, time1, start, stop, status) {
  use_named_response <- !missing(time1) || !missing(start) ||
    !missing(stop) || !missing(status)

  if (use_named_response) {
    args <- list()
    if (!missing(time)) {
      args$time <- time
    }
    if (!missing(time1)) {
      args$time1 <- time1
    }
    if (!missing(start)) {
      args$start <- start
    }
    if (!missing(time2)) {
      args$time2 <- time2
    }
    if (!missing(stop)) {
      args$stop <- stop
    }
    if (!missing(event)) {
      args$event <- event
    }
    if (!missing(status)) {
      args$status <- status
    }
  } else {
    args <- list(time)
    if (!missing(time2)) {
      args <- c(args, list(time2))
    }
    if (!missing(event)) {
      args <- c(args, list(event))
    }
  }
  args <- c(args, .compact_null(list(type = type, origin = origin)))
  .wrap_python(do.call(.python_attr("Surv"), args), c("survival_py_surv", "survival_py_object"))
}

is.Surv <- function(value) {
  .call_r_api("is_surv", value)
}

coxph.control <- function(eps = 1e-09, toler.chol = .Machine$double.eps^0.75,
                          iter.max = 20, toler.inf = sqrt(eps), outer.max = 10,
                          timefix = TRUE) {
  eps <- .as_finite_scalar(eps, "eps", positive = TRUE)
  toler.chol <- .as_finite_scalar(toler.chol, "toler.chol", positive = TRUE)
  iter.max <- .as_integer_scalar(iter.max, "iter.max", nonnegative = TRUE)
  toler.inf <- .as_finite_scalar(toler.inf, "toler.inf", positive = TRUE)
  outer.max <- .as_integer_scalar(outer.max, "outer.max", positive = TRUE)
  timefix <- .as_logical_scalar(timefix, "timefix")
  list(
    eps = eps,
    toler.chol = toler.chol,
    iter.max = iter.max,
    toler.inf = toler.inf,
    outer.max = outer.max,
    timefix = timefix
  )
}

survreg.control <- function(maxiter = 30, rel.tolerance = 1e-09,
                            toler.chol = 1e-10, iter.max, debug = 0,
                            outer.max = 10) {
  if (missing(iter.max)) {
    iter.max <- maxiter
  } else {
    maxiter <- iter.max
  }
  iter.max <- .as_integer_scalar(iter.max, "iter.max", nonnegative = TRUE)
  maxiter <- .as_integer_scalar(maxiter, "maxiter", nonnegative = TRUE)
  rel.tolerance <- .as_finite_scalar(rel.tolerance, "rel.tolerance", positive = TRUE)
  toler.chol <- .as_finite_scalar(toler.chol, "toler.chol", positive = TRUE)
  debug <- .as_finite_scalar(debug, "debug")
  outer.max <- .as_integer_scalar(outer.max, "outer.max", positive = TRUE)
  list(
    iter.max = iter.max,
    rel.tolerance = rel.tolerance,
    toler.chol = toler.chol,
    debug = debug,
    maxiter = maxiter,
    outer.max = outer.max
  )
}

survfit <- function(formula, ...) {
  UseMethod("survfit")
}

survfit.formula <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  .call_r_api(
    "survfit",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

survfit.character <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  .call_r_api(
    "survfit",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

survfit.survival_py_surv <- function(formula, ..., group = NULL, subset = NULL, na.action = "fail") {
  .call_r_api(
    "survfit",
    response = formula,
    group = if (is.null(group)) NULL else .as_python_vector(group),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

survfit.survival_py_coxph <- function(formula, newdata = NULL, ..., se.fit = TRUE) {
  .call_r_api(
    "survfit",
    response = formula,
    newdata = .as_python_data(newdata),
    `se.fit` = se.fit,
    ...,
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

survdiff <- function(formula, data = NULL, ..., group = NULL, subset = NULL, na.action = "fail") {
  .call_r_api(
    "survdiff",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    group = if (is.null(group)) NULL else .as_python_vector(group),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survdiff", "survival_py_object")
  )
}

coxph <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  .call_r_api(
    "coxph",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_coxph", "survival_py_model", "survival_py_object")
  )
}

survreg <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  .call_r_api(
    "survreg",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survreg", "survival_py_model", "survival_py_object")
  )
}

basehaz <- function(fit, ..., centered = TRUE) {
  .call_r_api(
    "basehaz",
    fit,
    centered = centered,
    ...,
    .wrap = c("survival_py_basehaz", "survival_py_object")
  )
}

concordance <- function(formula, data = NULL, ..., scores = NULL, risk.scores = NULL,
                        weights = NULL, cluster = NULL, subset = NULL, na.action = "fail") {
  .call_r_api(
    "concordance",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    scores = .as_python_optional_vector(scores),
    risk_scores = .as_python_optional_vector(risk.scores),
    weights = .as_python_optional_vector(weights),
    cluster = .as_python_optional_vector(cluster),
    subset = subset,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_concordance", "survival_py_object")
  )
}

cox.zph <- function(fit, ...) {
  .call_r_api("cox_zph", fit, ..., .wrap = c("survival_py_cox_zph", "survival_py_object"))
}

cox_zph <- function(fit, ...) {
  cox.zph(fit, ...)
}

coxph.detail <- function(fit, ...) {
  .call_r_api(
    "coxph_detail",
    fit,
    ...,
    .wrap = c("survival_py_coxph_detail", "survival_py_object")
  )
}

coxph_detail <- function(fit, ...) {
  coxph.detail(fit, ...)
}

coef.survival_py_model <- function(object, ...) {
  values <- .as_numeric_vector(.call_r_api("coef", object, ...))
  names(values) <- as.character(.call_r_api("coef_names", object))
  values
}

vcov.survival_py_model <- function(object, ..., complete = TRUE) {
  values <- .as_numeric_matrix(.call_r_api("vcov", object, complete = complete, ...))
  coefficient_names <- as.character(.call_r_api("coef_names", object, complete = complete))
  dimnames(values) <- list(coefficient_names, coefficient_names)
  values
}

confint.survival_py_model <- function(object, parm, level = 0.95, ...) {
  selected <- if (missing(parm)) NULL else parm
  result <- .call_r_api("confint", object, parm = selected, level = level, ...)
  .as_confint_matrix(result, level)
}

logLik.survival_py_model <- function(object, ...) {
  value <- as.numeric(.call_r_api("loglik", object, ...))
  structure(
    value,
    df = as.integer(.call_r_api("degrees_freedom", object)),
    nobs = as.integer(.call_r_api("nobs", object)),
    class = "logLik"
  )
}

nobs.survival_py_model <- function(object, ...) {
  as.integer(.call_r_api("nobs", object, ...))
}

df.residual.survival_py_survreg <- function(object, ...) {
  as.integer(.call_r_api("df_residual", object, ...))
}

extractAIC.survival_py_model <- function(fit, scale = 0, k = 2, ...) {
  values <- .as_numeric_vector(.call_r_api("extract_aic", fit, scale = scale, k = k, ...))
  names(values) <- c("df", "AIC")
  values
}

formula.survival_py_model <- function(x, ...) {
  stats::as.formula(.call_r_api("model_formula", x))
}

terms.survival_py_model <- function(x, ...) {
  stats::terms(stats::formula(x), ...)
}

weights.survival_py_model <- function(object, ...) {
  values <- .call_r_api("model_weights", object)
  if (is.null(values)) {
    return(NULL)
  }
  .as_numeric_vector(values)
}

model.matrix.survival_py_model <- function(object, ...) {
  .as_model_matrix(.call_r_api("model_matrix", object, ...))
}

model.frame.survival_py_model <- function(formula, ...) {
  columns <- .call_r_api("model_frame", formula, ...)
  as.data.frame(columns, ...)
}

fitted.survival_py_model <- function(object, ..., type = NULL, se.fit = FALSE) {
  result <- .call_r_api(
    "fitted",
    object,
    type = type,
    `se.fit` = se.fit,
    ...
  )
  .as_prediction_result(
    result,
    matrix_result = .predict_matrix_result(type),
    col.names = .predict_column_names(object, type)
  )
}

summary.survival_py_model <- function(object, ...) {
  result <- .call_r_api("model_summary", object, ...)
  result$coefficients <- .as_coefficient_table(result$coefficients)
  class(result) <- c("summary.survival_py_model", class(result))
  result
}

predict.survival_py_model <- function(object, newdata = NULL, ..., type = NULL, se.fit = FALSE) {
  result <- .call_r_api(
    "predict",
    object,
    newdata = .as_python_data(newdata),
    type = type,
    `se.fit` = se.fit,
    ...
  )
  .as_prediction_result(
    result,
    matrix_result = .predict_matrix_result(type),
    col.names = .predict_column_names(object, type)
  )
}

residuals.survival_py_model <- function(object, ..., type = "martingale") {
  result <- .call_r_api("residuals", object, type = type, ...)
  .as_residual_result(object, result, type)
}

anova.survival_py_model <- function(object, ..., test = "Chisq") {
  .call_r_api(
    "anova",
    object,
    ...,
    test = test,
    .wrap = c("survival_py_anova", "survival_py_object")
  )
}

summary.survival_py_survfit <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_survfit", ...)
}

summary.survival_py_basehaz <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_basehaz", ...)
}

summary.survival_py_survdiff <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_survdiff", ...)
}

summary.survival_py_concordance <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_concordance", ...)
}

summary.survival_py_cox_zph <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_cox_zph", ...)
}

summary.survival_py_coxph_detail <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_coxph_detail", ...)
}

summary.survival_py_anova <- function(object, ...) {
  .as_summary_data_frame(object, "summary.survival_py_anova", ...)
}

as.data.frame.survival_py_survfit <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_surv <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_basehaz <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_survdiff <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_concordance <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_cox_zph <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_coxph_detail <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_anova <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

print.survival_py_survfit <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_surv <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_basehaz <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_survdiff <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_concordance <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_cox_zph <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_coxph_detail <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_anova <- function(x, ...) {
  .print_tabular_result(x, ...)
}

print.survival_py_model <- function(x, ...) {
  cat("Call:\n")
  call <- tryCatch(stats::formula(x), error = function(e) NULL)
  if (is.null(call)) {
    cat("<unavailable>\n")
  } else {
    print(call)
  }

  cat("\nCoefficients:\n")
  print(coef(x), ...)

  likelihood <- logLik(x)
  cat(
    "\nlogLik=", format(as.numeric(likelihood)),
    " df=", attr(likelihood, "df"),
    " n=", attr(likelihood, "nobs"),
    "\n",
    sep = ""
  )
  invisible(x)
}

print.survival_py_object <- function(x, ...) {
  if (!inherits(x, "python.builtin.object")) {
    return(NextMethod())
  }
  cat(as.character(reticulate::py_str(x)), "\n")
  invisible(x)
}

print.summary.survival_py_model <- function(x, ...) {
  cat(x$model_type, "model summary\n", sep = "")
  print(x$coefficients, ...)
  cat("logLik=", x$loglik, " df=", x$df, " n=", x$n, "\n", sep = "")
  invisible(x)
}
