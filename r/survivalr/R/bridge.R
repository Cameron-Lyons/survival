.survival_python_module <- local({
  module <- NULL

  function() {
    if (is.null(module)) {
      module <<- reticulate::import("survival.r_api", convert = TRUE)
    }
    module
  }
})

if (getRversion() >= "2.15.1") {
  utils::globalVariables("nclass")
}

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

.survival_core_module <- local({
  module <- NULL

  function() {
    if (is.null(module)) {
      module <<- reticulate::import("survival.core", convert = TRUE)
    }
    module
  }
})

.core_attr <- function(name) {
  reticulate::py_get_attr(.survival_core_module(), name)
}

.survival_regression_module <- local({
  module <- NULL

  function() {
    if (is.null(module)) {
      module <<- reticulate::import("survival.regression", convert = TRUE)
    }
    module
  }
})

.regression_attr <- function(name) {
  reticulate::py_get_attr(.survival_regression_module(), name)
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

.formula_has_offset <- function(formula, data = NULL) {
  formula_value <- if (inherits(formula, "formula")) {
    formula
  } else if (is.character(formula)) {
    stats::as.formula(paste(formula, collapse = " "))
  } else {
    return(FALSE)
  }
  formula_terms <- tryCatch(
    if (is.null(data)) {
      stats::terms(formula_value)
    } else {
      stats::terms(formula_value, data = data)
    },
    error = function(e) NULL
  )
  !is.null(formula_terms) && length(attr(formula_terms, "offset")) > 0L
}

.restore_r_column_classes <- function(frame, data) {
  if (is.null(data) || !is.data.frame(data)) {
    return(frame)
  }
  for (column in names(data)) {
    source <- data[[column]]
    if (is.factor(source)) {
      if (column %in% names(frame)) {
        frame[[column]] <- factor(
          as.character(frame[[column]]),
          levels = levels(source),
          ordered = is.ordered(source)
        )
      }
      factor_name <- paste0("factor(", column, ")")
      if (factor_name %in% names(frame)) {
        frame[[factor_name]] <- factor(
          as.character(frame[[factor_name]]),
          levels = levels(factor(source))
        )
      }
    } else if (inherits(source, "Date") && column %in% names(frame)) {
      frame[[column]] <- as.Date(as.numeric(frame[[column]]), origin = "1970-01-01")
    } else if (inherits(source, "POSIXct") && column %in% names(frame)) {
      timezone <- attr(source, "tzone")
      if (is.null(timezone) || length(timezone) == 0L) {
        timezone <- ""
      }
      frame[[column]] <- as.POSIXct(
        as.numeric(frame[[column]]),
        origin = "1970-01-01",
        tz = timezone[[1L]]
      )
    }
  }
  frame
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

.eval_formula_dots <- function(dots, data, env, vector_args = character()) {
  if (is.null(dots) || length(dots) == 0L) {
    return(list())
  }
  dot_names <- names(dots)
  if (is.null(dot_names)) {
    dot_names <- rep("", length(dots))
  }
  values <- vector("list", length(dots))
  for (idx in seq_along(dots)) {
    value <- if (is.null(data)) {
      eval(dots[[idx]], env)
    } else {
      eval(dots[[idx]], data, env)
    }
    if (nzchar(dot_names[[idx]]) && dot_names[[idx]] %in% vector_args && !is.null(value)) {
      value <- .as_python_vector(value)
    }
    values[[idx]] <- value
  }
  names(values) <- dot_names
  values
}

.eval_formula_arg <- function(expr, missing_arg, data, env, vector = FALSE) {
  if (isTRUE(missing_arg)) {
    return(NULL)
  }
  value <- if (is.null(data)) {
    eval(expr, env)
  } else {
    eval(expr, data, env)
  }
  if (isTRUE(vector)) {
    .as_python_optional_vector(value)
  } else {
    value
  }
}

.strata_expr_labels <- function(exprs, n) {
  if (n == 0L) {
    return(character())
  }
  vapply(seq_len(n), function(idx) {
    paste(deparse(exprs[[idx]], width.cutoff = 500L), collapse = " ")
  }, character(1))
}

.as_strata_factor <- function(result) {
  codes <- result$codes
  if (!is.list(codes)) {
    codes <- as.list(codes)
  }
  code_values <- vapply(codes, function(value) {
    if (is.null(value) || length(value) == 0L || anyNA(value)) {
      NA_integer_
    } else {
      as.integer(value)
    }
  }, integer(1))
  factor(code_values, levels = seq_along(result$levels), labels = as.character(result$levels))
}

strata <- function(..., na.group = FALSE, shortlabel, sep = ", ") {
  values <- list(...)
  exprs <- as.list(substitute(list(...)))[-1L]
  if (length(values) == 1L && is.list(unclass(values[[1L]]))) {
    values <- unclass(values[[1L]])
    list_expr <- substitute(list(...))[[2L]]
    if (is.call(list_expr) && identical(list_expr[[1L]], quote(list))) {
      exprs <- as.list(list_expr)[-1L]
    } else {
      exprs <- as.list(seq_along(values))
    }
  }
  n_terms <- length(values)
  if (n_terms == 0L) {
    stop("strata requires at least one variable", call. = FALSE)
  }

  value_names <- names(values)
  expr_labels <- .strata_expr_labels(exprs, n_terms)
  if (is.null(value_names)) {
    arg_names <- expr_labels
    shortlabel_value <- if (missing(shortlabel)) {
      all(vapply(values, function(value) is.character(value) || is.factor(value), logical(1)))
    } else {
      shortlabel
    }
  } else {
    arg_names <- ifelse(value_names == "", expr_labels, value_names)
    shortlabel_value <- if (missing(shortlabel)) FALSE else shortlabel
  }

  py_values <- lapply(values, .as_python_vector)
  names(py_values) <- NULL
  args <- c(
    py_values,
    list(
      na_group = na.group,
      shortlabel = shortlabel_value,
      sep = sep,
      labels = as.list(arg_names)
    )
  )
  .as_strata_factor(do.call(.python_attr("strata"), args))
}

cluster <- function(x) {
  x
}

untangle.specials <- function(tt, special, order = 1) {
  spc <- attr(tt, "specials")[[special]]
  if (length(spc) == 0L) {
    return(list(vars = character(0), terms = numeric(0)))
  }
  facs <- attr(tt, "factors")
  fname <- dimnames(facs)
  ff <- apply(facs[spc, , drop = FALSE], 2L, sum)
  list(
    vars = fname[[1L]][spc],
    tvar = spc - attr(tt, "response"),
    terms = seq(ff)[ff & match(attr(tt, "order"), order, nomatch = 0)]
  )
}

attrassign <- function(object, ...) {
  UseMethod("attrassign")
}

attrassign.default <- function(object, tt, ...) {
  if (!inherits(tt, "terms")) {
    stop("need terms object", call. = FALSE)
  }
  aa <- attr(object, "assign")
  if (is.null(aa)) {
    stop("argument is not really a model matrix", call. = FALSE)
  }
  labels <- attr(tt, "term.labels")
  term_names <- c("(Intercept)", labels)[aa + 1L]
  split(seq(along.with = term_names), factor(term_names, levels = unique(term_names)))
}

attrassign.lm <- function(object, ...) {
  attrassign(stats::model.matrix(object), stats::terms(object))
}

.as_python_surv <- function(value) {
  if (inherits(value, "survival_py_surv")) {
    return(value)
  }
  if (!inherits(value, "Surv") || !is.matrix(value)) {
    return(value)
  }
  surv_type <- attr(value, "type")
  if (is.null(surv_type)) {
    surv_type <- if (ncol(value) == 3L) "counting" else "right"
  }
  if (ncol(value) == 2L) {
    return(.wrap_python(
      .python_attr("Surv")(as.numeric(value[, 1L]), as.numeric(value[, 2L]), type = surv_type),
      c("survival_py_surv", "survival_py_object")
    ))
  }
  if (ncol(value) == 3L) {
    return(.wrap_python(
      .python_attr("Surv")(
        as.numeric(value[, 1L]),
        as.numeric(value[, 2L]),
        as.numeric(value[, 3L]),
        type = surv_type
      ),
      c("survival_py_surv", "survival_py_object")
    ))
  }
  stop("unsupported Surv matrix shape", call. = FALSE)
}

.survsplit_arg_label <- function(expr, name = "") {
  if (!is.null(name) && nzchar(name)) {
    return(name)
  }
  paste(deparse(expr, width.cutoff = 500L), collapse = " ")
}

.survsplit_response_names <- function(formula, response) {
  response_names <- colnames(response)
  if (!is.null(response_names) && length(response_names) >= 2L) {
    return(response_names)
  }
  response_expr <- formula[[2L]]
  if (!is.call(response_expr)) {
    return(character())
  }
  args <- as.list(response_expr[-1L])
  arg_names <- names(args)
  if (is.null(arg_names)) {
    arg_names <- rep("", length(args))
  }
  vapply(seq_along(args), function(idx) {
    .survsplit_arg_label(args[[idx]], arg_names[[idx]])
  }, character(1))
}

.survsplit_minimal_surv <- function(args) {
  surv_type <- args[["type"]]
  origin <- args[["origin"]]
  if (is.null(origin)) {
    origin <- 0
  }
  args[["type"]] <- NULL
  args[["origin"]] <- NULL
  labels <- names(args)
  if (is.null(labels)) {
    labels <- rep("", length(args))
  }
  if (length(args) == 2L) {
    empty <- !nzchar(labels)
    labels[empty] <- c("time", "status")[empty]
    result <- cbind(as.numeric(args[[1L]]) - origin, as.numeric(args[[2L]]))
    colnames(result) <- labels
    attr(result, "type") <- if (is.null(surv_type)) "right" else surv_type
    class(result) <- "Surv"
    return(result)
  }
  if (length(args) >= 3L) {
    first_three <- seq_len(3L)
    empty <- !nzchar(labels[first_three])
    labels[first_three[empty]] <- c("start", "stop", "status")[empty]
    result <- cbind(as.numeric(args[[1L]]) - origin, as.numeric(args[[2L]]) - origin, as.numeric(args[[3L]]))
    colnames(result) <- labels[seq_len(3L)]
    attr(result, "type") <- if (is.null(surv_type)) "counting" else surv_type
    class(result) <- "Surv"
    return(result)
  }
  stop("Surv response requires at least two arguments", call. = FALSE)
}

.survsplit_model_frame_surv <- function(time, time2, event, type = NULL,
                                        origin = 0, time1, start, stop, status) {
  use_named_response <- !missing(time1) || !missing(start) ||
    !missing(stop) || !missing(status)

  if (use_named_response) {
    args <- list()
    if (!missing(time)) {
      args$time <- time
    }
    if (!missing(time1)) {
      args$time <- time1
    }
    if (!missing(start)) {
      args$time <- start
    }
    if (!missing(time2)) {
      args$time2 <- time2
    }
    if (!missing(stop)) {
      args$time2 <- stop
    }
    if (!missing(event)) {
      args$event <- event
    }
    if (!missing(status)) {
      args$event <- status
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
  .survsplit_minimal_surv(args)
}

.as_numeric_vector <- function(value) {
  as.numeric(unlist(value, recursive = TRUE, use.names = FALSE))
}

.as_nullable_character_vector <- function(value) {
  items <- if (is.list(value)) value else as.list(value)
  vapply(items, function(item) {
    if (is.null(item) || length(item) == 0L || anyNA(item)) {
      NA_character_
    } else {
      as.character(item)[[1L]]
    }
  }, character(1))
}

.as_nullable_numeric_vector <- function(value) {
  items <- if (is.list(value)) value else as.list(value)
  vapply(items, function(item) {
    if (is.null(item) || length(item) == 0L || anyNA(item)) {
      NA_real_
    } else {
      as.numeric(item)[[1L]]
    }
  }, numeric(1))
}

.as_nullable_logical_vector <- function(value) {
  items <- if (is.list(value)) value else as.list(value)
  vapply(items, function(item) {
    if (is.null(item) || length(item) == 0L || anyNA(item)) {
      NA
    } else {
      as.logical(item)[[1L]]
    }
  }, logical(1))
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

.as_coefficient_table <- function(rows, model_type = "coxph", robust = FALSE,
                                  scale = 1) {
  model_type <- as.character(model_type)[[1L]]
  robust <- length(robust) > 0L && isTRUE(as.logical(robust)[[1L]])
  is_survreg <- identical(model_type, "survreg")
  cox_scale <- if (is_survreg) 1 else as.numeric(scale)[[1L]]
  columns <- if (is_survreg) {
    if (robust) {
      c("Value", "Std. Err", "(Naive SE)", "z", "p")
    } else {
      c("Value", "Std. Error", "z", "p")
    }
  } else if (robust) {
    c("coef", "exp(coef)", "se(coef)", "robust se", "z", "Pr(>|z|)")
  } else {
    c("coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)")
  }
  if (length(rows) == 0L) {
    result <- matrix(numeric(), nrow = 0L, ncol = length(columns))
    dimnames(result) <- list(NULL, columns)
    return(result)
  }

  row_numeric <- function(row, name, fallback_name = NULL, default = NA_real_) {
    value <- row[[name]]
    if ((is.null(value) || length(value) == 0L) && !is.null(fallback_name)) {
      value <- row[[fallback_name]]
    }
    if (is.null(value) || length(value) == 0L) {
      return(default)
    }
    as.numeric(value)[[1L]]
  }
  row_values <- function(row) {
    coefficient <- row_numeric(row, "coef")
    if (is_survreg) {
      statistic <- row_numeric(row, "z", fallback_name = "statistic")
      values <- c(
        row_numeric(row, "value", fallback_name = "coef"),
        if (robust) {
          row_numeric(row, "robust_se", fallback_name = "se")
        } else {
          row_numeric(row, "se")
        }
      )
      if (robust) {
        values <- c(values, row_numeric(row, "naive_se", fallback_name = "se"))
      }
      return(c(values, statistic, row_numeric(row, "p")))
    }

    coefficient <- coefficient * cox_scale
    active_se <- row_numeric(
      row,
      if (robust) "robust_se" else "se",
      fallback_name = "se"
    ) * cox_scale
    statistic <- coefficient / active_se
    values <- c(
      coefficient,
      exp(coefficient)
    )
    if (robust) {
      values <- c(
        values,
        row_numeric(row, "naive_se", fallback_name = "se"),
        active_se
      )
    } else {
      values <- c(values, active_se)
    }
    c(values, statistic, stats::pchisq(statistic^2, 1, lower.tail = FALSE))
  }

  coefficient_names <- vapply(rows, function(row) as.character(row[["name"]]), character(1))
  values <- matrix(
    unlist(lapply(rows, row_values), use.names = FALSE),
    nrow = length(rows),
    ncol = length(columns),
    byrow = TRUE
  )
  dimnames(values) <- list(coefficient_names, columns)
  values
}

.as_cox_confint_table <- function(coefficient_table, conf.int) {
  coefficient <- coefficient_table[, "coef"]
  se_column <- if ("robust se" %in% colnames(coefficient_table)) {
    "robust se"
  } else {
    "se(coef)"
  }
  standard_error <- coefficient_table[, se_column]
  z <- stats::qnorm((1 + conf.int) / 2)
  values <- cbind(
    exp(coefficient),
    exp(-coefficient),
    exp(coefficient - z * standard_error),
    exp(coefficient + z * standard_error)
  )
  dimnames(values) <- list(
    rownames(coefficient_table),
    c(
      "exp(coef)",
      "exp(-coef)",
      paste("lower .", round(100 * conf.int, 2), sep = ""),
      paste("upper .", round(100 * conf.int, 2), sep = "")
    )
  )
  values
}

.cox_summary_test <- function(test, df, round.test = FALSE) {
  if (length(test) == 0L) {
    return(c(df = df))
  }
  raw_test <- as.numeric(test)[[1L]]
  c(
    test = if (round.test) round(raw_test, 2) else raw_test,
    df = df,
    pvalue = stats::pchisq(raw_test, df, lower.tail = FALSE)
  )
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

.survival_py_summary_logical <- function(value, name) {
  if (!is.logical(value) || length(value) != 1L || is.na(value)) {
    stop(name, " must be TRUE/FALSE", call. = FALSE)
  }
  value
}

.survival_py_summary_scale <- function(scale) {
  if (!is.numeric(scale) || length(scale) != 1L || !is.finite(scale) || scale <= 0) {
    stop("scale must be a positive finite number", call. = FALSE)
  }
  scale
}

.survival_py_summary_initial_value <- function(name, values) {
  switch(
    name,
    time = 0,
    n.event = 0,
    n.censor = 0,
    n.enter = 0,
    surv = 1,
    cumhaz = 0,
    std.err = 0,
    std.chaz = 0,
    lower = 1,
    upper = 1,
    if (length(values) > 0L) values[[1L]] else NA
  )
}

.survival_py_summary_augmented_curve <- function(curve) {
  if (nrow(curve) == 0L || !("time" %in% names(curve))) {
    return(curve)
  }
  if (min(curve$time, na.rm = TRUE) <= 0) {
    return(curve)
  }
  initial <- curve[1L, , drop = FALSE]
  for (name in names(initial)) {
    initial[[name]] <- .survival_py_summary_initial_value(name, curve[[name]])
  }
  rbind(initial, curve)
}

.survival_py_summary_delta <- function(values, indices) {
  cumulative <- c(0, cumsum(values))
  diff(c(0, cumulative[indices + 1L]))
}

.survival_py_summary_curve_at_times <- function(curve, times, extend, dosum) {
  if (nrow(curve) == 0L || !("time" %in% names(curve))) {
    return(curve[0L, , drop = FALSE])
  }

  selected_times <- times
  if (!extend) {
    selected_times <- selected_times[selected_times <= max(curve$time, na.rm = TRUE)]
  }
  if (length(selected_times) == 0L) {
    return(curve[0L, , drop = FALSE])
  }

  augmented <- .survival_py_summary_augmented_curve(curve)
  index1 <- findInterval(selected_times, augmented$time)
  step_index <- pmax(1L, index1)
  result <- augmented[step_index, , drop = FALSE]
  row.names(result) <- NULL
  result$time <- selected_times

  if ("n.risk" %in% names(result)) {
    risk_index <- 1L + findInterval(selected_times, augmented$time, left.open = TRUE)
    result[["n.risk"]] <- c(augmented[["n.risk"]], 0)[risk_index]
  }

  for (name in intersect(c("n.event", "n.censor", "n.enter"), names(result))) {
    if (dosum) {
      result[[name]] <- .survival_py_summary_delta(augmented[[name]], index1)
    } else {
      result[[name]] <- augmented[[name]][step_index]
    }
  }

  result
}

.survival_py_summary_split_frame <- function(frame) {
  group_columns <- intersect(c("strata", "curve"), names(frame))
  if (length(group_columns) == 0L) {
    return(list(frame))
  }
  key_frame <- frame[group_columns]
  keys <- do.call(paste, c(key_frame, sep = "\r"))
  split(frame, factor(keys, levels = unique(keys)), drop = TRUE)
}

.survival_py_survfit_summary_frame <- function(object, times, censored, scale,
                                               extend, data.frame, dosum, ...) {
  censored <- .survival_py_summary_logical(censored, "censored")
  extend <- .survival_py_summary_logical(extend, "extend")
  data.frame <- .survival_py_summary_logical(data.frame, "data.frame")
  scale <- .survival_py_summary_scale(scale)
  frame <- as.data.frame.survival_py_survfit(object, optional = TRUE)

  if (missing(times)) {
    if (!censored && "n.event" %in% names(frame)) {
      frame <- frame[frame[["n.event"]] > 0, , drop = FALSE]
      row.names(frame) <- NULL
    }
  } else {
    if (length(times) == 0L) {
      stop("no values in times vector", call. = FALSE)
    }
    if (inherits(times, "Date")) {
      times <- as.numeric(times)
    }
    if (!is.numeric(times)) {
      stop("times must be a numeric vector", call. = FALSE)
    }
    if (any(is.na(times))) {
      stop("times contains missing values", call. = FALSE)
    }
    if (missing(dosum)) {
      dosum <- all(diff(times) > 0)
    } else {
      dosum <- .survival_py_summary_logical(dosum, "dosum")
      if (dosum && !all(diff(times) > 0)) {
        stop("dosum=TRUE requires the times to be increasing", call. = FALSE)
      }
    }
    pieces <- lapply(
      .survival_py_summary_split_frame(frame),
      .survival_py_summary_curve_at_times,
      times = times,
      extend = extend,
      dosum = dosum
    )
    frame <- do.call(rbind, pieces)
    row.names(frame) <- NULL
  }

  if ("time" %in% names(frame) && scale != 1) {
    frame$time <- frame$time / scale
  }
  class(frame) <- c("summary.survival_py_survfit", class(frame))
  frame
}

.print_tabular_result <- function(x, ...) {
  print(as.data.frame(x), ...)
  invisible(x)
}

.as_model_matrix <- function(result) {
  values <- .as_numeric_matrix(result[["data"]])
  colnames(values) <- as.character(result[["columns"]])
  assign <- result[["assign"]]
  if (!is.null(assign)) {
    assign <- as.integer(unlist(assign, recursive = TRUE, use.names = FALSE))
    if (length(assign) != ncol(values)) {
      stop("model matrix assign metadata must match its column count", call. = FALSE)
    }
    attr(values, "assign") <- assign
  }
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

.pseudo_model_frame <- function(fit) {
  curve <- if (inherits(fit, "python.builtin.object")) {
    fit
  } else if (is.list(fit) && !is.data.frame(fit) && length(fit) > 0L) {
    fit[[1L]]
  } else {
    NULL
  }
  if (!inherits(curve, "python.builtin.object") || !reticulate::py_has_attr(curve, "model")) {
    return(NULL)
  }
  model <- reticulate::py_to_r(reticulate::py_get_attr(curve, "model"))
  if (!is.list(model)) {
    return(NULL)
  }
  model
}

.pseudo_group_values <- function(fit) {
  model <- .pseudo_model_frame(fit)
  if (is.null(model) || is.null(model[["group"]])) {
    return(NULL)
  }
  as.character(model[["group"]])
}

.pseudo_id_values <- function(fit) {
  model <- .pseudo_model_frame(fit)
  if (is.null(model) || is.null(model[["(id)"]])) {
    return(NULL)
  }
  unname(unlist(model[["(id)"]], use.names = FALSE))
}

.pseudo_id_row_names <- function(fit, n) {
  id_values <- .pseudo_id_values(fit)
  if (is.null(id_values)) {
    return(NULL)
  }
  if (length(id_values) == n) {
    return(id_values)
  }
  unique_ids <- unique(id_values)
  if (length(unique_ids) == n) {
    return(unique_ids)
  }
  NULL
}

.pseudo_apply_id_dimnames <- function(result, fit, col.names = NULL) {
  if (!is.matrix(result)) {
    return(result)
  }
  row_names <- .pseudo_id_row_names(fit, nrow(result))
  if (is.null(row_names)) {
    return(result)
  }
  if (!is.null(col.names) && length(col.names) == ncol(result)) {
    colnames(result) <- col.names
  }
  rownames(result) <- row_names
  names(dimnames(result)) <- c("(id)", "times")
  result
}

.is_grouped_pseudo_result <- function(value) {
  is.list(value) &&
    !is.data.frame(value) &&
    length(value) > 0L &&
    !is.null(names(value)) &&
    all(nzchar(names(value)))
}

.as_pseudo_matrix <- function(value, fit, matrix_result = FALSE,
                              drop_single_column = FALSE, col.names = NULL) {
  if (.is_grouped_pseudo_result(value)) {
    blocks <- lapply(value, .as_numeric_result, matrix_result = TRUE)
    widths <- vapply(blocks, ncol, integer(1))
    if (all(widths == widths[[1L]])) {
      group_values <- .pseudo_group_values(fit)
      if (!is.null(group_values)) {
        block_rows <- vapply(blocks, nrow, integer(1))
        group_rows <- vapply(names(blocks), function(label) {
          sum(group_values == label)
        }, integer(1))
        if (identical(unname(block_rows), unname(group_rows))) {
          result <- matrix(
            NA_real_,
            nrow = length(group_values),
            ncol = widths[[1L]]
          )
          for (label in names(blocks)) {
            indices <- which(group_values == label)
            result[indices, ] <- blocks[[label]]
          }
        } else {
          id_values <- .pseudo_id_values(fit)
          if (is.null(id_values) || length(id_values) != length(group_values)) {
            stop("grouped pseudo result does not match the stored survfit model frame",
                 call. = FALSE)
          }
          unique_ids <- unique(id_values)
          id_groups <- vapply(unique_ids, function(id_value) {
            group_values[which(id_values == id_value)[[1L]]]
          }, character(1))
          id_group_rows <- vapply(names(blocks), function(label) {
            sum(id_groups == label)
          }, integer(1))
          if (!identical(unname(block_rows), unname(id_group_rows))) {
            stop("grouped pseudo result does not match the stored survfit model frame",
                 call. = FALSE)
          }
          result <- matrix(
            NA_real_,
            nrow = length(unique_ids),
            ncol = widths[[1L]]
          )
          rownames(result) <- unique_ids
          for (label in names(blocks)) {
            indices <- which(id_groups == label)
            result[indices, ] <- blocks[[label]]
          }
        }
      } else {
        result <- do.call(rbind, blocks)
      }
      if (!is.null(col.names) && length(col.names) == ncol(result)) {
        colnames(result) <- col.names
      }
      result <- .pseudo_apply_id_dimnames(result, fit, col.names = col.names)
      if (drop_single_column && ncol(result) == 1L) {
        vector_result <- as.numeric(result[, 1L])
        names(vector_result) <- rownames(result)
        return(vector_result)
      }
      return(result)
    }
  }
  force_matrix <- !is.null(.pseudo_id_values(fit)) && drop_single_column
  result <- .as_numeric_result(
    value,
    matrix_result = matrix_result || force_matrix,
    drop_single_column = FALSE,
    col.names = col.names
  )
  result <- .pseudo_apply_id_dimnames(result, fit, col.names = col.names)
  if (drop_single_column && is.matrix(result) && ncol(result) == 1L) {
    vector_result <- as.numeric(result[, 1L])
    names(vector_result) <- rownames(result)
    return(vector_result)
  }
  if (drop_single_column && !is.matrix(result)) {
    return(as.numeric(result))
  }
  result
}

.as_pseudo_data_frame <- function(value, fit, times = NULL) {
  pseudo_matrix <- .as_pseudo_matrix(value, fit, matrix_result = TRUE)
  if (!is.matrix(pseudo_matrix)) {
    pseudo_matrix <- matrix(as.numeric(pseudo_matrix), ncol = 1L)
  }
  time_values <- if (is.null(times)) {
    seq_len(ncol(pseudo_matrix))
  } else {
    as.numeric(times)
  }
  if (length(time_values) != ncol(pseudo_matrix)) {
    time_values <- seq_len(ncol(pseudo_matrix))
  }
  id_values <- .pseudo_id_row_names(fit, nrow(pseudo_matrix))
  id_name <- if (is.null(id_values)) "id" else "(id)"
  if (is.null(id_values)) {
    id_values <- seq_len(nrow(pseudo_matrix))
  }
  frame <- data.frame(
    rep(id_values, each = ncol(pseudo_matrix)),
    time = rep(time_values, times = nrow(pseudo_matrix)),
    pseudo = as.numeric(t(pseudo_matrix))
  )
  names(frame)[[1L]] <- id_name
  group_values <- .pseudo_group_values(fit)
  if (!is.null(group_values) && length(group_values) == nrow(pseudo_matrix)) {
    frame <- data.frame(
      strata = rep(group_values, each = ncol(pseudo_matrix)),
      frame,
      row.names = NULL
    )
  }
  frame
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

.model_term_names <- function(object, terms = NULL) {
  as.character(.call_r_api("model_term_names", object, terms = terms))
}

.predict_matrix_result <- function(type) {
  if (is.null(type)) {
    return(FALSE)
  }
  value <- tolower(gsub("-", "_", trimws(type)))
  startsWith("terms", value) || startsWith("quantile", value) || startsWith("uquantile", value)
}

.predict_column_names <- function(object, type, terms = NULL) {
  if (is.null(type)) {
    return(NULL)
  }
  value <- tolower(gsub("-", "_", trimws(type)))
  if (startsWith("terms", value)) {
    return(.model_term_names(object, terms = terms))
  }
  NULL
}

.attach_term_prediction_constant <- function(value, object, type, reference = NULL) {
  if (is.null(type) || !inherits(object, "survival_py_coxph")) {
    return(value)
  }
  type_key <- tolower(gsub("-", "_", trimws(type)))
  if (!startsWith("terms", type_key)) {
    return(value)
  }
  reference_key <- if (is.null(reference)) {
    "sample"
  } else {
    tolower(gsub("-", "_", trimws(as.character(reference)[[1L]])))
  }
  if (identical(reference_key, "strata")) {
    return(value)
  }
  if (identical(reference_key, "zero")) {
    constant <- 0
  } else {
    module <- .survival_python_module()
    if (!reticulate::py_has_attr(module, "predict_terms_constant")) {
      return(value)
    }
    constant <- as.numeric(.call_r_api("predict_terms_constant", object))[[1L]]
  }
  if (is.list(value) && all(c("fit", "se.fit") %in% names(value))) {
    attr(value$fit, "constant") <- constant
  } else {
    attr(value, "constant") <- constant
  }
  value
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

.residual_column_names <- function(object, key, terms = NULL) {
  if (inherits(object, "survival_py_coxph") && identical(key, "partial")) {
    return(.model_term_names(object, terms = terms))
  }
  if (inherits(object, "survival_py_coxph") &&
      key %in% c("score", "schoenfeld", "scaledsch")) {
    return(as.character(.call_r_api("coef_names", object)))
  }
  if (inherits(object, "survival_py_survreg") && identical(key, "matrix")) {
    return(c("g", "dg", "ddg", "ds", "dds", "dsg"))
  }
  NULL
}

.as_residual_result <- function(object, value, type, terms = NULL) {
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
    col.names = .residual_column_names(object, key, terms = terms)
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

.core_nsk_basis <- function(x, df, knots, intercept, boundary_knots) {
  constructor <- .core_attr("NaturalSplineKnot")
  boundary_knots <- do.call(reticulate::tuple, as.list(as.numeric(boundary_knots)))
  spline <- do.call(
    constructor,
    .compact_null(list(
      knots = knots,
      boundary_knots = boundary_knots,
      df = df,
      intercept = intercept
    ))
  )
  do.call(reticulate::py_get_attr(spline, "basis"), list(x))
}

survival_python_config <- function() {
  reticulate::py_config()
}

nsk <- function(x, df = NULL, knots = NULL, intercept = FALSE, b = 0.05,
                Boundary.knots = stats::quantile(x, c(b, 1 - b), na.rm = TRUE)) {
  nx <- names(x)
  x <- as.numeric(as.vector(x))
  na_x <- is.na(x)
  x_fit <- x[!na_x]
  if (length(x_fit) == 0L) {
    stop("x must contain at least one non-missing value", call. = FALSE)
  }
  if (any(!is.finite(x_fit))) {
    stop("x must contain only finite values", call. = FALSE)
  }

  intercept <- .as_logical_scalar(intercept, "intercept")
  if (!is.null(df)) {
    df <- .as_integer_scalar(df, "df", positive = TRUE)
  }
  if (!is.null(knots)) {
    knots <- as.numeric(knots)
    if (any(!is.finite(knots))) {
      stop("non-finite knots", call. = FALSE)
    }
    knots <- sort(unique(knots))
  }

  if (is.logical(Boundary.knots)) {
    if (length(Boundary.knots) != 1L || is.na(Boundary.knots)) {
      stop("wrong length for Boundary.knots", call. = FALSE)
    }
    Boundary.knots <- if (Boundary.knots) range(x_fit) else NULL
  }

  boundary_names <- names(Boundary.knots)
  boundary_knots <- NULL
  core_knots <- knots

  if (length(Boundary.knots) == 0L) {
    if (is.null(knots) || length(knots) < 2L) {
      stop("wrong length for Boundary.knots", call. = FALSE)
    }
    all_knots <- sort(unique(knots))
    boundary_knots <- all_knots[c(1L, length(all_knots))]
    core_knots <- if (length(all_knots) > 2L) {
      all_knots[-c(1L, length(all_knots))]
    } else {
      numeric()
    }
  } else if (length(Boundary.knots) == 2L) {
    boundary_knots <- sort(as.numeric(Boundary.knots))
    if (any(!is.finite(boundary_knots)) || boundary_knots[[1L]] >= boundary_knots[[2L]]) {
      stop("Boundary.knots must be finite and strictly increasing", call. = FALSE)
    }
    if (!is.null(knots) && length(knots) > 0L) {
      kept_boundary <- boundary_knots
      if (kept_boundary[[2L]] <= max(knots)) {
        kept_boundary <- kept_boundary[[1L]]
      }
      if (length(kept_boundary) > 0L && kept_boundary[[1L]] >= min(knots)) {
        kept_boundary <- kept_boundary[-1L]
      }
      all_knots <- sort(unique(c(knots, kept_boundary)))
      if (length(all_knots) < 2L) {
        stop("at least two distinct finite knots are required", call. = FALSE)
      }
      boundary_knots <- all_knots[c(1L, length(all_knots))]
      core_knots <- if (length(all_knots) > 2L) {
        all_knots[-c(1L, length(all_knots))]
      } else {
        numeric()
      }
    }
  } else {
    stop("wrong length for Boundary.knots", call. = FALSE)
  }

  result <- .core_nsk_basis(
    x_fit,
    df = df,
    knots = core_knots,
    intercept = intercept,
    boundary_knots = boundary_knots
  )
  n_cols <- as.integer(.result_field(result, "n_cols"))
  basis_values <- as.numeric(.result_field(result, "basis"))
  fit_matrix <- matrix(basis_values, nrow = length(x_fit), ncol = n_cols, byrow = TRUE)
  out <- matrix(NA_real_, nrow = length(x), ncol = n_cols)
  out[!na_x, ] <- fit_matrix
  dimnames(out) <- list(nx, as.character(seq_len(n_cols)))

  boundary_attr <- as.numeric(.result_field(result, "boundary_knots"))
  if (
    length(boundary_names) == 2L &&
      isTRUE(all.equal(unname(boundary_attr), unname(as.numeric(Boundary.knots))))
  ) {
    names(boundary_attr) <- boundary_names
  }

  attr(out, "degree") <- 3L
  attr(out, "knots") <- as.numeric(.result_field(result, "knots"))
  attr(out, "Boundary.knots") <- boundary_attr
  attr(out, "intercept") <- intercept
  class(out) <- c("nsk", "ns", "basis", "matrix")
  out
}

ridge <- function(..., theta, df = nvar / 2, eps = 0.1, scale = TRUE) {
  x <- cbind(...)
  nvar <- ncol(x)
  xname <- as.character(parse(text = substitute(cbind(...))))[-1L]
  vars <- apply(x, 2L, function(z) stats::var(z[!is.na(z)]))
  class(x) <- "coxph.penalty"

  if (!missing(theta) && !missing(df)) {
    stop("Only one of df or theta can be specified", call. = FALSE)
  }

  if (scale) {
    pfun <- function(coef, theta, ndead, scale) {
      list(
        penalty = sum(coef^2 * scale) * theta / 2,
        first = theta * coef * scale,
        second = theta * scale,
        flag = FALSE
      )
    }
  } else {
    pfun <- function(coef, theta, ndead, scale) {
      list(
        penalty = sum(coef^2) * theta / 2,
        first = theta * coef,
        second = theta,
        flag = FALSE
      )
    }
  }

  if (!missing(theta)) {
    temp <- list(
      pfun = pfun,
      diag = TRUE,
      cfun = function(parms, iter, history) list(theta = parms$theta, done = TRUE),
      cparm = list(theta = theta),
      pparm = vars,
      varname = paste("ridge(", xname, ")", sep = "")
    )
  } else {
    temp <- list(
      pfun = pfun,
      diag = TRUE,
      cfun = get("frailty.controldf", envir = asNamespace("survival")),
      cargs = "df",
      cparm = list(df = df, eps = eps, thetas = 0, dfs = nvar, guess = 1),
      pparm = vars,
      varname = paste("ridge(", xname, ")", sep = "")
    )
  }

  attributes(x) <- c(attributes(x), temp)
  x
}

.frailty_encoded_x <- function(x, sparse) {
  levels <- levels(factor(x))
  result <- .call_r_api(
    "_frailty_encoding",
    x = .as_python_vector(x),
    levels = levels,
    sparse = sparse
  )
  sparse <- as.logical(.result_field(result, "sparse"))
  codes <- as.integer(.as_nullable_numeric_vector(.result_field(result, "codes")))
  levels <- as.character(.result_field(result, "levels"))

  if (sparse) {
    out <- as.numeric(codes)
    out[is.na(codes)] <- NA_real_
    names(out) <- names(x)
    class(out) <- c("coxph.penalty", "numeric")
    return(out)
  }

  values <- factor(
    ifelse(is.na(codes), NA_character_, levels[codes]),
    levels = levels
  )
  names(values) <- names(x)
  contrasts <- diag(length(levels))
  dimnames(contrasts) <- list(seq_len(length(levels)), seq_len(length(levels)))
  attr(values, "contrasts") <- contrasts
  class(values) <- c("coxph.penalty", "factor")
  values
}

.frailty_printfun <- function(include_loglik = FALSE) {
  force(include_loglik)
  function(coef, var, var2, df, history) {
    if (!is.null(history$history)) {
      theta <- history$history[nrow(history$history), 1]
    } else {
      theta <- history$theta
    }
    if (is.matrix(var)) {
      test <- coxph.wtest(var, coef)$test
    } else {
      test <- sum(coef^2 / var)
    }
    df2 <- max(df, 0.5)
    coef_row <- c(NA, NA, NA, test, df, stats::pchisq(test, df2, lower.tail = FALSE))
    history_text <- paste("Variance of random effect=", format(theta))
    if (include_loglik) {
      history_text <- paste(
        history_text,
        "  I-likelihood =",
        format(round(history$c.loglik, 1), digits = 10)
      )
    }
    list(coef = coef_row, history = history_text)
  }
}

.frailty_fixed_cfun <- function(parms, iter, old) {
  list(theta = parms$theta, done = TRUE)
}

.frailty_gamma_adjust_cfun <- function(control_name, needs_df = FALSE) {
  force(control_name)
  force(needs_df)
  control <- get(control_name, envir = asNamespace("survival"))
  add_correction <- function(temp, iter, old, group, status, loglik) {
    if (iter > 0) {
      if (old$theta == 0) {
        correct <- 0
      } else {
        if (is.matrix(group)) {
          group <- c(group %*% seq_len(ncol(group)))
        }
        deaths <- tapply(status, group, sum)
        correct <- get("frailty.gammacon", envir = asNamespace("survival"))(deaths, 1 / old$theta)
      }
      temp$c.loglik <- loglik + correct
    }
    temp
  }
  if (needs_df) {
    return(function(opt, iter, old, df, group, status, loglik) {
      temp <- control(opt, iter, old, df)
      add_correction(temp, iter, old, group, status, loglik)
    })
  }
  function(opt, iter, old, group, status, loglik, ...) {
    temp <- control(opt, iter, old, ...)
    add_correction(temp, iter, old, group, status, loglik)
  }
}

frailty <- function(x, distribution = "gamma", ...) {
  dlist <- c("gamma", "gaussian", "t")
  index <- pmatch(distribution, dlist)
  if (!is.na(index)) {
    distribution <- dlist[[index]]
  }
  name <- paste("frailty", distribution, sep = ".")
  if (!exists(name, mode = "function")) {
    stop(paste("Function '", name, "' not found", sep = ""), call. = FALSE)
  }
  get(name, mode = "function")(x, ...)
}

frailty.gamma <- function(x, sparse = (nclass > 5), theta, df, eps = 1e-05,
                          method = c("em", "aic", "df", "fixed"), ...) {
  nclass <- length(unique(x[!is.na(x)]))
  sparse <- .as_logical_scalar(sparse, "sparse")
  x <- .frailty_encoded_x(x, sparse)

  if (missing(method)) {
    if (!missing(theta)) {
      method <- "fixed"
      if (!missing(df)) {
        stop("Cannot give both a df and theta argument", call. = FALSE)
      }
    } else if (!missing(df)) {
      method <- "df"
    }
  }
  method <- match.arg(method)
  if (method == "df" && missing(df)) {
    stop("Method = df but no df argument", call. = FALSE)
  }
  if (method == "fixed" && missing(theta)) {
    stop("Method= fixed but no theta argument", call. = FALSE)
  }
  if (method != "df" && !missing(df)) {
    stop("Method is not df, but have a df argument", call. = FALSE)
  }
  if (method != "fixed" && !missing(theta)) {
    stop("Method is not 'fixed', but have a theta argument", call. = FALSE)
  }

  pfun <- function(coef, theta, ndeath) {
    if (theta == 0) {
      list(recenter = 0, penalty = 0, flag = TRUE)
    } else {
      recenter <- log(mean(exp(coef)))
      coef <- coef - recenter
      nu <- 1 / theta
      list(
        recenter = recenter,
        first = (exp(coef) - 1) * nu,
        second = exp(coef) * nu,
        penalty = -sum(coef) * nu,
        flag = FALSE
      )
    }
  }
  dots <- list(...)
  printfun <- .frailty_printfun(include_loglik = TRUE)

  if (method == "fixed") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("x", "status", "loglik"),
      cfun = get("frailty.controlgam", envir = asNamespace("survival")),
      cparm = c(list(theta = theta), dots)
    )
  } else if (method == "em") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("x", "status", "loglik"),
      cfun = get("frailty.controlgam", envir = asNamespace("survival")),
      cparm = c(list(eps = eps), dots)
    )
  } else if (method == "aic") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("x", "status", "loglik", "neff", "df", "plik"),
      cparm = c(list(eps = eps, lower = 0, init = c(0.1, 1)), dots),
      cfun = .frailty_gamma_adjust_cfun("frailty.controlaic")
    )
  } else {
    if (missing(eps)) {
      eps <- 0.1
    }
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("df", "x", "status", "loglik"),
      cparm = c(list(df = df, thetas = 0, dfs = 0, eps = eps, guess = 3 * df / length(unclass(x))), dots),
      cfun = .frailty_gamma_adjust_cfun("frailty.controldf", needs_df = TRUE)
    )
  }
  if (!sparse) {
    temp <- c(temp, list(varname = paste("gamma", levels(x), sep = ":")))
  }
  attributes(x) <- c(attributes(x), temp)
  x
}

frailty.gaussian <- function(x, sparse = (nclass > 5), theta, df,
                             method = c("reml", "aic", "df", "fixed"),
                             ...) {
  if (missing(method)) {
    if (!missing(theta)) {
      method <- "fixed"
      if (!missing(df)) {
        stop("Cannot give both a df and theta argument", call. = FALSE)
      }
    } else if (!missing(df)) {
      if (df == 0) {
        method <- "aic"
      } else {
        method <- "df"
      }
    }
  }
  method <- match.arg(method)
  if (method == "df" && missing(df)) {
    stop("Method = df but no df argument", call. = FALSE)
  }
  if (method == "fixed" && missing(theta)) {
    stop("Method= fixed but no theta argument", call. = FALSE)
  }
  if (method != "fixed" && !missing(theta)) {
    stop("Method is not 'fixed', but have a theta argument", call. = FALSE)
  }
  nclass <- length(unique(x[!is.na(x)]))
  sparse <- .as_logical_scalar(sparse, "sparse")
  x <- .frailty_encoded_x(x, sparse)
  if (!missing(theta) && !missing(df)) {
    stop("Cannot give both a df and theta argument", call. = FALSE)
  }

  pfun <- function(coef, theta, ndead) {
    if (theta == 0) {
      list(recenter = 0, penalty = 0, flag = TRUE)
    } else {
      recenter <- mean(coef)
      coef <- coef - recenter
      list(
        recenter = recenter,
        first = coef / theta,
        second = rep(1, length(coef)) / theta,
        penalty = 0.5 * sum(coef^2 / theta + log(2 * pi * theta)),
        flag = FALSE
      )
    }
  }
  dots <- list(...)
  printfun <- .frailty_printfun()

  if (method == "reml") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("coef", "trH", "loglik"),
      cfun = get("frailty.controlgauss", envir = asNamespace("survival")),
      cparm = dots
    )
  } else if (method == "fixed") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cfun = .frailty_fixed_cfun,
      cparm = c(list(theta = theta), dots)
    )
  } else if (method == "aic") {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("neff", "df", "plik"),
      cparm = c(list(lower = 0, init = c(0.1, 1)), dots),
      cfun = get("frailty.controlaic", envir = asNamespace("survival"))
    )
  } else {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = "df",
      cparm = c(list(df = df, thetas = 0, dfs = 0, guess = 3 * df / length(unclass(x))), dots),
      cfun = get("frailty.controldf", envir = asNamespace("survival"))
    )
  }
  if (!sparse) {
    temp <- c(temp, list(varname = paste("gauss", levels(x), sep = ":")))
  }
  attributes(x) <- c(attributes(x), temp)
  x
}

frailty.t <- function(x, sparse = (nclass > 5), theta, df, eps = 1e-05,
                      tdf = 5, method = c("aic", "df", "fixed"), ...) {
  nclass <- length(unique(x[!is.na(x)]))
  sparse <- .as_logical_scalar(sparse, "sparse")
  x <- .frailty_encoded_x(x, sparse)
  if (tdf <= 2) {
    stop("Cannot have df <3 for the t-frailty", call. = FALSE)
  }
  if (missing(method)) {
    if (!missing(theta)) {
      method <- "fixed"
      if (!missing(df)) {
        stop("Cannot give both a df and theta argument", call. = FALSE)
      }
    } else if (!missing(df)) {
      if (df == 0) {
        method <- "aic"
      } else {
        method <- "df"
      }
    }
  }
  method <- match.arg(method)
  if (method == "df" && missing(df)) {
    stop("Method = df but no df argument", call. = FALSE)
  }
  if (method == "fixed" && missing(theta)) {
    stop("Method= fixed but no theta argument", call. = FALSE)
  }
  if (method != "fixed" && !missing(theta)) {
    stop("Method is not 'fixed', but have a theta argument", call. = FALSE)
  }

  pfun <- function(coef, theta, ndead, tdf) {
    if (theta == 0) {
      list(recenter = 0, penalty = 0, flag = TRUE)
    } else {
      sig <- theta * (tdf - 2) / tdf
      temp <- 1 + coef^2 / (tdf * sig)
      temp1 <- coef / temp
      temp2 <- 1 / temp - (2 / (tdf * sig)) * coef^2 / temp^2
      recenter <- sum(temp1) / sum(temp2)
      coef <- coef - recenter
      const <- (tdf + 1) / (tdf * sig)
      temp <- 1 + coef^2 / (tdf * sig)
      list(
        recenter = recenter,
        first = const * coef / temp,
        second = const * (1 / temp - (2 / (tdf * sig)) * coef^2 / temp^2),
        penalty = sum(0.5 * log(pi * tdf * sig) + ((tdf + 1) / 2) * log(temp) + lgamma(tdf / 2) - lgamma((tdf + 1) / 2)),
        flag = FALSE
      )
    }
  }
  dots <- list(...)
  printfun <- .frailty_printfun()

  if (method == "fixed") {
    temp <- list(
      pfun = pfun,
      pparm = tdf,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cfun = .frailty_fixed_cfun,
      cparm = c(list(theta = theta), dots)
    )
  } else if (method == "aic") {
    temp <- list(
      pfun = pfun,
      pparm = tdf,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = c("neff", "df", "plik"),
      cparm = c(list(lower = 0, init = c(0.1, 1), eps = eps), dots),
      cfun = get("frailty.controlaic", envir = asNamespace("survival"))
    )
  } else {
    if (missing(eps)) {
      eps <- 0.1
    }
    temp <- list(
      pfun = pfun,
      pparm = tdf,
      printfun = printfun,
      diag = TRUE,
      sparse = sparse,
      cargs = "df",
      cparm = c(list(df = df, eps = eps, thetas = 0, dfs = 0, guess = 3 * df / length(unclass(x))), dots),
      cfun = get("frailty.controldf", envir = asNamespace("survival"))
    )
  }
  if (!sparse) {
    temp <- c(temp, list(varname = paste("t", levels(x), sep = ":")))
  }
  attributes(x) <- c(attributes(x), temp)
  x
}

pspline <- function(x, df = 4, theta, nterm = 2.5 * df, degree = 3,
                    eps = 0.1, method, Boundary.knots = range(x),
                    intercept = FALSE, penalty = TRUE, combine, ...) {
  xname <- deparse(substitute(x))
  x <- as.numeric(x)
  intercept <- .as_logical_scalar(intercept, "intercept")
  penalty <- .as_logical_scalar(penalty, "penalty")
  dots <- list(...)

  result <- .call_r_api(
    "pspline",
    x = .as_python_vector(x),
    df = df,
    theta = if (missing(theta)) NULL else theta,
    nterm = nterm,
    degree = degree,
    eps = if (missing(eps)) NULL else eps,
    method = if (missing(method)) NULL else method,
    boundary_knots = Boundary.knots,
    intercept = intercept,
    penalty = penalty,
    combine = if (missing(combine)) NULL else combine
  )

  out <- .as_numeric_matrix(.result_field(result, "basis"))
  out[is.nan(out)] <- NA_real_
  nvar <- ncol(out)
  nterm <- as.integer(.as_numeric_vector(.result_field(result, "nterm"))[[1L]])
  degree <- as.integer(.as_numeric_vector(.result_field(result, "degree"))[[1L]])
  df <- .as_numeric_vector(.result_field(result, "df"))[[1L]]
  eps <- .as_numeric_vector(.result_field(result, "eps"))[[1L]]
  method <- as.character(.result_field(result, "method"))
  boundary_knots <- .as_numeric_vector(.result_field(result, "boundary_knots"))
  dmat <- .as_numeric_matrix(.result_field(result, "dmat"))
  cbase <- .as_numeric_vector(.result_field(result, "cbase"))

  if (!penalty) {
    attributes(out) <- c(
      attributes(out),
      list(intercept = intercept, nterm = nterm, degree = degree, Boundary.knots = boundary_knots)
    )
    combine_result <- .result_field(result, "combine")
    if (!is.null(combine_result)) {
      attr(out, "combine") <- as.integer(.as_numeric_vector(combine_result))
    }
    class(out) <- "pspline"
    return(out)
  }

  pfun <- function(coef, theta, n, dmat) {
    if (theta >= 1) {
      list(penalty = 100 * (1 - theta), flag = TRUE)
    } else {
      if (theta <= 0) {
        lambda <- 0
      } else {
        lambda <- theta / (1 - theta)
      }
      list(
        penalty = c(coef %*% dmat %*% coef) * lambda / 2,
        first = c(dmat %*% coef) * lambda,
        second = c(dmat * lambda),
        flag = FALSE
      )
    }
  }
  printfun <- function(coef, var, var2, df, history, cbase) {
    test1 <- coxph.wtest(var, coef)$test
    xmat <- cbind(1, cbase)
    xsig <- coxph.wtest(var, xmat)$solve
    cmat <- coxph.wtest(t(xmat) %*% xsig, t(xsig))$solve[2, ]
    linear <- sum(cmat * coef)
    lvar1 <- c(cmat %*% var %*% cmat)
    lvar2 <- c(cmat %*% var2 %*% cmat)
    test2 <- linear^2 / lvar1
    cmat <- rbind(
      c(linear, sqrt(lvar1), sqrt(lvar2), test2, 1, stats::pchisq(test2, 1, lower.tail = FALSE)),
      c(NA, NA, NA, test1 - test2, df - 1, stats::pchisq(test1 - test2, max(0.5, df - 1), lower.tail = FALSE))
    )
    dimnames(cmat) <- list(c("linear", "nonlin"), NULL)
    nn <- nrow(history$thetas)
    if (length(nn)) {
      theta <- history$thetas[nn, 1]
    } else {
      theta <- history$theta
    }
    list(coef = cmat, history = paste("Theta=", format(theta)))
  }
  temp <- formals(printfun)
  temp$cbase <- cbase
  formals(printfun) <- temp
  environment(printfun) <- .GlobalEnv

  if (identical(method, "fixed")) {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      pparm = dmat,
      diag = FALSE,
      cparm = list(theta = .as_numeric_vector(.result_field(result, "theta"))[[1L]]),
      varname = paste("ps(", xname, ")", if (intercept) seq_len(nvar) else 1L + 2L:(nvar + 1L), sep = ""),
      cfun = function(parms, iter, old) list(theta = parms$theta, done = TRUE)
    )
  } else if (identical(method, "df")) {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      diag = FALSE,
      cargs = "df",
      cparm = c(list(df = df, eps = eps, thetas = c(1, 0), dfs = c(1, nterm), guess = 1 - df / nterm), dots),
      pparm = dmat,
      varname = paste("ps(", xname, ")", if (intercept) seq_len(nvar) else 1L + 2L:(nvar + 1L), sep = ""),
      cfun = get("frailty.controldf", envir = asNamespace("survival"))
    )
  } else {
    temp <- list(
      pfun = pfun,
      printfun = printfun,
      pparm = dmat,
      diag = FALSE,
      cargs = c("neff", "df", "plik"),
      cparm = c(list(eps = eps, init = c(0.5, 0.95), lower = 0, upper = 1), dots),
      varname = paste("ps(", xname, ")", if (intercept) seq_len(nvar) else 1L + 2L:(nvar + 1L), sep = ""),
      cfun = get("frailty.controlaic", envir = asNamespace("survival"))
    )
  }

  attributes(out) <- c(
    attributes(out),
    temp,
    list(intercept = intercept, nterm = nterm, degree = degree, df = df, Boundary.knots = boundary_knots)
  )
  combine_result <- .result_field(result, "combine")
  if (!is.null(combine_result)) {
    attr(out, "combine") <- as.integer(.as_numeric_vector(combine_result))
  }
  class(out) <- c("pspline", "coxph.penalty")
  out
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

.surv_factor_response <- function(args, type = NULL, origin = 0) {
  if (!(length(args) %in% c(2L, 3L)) || !is.factor(args[[length(args)]])) {
    return(NULL)
  }
  matched_type <- if (is.null(type)) {
    NULL
  } else {
    match.arg(type, c("right", "left", "interval", "counting", "interval2"))
  }
  if (length(args) == 2L && identical(matched_type, "interval2")) {
    return(NULL)
  }
  if (length(args) == 3L && identical(matched_type, "interval")) {
    return(NULL)
  }

  event <- args[[length(args)]]
  status <- as.integer(event) - 1L
  states <- levels(event)[-1L]
  input_attributes <- list(event = attributes(event))
  if (length(args) == 2L) {
    out <- cbind(
      time = as.numeric(args[[1L]]) - origin,
      status = status
    )
    attr(out, "type") <- "mright"
  } else {
    out <- cbind(
      start = as.numeric(args[[1L]]) - origin,
      stop = as.numeric(args[[2L]]) - origin,
      status = status
    )
    attr(out, "type") <- "mcounting"
  }
  attr(out, "states") <- states
  attr(out, "inputAttributes") <- input_attributes
  class(out) <- "Surv"
  out
}

.native_model_frame_surv <- function(time, time2, event, type = NULL,
                                     origin = 0, time1, start, stop, status) {
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
  factor_response <- .surv_factor_response(args, type = type, origin = origin)
  if (!is.null(factor_response)) {
    return(factor_response)
  }
  args <- c(args, .compact_null(list(type = type, origin = origin)))
  .survsplit_minimal_surv(args)
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
  factor_response <- .surv_factor_response(args, type = type, origin = origin)
  if (!is.null(factor_response)) {
    return(factor_response)
  }
  args <- c(args, .compact_null(list(type = type, origin = origin)))
  .wrap_python(do.call(.python_attr("Surv"), args), c("survival_py_surv", "survival_py_object"))
}

Surv2 <- function(time, event, repeated = FALSE) {
  if (missing(time)) {
    stop("must have a time argument", call. = FALSE)
  }
  if (inherits(time, "difftime")) {
    time <- unclass(time)
  }
  if (!is.numeric(time)) {
    stop("Time variable is not numeric", call. = FALSE)
  }
  if (missing(event)) {
    stop("must have an event argument", call. = FALSE)
  }
  if (!is.logical(repeated) || length(repeated) != 1L || is.na(repeated)) {
    stop("invalid value for repeated option", call. = FALSE)
  }
  time_values <- .as_python_vector(as.numeric(time))
  if (!is.list(time_values)) {
    time_values <- as.list(time_values)
  }
  event_values <- .as_python_vector(event)
  if (!is.list(event_values)) {
    event_values <- as.list(event_values)
  }
  result <- .call_r_api(
    "Surv2",
    time_values,
    event_values,
    repeated = repeated
  )
  status <- as.integer(.as_nullable_numeric_vector(.result_field(result, "status")))
  out <- cbind(time = as.numeric(time), status = status)
  attr(out, "states") <- as.character(.result_field(result, "states"))
  attr(out, "repeated") <- isTRUE(.result_field(result, "repeated"))
  class(out) <- "Surv2"
  out
}

is.Surv <- function(value) {
  if (inherits(value, "Surv")) {
    return(TRUE)
  }
  .call_r_api("is_surv", value)
}

is.ratetable <- function(x, verbose = FALSE) {
  datecheck <- function(value) {
    inherits(value, c("Date", "POSIXt", "date", "chron"))
  }
  att <- attributes(x)
  required <- c("dim", "dimnames", "cutpoints")
  is_date <- vapply(att$cutpoints, datecheck, logical(1))
  dimid <- names(att$dimnames)
  if (is.null(dimid)) {
    dimid <- att$dimid
  }

  if (!verbose) {
    if (!inherits(x, "ratetable")) {
      return(FALSE)
    }
    if (any(is.na(match(required, names(att))))) {
      return(FALSE)
    }
    if (is.null(att$dimid)) {
      att$dimid <- names(att$dimnames)
    }
    nd <- length(att$dim)
    if (length(x) != prod(att$dim)) {
      return(FALSE)
    }
    if (is.null(dimid)) {
      return(FALSE)
    }
    if (any(is.na(dimid) | dimid == "")) {
      return(FALSE)
    }
    if (!(is.list(att$dimnames) && is.list(att$cutpoints))) {
      return(FALSE)
    }
    if (length(att$dimnames) != nd || length(att$cutpoints) != nd) {
      return(FALSE)
    }
    if (!is.null(att$type)) {
      if (length(att$type) != nd) {
        return(FALSE)
      }
      if (any(is.na(match(att$type, 1:4)))) {
        return(FALSE)
      }
      if (sum(att$type == 4) > 1) {
        return(FALSE)
      }
      if (any((att$type > 2) != is_date)) {
        return(FALSE)
      }
      fac <- ifelse(att$type == 1, 1, 0)
    } else if (!is.null(att$factor)) {
      fac <- as.numeric(att$factor)
      if (any(is.na(fac))) {
        return(FALSE)
      }
      if (any(fac < 0)) {
        return(FALSE)
      }
      if (length(att$factor) != nd) {
        return(FALSE)
      }
      if (sum(fac > 1) > 1) {
        return(FALSE)
      }
      if (any(fac == 1 & is_date)) {
        return(FALSE)
      }
    } else {
      return(FALSE)
    }
    if (length(att$dimid) != nd) {
      return(FALSE)
    }
    for (i in seq_len(nd)) {
      n <- att$dim[i]
      if (length(att$dimnames[[i]]) != n) {
        return(FALSE)
      }
      if (fac[i] != 1 && length(att$cutpoints[[i]]) != n) {
        return(FALSE)
      }
      if (fac[i] != 1 && any(order(att$cutpoints[[i]]) != seq_len(n))) {
        return(FALSE)
      }
      if (fac[i] == 1 && !is.null(att$cutpoints[[i]])) {
        return(FALSE)
      }
      if (fac[i] > 1 && i < nd) {
        return(FALSE)
      }
    }
    return(TRUE)
  }

  msg <- NULL
  if (!inherits(x, "ratetable")) {
    msg <- c(msg, "wrong class")
  }
  missing_required <- is.na(match(required, names(att)))
  if (any(missing_required)) {
    msg <- c(msg, paste("missing attribute:", required[missing_required]))
  }
  if (is.null(att$dimid)) {
    att$dimid <- names(att$dimnames)
  }
  nd <- length(att$dim)
  if (length(x) != prod(att$dim)) {
    msg <- c(msg, "length of the data does not match prod(dim)")
  }
  if (!is.list(att$dimnames)) {
    msg <- c(msg, "dimnames is not a list")
  }
  if (!is.list(att$cutpoints)) {
    msg <- c(msg, "cutpoints is not a list")
  }
  if (length(att$dimnames) != nd) {
    msg <- c(msg, "wrong length for dimnames")
  }
  if (length(att$dimid) != nd) {
    msg <- c(msg, "wrong length for dimid, or dimnames do not have names")
  }
  if (any(att$dimid == "")) {
    msg <- c(msg, "one of the dimnames identifiers is blank")
  }
  if (length(att$cutpoints) != nd) {
    msg <- c(msg, "wrong length for cutpoints")
  }
  if (!is.null(att$type)) {
    if (any(is.na(match(att$type, 1:4)))) {
      msg <- c(msg, "type attribute must be 1, 2, 3, or 4")
    }
    type <- att$type
    if (length(type) != nd) {
      msg <- c(msg, "wrong length for type attribute")
    } else {
      indx <- which(type > 2 & !is_date)
      if (length(indx) > 0) {
        msg <- c(msg, paste0("type[", indx, "] is 3 or 4 but the cutpoint is not one of the date types"))
      }
      indx <- which(type < 3 & is_date)
      if (length(indx) > 0) {
        msg <- c(msg, paste0("type[", indx, "] is numeric or factor but the cutpoint is a date"))
      }
    }
    if (sum(type == 4) > 1) {
      msg <- c(msg, "two dimenesions idenitied as US ratetable years")
    }
  } else if (!is.null(att$factor)) {
    fac <- as.numeric(att$factor)
    if (any(is.na(fac))) {
      msg <- c(msg, "illegal 'factor' attribute of NA")
    }
    if (any(fac < 0)) {
      msg <- c(msg, "illegal 'factor' attribute of <0")
    }
    if (length(att$factor) != nd) {
      msg <- c(msg, "wrong length for factor")
    }
    type <- 1 * (fac == 1) + 2 * (fac == 0) + 4 * (fac > 1)
    if (sum(fac > 1) > 1) {
      msg <- c(msg, "two dimenesions idenitied as US ratetable years")
    }
  } else {
    msg <- c(msg, "missing the 'type' attribute")
    type <- rep(NA_integer_, nd)
  }
  for (i in (1:nd)) {
    n <- att$dim[i]
    if (length(att$dimnames[[i]]) != n) {
      msg <- c(msg, paste("dimname", i, "is the wrong length"))
    }
    if (type[i] > 1) {
      if (length(att$cutpoints[[i]]) != n) {
        msg <- c(msg, paste("wrong length for cutpoints", i))
      } else if (any(order(att$cutpoints[[i]]) != seq_len(n))) {
        msg <- c(msg, paste("unsorted cutpoints for dimension", i))
      }
    }
    if (type[i] == 1 && !is.null(att$cutpoints[[i]])) {
      msg <- c(msg, paste0("attribute type[", i, "] is continuous; cutpoint should be null"))
    }
    if (!is.null(att$fac) && type[i] == 4 && i < nd) {
      msg <- c(msg, "only the last dimension can be interpolated")
    }
  }
  if (length(msg) == 0) {
    TRUE
  } else {
    msg
  }
}

ratetableDate <- function(x) {
  if (!(inherits(x, "Date") || inherits(x, "POSIXt"))) {
    return(x)
  }
  days <- as.numeric(as.Date(x))
  structure(days, class = "rtabledate")
}

ratetable <- function(...) {
  datecheck <- function(x) {
    inherits(x, c("Date", "POSIXt", "date", "chron"))
  }
  args <- list(...)
  nargs <- length(args)
  lengths <- vapply(args, length, integer(1))
  n <- max(lengths)
  levlist <- vector("list", nargs)
  x <- matrix(0, n, nargs)
  dimnames(x) <- list(seq_len(n), names(args))
  isDate <- vapply(args, datecheck, logical(1))

  for (i in seq_len(nargs)) {
    if (lengths[[i]] == 1L) {
      args[[i]] <- rep(args[[i]], n)
    } else if (lengths[[i]] != n) {
      stop(paste("Arguments do not all have the same length (arg ", i, ")", sep = ""))
    }
    if (inherits(args[[i]], "category") || is.character(args[[i]])) {
      args[[i]] <- as.factor(args[[i]])
    }
    if (is.factor(args[[i]])) {
      levlist[[i]] <- levels(args[[i]])
      x[, i] <- as.numeric(args[[i]])
    } else {
      x[, i] <- ratetableDate(args[[i]])
    }
  }
  attr(x, "isDate") <- isDate
  attr(x, "levlist") <- levlist
  class(x) <- "ratetable2"
  x
}

match.ratetable <- function(R, ratetable) {
  datecheck <- function(x) {
    inherits(x, c("Date", "POSIXt", "date", "chron", "rtabledate"))
  }
  if (!is.ratetable(ratetable)) {
    stop("Invalid rate table")
  }

  dimid <- names(dimnames(ratetable))
  if (is.null(dimid)) {
    dimid <- attr(ratetable, "dimid")
  }
  datecut <- vapply(attr(ratetable, "cutpoints"), datecheck, logical(1))
  rtype <- attr(ratetable, "type")
  if (is.null(rtype)) {
    temp <- attr(ratetable, "factor")
    rtype <- 1 * (temp == 1) + ifelse(datecut, 3, 2) * (temp == 0) + 4 * (temp > 1)
  }

  if (is.matrix(R)) {
    attR <- attributes(R)
    attributes(R) <- attR["dim"]
    Rnames <- attR$dimnames[[2L]]
    isDate <- attR[["isDate"]]
    levlist <- attR[["levlist"]]
  } else {
    Rnames <- names(R)
    levlist <- lapply(R, levels)
    isDate <- vapply(R, datecheck, logical(1))
  }

  ord <- match(dimid, Rnames)
  if (any(is.na(ord))) {
    stop(paste("Argument '", dimid[is.na(ord)], "' needed by the ratetable was not found in the data", sep = ""))
  }
  if (any(duplicated(ord))) {
    stop("A ratetable argument appears twice in the data")
  }
  R <- R[, ord, drop = FALSE]
  levlist <- levlist[ord]
  isDate <- isDate[ord]
  dtemp <- dimnames(ratetable)

  if (any((rtype < 3) & isDate)) {
    indx <- which(rtype < 3 & isDate)
    stop(paste(
      "Data has a date type variable, but the reference",
      "ratetable is not a date variable:",
      paste(dimid[indx], collapse = " ")
    ))
  }

  for (i in seq_len(ncol(R))) {
    if (rtype[[i]] > 2) {
      R[, i] <- ratetableDate(R[, i])
    }
    if (length(levlist[[i]]) > 0L) {
      if (rtype[[i]] != 1) {
        stop(paste("for this ratetable,", dimid[[i]], "must be a continuous variable"))
      }
      temp <- charmatch(casefold(levlist[[i]]), casefold(dtemp[[i]]))
      if (any(is.na(temp))) {
        stop(paste("Levels do not match for ratetable() variable", dimid[[i]]))
      }
      if (any(temp == 0)) {
        stop(paste("Non-unique ratetable match for variable", dimid[[i]]))
      }
      R[, i] <- temp[as.numeric(R[, i])]
    } else {
      R[, i] <- unclass(R[, i])
      if (rtype[[i]] == 1) {
        temp <- R[, i]
        if (any(floor(temp) != temp) || any(temp <= 0) || max(temp) > length(dtemp[[i]])) {
          stop(paste("The variable", dimid[[i]], "is out of range"))
        }
      }
    }
  }

  R <- as.matrix(R)
  summ <- attr(ratetable, "summary")
  cutpoints <- lapply(attr(ratetable, "cutpoints"), ratetableDate)
  if (is.null(summ)) {
    list(R = R, cutpoints = cutpoints)
  } else {
    list(R = R, cutpoints = cutpoints, summ = summ(R))
  }
}

cipoisson <- function(k, time = 1, p = 0.95, method = c("exact", "anscombe")) {
  method <- match.arg(method)
  nn <- max(length(k), length(time), length(p))
  result <- .call_r_api(
    "cipoisson",
    k = .as_python_vector(k),
    time = .as_python_vector(time),
    p = .as_python_vector(p),
    method = method
  )
  values <- .as_numeric_vector(result)
  values[is.nan(values)] <- NA_real_
  if (length(values) == 0L) {
    return(numeric())
  }
  if (nn == 1L) {
    out <- values[seq_len(2L)]
    names(out) <- c("lower", "upper")
    return(out)
  }
  out <- matrix(values, ncol = 2L, byrow = TRUE)
  colnames(out) <- c("lower", "upper")
  out
}

lvcf <- function(id, x, time) {
  result <- .call_r_api(
    "lvcf",
    id = .as_python_vector(id),
    x = .as_python_vector(x),
    time = if (missing(time)) NULL else .as_python_vector(time)
  )
  if (is.factor(x)) {
    return(factor(.as_nullable_character_vector(result), levels = levels(x)))
  }
  if (is.integer(x)) {
    return(as.integer(.as_nullable_numeric_vector(result)))
  }
  if (is.numeric(x)) {
    return(.as_nullable_numeric_vector(result))
  }
  if (is.logical(x)) {
    return(.as_nullable_logical_vector(result))
  }
  .as_nullable_character_vector(result)
}

nostutter <- function(id, x, censor = 0, single = FALSE) {
  if (!(is.character(x) || is.numeric(x) || is.factor(x))) {
    stop("invalid variable type", call. = FALSE)
  }
  x_levels <- if (is.factor(x)) levels(x) else levels(factor(x))
  censor_value <- if (is.factor(x) || is.character(x)) as.character(censor)[[1L]] else censor
  result <- .call_r_api(
    "nostutter",
    id = .as_python_vector(id),
    x = if (is.factor(x)) as.character(x) else .as_python_vector(x),
    censor = censor_value,
    single = isTRUE(single)
  )
  factor(
    .as_nullable_character_vector(result),
    levels = unique(c(as.character(censor), x_levels))
  )
}

.bounded_link <- function(name, family, display_name, edge) {
  out <- stats::make.link(family)
  out$linkfun <- function(mu) {
    result <- .as_numeric_vector(.call_r_api(name, .as_python_vector(mu), edge = edge))
    result[is.nan(result)] <- NA_real_
    result
  }
  out$name <- display_name
  out
}

blogit <- function(edge = 0.05) {
  .bounded_link("blogit", "logit", "blogit", edge)
}

bprobit <- function(edge = 0.05) {
  .bounded_link("bprobit", "probit", "probit", edge)
}

bcloglog <- function(edge = 0.05) {
  .bounded_link("bcloglog", "cloglog", "bcloglog", edge)
}

blog <- function(edge = 0.05) {
  .bounded_link("blog", "log", "blog", edge)
}

.result_field <- function(result, name) {
  if (is.list(result) && !is.null(result[[name]])) {
    return(result[[name]])
  }
  if (inherits(result, "python.builtin.object") && reticulate::py_has_attr(result, name)) {
    return(reticulate::py_to_r(reticulate::py_get_attr(result, name)))
  }
  NULL
}

survexp <- function(formula, data, weights, subset, na.action, rmap, times,
                    method = c(
                      "ederer", "hakulinen", "conditional", "individual.h",
                      "individual.s"
                    ),
                    cohort = TRUE, conditional = FALSE, ratetable = NULL,
                    scale = 1, se.fit, model = FALSE, x = FALSE, y = FALSE,
                    time, age, year, sex = NULL) {
  direct_time <- NULL
  if (!missing(time)) {
    direct_time <- time
  } else if (!missing(formula) && !inherits(formula, "formula")) {
    direct_time <- formula
  }
  if (is.null(direct_time)) {
    call <- match.call()
    call[[1L]] <- quote(survival::survexp)
    return(eval.parent(call))
  }
  if (missing(age) || missing(year)) {
    stop("direct survexp bridge requires time, age, and year vectors", call. = FALSE)
  }
  if (!is.null(ratetable) && inherits(ratetable, "ratetable")) {
    stop(
      "direct survexp bridge requires a Python RateTable; omit ratetable to use the bundled Python table",
      call. = FALSE
    )
  }
  result <- .call_r_api(
    "survexp",
    time = .as_python_vector(direct_time),
    age = .as_python_vector(age),
    year = .as_python_vector(year),
    ratetable = ratetable,
    sex = if (is.null(sex)) NULL else .as_python_vector(sex),
    times = if (missing(times)) NULL else .as_python_vector(times),
    method = if (missing(method)) NULL else match.arg(method),
    cohort = cohort,
    conditional = conditional,
    scale = scale,
    se_fit = if (missing(se.fit)) NULL else se.fit
  )
  surv <- .result_field(result, "surv")
  if (is.null(surv)) {
    return(.as_numeric_vector(result))
  }
  out <- list(
    call = match.call(),
    surv = .as_numeric_vector(surv),
    n.risk = .as_numeric_vector(.result_field(result, "n_risk")),
    time = .as_numeric_vector(.result_field(result, "time")),
    cumhaz = .as_numeric_vector(.result_field(result, "cumhaz")),
    method = as.character(.result_field(result, "method")),
    n = as.integer(.result_field(result, "n"))
  )
  class(out) <- c("survexp", "survfit")
  out
}

.pyears_group_values <- function(value) {
  if (is.factor(value)) {
    return(levels(value))
  }
  sort(unique(as.character(value)))
}

.pyears_row_grid_keys <- function(values, levels) {
  if (length(levels) == 0L) {
    return(character(nrow(values)))
  }
  codes <- Map(function(column, column_levels) {
    match(as.character(column), column_levels)
  }, values, levels)
  missing <- Reduce(`|`, lapply(codes, is.na))
  dims <- vapply(levels, length, integer(1))
  grid_keys <- as.character(seq_len(prod(dims)))
  row_keys <- as.character(apply(do.call(cbind, codes), 1L, function(code) {
    if (any(is.na(code))) {
      return(NA_integer_)
    }
    1L + sum((code - 1L) * c(1L, cumprod(dims[-length(dims)])))
  }))
  row_keys[missing] <- NA_character_
  attr(row_keys, "grid_keys") <- grid_keys
  row_keys
}

.pyears_formula_group_info <- function(term_labels, data) {
  if (length(term_labels) == 0L) {
    return(NULL)
  }
  values <- data[term_labels]
  levels <- lapply(values, .pyears_group_values)
  names(levels) <- term_labels
  grid <- do.call(
    expand.grid,
    c(levels, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE))
  )
  row_keys <- .pyears_row_grid_keys(values, levels)
  list(
    names = term_labels,
    levels = levels,
    factor_info = lapply(values, function(value) {
      if (!is.factor(value)) {
        return(NULL)
      }
      list(levels = levels(value), ordered = is.ordered(value))
    }),
    class_info = lapply(values, function(value) {
      if (inherits(value, "Date")) {
        return(list(class = "Date"))
      }
      if (inherits(value, "POSIXct")) {
        timezone <- attr(value, "tzone")
        if (is.null(timezone) || length(timezone) == 0L) {
          timezone <- ""
        }
        return(list(class = "POSIXct", timezone = timezone[[1L]]))
      }
      NULL
    }),
    out_attrs_levels = Map(function(value, column_levels) {
      if (!is.factor(value)) {
        if (inherits(value, "POSIXct")) {
          timezone <- attr(value, "tzone")
          if (is.null(timezone) || length(timezone) == 0L) {
            timezone <- ""
          }
          seconds <- suppressWarnings(as.numeric(column_levels))
          return(as.character(as.POSIXct(seconds, origin = "1970-01-01", tz = timezone[[1L]])))
        }
        return(column_levels)
      }
      as.character(factor(column_levels, levels = levels(factor(value))))
    }, values, levels),
    keys = attr(row_keys, "grid_keys"),
    row_keys = unname(row_keys),
    grid = grid
  )
}

.pyears_restore_group_classes <- function(frame, group_info) {
  if (is.null(group_info) || is.null(group_info$factor_info)) {
    return(frame)
  }
  for (column in names(group_info$class_info)) {
    info <- group_info$class_info[[column]]
    if (is.null(info) || !(column %in% names(frame))) {
      next
    }
    if (identical(info$class, "Date")) {
      frame[[column]] <- as.Date(as.character(frame[[column]]))
    } else if (identical(info$class, "POSIXct")) {
      seconds <- suppressWarnings(as.numeric(as.character(frame[[column]])))
      frame[[column]] <- as.POSIXct(seconds, origin = "1970-01-01", tz = info$timezone)
    }
  }
  for (column in names(group_info$factor_info)) {
    info <- group_info$factor_info[[column]]
    if (is.null(info) || !(column %in% names(frame))) {
      next
    }
    frame[[column]] <- factor(
      as.character(frame[[column]]),
      levels = info$levels,
      ordered = isTRUE(info$ordered)
    )
  }
  frame
}

.pyears_fill_grid <- function(values, groups, group_info) {
  filled <- rep(0, length(group_info$keys))
  positions <- match(groups, group_info$keys)
  keep <- !is.na(positions)
  filled[positions[keep]] <- values[keep]
  structure(
    filled,
    dim = unname(vapply(group_info$levels, length, integer(1))),
    dimnames = group_info$levels
  )
}

.as_pyears_result <- function(result, call, data.frame = FALSE, terms = NULL,
                              group_name = NULL, group_info = NULL) {
  groups <- as.character(.result_field(result, "group"))
  pyears_values <- .as_numeric_vector(.result_field(result, "pyears"))
  n_values <- .as_numeric_vector(.result_field(result, "n"))
  grouped <- length(groups) == length(pyears_values) &&
    !(length(groups) == 1L && groups[[1L]] == "(all)")
  if (!is.null(group_info)) {
    pyears_values <- .pyears_fill_grid(pyears_values, groups, group_info)
    n_values <- .pyears_fill_grid(n_values, groups, group_info)
  } else if (grouped) {
    if (!is.null(group_name)) {
      dim_values <- length(pyears_values)
      dim_names <- stats::setNames(list(groups), group_name)
      pyears_values <- structure(pyears_values, dim = dim_values, dimnames = dim_names)
      n_values <- structure(n_values, dim = dim_values, dimnames = dim_names)
    } else {
      names(pyears_values) <- groups
      names(n_values) <- groups
    }
  }
  event_values <- .result_field(result, "event")
  expected_values <- .result_field(result, "expected")
  if (!is.null(group_info) && !is.null(event_values)) {
    event_values <- .pyears_fill_grid(.as_numeric_vector(event_values), groups, group_info)
  }
  if (!is.null(group_info) && !is.null(expected_values)) {
    expected_values <- .pyears_fill_grid(.as_numeric_vector(expected_values), groups, group_info)
  }
  if (isTRUE(data.frame)) {
    if (!is.null(group_info)) {
      keep <- as.numeric(pyears_values) != 0 | as.numeric(n_values) != 0
      if (!is.null(expected_values)) {
        keep <- keep | as.numeric(expected_values) != 0
      }
      if (!is.null(event_values)) {
        keep <- keep | as.numeric(event_values) != 0
      }
      frame <- group_info$grid[keep, , drop = FALSE]
      frame$pyears <- as.numeric(pyears_values)[keep]
      frame$n <- as.numeric(n_values)[keep]
      rownames(frame) <- NULL
      attr(frame, "out.attrs") <- list(
        dim = unname(vapply(group_info$levels, length, integer(1))),
        dimnames = stats::setNames(
          lapply(seq_along(group_info$levels), function(idx) {
            paste0("Var", idx, "=", group_info$out_attrs_levels[[idx]])
          }),
          paste0("Var", seq_along(group_info$levels))
        )
      )
      frame <- .pyears_restore_group_classes(frame, group_info)
    } else {
      frame <- data.frame(
        group = groups,
        pyears = unname(as.numeric(pyears_values)),
        n = unname(as.numeric(n_values)),
        stringsAsFactors = FALSE
      )
    }
    if (!is.null(expected_values)) {
      frame$expected <- if (is.null(group_info)) {
        .as_numeric_vector(expected_values)
      } else {
        as.numeric(expected_values)[keep]
      }
    }
    if (!is.null(event_values)) {
      frame$event <- if (is.null(group_info)) {
        .as_numeric_vector(event_values)
      } else {
        as.numeric(event_values)[keep]
      }
    }
    out <- list(
      call = call,
      data = frame,
      offtable = as.numeric(.result_field(result, "offtable")),
      tcut = isTRUE(.result_field(result, "tcut")),
      observations = as.integer(.result_field(result, "observations"))
    )
    if (!is.null(terms)) {
      out$terms <- terms
    }
    class(out) <- "pyears"
    return(out)
  }
  out <- list(
    call = call,
    pyears = pyears_values,
    n = n_values,
    offtable = as.numeric(.result_field(result, "offtable")),
    tcut = isTRUE(.result_field(result, "tcut")),
    observations = as.integer(.result_field(result, "observations"))
  )
  if (!is.null(expected_values)) {
    expected <- if (is.null(group_info)) .as_numeric_vector(expected_values) else expected_values
    if (is.null(group_info) && grouped && !is.null(group_name)) {
      expected <- structure(
        expected,
        dim = length(pyears_values),
        dimnames = stats::setNames(list(groups), group_name)
      )
    } else if (is.null(group_info) && grouped) {
      names(expected) <- groups
    }
    out$expected <- expected
  }
  if (!is.null(event_values)) {
    events <- if (is.null(group_info)) .as_numeric_vector(event_values) else event_values
    if (is.null(group_info) && grouped && !is.null(group_name)) {
      events <- structure(
        events,
        dim = length(pyears_values),
        dimnames = stats::setNames(list(groups), group_name)
      )
    } else if (is.null(group_info) && grouped) {
      names(events) <- groups
    }
    out$event <- events
  }
  if ("observations" %in% names(out)) {
    observations <- out$observations
    out$observations <- NULL
    out$observations <- observations
  }
  if (!is.null(terms)) {
    out$terms <- terms
  }
  class(out) <- "pyears"
  out
}

.pyears_formula_python_eligible <- function(formula, data, rmap, ratetable) {
  if (!inherits(formula, "formula") || missing(data) || is.null(data)) {
    return(FALSE)
  }
  if (!missing(rmap) || (!missing(ratetable) && !is.null(ratetable))) {
    return(FALSE)
  }
  TRUE
}

.pyears_formula_response_args <- function(response) {
  if (inherits(response, "Surv") && is.matrix(response)) {
    surv_type <- attr(response, "type")
    if (identical(surv_type, "right")) {
      return(list(
        direct = NULL,
        time = response[, 1L],
        start = NULL,
        stop = NULL,
        event = response[, 2L]
      ))
    }
    if (identical(surv_type, "counting")) {
      return(list(
        direct = NULL,
        time = NULL,
        start = response[, 1L],
        stop = response[, 2L],
        event = response[, 3L]
      ))
    }
    stop("Only right-censored and counting process survival types are supported", call. = FALSE)
  }
  if (any(response < 0)) {
    stop("Negative follow up time", call. = FALSE)
  }
  list(
    direct = as.numeric(response),
    time = NULL,
    start = NULL,
    stop = NULL,
    event = NULL
  )
}

pyears <- function(formula, data, weights, subset, na.action, rmap, ratetable,
                   scale = 365.25, expect = c("event", "pyears"),
                   model = FALSE, x = FALSE, y = FALSE, data.frame = FALSE,
                   time, start, stop, event = NULL, group = NULL) {
  direct_time <- NULL
  if (!missing(time)) {
    direct_time <- time
  } else if (!missing(formula) && !inherits(formula, "formula")) {
    direct_time <- formula
  }
  expect <- match.arg(expect)
  if (is.null(direct_time) && missing(start) && missing(stop)) {
    if (!missing(formula) && .pyears_formula_python_eligible(formula, data, rmap, ratetable)) {
      output_terms <- stats::terms(formula, data = data)
      model_formula <- formula
      model_env <- new.env(parent = environment(model_formula))
      model_env$Surv <- .survsplit_model_frame_surv
      environment(model_formula) <- model_env
      formula_terms <- stats::terms(model_formula, data = data)
      if (any(attr(formula_terms, "order") > 1L)) {
        stop("Pyears cannot have interaction terms", call. = FALSE)
      }
      model_call <- match.call()[c(1L, match(
        c("formula", "data", "weights", "subset", "na.action"),
        names(match.call()),
        nomatch = 0L
      ))]
      model_call$formula <- formula_terms
      model_call[[1L]] <- quote(stats::model.frame)
      mf <- eval(model_call, parent.frame())
      response <- stats::model.extract(mf, "response")
      if (is.null(response)) {
        stop("Follow-up time must appear in the formula", call. = FALSE)
      }
      response_args <- .pyears_formula_response_args(response)
      term_labels <- attr(formula_terms, "term.labels")
      term_values <- if (length(term_labels) == 0L) {
        NULL
      } else {
        mf[term_labels]
      }
      if (!is.null(term_values) && any(vapply(term_values, inherits, logical(1), "tcut"))) {
        call <- match.call()
        call[[1L]] <- quote(survival::pyears)
        return(eval.parent(call))
      }
      group_info <- .pyears_formula_group_info(term_labels, mf)
      weight_values <- stats::model.weights(mf)
      if (is.null(weight_values)) {
        weight_values <- rep(1, nrow(mf))
      }
      result <- .call_r_api(
        "pyears",
        if (is.null(response_args$direct)) NULL else .as_python_vector(response_args$direct),
        time = if (is.null(response_args$time)) NULL else .as_python_vector(response_args$time),
        start = if (is.null(response_args$start)) NULL else .as_python_vector(response_args$start),
        stop = if (is.null(response_args$stop)) NULL else .as_python_vector(response_args$stop),
        event = if (is.null(response_args$event)) NULL else .as_python_vector(response_args$event),
        group = if (is.null(group_info)) NULL else .as_python_vector(group_info$row_keys),
        weights = .as_python_vector(weight_values),
        scale = scale,
        `data_frame` = FALSE
      )
      return(.as_pyears_result(
        result,
        match.call(),
        data.frame = data.frame,
        terms = output_terms,
        group_name = if (length(term_labels) == 1L) term_labels[[1L]] else NULL,
        group_info = group_info
      ))
    }
    call <- match.call()
    call[[1L]] <- quote(survival::pyears)
    return(eval.parent(call))
  }
  if (!missing(ratetable) && !is.null(ratetable)) {
    stop("direct pyears bridge does not yet support ratetable; use formula pyears for R ratetables", call. = FALSE)
  }
  result <- .call_r_api(
    "pyears",
    if (is.null(direct_time)) NULL else .as_python_vector(direct_time),
    time = if (missing(time) || !is.null(direct_time)) NULL else .as_python_vector(time),
    start = if (missing(start)) NULL else .as_python_vector(start),
    stop = if (missing(stop)) NULL else .as_python_vector(stop),
    event = if (is.null(event)) NULL else .as_python_vector(event),
    group = if (is.null(group)) NULL else .as_python_vector(group),
    weights = if (missing(weights)) NULL else .as_python_vector(weights),
    subset = if (missing(subset)) NULL else subset,
    `na_action` = if (missing(na.action)) NULL else .as_na_action(na.action),
    scale = scale,
    `data_frame` = FALSE
  )
  .as_pyears_result(result, match.call(), data.frame = data.frame)
}

.finegray_model_frame_surv <- function(time, time2, event, type = NULL, origin = 0,
                                       time1, start, stop, status) {
  if (missing(time) && !missing(time1)) {
    time <- time1
  }
  if (missing(time) && !missing(start)) {
    time <- start
  }
  if (missing(time2) && !missing(stop)) {
    time2 <- stop
  }
  if (missing(event) && !missing(status)) {
    event <- status
  }
  if (missing(time)) {
    stop("Must have a time argument")
  }
  if (inherits(time, "difftime")) {
    time <- unclass(time)
  }
  if (!is.numeric(time)) {
    stop("Time variable is not numeric")
  }
  counting <- !missing(time2) && !missing(event)
  if (!counting && missing(event)) {
    event <- time2
  }
  if (missing(event)) {
    stop("Wrong number of args for this type of survival data")
  }
  if (length(event) != length(time)) {
    stop("Time and status are different lengths")
  }
  if (counting) {
    if (inherits(time2, "difftime")) {
      time2 <- unclass(time2)
    }
    if (!is.numeric(time2)) {
      stop("Time variable is not numeric")
    }
    if (length(time2) != length(time)) {
      stop("Start and stop are different lengths")
    }
  }
  if (is.factor(event)) {
    mstat <- as.factor(event)
    event_status <- as.numeric(mstat) - 1L
    states <- levels(mstat)[-1L]
    if (any(is.na(states) | states == "")) {
      stop("each state must have a non-blank name")
    }
    out <- if (counting) {
      cbind(start = as.numeric(time) - origin, stop = as.numeric(time2) - origin, status = event_status)
    } else {
      cbind(time = as.numeric(time) - origin, status = event_status)
    }
    attr(out, "type") <- if (counting) "mcounting" else "mright"
    attr(out, "states") <- states
    attr(out, "inputAttributes") <- list(event = attributes(event))
    class(out) <- "Surv"
    return(out)
  }
  event_status <- if (is.logical(event)) {
    as.numeric(event)
  } else if (is.numeric(event)) {
    if (max(event[!is.na(event)]) == 2) event - 1 else event
  } else {
    stop("Invalid status value, must be logical or numeric")
  }
  out <- if (counting) {
    cbind(start = as.numeric(time) - origin, stop = as.numeric(time2) - origin, status = event_status)
  } else {
    cbind(time = as.numeric(time) - origin, status = event_status)
  }
  attr(out, "type") <- if (counting) "counting" else "right"
  class(out) <- "Surv"
  out
}

.finegray_censor_surv <- function(start, stop, event) {
  event_times <- sort(unique(stop[event]))
  if (length(event_times) == 0L) {
    return(list(time = numeric(), surv = numeric(), n.event = numeric()))
  }
  surv <- 1
  out_time <- numeric()
  out_surv <- numeric()
  out_event <- numeric()
  for (time in event_times) {
    n_risk <- sum(start < time & stop >= time)
    n_event <- sum(event & stop == time)
    if (n_risk > 0L && n_event > 0L) {
      surv <- surv * (1 - n_event / n_risk)
      out_time <- c(out_time, time)
      out_surv <- c(out_surv, surv)
      out_event <- c(out_event, n_event)
    }
  }
  list(time = out_time, surv = out_surv, n.event = out_event)
}

.finegray_python_vector <- function(value) {
  as.list(.as_python_vector(value))
}

.finegray_formula_bridge <- function(call, env, na.action, etype, etype_missing,
                                     prefix, count, count_missing, timefix) {
  if (is.null(call$formula)) {
    return(NULL)
  }
  formula <- eval(call$formula, env)
  if (!inherits(formula, "formula")) {
    return(NULL)
  }
  model_formula <- formula
  model_env <- new.env(parent = environment(model_formula))
  model_env$Surv <- .finegray_model_frame_surv
  environment(model_formula) <- model_env

  data_value <- if (is.null(call$data)) NULL else eval(call$data, env)
  special <- c("strata", "cluster")
  formula_terms <- if (is.null(data_value)) {
    stats::terms(model_formula, specials = special)
  } else {
    stats::terms(model_formula, specials = special, data = data_value)
  }
  indx <- match(c("formula", "data", "weights", "subset", "id"), names(call), nomatch = 0L)
  model_call <- call[c(1L, indx)]
  model_call$formula <- formula_terms
  model_call$na.action <- na.action
  model_call[[1L]] <- quote(stats::model.frame)
  mf <- eval(model_call, env)
  if (nrow(mf) == 0L) {
    stop("No (non-missing) observations")
  }

  Terms <- stats::terms(mf)
  Y <- stats::model.extract(mf, "response")
  if (!inherits(Y, "Surv")) {
    stop("Response must be a survival object")
  }
  type <- attr(Y, "type")
  if (!identical(type, "mright") && !identical(type, "mcounting")) {
    return(NULL)
  }
  nY <- ncol(Y)
  states <- attr(Y, "states")
  if (length(states) < 2L) {
    stop("survival time has only a single state")
  }
  if (isTRUE(timefix)) {
    # The expansion path is otherwise independent of exact R Surv internals.
    # Right-multistate timefix only affects near-tied stop times, so leave
    # already-distinct times unchanged here.
    Y[, 1L] <- as.numeric(Y[, 1L])
  }

  strats <- attr(Terms, "specials")$strata
  if (length(strats)) {
    stemp <- untangle.specials(Terms, "strata", 1)
    if (length(stemp$vars) == 1L) {
      strata_values <- mf[[stemp$vars]]
    } else {
      strata_values <- strata(mf[, stemp$vars, drop = FALSE], shortlabel = TRUE)
    }
    istrat <- as.numeric(strata_values)
    mf[stemp$vars] <- NULL
  } else {
    istrat <- rep(1, nrow(mf))
  }

  id <- stats::model.extract(mf, "id")
  if (!is.null(id)) {
    mf["(id)"] <- NULL
  }
  user.weights <- stats::model.weights(mf)
  if (is.null(user.weights)) {
    user.weights <- rep(1, nrow(mf))
  }
  cluster <- attr(Terms, "specials")$cluster
  if (length(cluster)) {
    stop("a cluster() term is not valid")
  }

  if (isTRUE(etype_missing)) {
    enum <- 1L
  } else {
    index <- match(etype, states)
    if (any(is.na(index))) {
      stop("etype argument has a state that is not in the data")
    }
    enum <- index[[1L]]
    if (length(index) > 1L) {
      warning("only the first endpoint was used")
    }
  }
  count_name <- if (isTRUE(count_missing)) NULL else make.names(count)
  oname <- paste0(prefix, c("start", "stop", "status", "wt"))

  delay <- FALSE
  if (identical(type, "mcounting")) {
    if (is.null(id)) {
      stop("(start, stop] data requires a subject id")
    }
    index <- order(id, Y[, 2L])
    sorty <- Y[index, , drop = FALSE]
    first_index <- which(!duplicated(id[index]))
    last_index <- c(first_index[-1L] - 1L, length(id))
    if (any(sorty[-last_index, 3L] != 0)) {
      stop("a subject has a transition before their last time point")
    }
    delta <- c(sorty[-1L, 1L], 0) - sorty[, 2L]
    if (any(delta[-last_index] != 0)) {
      stop("a subject has gaps in time")
    }
    if (any(Y[first_index, 1L] > min(Y[, 2L]))) {
      delay <- TRUE
    }
    first <- last <- rep(FALSE, nrow(mf))
    first[index[first_index]] <- TRUE
    last[index[last_index]] <- TRUE
  } else {
    last <- rep(TRUE, nrow(mf))
  }
  if (nY == 2L) {
    temp <- min(Y[, 1L], na.rm = TRUE)
    zero <- if (temp > 0) 0 else 2 * temp - 1
    Y <- cbind(zero, Y)
  }
  utime <- sort(unique(c(Y[, 1:2])))
  newtime <- matrix(findInterval(Y[, 1:2], utime), ncol = 2L)
  status <- Y[, 3L]
  newtime[status != 0, 2L] <- newtime[status != 0, 2L] - 0.2

  stratfun <- function(stratum) {
    keep <- istrat == stratum
    times <- sort(unique(Y[keep & status == enum, 2L]))
    if (length(times) == 0L) {
      return(NULL)
    }
    tdata <- mf[keep, -1L, drop = FALSE]
    maxtime <- max(Y[keep, 2L])
    gsurv <- .finegray_censor_surv(
      newtime[keep, 1L],
      newtime[keep, 2L],
      last[keep] & status[keep] == 0
    )
    if (isTRUE(delay)) {
      hsurv <- .finegray_censor_surv(
        -newtime[keep, 2L],
        -newtime[keep, 1L],
        first[keep]
      )
      dtime <- rev(-hsurv$time[hsurv$n.event > 0])
      dprob <- c(rev(hsurv$surv[hsurv$n.event > 0])[-1L], 1)
      ctime_index <- gsurv$time[gsurv$n.event > 0]
      cprob_index <- c(1, gsurv$surv[gsurv$n.event > 0])
      temp <- sort(unique(c(dtime, ctime_index)))
      index1 <- findInterval(temp, dtime)
      index2 <- findInterval(temp, ctime_index)
      ctime <- utime[temp]
      cprob <- dprob[index1] * cprob_index[index2 + 1L]
    } else {
      ctime <- utime[gsurv$time[gsurv$n.event > 0]]
      cprob <- gsurv$surv[gsurv$n.event > 0]
    }
    ct2 <- c(ctime, maxtime)
    cp2 <- c(1, cprob)
    index <- findInterval(times, ct2, left.open = TRUE)
    index <- sort(unique(index))
    ckeep <- rep(FALSE, length(ct2))
    ckeep[index] <- TRUE
    expand <- (Y[keep, 3L] != 0 & Y[keep, 3L] != enum & last[keep])
    keep_arg <- c(TRUE, ckeep)[seq_along(ct2)]
    split <- .call_r_api(
      "finegray",
      .finegray_python_vector(Y[keep, 1L]),
      .finegray_python_vector(Y[keep, 2L]),
      .finegray_python_vector(ct2),
      .finegray_python_vector(cp2),
      .finegray_python_vector(expand),
      .finegray_python_vector(keep_arg)
    )
    rows <- as.integer(.result_field(split, "row"))
    tdata <- tdata[rows, , drop = FALSE]
    tstat <- ifelse((status[keep])[rows] == enum, 1, 0)
    tdata[[oname[[1L]]]] <- .as_numeric_vector(.result_field(split, "start"))
    tdata[[oname[[2L]]]] <- .as_numeric_vector(.result_field(split, "end"))
    tdata[[oname[[3L]]]] <- tstat
    tdata[[oname[[4L]]]] <- .as_numeric_vector(.result_field(split, "wt")) * user.weights[keep][rows]
    if (!is.null(count_name)) {
      tdata[[count_name]] <- as.integer(.result_field(split, "add"))
    }
    tdata
  }

  if (max(istrat) == 1L) {
    result <- stratfun(1L)
  } else {
    result <- do.call("rbind", lapply(seq_len(max(istrat)), stratfun))
  }
  rownames(result) <- NULL
  attr(result, "event") <- states[[enum]]
  result
}

finegray <- function(formula, data, weights, subset, na.action = na.pass,
                     etype, prefix = "fg", count = "", id, timefix = TRUE,
                     tstart, tstop, ctime, cprob, extend, keep) {
  direct_start <- NULL
  if (!missing(tstart)) {
    direct_start <- tstart
  } else if (!missing(formula) && !inherits(formula, "formula")) {
    direct_start <- formula
  }
  if (is.null(direct_start)) {
    call <- match.call()
    bridged <- .finegray_formula_bridge(
      call,
      parent.frame(),
      na.action,
      etype = if (missing(etype)) NULL else etype,
      etype_missing = missing(etype),
      prefix = prefix,
      count = count,
      count_missing = missing(count),
      timefix = timefix
    )
    if (!is.null(bridged)) {
      return(bridged)
    }
    call <- match.call()
    call[[1L]] <- quote(survival::finegray)
    return(eval.parent(call))
  }
  if (missing(tstop) || missing(ctime) || missing(cprob) || missing(extend) || missing(keep)) {
    stop("direct finegray bridge requires tstart, tstop, ctime, cprob, extend, and keep", call. = FALSE)
  }
  result <- .call_r_api(
    "finegray",
    .finegray_python_vector(direct_start),
    .finegray_python_vector(tstop),
    .finegray_python_vector(ctime),
    .finegray_python_vector(cprob),
    .finegray_python_vector(extend),
    .finegray_python_vector(keep)
  )
  data.frame(
    row = as.integer(.result_field(result, "row")),
    start = .as_numeric_vector(.result_field(result, "start")),
    end = .as_numeric_vector(.result_field(result, "end")),
    wt = .as_numeric_vector(.result_field(result, "wt")),
    add = as.integer(.result_field(result, "add"))
  )
}

survobrien <- function(formula, data, subset, na.action, transform,
                       time, status, covariate, strata = NULL) {
  direct_time <- NULL
  if (!missing(time)) {
    direct_time <- time
  } else if (!missing(formula) && !inherits(formula, "formula")) {
    direct_time <- formula
  }
  if (is.null(direct_time)) {
    if (.survobrien_formula_python_eligible(formula, data)) {
      result <- .call_r_api(
        "survobrien",
        .as_formula_string(formula),
        data = .as_python_data(data),
        subset = if (missing(subset)) NULL else subset,
        `na_action` = if (missing(na.action)) NULL else .as_na_action(na.action),
        transform = if (missing(transform)) NULL else transform
      )
      return(as.data.frame(result, check.names = TRUE, stringsAsFactors = FALSE))
    }
    call <- match.call()
    call[[1L]] <- quote(survival::survobrien)
    return(eval.parent(call))
  }
  if (missing(status) || missing(covariate)) {
    stop("direct survobrien bridge requires time, status, and covariate", call. = FALSE)
  }
  result <- .call_r_api(
    "survobrien",
    time = .as_python_vector(direct_time),
    status = .as_python_vector(status),
    covariate = .as_python_vector(covariate),
    strata = if (is.null(strata)) NULL else .as_python_vector(strata)
  )
  list(
    statistic = as.numeric(.result_field(result, "statistic")),
    p.value = as.numeric(.result_field(result, "p_value")),
    df = as.integer(.result_field(result, "df")),
    scores = .as_numeric_vector(.result_field(result, "scores")),
    score.sum = as.numeric(.result_field(result, "score_sum")),
    expected = as.numeric(.result_field(result, "expected")),
    variance = as.numeric(.result_field(result, "variance"))
  )
}

.survobrien_formula_numeric_term <- function(label, data) {
  if (label %in% names(data)) {
    return(is.numeric(data[[label]]) && !is.factor(data[[label]]))
  }
  match <- regexec(
    "^(log|sqrt|exp|identity|as\\.numeric)\\(([[:space:]]*[^()]+[[:space:]]*)\\)$",
    label
  )
  pieces <- regmatches(label, match)[[1L]]
  if (length(pieces) == 0L) {
    return(FALSE)
  }
  column <- trimws(pieces[[3L]])
  column %in% names(data) && is.numeric(data[[column]]) && !is.factor(data[[column]])
}

.survobrien_formula_python_eligible <- function(formula, data) {
  if (!inherits(formula, "formula") || missing(data) || is.null(data)) {
    return(FALSE)
  }
  terms <- stats::terms(formula, specials = c("strata", "cluster", "tt"), data = data)
  if (any(attr(terms, "order") > 1L)) {
    return(FALSE)
  }
  specials <- attr(terms, "specials")
  if (length(specials$cluster) > 1L || length(specials$tt) > 0L) {
    return(FALSE)
  }
  labels <- attr(terms, "term.labels")
  labels <- labels[!grepl("^(strata|cluster)\\(", labels)]
  if (length(labels) == 0L) {
    return(FALSE)
  }
  all(vapply(labels, .survobrien_formula_numeric_term, logical(1), data = data))
}

fromtimeline <- function(formula, data, id, istate = "istate") {
  call <- match.call()
  model_formula <- formula
  if (inherits(model_formula, "formula")) {
    model_env <- new.env(parent = environment(model_formula))
    model_env$Surv <- .native_model_frame_surv
    environment(model_formula) <- model_env
  }
  keep <- match(c("formula", "data", "id"), names(call), nomatch = 0L)
  tcall <- call[c(1L, keep)]
  tcall$formula <- model_formula
  tcall[[1L]] <- quote(stats::model.frame)
  mf <- eval(tcall, parent.frame())
  id_values <- stats::model.extract(mf, "id")
  response <- stats::model.response(mf)
  if (!inherits(response, "Surv")) {
    stop("response must be a Surv object", call. = FALSE)
  }
  response_type <- attr(response, "type")
  if (!(response_type %in% c("right", "mright"))) {
    stop("only valid for a right censored response", call. = FALSE)
  }

  tname <- c("tstart", "tstop")
  sname <- "state"
  lhs <- formula[[2L]]
  if ((is.name(lhs[[1L]]) && identical(lhs[[1L]], quote(Surv))) ||
      identical(deparse(lhs[[1L]]), "survival::Surv")) {
    if (is.name(lhs[[2L]])) {
      temp <- as.character(lhs[[2L]])
      tname <- paste0(temp, 1:2)
    }
    if (is.name(lhs[[3L]])) {
      sname <- as.character(lhs[[3L]])
    } else if (is.call(lhs[[3L]])) {
      temp <- lhs[[3L]]
      if (identical(deparse(temp[[1L]]), "factor") && is.name(temp[[2L]])) {
        sname <- as.character(temp[[2L]])
      }
    }
  } else if (is.name(lhs[[1L]])) {
    tname <- paste0(lhs[[1L]], 1:2)
  }

  if (is.name(call$id)) {
    idname <- as.character(call$id)
    if (is.na(match(idname, names(mf)))) {
      names(mf)[match("(id)", names(mf))] <- idname
    }
  } else {
    stop("id must be a simple variable name", call. = FALSE)
  }

  result <- .call_r_api(
    "fromtimeline",
    time = .as_python_vector(response[, 1L]),
    status = .as_python_vector(response[, 2L]),
    id = .as_python_vector(id_values),
    states = attr(response, "states"),
    data = .as_python_data(mf[-1L]),
    id_name = idname
  )
  removed_id <- .result_field(result, "removed_id")
  if (!is.null(removed_id) && length(removed_id) > 0L) {
    warning("identifiers with only 1 row were removed", call. = FALSE)
  }

  static <- as.logical(.result_field(result, "static"))
  static_rows <- as.integer(.as_numeric_vector(.result_field(result, "static_row"))) + 1L
  dynamic_rows <- as.integer(.as_numeric_vector(.result_field(result, "dynamic_row"))) + 1L
  n_out <- length(static_rows)
  data_names <- names(mf)[-1L]

  output <- list()
  static_names <- data_names[static]
  for (name in static_names) {
    output[[name]] <- mf[[name]][static_rows]
  }
  while (!is.na(match(sname, data_names))) {
    sname <- paste0(sname, "1")
  }
  output[[tname[[1L]]]] <- .as_numeric_vector(.result_field(result, "start"))
  output[[tname[[2L]]]] <- .as_numeric_vector(.result_field(result, "stop"))
  status_values <- as.integer(.as_numeric_vector(.result_field(result, "status")))
  state_levels <- as.character(.result_field(result, "state_levels"))
  if (length(state_levels) > 0L) {
    status_column <- factor(status_values, seq.int(0L, length(state_levels) - 1L), state_levels)
  } else {
    status_column <- as.numeric(status_values)
  }
  output[[sname]] <- status_column
  dynamic_names <- data_names[!static]
  if (length(dynamic_names) > 0L) {
    for (name in dynamic_names) {
      output[[name]] <- mf[[name]][dynamic_rows]
    }
  }

  if (any(istate == names(output))) {
    stop("istate option duplicates an existing variable", call. = FALSE)
  }
  istate_values <- as.integer(.as_numeric_vector(.result_field(result, "istate")))
  istate_levels <- as.character(.result_field(result, "istate_levels"))
  output[[istate]] <- if (length(istate_levels) > 0L) {
    factor(istate_values, seq_along(istate_levels), istate_levels)
  } else {
    as.numeric(istate_values)
  }

  out <- as.data.frame(output, stringsAsFactors = FALSE, optional = TRUE)
  row.names(out) <- seq_len(n_out)
  n_subject <- length(unique(static_rows))
  tcount_rows <- c(sname, dynamic_names)
  tcount <- matrix(
    0,
    nrow = length(tcount_rows),
    ncol = 9L,
    dimnames = list(
      tcount_rows,
      c("early", "late", "gap", "within", "boundary", "leading", "trailing", "tied", "missid")
    )
  )
  if (length(tcount_rows) > 0L) {
    middle_count <- max(0L, n_out - n_subject)
    tcount[sname, "within"] <- middle_count
    tcount[sname, "leading"] <- n_subject
    tcount[sname, "trailing"] <- n_subject
    if (length(dynamic_names) > 0L) {
      tcount[dynamic_names, "boundary"] <- middle_count
      tcount[dynamic_names, "leading"] <- n_subject
      tcount[dynamic_names, "trailing"] <- n_subject
    }
  }
  attr(out, "tm.retain") <- list(
    tname = list(idname = idname, tstartname = tname[[1L]], tstopname = tname[[2L]]),
    n = as.integer(n_out),
    tevent = list(name = sname, censor = stats::setNames(list(0), sname)),
    tdcvar = dynamic_names
  )
  attr(out, "tcount") <- tcount
  call_parts <- c(
    "tmerge(data1 = new, data2 = d2, id = ", idname,
    ", ", sname, " = event(.y1., .y2.)"
  )
  if (length(dynamic_names) > 0L) {
    call_parts <- c(
      call_parts,
      paste0(", ", dynamic_names, " = tdc(.y1., ", dynamic_names, ")", collapse = "")
    )
  }
  call_parts <- c(call_parts, ")")
  attr(out, "call") <- parse(text = paste0(call_parts, collapse = ""))[[1L]]
  class(out) <- c("tmerge", "data.frame")
  out
}

totimeline <- function(formula, data, id, istate) {
  if (missing(formula) || missing(data) || missing(id)) {
    stop("formula, data, and id arguments are required", call. = FALSE)
  }
  call <- match.call()
  model_formula <- formula
  if (inherits(model_formula, "formula")) {
    model_env <- new.env(parent = environment(model_formula))
    model_env$Surv <- .native_model_frame_surv
    environment(model_formula) <- model_env
  }
  tcall <- call
  tcall$formula <- model_formula
  tcall[[1L]] <- quote(stats::model.frame)
  mf <- eval(tcall, parent.frame())
  response <- stats::model.response(mf)
  id_values <- stats::model.extract(mf, "id")
  if (!inherits(response, "Surv")) {
    stop("response must be a Surv object", call. = FALSE)
  }
  response_type <- attr(response, "type")
  if (response_type %in% c("left", "interval")) {
    stop("not valid for interval censored or left censored data", call. = FALSE)
  }
  if (ncol(response) != 3L) {
    stop("initial data is not in (time1, time2) form", call. = FALSE)
  }

  tname <- "(time)"
  sname <- "(state)"
  lhs <- formula[[2L]]
  if ((is.name(lhs[[1L]]) && identical(lhs[[1L]], quote(Surv))) ||
      identical(deparse(lhs[[1L]]), "survival::Surv")) {
    if (is.name(lhs[[3L]])) {
      tname <- as.character(lhs[[3L]])
    }
    if (is.name(lhs[[4L]])) {
      sname <- as.character(lhs[[4L]])
    } else if (is.call(lhs[[4L]])) {
      temp <- lhs[[4L]]
      if (identical(deparse(temp[[1L]]), "factor") && is.name(temp[[2L]])) {
        sname <- as.character(temp[[2L]])
      }
    }
  }

  if (is.name(call$id)) {
    idname <- as.character(call$id)
    if (is.na(match(idname, names(mf)))) {
      names(mf)[match("(id)", names(mf))] <- idname
    }
  } else {
    stop("id must be a simple variable name", call. = FALSE)
  }

  if (missing(istate)) {
    istate_values <- NULL
    istate_levels <- NULL
  } else {
    istate_factor <- as.factor(stats::model.extract(mf, "istate"))
    istate_values <- as.character(istate_factor)
    istate_levels <- levels(istate_factor)
  }
  result <- .call_r_api(
    "totimeline",
    start = .as_python_vector(response[, 1L]),
    stop = .as_python_vector(response[, 2L]),
    status = .as_python_vector(response[, 3L]),
    states = attr(response, "states"),
    id = .as_python_vector(id_values),
    istate = if (is.null(istate_values)) NULL else .as_python_vector(istate_values),
    istate_levels = if (is.null(istate_levels)) NULL else .as_python_vector(istate_levels)
  )

  state_levels <- as.character(.result_field(result, "state_levels"))
  newdata <- cbind(
    data.frame(
      `(time)` = .as_numeric_vector(.result_field(result, "time")),
      `(state)` = factor(
        as.integer(.as_numeric_vector(.result_field(result, "status"))),
        seq.int(0L, length(state_levels) - 1L),
        state_levels
      ),
      check.names = FALSE
    ),
    mf[as.integer(.as_numeric_vector(.result_field(result, "data_row"))) + 1L, -1L, drop = FALSE]
  )
  row.names(newdata) <- NULL
  names(newdata)[1:2] <- c(tname, sname)
  indx <- match(c("(id)", "(istate)"), names(newdata), nomatch = 0L)
  newdata[, -indx]
}

.as_surv2data_response <- function(result) {
  response_type <- as.character(.result_field(result, "type"))
  states <- as.character(.result_field(result, "states"))
  start <- .as_numeric_vector(.result_field(result, "start"))
  stop <- .as_numeric_vector(.result_field(result, "stop"))
  status <- as.integer(.as_numeric_vector(.result_field(result, "status")))

  if (response_type %in% c("right", "mright")) {
    response <- cbind(time = stop, status = status)
  } else {
    response <- cbind(start = start, stop = stop, status = status)
  }
  colnames(response) <- NULL
  attr(response, "type") <- response_type
  if (length(states) > 0L) {
    attr(response, "states") <- states
  }
  class(response) <- "Surv"
  response
}

.as_surv2data_istate <- function(result) {
  states <- as.character(.result_field(result, "states"))
  istate <- .result_field(result, "istate")
  if (length(states) == 0L) {
    return(as.integer(.as_numeric_vector(istate)))
  }
  codes <- as.integer(.as_numeric_vector(istate))
  factor(codes, levels = seq_along(states), labels = states)
}

Surv2data <- function(formula, data, subset, id) {
  call <- match.call()
  indx <- match(
    c("formula", "data", "weights", "subset", "na.action", "cluster", "id", "istate"),
    names(call),
    nomatch = 0L
  )
  if (indx[[1L]] == 0L) {
    stop("A formula argument is required", call. = FALSE)
  }
  tform <- call[c(1L, indx)]
  tform$na.action <- stats::na.pass
  tform[[1L]] <- quote(stats::model.frame)
  mf <- eval(tform, parent.frame())
  response <- stats::model.response(mf)
  if (!inherits(response, "Surv2")) {
    stop("response must be a Surv2 object", call. = FALSE)
  }
  id_values <- stats::model.extract(mf, "id")
  result <- .call_r_api(
    "Surv2data",
    time = .as_python_vector(response[, 1L]),
    status = .as_python_vector(response[, 2L]),
    states = attr(response, "states"),
    repeated = isTRUE(attr(response, "repeated")),
    id = .as_python_vector(id_values)
  )
  rows <- as.integer(.as_numeric_vector(.result_field(result, "row"))) + 1L
  mf2 <- mf[rows, , drop = FALSE]
  mf2[["Surv2.y"]] <- .as_surv2data_response(result)

  id_result <- .result_field(result, "id")
  id_name <- paste(deparse(call$id, width.cutoff = 500L), collapse = " ")
  index <- match(id_name, names(mf2), nomatch = 0L)
  if (index > 0L) {
    mf2[[index]] <- id_result
  } else {
    mf2[["Surv2.id"]] <- id_result
  }
  mf2[["Surv2.istate"]] <- .as_surv2data_istate(result)
  attr(mf2, "terms") <- NULL
  mf2
}

yates <- function(fit, term, population = c("data", "factorial", "sas"),
                  levels, test = c("global", "trend", "pairwise"),
                  predict = "linear", options, nsim = 200,
                  method = c("direct", "sgtt")) {
  call <- match.call()
  call[[1L]] <- quote(survival::yates)
  eval.parent(call)
}

yates_setup <- function(fit, ...) {
  UseMethod("yates_setup", fit)
}

yates_setup.default <- function(fit, type, ...) {
  if (!missing(type) && !(type %in% c("linear", "link"))) {
    warning(
      "no yates_setup method exists for a model of class ",
      class(fit)[[1L]],
      " and estimate type ",
      type,
      ", linear predictor estimate used by default",
      call. = FALSE
    )
  }
  NULL
}

yates_setup.glm <- function(fit, predict = c("link", "response", "terms", "linear"), ...) {
  type <- match.arg(predict)
  if (type == "link" || type == "linear") {
    return(NULL)
  }
  if (type == "response") {
    finv <- stats::family(fit)$linkinv
    return(function(eta, X) finv(eta))
  }
  stop("type terms not yet supported", call. = FALSE)
}

.yates_setup_coxph <- function(fit, predict = c("lp", "risk", "expected", "terms",
                                                "survival", "linear"),
                               options, ...) {
  type <- match.arg(predict)
  if (type == "lp" || type == "linear") {
    return(NULL)
  }
  if (type == "risk") {
    return(function(eta, X) exp(eta))
  }
  if (type == "survival") {
    suppressWarnings(baseline <- survfit(fit, censor = FALSE))
    rmean <- if (missing(options) || is.null(options$rmean)) {
      max(baseline$time)
    } else {
      options$rmean
    }
    if (!is.null(baseline$strata)) {
      stop("stratified models not yet supported", call. = FALSE)
    }
    cumhaz <- c(0, baseline$cumhaz)
    tt <- c(diff(c(0, pmin(rmean, baseline$time))), 0)
    predict_fun <- function(eta, ...) {
      c2 <- outer(exp(drop(eta)), cumhaz)
      surv <- exp(-c2)
      meansurv <- apply(rep(tt, each = nrow(c2)) * surv, 1L, sum)
      cbind(meansurv, surv)
    }
    summary_fun <- function(surv, var) {
      bsurv <- t(surv[, -1L])
      std <- t(sqrt(var[, -1L]))
      chaz <- -log(bsurv)
      zstat <- -stats::qnorm((1 - baseline$conf.int) / 2)
      baseline$lower <- exp(-(chaz + zstat * std))
      baseline$upper <- exp(-(chaz - zstat * std))
      baseline$surv <- bsurv
      baseline$std.err <- std / bsurv
      baselinecumhaz <- chaz
      baseline
    }
    return(list(predict = predict_fun, summary = summary_fun))
  }
  stop("type expected is not supported", call. = FALSE)
}

yates_setup.coxph <- function(fit, predict = c("lp", "risk", "expected", "terms",
                                               "survival", "linear"),
                              options, ...) {
  .yates_setup_coxph(fit, predict = predict, options = if (missing(options)) NULL else options, ...)
}

yates_setup.survival_py_coxph <- function(fit, predict = c("lp", "risk", "expected",
                                                           "terms", "survival", "linear"),
                                          options, ...) {
  .yates_setup_coxph(fit, predict = predict, options = if (missing(options)) NULL else options, ...)
}

aareg <- function(formula, data, weights, subset, na.action, qrtol = 1e-07,
                  nmin, dfbeta = FALSE, taper = 1,
                  test = c("aalen", "variance", "nrisk"),
                  cluster, model = FALSE, x = FALSE, y = FALSE) {
  call <- match.call()
  call[[1L]] <- quote(survival::aareg)
  eval.parent(call)
}

cch <- function(formula, data, subcoh, id, stratum = NULL, cohort.size,
                method = c("Prentice", "SelfPrentice", "LinYing",
                           "I.Borgan", "II.Borgan"),
                robust = FALSE) {
  call <- match.call()
  call[[1L]] <- quote(survival::cch)
  eval.parent(call)
}

clogit <- function(formula, data, weights, subset, na.action,
                   method = c("exact", "approximate", "efron", "breslow"),
                   ...) {
  call <- match.call()
  indx <- match(c("formula", "data"), names(call), nomatch = 0)
  if (indx[[1L]] == 0L) {
    stop("A formula argument is required", call. = FALSE)
  }
  mf <- call[c(1L, indx)]
  mf[[1L]] <- quote(stats::model.frame)
  mf$na.action <- "na.pass"
  nrows <- NROW(eval(mf, parent.frame()))
  coxcall <- call
  coxcall[[1L]] <- quote(.survivalr_clogit_coxph)
  newformula <- formula
  newformula[[2L]] <- substitute(
    Surv(rep(1, nn), case),
    list(case = formula[[2L]], nn = nrows)
  )
  environment(newformula) <- environment(formula)
  coxcall$formula <- newformula
  method <- match.arg(method)
  coxcall$method <- switch(method,
    exact = "exact",
    efron = "efron",
    "breslow"
  )
  if (is.null(coxcall$eps) && is.null(coxcall$control)) {
    coxcall$eps <- 1e-09
  }
  if (method == "exact") {
    temp <- if (missing(data)) {
      terms(formula, specials = "cluster")
    } else {
      terms(formula, specials = "cluster", data = data)
    }
    if (!is.null(attr(temp, "specials")$cluster)) {
      stop("robust variance plus the exact method is not supported", call. = FALSE)
    }
    if (!is.null(coxcall$weights)) {
      coxcall$weights <- NULL
      warning("weights ignored: not possible for the exact method", call. = FALSE)
    }
  }
  eval_env <- new.env(parent = parent.frame())
  eval_env$.survivalr_clogit_coxph <- coxph
  fit <- eval(coxcall, eval_env)
  attr(fit, "userCall") <- sys.call()
  class(fit) <- unique(c("clogit", class(fit), "coxph"))
  fit
}

tmerge <- function(data1, data2, id, ..., tstart, tstop, options) {
  call <- match.call()
  call[[1L]] <- quote(survival::tmerge)
  eval.parent(call)
}

coxsurv.fit <- function(ctype, stype, se.fit, varmat, cluster, y, x, wt, risk,
                        position, strata, oldid, y2, x2, risk2, strata2,
                        id2, unlist = TRUE) {
  call <- match.call()
  call[[1L]] <- quote(survival::coxsurv.fit)
  eval.parent(call)
}

survfitcoxph.fit <- function(y, x, wt, x2, risk, newrisk, strata, se.fit,
                             survtype, vartype, varmat, id, y2, strata2,
                             unlist = TRUE) {
  call <- match.call()
  call[[1L]] <- quote(survival::survfitcoxph.fit)
  eval.parent(call)
}

survpenal.fit <- function(x, y, weights, offset, init, controlvals, dist,
                          scale = 0, nstrat = 1, strata, pcols, pattr,
                          assign, parms = NULL) {
  call <- match.call()
  call[[1L]] <- quote(survival::survpenal.fit)
  eval.parent(call)
}

survreg.fit <- function(x, y, weights, offset, init, controlvals, dist,
                        scale = 0, nstrat = 1, strata, parms = NULL,
                        assign) {
  call <- match.call()
  call[[1L]] <- quote(survival::survreg.fit)
  eval.parent(call)
}

.as_data_frame_model_matrix <- function(x, row.names = NULL, optional = FALSE,
                                        make.names = TRUE, ...) {
  dims <- dim(x)
  nrows <- dims[[1L]]
  row.names <- dimnames(x)[[1L]]
  value <- list(x)
  if (!optional) {
    names(value) <- deparse(substitute(x))[[1L]]
  }
  class(value) <- "data.frame"
  if (!is.null(row.names)) {
    row.names <- as.character(row.names)
    if (length(row.names) != nrows) {
      stop(
        sprintf(
          ngettext(
            length(row.names),
            "supplied %d row name for %d rows",
            "supplied %d row names for %d rows"
          ),
          length(row.names),
          nrows
        ),
        domain = NA
      )
    }
    .rowNamesDF(value, make.names = make.names) <- row.names
  } else {
    attr(value, "row.names") <- .set_row_names(nrows)
  }
  value
}

as.data.frame.Surv <- function(x, row.names = NULL, optional = FALSE,
                               make.names = TRUE, ...) {
  .as_data_frame_model_matrix(
    x,
    row.names = row.names,
    optional = optional,
    make.names = make.names,
    ...
  )
}

as.data.frame.Surv2 <- function(x, row.names = NULL, optional = FALSE,
                                make.names = TRUE, ...) {
  .as_data_frame_model_matrix(
    x,
    row.names = row.names,
    optional = optional,
    make.names = make.names,
    ...
  )
}

as.matrix.Surv <- function(x, ...) {
  y <- unclass(x)
  attr(y, "type") <- NULL
  attr(y, "states") <- NULL
  attr(y, "inputAttributes") <- NULL
  y
}

as.matrix.Surv2 <- function(x, ...) {
  y <- unclass(x)
  attr(y, "states") <- NULL
  attr(y, "repeated") <- NULL
  y
}

as.matrix.survival_py_surv <- function(x, ...) {
  as.matrix.Surv(.as_native_surv(x), ...)
}

length.Surv <- function(x) {
  nrow(x)
}

length.survival_py_surv <- function(x) {
  length.Surv(.as_native_surv(x))
}

names.Surv <- function(x) {
  rownames(x)
}

names.survival_py_surv <- function(x) {
  names.Surv(.as_native_surv(x))
}

`names<-.Surv` <- function(x, value) {
  rownames(x) <- value
  x
}

`names<-.survival_py_surv` <- function(x, value) {
  native <- .as_native_surv(x)
  rownames(native) <- value
  native
}

as.logical.Surv <- function(x, ...) {
  stop("invalid operation on a survival time", call. = FALSE)
}

as.logical.Surv2 <- function(x, ...) {
  stop("invalid operation on a survival time", call. = FALSE)
}

as.logical.survival_py_surv <- function(x, ...) {
  as.logical.Surv(.as_native_surv(x), ...)
}

.as_surv_arg <- function(x) {
  if (inherits(x, "survival_py_surv")) {
    return(.as_native_surv(x))
  }
  x
}

c.Surv <- function(...) {
  slist <- lapply(list(...), .as_surv_arg)
  if (!all(vapply(slist, inherits, logical(1L), what = "Surv"))) {
    stop("all elements must be of class Surv", call. = FALSE)
  }
  types <- vapply(slist, function(x) attr(x, "type"), character(1L))
  if (!all(types == types[[1L]])) {
    stop("all elements must be of the same Surv type", call. = FALSE)
  }
  if (types[[1L]] %in% c("mright", "mcounting")) {
    states <- lapply(slist, function(x) attr(x, "states"))
    if (any(diff(vapply(states, length, integer(1L))) != 0L)) {
      stop("all elements must have the same list of states", call. = FALSE)
    }
    if (!all(vapply(states, function(x) isTRUE(all.equal(x, states[[1L]])), logical(1L)))) {
      stop("all elements must have the same list of states", call. = FALSE)
    }
  }
  new <- do.call("rbind", lapply(slist, as.matrix))
  att1 <- attributes(slist[[1L]])
  att1 <- att1[is.na(match(names(att1), c("dim", "dimnames")))]
  attributes(new) <- c(attributes(new)[c("dim", "dimnames")], att1)
  new
}

c.survival_py_surv <- function(...) {
  c.Surv(...)
}

c.Surv2 <- function(...) {
  slist <- list(...)
  if (!all(vapply(slist, inherits, logical(1L), what = "Surv2"))) {
    stop("all elements must be of class Surv2", call. = FALSE)
  }
  states <- lapply(slist, function(x) attr(x, "states"))
  if (any(diff(vapply(states, length, integer(1L))) != 0L)) {
    stop("all elements must have the same list of states", call. = FALSE)
  }
  if (!is.null(states[[1L]]) &&
      !all(vapply(states, function(x) isTRUE(all.equal(x, states[[1L]])), logical(1L)))) {
    stop("all elements must have the same list of states", call. = FALSE)
  }
  new <- do.call("rbind", lapply(slist, as.matrix))
  att1 <- attributes(slist[[1L]])
  att1 <- att1[is.na(match(names(att1), c("dim", "dimnames")))]
  attributes(new) <- c(attributes(new)[c("dim", "dimnames")], att1)
  new
}

rev.Surv <- function(x) {
  x[rev(seq_len(nrow(x)))]
}

rev.survival_py_surv <- function(x) {
  rev.Surv(.as_native_surv(x))
}

rev.Surv2 <- function(x) {
  x[rev(seq_len(nrow(x)))]
}

rep.Surv <- function(x, ...) {
  x[rep(seq_len(nrow(x)), ...)]
}

rep.survival_py_surv <- function(x, ...) {
  rep.Surv(.as_native_surv(x), ...)
}

rep.Surv2 <- function(x, ...) {
  x[rep(seq_len(nrow(x)), ...)]
}

rep.int.Surv <- function(x, ...) {
  x[rep.int(seq_len(nrow(x)), ...)]
}

rep.int.survival_py_surv <- function(x, ...) {
  rep.int.Surv(.as_native_surv(x), ...)
}

rep.int.Surv2 <- function(x, ...) {
  x[rep.int(seq_len(nrow(x)), ...)]
}

rep_len.Surv <- function(x, ...) {
  x[rep_len(seq_len(nrow(x)), ...)]
}

rep_len.survival_py_surv <- function(x, ...) {
  rep_len.Surv(.as_native_surv(x), ...)
}

rep_len.Surv2 <- function(x, ...) {
  x[rep_len(seq_len(nrow(x)), ...)]
}

t.Surv <- function(x) {
  t(as.matrix(x))
}

t.survival_py_surv <- function(x) {
  t.Surv(.as_native_surv(x))
}

t.Surv2 <- function(x) {
  t(as.matrix(x))
}

head.Surv <- function(x, ...) {
  x[utils::head(seq_len(nrow(x)), ...)]
}

head.survival_py_surv <- function(x, ...) {
  head.Surv(.as_native_surv(x), ...)
}

tail.Surv <- function(x, ...) {
  x[utils::tail(seq_len(nrow(x)), ...)]
}

tail.survival_py_surv <- function(x, ...) {
  tail.Surv(.as_native_surv(x), ...)
}

tail.Surv2 <- function(x, ...) {
  x[utils::tail(seq_len(nrow(x)), ...)]
}

anyDuplicated.Surv <- function(x, ...) {
  anyDuplicated(as.matrix(x), ...)
}

anyDuplicated.survival_py_surv <- function(x, ...) {
  anyDuplicated.Surv(.as_native_surv(x), ...)
}

anyDuplicated.Surv2 <- function(x, ...) {
  anyDuplicated(as.matrix(x), ...)
}

duplicated.Surv <- function(x, ...) {
  duplicated(as.matrix(x), ...)
}

duplicated.survival_py_surv <- function(x, ...) {
  duplicated.Surv(.as_native_surv(x), ...)
}

duplicated.Surv2 <- function(x, ...) {
  duplicated(as.matrix(x), ...)
}

unique.Surv <- function(x, ...) {
  x[!duplicated(as.matrix(x), ...)]
}

unique.survival_py_surv <- function(x, ...) {
  unique.Surv(.as_native_surv(x), ...)
}

levels.Surv <- function(x) {
  attr(x, "states")
}

levels.survival_py_surv <- function(x) {
  levels.Surv(.as_native_surv(x))
}

Math.Surv <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

Math.survival_py_surv <- function(...) {
  Math.Surv(...)
}

Math.Surv2 <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

Ops.Surv <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

Ops.survival_py_surv <- function(...) {
  Ops.Surv(...)
}

Ops.Surv2 <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

Summary.Surv <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

Summary.survival_py_surv <- function(...) {
  Summary.Surv(...)
}

Summary.Surv2 <- function(...) {
  stop("Invalid operation on a survival time", call. = FALSE)
}

xtfrm.Surv <- function(x) {
  if (attr(x, "type") == "interval") {
    temp <- ifelse(x[, 3L] == 3L, (x[, 1L] + x[, 2L]) / 2, x[, 1L])
    index <- order(temp, match(x[, 3L], c(2L, 1L, 3L, 0L)))
  } else if (attr(x, "type") == "left") {
    index <- order(x[, 1L], x[, 2L])
  } else if (ncol(x) == 2L) {
    index <- order(x[, 1L], x[, 2L] == 0L, x[, 2L])
  } else {
    index <- order(x[, 2L], x[, 3L] == 0L, x[, 3L], x[, 1L])
  }
  temp <- integer(nrow(x))
  temp[index] <- seq_len(nrow(x))
  temp[is.na(x)] <- NA_integer_
  temp
}

xtfrm.survival_py_surv <- function(x) {
  xtfrm.Surv(.as_native_surv(x))
}

.survfit_quantile_findq <- function(x, y, p, tol) {
  if (max(y, na.rm = TRUE) < min(p)) {
    return(rep(NA_real_, length(p)))
  }
  xmax <- x[[length(x)]]
  dups <- duplicated(y)
  if (any(dups)) {
    x <- x[!dups]
    y <- y[!dups]
  }
  n <- length(y)
  indx1 <- stats::approx(y + tol, seq_len(n), p, method = "constant", f = 1)$y
  indx2 <- stats::approx(y - tol, seq_len(n), p, method = "constant", f = 1)$y
  quant <- (x[indx1] + x[indx2]) / 2
  quant[p == 0] <- x[[1L]]
  if (!is.na(y[[n]])) {
    lastpt <- abs(p - y[[n]]) < tol
    if (any(lastpt)) {
      quant[lastpt] <- (x[indx1[lastpt]] + xmax) / 2
    }
  }
  quant
}

.survfit_quantile_doquant <- function(p, time, surv, upper, lower, firstx, scale, tol) {
  qq <- .survfit_quantile_findq(c(firstx, time), c(0, 1 - surv), p, tol)
  if (missing(upper) || is.null(upper) || missing(lower) || is.null(lower)) {
    return(qq / scale)
  }
  rbind(
    qq,
    .survfit_quantile_findq(c(firstx, time), c(0, 1 - lower), p, tol),
    .survfit_quantile_findq(c(firstx, time), c(0, 1 - upper), p, tol)
  ) * (1 / scale)
}

.survfit_curve_quantile <- function(curve, probs, conf.int, scale, tolerance) {
  time <- .as_numeric_vector(.result_field(curve, "time"))
  surv <- .as_numeric_vector(.result_field(curve, "estimate"))
  if (length(time) == 0L || length(surv) == 0L) {
    stop("quantile requires Kaplan-Meier survfit time and survival estimates", call. = FALSE)
  }
  lower <- .as_numeric_vector(.result_field(curve, "conf_lower"))
  upper <- .as_numeric_vector(.result_field(curve, "conf_upper"))
  if (length(lower) == 0L || length(upper) == 0L) {
    conf.int <- FALSE
  } else {
    terminal <- !is.na(surv) & surv <= 0
    lower[terminal] <- NA_real_
    upper[terminal] <- NA_real_
  }
  if (isTRUE(conf.int)) {
    return(.survfit_quantile_doquant(
      probs, time, surv, upper, lower,
      firstx = 0, scale = scale, tol = tolerance
    ))
  }
  .survfit_quantile_doquant(probs, time, surv, firstx = 0, scale = scale, tol = tolerance)
}

.survfit_curve_has_confint <- function(curve) {
  length(.as_numeric_vector(.result_field(curve, "conf_lower"))) > 0L &&
    length(.as_numeric_vector(.result_field(curve, "conf_upper"))) > 0L
}

quantile.survival_py_survfit <- function(x, probs = c(0.25, 0.5, 0.75),
                                         conf.int = TRUE, scale,
                                         tolerance = sqrt(.Machine$double.eps),
                                         ...) {
  if (any(!is.numeric(probs)) || any(is.na(probs))) {
    stop("invalid probability", call. = FALSE)
  }
  if (any(probs < 0 | probs > 1)) {
    stop("Invalid probability", call. = FALSE)
  }
  if (missing(scale)) {
    scale <- 1
  }
  pname <- format(probs * 100)

  if (is.list(x) && !inherits(x, "python.builtin.object")) {
    curves <- unclass(x)
    nstrat <- length(curves)
    qmat <- matrix(0, nstrat, length(probs), dimnames = list(names(curves), pname))
    if (isTRUE(conf.int) && !all(vapply(curves, .survfit_curve_has_confint, logical(1L)))) {
      conf.int <- FALSE
    }
    if (isTRUE(conf.int)) {
      qlower <- qupper <- qmat
      for (idx in seq_along(curves)) {
        temp <- .survfit_curve_quantile(curves[[idx]], probs, TRUE, scale, tolerance)
        qmat[idx, ] <- temp[1L, ]
        qlower[idx, ] <- temp[2L, ]
        qupper[idx, ] <- temp[3L, ]
      }
      return(list(quantile = qmat, lower = qlower, upper = qupper))
    }
    for (idx in seq_along(curves)) {
      qmat[idx, ] <- .survfit_curve_quantile(curves[[idx]], probs, FALSE, scale, tolerance)
    }
    return(qmat)
  }

  if (isTRUE(conf.int)) {
    temp <- .survfit_curve_quantile(x, probs, TRUE, scale, tolerance)
    dimnames(temp) <- list(NULL, pname)
    return(list(quantile = temp[1L, ], lower = temp[2L, ], upper = temp[3L, ]))
  }
  temp <- .survfit_curve_quantile(x, probs, FALSE, scale, tolerance)
  names(temp) <- pname
  temp
}

median.survival_py_survfit <- function(x, ...) {
  quantile.survival_py_survfit(x, probs = 0.5, conf.int = FALSE, ...)
}

coef.survival_py_survfit <- function(object, ...) {
  stop("coef method not applicable for survfit objects", call. = FALSE)
}

vcov.survival_py_survfit <- function(object, ...) {
  stop("vcov method not applicable for survfit objects", call. = FALSE)
}

confint.survival_py_survfit <- function(object, ...) {
  stop(
    paste(
      "confint method not defined for survfit objects,",
      "use quantile for confidence intervals of the median survival"
    ),
    call. = FALSE
  )
}

.survfit_plot_groups <- function(x, cumhaz = FALSE, fun) {
  has_fun <- !missing(fun)
  fun_value <- if (has_fun) fun else NULL
  transform_values <- function(values) {
    if (!has_fun) {
      return(values)
    }
    if (is.function(fun_value)) {
      return(fun_value(values))
    }
    switch(
      tolower(as.character(fun_value)[[1L]]),
      event = ,
      f = 1 - values,
      pct = values * 100,
      logpct = log(pmax(values, .Machine$double.xmin) * 100),
      log = log(pmax(values, .Machine$double.xmin)),
      cloglog = log(-log(pmax(values, .Machine$double.xmin))),
      identity = ,
      s = ,
      surv = values,
      stop("Unrecognized function argument", call. = FALSE)
    )
  }
  frame <- as.data.frame(x)
  if (!all(c("time", "surv") %in% names(frame))) {
    stop("survfit plot requires time and survival estimates", call. = FALSE)
  }
  group <- if ("strata" %in% names(frame)) {
    frame$strata
  } else if ("curve" %in% names(frame) && length(unique(frame$curve)) > 1L) {
    frame$curve
  } else {
    rep(1L, nrow(frame))
  }
  groups <- split(frame, factor(group, levels = unique(group)), drop = TRUE)
  lapply(groups, function(curve) {
    curve <- curve[order(curve$time), , drop = FALSE]
    use_cumhaz <- isTRUE(cumhaz) || (has_fun && identical(tolower(as.character(fun_value)[[1L]]), "cumhaz"))
    values <- if (use_cumhaz) {
      if ("cumhaz" %in% names(curve)) curve$cumhaz else -log(curve$surv)
    } else {
      curve$surv
    }
    lower <- upper <- NULL
    if (all(c("lower", "upper") %in% names(curve))) {
      if (use_cumhaz) {
        lower <- -log(pmax(curve$upper, .Machine$double.xmin))
        upper <- -log(pmax(curve$lower, .Machine$double.xmin))
      } else {
        lower <- curve$lower
        upper <- curve$upper
      }
    }
    initial <- if (use_cumhaz) 0 else 1
    values <- c(initial, values)
    times <- c(0, curve$time)
    if (!is.null(lower) && !is.null(upper)) {
      lower <- c(initial, lower)
      upper <- c(initial, upper)
    }
    if (has_fun && !use_cumhaz) {
      values <- transform_values(values)
      if (!is.null(lower) && !is.null(upper)) {
        lower <- transform_values(lower)
        upper <- transform_values(upper)
      }
    }
    list(time = times, value = values, lower = lower, upper = upper, frame = curve)
  })
}

.survfit_plot_endpoints <- function(groups) {
  xend <- vapply(groups, function(curve) {
    max(curve$time[is.finite(curve$time)], na.rm = TRUE)
  }, numeric(1))
  yend <- vapply(groups, function(curve) {
    finite <- is.finite(curve$value)
    if (!any(finite)) {
      return(NA_real_)
    }
    tail(curve$value[finite], 1L)
  }, numeric(1))
  list(x = unname(xend), y = unname(yend))
}

plot.survival_py_survfit <- function(x, ..., conf.int, mark.time = FALSE, pch = 3,
                                     col = 1, lty = 1, lwd = 1, cex = 1,
                                     log = FALSE, xscale = 1, yscale = 1,
                                     xlim, ylim, xmax, fun, xlab = "", ylab = "",
                                     xaxs = "r", cumhaz = FALSE) {
  conf_requested <- !missing(conf.int) && isTRUE(conf.int)
  groups <- if (missing(fun)) .survfit_plot_groups(x, cumhaz = cumhaz) else .survfit_plot_groups(x, cumhaz = cumhaz, fun = fun)
  xrange <- range(unlist(lapply(groups, `[[`, "time"), use.names = FALSE), finite = TRUE)
  y_values <- lapply(groups, `[[`, "value")
  if (conf_requested) {
    bounds <- unlist(lapply(groups, function(curve) c(curve$lower, curve$upper)), use.names = FALSE)
    if (length(bounds) > 0L) {
      y_values <- c(y_values, list(bounds))
    } else {
      warning("confidence intervals are not available for this survfit object")
    }
  }
  yrange <- range(unlist(y_values, use.names = FALSE), finite = TRUE)
  if (!missing(xmax)) {
    xrange[[2L]] <- min(xrange[[2L]], xmax)
  }
  if (!missing(xlim)) {
    xrange <- xlim
  }
  if (!missing(ylim)) {
    yrange <- ylim
  }
  graphics::plot(
    xrange / xscale,
    yrange * yscale,
    type = "n",
    log = if (isTRUE(log)) "y" else if (identical(log, FALSE)) "" else log,
    xlab = xlab,
    ylab = ylab,
    xaxs = xaxs,
    ...
  )
  lines.survival_py_survfit(
    x,
    mark.time = mark.time,
    pch = pch,
    col = col,
    lty = lty,
    lwd = lwd,
    cex = cex,
    xmax = if (missing(xmax)) NULL else xmax,
    fun = if (missing(fun)) NULL else fun,
    conf.int = conf_requested,
    cumhaz = cumhaz,
    xscale = xscale,
    yscale = yscale
  )
  invisible(.survfit_plot_endpoints(groups))
}

lines.survival_py_survfit <- function(x, type = "s", pch = 3, col = 1,
                                      lty = 1, lwd = 1, cex = 1,
                                      mark.time = FALSE, xmax, fun,
                                      conf.int = FALSE, cumhaz = FALSE,
                                      xscale = 1, yscale = 1, ...) {
  groups <- if (missing(fun) || is.null(fun)) .survfit_plot_groups(x, cumhaz = cumhaz) else .survfit_plot_groups(x, cumhaz = cumhaz, fun = fun)
  ncurve <- length(groups)
  col <- rep(col, length.out = ncurve)
  lty <- rep(lty, length.out = ncurve)
  lwd <- rep(lwd, length.out = ncurve)
  pch <- rep(pch, length.out = ncurve)
  for (idx in seq_along(groups)) {
    curve <- groups[[idx]]
    keep <- is.finite(curve$time) & is.finite(curve$value)
    if (!missing(xmax) && !is.null(xmax)) {
      keep <- keep & curve$time <= xmax
    }
    graphics::lines(
      curve$time[keep] / xscale,
      curve$value[keep] * yscale,
      type = type,
      col = col[[idx]],
      lty = lty[[idx]],
      lwd = lwd[[idx]],
      ...
    )
    if (isTRUE(conf.int)) {
      if (is.null(curve$lower) || is.null(curve$upper)) {
        warning("confidence intervals are not available for this survfit object")
      } else {
        ci_lty <- if (is.numeric(lty[[idx]])) lty[[idx]] + 1 else lty[[idx]]
        graphics::lines(
          curve$time[keep] / xscale,
          curve$lower[keep] * yscale,
          type = type,
          col = col[[idx]],
          lty = ci_lty,
          lwd = lwd[[idx]],
          ...
        )
        graphics::lines(
          curve$time[keep] / xscale,
          curve$upper[keep] * yscale,
          type = type,
          col = col[[idx]],
          lty = ci_lty,
          lwd = lwd[[idx]],
          ...
        )
      }
    }
    if (is.numeric(mark.time) || isTRUE(mark.time)) {
      mark_x <- if (is.numeric(mark.time)) {
        mark.time
      } else if ("n.censor" %in% names(curve$frame)) {
        curve$frame$time[curve$frame$n.censor > 0]
      } else {
        numeric()
      }
      if (length(mark_x) > 0L) {
        mark_y <- stats::approx(curve$time, curve$value, mark_x, method = "constant", f = 0)$y
        graphics::points(mark_x / xscale, mark_y * yscale, pch = pch[[idx]], col = col[[idx]], cex = cex)
      }
    }
  }
  invisible(.survfit_plot_endpoints(groups))
}

points.survival_py_survfit <- function(x, fun, censor = FALSE, col = 1,
                                       pch, cumhaz = FALSE, xscale = 1,
                                       yscale = 1, ...) {
  groups <- if (missing(fun)) .survfit_plot_groups(x, cumhaz = cumhaz) else .survfit_plot_groups(x, cumhaz = cumhaz, fun = fun)
  ncurve <- length(groups)
  col <- rep(col, length.out = ncurve)
  if (missing(pch)) {
    pch <- rep(1, ncurve)
  } else {
    pch <- rep(pch, length.out = ncurve)
  }
  for (idx in seq_along(groups)) {
    curve <- groups[[idx]]
    frame <- curve$frame
    keep <- if (isTRUE(censor) && "n.censor" %in% names(frame)) {
      frame$n.censor > 0
    } else if ("n.event" %in% names(frame)) {
      frame$n.event > 0
    } else {
      rep(TRUE, nrow(frame))
    }
    if (any(keep)) {
      y <- stats::approx(curve$time, curve$value, frame$time[keep], method = "constant", f = 0)$y
      graphics::points(frame$time[keep] / xscale, y * yscale, col = col[[idx]], pch = pch[[idx]], ...)
    }
  }
  invisible(NULL)
}

.survival_py_survfit_residual_matrix <- function(result) {
  resid <- .as_numeric_matrix(result$resid)
  times <- .as_numeric_vector(result$time)
  id_values <- result$id
  time_labels <- signif(times, 4)
  if (any(duplicated(time_labels))) {
    time_labels <- NULL
  }
  dimnames(resid) <- list(id = id_values, times = time_labels)
  id_name <- result$id_name
  names(dimnames(resid))[[1L]] <- if (is.null(id_name)) "" else as.character(id_name)[[1L]]
  resid
}

.survival_py_survfit_residual_frame <- function(result, resid) {
  times <- .as_numeric_vector(result$time)
  id_values <- result$id
  frame <- data.frame(
    id = rep(id_values, times = length(times)),
    time = rep(times, each = nrow(resid)),
    resid = as.numeric(resid)
  )
  curve <- result$curve
  if (!is.null(curve) && length(curve) > 0L) {
    frame$curve <- rep(as.integer(.as_numeric_vector(curve)), times = length(times))
  }
  id_name <- result$id_name
  names(frame)[[1L]] <- if (is.null(id_name)) "(id)" else as.character(id_name)[[1L]]
  frame
}

residuals.survival_py_survfit <- function(object, times, type = "pstate",
                                          collapse = FALSE, weighted = collapse,
                                          data.frame = FALSE, extra = FALSE, ...) {
  if (inherits(object, "survival.r_api.CoxSurvfitResult")) {
    stop("residuals method for coxph survival curve not found", call. = FALSE)
  }
  if (missing(times)) {
    stop("the times argument is required", call. = FALSE)
  }
  result <- .call_r_api(
    "survfit_residuals",
    object,
    times = times,
    type = type,
    collapse = collapse,
    weighted = weighted,
    `data.frame` = data.frame,
    extra = extra,
    ...
  )
  resid <- .survival_py_survfit_residual_matrix(result)
  if (isTRUE(data.frame)) {
    return(.survival_py_survfit_residual_frame(result, resid))
  }
  if (isTRUE(extra)) {
    return(list(resid = resid, curve = result$curve))
  }
  resid
}

quantile.Surv <- function(x, probs = c(0.25, 0.5, 0.75), na.rm = FALSE, ...) {
  if (!na.rm && any(is.na(x))) {
    stop("missing values and NaN's not allowed if 'na.rm' is FALSE", call. = FALSE)
  }
  if (attr(x, "type") %in% c("mright", "mcounting")) {
    stop("quantile method not defined for multiple-endpoint Surv objects", call. = FALSE)
  }
  if (na.rm) {
    x <- x[!is.na(x)]
  }
  fit <- survfit.survival_py_surv(.as_python_surv(x))
  quantile.survival_py_survfit(fit, probs = probs, ...)
}

quantile.survival_py_surv <- function(x, probs = c(0.25, 0.5, 0.75),
                                      na.rm = FALSE, ...) {
  quantile.Surv(.as_native_surv(x), probs = probs, na.rm = na.rm, ...)
}

median.Surv <- function(x, na.rm = FALSE, ...) {
  quantile.Surv(x, probs = 0.5, na.rm = na.rm, ...)
}

median.survival_py_surv <- function(x, na.rm = FALSE, ...) {
  median.Surv(.as_native_surv(x), na.rm = na.rm, ...)
}

print.Surv <- function(x, quote = FALSE, ...) {
  invisible(print(as.character.Surv(x), quote = quote, ...))
}

print.Surv2 <- function(x, quote = FALSE, ...) {
  invisible(print(as.character.Surv2(x), quote = quote, ...))
}

.formula_has_unqualified_surv_response <- function(formula) {
  inherits(formula, "formula") &&
    length(formula) >= 3L &&
    is.call(formula[[2L]]) &&
    is.name(formula[[2L]][[1L]]) &&
    identical(formula[[2L]][[1L]], quote(Surv))
}

.formula_with_native_surv_response <- function(formula, env) {
  if (!.formula_has_unqualified_surv_response(formula)) {
    return(formula)
  }

  formula_env <- environment(formula)
  if (is.null(formula_env)) {
    formula_env <- env
  }
  model_env <- new.env(parent = formula_env)
  model_env$Surv <- .native_model_frame_surv
  environment(formula) <- model_env
  formula
}

model.frame.formula <- function(formula, ...) {
  stats::model.frame.default(
    .formula_with_native_surv_response(formula, parent.frame()),
    ...
  )
}

.as_native_surv <- function(x) {
  if (inherits(x, "Surv")) {
    return(x)
  }
  if (!inherits(x, "survival_py_surv")) {
    stop("argument is not a Surv object", call. = FALSE)
  }

  surv_type <- as.character(.result_field(x, "type"))
  time <- .as_numeric_vector(.result_field(x, "time"))
  status <- as.integer(.as_nullable_numeric_vector(.result_field(x, "event")))
  start <- .result_field(x, "start")
  time2 <- .result_field(x, "time2")

  if (!is.null(start)) {
    out <- cbind(
      start = .as_numeric_vector(start),
      stop = time,
      status = status
    )
    attr(out, "type") <- "counting"
  } else if (!is.null(time2)) {
    time2 <- .as_numeric_vector(time2)
    if (identical(surv_type, "interval2")) {
      time1 <- time
      internal_time2 <- rep(1, length(time))
      right_censored <- !is.na(status) & status == 2L
      interval_censored <- !is.na(status) & status == 3L
      missing_status <- is.na(status)
      time1[right_censored] <- time2[right_censored]
      internal_time2[interval_censored] <- time2[interval_censored]
      time1[missing_status] <- NA_real_
      out <- cbind(time1 = time1, time2 = internal_time2, status = status)
      attr(out, "type") <- "interval"
    } else {
      internal_time2 <- rep(1, length(time))
      interval_censored <- !is.na(status) & status == 3L
      internal_time2[interval_censored] <- time2[interval_censored]
      out <- cbind(time1 = time, time2 = internal_time2, status = status)
      attr(out, "type") <- surv_type
    }
  } else {
    out <- cbind(time = time, status = status)
    attr(out, "type") <- surv_type
  }
  class(out) <- "Surv"
  out
}

`[.survival_py_surv` <- function(x, i, j, drop = FALSE) {
  native <- .as_native_surv(x)
  if (missing(j)) {
    if (missing(i)) {
      native[]
    } else {
      native[i]
    }
  } else if (missing(i)) {
    if (missing(drop)) native[, j] else native[, j, drop = drop]
  } else {
    if (missing(drop)) native[i, j] else native[i, j, drop = drop]
  }
}

`[.Surv` <- function(x, i, j, drop = FALSE) {
  if (missing(j)) {
    xattr <- attributes(x)
    out <- unclass(x)[i, , drop = FALSE]
    attr(out, "type") <- xattr$type
    if (!is.null(xattr$states)) {
      attr(out, "states") <- xattr$states
    }
    if (!is.null(xattr$inputAttributes)) {
      attr(out, "inputAttributes") <- lapply(xattr$inputAttributes, function(value) {
        if (any(names(value) == "names")) {
          value$names <- value$names[i]
        }
        value
      })
    }
    class(out) <- "Surv"
    out
  } else {
    class(x) <- "matrix"
    NextMethod("[")
  }
}

`[.Surv2` <- function(x, i, j, drop = FALSE) {
  if (missing(j)) {
    xattr <- attributes(x)
    out <- unclass(x)[i, , drop = FALSE]
    if (!is.null(xattr$states)) {
      attr(out, "states") <- xattr$states
    }
    attr(out, "repeated") <- xattr$repeated
    class(out) <- "Surv2"
    out
  } else {
    class(x) <- "matrix"
    NextMethod("[")
  }
}

format.survival_py_surv <- function(x, ...) {
  as.character(.call_r_api("format_surv", x, ...))
}

as.character.survival_py_surv <- function(x, ...) {
  as.character.Surv(.as_native_surv(x), ...)
}

is.na.survival_py_surv <- function(x) {
  as.logical(.call_r_api("is_na_surv", x))
}

as.character.Surv <- function(x, ...) {
  if (inherits(x, "survival_py_surv")) {
    return(as.character(.call_r_api("format_surv", x, ...)))
  }
  new <- switch(attr(x, "type"),
    right = {
      temp <- x[, 2L]
      temp <- ifelse(is.na(temp), "?", ifelse(temp == 0, "+", ""))
      paste0(format(x[, 1L]), temp)
    },
    counting = {
      temp <- x[, 3L]
      temp <- ifelse(is.na(temp), "?", ifelse(temp == 0, "+", ""))
      paste0("(", format(x[, 1L]), ",", format(x[, 2L]), temp, "]")
    },
    left = {
      temp <- x[, 2L]
      temp <- ifelse(is.na(temp), "?", ifelse(temp == 0, "-", ""))
      paste0(format(x[, 1L]), temp)
    },
    interval = {
      stat <- x[, 3L]
      temp <- c("+", "", "-", "]")[stat + 1L]
      temp2 <- ifelse(
        stat == 3L,
        paste("[", format(x[, 1L]), ", ", format(x[, 2L]), sep = ""),
        format(x[, 1L])
      )
      ifelse(is.na(stat), "NA", paste0(temp2, temp))
    },
    mright = {
      temp <- x[, 2L]
      end <- c("+", paste(":", attr(x, "states"), sep = ""))
      temp <- ifelse(is.na(temp), "?", end[temp + 1L])
      paste0(format(x[, 1L]), temp)
    },
    mcounting = {
      temp <- x[, 3L]
      end <- c("+", paste(":", attr(x, "states"), sep = ""))
      temp <- ifelse(is.na(temp), "?", end[temp + 1L])
      paste0("(", format(x[, 1L]), ",", format(x[, 2L]), temp, "]")
    },
    stop("unsupported Surv type", call. = FALSE)
  )
  names(new) <- rownames(x)
  new
}

format.Surv <- function(x, ...) {
  if (inherits(x, "survival_py_surv")) {
    return(format.survival_py_surv(x, ...))
  }
  format(as.character.Surv(x), ...)
}

is.na.Surv <- function(x) {
  if (inherits(x, "survival_py_surv")) {
    return(is.na.survival_py_surv(x))
  }
  as.vector(rowSums(is.na(unclass(x))) > 0)
}

as.character.Surv2 <- function(x, ...) {
  states <- attr(x, "states")
  status <- x[, 2L]
  suffixes <- if (is.null(states)) {
    ifelse(is.na(status), "?", ifelse(status == 0, "+", ""))
  } else {
    endings <- c("+", paste0(":", states))
    ifelse(is.na(status), "?", endings[status + 1L])
  }
  result <- paste0(format(x[, 1L]), suffixes)
  names(result) <- rownames(x)
  result
}

format.Surv2 <- function(x, ...) {
  format(as.character.Surv2(x), ...)
}

is.na.Surv2 <- function(x) {
  as.vector(rowSums(is.na(unclass(x))) > 0)
}

.aeq_adjust_time_columns <- function(columns, tolerance) {
  adjusted <- lapply(columns, as.numeric)
  finite_values <- numeric()
  finite_positions <- list()
  for (column_index in seq_along(adjusted)) {
    column <- adjusted[[column_index]]
    finite <- is.finite(column)
    if (!any(finite)) {
      next
    }
    rows <- which(finite)
    finite_values <- c(finite_values, column[rows])
    finite_positions <- c(
      finite_positions,
      lapply(rows, function(row_index) c(column_index, row_index))
    )
  }
  if (length(finite_values) == 0L) {
    return(adjusted)
  }

  result <- .data_prep_attr("aeq_surv")(finite_values, tolerance)
  adjusted_values <- as.numeric(result$time)
  for (index in seq_along(finite_positions)) {
    position <- finite_positions[[index]]
    adjusted[[position[[1L]]]][position[[2L]]] <- adjusted_values[[index]]
  }
  adjusted
}

.aeq_raise_if_zero_interval <- function(left, right, adjusted_left, adjusted_right) {
  zero_length <- !is.na(left) & !is.na(right) &
    left != right & adjusted_left == adjusted_right
  if (any(zero_length)) {
    stop("aeqSurv exception, an interval has effective length 0", call. = FALSE)
  }
}

.aeq_native_multistate_surv <- function(x, tolerance) {
  surv_type <- attr(x, "type")
  if (!is.character(surv_type) || length(surv_type) != 1L ||
      !(surv_type %in% c("mright", "mcounting"))) {
    return(NULL)
  }
  if (tolerance <= 0) {
    return(x)
  }

  out <- x
  if (identical(surv_type, "mright")) {
    adjusted <- .aeq_adjust_time_columns(list(out[, 1L]), tolerance)
    out[, 1L] <- adjusted[[1L]]
  } else {
    adjusted <- .aeq_adjust_time_columns(list(out[, 1L], out[, 2L]), tolerance)
    .aeq_raise_if_zero_interval(out[, 1L], out[, 2L], adjusted[[1L]], adjusted[[2L]])
    out[, 1L] <- adjusted[[1L]]
    out[, 2L] <- adjusted[[2L]]
  }
  out
}

aeqSurv <- function(x, tolerance = sqrt(.Machine$double.eps)) {
  if (!is.numeric(tolerance) || length(tolerance) != 1L || !is.finite(tolerance)) {
    stop("invalid value for tolerance", call. = FALSE)
  }
  native_multistate <- .aeq_native_multistate_surv(x, tolerance)
  if (!is.null(native_multistate)) {
    return(native_multistate)
  }
  .call_r_api(
    "aeqSurv",
    .as_python_surv(x),
    tolerance = tolerance,
    .wrap = c("survival_py_surv", "survival_py_object")
  )
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

.coxph_fit_covariates <- function(x, n) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (nrow(x) != n) {
    stop("x and y have different numbers of rows", call. = FALSE)
  }
  lapply(seq_len(nrow(x)), function(idx) as.list(as.numeric(x[idx, , drop = TRUE])))
}

.coxph_fit_strata <- function(strata, n) {
  if (missing(strata) || is.null(strata)) {
    return(NULL)
  }
  if (length(strata) != n) {
    stop("strata and y have different numbers of rows", call. = FALSE)
  }
  as.integer(factor(strata, levels = unique(strata)))
}

.coxph_fit_core <- function(x, y, strata, offset, init, control, weights, method,
                            rownames, resid = TRUE, nocenter = NULL,
                            include_agreg_info = FALSE, include_class = TRUE,
                            method_label = NULL, linear_predictors_matrix = FALSE) {
  if (!inherits(y, "Surv") || !is.matrix(y)) {
    stop("y must be a Surv object", call. = FALSE)
  }
  if (ncol(y) == 2L) {
    entry_times <- NULL
    time <- as.numeric(y[, 1L])
    status <- as.integer(y[, 2L])
  } else if (ncol(y) == 3L) {
    entry_times <- as.numeric(y[, 1L])
    time <- as.numeric(y[, 2L])
    status <- as.integer(y[, 3L])
  } else {
    stop("y must have 2 or 3 columns", call. = FALSE)
  }
  n <- length(time)
  covariate_matrix <- as.matrix(x)
  x_names <- colnames(covariate_matrix)
  if (is.null(x_names)) {
    x_names <- paste0("X", seq_len(ncol(covariate_matrix)))
  }
  covariates <- .coxph_fit_covariates(covariate_matrix, n)

  if (missing(offset) || is.null(offset)) {
    offset <- rep(0, n)
  }
  offset <- as.numeric(offset)
  if (length(offset) != n) {
    stop("offset and y have different numbers of rows", call. = FALSE)
  }
  if (missing(weights) || is.null(weights)) {
    weights <- rep(1, n)
  }
  weights <- as.numeric(weights)
  if (length(weights) != n) {
    stop("weights and y have different numbers of rows", call. = FALSE)
  }
  if (missing(init)) {
    init <- NULL
  }
  if (!is.null(init)) {
    init <- as.numeric(init)
  }
  if (missing(control) || is.null(control)) {
    control <- coxph.control()
  }
  if (missing(method) || is.null(method)) {
    method <- "efron"
  }
  method <- match.arg(method, c("efron", "breslow", "exact"))
  if (missing(rownames) || is.null(rownames)) {
    rownames <- rownames(y)
    if (is.null(rownames)) {
      rownames <- as.character(seq_len(n))
    }
  }
  if (length(rownames) != n) {
    stop("rownames and y have different lengths", call. = FALSE)
  }

  fit <- do.call(
    .regression_attr("coxph_fit"),
    .compact_null(list(
      time = time,
      status = status,
      covariates = covariates,
      strata = .coxph_fit_strata(strata, n),
      weights = weights,
      offset = offset,
      initial_beta = if (is.null(init)) NULL else as.list(init),
      max_iter = as.integer(control[["iter.max"]]),
      eps = as.numeric(control[["eps"]]),
      toler = as.numeric(control[["toler.chol"]]),
      method = method,
      entry_times = entry_times,
      nocenter = if (is.null(nocenter)) NULL else as.list(as.numeric(nocenter))
    ))
  )

  coefficient_matrix <- .as_numeric_matrix(.result_field(fit, "coefficients"))
  coefficients <- if (nrow(coefficient_matrix) == 0L) {
    numeric()
  } else {
    as.numeric(coefficient_matrix[1L, ])
  }
  names(coefficients) <- x_names
  means <- .as_numeric_vector(.result_field(fit, "means"))
  variance <- .as_numeric_matrix(.result_field(fit, "information_matrix"))
  loglik <- .as_numeric_vector(.result_field(fit, "log_likelihood"))
  linear_predictors <- as.numeric(covariate_matrix %*% coefficients) + offset
  if (length(means) == length(coefficients)) {
    linear_predictors <- linear_predictors - sum(means * coefficients)
  }

  if (isTRUE(linear_predictors_matrix)) {
    linear_predictors <- matrix(linear_predictors, ncol = 1L)
    rownames(linear_predictors) <- as.character(rownames)
  }
  out <- list(
    coefficients = coefficients,
    var = variance,
    loglik = loglik,
    score = as.numeric(.result_field(fit, "score_test")),
    iter = as.integer(.result_field(fit, "iterations")),
    linear.predictors = linear_predictors,
    means = means,
    method = if (is.null(method_label)) as.character(.result_field(fit, "method")) else method_label
  )
  if (isTRUE(include_class)) {
    out$class <- "coxph"
  }
  if (isTRUE(resid)) {
    martingale <- .as_numeric_vector(do.call(reticulate::py_get_attr(fit, "martingale_residuals"), list()))
    names(martingale) <- as.character(rownames)
    out$residuals <- martingale
    out <- out[c(
      "coefficients", "var", "loglik", "score", "iter",
      "linear.predictors", "residuals", "means", "method", "class"
    )[c(
      TRUE, TRUE, TRUE, TRUE, TRUE,
      TRUE, TRUE, TRUE, TRUE, "class" %in% names(out)
    )]]
  }
  if (isTRUE(include_agreg_info)) {
    names(out$means) <- x_names
    out$first <- .as_numeric_vector(.result_field(fit, "score_vector"))
    out$info <- c(rank = length(coefficients), rescale = 0L, `step halving` = 0L, convergence = 0L)
    out <- out[c(
      "coefficients", "var", "loglik", "score", "iter",
      "linear.predictors", if ("residuals" %in% names(out)) "residuals",
      "means", "first", "info", "method", "class"
    )]
  }
  out
}

coxph.fit <- function(x, y, strata, offset, init, control, weights, method,
                      rownames, resid = TRUE, nocenter = NULL) {
  .coxph_fit_core(x, y, strata, offset, init, control, weights, method,
                  rownames, resid = resid, nocenter = nocenter)
}

agreg.fit <- function(x, y, strata, offset, init, control, weights, method,
                      rownames, resid = TRUE, nocenter = NULL) {
  .coxph_fit_core(x, y, strata, offset, init, control, weights, method,
                  rownames, resid = resid, nocenter = nocenter,
                  include_agreg_info = TRUE)
}

agexact.fit <- function(x, y, strata, offset, init, control, weights, method,
                        rownames, resid = TRUE, nocenter = NULL) {
  if (!missing(weights) && !is.null(weights) && any(as.numeric(weights) != 1)) {
    stop("Case weights are not supported for the exact method", call. = FALSE)
  }
  .coxph_fit_core(x, y, strata, offset, init, control, rep(1, nrow(as.matrix(x))), "exact",
                  rownames, resid = resid, nocenter = nocenter,
                  include_class = FALSE, method_label = "coxph",
                  linear_predictors_matrix = TRUE)
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
  dots <- match.call(expand.dots = FALSE)$...
  evaluated_dots <- .eval_formula_dots(
    dots,
    data,
    parent.frame(),
    vector_args = c("weights", "id", "cluster", "group")
  )
  do.call(
    .call_r_api,
    c(
      list(
        "survfit",
        response = .as_formula_string(formula),
        data = .as_python_data(data),
        subset = subset,
        `na.action` = .as_na_action(na.action)
      ),
      evaluated_dots,
      list(.wrap = c("survival_py_survfit", "survival_py_object"))
    )
  )
}

survfit.character <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  dots <- match.call(expand.dots = FALSE)$...
  evaluated_dots <- .eval_formula_dots(
    dots,
    data,
    parent.frame(),
    vector_args = c("weights", "id", "cluster", "group")
  )
  do.call(
    .call_r_api,
    c(
      list(
        "survfit",
        response = .as_formula_string(formula),
        data = .as_python_data(data),
        subset = subset,
        `na.action` = .as_na_action(na.action)
      ),
      evaluated_dots,
      list(.wrap = c("survival_py_survfit", "survival_py_object"))
    )
  )
}

survfit.Surv <- function(formula, ..., group = NULL, subset = NULL, na.action = "fail") {
  survfit.survival_py_surv(
    .as_python_surv(formula),
    ...,
    group = group,
    subset = subset,
    na.action = na.action
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

.survfitKM_type <- function(y) {
  if (inherits(y, "survival_py_surv")) {
    y <- .as_native_surv(y)
  }
  surv_type <- attr(y, "type")
  if (is.null(surv_type)) {
    return(if (ncol(y) == 3L) "counting" else "right")
  }
  as.character(surv_type)
}

.survfitKM_computation_type <- function(stype, ctype, type) {
  if (!missing(type)) {
    if (!is.character(type)) {
      stop("type argument must be character", call. = FALSE)
    }
    matched <- charmatch(type, c("kaplan-meier", "fleming-harrington", "fh2"))
    if (is.na(matched)) {
      stop("invalid value for 'type'", call. = FALSE)
    }
    return(c(1L, 3L, 4L)[[matched]])
  }
  if (!(ctype %in% 1:2)) {
    stop("ctype must be 1 or 2", call. = FALSE)
  }
  if (!(stype %in% 1:2)) {
    stop("stype must be 1 or 2", call. = FALSE)
  }
  as.integer(2L * stype + ctype - 2L)
}

.survfitKM_robust_active <- function(y, weights, id, cluster, robust) {
  if (!is.null(robust)) {
    return(isTRUE(robust))
  }
  if (!is.null(cluster) && length(cluster) > 0L) {
    return(TRUE)
  }
  if (any(weights != floor(weights))) {
    return(TRUE)
  }
  if (!is.null(id) && length(id) > 0L) {
    if (inherits(y, "survival_py_surv")) {
      y <- .as_native_surv(y)
    }
    status <- if (inherits(y, "Surv") && is.matrix(y)) {
      y[, ncol(y)]
    } else {
      y_frame <- as.data.frame(y)
      y_frame[[ncol(y_frame)]]
    }
    event_id <- id[status == 1]
    return(anyDuplicated(event_id) > 0L)
  }
  FALSE
}

.survfitKM_right_time_status <- function(y, y.frame) {
  if (inherits(y, "Surv") && is.matrix(y)) {
    return(list(
      time = as.numeric(y[, 1L]),
      status = as.numeric(y[, ncol(y)])
    ))
  }
  if (ncol(y.frame) == 1L && inherits(y.frame[[1L]], "Surv") && is.matrix(y.frame[[1L]])) {
    y_matrix <- y.frame[[1L]]
    return(list(
      time = as.numeric(y_matrix[, 1L]),
      status = as.numeric(y_matrix[, ncol(y_matrix)])
    ))
  }
  if (all(c("time", "status") %in% names(y.frame))) {
    return(list(
      time = as.numeric(y.frame$time),
      status = as.numeric(y.frame$status)
    ))
  }
  list(
    time = as.numeric(y.frame[[1L]]),
    status = as.numeric(y.frame[[ncol(y.frame)]])
  )
}

.survfitKM_counting_time_status <- function(y, y.frame) {
  if (inherits(y, "Surv") && is.matrix(y)) {
    return(list(
      start = as.numeric(y[, 1L]),
      stop = as.numeric(y[, 2L]),
      status = as.numeric(y[, ncol(y)])
    ))
  }
  if (ncol(y.frame) == 1L && inherits(y.frame[[1L]], "Surv") && is.matrix(y.frame[[1L]])) {
    y_matrix <- y.frame[[1L]]
    return(list(
      start = as.numeric(y_matrix[, 1L]),
      stop = as.numeric(y_matrix[, 2L]),
      status = as.numeric(y_matrix[, ncol(y_matrix)])
    ))
  }
  if (all(c("start", "stop", "status") %in% names(y.frame))) {
    return(list(
      start = as.numeric(y.frame$start),
      stop = as.numeric(y.frame$stop),
      status = as.numeric(y.frame$status)
    ))
  }
  list(
    start = as.numeric(y.frame[[1L]]),
    stop = as.numeric(y.frame[[2L]]),
    status = as.numeric(y.frame[[ncol(y.frame)]])
  )
}

.survfitKM_std_err <- function(surv, std_err, std_chaz, stype, logse) {
  if (length(std_err) == 0L) {
    return(std_err)
  }
  if (as.integer(stype) != 1L) {
    return(std_chaz)
  }
  if (!isTRUE(logse)) {
    return(std_err)
  }
  out <- rep(NA_real_, length(surv))
  positive <- !is.na(surv) & surv > 0
  out[positive] <- std_err[positive] / surv[positive]
  out[!positive & !is.na(surv)] <- Inf
  out
}

.survfitKM_modified_std_low <- function(std.err, n.risk, n.event) {
  events <- n.event > 0
  if (length(events) > 0L) {
    events[[1L]] <- TRUE
  }
  if (!any(events)) {
    return(std.err)
  }
  positions <- seq_along(events)
  n.lag <- rep(n.risk[events], diff(c(positions[events], 1L + max(positions))))
  std.err * sqrt(n.lag / n.risk)
}

.survfitKM_std_low <- function(fields, conf.lower) {
  switch(
    conf.lower,
    usual = fields$std.err,
    peto = sqrt((1 - fields$surv) / fields$n.risk),
    modified = .survfitKM_modified_std_low(
      fields$std.err,
      fields$n.risk,
      fields$n.event
    )
  )
}

.survfitKM_curve_fields <- function(result, se.fit, stype, conf.type,
                                    conf.lower, conf.int, logse) {
  surv <- .as_numeric_vector(.result_field(result, "estimate"))
  fields <- list(
    time = .as_numeric_vector(.result_field(result, "time")),
    n.risk = .as_numeric_vector(.result_field(result, "n_risk")),
    n.event = .as_numeric_vector(.result_field(result, "n_event")),
    n.censor = .as_numeric_vector(.result_field(result, "n_censor")),
    surv = surv
  )
  n_enter <- .result_field(result, "n_enter")
  if (!is.null(n_enter)) {
    fields$n.enter <- .as_numeric_vector(n_enter)
  }
  n_risk_count <- .result_field(result, "n_risk_count")
  n_event_count <- .result_field(result, "n_event_count")
  n_censor_count <- .result_field(result, "n_censor_count")
  n_enter_count <- .result_field(result, "n_enter_count")
  if (!is.null(n_risk_count)) {
    fields$n.risk.count <- .as_numeric_vector(n_risk_count)
  }
  if (!is.null(n_event_count)) {
    fields$n.event.count <- .as_numeric_vector(n_event_count)
  }
  if (!is.null(n_censor_count)) {
    fields$n.censor.count <- .as_numeric_vector(n_censor_count)
  }
  if (!is.null(n_enter_count)) {
    fields$n.enter.count <- .as_numeric_vector(n_enter_count)
  }
  if (isTRUE(se.fit)) {
    std_chaz <- .as_numeric_vector(.result_field(result, "std_chaz"))
    fields$std.err <- .survfitKM_std_err(
      surv,
      .as_numeric_vector(.result_field(result, "std_err")),
      std_chaz,
      stype,
      logse
    )
    fields$cumhaz <- .as_numeric_vector(.result_field(result, "cumhaz"))
    fields$std.chaz <- std_chaz
    if (!identical(conf.type, "none")) {
      if (identical(conf.lower, "usual")) {
        lower <- .as_numeric_vector(.result_field(result, "conf_lower"))
        upper <- .as_numeric_vector(.result_field(result, "conf_upper"))
      } else {
        ci <- survfit_confint(
          surv,
          fields$std.err,
          logse = logse,
          conf.type = conf.type,
          conf.int = conf.int,
          selow = .survfitKM_std_low(fields, conf.lower)
        )
        lower <- ci$lower
        upper <- ci$upper
      }
      zero_surv <- !is.na(surv) & surv <= 0 & fields$std.err > 0
      if (isTRUE(logse) && conf.type %in% c("log", "log-log", "logit")) {
        lower[zero_surv] <- NA_real_
        upper[zero_surv] <- NA_real_
      }
      fields$lower <- lower
      fields$upper <- upper
    }
  } else {
    fields$cumhaz <- .as_numeric_vector(.result_field(result, "cumhaz"))
  }
  fields
}

.survfitKM_cbind_fields <- function(curves, name) {
  values <- lapply(curves, function(curve) curve[[name]])
  unlist(values, use.names = FALSE)
}

.survfitKM_counts_matrix <- function(fields) {
  columns <- list(
    nrisk = fields$n.risk.count,
    nevent = fields$n.event.count,
    ncensor = fields$n.censor.count
  )
  if (!is.null(fields$n.enter.count)) {
    columns$nenter <- fields$n.enter.count
  }
  matrix(
    unlist(columns, use.names = FALSE),
    ncol = length(columns),
    dimnames = list(NULL, names(columns))
  )
}

.survfitKM_influence_value <- function(influence) {
  if (is.logical(influence)) {
    influence <- if (influence) 3L else 0L
  }
  if (!(influence %in% 0:3)) {
    stop("influence argument must be 0, 1, 2, or 3", call. = FALSE)
  }
  as.integer(influence)
}

.survfitKM_influence_matrix <- function(result, name, row.labels) {
  out <- .as_numeric_matrix(.result_field(result, name))
  rownames(out) <- as.character(row.labels)
  out
}

.survfitKM_influence_fields <- function(time, status, weights, cluster,
                                        stype, ctype, conf.int, conf.type) {
  if (anyNA(cluster)) {
    stop("cluster contains missing values", call. = FALSE)
  }
  cluster_labels <- unique(cluster)
  cluster_codes <- match(cluster, cluster_labels)
  result <- .call_r_api(
    "survfitkm_influence",
    time = .as_python_vector(as.numeric(time)),
    status = .as_python_vector(as.numeric(status)),
    cluster = .as_python_vector(cluster_codes),
    weights = .as_python_vector(as.numeric(weights)),
    stype = as.integer(stype),
    ctype = as.integer(ctype),
    conf_level = conf.int,
    conf_type = conf.type
  )
  list(
    influence.surv = .survfitKM_influence_matrix(
      result,
      "influence_surv",
      cluster_labels
    ),
    influence.chaz = .survfitKM_influence_matrix(
      result,
      "influence_chaz",
      cluster_labels
    )
  )
}

.survfitKM_counting_influence_fields <- function(start, stop, status, curve.time,
                                                 curve.estimate, weights, cluster,
                                                 stype, ctype, conf.int, conf.type) {
  if (anyNA(cluster)) {
    stop("cluster contains missing values", call. = FALSE)
  }
  cluster_labels <- unique(cluster)
  cluster_codes <- match(cluster, cluster_labels)
  result <- .call_r_api(
    "survfitkm_counting_influence",
    start = .as_python_vector(as.numeric(start)),
    stop = .as_python_vector(as.numeric(stop)),
    status = .as_python_vector(as.integer(status)),
    curve_time = .as_python_vector(as.numeric(curve.time)),
    curve_estimate = .as_python_vector(as.numeric(curve.estimate)),
    cluster = .as_python_vector(cluster_codes),
    weights = .as_python_vector(as.numeric(weights)),
    stype = as.integer(stype),
    ctype = as.integer(ctype),
    conf_level = conf.int,
    conf_type = conf.type
  )
  list(
    influence.surv = .survfitKM_influence_matrix(
      result,
      "influence_surv",
      cluster_labels
    ),
    influence.chaz = .survfitKM_influence_matrix(
      result,
      "influence_chaz",
      cluster_labels
    )
  )
}

.survfitKM_add_influence <- function(fields, influence.fields, influence,
                                     chaz.first = FALSE) {
  fields$.influence_chaz_first <- isTRUE(chaz.first)
  if (influence %in% c(1L, 3L)) {
    fields$influence.surv <- influence.fields$influence.surv
  }
  if (influence %in% c(2L, 3L)) {
    fields$influence.chaz <- influence.fields$influence.chaz
  }
  fields
}

.survfitKM_apply_influence_se <- function(fields, influence.fields, stype, ctype,
                                          conf.type, conf.lower, conf.int,
                                          logse) {
  if (is.null(fields$std.err)) {
    return(fields)
  }
  surv_se <- sqrt(colSums(influence.fields$influence.surv^2))
  chaz_se <- sqrt(colSums(influence.fields$influence.chaz^2))
  fields$std.err <- if (as.integer(stype) == 1L && as.integer(ctype) == 2L && isTRUE(logse)) {
    surv_se
  } else {
    .survfitKM_std_err(fields$surv, surv_se, chaz_se, stype, logse)
  }
  fields$std.chaz <- chaz_se
  if (!identical(conf.type, "none")) {
    if (identical(conf.lower, "usual")) {
      ci <- survfit_confint(
        fields$surv,
        fields$std.err,
        logse = logse,
        conf.type = conf.type,
        conf.int = conf.int
      )
    } else {
      ci <- survfit_confint(
        fields$surv,
        fields$std.err,
        logse = logse,
        conf.type = conf.type,
        conf.int = conf.int,
        selow = .survfitKM_std_low(fields, conf.lower)
      )
    }
    zero_surv <- !is.na(fields$surv) & fields$surv <= 0 & fields$std.err > 0
    if (isTRUE(logse) && conf.type %in% c("log", "log-log", "logit")) {
      ci$lower[zero_surv] <- NA_real_
      ci$upper[zero_surv] <- NA_real_
    }
    fields$lower <- ci$lower
    fields$upper <- ci$upper
  }
  fields
}

.survfitKM_r_list <- function(fields, n, strata, n.id, curve_type, se.fit,
                              conf.int, conf.type, conf.lower, logse, add.counts,
                              t0) {
  out <- list(
    n = n,
    time = fields$time,
    n.risk = fields$n.risk,
    n.event = fields$n.event,
    n.censor = fields$n.censor,
    surv = fields$surv
  )
  if (isTRUE(se.fit)) {
    out$std.err <- fields$std.err
  }
  out$cumhaz <- fields$cumhaz
  if (isTRUE(se.fit)) {
    out$std.chaz <- fields$std.chaz
  }
  if (!is.null(strata)) {
    out$strata <- strata
  }
  if (!is.null(fields$n.enter)) {
    out$n.enter <- fields$n.enter
  }
  if (isTRUE(add.counts)) {
    out$counts <- .survfitKM_counts_matrix(fields)
  }
  if (!is.null(n.id)) {
    out$n.id <- n.id
  }
  out$type <- curve_type
  if (isTRUE(se.fit)) {
    out$logse <- logse
    out$conf.int <- conf.int
    out$conf.type <- conf.type
    if (!identical(conf.lower, "usual")) {
      out$conf.lower <- conf.lower
    }
    if (!identical(conf.type, "none")) {
      out$lower <- fields$lower
      out$upper <- fields$upper
    }
  }
  if (isTRUE(fields$.influence_chaz_first) && !is.null(fields[["influence.chaz", exact = TRUE]])) {
    out$influence.chaz <- fields$influence.chaz
  }
  if (!is.null(fields[["influence.surv", exact = TRUE]])) {
    out$influence.surv <- fields$influence.surv
  }
  if (!isTRUE(fields$.influence_chaz_first) && !is.null(fields[["influence.chaz", exact = TRUE]])) {
    out$influence.chaz <- fields$influence.chaz
  }
  out$t0 <- t0
  out
}

survfitKM <- function(x, y, weights = rep(1, length(x)), stype = 1, ctype = 1,
                      se.fit = TRUE, conf.int = 0.95,
                      conf.type = c("log", "log-log", "plain", "none", "logit", "arcsin"),
                      conf.lower = c("usual", "peto", "modified"), start.time,
                      id, cluster, robust, influence = FALSE, type,
                      entry = FALSE, time0 = FALSE) {
  if (!is.factor(x)) {
    stop("x must be a factor", call. = FALSE)
  }
  if (!inherits(y, "survival_py_surv") && !inherits(y, "Surv")) {
    stop("y must be a Surv object", call. = FALSE)
  }
  y_frame <- as.data.frame(y)
  if (length(x) != nrow(y_frame)) {
    stop("x and y have different lengths", call. = FALSE)
  }
  if (length(weights) != length(x)) {
    stop("weights and x have different lengths", call. = FALSE)
  }
  conf.type <- match.arg(conf.type)
  conf.lower <- match.arg(conf.lower)
  survfit_type <- .survfitKM_computation_type(stype, ctype, type)
  influence_value <- .survfitKM_influence_value(influence)
  if (influence_value > 0L && !missing(robust) && isFALSE(robust)) {
    warning("robust=FALSE implies influence=FALSE", call. = FALSE)
    influence_value <- 0L
  }
  robust_active <- .survfitKM_robust_active(
    y,
    weights,
    if (missing(id)) NULL else id,
    if (missing(cluster)) NULL else cluster,
    if (missing(robust)) NULL else robust
  )
  if (influence_value > 0L) {
    robust_active <- TRUE
  }
  logse <- !robust_active || survfit_type %in% c(2L, 4L)
  add_counts <- !isTRUE(all.equal(weights, rep(1, length(weights))))
  curve_type <- .survfitKM_type(y)
  if (influence_value > 0L && !(curve_type %in% c("right", "counting"))) {
    stop(
      "survfitKM influence output is currently supported only for right-censored or counting-process data",
      call. = FALSE
    )
  }
  influence_cluster <- NULL
  if (influence_value > 0L) {
    influence_cluster <- if (!missing(cluster)) {
      cluster
    } else if (!missing(id)) {
      id
    } else {
      seq_along(x)
    }
    if (length(influence_cluster) != length(x)) {
      stop("cluster and x have different lengths", call. = FALSE)
    }
  }
  y_influence <- if (influence_value > 0L && identical(curve_type, "right")) {
    .survfitKM_right_time_status(y, y_frame)
  } else if (influence_value > 0L && identical(curve_type, "counting")) {
    .survfitKM_counting_time_status(y, y_frame)
  } else {
    NULL
  }

  args <- list(
    .as_python_surv(y),
    group = if (nlevels(x) > 1L) .as_python_vector(as.character(x)) else NULL,
    weights = .as_python_vector(weights),
    stype = as.integer(stype),
    ctype = as.integer(ctype),
    `se.fit` = isTRUE(se.fit),
    `conf.int` = conf.int,
    `conf.type` = conf.type,
    id = if (missing(id)) NULL else .as_python_vector(id),
    cluster = if (missing(cluster)) NULL else .as_python_vector(cluster),
    robust = if (missing(robust)) NULL else robust,
    entry = entry,
    time0 = time0
  )
  if (!missing(start.time)) {
    args$`start.time` <- start.time
  }
  if (!missing(type)) {
    args$type <- type
  }
  result <- do.call(.python_attr("survfit"), .compact_null(args))
  t0 <- if (missing(start.time)) 0 else as.numeric(start.time)[[1L]]
  group_n <- as.integer(tabulate(as.integer(x), nbins = nlevels(x)))
  names(group_n) <- levels(x)
  group_n_id <- if (missing(id)) {
    NULL
  } else {
    vapply(levels(x), function(level) {
      length(unique(id[x == level]))
    }, integer(1))
  }

  if (nlevels(x) <= 1L) {
    fields <- .survfitKM_curve_fields(
      result,
      isTRUE(se.fit),
      stype,
      conf.type,
      conf.lower,
      conf.int,
      logse
    )
    if (influence_value > 0L) {
      influence_stype <- if (survfit_type %in% c(1L, 2L)) 1L else 2L
      influence_ctype <- if (survfit_type %in% c(1L, 3L)) 1L else 2L
      influence_fields <- if (identical(curve_type, "counting")) {
        .survfitKM_counting_influence_fields(
          y_influence$start,
          y_influence$stop,
          y_influence$status,
          fields$time,
          fields$surv,
          weights,
          influence_cluster,
          influence_stype,
          influence_ctype,
          conf.int,
          conf.type
        )
      } else {
        .survfitKM_influence_fields(
          y_influence$time,
          y_influence$status,
          weights,
          influence_cluster,
          influence_stype,
          influence_ctype,
          conf.int,
          conf.type
        )
      }
      fields <- .survfitKM_apply_influence_se(
        fields,
        influence_fields,
        influence_stype,
        influence_ctype,
        conf.type,
        conf.lower,
        conf.int,
        logse
      )
      fields <- .survfitKM_add_influence(
        fields,
        influence_fields,
        influence_value,
        chaz.first = influence_stype == 2L
      )
    }
    return(.survfitKM_r_list(
      fields,
      as.integer(nrow(y_frame)),
      NULL,
      if (is.null(group_n_id)) NULL else unname(group_n_id[[1L]]),
      curve_type,
      isTRUE(se.fit),
      conf.int,
      conf.type,
      conf.lower,
      logse,
      add_counts,
      t0
    ))
  }

  nonempty_levels <- levels(x)[group_n > 0L]
  curves <- lapply(nonempty_levels, function(level) {
    .survfitKM_curve_fields(
      result[[level]],
      isTRUE(se.fit),
      stype,
      conf.type,
      conf.lower,
      conf.int,
      logse
    )
  })
  names(curves) <- nonempty_levels
  influence_curves <- NULL
  influence_stype <- NULL
  if (influence_value > 0L) {
    influence_stype <- if (survfit_type %in% c(1L, 2L)) 1L else 2L
    influence_ctype <- if (survfit_type %in% c(1L, 3L)) 1L else 2L
    influence_curves <- lapply(nonempty_levels, function(level) {
      keep <- which(x == level)
      if (identical(curve_type, "counting")) {
        .survfitKM_counting_influence_fields(
          y_influence$start[keep],
          y_influence$stop[keep],
          y_influence$status[keep],
          curves[[level]]$time,
          curves[[level]]$surv,
          weights[keep],
          influence_cluster[keep],
          influence_stype,
          influence_ctype,
          conf.int,
          conf.type
        )
      } else {
        .survfitKM_influence_fields(
          y_influence$time[keep],
          y_influence$status[keep],
          weights[keep],
          influence_cluster[keep],
          influence_stype,
          influence_ctype,
          conf.int,
          conf.type
        )
      }
    })
    curves <- Map(function(curve, influence_curve) {
      .survfitKM_apply_influence_se(
        curve,
        influence_curve,
        influence_stype,
        influence_ctype,
        conf.type,
        conf.lower,
        conf.int,
        logse
      )
    }, curves, influence_curves)
  }
  field_names <- unique(unlist(lapply(curves, names), use.names = FALSE))
  fields <- stats::setNames(lapply(field_names, function(name) {
    .survfitKM_cbind_fields(curves, name)
  }), field_names)
  if (influence_value > 0L) {
    if (influence_value %in% c(1L, 3L)) {
      fields$influence.surv <- lapply(
        influence_curves,
        function(curve) curve$influence.surv
      )
    }
    if (influence_value %in% c(2L, 3L)) {
      fields$influence.chaz <- lapply(
        influence_curves,
        function(curve) curve$influence.chaz
      )
    }
    fields$.influence_chaz_first <- influence_stype == 2L
  }
  strata <- group_n
  strata[group_n > 0L] <- vapply(curves, function(curve) length(curve$time), integer(1))
  .survfitKM_r_list(
    fields,
    unname(group_n),
    strata,
    if (is.null(group_n_id)) NULL else unname(group_n_id),
    curve_type,
    isTRUE(se.fit),
    conf.int,
    conf.type,
    conf.lower,
    logse,
    add_counts,
    t0
  )
}

survfit0 <- function(x, ...) {
  .call_r_api(
    "survfit0",
    x,
    ...,
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

survfit_confint <- function(p, se, logse = TRUE, conf.type, conf.int = 0.95,
                            selow, ulimit = TRUE) {
  result <- .call_r_api(
    "survfit_confint",
    p = .as_python_vector(p),
    se = .as_python_vector(se),
    logse = logse,
    conf_type = conf.type,
    conf_int = conf.int,
    selow = if (missing(selow)) NULL else .as_python_vector(selow),
    ulimit = ulimit
  )
  list(
    lower = .as_numeric_vector(.result_field(result, "lower")),
    upper = .as_numeric_vector(.result_field(result, "upper"))
  )
}

pseudo <- function(fit, times, type, collapse = TRUE, data.frame = FALSE, ...) {
  times_value <- if (missing(times)) NULL else .as_python_vector(times)
  result <- .call_r_api(
    "pseudo",
    fit,
    times = times_value,
    type = if (missing(type)) NULL else type,
    collapse = collapse,
    `data.frame` = FALSE,
    ...
  )
  if (isTRUE(data.frame)) {
    return(.as_pseudo_data_frame(
      result,
      fit,
      times = if (missing(times)) NULL else times
    ))
  }
  .as_pseudo_matrix(
    result,
    fit,
    matrix_result = !missing(times) && length(times) > 1L,
    drop_single_column = TRUE,
    col.names = if (missing(times)) NULL else as.character(times)
  )
}

survdiff <- function(formula, data = NULL, ..., group = NULL, subset = NULL, na.action = "fail") {
  env <- parent.frame()
  group_values <- .eval_formula_arg(substitute(group), missing(group), data, env, vector = TRUE)
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data, env, vector = TRUE)
  .call_r_api(
    "survdiff",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    group = group_values,
    subset = subset_values,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_survdiff", "survival_py_object")
  )
}

survcheck <- function(formula, data = NULL, subset = NULL, na.action = na.pass,
                      id, istate, istate0 = "(s0)", timefix = TRUE, ...) {
  env <- parent.frame()
  id_values <- if (missing(id)) {
    NULL
  } else {
    eval(substitute(id), data, env)
  }
  istate_values <- if (missing(istate)) {
    NULL
  } else {
    eval(substitute(istate), data, env)
  }
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data, env, vector = TRUE)
  .call_r_api(
    "survcheck",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset_values,
    `na.action` = .as_na_action(na.action),
    id = if (is.null(id_values)) NULL else .as_python_vector(id_values),
    istate = if (is.null(istate_values)) NULL else .as_python_vector(istate_values),
    istate0 = istate0,
    timefix = timefix,
    ...,
    .wrap = c("survival_py_survcheck", "survival_py_object")
  )
}

rttright <- function(formula, data, weights, subset, na.action,
                     times, id, timefix = TRUE, renorm = TRUE) {
  env <- parent.frame()
  weight_values <- .eval_formula_arg(substitute(weights), missing(weights), data, env, vector = TRUE)
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data, env, vector = TRUE)
  id_values <- .eval_formula_arg(substitute(id), missing(id), data, env, vector = TRUE)
  if (if (missing(data)) .formula_has_offset(formula) else .formula_has_offset(formula, data)) {
    warning("Offset term ignored", call. = FALSE)
  }
  result <- .call_r_api(
    "rttright",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    weights = weight_values,
    subset = subset_values,
    `na.action` = if (missing(na.action)) NULL else .as_na_action(na.action),
    times = if (missing(times)) NULL else .as_python_vector(times),
    id = id_values,
    timefix = timefix,
    renorm = renorm,
    `_warn_offset` = FALSE
  )
  if (missing(times)) {
    .as_numeric_vector(result)
  } else if (length(times) == 0L) {
    matrix(numeric(), nrow = length(result), ncol = 0L)
  } else {
    .as_numeric_result(
      result,
      matrix_result = length(times) > 1L,
      drop_single_column = TRUE,
      col.names = as.character(times)
    )
  }
}

statefig <- function(layout, connect, margin = 0.03, box = TRUE, cex = 1,
                     col = 1, lwd = 1, lty = 1, bcol = col, acol = col,
                     alwd = lwd, alty = lty, offset = 0) {
  if (!is.numeric(layout)) {
    stop("layout must be a numeric vector or matrix", call. = FALSE)
  }
  if (!is.matrix(connect) || nrow(connect) != ncol(connect)) {
    stop("connect must be a square matrix", call. = FALSE)
  }
  connect_names <- dimnames(connect)
  statenames <- if (!is.null(connect_names[[1L]])) {
    connect_names[[1L]]
  } else if (!is.null(connect_names[[2L]])) {
    connect_names[[2L]]
  } else {
    stop("connect must have the state names as dimnames", call. = FALSE)
  }
  layout_value <- if (is.matrix(layout)) {
    lapply(seq_len(nrow(layout)), function(idx) as.numeric(layout[idx, ]))
  } else {
    as.numeric(layout)
  }
  connect_value <- lapply(seq_len(nrow(connect)), function(idx) as.numeric(connect[idx, ]))
  result <- .call_r_api(
    "statefig",
    layout = layout_value,
    connect = connect_value,
    states = as.list(statenames),
    margin = margin,
    box = box,
    cex = cex,
    col = col,
    lwd = lwd,
    lty = lty,
    bcol = bcol,
    acol = acol,
    alwd = alwd,
    alty = alty,
    offset = offset
  )
  coords <- .as_numeric_matrix(result[["positions"]])
  dimnames(coords) <- list(as.character(result[["states"]]), c("x", "y"))
  invisible(coords)
}

survSplit <- function(formula, data, subset, na.action = na.pass, cut,
                      start = "tstart", id, zero = 0, episode,
                      end = "tstop", event = "event", added) {
  if (missing(formula)) {
    stop("either a formula or the end and event arguments are required", call. = FALSE)
  }
  if (missing(cut)) {
    stop("cut must be supplied", call. = FALSE)
  }
  call <- match.call()
  model_formula <- formula
  if (inherits(model_formula, "formula")) {
    model_env <- new.env(parent = environment(model_formula))
    model_env$Surv <- .survsplit_model_frame_surv
    environment(model_formula) <- model_env
  }
  keep <- match(c("data", "subset"), names(call), nomatch = 0L)
  model_call <- call[c(1L, keep)]
  model_call$formula <- model_formula
  model_call$na.action <- na.action
  model_call[[1L]] <- quote(stats::model.frame)
  model_frame <- eval.parent(model_call)
  response <- stats::model.response(model_frame)
  py_response <- .as_python_surv(response)
  if (!is.Surv(py_response)) {
    stop("the model must have a Surv object as the response", call. = FALSE)
  }
  response_names <- .survsplit_response_names(formula, response)
  counting_response <- length(response_names) >= 3L
  start_name <- if (missing(start) && counting_response) response_names[[1L]] else start
  end_name <- if (missing(end) && length(response_names) >= 2L) {
    response_names[[if (counting_response) 2L else 1L]]
  } else {
    end
  }
  event_name <- if (missing(event) && length(response_names) >= 2L) {
    response_names[[if (counting_response) 3L else 2L]]
  } else {
    event
  }
  covariates <- model_frame[-1L]
  result <- .call_r_api(
    "survSplit",
    response = py_response,
    data = .as_python_data(covariates),
    cut = as.list(as.numeric(cut)),
    start = start_name,
    end = end_name,
    event = event_name,
    episode = if (missing(episode)) NULL else as.character(episode),
    id = if (missing(id)) NULL else as.character(id),
    zero = zero
  )
  .restore_r_column_classes(
    as.data.frame(result, stringsAsFactors = FALSE, optional = TRUE),
    covariates
  )
}

survcondense <- function(formula, data, subset, weights, na.action = na.pass,
                         id, start = "tstart", end = "tstop", event = "event") {
  if (missing(id)) {
    stop("id is required", call. = FALSE)
  }
  env <- parent.frame()
  id_name <- paste(deparse(substitute(id), width.cutoff = 500L), collapse = " ")
  weight_name <- if (missing(weights)) {
    NULL
  } else {
    paste(deparse(substitute(weights), width.cutoff = 500L), collapse = " ")
  }
  id_values <- .eval_formula_arg(substitute(id), missing(id), data, env, vector = TRUE)
  weight_values <- .eval_formula_arg(substitute(weights), missing(weights), data, env, vector = TRUE)
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data, env, vector = TRUE)
  result <- .call_r_api(
    "survcondense",
    formula = .as_formula_string(formula),
    data = .as_python_data(data),
    subset = subset_values,
    weights = weight_values,
    `na.action` = .as_na_action(na.action),
    id = id_values,
    start = start,
    end = end,
    event = event,
    `_id_name` = id_name,
    `_weights_name` = weight_name
  )
  frame <- as.data.frame(result, stringsAsFactors = FALSE, optional = TRUE)
  strata_columns <- grep("^strata\\(", names(frame), value = TRUE)
  for (column in strata_columns) {
    frame[[column]] <- factor(frame[[column]])
  }
  .restore_r_column_classes(frame, data)
}

coxph <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  dots <- match.call(expand.dots = FALSE)$...
  evaluated_dots <- .eval_formula_dots(
    dots,
    data,
    parent.frame(),
    vector_args = c("weights", "offset", "strata", "cluster", "id")
  )
  do.call(
    .call_r_api,
    c(
      list(
        "coxph",
        response = .as_formula_string(formula),
        data = .as_python_data(data),
        subset = subset,
        `na.action` = .as_na_action(na.action)
      ),
      evaluated_dots,
      list(.wrap = c("survival_py_coxph", "survival_py_model", "survival_py_object"))
    )
  )
}

survreg.distributions <- list(
  extreme = list(
    name = "Extreme value",
    variance = function(parm) pi^2 / 6,
    init = function(x, weights, ...) {
      mean <- sum(x * weights) / sum(weights)
      var <- sum(weights * (x - mean)^2) / sum(weights)
      c(mean + 0.572, var / 1.64)
    },
    deviance = function(y, scale, parms) {
      status <- y[, ncol(y)]
      width <- ifelse(status == 3, (y[, 2] - y[, 1]) / scale, 1)
      temp <- width / (exp(width) - 1)
      center <- ifelse(status == 3, y[, 1] - log(temp), y[, 1])
      temp3 <- (-temp) + log(1 - exp(-exp(width)))
      loglik <- ifelse(status == 1, -(1 + log(scale)), ifelse(status == 3, temp3, 0))
      list(center = center, loglik = loglik)
    },
    density = function(x, parms) {
      w <- exp(x)
      ww <- exp(-w)
      cbind(1 - ww, ww, w * ww, (1 - w), w * (w - 3) + 1)
    },
    quantile = function(p, parms) log(-log(1 - p))
  ),
  logistic = list(
    name = "Logistic",
    variance = function(parm) pi^2 / 3,
    init = function(x, weights, ...) {
      mean <- sum(x * weights) / sum(weights)
      var <- sum(weights * (x - mean)^2) / sum(weights)
      c(mean, var / 3.2)
    },
    deviance = function(y, scale, parms) {
      status <- y[, ncol(y)]
      width <- ifelse(status == 3, (y[, 2] - y[, 1]) / scale, 0)
      center <- ifelse(status == 3, (y[, 1] + y[, 2]) / 2, y[, 1])
      temp2 <- ifelse(status == 3, exp(width / 2), 2)
      temp3 <- log((temp2 - 1) / (temp2 + 1))
      loglik <- ifelse(status == 1, -log(4 * scale), ifelse(status == 3, temp3, 0))
      list(center = center, loglik = loglik)
    },
    density = function(x, parms) {
      w <- exp(x)
      cbind(
        w / (1 + w),
        1 / (1 + w),
        w / (1 + w)^2,
        (1 - w) / (1 + w),
        (w * (w - 4) + 1) / (1 + w)^2
      )
    },
    quantile = function(p, parms) log(p / (1 - p))
  ),
  gaussian = list(
    name = "Gaussian",
    variance = function(parm) 1,
    init = function(x, weights, ...) {
      mean <- sum(x * weights) / sum(weights)
      var <- sum(weights * (x - mean)^2) / sum(weights)
      c(mean, var)
    },
    deviance = function(y, scale, parms) {
      status <- y[, ncol(y)]
      width <- ifelse(status == 3, (y[, 2] - y[, 1]) / scale, 0)
      center <- ifelse(status == 3, (y[, 1] + y[, 2]) / 2, y[, 1])
      temp2 <- log(2 * stats::pnorm(width / 2) - 1)
      loglik <- ifelse(status == 1, -log(sqrt(2 * pi) * scale), ifelse(status == 3, temp2, 0))
      list(center = center, loglik = loglik)
    },
    density = function(x, parms) {
      cbind(stats::pnorm(x), stats::pnorm(-x), stats::dnorm(x), -x, x^2 - 1)
    },
    quantile = function(p, parms) stats::qnorm(p)
  ),
  weibull = list(
    name = "Weibull",
    dist = "extreme",
    trans = function(y) log(y),
    dtrans = function(y) 1 / y,
    itrans = function(x) exp(x)
  ),
  exponential = list(
    name = "Exponential",
    dist = "extreme",
    trans = function(y) log(y),
    dtrans = function(y) 1 / y,
    scale = 1,
    itrans = function(x) exp(x)
  ),
  rayleigh = list(
    name = "Rayleigh",
    dist = "extreme",
    trans = function(y) log(y),
    dtrans = function(y) 1 / y,
    itrans = function(x) exp(x),
    scale = 0.5
  ),
  loggaussian = list(
    name = "Log Normal",
    dist = "gaussian",
    trans = function(y) log(y),
    itrans = function(x) exp(x),
    dtrans = function(y) 1 / y
  ),
  lognormal = list(
    name = "Log Normal",
    dist = "gaussian",
    trans = function(y) log(y),
    itrans = function(x) exp(x),
    dtrans = function(y) 1 / y
  ),
  loglogistic = list(
    name = "Log logistic",
    dist = "logistic",
    trans = function(y) log(y),
    dtrans = function(y) 1 / y,
    itrans = function(x) exp(x)
  ),
  t = list(
    name = "Student-t",
    variance = function(df) df / (df - 2),
    parms = c(df = 4),
    init = function(x, weights, df) {
      if (df <= 2) {
        stop("Degrees of freedom must be >=3")
      }
      mean <- sum(x * weights) / sum(weights)
      var <- sum(weights * (x - mean)^2) / sum(weights)
      c(mean, var * (df - 2) / df)
    },
    deviance = function(y, scale, parms) {
      status <- y[, ncol(y)]
      width <- ifelse(status == 3, (y[, 2] - y[, 1]) / scale, 0)
      center <- ifelse(status == 3, rowMeans(y), y[, 1])
      temp2 <- log(1 - 2 * stats::pt(width / 2, df = parms))
      loglik <- ifelse(status == 1, -log(stats::dt(0, df = parms) * scale), ifelse(status == 3, temp2, 0))
      list(center = center, loglik = loglik)
    },
    density = function(x, df) {
      cbind(
        stats::pt(x, df),
        stats::pt(-x, df),
        stats::dt(x, df),
        -(df + 1) * x / (df + x^2),
        (df + 1) * (x^2 * (df + 3) / (df + x^2) - 1) / (df + x^2)
      )
    },
    quantile = function(p, df) stats::qt(p, df)
  )
)

survregDtest <- function(dlist, verbose = FALSE) {
  errlist <- NULL
  if (is.null(dlist$name)) {
    errlist <- c(errlist, "Missing a name")
  } else if (length(dlist$name) != 1L || !is.character(dlist$name)) {
    errlist <- c(errlist, "Invalid name")
  }
  if (!is.null(dlist$dist)) {
    if (!is.character(dlist$dist) || is.null(match(dlist$dist, names(survreg.distributions)))) {
      errlist <- c(errlist, "Reference distribution not found")
    } else {
      if (!is.function(dlist$trans)) {
        errlist <- c(errlist, "Missing or invalid trans component")
      }
      if (!is.function(dlist$itrans)) {
        errlist <- c(errlist, "Missing or invalid itrans component")
      }
      if (!is.function(dlist$dtrans)) {
        errlist <- c(errlist, "Missing or invalid dtrans component")
      }
    }
    if (is.null(errlist)) {
      if (!all.equal(dlist$itrans(dlist$trans(1:10)), 1:10)) {
        errlist <- c(errlist, "trans and itrans must be inverses of each other")
      }
      if (length(dlist$dtrans(1:10)) != 10L) {
        errlist <- c(errlist, "dtrans must be a 1-1 function")
      }
    }
  } else {
    if (!is.function(dlist$init)) {
      errlist <- c(errlist, "Missing or invalid init function")
    }
    if (!is.function(dlist$deviance)) {
      errlist <- c(errlist, "Missing or invalid deviance function")
    }
    if (!is.function(dlist$density)) {
      errlist <- c(errlist, "Missing or invalid density function")
    } else {
      if (is.null(dlist$parms)) {
        temp <- dlist$density(1:10 / 10)
      } else {
        temp <- dlist$density(1:10 / 10, unlist(dlist$parms))
      }
      if (!is.numeric(temp) || !is.matrix(temp) || nrow(temp) != 10L || ncol(temp) != 5L) {
        errlist <- c(errlist, "Density function must return a 5 column matrix")
      }
    }
    if (!is.function(dlist$quantile)) {
      errlist <- c(errlist, "Missing or invalid quantile function")
    }
  }
  if (is.null(errlist)) {
    TRUE
  } else if (verbose) {
    errlist
  } else {
    FALSE
  }
}

dsurvreg <- function(x, mean, scale = 1, distribution = "weibull", parms) {
  .as_numeric_vector(.call_r_api(
    "dsurvreg",
    x = .as_python_vector(x),
    mean = .as_python_vector(mean),
    scale = .as_python_vector(scale),
    distribution = distribution,
    parms = if (missing(parms)) NULL else parms
  ))
}

psurvreg <- function(q, mean, scale = 1, distribution = "weibull", parms) {
  .as_numeric_vector(.call_r_api(
    "psurvreg",
    q = .as_python_vector(q),
    mean = .as_python_vector(mean),
    scale = .as_python_vector(scale),
    distribution = distribution,
    parms = if (missing(parms)) NULL else parms
  ))
}

qsurvreg <- function(p, mean, scale = 1, distribution = "weibull", parms) {
  .as_numeric_vector(.call_r_api(
    "qsurvreg",
    p = .as_python_vector(p),
    mean = .as_python_vector(mean),
    scale = .as_python_vector(scale),
    distribution = distribution,
    parms = if (missing(parms)) NULL else parms
  ))
}

rsurvreg <- function(n, mean, scale = 1, distribution = "weibull", parms) {
  args <- .compact_null(list(
    n = .as_integer_scalar(n, "n", nonnegative = TRUE),
    mean = .as_python_vector(mean),
    scale = .as_python_vector(scale),
    distribution = distribution,
    parms = if (missing(parms)) NULL else parms
  ))
  .as_numeric_vector(do.call(.python_attr("rsurvreg"), args))
}

.survreg_distribution_name <- function(value) {
  if (is.character(value)) {
    return(value)
  }
  if (!is.list(value) || is.null(value$name)) {
    return(value)
  }
  switch(
    as.character(value$name),
    "Weibull" = "weibull",
    "Exponential" = "exponential",
    "Rayleigh" = "rayleigh",
    "Log Normal" = "lognormal",
    "Log logistic" = "loglogistic",
    "Gaussian" = "gaussian",
    "Logistic" = "logistic",
    "Student-t" = "t",
    value
  )
}

.normalize_survreg_distribution_dots <- function(dots) {
  dist_name <- if ("dist" %in% names(dots)) {
    "dist"
  } else if ("distribution" %in% names(dots)) {
    "distribution"
  } else {
    NULL
  }
  if (is.null(dist_name)) {
    return(dots)
  }
  dist_value <- dots[[dist_name]]
  if (is.list(dist_value) && !is.null(dist_value$parms) && !("parms" %in% names(dots))) {
    dots$parms <- unname(dist_value$parms)
  }
  dots[[dist_name]] <- .survreg_distribution_name(dist_value)
  dots
}

survreg <- function(formula, data = NULL, ..., subset = NULL, na.action = "fail") {
  dots <- match.call(expand.dots = FALSE)$...
  evaluated_dots <- .eval_formula_dots(
    dots,
    data,
    parent.frame(),
    vector_args = c("weights", "offset", "cluster")
  )
  evaluated_dots <- .normalize_survreg_distribution_dots(evaluated_dots)
  do.call(
    .call_r_api,
    c(
      list(
        "survreg",
        response = .as_formula_string(formula),
        data = .as_python_data(data),
        subset = subset,
        `na.action` = .as_na_action(na.action)
      ),
      evaluated_dots,
      list(.wrap = c("survival_py_survreg", "survival_py_model", "survival_py_object"))
    )
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

.as_brier_result <- function(result) {
  out <- list(
    rsquared = .as_numeric_vector(result[["rsquared"]]),
    brier = .as_numeric_vector(result[["brier"]]),
    times = .as_numeric_vector(result[["times"]])
  )
  if (!is.null(result[["p0"]])) {
    out$p0 <- .as_numeric_vector(result[["p0"]])
  }
  if (!is.null(result[["phat"]])) {
    out$phat <- .as_numeric_matrix(result[["phat"]])
    colnames(out$phat) <- as.character(seq_len(ncol(out$phat)))
  }
  if (!is.null(result[["eff.n"]])) {
    out$eff.n <- .as_numeric_vector(result[["eff.n"]])
  }
  out
}

brier <- function(fit, times, newdata, ties = TRUE, detail = FALSE,
                  timefix = TRUE, efron = FALSE) {
  .as_brier_result(.call_r_api(
    "brier",
    fit,
    times = if (missing(times)) NULL else .as_python_vector(times),
    newdata = if (missing(newdata)) NULL else .as_python_data(newdata),
    ties = ties,
    detail = detail,
    timefix = timefix,
    efron = efron
  ))
}

concordance <- function(object, ..., formula) {
  if (missing(object)) {
    if (!missing(formula)) {
      return(concordance(formula, ...))
    }
  }
  UseMethod("concordance")
}

concordance.default <- function(object, data = NULL, ..., scores = NULL, risk.scores = NULL,
                                weights = NULL, cluster = NULL, subset = NULL, na.action = "fail") {
  formula <- object
  env <- parent.frame()
  score_values <- .eval_formula_arg(substitute(scores), missing(scores), data, env, vector = TRUE)
  risk_score_values <- .eval_formula_arg(
    substitute(risk.scores),
    missing(risk.scores),
    data,
    env,
    vector = TRUE
  )
  weight_values <- .eval_formula_arg(substitute(weights), missing(weights), data, env, vector = TRUE)
  cluster_values <- .eval_formula_arg(substitute(cluster), missing(cluster), data, env, vector = TRUE)
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data, env, vector = TRUE)
  .call_r_api(
    "concordance",
    response = .as_formula_string(formula),
    data = .as_python_data(data),
    scores = score_values,
    risk_scores = risk_score_values,
    weights = weight_values,
    cluster = cluster_values,
    subset = subset_values,
    `na.action` = .as_na_action(na.action),
    ...,
    .wrap = c("survival_py_concordance", "survival_py_object")
  )
}

.model_concordance_response <- function(object, newdata) {
  if (missing(newdata)) {
    response <- object$y_response
    if (inherits(response, "python.builtin.object")) {
      response <- .wrap_python(response, c("survival_py_surv", "survival_py_object"))
    }
    return(.as_native_surv(response))
  }

  frame <- stats::model.frame(stats::terms(object), data = newdata)
  response <- stats::model.response(frame)
  if (is.Surv(response)) {
    return(response)
  }
  if (is.numeric(response) && is.vector(response)) {
    return(Surv(response))
  }
  stop("left hand side of the formula must be a numeric vector or a survival object", call. = FALSE)
}

concordance.survival_py_model <- function(object, ..., newdata, cluster, ymin, ymax,
                                          timewt = c("n", "S", "S/G", "n/G2", "I"),
                                          influence = 0, ranks = FALSE, timefix = TRUE,
                                          keepstrata = 10) {
  extra <- list(...)
  if (length(extra) > 0L) {
    stop("multiple fitted model concordance is not implemented", call. = FALSE)
  }
  timewt <- match.arg(timewt)
  response <- if (missing(newdata)) {
    .model_concordance_response(object)
  } else {
    .model_concordance_response(object, newdata)
  }
  scores <- if (missing(newdata)) {
    predict(object, type = "lp")
  } else {
    predict(object, newdata = newdata, type = "lp")
  }
  fit_args <- list(
    y = response,
    x = scores,
    timewt = timewt,
    influence = influence,
    ranks = ranks,
    reverse = inherits(object, "survival_py_coxph"),
    timefix = timefix,
    keepstrata = keepstrata
  )
  if (!missing(cluster)) {
    fit_args$cluster <- cluster
  }
  if (!missing(ymin)) {
    fit_args$ymin <- ymin
  }
  if (!missing(ymax)) {
    fit_args$ymax <- ymax
  }
  result <- do.call(concordancefit, fit_args)
  result$call <- match.call()
  class(result) <- "concordance"
  result
}

coef.survival_py_concordance <- function(object, ...) {
  frame <- as.data.frame(object)
  values <- as.numeric(frame$concordance)
  if (nrow(frame) > 1L) {
    names(values) <- as.character(frame$score)
  }
  values
}

vcov.survival_py_concordance <- function(object, ...) {
  frame <- as.data.frame(object)
  if (nrow(frame) == 1L) {
    return(4 * as.numeric(frame$variance)[[1L]])
  }
  dfbeta <- .result_field(object, "dfbeta")
  if (is.null(dfbeta)) {
    values <- 4 * as.numeric(frame$variance)
    return(diag(values, nrow = length(values)))
  }
  dfbeta_columns <- lapply(dfbeta, .as_numeric_vector)
  n_obs <- length(dfbeta_columns[[1L]])
  if (any(vapply(dfbeta_columns, length, integer(1)) != n_obs)) {
    stop("concordance dfbeta values must be rectangular", call. = FALSE)
  }
  dfbeta_matrix <- do.call(cbind, dfbeta_columns)
  4 * crossprod(dfbeta_matrix)
}

coef.concordance <- function(object, ...) {
  object$concordance
}

vcov.concordance <- function(object, ...) {
  object$var
}

print.concordance <- function(x, digits = max(1L, getOption("digits") - 3L), ...) {
  if (!is.null(call <- x$call)) {
    cat("Call:\n")
    dput(call)
    cat("\n")
  }
  omit <- x$na.action
  if (length(omit)) {
    cat("n=", x$n, " (", stats::naprint(omit), ")\n", sep = "")
  } else {
    cat("n=", x$n, "\n")
  }
  if (is.null(x$var)) {
    cat("Concordance= ", format(x$concordance, digits = digits), "\n")
  } else if (length(x$concordance) > 1L) {
    table <- cbind(concordance = x$concordance, se = sqrt(diag(x$var)))
    print(round(table, digits = digits), ...)
    cat("\n")
  } else {
    cat(
      "Concordance= ",
      format(x$concordance, digits = digits),
      " se= ",
      format(sqrt(x$var), digits = digits),
      "\n",
      sep = ""
    )
  }
  if (!is.matrix(x$count) || nrow(x$count) < 11L) {
    print(round(x$count, 2))
  }
  invisible(x)
}

royston <- function(fit, newdata, ties = TRUE, adjust = FALSE) {
  result <- .call_r_api(
    "royston",
    fit,
    newdata = if (missing(newdata)) NULL else .as_python_data(newdata),
    ties = ties,
    adjust = adjust
  )
  values <- as.numeric(unlist(result, use.names = FALSE))
  names(values) <- names(result)
  values
}

survConcordance <- function(formula, data, weights, subset, na.action) {
  .Deprecated("concordance")
  env <- parent.frame()
  data_values <- if (missing(data)) NULL else data
  weight_values <- .eval_formula_arg(substitute(weights), missing(weights), data_values, env, vector = TRUE)
  subset_values <- .eval_formula_arg(substitute(subset), missing(subset), data_values, env, vector = TRUE)
  .call_r_api(
    "survConcordance",
    formula = .as_formula_string(formula),
    data = .as_python_data(data_values),
    weights = weight_values,
    subset = subset_values,
    `na.action` = if (missing(na.action)) "fail" else .as_na_action(na.action),
    .wrap = c("survival_py_concordance", "survival_py_object")
  )
}

survConcordance.fit <- function(y, x, strata, weight) {
  .Deprecated("concordance")
  stats <- .call_r_api(
    "survConcordance_fit",
    .as_python_surv(y),
    .as_python_vector(x),
    strata = if (missing(strata)) NULL else .as_python_vector(strata),
    weight = if (missing(weight)) NULL else .as_python_vector(weight)
  )
  values <- as.numeric(unlist(stats, use.names = FALSE))
  names(values) <- names(stats)
  values
}

.concordancefit_count <- function(result) {
  concordant <- .as_numeric_vector(.result_field(result, "concordant"))
  comparable <- .as_numeric_vector(.result_field(result, "comparable"))
  tied_x <- .as_numeric_vector(.result_field(result, "tied_x"))
  tied_y <- .as_numeric_vector(.result_field(result, "tied_y"))
  tied_xy <- .as_numeric_vector(.result_field(result, "tied_xy"))
  if (length(tied_x) == 0L) tied_x <- rep(0, length(concordant))
  if (length(tied_y) == 0L) tied_y <- rep(0, length(concordant))
  if (length(tied_xy) == 0L) tied_xy <- rep(0, length(concordant))
  strict_concordant <- concordant - 0.5 * tied_x
  discordant <- pmax(comparable - strict_concordant - tied_x, 0)
  count <- cbind(
    concordant = strict_concordant,
    discordant = discordant,
    tied.x = tied_x,
    tied.y = tied_y,
    tied.xy = tied_xy
  )
  if (nrow(count) == 1L) {
    out <- as.numeric(count[1L, ])
    names(out) <- colnames(count)
    return(out)
  }
  rownames(count) <- .result_field(result, "score_names")
  count
}

.concordancefit_rows <- function(rows) {
  if (is.null(rows) || !is.list(rows) || length(rows) == 0L) {
    return(NULL)
  }
  data.frame(
    time = .as_numeric_vector(lapply(rows, `[[`, "time")),
    rank = .as_numeric_vector(lapply(rows, `[[`, "rank")),
    timewt = .as_numeric_vector(lapply(rows, `[[`, "timewt")),
    casewt = .as_numeric_vector(lapply(rows, `[[`, "casewt"))
  )
}

concordancefit <- function(y, x, strata, weights, ymin = NULL, ymax = NULL,
                           timewt = c("n", "S", "S/G", "n/G2", "I"),
                           cluster, influence = 0, ranks = FALSE,
                           reverse = FALSE, timefix = TRUE, keepstrata = 10,
                           std.err = TRUE) {
  if (any(is.na(x)) || any(is.na(y))) {
    return(NULL)
  }
  timewt <- match.arg(timewt)
  influence <- as.integer(influence)
  if (!isTRUE(std.err)) {
    ranks <- FALSE
    influence <- 0L
  }
  internal_influence <- if (isTRUE(std.err) && influence == 0L) 1L else influence
  result <- .call_r_api(
    "concordance",
    .as_python_surv(y),
    scores = .as_python_optional_vector(x),
    strata = if (missing(strata)) NULL else .as_python_vector(strata),
    weights = if (missing(weights)) NULL else .as_python_vector(weights),
    ymin = ymin,
    ymax = ymax,
    timewt = timewt,
    cluster = if (missing(cluster)) NULL else .as_python_vector(cluster),
    influence = internal_influence,
    ranks = ranks,
    reverse = !isTRUE(reverse),
    timefix = timefix,
    keepstrata = keepstrata
  )

  concordance <- .as_numeric_vector(.result_field(result, "concordance"))
  out <- list(
    concordance = if (length(concordance) == 1L) concordance[[1L]] else concordance,
    count = .concordancefit_count(result),
    n = as.integer(.result_field(result, "n"))
  )
  variance <- .result_field(result, "variance")
  if (isTRUE(std.err) && !is.null(variance)) {
    out$var <- 4 * .as_numeric_vector(variance)
    out$cvar <- out$var
  }
  if (influence %in% c(1L, 3L)) {
    dfbeta <- .result_field(result, "dfbeta")
    if (!is.null(dfbeta)) {
      out$dfbeta <- 2 * .as_numeric_vector(dfbeta)
    }
  }
  if (influence >= 2L) {
    influence_rows <- .result_field(result, "influence")
    if (!is.null(influence_rows)) {
      out$influence <- 2 * .as_numeric_matrix(influence_rows)
      colnames(out$influence) <- c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")
    }
  }
  if (isTRUE(ranks)) {
    out$ranks <- .concordancefit_rows(.result_field(result, "ranks"))
  }
  out
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

coxph.wtest <- function(var, b, toler.chol = 1e-09) {
  result <- .call_r_api(
    "coxph_wtest",
    var = var,
    b = b,
    toler_chol = toler.chol
  )
  solve_value <- .result_field(result, "solve")
  if (is.matrix(b) && length(dim(b)) == 2L && ncol(b) > 1L && !is.null(solve_value)) {
    solve_value <- .as_numeric_matrix(solve_value)
  } else if (!is.null(solve_value)) {
    solve_value <- .as_numeric_vector(solve_value)
  }
  list(
    test = .as_numeric_vector(.result_field(result, "test")),
    df = as.integer(.result_field(result, "df")),
    solve = solve_value
  )
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
  result <- structure(
    value,
    df = as.integer(.call_r_api("degrees_freedom", object)),
    class = "logLik"
  )
  if (inherits(object, "survival_py_coxph")) {
    attr(result, "nobs") <- as.integer(.call_r_api("nobs", object))
  }
  result
}

deviance.survival_py_model <- function(object, ...) {
  NULL
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

labels.survival_py_model <- function(object, ...) {
  attr(stats::terms(object), "term.labels")
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

model.frame.survival_py_survfit <- function(formula, ...) {
  object <- formula
  if (.is_grouped_survival_py_survfit(object)) {
    if (length(object) == 0L) {
      stop("model.frame requires a non-empty grouped survfit result", call. = FALSE)
    }
    object <- unclass(object)[[1L]]
  }
  columns <- .call_r_api("model_frame", object, ...)
  as.data.frame(columns, ...)
}

fitted.survival_py_model <- function(object, ..., type = NULL, se.fit = FALSE) {
  dots <- list(...)
  result <- .call_r_api(
    "fitted",
    object,
    type = type,
    `se.fit` = se.fit,
    ...
  )
  value <- .as_prediction_result(
    result,
    matrix_result = .predict_matrix_result(type),
    col.names = .predict_column_names(object, type, terms = dots[["terms"]])
  )
  .attach_term_prediction_constant(value, object, type, reference = dots[["reference"]])
}

summary.survival_py_model <- function(object, conf.int = 0.95, scale = 1, ...) {
  result <- .call_r_api("model_summary", object)
  model_type <- as.character(result$model_type)[[1L]]
  robust <- length(result$robust) > 0L && isTRUE(as.logical(result$robust)[[1L]])
  coefficient_table <- .as_coefficient_table(
    result$coefficients,
    model_type = model_type,
    robust = robust,
    scale = scale
  )
  if (identical(model_type, "survreg")) {
    location_coefficients <- result$location_coefficients
    location_names <- result$location_coefficient_names
    if (is.null(location_coefficients) || is.null(location_names)) {
      location_coefficients <- coef(object)
      location_names <- names(location_coefficients)
    }
    location_coefficients <- .as_numeric_vector(location_coefficients)
    location_names <- as.character(location_names)
    if (length(location_coefficients) != length(location_names)) {
      stop("survreg summary coefficient names must match its location coefficients", call. = FALSE)
    }
    names(location_coefficients) <- location_names
    result$coefficients <- location_coefficients
    result$table <- coefficient_table

    scales <- result$scales
    if (is.null(scales)) {
      scales <- result$scale
    }
    scales <- .as_numeric_vector(scales)
    if (length(scales) > 1L) {
      scale_rows <- seq.int(length(location_coefficients) + 1L, nrow(coefficient_table))
      if (length(scale_rows) != length(scales)) {
        stop("survreg summary scale rows must match its fitted scales", call. = FALSE)
      }
      names(scales) <- rownames(coefficient_table)[scale_rows]
      result$scale <- scales
    } else if (length(scales) == 1L) {
      result$scale <- scales[[1L]]
    }
    result$scales <- scales
  } else {
    if (nrow(coefficient_table) == 0L) {
      result$coefficients <- NULL
      result$used.robust <- NULL
    } else {
      result$coefficients <- coefficient_table
      result$used.robust <- robust
      if (conf.int) {
        result$conf.int <- .as_cox_confint_table(coefficient_table, conf.int)
      } else {
        result$conf.int <- NULL
      }

      null_loglik <- as.numeric(result$null_loglik)[[1L]]
      full_loglik <- as.numeric(result$loglik)[[1L]]
      result$loglik <- c(null_loglik, full_loglik)
      result$nevent <- as.numeric(result$n_event)[[1L]]

      df <- as.integer(result$df)[[1L]]
      likelihood_test <- -2 * (null_loglik - full_loglik)
      result$logtest <- .cox_summary_test(likelihood_test, df)
      result$sctest <- .cox_summary_test(result$score_test, df)
      result$rsq <- c(
        rsq = 1 - exp(-likelihood_test / result$n),
        maxrsq = 1 - exp(2 * null_loglik / result$n)
      )

      unscaled_coefficients <- stats::coef(object)
      keep <- !is.na(unscaled_coefficients)
      if (df > 0L && any(keep)) {
        active_variance <- stats::vcov(object, complete = TRUE)[keep, keep, drop = FALSE]
        wald <- coxph.wtest(
          active_variance,
          unname(as.list(unscaled_coefficients[keep]))
        )
        wald_test <- wald$test
      } else {
        wald_test <- 0
      }
      result$waldtest <- .cox_summary_test(wald_test, df, round.test = TRUE)
    }
  }
  class(result) <- c("summary.survival_py_model", class(result))
  result
}

predict.survival_py_model <- function(object, newdata = NULL, ..., type = NULL, se.fit = FALSE) {
  dots <- list(...)
  result <- .call_r_api(
    "predict",
    object,
    newdata = .as_python_data(newdata),
    type = type,
    `se.fit` = se.fit,
    ...
  )
  value <- .as_prediction_result(
    result,
    matrix_result = .predict_matrix_result(type),
    col.names = .predict_column_names(object, type, terms = dots[["terms"]])
  )
  .attach_term_prediction_constant(value, object, type, reference = dots[["reference"]])
}

residuals.survival_py_model <- function(object, ..., type = "martingale") {
  dots <- list(...)
  result <- .call_r_api("residuals", object, type = type, ...)
  .as_residual_result(object, result, type, terms = dots[["terms"]])
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

summary.survival_py_survfit <- function(object, times, censored = FALSE, scale = 1,
                                        extend = FALSE, rmean = getOption("survfit.rmean"),
                                        data.frame = FALSE, dosum, ...) {
  .survival_py_survfit_summary_frame(
    object,
    times = times,
    censored = censored,
    scale = scale,
    extend = extend,
    data.frame = data.frame,
    dosum = dosum,
    ...
  )
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

.is_grouped_survival_py_survfit <- function(x) {
  is.list(x) && !inherits(x, "python.builtin.object")
}

.survival_py_survfit_field_alias <- function(name) {
  aliases <- c(
    estimate = "surv",
    n_risk = "n.risk",
    n_event = "n.event",
    n_censor = "n.censor",
    n_enter = "n.enter",
    std_err = "std.err",
    std_chaz = "std.chaz",
    conf_lower = "lower",
    conf_upper = "upper",
    cumulative_hazard = "cumhaz",
    cumulative_hazard_std_err = "std.chaz"
  )
  if (length(name) == 1L && !is.na(name) && name %in% names(aliases)) {
    aliases[[name]]
  } else {
    name
  }
}

.as_survival_py_survfit_list <- function(x) {
  if (.is_grouped_survival_py_survfit(x)) {
    return(unclass(x))
  }
  as.list(as.data.frame.survival_py_survfit(x, optional = TRUE))
}

.as_survival_py_survfit_curve <- function(x) {
  class(x) <- unique(c("survival_py_survfit", "survival_py_object", class(x)))
  x
}

.survival_py_survfit_group_indices <- function(i, targets) {
  if (is.character(i)) {
    idx <- match(i, targets)
    if (any(is.na(idx))) {
      stop(
        paste("strata", paste(i[is.na(idx)], collapse = " "), "not matched"),
        call. = FALSE
      )
    }
    return(idx)
  }
  idx <- seq_along(targets)[i]
  if (any(is.na(idx))) {
    stop(
      paste("strata", paste(i[is.na(idx)], collapse = " "), "not matched"),
      call. = FALSE
    )
  }
  idx
}

.survival_py_survfit_aggregate_groups <- function(by, data_count) {
  if (is.null(by)) {
    return(NULL)
  }
  by_list <- if (is.list(by)) by else list(by)
  if (length(by_list) == 0L) {
    stop("arguments must have the same length", call. = FALSE)
  }
  lengths <- vapply(by_list, length, integer(1))
  if (any(lengths != data_count)) {
    stop("arguments must have the same length", call. = FALSE)
  }
  if (length(by_list) == 1L) {
    groups <- as.integer(as.factor(by_list[[1L]]))
  } else {
    group_frame <- as.data.frame(by_list, stringsAsFactors = TRUE)
    groups <- as.integer(interaction(group_frame, drop = TRUE))
  }
  if (all(groups == groups[[1L]])) {
    NULL
  } else {
    groups
  }
}

aggregate.survival_py_survfit <- function(x, by = NULL, FUN = mean, ...) {
  dims <- dim(x)
  data_count <- dims["data"]
  if (is.null(data_count) || is.na(data_count)) {
    stop("survfit object does not have a 'data' margin", call. = FALSE)
  }
  if (!identical(FUN, mean)) {
    stop("FUN must be mean for Python-backed survfit objects", call. = FALSE)
  }
  if (length(list(...)) > 0L) {
    stop("additional FUN arguments are not supported", call. = FALSE)
  }
  .call_r_api(
    "aggregate_survfit_result",
    x,
    groups = .survival_py_survfit_aggregate_groups(by, data_count),
    .wrap = c("survival_py_survfit", "survival_py_object")
  )
}

as.list.survival_py_survfit <- function(x, ...) {
  .as_survival_py_survfit_list(x)
}

names.survival_py_survfit <- function(x) {
  names(.as_survival_py_survfit_list(x))
}

length.survival_py_survfit <- function(x) {
  length(.as_survival_py_survfit_list(x))
}

dim.survival_py_survfit <- function(x) {
  if (.is_grouped_survival_py_survfit(x)) {
    return(stats::setNames(length(x), "strata"))
  }

  frame <- as.data.frame.survival_py_survfit(x, optional = TRUE)
  if ("strata" %in% names(frame)) {
    strata_count <- length(unique(frame$strata))
    if (strata_count > 0L) {
      return(stats::setNames(strata_count, "strata"))
    }
  }
  if ("curve" %in% names(frame)) {
    curve_count <- length(unique(frame$curve))
    if (curve_count > 1L) {
      return(stats::setNames(curve_count, "data"))
    }
  }
  NULL
}

`[.survival_py_survfit` <- function(x, i, ..., drop = TRUE) {
  if (length(list(...)) > 0L) {
    stop("incorrect number of dimensions", call. = FALSE)
  }
  if (missing(i)) {
    return(x)
  }
  if (.is_grouped_survival_py_survfit(x)) {
    curves <- unclass(x)
    idx <- .survival_py_survfit_group_indices(i, names(curves))
    selected <- curves[idx]
    if (isTRUE(drop) && length(selected) == 1L) {
      return(.as_survival_py_survfit_curve(selected[[1L]]))
    }
    return(structure(
      selected,
      class = c("survival_py_survfit", "survival_py_object", "list")
    ))
  }
  if (isTRUE(all(i == 1))) {
    return(x)
  }
  stop("subscript out of bounds", call. = FALSE)
}

`[[.survival_py_survfit` <- function(x, i, ..., exact = TRUE) {
  fields <- .as_survival_py_survfit_list(x)
  if (is.character(i) && length(i) == 1L && !(i %in% names(fields))) {
    i <- .survival_py_survfit_field_alias(i)
  }
  fields[[i, ..., exact = exact]]
}

`$.survival_py_survfit` <- function(x, name) {
  fields <- .as_survival_py_survfit_list(x)
  if (!(name %in% names(fields))) {
    name <- .survival_py_survfit_field_alias(name)
  }
  fields[[name, exact = FALSE]]
}

as.data.frame.survival_py_survfit <- function(x, row.names = NULL, optional = FALSE, ...) {
  .as_r_data_frame(x, row.names = row.names, optional = optional, ...)
}

as.data.frame.survival_py_surv <- function(x, row.names = NULL, optional = FALSE, ...) {
  as.data.frame.Surv(.as_native_surv(x), row.names = row.names, optional = optional, ...)
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
  print(.as_native_surv(x), ...)
  invisible(x)
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
  metadata <- .call_r_api("model_summary", x)
  cat(
    "\nlogLik=", format(as.numeric(likelihood)),
    " df=", attr(likelihood, "df"),
    " n=", metadata$n,
    sep = ""
  )
  if (inherits(x, "survival_py_coxph")) {
    cat(" events=", metadata$n_event, sep = "")
  }
  cat("\n")
  invisible(x)
}

print.survival_py_object <- function(x, ...) {
  if (!inherits(x, "python.builtin.object")) {
    return(NextMethod())
  }
  cat(as.character(reticulate::py_str(x)), "\n")
  invisible(x)
}

print.summary.survival_py_model <- function(
    x,
    digits = max(getOption("digits") - 3, 3),
    signif.stars = getOption("show.signif.stars"),
    expand = FALSE,
    ...) {
  if (identical(x$model_type, "coxph")) {
    if (!is.null(x$call)) {
      cat("Call:\n")
      dput(x$call)
      cat("\n")
    }
    if (!is.null(x$fail)) {
      cat(" Coxreg failed.", x$fail, "\n")
      return(invisible(x))
    }

    saved_digits <- options(digits = digits)
    on.exit(options(saved_digits))
    cat("  n=", x$n)
    if (!is.null(x$nevent)) {
      cat(", number of events=", x$nevent, "\n")
    } else {
      cat("\n")
    }
    if (is.null(x$coefficients)) {
      cat("   Null model\n")
      return(invisible(x))
    }

    cat("\n")
    stats::printCoefmat(
      x$coefficients,
      digits = digits,
      signif.stars = signif.stars,
      ...
    )
    if (!is.null(x$conf.int)) {
      cat("\n")
      print(x$conf.int)
    }

    p_digits <- max(1, getOption("digits") - 4)
    cat("\n")
    cat(
      "Likelihood ratio test= ",
      format(round(x$logtest["test"], 2)),
      "  on ", x$logtest["df"], " df,",
      "   p=", format.pval(x$logtest["pvalue"], digits = p_digits),
      "\n",
      sep = ""
    )
    cat(
      "Wald test            = ",
      format(round(x$waldtest["test"], 2)),
      "  on ", x$waldtest["df"], " df,",
      "   p=", format.pval(x$waldtest["pvalue"], digits = p_digits),
      "\n",
      sep = ""
    )
    cat(
      "Score (logrank) test = ",
      format(round(x$sctest["test"], 2)),
      "  on ", x$sctest["df"], " df,",
      "   p=", format.pval(x$sctest["pvalue"], digits = p_digits),
      "\n\n",
      sep = ""
    )
    if (isTRUE(x$used.robust)) {
      cat(
        "  (Note: the likelihood ratio and score tests assume independence of\n",
        "     observations within a cluster, the Wald and robust score tests do not).\n",
        sep = ""
      )
    }
    return(invisible(x))
  }

  cat(x$model_type, "model summary\n", sep = "")
  coefficient_table <- if (identical(x$model_type, "survreg") && !is.null(x$table)) {
    x$table
  } else {
    x$coefficients
  }
  print(coefficient_table, digits = digits, ...)
  cat("logLik=", x$loglik, " df=", x$df, " n=", x$n, "\n", sep = "")
  invisible(x)
}
