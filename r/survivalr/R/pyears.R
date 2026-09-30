.pyears_formula_group_info <- function(term_labels, data, data_frame, category_levels) {
  if (length(term_labels) == 0L) {
    return(NULL)
  }
  values <- data[term_labels]
  levels <- category_levels
  names(levels) <- term_labels
  grid <- if (isTRUE(data_frame)) {
    columns <- Map(function(value, labels) {
      if (inherits(value, "tcut")) {
        if (is.numeric(labels)) labels else factor(labels, levels = labels)
      } else value[match(labels, value)]
    }, values, levels)
    do.call(expand.grid, c(columns, list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)))
  } else NULL
  list(
    names = term_labels,
    levels = levels,
    factor_info = lapply(values, function(value) {
      if (inherits(value, "tcut")) {
        return(list(levels = attr(value, "labels"), ordered = FALSE))
      }
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
    keys = as.character(seq_len(prod(vapply(levels, length, integer(1))))),
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
    } else if (identical(info$class, "POSIXct") && !inherits(frame[[column]], "POSIXct")) {
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
                              group_name = NULL, group_info = NULL,
                              model_frame = NULL, x_values = NULL, y_values = NULL,
                              include_model = FALSE, include_x = FALSE,
                              include_y = FALSE) {
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
      keep <- as.numeric(pyears_values) > 0
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
  } else {
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
  }

  observations <- out$observations
  out$observations <- NULL
  summary_values <- .result_field(result, "summary")
  if (!is.null(summary_values)) {
    events <- out$event
    out$event <- NULL
    out$summary <- summary_values
    if (!is.null(events)) out$event <- events
  }
  out$observations <- observations
  if (!is.null(terms)) {
    out$terms <- terms
  }
  if (!is.null(model_frame)) {
    omitted <- attr(model_frame, "na.action")
    if (length(omitted)) {
      out$na.action <- omitted
    }
    if (isTRUE(include_model)) {
      out$model <- model_frame
    } else {
      if (isTRUE(include_x)) out$x <- x_values
      if (isTRUE(include_y)) out$y <- y_values
    }
  }
  class(out) <- "pyears"
  out
}

# Numeric two-column responses follow R's two kernels: (time, event)
# without a rate table, and (start, stop) without events with a rate table.
.pyears_formula_response_args <- function(response, has_ratetable) {
  if (inherits(response, "Surv")) {
    type <- attr(response, "type")
    if (!type %in% c("right", "counting")) {
      base::stop("Only right-censored and counting process survival types are supported", call. = FALSE)
    }
    if (type == "right") {
      if (any(response[, 1L] < 0)) base::stop("Negative survival time", call. = FALSE)
      nzero <- sum(response[, 1L] == 0 & response[, 2L] == 1)
      if (nzero > 0L) warning(nzero, " observations with an event and 0 follow-up time, ",
        "any rate calculations are statistically questionable", call. = FALSE)
    }
    return(list(stop = as.numeric(response[, ncol(response) - 1L]),
      start = if (type == "counting") as.numeric(response[, 1L]) else NULL,
      event = as.numeric(response[, ncol(response)])))
  }
  if (any(response < 0)) base::stop("Negative follow up time", call. = FALSE)
  response <- as.matrix(response)
  if (ncol(response) > 2L) base::stop("Y has too many columns", call. = FALSE)
  if (ncol(response) == 2L && has_ratetable) {
    list(start = as.numeric(response[, 1L]), stop = as.numeric(response[, 2L]), event = NULL)
  } else list(start = NULL, stop = as.numeric(response[, 1L]),
    event = if (ncol(response) == 2L) as.numeric(response[, 2L]) else NULL)
}

.pyears_categories <- function(term_values, n) {
  p <- length(term_values)
  x <- matrix(0, n, p)
  factors <- integer(p)
  levels <- cuts <- vector("list", p)
  for (column in seq_len(p)) {
    value <- term_values[[column]]
    if (inherits(value, "tcut")) {
      x[, column] <- as.numeric(value)
      cuts[[column]] <- as.list(as.numeric(attr(value, "cutpoints")))
      levels[[column]] <- attr(value, "labels")
    } else {
      factor <- as.factor(value)
      x[, column] <- as.numeric(factor)
      factors[column] <- 1L
      cuts[[column]] <- list()
      levels[[column]] <- levels(factor)
    }
  }
  list(x = x, factors = factors, dims = vapply(levels, length, integer(1)), cuts = cuts, levels = levels)
}

.pyears_native <- function(followup, categories, weights, scale, expect,
                           ratetable = NULL, expected_data = NULL, summary = NULL) {
  scale <- .as_finite_scalar(scale, "scale", positive = TRUE)
  n <- length(followup$stop)
  groups <- if (length(categories$dims)) as.character(seq_len(prod(categories$dims))) else "(all)"
  raw <- do.call(.population_attr("pyears"), .compact_null(list(
    stop = array(followup$stop),
    start = if (is.null(followup$start)) NULL else array(followup$start),
    event = if (is.null(followup$event)) NULL else array(followup$event),
    weights = if (is.null(weights)) NULL else array(as.numeric(weights)),
    factors = as.list(categories$factors), dims = as.list(unname(categories$dims)), cuts = categories$cuts,
    categories_data = categories$x,
    ratetable = if (is.null(ratetable)) NULL else .as_python_ratetable(ratetable),
    ratetable_positions = expected_data, expect = expect, scale = scale
  )))$to_arrays()
  list(pyears = as.numeric(raw$pyears), n = as.numeric(raw$n), offtable = raw$offtable,
    group = groups, observations = n,
    event = if (is.null(raw$event)) NULL else as.numeric(raw$event),
    expected = if (is.null(raw$expected)) NULL else as.numeric(raw$expected),
    tcut = any(categories$factors == 0L), summary = summary)
}

# Build a mapping call without parsing/deparsing variable names. Added variables
# pass through model.frame's subset and missing-data handling with the response.
.pyears_rate_call <- function(rate_call, ratetable) {
  names <- names(dimnames(ratetable))
  if (is.null(names)) names <- attr(ratetable, "dimid")
  entries <- if (is.null(rate_call)) list() else as.list(rate_call)[-1L]
  unknown <- setdiff(names(entries), names)
  if (length(unknown)) base::stop("Variable not found in the ratetable: ", paste(unknown, collapse = ", "), call. = FALSE)
  for (name in setdiff(names, names(entries))) entries[[name]] <- as.name(name)
  as.call(c(list(quote(list)), entries))
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
    if (missing(formula)) base::stop("A formula argument is required", call. = FALSE)
    Call <- match.call()
    original_formula <- formula
    output_terms <- if (missing(data)) stats::terms(formula) else stats::terms(formula, data = data)
    if (any(attr(output_terms, "order") > 1L)) base::stop("Pyears cannot have interaction terms", call. = FALSE)
    has_ratetable <- !missing(rmap) || !missing(ratetable)
    rate_call <- NULL
    if (has_ratetable) {
      if (missing(ratetable)) base::stop("No rate table specified", call. = FALSE)
      if (!is.ratetable(ratetable)) {
        if (inherits(ratetable, "coxph") && !inherits(ratetable, "coxphms")) {
          if (length(attr(ratetable$terms, "offset"))) base::stop("Cannot deal with models that contain an offset", call. = FALSE)
          if (length(attr(ratetable$terms, "specials")$strata)) base::stop("pyears cannot handle stratified Cox models", call. = FALSE)
          base::stop("pyears with a Cox rate table is not supported by survival 3.8-12", call. = FALSE)
        }
        base::stop("Invalid rate table", call. = FALSE)
      }
      rate_call <- if (missing(rmap)) NULL else substitute(rmap)
      if (!is.null(rate_call) && (!is.call(rate_call) || !identical(rate_call[[1L]], quote(list)))) {
        base::stop("Invalid rcall argument", call. = FALSE)
      }
      rate_call <- .pyears_rate_call(rate_call, ratetable)
    }
    model_formula <- .formula_with_native_surv_response(formula(output_terms), parent.frame())
    for (name in all.vars(rate_call)) model_formula[[3L]] <- call("+", model_formula[[3L]], as.name(name))
    formula_terms <- if (missing(data)) stats::terms(model_formula) else stats::terms(model_formula, data = data)
    if (!attr(formula_terms, "response")) base::stop("Follow-up time must appear in the formula", call. = FALSE)
    # Cache a possibly Python-backed response and let model.frame evaluate other
    # terms once. This also supports response functions in a formula environment.
    response <- eval(attr(formula_terms, "variables")[[2L]], if (missing(data)) NULL else data, environment(formula_terms))
    if (inherits(response, "survival_py_surv")) response <- .as_native_surv(response)
    predvars <- attr(formula_terms, "variables")
    predvars[[2L]] <- as.call(list(function() response))
    attr(formula_terms, "predvars") <- predvars
    indices <- match(c("formula", "data", "weights", "subset", "na.action"), names(Call), nomatch = 0L)
    model_call <- Call[c(1L, indices[indices > 0L])]
    model_call[[1L]] <- quote(stats::model.frame)
    model_call$formula <- formula_terms
    mf <- eval(model_call, parent.frame())
    if (!nrow(mf)) base::stop("Data set has 0 observations", call. = FALSE)
    response <- stats::model.response(mf)
    retained_terms <- attr(mf, "terms")
    retained_predvars <- attr(retained_terms, "predvars")
    retained_predvars[[2L]] <- attr(formula_terms, "variables")[[2L]]
    attr(retained_terms, "predvars") <- retained_predvars
    environment(retained_terms) <- environment(original_formula)
    attr(mf, "terms") <- retained_terms
    if (anyNA(response)) base::stop("missing values in the response", call. = FALSE)
    followup <- .pyears_formula_response_args(response, has_ratetable)
    term_labels <- attr(output_terms, "term.labels")
    term_values <- mf[term_labels]
    categories <- .pyears_categories(term_values, nrow(mf))
    x_values <- if (length(term_values)) categories$x else rep(1, nrow(mf))
    y_values <- if (inherits(response, "Surv")) response else as.matrix(response)
    matched <- NULL
    if (has_ratetable) {
      rate_data <- data.frame(eval(rate_call, mf, environment(formula_terms)), stringsAsFactors = TRUE)
      matched <- match.ratetable(rate_data, ratetable)
    }
    result <- .pyears_native(followup, categories, stats::model.weights(mf), scale, expect,
      ratetable = if (has_ratetable) ratetable else NULL,
      expected_data = if (is.null(matched)) NULL else matched$R,
      summary = if (is.null(matched)) NULL else matched$summ)
    group_info <- .pyears_formula_group_info(term_labels, mf, data.frame, categories$levels)
    out <- .as_pyears_result(result, Call, data.frame = data.frame, terms = output_terms,
      group_name = if (length(term_labels) == 1L) term_labels[[1L]] else NULL,
      group_info = group_info, model_frame = mf, x_values = x_values, y_values = y_values,
      include_model = model, include_x = x, include_y = y)
    # R drops dimensions when the full category grid has only one cell.
    if (!isTRUE(data.frame) && length(result$pyears) == 1L) {
      for (name in c("pyears", "n", "event", "expected")) {
        if (!is.null(out[[name]])) out[[name]] <- as.numeric(out[[name]])
      }
    }
    # No synthetic group column belongs to a formula with no covariates.
    if (isTRUE(data.frame) && !length(term_labels)) out$data$group <- NULL
    return(out)
  }
  if (!missing(ratetable) && !is.null(ratetable)) {
    base::stop("direct pyears bridge does not yet support ratetable; use formula pyears for R ratetables", call. = FALSE)
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
    subset = if (missing(subset)) NULL else .as_python_formula_subset(
      subset, n = if (!is.null(direct_time)) NROW(direct_time) else length(stop)
    ),
    `na_action` = if (missing(na.action)) NULL else .as_na_action(na.action),
    scale = scale,
    `data_frame` = FALSE
  )
  .as_pyears_result(result, match.call(), data.frame = data.frame)
}
