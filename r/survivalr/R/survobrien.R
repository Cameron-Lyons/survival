# R evaluates formulas and retains column classes. The shared kernel constructs
# risk sets and ranks; source rows align protected columns after subset/NA removal.
survobrien <- function(formula, data, subset, na.action, transform) {
  Call <- match.call()
  if (!missing(transform) && length(transform(seq_len(10L))) != 10L) {
    stop("Transform function must be 1 to 1", call. = FALSE)
  }
  if (missing(formula)) stop("A formula argument is required", call. = FALSE)
  formula <- .formula_with_native_surv_response(formula, parent.frame())
  Terms <- if (missing(data)) {
    stats::terms(formula, specials = c("strata", "cluster", "tt"))
  } else stats::terms(formula, specials = c("strata", "cluster", "tt"), data = data)
  if (attr(Terms, "response") == 0L) stop("Response must be a survival object", call. = FALSE)
  source <- if (missing(data)) NULL else data
  formula_env <- environment(Terms)
  # Evaluate the response once and reuse it in model.frame. This also supplies
  # a row index for subset/NA alignment without reevaluating the response call.
  response <- eval(attr(Terms, "variables")[[2L]], source, formula_env)
  if (inherits(response, "survival_py_surv")) response <- .as_native_surv(response)
  predvars <- attr(Terms, "variables")
  predvars[[2L]] <- as.call(list(function() response))
  attr(Terms, "predvars") <- predvars
  indices <- match(c("formula", "data", "subset", "na.action"), names(Call), nomatch = 0L)
  frame_call <- Call[c(1L, indices[indices > 0L])]
  frame_call[[1L]] <- quote(stats::model.frame)
  frame_call$formula <- Terms
  frame_call$.survobrien_source <- seq_len(NROW(response))
  frame <- eval(frame_call, parent.frame())
  source_rows <- frame[["(.survobrien_source)"]]
  frame[["(.survobrien_source)"]] <- NULL
  if (!nrow(frame)) stop("No (non-missing) observations", call. = FALSE)
  response <- stats::model.response(frame)
  if (!inherits(response, "Surv")) stop("Response must be a survival object", call. = FALSE)
  type <- attr(response, "type")
  if (!type %in% c("right", "counting")) {
    stop("Response must be right censored or (start, stop] data", call. = FALSE)
  }
  Terms <- attr(frame, "terms")
  clusters <- untangle.specials(Terms, "cluster")
  if (length(clusters$terms) > 1L) stop("Can have only 1 cluster term", call. = FALSE)
  strata_terms <- untangle.specials(Terms, "strata")
  omitted <- unique(c(clusters$terms, strata_terms$terms))
  covariates <- if (length(omitted)) stats::drop.terms(Terms, omitted, keep.response = TRUE) else Terms
  if (any(attr(covariates, "order") > 1L)) {
    stop("This function cannot deal with iteraction terms", call. = FALSE)
  }
  variables <- attr(covariates, "term.labels")
  keep <- vapply(frame[variables], function(x) is.factor(x) || inherits(x, "AsIs"), logical(1))
  if (all(keep)) stop("No continuous variables to modify", call. = FALSE)
  strata_codes <- NULL
  if (length(strata_terms$vars)) {
    strata_values <- if (length(strata_terms$vars) == 1L) {
      frame[[strata_terms$vars]]
    } else strata(frame[strata_terms$vars], shortlabel = TRUE)
    if (anyNA(strata_values)) stop("missing values in the strata", call. = FALSE)
    strata_codes <- as.integer(factor(strata_values))
  }
  columns <- frame[variables[!keep]]
  n <- nrow(frame)
  # R indexes a matrix-valued continuous term linearly, using its first column.
  continuous <- if (missing(transform)) lapply(columns, function(value) {
    value <- value[seq_len(n)]
    if (is.character(value)) {
      match(value, sort(unique(value), na.last = NA))
    } else as.numeric(value)
  }) else list()
  expansion <- .validation_attr("survobrien")(
    time = as.numeric(response[, ncol(response) - 1L]),
    status = as.integer(response[, ncol(response)]),
    continuous = unname(continuous),
    start = if (type == "counting") as.numeric(response[, 1L]) else NULL,
    strata = strata_codes, transform = missing(transform)
  )$to_arrays()
  rows <- as.integer(expansion$row) + 1L
  output <- lapply(seq_len(ncol(response) - 1L), function(column) response[rows, column])
  output[[ncol(response)]] <- as.integer(expansion$status)
  names(output) <- colnames(response)
  referenced <- function(terms) unlist(lapply(terms, function(term) all.vars(str2lang(term))), use.names = FALSE)
  kept_names <- c(referenced(variables[keep]), referenced(strata_terms$vars))
  copy_columns <- function(names) {
    result <- lapply(names, function(name) {
      values <- eval(as.name(name), source, formula_env)
      values[source_rows][rows]
    })
    names(result) <- names
    result
  }
  if (length(kept_names)) {
    kept <- copy_columns(kept_names)
    names(kept) <- make.unique(names(kept))
    output <- c(output, kept)
  }
  if (length(clusters$vars)) {
    output <- c(output, copy_columns(referenced(clusters$vars)))
  } else output <- c(output, list(.id. = rows))
  if (missing(transform)) {
    transformed <- lapply(expansion$transformed, as.numeric)
    transformed <- lapply(transformed, function(column) {
      column[is.nan(column)] <- NA_real_
      column
    })
  } else {
    offsets <- as.numeric(expansion$block_offsets)
    transformed <- lapply(columns, function(column) {
      values <- lapply(seq_len(length(offsets) - 1L), function(block) {
        index <- seq.int(offsets[block] + 1L, offsets[block + 1L])
        value <- transform(column[rows[index]])
        if (length(value) != length(index)) stop("Transform function must be 1 to 1", call. = FALSE)
        value
      })
      if (length(values)) unlist(values) else numeric()
    })
  }
  names(transformed) <- names(columns)
  as.data.frame(c(output, transformed, list(.strata. = as.integer(expansion$strata))),
                check.names = TRUE, stringsAsFactors = FALSE)
}
