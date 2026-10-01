# Formulas and covariate classes stay in R; Rust prepares censoring/entry curves
# and expands intervals, returning source rows for the fitted model frame.
.finegray_array <- function(value) if (is.null(value)) NULL else array(value)

finegray <- function(formula, data, weights, subset, na.action = na.pass,
                     etype, prefix = "fg", count = "", id, timefix = TRUE,
                     tstart, tstop, ctime, cprob, extend, keep) {
  direct_start <- if (!missing(tstart)) tstart else {
    if (!missing(formula) && !inherits(formula, "formula")) formula else NULL
  }
  if (!is.null(direct_start)) {
    if (missing(tstop) || missing(ctime) || missing(cprob) || missing(extend) || missing(keep)) {
      stop("direct finegray bridge requires tstart, tstop, ctime, cprob, extend, and keep", call. = FALSE)
    }
    result <- .call_regression("finegray", array(as.numeric(direct_start)), array(as.numeric(tstop)),
      array(as.numeric(ctime)), array(as.numeric(cprob)), array(as.logical(extend)), array(as.logical(keep)))$to_arrays()
    return(data.frame(row = as.integer(result$row), start = as.numeric(result$start),
      end = as.numeric(result$end), wt = as.numeric(result$wt), add = as.integer(result$add)))
  }
  if (missing(formula)) stop("A formula argument is required", call. = FALSE)
  Call <- match.call()
  formula <- .formula_with_native_surv_response(formula, parent.frame())
  Terms <- if (missing(data)) stats::terms(formula, c("strata", "cluster")) else {
    stats::terms(formula, c("strata", "cluster"), data = data)
  }
  if (!attr(Terms, "response")) stop("Response must be a survival object", call. = FALSE)
  response <- eval(attr(Terms, "variables")[[2L]], if (missing(data)) NULL else data, environment(Terms))
  if (inherits(response, "survival_py_surv")) response <- .as_native_surv(response)
  predvars <- attr(Terms, "variables")
  predvars[[2L]] <- as.call(list(function() response))
  attr(Terms, "predvars") <- predvars
  indices <- match(c("formula", "data", "weights", "subset", "id"), names(Call), nomatch = 0L)
  frame_call <- Call[c(1L, indices[indices > 0L])]
  frame_call[[1L]] <- quote(stats::model.frame)
  frame_call$formula <- Terms
  frame_call$na.action <- na.action
  frame <- eval(frame_call, parent.frame())
  if (!nrow(frame)) stop("No (non-missing) observations", call. = FALSE)
  response <- stats::model.response(frame)
  if (!inherits(response, "Surv")) stop("Response must be a survival object", call. = FALSE)
  type <- attr(response, "type")
  if (!type %in% c("mright", "mcounting")) {
    stop("Fine-Gray model requires a multi-state survival", call. = FALSE)
  }
  states <- attr(response, "states")
  if (length(states) < 2L) stop("survival time has only a single state", call. = FALSE)
  if (anyNA(response)) stop("missing values in the response", call. = FALSE)
  Terms <- attr(frame, "terms")
  if (length(attr(Terms, "specials")$cluster)) stop("a cluster() term is not valid", call. = FALSE)
  strata_terms <- untangle.specials(Terms, "strata", 1)
  strata_codes <- NULL
  if (length(strata_terms$vars)) {
    values <- if (length(strata_terms$vars) == 1L) frame[[strata_terms$vars]] else {
      strata(frame[strata_terms$vars], shortlabel = TRUE)
    }
    if (anyNA(values)) stop("strata must not contain missing values", call. = FALSE)
    strata_codes <- as.integer(factor(values))
    frame[strata_terms$vars] <- NULL
  }
  ids <- stats::model.extract(frame, "id")
  if (!is.null(ids)) {
    if (anyNA(ids)) stop("id must not contain missing values", call. = FALSE)
    ids <- match(ids, unique(ids))
    frame["(id)"] <- NULL
  }
  user_weights <- stats::model.weights(frame)
  enum <- if (missing(etype)) 1L else {
    index <- match(etype, states)
    if (!length(index) || anyNA(index)) stop("etype argument has a state that is not in the data", call. = FALSE)
    if (length(index) > 1L) warning("only the first endpoint was used", call. = FALSE)
    index[1L]
  }
  raw <- .call_regression("finegray_expand", array(as.numeric(response[, ncol(response) - 1L])),
    array(as.integer(response[, ncol(response)])), as.integer(enum),
    start = if (type == "mcounting") array(as.numeric(response[, 1L])) else NULL,
    strata = .finegray_array(strata_codes), id = .finegray_array(ids),
    weights = if (is.null(user_weights)) NULL else array(as.numeric(user_weights)),
    timefix = if (timefix) TRUE else FALSE)$to_arrays()
  rows <- as.integer(raw$row)
  output <- frame[rows, -1L, drop = FALSE]
  names <- paste0(prefix, c("start", "stop", "status", "wt"))
  output[[names[1L]]] <- as.numeric(raw$start)
  output[[names[2L]]] <- as.numeric(raw$end)
  output[[names[3L]]] <- as.numeric(response[rows, ncol(response)] == enum)
  output[[names[4L]]] <- as.numeric(raw$wt)
  if (!missing(count)) output[[make.names(count)]] <- as.integer(raw$add)
  rownames(output) <- NULL
  attr(output, "event") <- states[enum]
  output
}
