# External lm/GLM objects retain their own formula evaluation and prediction.
# Concordance counts, influence functions and model comparisons use Rust/Python.
.concordance_lm_data <- function(object, newdata, cluster) {
  if (is.null(newdata)) {
    frame <- object$model
    if (is.null(frame)) frame <- stats::model.frame(object)
    response <- object$y
    if (is.null(response)) response <- stats::model.response(frame)
    score <- object$linear.predictors
    if (is.null(score)) score <- object$fitted.values
    if (is.null(score)) {
      object$na.action <- NULL
      score <- stats::predict(object)
    }
    weights <- stats::model.weights(frame)
  } else {
    frame <- stats::model.frame(stats::terms(object), data = newdata,
                               na.action = stats::na.omit, xlev = object$xlevels)
    response <- stats::model.response(frame)
    score <- stats::predict(object, newdata = newdata, na.action = stats::na.pass)
    omitted <- attr(frame, "na.action")
    if (length(omitted)) score <- score[-as.integer(omitted)]
    weights <- NULL
  }
  if (!is.numeric(response) || !is.null(dim(response))) {
    stop("left hand side of the formula must be a numeric vector", call. = FALSE)
  }
  if (length(score) != length(response)) {
    stop("model predictions must match the response length", call. = FALSE)
  }
  list(y = as.list(unname(response)), x = as.list(unname(score)),
       weights = if (is.null(weights)) NULL else as.list(unname(weights)),
       cluster = if (!length(cluster)) NULL else
         .as_python_vector(match(cluster, sort(unique(cluster))) - 1L))
}

concordance.lm <- function(object, ..., newdata, cluster, ymin, ymax,
                           influence = 0, ranks = FALSE, timefix = TRUE,
                           keepstrata = 10) {
  Call <- match.call()
  fits <- list(object, ...)
  if (!all(vapply(fits, inherits, logical(1), what = "lm"))) {
    stop("argument is not an appropriate fit object", call. = FALSE)
  }
  fit_names <- as.character(Call)[1L + seq_along(fits)]
  data <- lapply(fits, .concordance_lm_data,
                 newdata = if (missing(newdata)) NULL else newdata,
                 cluster = if (missing(cluster)) NULL else cluster)
  options <- list(influence = as.integer(influence), ranks = ranks, timefix = timefix,
                  keepstrata = keepstrata, reverse = FALSE,
                  ymin = if (missing(ymin)) NULL else ymin,
                  ymax = if (missing(ymax)) NULL else ymax)
  captured <- .pybridge_attr("_call_fit_with_warnings")(
    .pybridge_attr("_concordance_lm_data"),
    list(data = data, names = as.list(fit_names), options = options,
         newdata = !missing(newdata) && !is.null(newdata))
  )
  for (message in captured$warnings) warning(message, call. = FALSE)
  .as_model_concordance_list(captured$result, fit_names,
                             if (missing(cluster)) NULL else cluster, Call)
}

.as_model_concordance_list <- function(result, fit_names, cluster, Call) {
  out <- .as_concordance_list(result, fit_names)
  if (length(fit_names) > 1L) {
    out$cvar <- as.numeric(out$cvar)
    if (!is.null(out$dfbeta)) dimnames(out$dfbeta) <- NULL
    if (!is.null(out$influence)) dimnames(out$influence) <- NULL
  }
  if (length(cluster) && !is.null(out$dfbeta)) {
    labels <- as.character(sort(unique(cluster)))
    if (length(fit_names) == 1L) names(out$dfbeta) <- labels else rownames(out$dfbeta) <- labels
  }
  out$call <- Call
  class(out) <- "concordance"
  out
}

concordance.survival_py_model <- function(object, ..., newdata, cluster, ymin, ymax,
                                          timewt = c("n", "S", "S/G", "n/G2", "I"),
                                          influence = 0, ranks = FALSE, timefix = TRUE,
                                          keepstrata = 10) {
  Call <- match.call()
  fits <- list(object, ...)
  nfit <- length(fits)
  fit_names <- as.character(Call)[1L + seq_len(nfit)]
  fit_class <- if (inherits(object, "survival_py_coxph")) {
    "survival_py_coxph"
  } else if (inherits(object, "survival_py_survreg")) {
    "survival_py_survreg"
  } else {
    stop("object is not an appropriate fit object", call. = FALSE)
  }
  valid_fit <- vapply(fits, inherits, logical(1), what = fit_class)
  if (any(!valid_fit)) {
    index <- which(!valid_fit)[[1L]]
    call_name <- names(Call)[index + 1L]
    label <- if (!is.null(call_name) && nzchar(call_name) && call_name != "object") {
      call_name
    } else {
      fit_names[[index]]
    }
    stop(label, " argument is not an appropriate fit object", call. = FALSE)
  }
  newdata <- if (missing(newdata)) NULL else newdata
  explicit_cluster <- if (missing(cluster)) NULL else cluster
  clusters <- lapply(fits, function(fit) {
    value <- explicit_cluster
    if (is.null(value) && is.null(newdata) && inherits(fit, "survival_py_coxph")) {
      value <- unlist(fit$cluster, use.names = FALSE)
    }
    value
  })
  codes <- lapply(clusters, function(value) {
    if (is.null(value)) NULL else .as_python_vector(match(value, sort(unique(value))) - 1L)
  })
  options <- list(timewt = match.arg(timewt), influence = as.integer(influence),
                  ranks = ranks, timefix = timefix, keepstrata = keepstrata,
                  reverse = inherits(object, "survival_py_coxph"),
                  ymin = if (missing(ymin)) NULL else ymin,
                  ymax = if (missing(ymax)) NULL else ymax)
  captured <- .pybridge_attr("_call_fit_with_warnings")(
    .pybridge_attr("_concordance_survival_models"),
    list(fits = fits, names = as.list(fit_names), options = options,
         newdata = .as_python_data(newdata), clusters = codes)
  )
  for (message in captured$warnings) warning(message, call. = FALSE)
  .as_model_concordance_list(captured$result, fit_names, clusters[[1L]], Call)
}
