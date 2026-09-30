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
  out <- .as_concordance_list(captured$result, fit_names)
  if (length(fits) > 1L) {
    out$cvar <- as.numeric(out$cvar)
    if (!is.null(out$dfbeta)) dimnames(out$dfbeta) <- NULL
    if (!is.null(out$influence)) dimnames(out$influence) <- NULL
  }
  if (!missing(cluster) && length(cluster) && !is.null(out$dfbeta)) {
    labels <- as.character(sort(unique(cluster)))
    if (length(fits) == 1L) names(out$dfbeta) <- labels else rownames(out$dfbeta) <- labels
  }
  out$call <- Call
  class(out) <- "concordance"
  out
}
