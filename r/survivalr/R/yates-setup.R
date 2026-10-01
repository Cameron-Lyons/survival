.yates_survival_setup <- function(baseline, rmean) {
  prediction <- .validation_attr("YatesPrediction")(
    time = as.list(as.numeric(baseline$time)),
    cumhaz = as.list(as.numeric(baseline$cumhaz)), rmean = rmean
  )
  predict_fun <- function(eta, ...) {
    eta <- drop(eta)
    if (length(dim(eta)) > 1L) stop("eta must be a vector or single-row/column matrix", call. = FALSE)
    result <- .as_numeric_matrix(prediction$predict(as.list(as.numeric(eta))))
    dimnames(result) <- list(names(eta), c("meansurv", rep("", ncol(result) - 1L)))
    result
  }
  summary_fun <- function(surv, var) {
    if (!is.matrix(surv) || ncol(surv) != length(baseline$time) + 2L) {
      stop("surv columns must contain restricted mean, time zero, and baseline times", call. = FALSE)
    }
    curves <- .validation_attr("yates_survival_summary")(surv, var, baseline$conf.int)
    for (name in c("surv", "cumhaz", "std_err", "lower", "upper")) {
      value <- .as_numeric_matrix(.result_field(curves, name))
      if (!length(baseline$time)) value <- matrix(numeric(), 0L, nrow(surv))
      dimnames(value) <- list(colnames(surv)[-c(1L, 2L)], rownames(surv))
      baseline[[if (name == "std_err") "std.err" else name]] <- value
    }
    baseline$std.chaz <- baseline$std.err
    baseline
  }
  structure(
    list(predict = predict_fun, summary = summary_fun),
    survivalr_prediction = list(kind = "survival", baseline = baseline, rmean = rmean)
  )
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
    return(structure(function(eta, X) exp(eta), survivalr_prediction = list(kind = "risk")))
  }
  if (type == "survival") {
    suppressWarnings(baseline <- if (inherits(fit, "survival_py_coxph")) {
      survfit(fit, censor = FALSE)
    } else {
      .cox_yates_baseline(fit)
    })
    rmean <- if (missing(options) || is.null(options$rmean)) {
      if (length(baseline$time)) max(baseline$time) else -Inf
    } else {
      options$rmean
    }
    if (!is.null(baseline$strata)) {
      stop("stratified models not yet supported", call. = FALSE)
    }
    return(.yates_survival_setup(baseline, rmean))
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

