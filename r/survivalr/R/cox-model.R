# Prepare fitted R Cox data for the shared numerical kernels. Formula terms,
# contrasts and retained model frames remain R's responsibility.
.cox_fixed_design <- function(fit, frame, training = FALSE) {
  stored <- fit[["x"]]
  design <- if (training && !is.null(stored)) stored else stats::model.matrix(fit, frame)
  if (!is.null(fit$frail)) {
    sparse <- names(fit$pterms)[fit$pterms == 2L]
    design <- design[, !colnames(design) %in% sparse, drop = FALSE]
  }
  design
}

.cox_model_data <- function(fit, design = FALSE, curve = FALSE) {
  Terms <- fit$terms
  if (inherits(fit, "coxphms")) stop("multi-state coxph not yet supported", call. = FALSE)
  if (length(attr(Terms, "specials")$tt)) {
    stop("function not defined for models with tt() terms", call. = FALSE)
  }
  strata_terms <- untangle.specials(Terms, "strata")
  frame <- fit$model
  if (is.null(frame) && (is.null(fit$y) || (design && is.null(fit[["x"]])) ||
      length(attr(Terms, "offset")) || !is.null(fit$call$weights) ||
      (length(strata_terms$vars) && is.null(fit$strata)))) {
    frame <- stats::model.frame(fit)
  }
  response <- if (is.null(fit$y)) stats::model.response(frame) else fit$y
  if (is.null(fit$y) && !identical(fit$timefix, FALSE)) {
    response <- .as_native_surv(aeqSurv(response))
  }
  if (!attr(response, "type") %in% c("right", "counting")) {
    stop("Invalid Cox survival response", call. = FALSE)
  }
  if ((!is.null(frame) && nrow(response) != nrow(frame)) ||
      nrow(response) != length(fit$linear.predictors)) {
    stop("Failed to reconstruct the original data set", call. = FALSE)
  }
  weights <- if (is.null(frame)) fit$weights else stats::model.weights(frame)
  if (is.null(weights)) weights <- rep(1, nrow(response))
  offset <- if (is.null(frame)) NULL else stats::model.offset(frame)
  offset_mean <- if (is.null(offset)) 0 else sum(offset * weights) / sum(weights)
  beta <- ifelse(is.na(fit$coefficients), 0, fit$coefficients)
  center <- sum(fit$means * beta) + offset_mean
  x <- if (design || (curve && !is.null(fit$frail))) {
    .cox_fixed_design(fit, frame, training = TRUE)
  } else NULL
  # Individual predictions retain fitted frailty in the training risk sets;
  # the default survfit baseline excludes frailty, matching the Cox curve API.
  risk <- if (curve && !is.null(fit$frail)) {
    exp(drop(x %*% beta) + (if (is.null(offset)) 0 else offset) - center)
  } else exp(fit$linear.predictors - offset_mean)
  groups <- if (length(strata_terms$vars)) {
    if (is.null(frame)) fit$strata else strata(frame[strata_terms$vars], shortlabel = TRUE)
  } else factor(rep(0, nrow(response)))
  list(terms = Terms, frame = frame, y = response, x = x, weights = weights,
    beta = beta, center = center, risk = risk, strata = groups, strata_terms = strata_terms)
}

# Event-time baseline at the model's centering point. This is the retained
# survfit object needed by Yates summaries, including uncertainty and labels.
.cox_yates_baseline <- function(fit) {
  input <- .cox_model_data(fit, design = TRUE, curve = TRUE)
  if (nlevels(input$strata) > 1L) stop("stratified models not yet supported", call. = FALSE)
  factors <- attr(input$terms, "factors")
  if (length(input$strata_terms$vars)) {
    if (any(factors[input$strata_terms$vars, , drop = FALSE] *
            rep(attr(input$terms, "order"), each = length(input$strata_terms$vars)) > 1L)) {
      stop("Models with strata by covariate interaction terms require newdata", call. = FALSE)
    }
    for (name in input$strata_terms$vars) factors <- factors[, factors[name, ] == 0, drop = FALSE]
  }
  if (any(factors > 1L)) {
    stop(paste("not able to create a curve for models that contain an interaction",
               "without the lower order effect"), call. = FALSE)
  }
  p <- ncol(input$x)
  raw <- .survival_analysis_attr("coxsurv_fit")(
    y = unclass(input$y), x = input$x, weights = array(input$weights), risk = array(input$risk),
    strata = array(rep(0L, nrow(input$y))), nstrata = 1L,
    x2 = matrix(as.numeric(fit$means), 1L, p), risk2 = array(1),
    stype = 2L, ctype = if (fit$method == "efron") 2L else 1L,
    varmat = if (p) fit$var else matrix(numeric(), 0L, 0L))
  keep <- as.numeric(raw$n_event) > 0
  value <- function(name) as.numeric(reticulate::py_get_attr(raw, name))[keep]
  result <- list(n = as.integer(raw$n), time = value("time"), n.risk = value("n_risk"),
    n.event = value("n_event"), n.censor = value("n_censor"), surv = value("surv"),
    cumhaz = value("cumhaz"), std.err = value("std_err"), logse = TRUE)
  result$std.chaz <- result$std.err
  band <- .survival_analysis_attr("survfit_confint")(
    as.list(result$surv), as.list(result$std.err), logse = TRUE, conf_type = "log", conf_int = .95)
  result$lower <- .as_numeric_vector(band$lower)
  result$upper <- .as_numeric_vector(band$upper)
  result$conf.type <- "log"
  result$conf.int <- .95
  result$call <- quote(survfit(formula = fit, censor = FALSE))
  class(result) <- c("survfitcox", "survfit")
  result
}
