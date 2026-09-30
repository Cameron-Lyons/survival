# The Python survexp kernel on matched rate-table positions: the individual
# methods give one value per subject, the cohort methods the fit at each of `times`
# (which already holds every output time).
.survexp_ratetable_fit <- function(group, expected_data, followup, times,
                                   method, ratetable) {
  times <- sort(unique(times))
  n_times <- length(times)
  n_groups <- max(group)
  raw <- .population_attr("survexp")(
    ratetable = .as_python_ratetable(ratetable),
    positions = expected_data,
    y = array(as.numeric(followup)),
    group = as.list(as.integer(group) - 1L),
    times = array(as.numeric(times)),
    method = method
  )$to_arrays()
  if (startsWith(method, "individual")) {
    return(as.numeric(raw$surv))
  }
  survival <- raw$surv
  n_risk <- matrix(as.integer(raw$n_risk), n_times, n_groups)
  if (n_groups > 1L) {
    list(surv = survival, n = n_risk)
  } else {
    list(surv = c(survival), n = c(n_risk))
  }
}

# R handles terms, contrasts and factor levels; baseline hazards and cohort
# reduction use the same Rust kernels as Python's fitted-model methods.
.survexp_cox_fit <- function(fit, data, groups, weights, response, times, method) {
  individual <- startsWith(method, "individual")
  if (inherits(fit, "survival_py_coxph")) {
    if (!ncol(data)) data[[".row"]] <- seq_len(nrow(data))
    raw <- .pybridge_attr("_survexp_cox_fit")(
      .restore_python(fit), .as_python_data(data), array(as.integer(groups) - 1L), array(as.numeric(weights)),
      if (is.null(response)) NULL else array(as.numeric(response)),
      if (is.null(times)) NULL else array(as.numeric(times)), method)
    if (individual) return(as.numeric(raw))
  } else {
    Terms <- fit$terms
    if (length(attr(Terms, "specials")$tt)) stop("function not defined for models with tt() terms", call. = FALSE)
    if (!is.null(fit$frail) && !individual) stop("Newdata cannot be used when a model has frailty terms", call. = FALSE)
    factors <- attr(Terms, "factors")
    strata_terms <- untangle.specials(Terms, "strata")
    for (name in strata_terms$vars) factors <- factors[, factors[name, ] == 0, drop = FALSE]
    if (!individual && any(factors > 1L)) stop("not able to create a curve for models that contain an interaction without the lower order effect", call. = FALSE)
    input <- .cox_model_data(fit)
    train_y <- input$y
    beta <- input$beta
    new_terms <- if (individual) Terms else stats::delete.response(Terms)
    new_frame <- stats::model.frame(new_terms, data, xlev = fit$xlevels, na.action = stats::na.fail)
    new_x <- .cox_fixed_design(fit, new_frame)
    new_offset <- stats::model.offset(new_frame)
    if (is.null(new_offset)) new_offset <- rep(0, nrow(new_frame))
    new_risk <- exp(drop(new_x %*% beta) + new_offset - input$center)
    train_strata <- input$strata
    new_strata <- if (length(strata_terms$vars)) {
      match(as.character(strata(new_frame[strata_terms$vars], shortlabel = TRUE)), levels(train_strata))
    } else rep(1L, nrow(new_frame))
    if (anyNA(new_strata)) stop("New data set has strata levels not found in the original", call. = FALSE)
    baselines <- lapply(split(seq_len(nrow(train_y)), train_strata), function(rows) {
      if (!length(rows)) return(list(time = numeric(), cumhaz = numeric()))
      curve <- .survival_analysis_attr("cox_survfit_baseline")(
        unclass(train_y[rows, , drop = FALSE]), matrix(0, length(rows), 0L),
        array(input$weights[rows]), array(input$risk[rows]),
        if (fit$method == "efron") 3L else 2L, if (fit$method == "efron") 3L else 2L)
      keep <- as.numeric(curve$n_event) > 0
      list(time = as.numeric(curve$time)[keep], cumhaz = as.numeric(curve$cumhaz)[keep])
    })
    if (individual) {
      prediction_y <- stats::model.response(new_frame)
      if (attr(prediction_y, "type") != attr(train_y, "type")) stop("New data has a different survival type than the model", call. = FALSE)
      value <- numeric(nrow(new_frame))
      for (s in unique(new_strata)) {
        rows <- which(new_strata == s)
        baseline <- baselines[[s]]
        at <- function(time) .as_numeric_vector(.survival_analysis_attr("step_values_at")(
          as.list(baseline$time), as.list(baseline$cumhaz), as.list(as.numeric(time)), 0))
        hazard <- at(prediction_y[rows, ncol(prediction_y) - 1L])
        if (ncol(prediction_y) == 3L) hazard <- hazard - at(prediction_y[rows, 1L])
        value[rows] <- hazard * new_risk[rows]
      }
      return(if (method == "individual.h") value else exp(-value))
    }
    raw <- .population_attr("survexp_cox_prepared")(
      array(unlist(lapply(baselines, `[[`, "time"))), array(unlist(lapply(baselines, `[[`, "cumhaz"))),
      array(vapply(baselines, function(b) length(b$time), integer(1))), array(new_risk),
      array(new_strata - 1L), array(as.integer(groups) - 1L), array(as.numeric(weights)),
      if (is.null(response)) NULL else array(as.numeric(response)),
      if (is.null(times)) NULL else array(as.numeric(times)), method)$to_arrays()
  }
  # The reference exposes counts only for Ederer Cox cohorts.
  list(time = as.numeric(raw$time), surv = raw$surv,
    n = if (method == "ederer") raw$n_risk else NULL)
}

.survexp_formula <- function(Call, evaluation_env, ratetable,
                                       times_missing, method_missing,
                                       cohort_missing, conditional_missing,
                                       se_fit_missing, method, cohort,
                                       conditional, scale, se_fit, model, x, y) {
  model_args <- match(
    c("formula", "data", "weights", "subset", "na.action"),
    names(Call),
    nomatch = 0L
  )
  if (model_args[[1L]] == 0L) {
    stop("A formula argument is required", call. = FALSE)
  }
  model_call <- Call[c(1L, model_args)]
  model_call[[1L]] <- quote(stats::model.frame)
  formula_value <- eval(Call$formula, evaluation_env)
  Terms <- if ("data" %in% names(Call)) {
    stats::terms(formula_value, data = eval(Call$data, evaluation_env))
  } else {
    stats::terms(formula_value)
  }

  rate_call <- if ("rmap" %in% names(Call)) Call$rmap else NULL
  if (!is.null(rate_call) && (!is.call(rate_call) || rate_call[[1L]] != as.name("list"))) {
    stop("Invalid rate-call argument", call. = FALSE)
  }
  israte <- is.ratetable(ratetable)
  python_cox <- inherits(ratetable, "survival_py_coxph")
  if (!israte && !python_cox && (!inherits(ratetable, "coxph") || inherits(ratetable, "coxphms"))) {
    stop("Invalid rate table", call. = FALSE)
  }
  cox_terms <- if (python_cox) stats::terms(stats::as.formula(.result_field(ratetable, "formula"))) else if (!israte) ratetable$terms else NULL
  varlist <- if (israte) names(dimnames(ratetable)) else all.vars(stats::delete.response(cox_terms))
  if (israte && is.null(varlist)) varlist <- attr(ratetable, "dimid")
  mapped <- match(names(rate_call)[-1L], varlist)
  if (any(is.na(mapped))) {
    stop("Variable not found in the ratetable: ", names(rate_call)[-1L][is.na(mapped)], call. = FALSE)
  }
  if (any(!(varlist %in% names(rate_call)))) {
    entries <- if (is.null(rate_call)) list() else as.list(rate_call)[-1L]
    for (name in setdiff(varlist, names(entries))) entries[[name]] <- as.name(name)
    rate_call <- as.call(c(list(quote(list)), entries))
  }
  if (is.null(rate_call)) rate_call <- quote(list())
  new_variables <- all.vars(rate_call)
  # Individual Cox predictions read the fitted response, which can differ from
  # this formula's follow-up. Include its variables before subset/NA processing.
  individual <- (!method_missing && startsWith(method, "individual")) || (method_missing && !isTRUE(cohort))
  if (!israte && individual) new_variables <- union(new_variables, all.vars(stats::formula(cox_terms)[[2L]]))
  expanded <- .formula_with_native_surv_response(stats::formula(Terms), evaluation_env)
  for (name in new_variables) expanded[[length(expanded)]] <- call("+", expanded[[length(expanded)]], as.name(name))
  frame_terms <- stats::terms(expanded)
  if (attr(frame_terms, "response")) {
    source_data <- if ("data" %in% names(Call)) eval(Call$data, evaluation_env) else NULL
    response <- eval(attr(frame_terms, "variables")[[2L]], source_data, environment(frame_terms))
    if (inherits(response, "survival_py_surv")) response <- .as_native_surv(response)
    predvars <- attr(frame_terms, "variables")
    predvars[[2L]] <- as.call(list(function() response))
    attr(frame_terms, "predvars") <- predvars
  }
  model_call$formula <- frame_terms
  model_frame <- eval(model_call, evaluation_env)
  if (attr(frame_terms, "response")) {
    retained <- attr(model_frame, "terms")
    predvars <- attr(retained, "predvars")
    predvars[[2L]] <- attr(frame_terms, "variables")[[2L]]
    attr(retained, "predvars") <- predvars
    environment(retained) <- environment(Terms)
    attr(model_frame, "terms") <- retained
  }
  n <- nrow(model_frame)
  if (n == 0L) {
    stop("Data set has 0 rows", call. = FALSE)
  }
  if (!se_fit_missing && isTRUE(se_fit)) {
    warning("se.fit value ignored")
  }
  weights <- stats::model.weights(model_frame)
  if (length(weights) == 0L) {
    weights <- rep(1, n)
  }
  if (israte && any(weights != 1)) {
    warning("weights ignored")
  }
  if (any(attr(Terms, "order") > 1L)) {
    stop("Survexp cannot have interaction terms", call. = FALSE)
  }

  requested_times <- if (times_missing) NULL else as.numeric(eval(Call$times, evaluation_env))
  if (!times_missing) {
    if (!length(requested_times) || any(!is.finite(requested_times))) stop("times must be nonempty and finite", call. = FALSE)
    if (any(requested_times < 0)) {
      stop("Invalid time point requested", call. = FALSE)
    }
    if (length(requested_times) > 1L && any(diff(requested_times) < 0)) {
      stop("Times must be in increasing order", call. = FALSE)
    }
  }
  response <- stats::model.response(model_frame)
  if (anyNA(response)) stop("missing values in the response", call. = FALSE)
  no_response <- is.null(response)
  if (no_response) {
    if (times_missing && israte) {
      stop("either a times argument or a response is needed", call. = FALSE)
    }
    new_time <- requested_times
  } else {
    if (is.matrix(response)) {
      if (inherits(response, "Surv") && attr(response, "type") == "right") {
        response <- response[, 1L]
      } else {
        stop("Illegal response value", call. = FALSE)
      }
    }
    if (any(response < 0)) {
      stop("Negative follow up time", call. = FALSE)
    }
    unique_response <- unique(response)
    new_time <- if (times_missing) {
      sort(unique_response)
    } else {
      sort(unique(c(requested_times, unique_response[unique_response < max(requested_times)])))
    }
  }

  if (!method_missing) {
    method <- match.arg(method, c(
      "ederer", "hakulinen", "conditional", "individual.h", "individual.s"
    ))
  } else {
    method <- if (!conditional_missing && isTRUE(conditional)) {
      "conditional"
    } else if (no_response) {
      "ederer"
    } else {
      "hakulinen"
    }
    if (!cohort_missing && !isTRUE(cohort)) {
      method <- "individual.s"
    }
  }
  if (no_response && method != "ederer") {
    stop("a response is required in the formula unless method='ederer'", call. = FALSE)
  }

  output_variables <- attr(Terms, "term.labels")
  rate_data <- data.frame(eval(rate_call, model_frame, environment(Terms)), stringsAsFactors = TRUE, check.names = FALSE)
  if (!ncol(rate_data)) rate_data <- data.frame(row.names = row.names(model_frame))
  if (!israte && individual) {
    for (name in setdiff(all.vars(stats::formula(cox_terms)[[2L]]), names(rate_data))) rate_data[[name]] <- model_frame[[name]]
  }
  if (no_response && israte) {
    response <- rep(max(requested_times), n)
  }
  matched <- if (israte) match.ratetable(rate_data, ratetable) else NULL
  expected_data <- if (israte) matched$R else NULL

  if (startsWith(method, "individual")) {
    if (no_response) {
      stop("for individual survival an observation time must be given", call. = FALSE)
    }
    values <- if (israte) .survexp_ratetable_fit(
      seq_len(n), expected_data, response, max(response), method, ratetable
    ) else .survexp_cox_fit(ratetable, rate_data, rep(1L, n), weights, response, NULL, method)
    names(values) <- row.names(model_frame)
    omitted <- attr(model_frame, "na.action")
    if (length(omitted)) stats::naresid(omitted, values) else values
  } else {
    if (length(output_variables) == 0L) {
      groups <- rep(1L, n)
    } else {
      for (variable in output_variables) {
        if (inherits(model_frame[[variable]], "tcut")) {
          stop("Can't use tcut variables in expected survival", call. = FALSE)
        }
      }
      groups <- strata(model_frame[output_variables])
    }
    fit <- if (israte) .survexp_ratetable_fit(
      as.numeric(groups), expected_data, response, new_time,
      if (method == "conditional") "conditional" else "hakulinen", ratetable
    ) else .survexp_cox_fit(ratetable, rate_data, as.integer(groups), weights,
      if (no_response) NULL else response, requested_times, method)
    if (!israte) new_time <- fit$time
    if (times_missing || !israte) {
      n_risk <- fit$n
      survival <- fit$surv
    } else {
      keep <- match(requested_times, new_time)
      if (is.matrix(fit$surv)) {
        survival <- rbind(1, fit$surv)[keep + 1L, , drop = FALSE]
        n_risk <- fit$n[pmax(1L, keep), , drop = FALSE]
      } else {
        survival <- c(1, fit$surv)[keep + 1L]
        n_risk <- fit$n[pmax(1L, keep)]
      }
      new_time <- requested_times
    }
    new_time <- new_time / .as_finite_scalar(scale, "scale", positive = TRUE)
    if (is.matrix(survival)) {
      dimnames(survival) <- list(NULL, levels(groups))
      out <- list(
        call = Call,
        surv = drop(survival),
        n.risk = drop(n_risk),
        time = new_time
      )
    } else {
      out <- list(
        call = Call,
        surv = c(survival),
        n.risk = c(n_risk),
        time = new_time
      )
    }
    if (model) {
      out$model <- model_frame
    } else {
      if (x) out$x <- groups
      if (y) out$y <- response
    }
    if (!is.null(matched$summ)) {
      out$summ <- matched$summ
    }
    out$method <- if (no_response) {
      "Ederer"
    } else if (isTRUE(conditional)) {
      "conditional"
    } else {
      "cohort"
    }
    class(out) <- c("survexp", "survfit")
    out
  }
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
    formula_ratetable <- if (missing(ratetable) || is.null(ratetable)) {
      survival::survexp.us
    } else {
      ratetable
    }
    return(.survexp_formula(
      Call = match.call(),
      evaluation_env = parent.frame(),
      ratetable = formula_ratetable,
      times_missing = missing(times),
      method_missing = missing(method),
      cohort_missing = missing(cohort),
      conditional_missing = missing(conditional),
      se_fit_missing = missing(se.fit),
      method = method,
      cohort = cohort,
      conditional = conditional,
      scale = scale,
      se_fit = if (missing(se.fit)) NULL else se.fit,
      model = model,
      x = x,
      y = y
    ))
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
  if (!is.null(sex) && is.numeric(sex) && all(sex %in% c(0, 1))) {
    # this interface's 0/1 codes are survexp.us's male/female
    sex <- c("male", "female")[sex + 1L]
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
