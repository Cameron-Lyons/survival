# R owns formula evaluation and result attributes. Marginal means, estimability,
# simulation, and contrasts use the shared numerical implementation.
.yates_frame <- function(fit, Terms) {
  if (inherits(fit, "survival_py_model")) {
    raw <- as.data.frame(.call_r_api("model_frame", fit), check.names = FALSE)
    metadata <- .pybridge_attr("_yates_model_metadata")(fit)
    xlevels <- metadata$xlevels
    for (name in names(metadata$raw_levels)) {
      raw[[name]] <- factor(raw[[name]], levels = metadata$raw_levels[[name]])
    }
    frame <- stats::model.frame(Terms, raw, xlev = xlevels, na.action = stats::na.pass)
    extra_factors <- setdiff(names(frame)[vapply(frame, is.factor, logical(1))], names(xlevels))
    xlevels <- c(xlevels, lapply(frame[extra_factors], levels))
    for (name in intersect(c("(weights)", "(id)"), names(raw))) frame[[name]] <- raw[[name]]
    Terms <- attr(frame, "terms")
    contrasts <- attr(stats::model.matrix(Terms, frame), "contrasts")
  } else {
    frame <- fit$model
    if (is.null(frame)) frame <- stats::model.frame(fit)
    xlevels <- fit$xlevels
    contrasts <- fit$contrasts
  }
  list(frame = frame, terms = Terms, xlevels = xlevels, contrasts = contrasts)
}

.yates_levels <- function(Terms, term, xlevels, levels = NULL) {
  labels <- attr(Terms, "term.labels")
  if (is.character(term)) {
    term <- stats::as.formula(paste("~", paste(term, collapse = "+")))
  } else if (is.numeric(term)) {
    if (!length(term) || anyNA(term) || any(term != floor(term) | term < 1L | term > length(labels))) {
      stop("a numeric term must be an integer between 1 and max terms in the fit", call. = FALSE)
    }
    term <- stats::as.formula(paste("~", paste(labels[term], collapse = "+")))
  } else if (!inherits(term, "formula")) {
    stop("the term must be a formula or integer", call. = FALSE)
  }
  variables <- as.list(attr(Terms, "variables"))[-1L]
  key <- vapply(variables, function(x) all.vars(x)[1L], character(1))
  raw <- all.vars(stats::delete.response(stats::terms(term)))
  if (!length(raw)) stop("a term must select variables from the fit", call. = FALSE)
  index <- match(raw, key)
  if (anyNA(index)) stop("variable ", paste(raw[is.na(index)], collapse = " "), " not found in the formula", call. = FALSE)
  parts <- rownames(attr(Terms, "factors"))[index]
  categorical <- attr(Terms, "dataClasses")[parts] %in% c("factor", "character", "ordered")
  if (is.null(levels)) {
    if (!all(parts %in% names(xlevels))) stop("continuous variables require the levels argument", call. = FALSE)
    values <- expand.grid(xlevels[parts], stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  } else if (is.list(levels)) {
    if (is.null(names(levels))) {
      if (length(raw) != 1L || length(levels) != 1L) stop("levels list requires named elements", call. = FALSE)
      names(levels) <- raw
    }
    values <- vector("list", length(raw))
    names(values) <- raw
    for (i in seq_along(raw)) {
      value <- if (raw[i] %in% names(levels)) levels[[raw[i]]] else xlevels[[parts[i]]]
      if (is.null(value)) stop("levels information not found for: ", raw[i], call. = FALSE)
      values[[i]] <- value
    }
    values <- expand.grid(values, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
    if (anyDuplicated(values)) stop("levels data frame has duplicates", call. = FALSE)
  } else if (is.matrix(levels)) {
    if (ncol(levels) != length(parts)) stop("levels matrix has the wrong number of columns", call. = FALSE)
    if (!is.null(colnames(levels))) {
      index <- match(raw, colnames(levels))
      if (anyNA(index)) stop("matrix column names do not match the variable list", call. = FALSE)
      levels <- levels[, index, drop = FALSE]
    } else if (ncol(levels) > 1L) stop("multicolumn levels matrix requires column names", call. = FALSE)
    if (anyDuplicated(levels)) stop("levels matrix has duplicated rows", call. = FALSE)
    values <- as.data.frame(levels, stringsAsFactors = FALSE)
    names(values) <- raw
  } else {
    if (length(parts) > 1L) stop("levels should be a data frame or matrix", call. = FALSE)
    values <- data.frame(unique(levels), check.names = FALSE)
    names(values) <- raw
  }
  if (!nrow(values)) stop("levels must not be empty", call. = FALSE)
  for (i in which(categorical)) {
    if (anyNA(match(values[[i]], xlevels[[parts[i]]]))) stop("invalid level for term ", raw[i], call. = FALSE)
  }
  list(values = values, raw = raw, parts = parts)
}

.yates_factorial <- function(frame, names, xlevels, categorical) {
  if (!all(categorical[names])) stop("population=factorial only applies if all the adjusting terms are categorical", call. = FALSE)
  grid <- if (length(names)) expand.grid(xlevels[names], stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE) else data.frame(.row = 1L)
  result <- frame[rep(1L, nrow(grid)), , drop = FALSE]
  rownames(result) <- NULL
  for (name in names) {
    result[[name]] <- if (is.factor(frame[[name]])) {
      factor(grid[[name]], levels = xlevels[[name]], ordered = is.ordered(frame[[name]]))
    } else grid[[name]]
  }
  result
}

.yates_matrices <- function(prepared, selection, population) {
  Terms <- prepared$terms
  frame <- prepared$frame
  classes <- attr(Terms, "dataClasses")[rownames(attr(Terms, "factors"))]
  categorical <- classes %in% c("factor", "character", "ordered")
  names(categorical) <- names(classes)
  adjusting <- setdiff(names(classes), selection$parts)
  if (is.data.frame(population)) {
    pdata <- population
  } else if (population == "data") {
    pdata <- frame
  } else if (population == "factorial") {
    pdata <- .yates_factorial(frame, adjusting, prepared$xlevels, categorical)
  } else {
    categories <- adjusting[categorical[adjusting]]
    continuous <- setdiff(adjusting, categories)
    if (!length(categories)) {
      pdata <- frame
    } else {
      pdata <- .yates_factorial(frame, categories, prepared$xlevels, categorical)
      if (length(continuous)) {
        k <- rep(seq_len(nrow(frame)), nrow(pdata))
        pdata <- pdata[rep(seq_len(nrow(pdata)), each = nrow(frame)), , drop = FALSE]
        for (name in continuous) {
          pdata[[name]] <- if (is.matrix(frame[[name]])) frame[[name]][k, , drop = FALSE] else frame[[name]][k]
        }
      }
    }
  }
  if (!nrow(pdata)) stop("population must have at least one row", call. = FALSE)
  values <- selection$values
  names(values) <- selection$raw
  for (i in seq_along(selection$parts)) {
    part <- selection$parts[i]
    if (categorical[[part]] && identical(part, selection$raw[i])) {
      values[[i]] <- factor(values[[i]], levels = prepared$xlevels[[part]], ordered = is.ordered(frame[[part]]))
    }
  }
  if (!is.null(attr(pdata, "terms"))) {
    # Select evaluated variables directly; keep fitted predvars, including spline knots.
    index <- match(selection$parts, rownames(attr(Terms, "factors")))
    selected_terms <- Terms
    attr(selected_terms, "variables") <- as.call(c(quote(list), as.list(attr(Terms, "variables"))[-1L][index]))
    predvars <- attr(Terms, "predvars")
    if (!is.null(predvars)) attr(selected_terms, "predvars") <- as.call(c(quote(list), as.list(predvars)[-1L][index]))
    attr(selected_terms, "dataClasses") <- classes[index]
    attr(selected_terms, "factors") <- attr(Terms, "factors")[index, , drop = FALSE]
    selected_frame <- stats::model.frame(selected_terms, values, xlev = prepared$xlevels[selection$parts], na.action = stats::na.pass)
  }
  lapply(seq_len(nrow(values)), function(i) {
    data <- pdata
    if (is.null(attr(pdata, "terms"))) {
      for (name in selection$raw) data[[name]] <- rep(values[[name]][i], nrow(data))
    } else {
      for (name in names(selected_frame)) {
        data[[name]] <- if (is.matrix(selected_frame[[name]])) {
          selected_frame[[name]][rep(i, nrow(data)), , drop = FALSE]
        } else rep(selected_frame[[name]][i], nrow(data))
      }
    }
    stats::model.matrix(Terms, data, xlev = prepared$xlevels, contrasts.arg = prepared$contrasts)
  })
}

.yates_test_matrix <- function(rows, sigma2 = NULL) {
  result <- t(vapply(rows, function(row) {
    value <- c(chisq = row$chisq, df = if (is.null(row$df)) NA_real_ else row$df)
    if (!is.null(sigma2)) value <- c(value, ss = row$ss)
    value
  }, numeric(if (is.null(sigma2)) 2L else 3L)))
  rownames(result) <- vapply(rows, function(row) row$name, character(1))
  result[is.nan(result)] <- NA_real_
  result
}

.yates_sgtt <- function(prepared, selection, beta, covariance, assign, sigma2, cox, fitted_terms) {
  contrasts <- prepared$contrasts
  if (!all(vapply(contrasts, function(x) is.character(x) && x %in% c("contr.SAS", "contr.treatment"), logical(1)))) {
    stop("yates sgtt method can only handle contr.SAS or contr.treatment", call. = FALSE)
  }
  factor_names <- names(contrasts)
  full <- lapply(seq_along(factor_names), function(i) {
    levels <- prepared$xlevels[[factor_names[i]]]
    value <- diag(length(levels))
    dimnames(value) <- list(levels, levels)
    if ((i > 1L || attr(prepared$terms, "intercept") == 1L) && contrasts[[i]] == "contr.treatment") {
      value <- value[, c(seq.int(2L, length(levels)), 1L), drop = FALSE]
    }
    value
  })
  names(full) <- factor_names
  x <- stats::model.matrix(prepared$terms, prepared$frame, xlev = prepared$xlevels, contrasts.arg = full)
  xassign <- attr(x, "assign")
  retain <- xassign %in% c(0L, fitted_terms)
  x <- x[, retain, drop = FALSE]
  xassign <- xassign[retain]
  factors <- attr(prepared$terms, "factors")
  categorical <- attr(prepared$terms, "dataClasses")[rownames(factors)] %in% c("factor", "character", "ordered")
  tcat <- colSums(factors[!categorical, , drop = FALSE]) == 0L
  share <- crossprod(factors)
  adjustments <- lapply(seq_len(ncol(factors)), function(i) {
    if (!tcat[i]) return(list())
    as.list(unname(which(share[i, ] > 0L & tcat & seq_len(ncol(factors)) > i)))
  })
  keep <- match(selection$parts, colnames(factors))
  if (anyNA(keep)) stop("sgtt requires fitted main-effect terms", call. = FALSE)
  tests <- Map(function(i, name) reticulate::tuple(as.integer(i), name), keep, selection$parts)
  result <- .validation_attr("yates_sgtt")(
    x, as.list(as.integer(xassign)), adjustments, as.list(unname(beta)), covariance,
    as.list(as.integer(assign)), tests, sigma2 = sigma2, include_intercept = !cox
  )
  sas <- .as_numeric_matrix(result$sas)
  columns <- as.integer(unlist(result$columns)) + 1L
  dimnames(sas) <- list(paste0("L", columns), colnames(x)[columns])
  list(test = .yates_test_matrix(result$test, sigma2), SAS = sas)
}


.yates_simulation <- function(matrices, beta, covariance, means, estimable,
                              setup, nsim, test, terms) {
  if (length(nsim) != 1L || !is.numeric(nsim) || !is.finite(nsim) ||
      nsim < 2 || nsim != floor(nsim) || nsim > .Machine$integer.max) {
    stop("nsim must be an integer of at least two", call. = FALSE)
  }
  builtin <- attr(setup, "survivalr_prediction", exact = TRUE)
  arguments <- list(
    xmatlist = matrices, beta = as.list(unname(beta)), vmat = covariance,
    means = as.list(means), estimable = as.list(unname(estimable)),
    nsim = as.integer(nsim), test = test,
    term = if (length(terms) == 1L) terms else "global",
    normal_draws = function(n, p) matrix(stats::rnorm(n * p), nrow = n, ncol = p)
  )
  if (!is.null(builtin)) {
    if (builtin$kind == "survival") {
      arguments$time <- as.list(builtin$baseline$time)
      arguments$cumhaz <- as.list(builtin$baseline$cumhaz)
      arguments$rmean <- builtin$rmean
      arguments$conf_int <- builtin$baseline$conf.int
    }
    return(do.call(.validation_attr(paste0("yates_", builtin$kind)), arguments))
  }
  if (is.function(setup)) {
    predfun <- setup
  } else if (is.list(setup) && is.function(setup$predict) && is.function(setup$summary)) {
    predfun <- setup$predict
  } else {
    stop("the prediction should be a function, or a list with two functions", call. = FALSE)
  }
  xall <- do.call(rbind, matrices)
  first <- TRUE
  arguments$predict <- function(eta) {
    eta <- matrix(as.numeric(eta), ncol = 1L, dimnames = list(rownames(xall), NULL))
    value <- if (first) {
      first <<- FALSE
      predfun(eta, xall)
    } else predfun(eta)
    if (!is.numeric(value)) {
      stop("prediction function should return a vector or matrix", call. = FALSE)
    }
    if (is.null(dim(value))) value <- matrix(value, ncol = 1L)
    value
  }
  do.call(.validation_attr("yates_predict"), arguments)
}

yates <- function(fit, term, population = c("data", "factorial", "sas"),
                  levels, test = c("global", "trend", "pairwise"),
                  predict = "linear", options, nsim = 200,
                  method = c("direct", "sgtt")) {
  Call <- match.call()
  if (missing(fit)) stop("a fit argument is required", call. = FALSE)
  if (missing(term)) stop("a term argument is required", call. = FALSE)
  Terms <- try(stats::delete.response(stats::terms(fit)), silent = TRUE)
  if (inherits(Terms, "try-error")) stop("the fit does not have a terms structure", call. = FALSE)
  if (inherits(fit, c("coxphms", "survival_py_coxphms"))) stop("multi-state coxph not yet supported", call. = FALSE)
  if (is.list(predict) || is.function(predict)) stop("user written prediction functions are not yet supported", call. = FALSE)
  setup_args <- list(fit = fit)
  if (!missing(predict)) setup_args$predict <- predict
  if (!missing(options)) setup_args$options <- options
  setup <- do.call(yates_setup, setup_args)
  if (is.null(setup)) predict <- "linear"
  method <- match.arg(tolower(method), c("direct", "sgtt"))
  if (method == "sgtt" && missing(population)) population <- "sas"
  if (!is.data.frame(population)) {
    if (!is.character(population)) stop("the population argument must be a data frame or character", call. = FALSE)
    population <- match.arg(tolower(population[1L]), c("data", "factorial", "sas", "empirical", "yates"))
    if (population == "empirical") population <- "data"
    if (population == "yates") population <- "factorial"
  }
  linear <- identical(predict, "linear") || is.null(setup)
  if (method == "sgtt" && (!identical(population, "sas") || !linear)) {
    stop("sgtt method only applies if population = sas and predict = linear", call. = FALSE)
  }
  test <- match.arg(test)
  prepared <- .yates_frame(fit, Terms)
  selection <- .yates_levels(prepared$terms, term, prepared$xlevels, if (missing(levels)) NULL else levels)
  matrices <- .yates_matrices(prepared, selection, population)
  own <- inherits(fit, "survival_py_model")
  beta <- if (own) stats::coef(fit) else stats::coef(fit, complete = TRUE)
  kept <- !is.na(beta)
  covariance <- if (own) stats::vcov(fit) else stats::vcov(fit, complete = FALSE)
  if (all(names(beta)[kept] %in% rownames(covariance))) {
    covariance <- covariance[names(beta)[kept], names(beta)[kept], drop = FALSE]
  } else if (nrow(covariance) == length(beta)) {
    covariance <- covariance[kept, kept, drop = FALSE]
  }
  xold <- stats::model.matrix(fit)
  cox <- inherits(fit, c("coxph", "survival_py_coxph"))
  columns <- match(names(beta), colnames(matrices[[1L]]))
  if (anyNA(columns)) stop("population design does not match fitted coefficients", call. = FALSE)
  matrices <- lapply(matrices, function(x) x[, columns, drop = FALSE])
  estimable <- if (all(kept)) rep(TRUE, length(matrices)) else {
    .validation_attr("yates_estimable")(
      if (cox) lapply(matrices, function(x) cbind(1, x)) else matrices,
      xold, intercept = cox
    )
  }
  matrices <- lapply(matrices, function(x) x[, kept, drop = FALSE])
  means <- if (cox) as.numeric(fit$means)[kept] else rep(0, sum(kept))
  sigma2 <- if (identical(class(fit)[1L], "lm")) summary(fit)$sigma^2 else NULL
  if (linear) {
    weights <- NULL
    if (identical(population, "data")) {
      weights <- stats::model.extract(prepared$frame, "weights")
      if (is.null(weights)) {
        id <- stats::model.extract(prepared$frame, "id")
        if (!is.null(id)) weights <- 1 / as.numeric(table(id)[as.character(id)])
      }
    }
    cmat <- .as_numeric_matrix(.validation_attr("yates_population_means")(
      matrices, if (is.null(weights)) NULL else as.list(unname(weights))
    ))
    result <- .validation_attr("yates")(
      cmat, as.list(unname(beta[kept])), covariance, offset = -sum(means * beta[kept]),
      sigma2 = sigma2, estimable = as.list(unname(estimable)), test = test
    )
  } else {
    result <- .yates_simulation(matrices, beta[kept], covariance, means, estimable,
                                setup, nsim, test, selection$parts)
  }
  estimate <- selection$values
  estimate$pmm <- vapply(result$estimate, function(x) x$pmm, numeric(1))
  estimate$std <- vapply(result$estimate, function(x) x$std, numeric(1))
  estimate$pmm[is.nan(estimate$pmm)] <- NA_real_
  tests <- .yates_test_matrix(result$test, if (linear) sigma2 else NULL)
  if (test == "pairwise" && nrow(estimate) == 2L) {
    rownames(tests) <- if (linear || length(selection$parts) > 1L) "global" else selection$parts
  }
  mvar <- if (any(estimable)) .as_numeric_matrix(result$mvar) else NA_real_
  output <- list(estimate = estimate, test = tests, mvar = mvar)
  if (linear && any(estimable)) {
    colnames(cmat) <- names(beta)[kept]
    output$cmat <- cmat
  }
  if (method == "sgtt") {
    sgtt <- .yates_sgtt(prepared, selection, beta[kept], covariance, attr(xold, "assign")[kept], sigma2, cox, unique(attr(xold, "assign")))
    output$test <- sgtt$test
    output$SAS <- sgtt$SAS
  }
  if (!is.null(result$prediction_mean)) {
    mean <- .as_numeric_matrix(result$prediction_mean)
    variance <- .as_numeric_matrix(result$prediction_variance)
    if (is.list(setup)) output$summary <- setup$summary(mean, variance) else {
      output$pmm <- mean
      output$mvar2 <- variance
    }
  }
  if (!is.null(result$summary)) {
    baseline <- attr(setup, "survivalr_prediction", exact = TRUE)$baseline
    for (name in c("surv", "cumhaz", "std_err", "lower", "upper")) {
      value <- .as_numeric_matrix(.result_field(result$summary, name))
      if (!length(baseline$time)) value <- matrix(numeric(), 0L, nrow(estimate))
      baseline[[if (name == "std_err") "std.err" else name]] <- value
    }
    baseline$std.chaz <- baseline$std.err
    output$summary <- baseline
  }
  output$call <- Call
  class(output) <- "yates"
  output
}
