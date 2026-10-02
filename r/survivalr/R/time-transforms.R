.time_transform_callback <- function(fun) {
  force(fun)
  callback <- function(x, time, riskset, weights, input_levels, status) {
    if (!is.null(input_levels)) x <- factor(unlist(x), levels = unlist(input_levels))
    value <- fun(x, time, riskset, weights)
    if (inherits(value, "coxph.penalty")) {
      return(.time_transform_penalty(value, status))
    }
    if (is.factor(value)) {
      if (nlevels(value) < 2L) {
        stop("contrasts can be applied only to factors with 2 or more levels", call. = FALSE)
      }
      contrast <- stats::contrasts(value)
      declared <- attr(value, "contrasts")
      contrast_label <- if (is.null(declared)) getOption("contrasts")[[if(is.ordered(value)) 2L else 1L]] else
        if (is.character(declared)) declared[[1L]] else NULL
      return(list(`_survival_tt_kind` = "factor", values = as.character(value),
                  levels = as.list(levels(value)), contrasts = unname(contrast), contrast_label = contrast_label,
                  contrast_metadata = if (is.null(contrast_label)) list(data = unname(contrast),
                    rows = rownames(contrast), columns = colnames(contrast)) else NULL,
                  contrast_names = as.list(if (is.null(colnames(contrast)))
                    as.character(seq_len(ncol(contrast))) else colnames(contrast))))
    }
    if (is.matrix(value)) {
      result <- list(`_survival_tt_kind` = "matrix", values = unname(value),
                     names = if (is.null(colnames(value))) NULL else as.list(colnames(value)))
      return(result)
    }
    value
  }
  .pybridge_attr("_r_time_transform")(callback)
}

.time_transform_penalty <- function(value, status) {
  if (is.null(status)) stop("time transform penalty is missing event status", call. = FALSE)
  attribute <- attributes(value)
  sparse <- isTRUE(attribute$sparse)
  basis <- if (is.factor(value) && !sparse) {
    # Cox penalty factors use every group, including the reference group.
    stats::model.matrix(~ value - 1)
  } else if (is.matrix(value)) value else matrix(as.numeric(value), ncol = 1L)
  controller <- .survpenal_controller(attribute, seq_len(ncol(basis)), basis, status)
  result <- list(`_survival_tt_kind` = "matrix", values = unname(basis),
    names = if (is.factor(value) && !sparse) as.list(as.character(seq_len(ncol(basis))))
      else if (is.null(colnames(basis))) NULL else as.list(colnames(basis)),
    penalty = controller$penalty,
    penalty_names = if (is.null(attribute$varname)) NULL else as.list(attribute$varname),
    history = controller$history)
  if (is.factor(value) && !sparse) {
    contrast <- stats::contrasts(value)
    result$contrast_metadata <- list(data = unname(contrast),
      rows = rownames(contrast), columns = colnames(contrast))
  }
  if (is.function(controller$printfun)) {
    result$report <- function(coef, var, var2, df, digits) {
      previous <- options(digits = as.integer(digits))
      on.exit(options(previous))
      as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
      report <- if (sparse) {
        controller$printfun(unlist(coef), unlist(var), , df, controller$history())
      } else controller$printfun(unlist(coef), as_matrix(var), as_matrix(var2),
                                 df, controller$history())
      tab <- report$coef
      names <- if (is.matrix(tab)) rownames(tab) else NULL
      if (!is.matrix(tab)) tab <- matrix(tab, nrow = 1L)
      list(coefficients = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
           names = if (is.null(names)) rep(list(NULL), nrow(tab)) else as.list(names),
           history = as.list(report$history))
    }
  }
  result
}

.time_transform_functions <- function(tt) {
  if (is.function(tt)) return(.time_transform_callback(tt))
  if (is.list(tt)) return(lapply(tt, function(fun) {
    if (is.function(fun)) .time_transform_callback(fun) else fun
  }))
  tt
}
