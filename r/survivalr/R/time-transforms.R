.time_transform_callback <- function(fun) {
  force(fun)
  callback <- function(x, time, riskset, weights, input_levels, status) {
    if (!is.null(input_levels)) x <- factor(unlist(x), levels = unlist(input_levels))
    value <- fun(x, time, riskset, weights)
    if (is.factor(value)) {
      if (inherits(value, "coxph.penalty")) {
        stop("factor-valued time-transform penalties are not supported", call. = FALSE)
      }
      if (nlevels(value) < 2L) {
        stop("contrasts can be applied only to factors with 2 or more levels", call. = FALSE)
      }
      contrast <- stats::contrasts(value)
      return(list(`_survival_tt_kind` = "factor", values = as.character(value),
                  levels = as.list(levels(value)), contrasts = unname(contrast),
                  contrast_names = as.list(if (is.null(colnames(contrast)))
                    as.character(seq_len(ncol(contrast))) else colnames(contrast))))
    }
    if (is.matrix(value)) {
      result <- list(`_survival_tt_kind` = "matrix", values = unname(value),
                     names = if (is.null(colnames(value))) NULL else as.list(colnames(value)))
      if (inherits(value, "coxph.penalty")) {
        # The controller's x/status requests refer to this expanded risk-set frame.
        if (is.null(status)) stop("time transform penalty is missing event status", call. = FALSE)
        controller <- .survpenal_controller(attributes(value), seq_len(ncol(value)), value, status)
        result$penalty <- controller$penalty
        result$penalty_names <- if (is.null(attr(value, "varname"))) NULL else as.list(attr(value, "varname"))
        result$history <- controller$history
        if (is.function(controller$printfun)) {
          result$report <- function(coef, var, var2, df, digits) {
            previous <- options(digits = as.integer(digits))
            on.exit(options(previous))
            as_matrix <- function(rows) do.call(rbind, lapply(rows, unlist))
            report <- controller$printfun(unlist(coef), as_matrix(var), as_matrix(var2),
                                         df, controller$history())
            tab <- report$coef
            names <- if (is.matrix(tab)) rownames(tab) else NULL
            if (!is.matrix(tab)) tab <- matrix(tab, nrow = 1L)
            list(coefficients = lapply(seq_len(nrow(tab)), function(i) as.list(tab[i, ])),
                 names = if (is.null(names)) rep(list(NULL), nrow(tab)) else as.list(names),
                 history = as.list(report$history))
          }
        }
      }
      return(result)
    }
    value
  }
  .pybridge_attr("_r_time_transform")(callback)
}

.time_transform_functions <- function(tt) {
  if (is.function(tt)) return(.time_transform_callback(tt))
  if (is.list(tt)) return(lapply(tt, function(fun) {
    if (is.function(fun)) .time_transform_callback(fun) else fun
  }))
  tt
}
