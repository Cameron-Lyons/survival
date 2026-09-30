# R callback signatures and history attributes around the shared Rust searches.
.penalty_control <- function(method, opt, iter, old = NULL, ..., gamma = FALSE) {
  eps <- opt$eps
  if (!length(eps)) eps <- if (method == "df") 0.1 else 1e-5
  options <- list(method = method, eps = eps)
  if (method == "gamma") options$theta <- opt$theta
  if (method %in% c("gamma", "gaussian", "aic")) options$init <- opt$init
  if (method == "df") {
    options <- c(options, list(target_df = opt$df, thetas = as.list(opt$thetas),
                              dfs = as.list(opt$dfs), guess = opt$guess))
  }
  if (method == "aic") {
    options$lower <- if (is.null(opt$lower)) 0 else opt$lower
    options$upper <- opt$upper
    options$caic <- isTRUE(as.logical(opt$caic))
  }
  if (!is.null(options$init)) options$init <- as.list(options$init)
  if (gamma) options$gamma_correction <- TRUE
  if (iter == 0L) {
    state <- .call_r_api("_penalty_control", options, as.integer(iter))
    theta <- state$theta
    if (method == "df") theta <- opt$guess else
      if (method %in% c("gamma", "gaussian") && !is.null(opt$theta)) theta <- opt$theta else
        if (!is.null(opt$init)) names(theta) <- names(opt$init[1L])
    if (method %in% c("gamma", "gaussian")) return(list(theta = theta))
    out <- list(theta = theta, done = FALSE)
    if (method == "df") out$history <- cbind(thetas = opt$thetas, dfs = opt$dfs)
    return(out)
  }
  history <- old$history
  if (!is.null(history) && !is.matrix(history)) history <- matrix(history, nrow = 1L)
  previous <- list(theta = as.numeric(old$theta), history = history,
                   half = if (is.null(old$half)) NULL else as.integer(old$half))
  state <- .call_r_api("_penalty_control", options, as.integer(iter), previous, list(...))
  out <- list(theta = state$theta, done = state$done)
  if (length(state$row)) {
    row <- as.numeric(state$row)
    if (method == "df") {
      thetas <- c(old$history[, 1], old$theta)
      out$history <- cbind(thetas = thetas,
                           dfs = c(old$history[, 2], row[[2L]]))
      if (!is.null(state$theta_history_index)) {
        names(out$theta) <- names(thetas[state$theta_history_index + 1L])
      }
    } else if (iter == 1L) {
      names(row) <- as.character(state$columns)
      out$history <- row
    } else {
      out$history <- rbind(old$history, as.vector(row))
    }
  }
  if (!is.null(state$half)) out$half <- as.numeric(state$half)
  if (!is.null(state$c_loglik)) {
    out$c.loglik <- state$c_loglik
    names(out$c.loglik) <- names(list(...)$loglik)
  }
  if (method == "gamma" && !is.null(opt$theta)) names(out$theta) <- names(old$theta)
  if (method %in% c("gamma", "gaussian") && length(out$history)) {
    if (iter == 1L) {
      names(out$theta) <- if (!is.null(opt$init)) names(opt$init[2L]) else
        if (method == "gaussian") names(old$theta) else NULL
    } else if (iter == 2L) {
      inherits_label <- if (method == "gamma") out$history[2L, 3L] >= out$history[1L, 3L] + 1 else
        all(out$history[, 2L] > 0) || all(out$history[, 2L] < 0)
      if (inherits_label) names(out$theta) <- names(out$history[2L, 1L])
    } else {
      column <- if (method == "gamma") 3L else 2L
      names(out$done) <- names(out$history[iter, column])
    }
  }
  trace <- isTRUE(as.logical(opt$trace)) && switch(method,
    gamma = is.null(opt$theta) && iter >= 2L,
    gaussian = iter >= 2L,
    aic = iter >= 3L,
    df = nrow(out$history) > 2L)
  if (trace) {
    print(out$history)
    prefix <- if (method == "df") {
      if (out$half > 0) "  bisect:new theta=" else "  new theta="
    } else "    new theta="
    theta <- if (method %in% c("gamma", "gaussian") && iter == 2L) out$theta else format(out$theta)
    cat(prefix, theta, "\n\n")
  }
  out
}

.penalty_group_events <- function(group, status) {
  if (is.matrix(group)) group <- c(group %*% seq_len(ncol(group)))
  as.list(as.numeric(tapply(status, group, sum)))
}

.frailty_controlgam <- function(opt, iter, old, group, status, loglik) {
  if (iter == 0L) return(.penalty_control("gamma", opt, iter))
  .penalty_control("gamma", opt, iter, old, loglik = loglik,
                   events_by_group = if (old$theta == 0) NULL else .penalty_group_events(group, status))
}

.frailty_controldf <- function(parms, iter, old, df) {
  if (iter == 0L) return(.penalty_control("df", parms, iter))
  .penalty_control("df", parms, iter, old, df = df)
}

.frailty_controlaic <- function(parms, iter, old, n, df, loglik) {
  if (iter == 0L) return(.penalty_control("aic", parms, iter))
  .penalty_control("aic", parms, iter, old, neff = n, df = df, plik = loglik)
}

.frailty_controlgauss <- function(opt, iter, old, fcoef, trH, loglik) {
  if (iter == 0L) return(.penalty_control("gaussian", opt, iter))
  .penalty_control("gaussian", opt, iter, old, coef = as.list(fcoef), trh = trH)
}

.frailty_gamma_df_cfun <- function(opt, iter, old, df, group, status, loglik) {
  if (iter == 0L) return(.penalty_control("df", opt, iter, gamma = TRUE))
  .penalty_control("df", opt, iter, old, df = df, loglik = loglik, gamma = TRUE,
                   events_by_group = if (old$theta == 0) NULL else .penalty_group_events(group, status))
}

.frailty_gamma_aic_cfun <- function(opt, iter, old, group, status, loglik, n, df, plik) {
  if (iter == 0L) return(.penalty_control("aic", opt, iter, gamma = TRUE))
  .penalty_control("aic", opt, iter, old, neff = n, df = df, plik = plik,
                   loglik = loglik, gamma = TRUE,
                   events_by_group = if (old$theta == 0) NULL else .penalty_group_events(group, status))
}
