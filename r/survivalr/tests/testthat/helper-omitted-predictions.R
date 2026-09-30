.omitted_prediction_setup <- function() {
  d <- survival::ovarian
  row.names(d) <- paste0("patient / ", seq_len(nrow(d)))
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
  list(data = d, names = c("四", "two / 2", "six:6", "one"), specs = list(
    cox = c("coxph", "age + rx"), cox_strata = c("coxph", "age + rx + strata(cl)"),
    cox_ridge = c("coxph", "age + ridge(rx, theta = 2)"),
    cox_sparse = c("coxph", "age + frailty(cl, theta = 0.5, sparse = TRUE)"),
    cox_sparse_only = c("coxph", "frailty(cl, theta = 0.5, sparse = TRUE)"),
    cox_factor = c("coxph", "age + cl"),
    aft = c("survreg", "age + rx"), aft_strata = c("survreg", "age + rx + strata(cl)"),
    aft_ridge = c("survreg", "age + ridge(rx, theta = 2)"),
    aft_spline = c("survreg", "pspline(age, df = 3) + rx"),
    aft_fixed = c("survreg", "age + rx"), aft_factor = c("survreg", "age + cl")
  ))
}

.omitted_prediction_fit <- function(name, setup, namespace) {
  spec <- setup$specs[[name]]
  options <- list(as.formula(paste("Surv(futime,fustat) ~", spec[[2L]])), setup$data, x = TRUE)
  if (name == "aft_fixed") options$scale <- 1
  do.call(get(spec[[1L]], envir = asNamespace(namespace)), options)
}

.omitted_prediction_cases <- function(name) {
  cox <- startsWith(name, "cox")
  types <- if (name == "cox_sparse_only") c("lp", "risk", "terms") else if (cox)
    c("lp", "risk", "expected", "survival", "terms") else
    c("response", "link", "terms", "quantile", "uquantile")
  cases <- list()
  for (type in types) {
    probabilities <- if (type %in% c("quantile", "uquantile")) list(.5, c(.1,.5,.9)) else list(NULL)
    for (p in probabilities) for (keep in c(0L,1L,2L,4L)) for (kind in c("NA", "NaN"))
      for (action in c("na.pass", "na.omit", "na.exclude")) for (se in c(FALSE,TRUE))
        for (grouped in if (cox) c(FALSE,TRUE) else FALSE) {
          cases[[length(cases) + 1L]] <- list(
            name = paste(name,type,length(p),keep,kind,action,se,grouped,sep="/"),
            model = name, type = type, p = p, keep = keep, kind = kind,
            na_action = action, se_fit = se, grouped = grouped)
        }
  }
  cases
}

.omitted_prediction_data <- function(setup, case) {
  nd <- setup$data[3:6, ]; row.names(nd) <- setup$names
  gaps <- setdiff(seq_len(4L), c(3L,1L,2L,4L)[seq_len(case$keep)])
  if (case$model %in% c("cox_sparse_only", "cox_factor", "aft_factor")) nd$cl[gaps] <- NA else
    nd$age[gaps] <- if (case$kind == "NA") NA_real_ else NaN
  nd
}

.omitted_prediction_call <- function(object, setup, case, newdata = .omitted_prediction_data(setup, case)) {
  args <- list(object, newdata = newdata, type = case$type,
               na.action = case$na_action, se.fit = case$se_fit)
  if (!is.null(case$p)) args$p <- case$p
  if (case$grouped) args$collapse <- c("b", "a", "b", "a")
  # The penalized wrapper refuses survival; compare the ordinary prediction
  # method on the same fitted state, as in the existing row-reference suite.
  if (case$type == "survival" && inherits(args[[1L]], "coxph.penal")) class(args[[1L]]) <- "coxph"
  do.call(predict, args)
}

.omitted_prediction_capture <- function(fun) {
  messages <- character()
  value <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = messages)
}

.omitted_prediction_expected <- function(object, setup, case) {
  result <- .omitted_prediction_capture(function() .omitted_prediction_call(object, setup, case))
  if (is.list(result$value) && !is.null(result$value$error)) result$raw_error <- result$value$error
  if (case$grouped || case$model == "cox_sparse") {
    ordinary <- object
    if (case$model == "cox_sparse") {
      ordinary$x <- object$x[, "age", drop = FALSE]
      ordinary$terms <- stats::terms(survival::Surv(futime,fustat) ~ age)
      ordinary$assign <- list(age = 1L); ordinary$pterms <- c(age = 0)
      ordinary$means <- object$means[1L]
      class(ordinary) <- "coxph"
    }
    single <- case; single$grouped <- FALSE
    value <- .omitted_prediction_call(ordinary, setup, single)
    if (case$model == "cox_sparse" && case$type == "terms") {
      append <- function(x) {
        frailty <- rep(0, nrow(x))
        if (case$na_action == "na.exclude")
          frailty[setdiff(seq_len(4L), c(3L,1L,2L,4L)[seq_len(case$keep)])] <- NA_real_
        y <- cbind(x, frailty)
        colnames(y) <- c(colnames(x), attr(object$terms, "term.labels")[[2L]])
        y
      }
      if (case$se_fit) {value$fit <- append(value$fit); value$se.fit <- append(value$se.fit)} else value <- append(value)
    }
    if (case$grouped) {
      groups <- c("b", "a", "b", "a")
      if (case$na_action == "na.omit" && case$model != "cox_sparse_only")
        groups <- groups[seq_len(4L) %in% c(3L,1L,2L,4L)[seq_len(case$keep)]]
      aggregate <- function(x, errors = FALSE) {
        x <- if (errors) sqrt(rowsum(x^2, groups)) else rowsum(x, groups)
        if (case$type == "terms" && case$model != "cox_sparse_only") x else drop(x)
      }
      if (case$se_fit) value <- list(fit = aggregate(value$fit), se.fit = aggregate(value$se.fit, TRUE)) else value <- aggregate(value)
    }
    result$value <- value
  }
  if (case$model == "aft_spline" && case$keep == 0L) {
    # Stock splines fail with empty derivatives when every source value is
    # missing. Append a valid control row to obtain the missing basis from R.
    nd <- rbind(.omitted_prediction_data(setup, case), setup$data[3L,,drop=FALSE])
    control <- case; control$na_action <- "na.pass"
    value <- .omitted_prediction_call(object, setup, control, nd)
    trim <- function(x) {
      if (case$na_action == "na.pass") {
        if (is.matrix(x)) return(x[1:4,,drop=FALSE]) else return(x[1:4])
      }
      if (is.matrix(x)) {
        y <- x[integer(),,drop=FALSE]
        if (!is.null(dimnames(y))) dimnames(y) <- list(NULL, colnames(y))
      } else y <- numeric()
      if (case$na_action == "na.exclude") {
        omit <- setNames(seq_len(4L), setup$names); class(omit) <- "exclude"
        y <- stats::naresid(omit, y)
      }
      y
    }
    result$value <- if (case$se_fit) list(fit = trim(value$fit), se.fit = trim(value$se.fit)) else trim(value)
  }
  # Keep the existing repair for stock's zero-row attrassign/1:0 term loops.
  # Reconstruct the empty matrix from independently obtained stock term labels.
  if (is.list(result$value) && !is.null(result$value$error) &&
      startsWith(case$model, "aft") && case$type == "terms" && case$keep == 0L &&
      case$na_action != "na.pass") {
    result$raw_error <- result$value$error
    complete <- case; complete$keep <- 4L; complete$se_fit <- FALSE
    labels <- colnames(.omitted_prediction_call(object, setup, complete))
    empty <- matrix(numeric(), 0L, length(labels), dimnames = list(NULL, labels))
    if (case$na_action == "na.exclude") {
      omit <- setNames(seq_len(4L), setup$names); class(omit) <- "exclude"
      empty <- stats::naresid(omit, empty)
    }
    result$value <- if (case$se_fit) list(fit = empty, se.fit = empty) else empty
  }
  result
}
