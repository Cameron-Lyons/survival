.snapshot_callback <- function(value) {
  force(value)
  function() .pybridge_attr("_serialize_r_object")(value)
}

.snapshot_python <- function(value) {
  if (!inherits(value, "python.builtin.object")) return(value)
  if (is.null(attr(value, "survival_state", exact = TRUE))) {
    attr(value, "survival_state") <- .Call(C_snapshot_new, .snapshot_callback(value))
    if (!inherits(value, "survival_py_object")) {
      class(value) <- c("survival_py_object", class(value))
    }
  }
  value
}

.restore_python <- function(value) {
  if (!inherits(value, "python.builtin.object") || !reticulate::py_is_null_xptr(value)) {
    return(value)
  }
  state <- attr(value, "survival_state", exact = TRUE)
  if (is.null(state)) {
    stop("This Python object was saved without model state; recreate it before saving with survivalr", call. = FALSE)
  }
  restored <- .pybridge_attr("_unserialize_r_object")(.Call(C_snapshot_get, state))
  list2env(as.list.environment(restored, all.names = TRUE), envir = value)
  .Call(C_snapshot_bind, state, .snapshot_callback(value))
  value
}

# Python Surv values (fit$y, fit$model) reach R as survival's Surv matrix.
.r_value_from_python <- function(value) {
  if (inherits(value, "survival.r._surv.Surv")) return(.as_native_surv(value))
  # Only a dict such as fit$model can hold one; row lists stay unscanned.
  if (is.list(value) && !is.object(value) && !is.null(names(value))) {
    surv <- vapply(value, inherits, logical(1), what = "survival.r._surv.Surv")
    if (any(surv)) value[surv] <- lapply(value[surv], .as_native_surv)
  }
  value
}

`$.survival_py_object` <- function(x, name) {
  x <- .restore_python(x)
  .snapshot_python(.r_value_from_python(NextMethod("$")))
}

`[[.survival_py_object` <- function(x, ...) {
  x <- .restore_python(x)
  .snapshot_python(.r_value_from_python(NextMethod("[[")))
}

`$<-.survival_py_object` <- function(x, name, value) {
  x <- .restore_python(x)
  NextMethod("$<-")
}
