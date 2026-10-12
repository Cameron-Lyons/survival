reference_survfit_complete_event_grid <- local({
  ns <- asNamespace("survival")
  km <- get("survfitKM", ns)
  replace_grid <- function(expr) {
    if (identical(expr, quote(x == i & position > 1))) {
      return(quote(x == i & (position > 1 | y[, 3] == 1)))
    }
    if (is.call(expr)) {
      for (j in seq_along(expr)) expr[[j]] <- replace_grid(expr[[j]])
    }
    expr
  }
  body(km) <- replace_grid(body(km))
  fit <- get("survfit.formula", ns)
  environment(fit) <- list2env(list(survfitKM = km), parent = ns)
  fit
})
