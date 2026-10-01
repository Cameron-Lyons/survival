#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/penalty_control_reference.json"
cases <- list()
rows <- function(x) {
  if (is.null(x)) return(list())
  if (!is.matrix(x)) x <- matrix(x, nrow = 1L)
  lapply(seq_len(nrow(x)), function(i) I(as.numeric(x[i, ])))
}
add <- function(name, method, opt, response, steps = 7L) {
  control <- get(paste0("frailty.control", switch(method, gamma = "gam", gaussian = "gauss", method)),
                 asNamespace("survival"))
  old <- control(opt, 0L)
  start <- old
  path <- list()
  for (iter in seq_len(steps)) {
    input <- response(old$theta, iter)
    call <- switch(method,
      gamma = list(group = seq_along(input$events_by_group), status = input$events_by_group,
                   loglik = input$loglik),
      df = list(df = input$df),
      aic = list(n = input$neff, df = input$df, loglik = input$plik),
      gaussian = list(fcoef = input$coef, trH = input$trh, loglik = 0))
    old <- do.call(control, c(list(opt, iter, old), call))
    stopifnot(is.finite(old$theta), all(is.finite(unlist(old$history))))
    path[[iter]] <- list(input = lapply(input, function(x) if (length(x) > 1L) I(x) else x),
      theta = old$theta, done = old$done, history = rows(old$history),
      c_loglik = old$c.loglik, half = old$half)
    if (isTRUE(old$done)) break
  }
  options <- opt
  if (method == "df") { options$target_df <- opt$df; options$df <- NULL }
  for (field in intersect(names(options), c("init", "thetas", "dfs"))) options[[field]] <- I(options[[field]])
  cases[[length(cases) + 1L]] <<- list(name = name, method = method, options = options,
    initial = list(theta = start$theta, done = isTRUE(start$done), history = rows(start$history)),
    path = path)
}
for (name in c("default", "init", "fixed", "taylor")) {
  opt <- switch(name, default = list(eps = 1e-6), init = list(eps = 1e-6, init = c(.1, 2)),
                fixed = list(theta = .4), taylor = list(theta = 1e-10))
  add(paste0("gamma-", name), "gamma", opt,
      function(theta, iter) list(loglik = -100 - 3 * (theta - .7)^2, events_by_group = c(2, 0, 1, 4, 3)),
      if (name %in% c("fixed", "taylor")) 1L else 7L)
}
add("df-ridge", "df", list(df = 1.7, eps = .001, thetas = 0, dfs = 4, guess = .7),
    function(theta, iter) list(df = 4 / (1 + theta)))
add("df-frailty", "df", list(df = 2.3, eps = .001, thetas = 0, dfs = 0, guess = .4),
    function(theta, iter) list(df = 5 * theta / (1 + theta)))
add("df-spline", "df", list(df = 3, eps = .001, thetas = c(1, 0), dfs = c(1, 8), guess = .6),
    function(theta, iter) list(df = 8 - 7 * theta^(1 / 3)))
for (caic in c(FALSE, TRUE)) for (bounded in c(FALSE, TRUE)) for (n in c(3, 50)) {
  opt <- list(eps = 1e-7, init = c(.1, 1), lower = 0, caic = caic)
  if (bounded) opt$upper <- 1.2
  add(paste("aic", caic, bounded, n, sep = "-"), "aic", opt,
      function(theta, iter) list(plik = -100 - 3 * (theta - .7)^2 + iter * 1e-6, df = 2 + theta / 2, neff = n))
}
for (name in c("default", "init", "increase", "decrease")) {
  opt <- list(eps = 1e-7)
  if (name == "init") opt$init <- c(.1, 1)
  target <- switch(name, increase = 20, decrease = .001, .7)
  add(paste0("gaussian-", name), "gaussian", opt,
      function(theta, iter) list(coef = sqrt(target) * c(-1, 1, -1, 1), trh = theta * .2 / (1 + theta)))
}
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
                cases = cases), output, auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null")
