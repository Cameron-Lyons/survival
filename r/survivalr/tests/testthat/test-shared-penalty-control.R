test_that("native controller trajectories preserve R histories and trace output", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (kind in c("gam", "df", "aic", "gauss")) {
    actual <- get(paste0(".frailty_control", kind), asNamespace("survivalr"))
    expected <- get(paste0("frailty.control", kind), asNamespace("survival"))
    for (initial in c(FALSE, TRUE)) {
      opt <- list(eps2 = 1e-6, trace = TRUE)
      if (initial) opt$init <- c(.1, 2)
      if (kind == "df") opt <- c(opt, list(df = 1.7, thetas = 0, dfs = 4, guess = .7))
      if (kind == "aic") opt <- c(opt, list(lower = 0, upper = 3, caic = initial))
      old <- actual(opt, 0L)
      ref <- expected(opt, 0L)
      expect_equal(old, ref)
      for (iter in seq_len(6L)) {
        theta <- ref$theta
        inputs <- switch(kind,
          gam = list(group = c(1, 1, 2, 3, 3, 3), status = c(1, 1, 0, 1, 0, 1),
                     loglik = -100 - 3 * (theta - .7)^2),
          df = list(df = 4 / (1 + theta)),
          aic = list(n = 50, df = 2 + theta / 2, loglik = -100 - 3 * (theta - .7)^2 + iter * 1e-6),
          gauss = list(fcoef = sqrt(.7) * c(-1, 1, -1, 1), trH = theta * .2 / (1 + theta), loglik = -100))
        actual_trace <- capture.output(old <- do.call(actual, c(list(opt, iter, old), inputs)))
        expected_trace <- capture.output(ref <- do.call(expected, c(list(opt, iter, ref), inputs)))
        expect_equal(old, ref, tolerance = 1e-8, info = paste(kind, initial, iter))
        expect_equal(actual_trace, expected_trace, info = paste(kind, initial, iter))
        if (isTRUE(ref$done)) break
      }
    }
  }
})

test_that("gamma constructor corrections work for dense and sparse groups", {
  for (method in c("fixed", "em", "aic", "df")) for (dense in c(FALSE, TRUE)) {
    opt <- switch(method, fixed = list(theta = .4), em = list(), aic = list(method = "aic"), df = list(df = 1.7))
    group <- rep(1:4, each = 3)
    actual_term <- do.call(frailty.gamma, c(list(group), opt))
    ref_term <- do.call(survival::frailty.gamma, c(list(group), opt))
    actual <- attr(actual_term, "cfun")
    expected <- attr(ref_term, "cfun")
    parms <- attr(ref_term, "cparm")
    old <- actual(parms, 0L)
    ref <- expected(parms, 0L)
    expect_equal(old, ref)
    if (dense) group <- model.matrix(~ factor(group) - 1)
    for (iter in seq_len(if (method == "fixed") 1L else 4L)) {
      # Match cargs positionally, including gamma's penalized and unpenalized likelihoods.
      inputs <- list(group, rep(c(1, 1, 0), 4), -100 - 3 * (ref$theta - .7)^2)
      if (method == "df") inputs <- c(list(4 * ref$theta / (1 + ref$theta)), inputs)
      if (method == "aic") inputs <- c(inputs, list(8, 2 + ref$theta / 2, -90 - 3 * (ref$theta - .7)^2))
      old <- do.call(actual, c(list(parms, iter, old), inputs))
      ref <- do.call(expected, c(list(parms, iter, ref), inputs))
      expect_equal(old, ref, tolerance = 1e-8, info = paste(method, dense, iter))
      if (isTRUE(ref$done)) break
    }
  }
})

test_that("constructors and AFT fitting work with reference controllers disabled", {
  data <- list(x = cbind(Intercept = 1,
    age = c(2, -1, 3, 0, 1, 2, -2, 0, 3, 1, -1, 2),
    group = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0)),
    y = cbind(c(1.2, 2.5, .9, 3, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9),
              c(1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1)))
  cases <- list()
  for (kind in c("ridge", "spline", "gamma", "gaussian", "t")) {
    constructor <- switch(kind, ridge = "ridge", spline = "pspline", paste0("frailty.", kind))
    options <- switch(kind,
      ridge = list(data$x[, 2:3], df = 1),
      spline = list(seq(.1, 1.2, length.out = 12), df = 2, nterm = 4),
      gamma = list(rep(1:3, 4), sparse = TRUE),
      gaussian = list(rep(1:3, 4), sparse = TRUE),
      t = list(rep(1:3, 4), df = 1, sparse = TRUE))
    term <- do.call(get(constructor, asNamespace("survivalr")), options)
    reference <- do.call(getExportedValue("survival", constructor), options)
    x <- cbind(Intercept = 1, term)
    args <- list(x = x, y = data$y, weights = NULL, offset = NULL, init = NULL,
      controlvals = survival::survreg.control(), dist = "gaussian",
      pcols = list(2:ncol(x)), pattr = list(attributes(reference)),
      assign = list(Intercept = 1, term = 2:ncol(x)))
    expected <- do.call(survival::survpenal.fit, args)
    args$pattr <- list(attributes(term))
    cases[[kind]] <- list(args = args, expected = expected, constructor = constructor, options = options)
  }
  blocked <- function(...) stop("reference controller called")
  testthat::local_mocked_bindings(
    frailty.controlgam = blocked, frailty.controldf = blocked,
    frailty.controlaic = blocked, frailty.controlgauss = blocked,
    frailty.gammacon = blocked, frailty.brent = blocked, .package = "survival"
  )
  for (kind in names(cases)) {
    case <- cases[[kind]]
    term <- do.call(get(case$constructor, asNamespace("survivalr")), case$options)
    case$args$pattr <- list(attributes(term))
    actual <- do.call(survpenal.fit, case$args)
    expect_length(actual$printfun, length(case$expected$printfun))
    expect_true(all(vapply(actual$printfun, function(x) is.null(x) || is.function(x), logical(1))))
    for (field in setdiff(names(case$expected), "printfun")) {
      comparison <- all.equal(actual[[field]], case$expected[[field]], tolerance = 4e-7, scale = 1)
      expect_true(isTRUE(comparison), info = paste(kind, field, paste(comparison, collapse = "; ")))
    }
  }
})
