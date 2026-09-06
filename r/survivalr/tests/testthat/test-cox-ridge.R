test_that("fixed-theta ridge summaries retain R uncertainty and fractional degrees of freedom", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 2, 3, 4, 4, 5, 6, 7, 8),
    event = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 1),
    x = c(.2, .5, .7, .1, .4, .8, .3, .9, .6, 1.2),
    z = c(.3, 1.2, .4, .8, 1.1, .6, .2, .7, 1.4, .9),
    w = c(1, 2, 1, .5, 1.5, 1, 2, 1, 1, 1)
  )

  for (rhs in c(
    "z + ridge(x, theta=2)",
    "ridge(x, z, theta=2)",
    "ridge(x, theta=2) + ridge(z, theta=.5, scale=FALSE)"
  )) {
    formula <- stats::as.formula(paste("Surv(time, event) ~", rhs))
    reference_formula <- formula
    environment(reference_formula) <- asNamespace("survival")
    bridged <- coxph(formula, data = data, weights = w, robust = FALSE)
    reference <- survival::coxph(reference_formula, data = data, weights = w, robust = FALSE)

    expect_equal(as.numeric(logLik(bridged)), as.numeric(logLik(reference)), tolerance = 1e-8)
    expect_equal(attr(logLik(bridged), "df"), attr(logLik(reference), "df"), tolerance = 1e-8)
    expect_equal(AIC(bridged), AIC(reference), tolerance = 1e-8)
    expect_equal(BIC(bridged), BIC(reference), tolerance = 1e-8)
    expect_equal(unname(extractAIC(bridged)), unname(extractAIC(reference)), tolerance = 1e-8)

    for (scale_value in c(1, 2)) {
      actual <- summary(bridged, conf.int = .9, scale = scale_value)
      expected <- summary(reference, conf.int = .9, scale = scale_value)
      expect_equal(actual$coefficients, expected$coefficients, tolerance = 1e-8)
      expect_equal(actual$conf.int, expected$conf.int, tolerance = 1e-8)
      expect_equal(actual$df, expected$df, tolerance = 1e-8)
      expect_equal(actual$logtest, expected$logtest, tolerance = 1e-8)
      expect_equal(actual$loglik, expected$loglik, tolerance = 1e-8)
      expect_equal(actual$iter, expected$iter)
      expect_null(actual$sctest)
      expect_null(actual$waldtest)
      expect_null(actual$rsq)
      expect_null(actual$used.robust)
    }

    without_confidence <- summary(bridged, conf.int = FALSE)
    grouped <- summary(bridged, terms = TRUE)
    grouped_reference <- summary(reference, terms = TRUE)
    expect_equal(grouped$coefficients, grouped_reference$coefficients, tolerance = 1e-8)
    expect_equal(grouped$conf.int, grouped_reference$conf.int, tolerance = 1e-8)
    expect_equal(grouped$df, grouped_reference$df, tolerance = 1e-8)
    expect_null(without_confidence$conf.int)
    printed <- paste(capture.output(print(without_confidence)), collapse = "\n")
    expect_match(printed, "se2", fixed = TRUE)
    expect_match(printed, "Degrees of freedom for terms=", fixed = TRUE)
    expect_match(printed, "Likelihood ratio test=", fixed = TRUE)
    expect_false(grepl("Score (logrank)", printed, fixed = TRUE))
    expect_false(grepl("Wald test", printed, fixed = TRUE))
  }

  initialized <- coxph(
    Surv(time, event) ~ z + ridge(x, theta = 2), data = data,
    weights = w, robust = FALSE, init = c(.4, -.2), max_iter = 0
  )
  initialized_reference <- survival::coxph(
    survival::Surv(time, event) ~ z + survival::ridge(x, theta = 2), data = data,
    weights = w, robust = FALSE, init = c(.4, -.2),
    control = survival::coxph.control(iter.max = 0)
  )
  expect_equal(
    summary(initialized)$loglik, summary(initialized_reference)$loglik, tolerance = 1e-10
  )
  expect_equal(
    summary(initialized)$logtest, summary(initialized_reference)$logtest, tolerance = 1e-10
  )
})

test_that("automatic ridge selection retains R controller summaries and iteration controls", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 2, 3, 4, 4, 5, 6, 7, 8),
    event = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 1),
    x = c(.2, .5, .7, .1, .4, .8, .3, .9, .6, 1.2),
    z = c(.3, 1.2, .4, .8, 1.1, .6, .2, .7, 1.4, .9),
    w = c(1, 2, 1, .5, 1.5, 1, 2, 1, 1, 1)
  )
  for (rhs in c(
    "z + ridge(x)",
    "ridge(x, z, df=.6, eps=.001)",
    "ridge(x, df=.4) + ridge(z, theta=2)"
  )) {
    formula <- stats::as.formula(paste("Surv(time, event) ~", rhs))
    reference_formula <- formula
    environment(reference_formula) <- asNamespace("survival")
    control <- survival::coxph.control(iter.max = 50, outer.max = 10, eps = 1e-11)
    bridged <- coxph(
      formula, data = data, weights = w, robust = FALSE,
      control = control[c("iter.max", "outer.max", "eps")]
    )
    reference <- survival::coxph(
      reference_formula, data = data, weights = w, robust = FALSE, control = control
    )

    expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 1e-8)
    expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 1e-8)
    expect_equal(as.numeric(logLik(bridged)), as.numeric(logLik(reference)), tolerance = 1e-8)
    expect_equal(attr(logLik(bridged), "df"), attr(logLik(reference), "df"), tolerance = 1e-8)
    expect_equal(AIC(bridged), AIC(reference), tolerance = 1e-8)
    expect_equal(BIC(bridged), BIC(reference), tolerance = 1e-8)
    actual <- summary(bridged, terms = TRUE)
    expected <- summary(reference, terms = TRUE)
    expect_equal(actual$coefficients, expected$coefficients, tolerance = 1e-8)
    expect_equal(actual$conf.int, expected$conf.int, tolerance = 1e-8)
    expect_equal(actual$df, expected$df, tolerance = 1e-8)
    expect_equal(actual$logtest, expected$logtest, tolerance = 1e-8)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-8)
    expect_equal(actual$penalty, unname(reference$penalty), tolerance = 1e-8)
    expect_equal(actual$iter, expected$iter)
  }

  for (limits in list(c(50, 1), c(0, 10))) {
    formula <- Surv(time, event) ~ z + ridge(x, df = .5, eps = .001)
    reference_formula <- formula
    environment(reference_formula) <- asNamespace("survival")
    control <- survival::coxph.control(iter.max = limits[1], outer.max = limits[2], eps = 1e-11)
    bridged <- coxph(
      formula, data = data, weights = w, init = c(.4, -.2), robust = FALSE,
      control = control[c("iter.max", "outer.max", "eps")]
    )
    reference <- survival::coxph(
      reference_formula, data = data, weights = w, init = c(.4, -.2), robust = FALSE,
      control = control
    )
    actual <- summary(bridged)
    expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 1e-8)
    expect_equal(actual$df, unname(reference$df), tolerance = 1e-8)
    expect_equal(actual$loglik, unname(reference$loglik), tolerance = 1e-8)
    expect_equal(actual$penalty, unname(reference$penalty), tolerance = 1e-8)
    expect_equal(actual$iter, unname(reference$iter))
  }
})
