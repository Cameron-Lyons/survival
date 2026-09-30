.yates_test_data <- function() {
  i <- seq_len(60L)
  data.frame(
    y = 2 + i / 20 + sin(i), time = 2 + i %% 13 + i / 70, status = as.integer(i %% 4 > 0),
    a = factor(rep(c("c", "a", "b"), 20), levels = c("c", "b", "a")),
    b = factor(rep(c("v", "u"), each = 30)), z = 2 + cos(i),
    w = 1 + i %% 5, id = rep(seq_len(30), each = 2)
  )
}

.yates_compare <- function(fit, ..., tolerance = 1e-8) {
  set.seed(928)
  actual <- yates(fit, ...)
  state <- .Random.seed
  next_draws <- rnorm(7)
  set.seed(928)
  expected <- .yates_reference(fit, ...)
  expect_identical(.Random.seed, state)
  expect_identical(rnorm(7), next_draws)
  actual$call <- expected$call
  expect_equal(actual, expected, tolerance = tolerance)
  invisible(actual)
}

.yates_reference <- function(fit, ...) {
  # S3 registrations share the generic name. Resolve stock setup methods
  # explicitly so attaching the bridge cannot substitute its curve summary.
  yates_setup <- function(fit, ...) {
    class <- class(fit)[1L]
    method <- if (class %in% c("coxph", "glm")) {
      get(paste0("yates_setup.", class), envir = asNamespace("survival"))
    } else getS3method("yates_setup", class, optional = TRUE, envir = asNamespace("survival"))
    if (is.null(method)) method <- get("yates_setup.default", envir = asNamespace("survival"))
    method(fit, ...)
  }
  survival::yates(fit, ...)
}

test_that("linear Yates preserves R populations, contrasts, transformations and attributes", {
  data <- .yates_test_data()
  for (formula in list(y ~ a * b, y ~ a * b + z, y ~ a + b + log(z), y ~ a + b + splines::ns(z, 3))) {
    fit <- lm(formula, data, weights = w)
    for (population in c("data", "sas")) for (test in c("global", "pairwise")) {
      # Stock R restores the original matrix dimensions after replicating rows.
      if (population == "sas" && grepl("ns", deparse(formula), fixed = TRUE)) next
      .yates_compare(fit, "a", population = population, test = test)
      .yates_compare(fit, "b", population = population, test = test)
    }
    .yates_compare(fit, "a", population = data[c(1, 3, 8, 9), ])
    if (!"z" %in% all.vars(formula)) .yates_compare(fit, "a", population = "factorial")
  }
  fit <- lm(y ~ a + z, data)
  .yates_compare(fit, "z", levels = c(1.5, 2, 2.5))
  .yates_compare(fit, "a", levels = list(a = c("b", "c")))
  .yates_compare(fit, "a", levels = matrix(c("b", "c"), ncol = 1L))
  .yates_compare(fit, "a", population = "empirical")
  simple <- lm(y ~ a, data)
  expect_equal(yates(simple, "a", population = "yates")$estimate, yates(simple, "a")$estimate)
  fit <- lm(y ~ factor(a) + log(z), data)
  .yates_compare(fit, "a", levels = c("a", "b"))
  .yates_compare(fit, "z", levels = c(1.5, 2.5))
})

test_that("Yates SAS type III tests use the shared kernel", {
  data <- .yates_test_data()
  for (formula in list(y ~ a*b, y ~ a*b+z, y ~ a*b-1)) {
    for (contrast in c("contr.treatment", "contr.SAS")) {
      fit <- lm(formula, data, contrasts = list(a = contrast, b = contrast))
      for (term in c("a", "b", if ("z" %in% all.vars(formula)) "a+b")) .yates_compare(fit, term, method = "sgtt")
    }
  }
})

test_that("GLM response simulations preserve R's RNG kind, state and callback order", {
  data <- .yates_test_data()
  old <- RNGkind()
  on.exit(do.call(RNGkind, as.list(old)), add = TRUE)
  for (kind in c("Inversion", "Box-Muller", "Ahrens-Dieter")) {
    RNGkind(normal.kind = kind)
    for (family in list(gaussian(), gaussian("log"), poisson(), Gamma("log"))) {
      fit <- suppressWarnings(glm(y ~ a+b+z, data, weights = w, family = family))
      .yates_compare(fit, "a", predict = "response", nsim = 17)
      .yates_compare(fit, "b", predict = "response", nsim = 17, test = "pairwise", population = "sas")
    }
  }
  fit <- glm(status ~ a + z, data, family = binomial())
  .yates_compare(fit, "a", predict = "link", nsim = 2)
  .yates_compare(fit, "a", predict = "response", population = data[1:5, ], nsim = 13)
})

test_that("Cox Yates supports stock and Python-backed fitted models", {
  data <- .yates_test_data()
  for (formula in list(Surv(time, status) ~ a + z, Surv(time, status) ~ factor(a) * b + log(z))) {
    reference <- survival::coxph(formula, data, weights = w, id = id, model = TRUE)
    own <- coxph(formula, data, weights = w, id = id, model = TRUE)
    for (predict in c("linear", "risk")) for (population in c("data", "sas")) {
      expected <- .yates_compare(reference, "a", predict = predict, population = population, nsim = 23)
      set.seed(928)
      actual <- yates(own, "a", predict = predict, population = population, nsim = 23)
      actual$call <- expected$call
      expect_equal(actual, expected, tolerance = 1e-7)
    }
  }
})

test_that("Cox survival simulations retain aligned baseline metadata", {
  data <- .yates_test_data()
  fit <- survival::coxph(Surv(time, status) ~ a + z, data)
  for (levels in list(NULL, "a")) for (horizon in c(0, 7, Inf)) {
    set.seed(928)
    actual <- yates(fit, "a", levels = levels, predict = "survival", options = list(rmean = horizon), nsim = 11)
    if (is.null(levels) && horizon > 0) {
      set.seed(928)
      expected <- .yates_reference(fit, "a", predict = "survival", options = list(rmean = horizon), nsim = 11)
      expect_equal(actual$estimate, expected$estimate, tolerance = 1e-8)
      expect_equal(actual$mvar, expected$mvar, tolerance = 1e-8)
      expect_equal(actual$summary$surv, unname(expected$summary$surv[-1L, , drop = FALSE]), tolerance = 1e-8)
    }
    if (horizon == 0) {
      expect_equal(actual$estimate$pmm, rep(0, nrow(actual$estimate)))
      expect_equal(actual$mvar, matrix(0, nrow(actual$estimate), nrow(actual$estimate)))
    }
    expect_equal(actual$summary$cumhaz, -log(actual$summary$surv))
    expect_equal(actual$summary$std.chaz, actual$summary$std.err)
    expect_equal(nrow(actual$summary$surv), length(actual$summary$time))
    expect_equal(ncol(actual$summary$surv), nrow(actual$estimate))
  }
})

test_that("custom S3 prediction matrices and summaries preserve random draw ordering", {
  data <- .yates_test_data()
  fit <- lm(y ~ a + z, data)
  class(fit) <- c("yates_custom_fixture", class(fit))
  for (with_summary in c(FALSE, TRUE)) {
    setup <- function(fit, ...) {
      predict <- function(eta, X) {
        shift <- runif(1)
        if (!missing(X)) expect_equal(nrow(X), length(eta))
        cbind(eta = drop(eta) + shift, square = drop(eta)^2)
      }
      if (with_summary) list(predict = predict, summary = function(mean, variance) list(mean = mean, variance = variance)) else predict
    }
    registerS3method("yates_setup", "yates_custom_fixture", setup, envir = asNamespace("survivalr"))
    registerS3method("yates_setup", "yates_custom_fixture", setup, envir = asNamespace("survival"))
    .yates_compare(fit, "a", predict = "custom", nsim = 19)
  }
})

test_that("reordered variables, joint settings and final numeric term use their fitted columns", {
  data <- .yates_test_data()
  fit <- lm(y ~ a*b+z, data)
  forward <- yates(fit, "a+b")
  reverse <- yates(fit, "b+a")
  key <- paste(forward$estimate$b, forward$estimate$a)
  order <- match(paste(reverse$estimate$b, reverse$estimate$a), key)
  expect_equal(reverse$estimate$pmm, forward$estimate$pmm[order])
  expect_equal(reverse$mvar, forward$mvar[order, order])
  joint <- cbind(b = c("u", "v"), a = c("a", "c"))
  result <- yates(fit, "a+b", levels = joint)
  expect_equal(nrow(result$estimate), 2L)
  expect_equal(result$estimate$a, c("a", "c"))
  expected <- forward$estimate$pmm[match(paste(joint[, "b"], joint[, "a"]), key)]
  expect_equal(result$estimate$pmm, expected)
  fit <- lm(y ~ a+z, data)
  explicit <- yates(fit, "z", levels = c(1.5, 2.5))
  numbered <- yates(fit, 2L, levels = c(1.5, 2.5))
  expect_equal(numbered$estimate, explicit$estimate)
})

test_that("Yates estimability handles aliased columns and no estimable populations", {
  data <- .yates_test_data()
  data$duplicate <- data$z
  fit <- lm(y ~ a + z + duplicate, data)
  .yates_compare(fit, "a")
  result <- yates(fit, "z", levels = c(0, 4))
  expect_true(all(is.na(result$estimate$pmm)))
  expect_true(all(is.na(result$test)))
  expect_identical(result$mvar, NA_real_)
  expect_null(result$cmat)
})

test_that("Yates rejects unsupported and malformed requests", {
  fit <- lm(y ~ a+z, .yates_test_data())
  expect_error(yates(), "fit argument")
  expect_error(yates(fit), "term argument")
  expect_error(yates(fit, "missing"), "not found")
  expect_error(yates(fit, 3), "numeric term")
  expect_error(yates(fit, "z"), "continuous")
  expect_error(yates(fit, "a+z", levels = list(a = "a")), "levels information not found for: z")
  expect_error(yates(fit, "a", levels = "missing"), "invalid level")
  expect_error(yates(fit, "a", levels = list(a = c("a", "a"))), "duplicates")
  expect_error(yates(fit, "a", levels = character()), "must not be empty")
  expect_error(yates(fit, "a", population = "factorial"), "categorical")
  expect_error(yates(fit, "a", population = 1), "data frame or character")
  expect_error(yates(fit, "a", population = "data", method = "sgtt"), "only applies")
  expect_error(yates(fit, "a", test = "trend"), "not supported")
  expect_error(yates(fit, "a", predict = exp), "not yet supported")
  expect_error(yates(fit, "a", population = .yates_test_data()[FALSE, ]), "at least one row")
})

test_that("R Yates does not invoke stock numerical or formula helpers", {
  data <- .yates_test_data()
  fit <- lm(y ~ a*b+z, data)
  glm <- glm(status ~ a+b+z, data, family = binomial())
  bridge <- yates
  unavailable <- function(...) stop("reference Yates helper invoked")
  local_mocked_bindings(yates = unavailable, cmatrix = unavailable, yates_xmat = unavailable,
                       yates_factorial_pop = unavailable, estfun = unavailable, testfun = unavailable,
                       .package = "survival")
  expect_s3_class(bridge(fit, "a"), "yates")
  expect_s3_class(bridge(fit, "a", method = "sgtt"), "yates")
  expect_s3_class(bridge(glm, "a", predict = "response", nsim = 5), "yates")
})

test_that("SAS populations replicate matrix-valued adjusters and empty adjustment sets", {
  data <- .yates_test_data()
  fit <- lm(y ~ a+b+splines::ns(z, 3), data)
  population <- data[rep(seq_len(nrow(data)), nlevels(data$b)), ]
  population$b <- factor(rep(levels(data$b), each = nrow(data)), levels = levels(data$b))
  explicit <- yates(fit, "a", population = population)
  sas <- yates(fit, "a", population = "sas")
  expect_equal(sas$estimate, explicit$estimate)
  expect_equal(sas$mvar, explicit$mvar)
  fit <- lm(y ~ a*b, data)
  sas <- yates(fit, "a+b", method = "sgtt")
  expect_equal(rownames(sas$test), c("a", "b"))
  expect_equal(sas$test[1L, ], yates(fit, "a", method = "sgtt")$test[1L, ])
  expect_equal(sas$test[2L, ], yates(fit, "b", method = "sgtt")$test[1L, ])
})

test_that("AFT Yates uses coefficient covariance without scale parameters", {
  data <- .yates_test_data()
  reference <- survival::survreg(Surv(time, status) ~ a+z, data, model = TRUE)
  own <- survreg(Surv(time, status) ~ a+z, data, model = TRUE)
  expected <- yates(reference, "a")
  actual <- yates(own, "a")
  expect_equal(actual$estimate, expected$estimate, tolerance = 1e-7)
  expect_equal(actual$mvar, expected$mvar, tolerance = 1e-7)
  expect_equal(expected$estimate$pmm, drop(expected$cmat %*% coef(reference)))
  covariance <- vcov(reference)[names(coef(reference)), names(coef(reference))]
  expect_equal(expected$mvar, unname(expected$cmat %*% covariance %*% t(expected$cmat)))
})

test_that("stratified Cox tests select fitted coefficient columns", {
  data <- .yates_test_data()
  fit <- survival::coxph(Surv(time, status) ~ a+z+strata(b), data)
  own <- coxph(Surv(time, status) ~ a+z+strata(b), data, model = TRUE)
  for (method in c("direct", "sgtt")) {
    actual <- yates(fit, "a", method = method)
    native <- yates(own, "a", method = method)
    expect_equal(native$estimate, actual$estimate, tolerance = 1e-7)
    expect_equal(native$test, actual$test, tolerance = 1e-7)
    expect_equal(native$mvar, actual$mvar, tolerance = 1e-7)
    expect_equal(colnames(actual$cmat), names(coef(fit)))
  }
})

test_that("case ids, offsets, alias masks and explicit contrast coding are preserved", {
  data <- .yates_test_data()
  fit <- survival::coxph(Surv(time, status) ~ a+z, data, id = rep(1:20, times = rep(c(1, 5), 10)))
  .yates_compare(fit, "a")
  own <- coxph(Surv(time, status) ~ a+z, data, id = rep(1:20, times = rep(c(1, 5), 10)), model = TRUE)
  expect_equal(yates(own, "a")$estimate, yates(fit, "a")$estimate, tolerance = 1e-7)
  fit <- lm(y ~ a+z+offset(z/10), data, contrasts = list(a = "contr.sum"))
  .yates_compare(fit, "a")
  actual <- yates(fit, "a", population = data[1:9, ])
  cmat <- t(vapply(levels(data$a), function(level) {
    population <- data[1:9, ]
    population$a <- factor(level, levels = levels(data$a))
    colMeans(model.matrix(delete.response(terms(fit)), population, contrasts.arg = fit$contrasts))
  }, numeric(length(coef(fit)))))
  expect_equal(actual$estimate$pmm, unname(drop(cmat %*% coef(fit))))
  expect_equal(actual$mvar, unname(cmat %*% vcov(fit) %*% t(cmat)))
  fit <- glm(y ~ a+z+I(z), data, family = gaussian())
  .yates_compare(fit, "a", predict = "response", nsim = 9)
})
