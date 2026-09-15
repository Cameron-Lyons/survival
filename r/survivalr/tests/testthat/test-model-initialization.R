initialization_data <- function() {
  data.frame(
    time = c(1, 2, 3, 4, 5, 6, 7, 8),
    status = c(1, 1, 0, 1, 1, 0, 1, 1),
    x = c(0.2, 1.1, 0.3, -0.2, 0.7, 1.3, -0.5, 0.9),
    z = c(0.8, -0.1, 0.5, 1.2, -0.4, 0.1, 0.7, -0.3)
  )
}

require_initialization_bridge <- function() {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
}

expect_initialization_fit <- function(bridged, reference) {
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 1e-12)
  expect_equal(as.numeric(logLik(bridged)), as.numeric(logLik(reference)), tolerance = 1e-10)
  expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 1e-9)
}

test_that("single-coefficient Cox initialization accepts scalar aliases", {
  require_initialization_bridge()
  data <- initialization_data()
  starting_value <- 0.25
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x, data = data, init = starting_value,
    control = survival::coxph.control(iter.max = 0)
  )

  for (alias in c("init", "initial_beta")) {
    for (value in list(starting_value, c(x = starting_value), list(starting_value))) {
      arguments <- list(Surv(time, status) ~ x, data = data, max_iter = 0)
      arguments[[alias]] <- value
      expect_initialization_fit(do.call(coxph, arguments), reference)
    }
  }
  from_expression <- coxph(
    Surv(time, status) ~ x, data = data, init = starting_value, max_iter = 0
  )
  expect_initialization_fit(from_expression, reference)
})

test_that("fixed-scale intercept AFT initialization accepts scalar aliases", {
  require_initialization_bridge()
  data <- initialization_data()
  reference <- survival::survreg(
    survival::Surv(time, status) ~ 1, data = data, dist = "logistic",
    init = 2, scale = 1.3, control = survival::survreg.control(maxiter = 0)
  )

  for (alias in c("init", "initial", "initial_beta")) {
    for (value in list(2, 2L, c(`(Intercept)` = 2), list(2))) {
      arguments <- list(Surv(time, status) ~ 1, data = data, dist = "logistic",
                        scale = 1.3, max_iter = 0)
      arguments[[alias]] <- value
      expect_initialization_fit(do.call(survreg, arguments), reference)
    }
  }
})

test_that("multiple initial coefficients preserve order and model dimensions", {
  require_initialization_bridge()
  data <- initialization_data()
  cox_reference <- survival::coxph(
    survival::Surv(time, status) ~ x + z, data = data, init = c(0.25, -0.15),
    control = survival::coxph.control(iter.max = 0)
  )
  aft_reference <- survival::survreg(
    survival::Surv(time, status) ~ x, data = data, dist = "logistic",
    init = c(2, 0.25), scale = 1.3, control = survival::survreg.control(maxiter = 0)
  )
  for (alias in c("init", "initial_beta")) {
    arguments <- list(Surv(time, status) ~ x + z, data = data, max_iter = 0)
    arguments[[alias]] <- c(x = 0.25, z = -0.15)
    expect_initialization_fit(do.call(coxph, arguments), cox_reference)
  }
  for (alias in c("init", "initial", "initial_beta")) {
    arguments <- list(Surv(time, status) ~ x, data = data, dist = "logistic",
                      scale = 1.3, max_iter = 0)
    arguments[[alias]] <- c(`(Intercept)` = 2, x = 0.25)
    expect_initialization_fit(do.call(survreg, arguments), aft_reference)
  }

  for (value in list(numeric(), c(0, 1))) {
    expect_error(
      coxph(Surv(time, status) ~ x, data = data, init = value, max_iter = 0),
      "initial_beta has .* values but covariates has 1 columns"
    )
    expect_error(
      survival::coxph(survival::Surv(time, status) ~ x, data = data, init = value),
      "wrong length for init"
    )
    expect_error(
      survreg(Surv(time, status) ~ 1, data = data, init = value,
              scale = 1.3, dist = "logistic", max_iter = 0),
      "initial_beta has .* values but model expects 1"
    )
    expect_error(
      survival::survreg(survival::Surv(time, status) ~ 1, data = data,
                        init = value, scale = 1.3, dist = "logistic"),
      "Wrong length for initial parameters"
    )
  }
  expect_error(
    coxph(Surv(time, status) ~ x, data = data, init = 0, initial_beta = 0),
    "use only one of init or initial_beta"
  )
  expect_error(
    survreg(Surv(time, status) ~ 1, data = data, init = 0, initial = 0),
    "use only one of init, initial, or initial_beta"
  )
})

test_that("explicit NULL initial values do not delete other forwarded arguments", {
  require_initialization_bridge()
  data <- initialization_data()
  # NULL retains the bridge's existing meaning of omitting an optional value.
  # Its position in ... must not change the names or number of other arguments.
  cox_default <- coxph(Surv(time, status) ~ x, data = data, max_iter = 0)
  aft_default <- survreg(Surv(time, status) ~ 1, data = data, dist = "logistic",
                         scale = 1.3, max_iter = 0)
  for (alias in c("init", "initial_beta")) {
    arguments <- list(Surv(time, status) ~ x, data = data, max_iter = 0)
    arguments[alias] <- list(NULL)
    actual <- do.call(coxph, arguments)
    expect_equal(coef(actual), coef(cox_default))
    expect_equal(logLik(actual), logLik(cox_default))
    expect_equal(vcov(actual), vcov(cox_default))
  }
  for (alias in c("init", "initial", "initial_beta")) {
    arguments <- list(Surv(time, status) ~ 1, data = data, dist = "logistic",
                      scale = 1.3, max_iter = 0)
    arguments[alias] <- list(NULL)
    actual <- do.call(survreg, arguments)
    expect_equal(coef(actual), coef(aft_default))
    expect_equal(logLik(actual), logLik(aft_default))
    expect_equal(vcov(actual), vcov(aft_default))
  }
  before_an_alias <- coxph(Surv(time, status) ~ x, data = data,
                          init = NULL, initial_beta = 0.25, max_iter = 0)
  after_an_alias <- coxph(Surv(time, status) ~ x, data = data,
                         initial_beta = 0.25, max_iter = 0, init = NULL)
  expect_equal(coef(before_an_alias), coef(after_an_alias))
  expect_equal(logLik(before_an_alias), logLik(after_an_alias))
})
