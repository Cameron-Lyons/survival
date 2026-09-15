test_that("Cox controls preserve survival check settings", {
  skip_if_not_installed("survival")

  control <- coxph.control()
  expect_named(control, c(
    "eps", "toler.chol", "iter.max", "toler.inf", "outer.max", "timefix",
    "survcheckallow"
  ))
  expect_identical(control$survcheckallow, "gap")

  reference_has_survcheckallow <- "survcheckallow" %in%
    names(formals(survival::coxph.control))
  if (reference_has_survcheckallow) {
    expect_identical(control, survival::coxph.control())
  }

  # R preserves this value without validation. Only multistate Cox fits
  # inspect it; ordinary Cox fits also accept NULL and undocumented values.
  settings <- list(
    "gap", "jump", "teleport", "overlap", c("gap", "overlap"),
    character(), NULL, "unknown", 42, TRUE, NA_character_
  )
  for (setting in settings) {
    actual <- coxph.control(survcheckallow = setting)
    expect_true("survcheckallow" %in% names(actual))
    expect_identical(actual$survcheckallow, setting)
    if (reference_has_survcheckallow) {
      reference <- survival::coxph.control(survcheckallow = setting)
      expect_identical(actual, reference)
    }
  }
})

test_that("native R Cox controls work for ordinary right and counting responses", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    id = rep(1:6, each = 2),
    start = c(0, 3, 0, 2, 0, 4, 0, 2, 0, 5, 0, 3),
    stop = c(2, 5, 3, 6, 4, 7, 2, 8, 4, 9, 3, 10),
    status = c(0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0),
    x = c(0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.9, 0.6, 1.2, 0.5, 0.2)
  )
  checked <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = data,
    id = id
  )
  expect_gt(unname(checked$flag["overlap"]), 0)
  expect_gt(unname(checked$flag["gap"]), 0)

  reference_has_survcheckallow <- "survcheckallow" %in%
    names(formals(survival::coxph.control))
  settings <- list(
    "gap", "overlap", c("gap", "jump", "teleport", "overlap"),
    character(), NULL, "unknown"
  )
  expect_same_fit <- function(actual, expected) {
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-8)
    expect_equal(unname(vcov(actual)), unname(vcov(expected)), tolerance = 1e-8)
    expect_equal(as.numeric(logLik(actual)), as.numeric(logLik(expected)), tolerance = 1e-8)
  }

  for (response_type in c("right", "counting")) {
    if (response_type == "right") {
      bridge_formula <- Surv(stop, status) ~ x
      reference_formula <- survival::Surv(stop, status) ~ x
    } else {
      bridge_formula <- Surv(start, stop, status) ~ x
      reference_formula <- survival::Surv(start, stop, status) ~ x
    }
    baseline <- survival::coxph(
      reference_formula,
      data = data,
      id = data$id,
      robust = FALSE,
      control = survival::coxph.control(iter.max = 40)
    )
    for (setting in settings) {
      # Older survival versions have no such control argument. Their
      # ordinary fits remain a valid numerical reference for these settings.
      native_control <- if (reference_has_survcheckallow) {
        survival::coxph.control(iter.max = 40, survcheckallow = setting)
      } else {
        survival::coxph.control(iter.max = 40)
      }
      native_fit <- survival::coxph(
        reference_formula,
        data = data,
        id = data$id,
        robust = FALSE,
        control = native_control
      )
      expect_same_fit(native_fit, baseline)

      with_native_control <- coxph(
        bridge_formula,
        data = data,
        id = data$id,
        robust = FALSE,
        control = native_control
      )
      with_bridge_control <- coxph(
        bridge_formula,
        data = data,
        id = data$id,
        robust = FALSE,
        control = coxph.control(iter.max = 40, survcheckallow = setting)
      )
      expect_same_fit(with_native_control, native_fit)
      expect_same_fit(with_bridge_control, native_fit)
    }
  }
})
