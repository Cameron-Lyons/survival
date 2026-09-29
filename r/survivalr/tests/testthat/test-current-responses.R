test_that("Surv2 preserves current binary and factor status semantics", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  cases <- list(
    c(0, 1, NA), c(1, 2, NA), c(FALSE, TRUE, NA), rep(NA, 3),
    factor(c("lost", "ill", NA), levels = c("lost", "ill", "death")),
    factor(rep("lost", 3), levels = "lost"),
    ordered(c("lost", "ill", NA), levels = c("lost", "ill")),
    setNames(c(0, 1, NA), c("a", "b", "c"))
  )
  for (event in cases) for (repeated in list(FALSE, TRUE, "first")) {
    time <- setNames(c(1, 2, 3), c("x", "y", "z"))
    actual <- Surv2(time, event, repeated)
    expected <- survival::Surv2(time, event, repeated)
    expect_equal(actual, expected)
    expect_equal(format(actual), getFromNamespace("format.Surv2", "survival")(expected))
    expect_equal(actual[1:2], getFromNamespace("[.Surv2", "survival")(expected, 1:2))
    expect_equal(as.matrix(actual), getFromNamespace("as.matrix.Surv2", "survival")(expected))
  }
  expect_error(Surv2(1:3, c("a", "b", "c")), "invalid status")
  expect_warning(actual <- Surv2(1:3, c(0, 3, 1)), "Invalid status value, converted to NA")
  expected <- suppressWarnings(survival::Surv2(1:3, c(0, 3, 1)))
  expect_equal(actual, expected)
  expect_identical(is.na(Surv2(1:3, c(0, NA, 1))), c(FALSE, TRUE, FALSE))
})

test_that("multistate responses preserve censor labels through conversion", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  event <- factor(c("lost", "censored", NA), levels = c("lost", "censored", "censor"))
  for (counting in c(FALSE, TRUE)) {
    args <- if (counting) list(0:2, 1:3, event) else list(1:3, event)
    actual <- do.call(Surv, args)
    expected <- do.call(survival::Surv, args)
    expect_equal(actual, expected)
    expect_equal(actual[1:2], getFromNamespace("[.Surv", "survival")(expected, 1:2))
    expect_equal(format(actual), survival::format.Surv(expected))
    for (legacy in c(FALSE, TRUE)) {
      source <- expected
      if (legacy) attr(source, "clabel") <- NULL
      converted <- .as_native_surv(.as_python_surv(source))
      # Python responses retain states/censor labels, not arbitrary R input attributes.
      attr(source, "inputAttributes") <- NULL
      expect_equal(converted, source)
      expect_equal(format(converted), survival::format.Surv(source))
    }
  }
  one <- survival::Surv(1, factor("lost", levels = c("lost", "event")))
  expect_identical(.result_field(.as_python_surv(one), "clabel"), "lost")
})

test_that("grouped recurrent-event KM matches separate R curves", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  d <- data.frame(start = c(0,2,0,3,0,4,0,5), stop = c(2,5,3,6,4,7,5,8),
                  status = c(0,1,1,0,0,1,1,0), id = rep(1:4, each = 2),
                  group = rep(c("a", "b"), each = 4))
  for (entry in c(FALSE, TRUE)) {
    fit <- survfit(Surv(start, stop, status) ~ group, data = d, id = id, entry = entry)
    reference <- reference_survfit_complete_event_grid(
      survival::Surv(start, stop, status) ~ group, data = d, id = id, entry = entry)
    expect_equal(sum(reference$n.event), 4)
    for (group in c("a", "b")) {
      single <- getFromNamespace("survfit.formula", "survival")(
        survival::Surv(start, stop, status) ~ 1,
        data = d[d$group == group, ], id = id, entry = entry)
      for (field in c("time", "n.event", "n.risk", "surv", "cumhaz")) {
        actual <- .as_numeric_vector(.result_field(fit[[group]], gsub(".", "_", field, fixed = TRUE)))
        expect_equal(actual, single[[field]], tolerance = 1e-12)
        expect_equal(reference[paste0("group=", group)][[field]], single[[field]], tolerance = 1e-12)
      }
    }
  }
})
