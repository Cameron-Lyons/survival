test_that("multistate Cox summaries forward explicit count selection", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- data.frame(
    time = c(1, 1, 2, 3, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8, 9),
    event = factor(c("a", "censor", "b", "a", "censor", "b", "censor", "censor",
                     "b", "censor", "a", "censor", "b", "a", "censor", "b"),
                   levels = c("censor", "a", "b")),
    x = c(2, 0, 1, 3, 1, 0, 2, 3, 0, 2, 1, 0, 3, 1, 2, 3),
    id = 1:16
  )
  actual_fit <- coxph(Surv(time, event) ~ x, data, id = id)
  stock_fit <- survival::coxph(Surv(time, event) ~ x, data, id = id)
  newdata <- data.frame(x = c(0, 2))
  actual <- suppressWarnings(survfit(actual_fit, newdata = newdata))
  stock <- survival::survfit(stock_fit, newdata = newdata)
  for (times in list(c(0, 2, 4, 7, 10), c(7, 2, 2, 0, 10))) {
    for (dosum in c(FALSE, TRUE)) {
      result <- summary(actual, times = times, extend = TRUE, dosum = dosum)
      expected <- summary(stock, times = times, extend = TRUE, dosum = dosum)
      for (field in c("time", "n.risk", "n.event", "n.censor", "n.transition")) {
        expect_equal(result[[field]], expected[[field]], ignore_attr = TRUE)
      }
      for (field in c("pstate", "cumhaz")) {
        expect_equal(result[[field]], expected[[field]], tolerance = 1e-8, ignore_attr = TRUE)
      }
    }
  }
  expect_error(summary(actual, times = c(1, 2), dosum = "yes"), "dosum must be TRUE/FALSE")
})
