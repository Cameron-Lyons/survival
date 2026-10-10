test_that("gamma EM constructor dots flatten vector init like stock R", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  group <- c("a", "b", "a")
  for (initial in list(.2, c(.2, 2))) {
    actual <- frailty.gamma(group, sparse = TRUE, init = initial)
    stock <- survival::frailty.gamma(group, sparse = TRUE, init = initial)
    expect_identical(attr(actual, "cparm"), attr(stock, "cparm"))
    if (length(initial) > 1L) {
      actual_state <- attr(actual, "cfun")(attr(actual, "cparm"), 0, NULL)
      stock_state <- attr(stock, "cfun")(attr(stock, "cparm"), 0, NULL)
      expect_equal(actual_state, stock_state)
      expect_identical(actual_state$theta, 0)
    }
  }
  for (options in list(list(theta = .5), list(method = "aic"), list(df = 1))) {
    actual <- do.call(frailty.gamma, c(list(group, sparse = TRUE, init = c(.2, 2)), options))
    stock <- do.call(survival::frailty.gamma,
      c(list(group, sparse = TRUE, init = c(.2, 2)), options))
    expect_identical(attr(actual, "cparm"), attr(stock, "cparm"))
  }
  actual <- frailty.gaussian(group, sparse = TRUE, init = c(.2, 2))
  stock <- survival::frailty.gaussian(group, sparse = TRUE, init = c(.2, 2))
  expect_identical(attr(actual, "cparm"), attr(stock, "cparm"))
  expect_identical(attr(actual, "cfun")(attr(actual, "cparm"), 0, NULL)$theta, .2)
})
