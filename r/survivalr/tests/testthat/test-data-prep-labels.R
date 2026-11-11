test_that("nostutter factor transport retains event states through Python", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  cases <- list(
    list(x = c(0, 1, 1, 2), censor = 0),
    list(x = c("censor", "b", "b", "a"), censor = "censor"),
    list(x = factor(c("b", "b", "a", "a"), levels = c("censor", "b", "a")),
         censor = "censor"),
    list(x = ordered(c("b", "b", "a", "a"), levels = c("censor", "b", "a")),
         censor = "censor")
  )
  for (case in cases) {
    args <- c(list(id = rep(1, 4)), case)
    actual <- do.call(nostutter, args)
    expected <- do.call(survival::nostutter, args)
    expect_identical(actual, expected)
    expect_identical(
      unclass(survival::Surv(1:4, actual)),
      unclass(survival::Surv(1:4, expected))
    )
  }
})

test_that("neardate matches character labels across identifier types", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  cases <- list(
    list(c(1, 2, 1), c("1", "2", "1")),
    list(c(TRUE, FALSE, TRUE), c("TRUE", "FALSE", "TRUE")),
    list(c(NA_real_, NaN, 1), c(NA_real_, NaN, 1)),
    list(c(TRUE, NaN, 2), c("1", "NaN", "2")),
    list(c(TRUE, 2L, 3L), c("1", "2", "3")),
    list(factor(c(1, 2, 1), levels = c(2, 1, 3)), c(1, 2, 1)))
  for (case in cases) for (best in c("after", "prior")) {
    args <- list(id1 = case[[1]], id2 = case[[2]], y1 = c(1, 2, 5),
                 y2 = c(2, 3, 4), best = best, nomatch = -1L)
    expect_identical(do.call(neardate, args), do.call(survival::neardate, args))
  }
})
