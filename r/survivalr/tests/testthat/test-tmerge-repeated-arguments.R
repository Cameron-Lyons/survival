test_that("tmerge applies repeated arguments in order and survcheck tracks states", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  d <- data.frame(id = 1:3, fu = c(10, 10, 10), e1 = c(2, 3, NA), e2 = c(5, NA, NA), e3 = c(7, 8, 9))
  actual <- tmerge(tmerge(d[, 1:2], d, id = id, tstop = fu), d, id = id,
    ev = event(e1), ev = event(e2), ev = event(e3))
  expected <- survival::tmerge(survival::tmerge(d[, 1:2], d, id = id, tstop = fu), d, id = id,
    ev = event(e1), ev = event(e2), ev = event(e3))
  expect_equal(as.list(actual), as.list(expected), ignore_attr = TRUE)
  expect_equal(attr(actual, "tcount"), attr(expected, "tcount"), ignore_attr = TRUE)
  expect_identical(rownames(attr(actual, "tcount")), rownames(attr(expected, "tcount")))
  # The cgd vignette's pattern: seven infection times into one event column.
  cgd0 <- survival::cgd0
  vignette <- function(tm) {
    n <- tm(cgd0[, 1:13], cgd0, id = id, tstop = futime)
    tm(n, cgd0, id = id, infect = event(etime1), infect = event(etime2), infect = event(etime3),
       infect = event(etime4), infect = event(etime5), infect = event(etime6), infect = event(etime7))
  }
  actual <- vignette(tmerge)
  expected <- vignette(survival::tmerge)
  expect_identical(nrow(actual), nrow(expected))
  expect_equal(actual$infect, expected$infect)
  expect_equal(attr(actual, "tcount"), attr(expected, "tcount"), ignore_attr = TRUE)
  # A character tdc is missing before its first value.
  d1 <- data.frame(id = 1:2, fu = c(10, 10))
  d2 <- data.frame(id = c(1, 2), t = c(4, 6), v = c("lo", "hi"))
  actual <- tmerge(tmerge(d1, d1, id = id, tstop = fu), d2, id = id, z = tdc(t, v))
  expected <- survival::tmerge(survival::tmerge(d1, d1, id = id, tstop = fu), d2, id = id, z = tdc(t, v))
  expect_identical(actual$z, expected$z)
  # survcheck tabulates (start, stop] transitions from the states it works out.
  expect_equal(survcheck(Surv(tstart, tstop, status) ~ 1, survival::cgd, id = id)$transitions,
    survival::survcheck(Surv(tstart, tstop, status) ~ 1, survival::cgd, id = id)$transitions,
    ignore_attr = TRUE)
})
