test_that("survSplit keeps the data's column order and passes timefix and added", {
  d <- data.frame(
    time = c(5, 10, 3, 12),
    status = c(1, 0, 1, 1),
    g = c("a", "b", "a", "b"),
    id = 1:4
  )
  bridged <- survSplit(Surv(time, status) ~ ., d, cut = c(4, 9), episode = "ep", id = "id")
  reference <- survival::survSplit(Surv(time, status) ~ ., d, cut = c(4, 9), episode = "ep", id = "id")
  expect_equal(bridged, reference)
  expect_type(bridged$status, "double")
  expect_equal(
    survSplit(Surv(time, status) ~ ., d, cut = c(4, 9), timefix = FALSE),
    survival::survSplit(Surv(time, status) ~ ., d, cut = c(4, 9), timefix = FALSE)
  )
  expect_equal(
    survSplit(Surv(time, status) ~ g, d, cut = 4, added = "new"),
    survival::survSplit(Surv(time, status) ~ g, d, cut = 4, added = "new")
  )
})
