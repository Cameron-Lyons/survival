test_that("multi-state Cox residuals and predictions are R's transition matrices", {
  mg <- survival::mgus2
  mg$etime <- with(mg, ifelse(pstat == 0, futime, ptime))
  mg$event <- factor(with(mg, ifelse(pstat == 0, 2 * death, 1)), 0:2,
    labels = c("censor", "pcm", "death"))
  fit <- coxph(Surv(etime, event) ~ sex + age, data = mg, id = id)
  stock <- survival::coxph(survival::Surv(etime, event) ~ sex + age, data = mg, id = id)
  for (type in c("martingale", "score", "dfbeta", "dfbetas")) {
    expect_equal(residuals(fit, type = type), residuals(stock, type = type), tolerance = 1e-6)
  }
  expect_equal(residuals(fit, collapse = TRUE), residuals(stock, collapse = TRUE), tolerance = 1e-6)
  expect_equal(predict(fit), predict(stock), tolerance = 1e-6)
  expect_equal(predict(fit, type = "risk"), predict(stock, type = "risk"), tolerance = 1e-6)
  expect_equal(predict(fit, se.fit = TRUE), predict(stock, se.fit = TRUE), tolerance = 1e-6)
  newdata <- data.frame(sex = "F", age = 60)
  expect_equal(predict(fit, newdata = newdata), predict(stock, newdata = newdata), tolerance = 1e-6)

  counting <- coxph(Surv(start, stop, event) ~ age + sex, data = survival::mgus1, id = id)
  reference <- survival::coxph(survival::Surv(start, stop, event) ~ age + sex,
    data = survival::mgus1, id = id)
  expect_equal(residuals(counting), residuals(reference), tolerance = 1e-6)
  expect_equal(residuals(counting, type = "score"), residuals(reference, type = "score"),
    tolerance = 1e-6)
  expect_equal(predict(counting), predict(reference), tolerance = 1e-6)
})
