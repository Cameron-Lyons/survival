test_that("brier(detail = TRUE) names phat by the rows scored", {
  fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, survival::lung)
  stock <- survival::coxph(survival::Surv(time, status) ~ age + sex + ph.ecog, survival::lung)
  result <- brier(fit, times = c(100, 365), detail = TRUE)
  reference <- survival::brier(stock, times = c(100, 365), detail = TRUE)
  expect_identical(colnames(result$phat), colnames(reference$phat))
  expect_equal(result$phat, reference$phat, tolerance = 1e-6)
})

test_that("concordancefit returns R's cvar vector and cluster-named dfbeta", {
  lung <- survival::lung
  y <- with(lung, survival::Surv(time, status))
  x <- cbind(age = lung$age, sex = lung$sex)
  expect_equal(concordancefit(y, x)$cvar, survival::concordancefit(y, x)$cvar)
  expect_equal(
    concordancefit(y, lung$age, cluster = lung$sex, influence = 1)$dfbeta,
    survival::concordancefit(y, lung$age, cluster = lung$sex, influence = 1)$dfbeta
  )
})
