test_that("finegray carries a missing case weight into its subject's rows only", {
  d <- data.frame(t = 1:8, s = factor(c(1, 0, 2, 1, 0, 2, 1, 0), 0:2, labels = c("c", "x", "y")),
    a = 1:8, w = c(1, 2, NA, 1, 1, 2, 1, 1))
  expect_equal(
    as.data.frame(finegray(Surv(t, s) ~ a, data = d, weights = w)),
    as.data.frame(survival::finegray(survival::Surv(t, s) ~ a, data = d, weights = w)),
    ignore_attr = TRUE
  )
  mg <- survival::mgus2
  mg$etime <- with(mg, ifelse(pstat == 0, futime, ptime))
  mg$event <- factor(with(mg, ifelse(pstat == 0, 2 * death, 1)), 0:2,
    labels = c("censor", "pcm", "death"))
  result <- finegray(Surv(etime, event) ~ age, data = mg, weights = hgb)
  reference <- survival::finegray(survival::Surv(etime, event) ~ age, data = mg, weights = hgb)
  expect_equal(nrow(result), nrow(reference))
  expect_equal(result$fgwt, reference$fgwt)
})

test_that("survexp with a Cox rate table reads a numeric column for a factor", {
  heart <- survival::heart
  fit <- survival::coxph(survival::Surv(start, stop, event) ~ age + transplant, data = heart)
  newdata <- data.frame(age = c(-10, 0, 10), transplant = c(0, 1, 1))
  expect_warning(
    result <- survexp(~ 1, data = newdata, ratetable = fit, times = c(10, 100, 500)),
    "variable 'transplant' is not a factor"
  )
  expect_equal(result$surv, suppressWarnings(
    survival::survexp(~ 1, data = newdata, ratetable = fit, times = c(10, 100, 500))$surv
  ))
})

test_that("royston of a null Cox model has R's NA values", {
  result <- royston(coxph(Surv(time, status) ~ 1, survival::lung))
  reference <- survival::royston(survival::coxph(survival::Surv(time, status) ~ 1, survival::lung))
  expect_equal(result, reference)
  expect_identical(is.nan(result), is.nan(reference))
})

test_that("survConcordance keeps R's na.action component", {
  result <- suppressWarnings(survConcordance(Surv(time, status) ~ ph.karno, survival::lung))
  reference <- suppressWarnings(
    survival::survConcordance(survival::Surv(time, status) ~ ph.karno, survival::lung)
  )
  expect_named(result, names(reference))
  expect_identical(result$na.action, reference$na.action)
})
