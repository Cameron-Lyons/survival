# A stock fit whose formula sees survival's own Surv, pspline and ridge.
stock_cox <- function(form, frame) {
  environment(form) <- list2env(list(Surv = survival::Surv, pspline = survival::pspline,
    ridge = survival::ridge, frame = frame), parent = environment(form))
  eval(bquote(survival::coxph(.(form), data = frame)), environment(form))
}

yates_data <- function() {
  d <- survival::lung
  d$ecog <- factor(d$ph.ecog)
  d$sex <- factor(d$sex)
  d$oe <- factor(d$ph.ecog, ordered = TRUE)
  d
}

test_that("yates runs on Cox fits with penalized terms as R does", {
  d <- yates_data()
  for (form in list(Surv(time, status) ~ ecog + pspline(age, df = 3),
                    Surv(time, status) ~ ecog + ridge(age, theta = 1))) {
    expect_equal(
      yates(coxph(form, d), ~ ecog)$estimate,
      survival::yates(stock_cox(form, d), ~ ecog)$estimate,
      tolerance = 1e-6
    )
  }
})

test_that("yates over an ordered factor averages the fit's own predictions", {
  d <- yates_data()
  fit <- coxph(Surv(time, status) ~ oe + sex, d)
  kept <- d[!is.na(d$ph.ecog), ]
  direct <- vapply(0:3, function(k) {
    mean(predict(fit, newdata = transform(kept, oe = factor(k, levels = 0:3, ordered = TRUE)),
      type = "lp"))
  }, numeric(1))
  expect_equal(yates(fit, ~ oe)$estimate$pmm, direct, tolerance = 1e-8)
  fit <- lm(age ~ oe + sex, d)
  direct <- vapply(0:3, function(k) {
    mean(predict(fit, newdata = transform(kept, oe = factor(k, levels = 0:3, ordered = TRUE))))
  }, numeric(1))
  expect_equal(yates(fit, ~ oe)$estimate$pmm, direct, tolerance = 1e-8)
})

test_that("yates predicts survival for survivalr Cox fits", {
  d <- yates_data()
  d$sex <- factor(d$sex, labels = c("M", "F"))
  set.seed(1)
  result <- yates(coxph(Surv(time, status) ~ sex + age, d), ~ sex, predict = "survival",
    nsim = 10, options = list(rmean = 400))
  set.seed(1)
  reference <- survival::yates(stock_cox(Surv(time, status) ~ sex + age, d), ~ sex,
    predict = "survival", nsim = 10, options = list(rmean = 400))
  expect_equal(result$estimate, reference$estimate, tolerance = 1e-8)
})

test_that("yates results print as R's", {
  d <- yates_data()
  printed <- function(x) sub(" +$", "", capture.output(print(x)))
  expect_identical(
    printed(yates(coxph(Surv(time, status) ~ ecog + sex + age, d), ~ ecog)),
    printed(survival::yates(stock_cox(Surv(time, status) ~ ecog + sex + age, d), ~ ecog))
  )
  expect_identical(
    printed(yates(lm(age ~ ecog + sex, d), ~ ecog, test = "pairwise")),
    printed(survival::yates(lm(age ~ ecog + sex, d), ~ ecog, test = "pairwise"))
  )
})
