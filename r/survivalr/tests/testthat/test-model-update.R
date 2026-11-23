# update(), getCall(), drop1(), add1() and step() refit bridge fits from their call,
# as R's generics do for stock fits.
test_that("Cox fits support R's model-selection generics", {
  data <- na.omit(survival::lung[, c("time", "status", "age", "sex", "ph.ecog")])
  fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = data)
  stock <- survival::coxph(survival::Surv(time, status) ~ age + sex + ph.ecog, data = data)
  expect_equal(deparse(getCall(fit)),
    "coxph(formula = Surv(time, status) ~ age + sex + ph.ecog, data = data)")
  expect_equal(unname(coef(update(fit, . ~ . - age))), unname(coef(update(stock, . ~ . - age))))
  expect_equal(unname(coef(update(fit, data = data[1:150, ]))),
    unname(coef(update(stock, data = data[1:150, ]))))
  expect_equal(as.data.frame(drop1(fit, test = "Chisq")),
    as.data.frame(drop1(stock, test = "Chisq")), ignore_attr = TRUE)
  smaller <- update(fit, . ~ . - ph.ecog)
  expect_equal(as.data.frame(add1(smaller, ~ . + ph.ecog, test = "Chisq")),
    as.data.frame(add1(update(stock, . ~ . - ph.ecog), ~ . + ph.ecog, test = "Chisq")),
    ignore_attr = TRUE)
  chosen <- step(fit, trace = 0)
  expect_equal(attr(terms(chosen), "term.labels"),
    attr(terms(step(stock, trace = 0)), "term.labels"))
  expect_equal(unname(coef(fit)), unname(coef(stock)))
})

test_that("survreg fits support R's model-selection generics", {
  lung <- survival::lung
  fit <- survreg(Surv(time, status) ~ age + sex, data = lung)
  stock <- survival::survreg(survival::Surv(time, status) ~ age + sex, data = lung)
  expect_equal(unname(coef(update(fit, . ~ . - sex))), unname(coef(update(stock, . ~ . - sex))))
  expect_equal(as.data.frame(drop1(fit, test = "Chisq")),
    as.data.frame(drop1(stock, test = "Chisq")), ignore_attr = TRUE)
  expect_equal(attr(terms(step(fit, trace = 0)), "term.labels"),
    attr(terms(step(stock, trace = 0)), "term.labels"))
})
