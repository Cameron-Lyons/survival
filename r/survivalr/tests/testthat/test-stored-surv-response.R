# A response that is already a Surv object (R's y ~ x idiom), whether survival's
# matrix or survivalr's own Surv, fits as the Surv(...) call does in stock R.
test_that("a Surv column of the data is a model response", {
  lung <- survival::lung
  native <- lung
  native$SurvObj <- survival::Surv(lung$time, lung$status == 2)
  bridged <- lung
  bridged$SurvObj <- with(lung, Surv(time, status == 2))
  # survival::survfit(<formula>) here would dispatch to survivalr's own method.
  stock <- function(fit) {
    eval(fit, list(Surv = survival::Surv, lung = lung,
      survfit_formula = get("survfit.formula", asNamespace("survival"))))
  }
  cox <- stock(quote(survival::coxph(Surv(time, status == 2) ~ sex, data = lung)))
  km <- stock(quote(survfit_formula(Surv(time, status == 2) ~ sex, data = lung)))
  aft <- stock(quote(survival::survreg(Surv(time, status == 2) ~ age, data = lung)))
  test <- stock(quote(survival::survdiff(Surv(time, status == 2) ~ sex, data = lung)))
  cindex <- stock(quote(survival::concordance(Surv(time, status == 2) ~ age, data = lung)))
  for (data in list(native, bridged)) {
    expect_equal(unname(coef(coxph(SurvObj ~ sex, data = data))), unname(coef(cox)))
    expect_equal(
      summary(survfit(SurvObj ~ sex, data = data), times = c(100, 300))$surv,
      summary(km, times = c(100, 300))$surv
    )
    expect_equal(unname(coef(survreg(SurvObj ~ age, data = data))), unname(coef(aft)))
    expect_equal(survdiff(SurvObj ~ sex, data = data)$chisq, test$chisq)
    expect_equal(concordance(SurvObj ~ age, data = data)$concordance, cindex$concordance)
  }
  frame <- data.frame(y = with(lung, Surv(time, status)), age = lung$age)
  expect_equal(
    unname(coef(coxph(y ~ age, data = frame))),
    unname(coef(stock(quote(survival::coxph(Surv(time, status) ~ age, data = lung)))))
  )
  expect_error(coxph(age ~ sex, data = lung), "response must be a survival object")
})

test_that("survcondense stores a condensed Surv response under its name", {
  data <- data.frame(id = c(1, 1, 2), x = c(1, 1, 2))
  data$y <- survival::Surv(c(0, 2, 0), c(2, 5, 3), c(0, 1, 0))
  result <- survcondense(y ~ x, data = data, id = id)
  reference <- survival::survcondense(y ~ x, data = data, id = id)
  expect_named(result, names(reference))
  expect_equal(unclass(result$y)[, 1:3], unclass(reference$y)[, 1:3], ignore_attr = TRUE)
})
