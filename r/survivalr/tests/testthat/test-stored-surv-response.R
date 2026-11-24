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

test_that("an unused Surv column of newdata is carried through curves and summaries", {
  lung <- survival::lung[1:80, c("time", "status", "age")]
  cox <- coxph(Surv(time, status) ~ age, data = lung)
  stock_cox <- eval(quote(survival::coxph(Surv(time, status) ~ age, data = lung)),
    list(Surv = survival::Surv, lung = lung))
  newdata <- lung[1:3, ]
  newdata$sy <- survival::Surv(newdata$time, newdata$status)
  curves <- survfit(cox, newdata = newdata)
  stock <- survival::survfit(stock_cox, newdata = newdata)
  expect_equal(curves[2]$surv, stock[2]$surv)
  ours <- summary(curves, times = 200, data.frame = TRUE)
  reference <- summary(stock, times = 200, data.frame = TRUE)
  expect_named(ours, names(reference))
  expect_equal(ours$surv, reference$surv)
  expect_equal(unclass(ours$sy_)[], unclass(reference$sy_)[])

  d <- survival::mgus2[1:300, ]
  d$etime <- with(d, ifelse(pstat == 0, futime, ptime))
  d$event <- factor(with(d, ifelse(pstat == 0, 2 * death, 1)), 0:2, c("censor", "pcm", "death"))
  multistate <- coxph(Surv(etime, event) ~ age, data = d, id = id)
  nd <- d[1:2, c("age", "etime", "event", "futime", "death")]
  with_surv <- nd
  with_surv$sy <- survival::Surv(nd$futime, nd$death)
  expect_equal(
    suppressWarnings(survfit(multistate, newdata = with_surv))$pstate,
    suppressWarnings(survfit(multistate, newdata = nd))$pstate
  )
  stock_multistate <- eval(quote(survival::coxph(Surv(etime, event) ~ age, data = d, id = id)),
    list(Surv = survival::Surv, d = d))
  ours <- summary(suppressWarnings(survfit(multistate, newdata = with_surv)), times = c(30, 60),
    data.frame = TRUE)
  reference <- summary(survival::survfit(stock_multistate, newdata = with_surv), times = c(30, 60),
    data.frame = TRUE)
  expect_identical(dim(ours), dim(reference))
  expect_equal(ours$pstate, reference$pstate)
  expect_equal(unclass(ours$sy)[], unclass(reference$sy)[])
  expect_s3_class(ours$sy, "Surv")
  for (column in list(with_surv$sy, survival::Surv2(nd$futime, nd$death))) {
    shown <- nd
    shown$sy <- column
    curves <- suppressWarnings(survfit(multistate, newdata = shown))
    expect_output(print(curves))
    expect_output(print(curves[1, ]))
    expect_s3_class(as.data.frame(curves)$sy, class(column)[[1L]])
  }
  # A Surv2 column is carried whole too.
  with_surv2 <- newdata
  with_surv2$s2 <- Surv2(newdata$time, newdata$status - 1)
  frame <- summary(survfit(cox, newdata = with_surv2), times = 200, data.frame = TRUE)
  expect_s3_class(frame$s2_, "Surv2")
  expect_equal(unclass(frame$s2_)[], unclass(survival::Surv2(newdata$time, newdata$status - 1))[])
})

test_that("a Surv column keeps its place and name among other newdata columns", {
  lung <- survival::lung[1:80, c("time", "status", "age")]
  cox <- coxph(Surv(time, status) ~ age, data = lung)
  stock_cox <- eval(quote(survival::coxph(Surv(time, status) ~ age, data = lung)),
    list(Surv = survival::Surv, lung = lung))
  newdata <- data.frame(.survival_response = c(7, 8), age = c(50, 60))
  newdata$`my y` <- survival::Surv(c(100, 200), c(1, 0))
  ours <- summary(survfit(cox, newdata = newdata), times = c(100, 200), data.frame = TRUE)
  stock <- summary(survival::survfit(stock_cox, newdata = newdata), times = c(100, 200),
    data.frame = TRUE)
  expect_named(ours, names(stock))
  expect_equal(ours$.survival_response, stock$.survival_response)
  expect_equal(unclass(ours$`my y`)[], unclass(stock$`my y`)[])
  d <- survival::mgus2[1:300, ]
  d$etime <- with(d, ifelse(pstat == 0, futime, ptime))
  d$event <- factor(with(d, ifelse(pstat == 0, 2 * death, 1)), 0:2, c("censor", "pcm", "death"))
  multistate <- coxph(Surv(etime, event) ~ age, data = d, id = id)
  nd <- data.frame(age = c(60, 70))
  nd$`my y` <- survival::Surv(c(1, 2), c(1, 0))
  curves <- suppressWarnings(survfit(multistate, newdata = nd))
  frame <- as.data.frame(curves)
  expect_s3_class(frame$my.y, "Surv")
  expect_s3_class(as.data.frame(curves, optional = TRUE)$`my y`, "Surv")
})

test_that("a one-row frame with a Surv column converts", {
  lung <- survival::lung[1:80, c("time", "status", "age")]
  lung$y <- survival::Surv(lung$time, lung$status)
  cox <- coxph(Surv(time, status) ~ age, data = lung)
  stock <- eval(quote(survival::coxph(Surv(time, status) ~ age, data = lung)),
    list(Surv = survival::Surv, lung = lung))
  expect_equal(unname(predict(cox, newdata = lung[1, ])), unname(predict(stock, newdata = lung[1, ])))
  expect_equal(survfit(cox, newdata = lung[1, ])$surv,
    survival::survfit(stock, newdata = lung[1, ])$surv)
  heart <- survival::heart[1:60, ]
  heart$y <- survival::Surv(heart$start, heart$stop, heart$event)
  fit <- coxph(Surv(start, stop, event) ~ age, data = heart)
  expect_equal(unname(predict(fit, newdata = heart[1, ])),
    unname(predict(fit, newdata = heart[1, c("start", "stop", "event", "age")])))
  stored <- coxph(y ~ age, data = lung)
  expect_equal(unname(predict(stored, newdata = lung[1, ])), unname(predict(stock, newdata = lung[1, ])))
})
