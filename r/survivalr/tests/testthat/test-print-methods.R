# Bridge print methods render Python's R-format reports with R's call header, so the
# output equals stock R's (apart from the trailing spaces Python's reports drop).
printed <- function(expr, env) sub(" +$", "", capture.output(eval(expr, env)))

stock_print_env <- function() {
  list2env(list(
    Surv = survival::Surv, coxph = survival::coxph, survreg = survival::survreg,
    survfit = get("survfit.formula", asNamespace("survival")), survdiff = survival::survdiff,
    concordance = survival::concordance, clogit = survival::clogit, cox.zph = survival::cox.zph,
    frailty = survival::frailty, strata = survival::strata, pspline = survival::pspline,
    lung = survival::lung, infert = datasets::infert
  ), parent = globalenv())
}

test_that("model, curve and test prints equal stock R's", {
  port <- list2env(list(lung = survival::lung, infert = datasets::infert),
    parent = environment())
  stock <- stock_print_env()
  cases <- list(
    quote(print(coxph(Surv(time, status) ~ age + ph.ecog, data = lung))),
    quote(print(coxph(Surv(time, status) ~ 1, data = lung))),
    quote(print(coxph(Surv(time, status) ~ age + frailty(inst), data = lung))),
    quote(print(coxph(Surv(time, status) ~ pspline(age), data = lung))),
    quote(print(coxph(Surv(time, status) ~ age + strata(sex), data = lung))),
    quote(print(coxph(Surv(time, status) ~ age + sex, data = lung, cluster = inst))),
    quote(print(clogit(case ~ spontaneous + induced + strata(stratum), infert))),
    quote(print(survreg(Surv(time, status) ~ age + sex + strata(ph.ecog), data = lung,
      robust = TRUE))),
    quote(print(survreg(Surv(time, status) ~ age, data = lung, dist = "lognormal"))),
    quote(print(survfit(Surv(time, status) ~ sex, data = lung))),
    quote(print(survfit(Surv(time, status) ~ 1, data = lung))),
    quote(print(survfit(Surv(time, status) ~ sex, data = lung), rmean = "common")),
    quote(print(survdiff(Surv(time, status) ~ sex, lung))),
    quote(print(survdiff(Surv(time, status) ~ ph.ecog, lung, rho = 1))),
    quote(print(cox.zph(coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)))),
    quote(print(concordance(Surv(time, status) ~ ph.karno, lung))),
    quote(print(concordance(Surv(time, status) ~ ph.karno + strata(sex), lung))),
    quote(print(anova(coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)))),
    quote(print(anova(coxph(Surv(time, status) ~ age, data = lung),
      coxph(Surv(time, status) ~ age + sex, data = lung)))),
    quote(print(anova(survreg(Surv(time, status) ~ age, data = lung),
      survreg(Surv(time, status) ~ age + sex, data = lung))))
  )
  for (case in cases) {
    expect_identical(printed(case, port), printed(case, stock), label = deparse(case)[1])
  }
})

test_that("anova results are R's anova data frames", {
  lung <- survival::lung
  fit <- coxph(Surv(time, status) ~ age + sex, data = lung)
  stock <- survival::coxph(survival::Surv(time, status) ~ age + sex, data = lung)
  expect_equal(anova(fit), anova(stock), ignore_attr = "heading", tolerance = 1e-6)
  expect_s3_class(anova(fit), "anova")
  # anova.survreg's heading names the distribution from object$family, which survreg
  # fits lack; the port leaves that empty line out.
  aft <- anova(survreg(Surv(time, status) ~ age + sex, data = lung))
  expect_false(any(grepl("distribution with", attr(aft, "heading"))))
})
