# Surv() and every Surv a Python fit hands back are survival's own Surv matrix.

.surv_outcome <- function(expr) {
  warnings <- character()
  error <- NULL
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) {
      error <<- conditionMessage(e)
      NULL
    }),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = warnings, error = error)
}

test_that("Surv() returns survival's matrix, attributes, NaN and warnings", {
  weeks <- structure(as.difftime(c(1, 2, 3), units = "weeks"), names = c("a", "b", "c"))
  labelled <- structure(c(1, 2, 3), label = "Days", class = "labelled")
  cases <- list(
    right = quote(S(c(5, 3, 8), c(1, 0, 1))),
    logical = quote(S(c(5, 3, 8), c(TRUE, FALSE, NA))),
    one_two = quote(S(c(5, 3, 8), c(2, 1, 2))),
    time_only = quote(S(c(a = NaN, b = 2))),
    nan_time = quote(S(c(NaN, 3, Inf), c(1, 0, 1))),
    integer_names = quote(S(c(a = 1L, b = 2L), c(1L, 0L))),
    invalid_status = quote(S(c(5, 3, 8), c(1, 4, 0))),
    origin = quote(S(c(5, 3, 8), c(1, 0, 1), origin = 2)),
    left = quote(S(c(NaN, 2), c(1, 0), type = "left")),
    counting = quote(S(c(a = 0, b = 1), c(c = 2, d = 3), c(e = 1, f = 0))),
    counting_backwards = quote(S(c(3, 1), c(2, 3), c(1, 0))),
    counting_nan = quote(S(c(0, NaN, 1), c(2, 3, NaN), c(1, 0, 1))),
    counting_difftime = quote(S(c(0, 1), as.difftime(c(2, 3), units = "days"), c(1, 0))),
    interval_no_three = quote(S(c(1, 2), c(a = 2, b = 3), c(0, 1), type = "interval")),
    interval_three = quote(S(c(1, 2), c(a = 2, b = 3), c(3, 1), type = "interval")),
    interval_nan = quote(S(c(NaN, 2), c(NaN, 3), c(3, 1), type = "interval")),
    interval2 = quote(S(c(1, NA, 2, 4), c(2, 3, Inf, 4), type = "interval2")),
    interval2_labelled = quote(S(labelled, c(2, NA, 5), type = "interval2")),
    interval2_nan = quote(S(c(NaN, 2), c(3, NaN), type = "interval2")),
    interval2_named_type = quote(S(c(NaN, 2, 1), c(3, NaN, 4), type = c(cens = "interval2"))),
    difftime_time = quote(S(weeks, c(1, 0, 1))),
    labelled_time = quote(S(labelled, c(1, 0, 1))),
    named_event = quote(S(c(1, 2, 3), c(a = 1, b = 0, c = 1))),
    keyword_time_event = quote(S(time = c(a = 1, b = 2), event = c(1, 0))),
    keyword_time_time2 = quote(S(time = c(a = 1, b = 2), time2 = c(x = 1, y = 0))),
    mstate = quote(S(c(p = 1, q = 2, r = 3), factor(c("c", "a", "b")))),
    mstate_named_event = quote(S(c(1, 2, 3), factor(c(u = "c", v = "a", w = "c")))),
    mstate_counting = quote(S(c(p = 0, q = 0), c(1, 2), factor(c("c", "a")))),
    empty = quote(S(numeric(0), numeric(0))),
    missing_status = quote(S(c(1, 2), c(NA_real_, NA_real_))),
    missing_counting_status = quote(S(c(0, 1), c(2, 3), c(NA_real_, NA_real_))),
    wrong_type = quote(S(c(1, 2), c(1, 0), type = "counting")),
    not_numeric = quote(S(c("a", "b"), c(1, 0)))
  )
  for (name in names(cases)) {
    env <- list(weeks = weeks, labelled = labelled)
    ours <- .surv_outcome(eval(cases[[name]], c(env, S = Surv)))
    stock <- .surv_outcome(eval(cases[[name]], c(env, S = survival::Surv)))
    expect_identical(ours$value, stock$value, info = name)
    expect_identical(ours$warnings, stock$warnings, info = name)
    expect_identical(is.null(ours$error), is.null(stock$error), info = name)
  }
})

test_that("a Surv() value is a plain matrix that base R and saveRDS handle", {
  y <- Surv(c(5, 3, 8, 2), c(1, 0, 1, 1))
  stock <- survival::Surv(c(5, 3, 8, 2), c(1, 0, 1, 1))
  expect_true(is.matrix(y))
  expect_identical(typeof(y), "double")
  expect_identical(dimnames(y), list(NULL, c("time", "status")))
  expect_identical(colSums(unclass(y)), colSums(unclass(stock)))
  expect_identical(unclass(rbind(y, y)), unclass(rbind(stock, stock)))
  y[1, 1] <- 99
  stock[1, 1] <- 99
  expect_identical(y, stock)
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(y, path)
  expect_identical(readRDS(path), stock)
  expect_true(is.Surv(y))
  expect_false(is.Surv(unclass(y)))
})

test_that("a Surv column behaves as survival's in a data frame", {
  d <- data.frame(x = c(1, 2, 3, 4))
  d$y <- Surv(c(5, 3, 8, 2), c(1, 0, 1, 1))
  stock <- data.frame(x = c(1, 2, 3, 4))
  stock$y <- survival::Surv(c(5, 3, 8, 2), c(1, 0, 1, 1))
  expect_identical(d, stock)
  expect_identical(d[2:3, ], stock[2:3, ])
  expect_identical(stats::model.frame(y ~ x, d), stats::model.frame(y ~ x, stock))
})

test_that("Python fits hand back their responses as survival's Surv", {
  lung <- survival::lung
  cox <- coxph(Surv(time, status) ~ age, data = lung)
  response <- survival::Surv(lung$time, lung$status)
  expect_identical(cox$y, response)
  expect_identical(cox[["y"]], response)
  aft <- survreg(Surv(time, status) ~ age, data = lung, y = TRUE)
  expect_identical(aft$y, response)
  frame <- coxph(Surv(time, status) ~ age, data = lung, model = TRUE)$model
  expect_s3_class(frame[[1L]], "Surv")
  expect_identical(unclass(frame[[1L]])[], unclass(response)[])
})

test_that("aeqSurv returns survival's result with the input's attributes", {
  right <- Surv(c(1, 1 + 1e-10, 2), c(1, 0, 1))
  counting <- Surv(c(0, 0), c(1, 1 + 1e-10), c(1, 0))
  interval <- Surv(c(1, 2), c(2, 3 + 1e-12), c(3, 3), type = "interval")
  named <- Surv(c(a = 1, b = 1 + 1e-10), c(1, 1))
  untied_nan <- Surv(c(NaN, 1, 2), c(1, 0, 1))
  tied_nan <- Surv(c(NaN, 1, 1 + 1e-10, 2), c(1, 0, 1, 1))
  for (y in list(right, counting, interval, named, untied_nan, tied_nan)) {
    expect_identical(aeqSurv(y), survival::aeqSurv(y))
  }
})

test_that("Surv() responses reach the direct-response calls as survival's do", {
  lung <- survival::lung
  y <- Surv(lung$time, lung$status)
  formula_km <- survfit(Surv(time, status) ~ sex, data = lung)
  expect_equal(summary(survfit(y), times = c(100, 300))$surv,
    summary(survfit(Surv(time, status) ~ 1, data = lung), times = c(100, 300))$surv)
  expect_equal(summary(survfit(y, group = lung$sex), times = 200)$surv,
    summary(formula_km, times = 200)$surv)
  x <- data.frame(age = lung$age)
  expect_equal(unname(coef(coxph(y, x = x))),
    unname(coef(coxph(Surv(time, status) ~ age, data = lung))))
  expect_equal(unname(coef(survreg(y, x = cbind(1, age = lung$age)))),
    unname(coef(survreg(Surv(time, status) ~ age, data = lung))))
  low <- coxph.fit(as.matrix(lung["age"]), y, strata = NULL, offset = NULL, init = NULL,
    control = coxph.control(), weights = NULL, method = "efron", rownames = NULL)
  expect_equal(unname(low$coefficients),
    unname(coef(coxph(Surv(time, status) ~ age, data = lung))))
})

test_that("an unused Surv column does not stop a fit", {
  lung <- survival::lung[, c("time", "status", "age", "sex")]
  lung$y <- Surv(lung$time, lung$status)
  fit <- coxph(Surv(time, status) ~ age, data = lung)
  expect_equal(
    unname(coef(fit)),
    unname(coef(coxph(Surv(time, status) ~ age, data = survival::lung)))
  )
  expect_equal(
    summary(survfit(Surv(time, status) ~ sex, data = lung), times = 200)$surv,
    summary(survfit(Surv(time, status) ~ sex, data = survival::lung), times = 200)$surv
  )
  expect_length(predict(fit, newdata = lung[1:5, ], type = "expected"), 5L)
  plain <- lung[1:3, c("time", "status", "age")]
  curves <- survfit(fit, newdata = lung[1:3, ])
  expect_equal(curves[2]$surv, survfit(fit, newdata = plain)[2]$surv)
  d <- survival::mgus2[1:300, ]
  d$etime <- with(d, ifelse(pstat == 0, futime, ptime))
  d$event <- factor(with(d, ifelse(pstat == 0, 2 * death, 1)), 0:2, c("censor", "pcm", "death"))
  multistate <- coxph(Surv(etime, event) ~ age, data = d, id = id)
  nd <- d[1:2, c("age", "etime", "event", "futime", "death")]
  with_surv <- nd
  with_surv$y <- Surv(nd$futime, nd$death)
  expect_equal(
    suppressWarnings(survfit(multistate, newdata = with_surv))$pstate,
    suppressWarnings(survfit(multistate, newdata = nd))$pstate
  )
})

test_that("a one-row Surv converts for survfit, prediction and pyears", {
  expect_equal(survfit(Surv(5, 1))$surv, 0)
  lung <- survival::lung[, c("time", "status", "age")]
  lung$y <- Surv(lung$time, lung$status)
  fit <- coxph(Surv(time, status) ~ age, data = lung)
  expect_equal(
    unname(predict(fit, newdata = lung[1, ])),
    unname(predict(fit, newdata = lung[1, c("time", "status", "age")]))
  )
  direct <- pyears(Surv(lung$time, lung$status), scale = 1)
  formula <- pyears(Surv(time, status) ~ 1, data = lung, scale = 1)
  expect_equal(c(direct$pyears, direct$event, direct$n), c(formula$pyears, formula$event, formula$n))
})

test_that("multi-state responses in survcheck's own Surv binding keep their states", {
  d <- data.frame(id = 1:6, etime = c(5, 8, 3, 9, 4, 7),
    ev = factor(c("censor", "pcm", "death", "pcm", "censor", "death"),
      levels = c("censor", "pcm", "death")))
  ours <- survcheck((Surv(etime, ev)) ~ 1, data = d, id = id)
  stock <- survival::survcheck((survival::Surv(etime, ev)) ~ 1, data = d, id = id)
  expect_equal(unclass(ours$transitions), unclass(stock$transitions), ignore_attr = TRUE)
  expect_identical(dimnames(ours$transitions), dimnames(stock$transitions))
})

test_that("survreg.fit takes a Surv response's fitter codes as given", {
  lung <- survival::lung
  x <- cbind(1, age = lung$age)
  # survreg() hands survreg.fit a left-censored Surv recoded to 1 = exact, 2 = left.
  y <- survival::Surv(lung$time / 100, lung$status - 1, type = "left")
  y[, 2] <- 2 - y[, 2]
  fit <- function(fitter, distributions) {
    fitter(x, y, NULL, NULL, NULL, survival::survreg.control(), dist = distributions$gaussian,
      scale = 0, nstrat = 1, strata = 0, parms = NULL)$coefficients
  }
  expect_equal(
    unname(fit(survreg.fit, survreg.distributions)),
    unname(fit(survival::survreg.fit, survival::survreg.distributions)),
    tolerance = 1e-6
  )
})

test_that("wrappers that build R model frames work without survivalr attached", {
  skip_if_not_installed("callr")
  result <- callr::r(function() {
    loadNamespace("survivalr")
    lung <- survival::lung
    d <- survival::mgus2
    d$etime <- with(d, ifelse(pstat == 0, futime, ptime))
    d$event <- factor(with(d, ifelse(pstat == 0, 2 * death, 1)), 0:2, c("censor", "pcm", "death"))
    nwtco <- survival::nwtco
    cases <- nwtco[nwtco$rel == 1 | nwtco$in.subcohort == 1, ]
    unattached <- tryCatch(stats::model.frame(Surv(time, status) ~ age, lung),
      error = conditionMessage)
    list(
      aareg = survivalr::aareg(Surv(time, status) ~ age, data = lung)$test.statistic,
      cch = coef(survivalr::cch(Surv(edrel, rel) ~ age, data = cases, subcoh = ~in.subcohort,
        id = ~seqno, cohort.size = 4028)),
      split = nrow(survivalr::survSplit(Surv(time, status) ~ ., data = lung, cut = c(100, 300))),
      transitions = unname(survivalr::survcheck(Surv(etime, event) ~ 1, data = d, id = id)$transitions),
      pyears = sum(survivalr::pyears(Surv(time, status) ~ sex, data = lung, scale = 1)$pyears),
      unattached = unattached,
      registered = !is.null(getS3method("model.frame", "formula", optional = TRUE))
    )
  }, libpath = .libPaths())
  lung <- survival::lung
  nwtco <- survival::nwtco
  cases <- nwtco[nwtco$rel == 1 | nwtco$in.subcohort == 1, ]
  expect_equal(result$aareg, aareg(Surv(time, status) ~ age, data = lung)$test.statistic)
  expect_equal(unname(result$cch), unname(coef(survival::cch(survival::Surv(edrel, rel) ~ age,
    data = cases, subcoh = ~in.subcohort, id = ~seqno, cohort.size = 4028))), tolerance = 1e-6)
  expect_identical(result$split, nrow(survSplit(Surv(time, status) ~ ., data = lung, cut = c(100, 300))))
  expect_equal(sum(result$transitions), 1384)
  expect_equal(result$pyears, sum(pyears(Surv(time, status) ~ sex, data = lung, scale = 1)$pyears))
  # As without survivalr: an unqualified Surv() is not found when nothing attaches it.
  expect_match(result$unattached, "could not find function \"Surv\"")
  expect_false(result$registered)
})

test_that("a Surv proxy saved by an earlier build converts", {
  native <- Surv(c(5, 3, NA, 8, 2), c(1, 0, 1, 1, 0))
  # survivalr 2.1.0's Surv() returned this proxy class.
  legacy <- .wrap_python(.as_python_surv(native), c("survival_py_surv", "survival_py_object"))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path))
  saveRDS(legacy, path)
  # Each check reads the file afresh, before anything has restored the Python object.
  fresh <- function() readRDS(path)
  expect_identical(Surv(fresh()), native)
  expect_identical(fresh()[2:3], native[2:3])
  expect_identical(fresh()[, 1], native[, 1])
  expect_identical(head(fresh(), 2), head(native, 2))
  expect_identical(tail(fresh(), 2), tail(native, 2))
  expect_identical(length(fresh()), length(native))
  expect_identical(rev(fresh()), rev(native))
  expect_identical(c(fresh(), fresh()), c(native, native))
  expect_identical(format(fresh()), format(native))
  expect_identical(as.character(fresh()), as.character(native))
  expect_identical(is.na(fresh()), is.na(native))
  expect_identical(as.matrix(fresh()), as.matrix(native))
  expect_output(print(fresh()), "3+", fixed = TRUE)
  expect_equal(survfit(fresh())$surv, survfit(native)$surv)
  expect_equal(survfit(formula = fresh())$surv, survfit(native)$surv)
  expect_identical(c(a = fresh(), b = fresh()), c(a = native, b = native))
  expect_identical(summary(fresh()), summary(native))
  expect_identical(fresh()[[2]], native[[2]])
  expect_identical(fresh()[["type"]], "right")
  printed <- fresh()
  invisible(capture.output(print(printed)))
  expect_equal(
    survdiff(printed, group = c(1, 1, 2, 2, 2), subset = c(TRUE, TRUE, TRUE, TRUE, FALSE))$chisq,
    survdiff(native, group = c(1, 1, 2, 2, 2), subset = c(TRUE, TRUE, TRUE, TRUE, FALSE))$chisq
  )
  expect_equal(
    survdiff(fresh(), group = c(1, 1, 2, 2, 2), subset = c(TRUE, TRUE, TRUE, TRUE, FALSE))$chisq,
    survdiff(native, group = c(1, 1, 2, 2, 2), subset = c(TRUE, TRUE, TRUE, TRUE, FALSE))$chisq
  )
  expect_error(Surv(fresh(), type = "left"), "takes no other arguments")
  lung <- survival::lung
  response <- .wrap_python(.as_python_surv(Surv(lung$time, lung$status)),
    c("survival_py_surv", "survival_py_object"))
  expect_equal(
    sum(pyears(response ~ sex, data = lung, scale = 1)$pyears),
    sum(pyears(Surv(time, status) ~ sex, data = lung, scale = 1)$pyears)
  )
  x <- cbind(`(Intercept)` = 1, age = lung$age)
  expect_named(coef(survreg(response, x = x)), c("(Intercept)", "age"))
})

test_that("survSplit and survcheck read the response with survivalr's full Surv()", {
  lung <- survival::lung
  ours <- survcheck(Surv(time, status) ~ 1, data = lung, id = seq_len(nrow(lung)))
  stock <- survival::survcheck(survival::Surv(time, status) ~ 1, data = lung,
    id = seq_len(nrow(lung)))
  expect_equal(unclass(ours$transitions), unclass(stock$transitions), ignore_attr = TRUE)
  split <- survSplit(Surv(time, status) ~ age, data = lung, cut = c(100, 300), episode = "ep")
  # survival's survSplit reads a bare Surv() on the left-hand side.
  reference <- eval(quote(survival::survSplit(Surv(time, status) ~ age, data = lung,
    cut = c(100, 300), episode = "ep")), list(Surv = survival::Surv, lung = lung))
  expect_equal(split$status, reference$status)
  expect_equal(split$tstart, reference$tstart)
})
