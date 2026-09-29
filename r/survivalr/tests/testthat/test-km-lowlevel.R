normalize_km_missing <- function(x) {
  if (is.numeric(x)) x[is.nan(x)] <- NA_real_
  else if (is.list(x)) x <- lapply(x, normalize_km_missing)
  x
}

test_that("direct KM delegates prepared inputs and preserves raw R components", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"))
  times <- c(1, 1, 2, 3, 3, 4, 5, 6)
  event <- c(1, 0, 1, 1, 0, 1, 0, 0)
  right <- survival::Surv(times, event)
  counting <- survival::Surv(c(0,2,0,2,1,3,0,4), c(2,5,2,6,3,7,4,8),
                             c(0,1,1,0,0,1,0,1))
  groups <- factor(rep(c("b", "a"), 4), levels = c("b", "a"))
  single <- factor(rep("one", 8))
  compare <- function(x = groups, y = right, weights = rep(1, 8), id = NULL,
                      cluster = NULL, ...) {
    args <- c(list(x = x, y = y, weights = weights, id = id, cluster = cluster), list(...))
    actual <- suppressWarnings(do.call(survfitKM, args))
    expected <- suppressWarnings(do.call(survival::survfitKM, args))
    expect_identical(names(actual), names(expected))
    expect_equal(normalize_km_missing(actual), normalize_km_missing(expected), tolerance = 1e-10)
  }
  for (stype in 1:2) for (ctype in 1:2) {
    compare(stype = stype, ctype = ctype)
    compare(stype = stype, ctype = ctype, influence = TRUE)
    compare(stype = stype, ctype = ctype, cluster = rep(c("z", "b"), 4), influence = 3)
    compare(y = counting, stype = stype, ctype = ctype, id = rep(1:4, each = 2), entry = TRUE)
    compare(x = single, y = counting, stype = stype, ctype = ctype,
            id = rep(1:4, each = 2), entry = TRUE, influence = TRUE)
  }
  for (conf in c("plain", "log", "log-log", "logit", "arcsin", "none")) {
    for (lower in c("usual", "peto", "modified")) {
      compare(conf.type = conf, conf.lower = lower)
    }
  }
  compare(x = factor(groups, levels = c("empty", "b", "a", "unused")))
  compare(x = factor(c(rep("early", 4), rep("late", 4))), start.time = 4)
  compare(start.time = 2.5, influence = TRUE, cluster = 1:8)
  compare(y = survival::Surv(times - 4, event))
  compare(y = survival::Surv(c(1, 1 + 1e-10, 2, 3, 3 + 1e-10, 4, 5, 6), event))
  compare(weights = rep(1 + 1e-10, 8))
  compare(weights = c(1, 1, 1, 1, 1, 1, 1, 1 + 2e-8))
  compare(weights = c(0, 1, 1, 0, 1, 1, 1, 1))
  compare(weights = rep(c(.5, 1.5), 4), se.fit = FALSE)
  compare(se.fit = FALSE, influence = TRUE, cluster = 1:8)
  compare(conf.int = FALSE)
  compare(type = "fh2", stype = 99, ctype = 99)
  expect_warning(survfitKM(single, right, influence = TRUE, robust = FALSE),
                 "robust=FALSE implies influence=FALSE")
  expect_warning(survfitKM(single, right, cluster = 1:8, robust = FALSE),
                 "cluster specified with robust=FALSE, cluster ignored")
})

test_that("prepared KM handles empty levels and omitted arguments without retaining rows", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"))
  x <- factor(c("a", "a", "b", "b"), levels = c("empty", "a", "b", "unused"))
  y <- survival::Surv(c(1, 2, 3, 4), c(1, 0, 1, 0))
  fit <- survfitKM(x, y, start.time = 3, influence = TRUE)
  expect_identical(fit$n, c(0L, 0L, 2L, 0L))
  expect_identical(fit$strata, c(empty = 0L, a = 0L, b = 2L, unused = 0L))
  expect_length(fit$influence.surv, 4L)
  expect_null(fit$influence.surv[[1L]])
  expect_null(fit$influence.surv[[2L]])
  expect_null(fit$influence.surv[[4L]])
  expect_equal(fit$n.risk, c(2, 1))
  expect_null(fit$model)
  expect_error(survfitKM(1:4, y), "x must be a factor")
  expect_error(survfitKM(x, survival::Surv(1:4, c(1, 0, 1, 0), type = "left")),
               "Can only handle right censored or counting data")
})
