test_that("prepared Yates prediction preserves R values and names", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  fit <- survival::coxph(survival::Surv(time, status) ~ age, survival::lung)
  for (horizon in c(0, 100, Inf)) {
    setup <- yates_setup(fit, "survival", options = list(rmean = horizon))
    reference <- survival::yates_setup(fit, "survival", options = list(rmean = horizon))
    for (eta in list(0, c(a = -1, b = .5), matrix(c(-1, .5), ncol = 1,
                     dimnames = list(c("a", "b"), NULL)), matrix(c(-1, .5), nrow = 1))) {
      expect_equal(setup$predict(eta), reference$predict(eta), tolerance = 1e-11)
    }
    expect_error(setup$predict(matrix(1, 2, 2)), "vector or single-row/column")
  }
  own_fit <- coxph(Surv(time, status) ~ age, survival::lung)
  own_setup <- yates_setup(own_fit, "survival", options = list(rmean = 100))
  reference <- survival::yates_setup(fit, "survival", options = list(rmean = 100))
  expect_equal(own_setup$predict(c(-1, 0, .5)), reference$predict(c(-1, 0, .5)), tolerance = 1e-10)
})

test_that("Yates summary aligns baseline times and derived curve uncertainty", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (kind in c("ordinary", "single", "empty")) for (nlevel in c(1L, 3L)) {
    data <- switch(kind,
      ordinary = survival::lung,
      single = data.frame(time = c(2, 2, 2), status = c(1, 1, 0)),
      empty = data.frame(time = c(1, 2, 3), status = c(0, 0, 0)))
    fit <- survival::coxph(survival::Surv(time, status) ~ 1, data)
    setup <- yates_setup(fit, "survival")
    mean <- setup$predict(seq_len(nlevel) / 10)
    variance <- matrix(.0001, nrow(mean), ncol(mean))
    result <- setup$summary(mean, variance)
    expected <- t(mean[, -c(1L, 2L), drop = FALSE])
    std <- t(sqrt(variance[, -c(1L, 2L), drop = FALSE]))
    expect_equal(result$surv, expected, tolerance = 1e-12)
    expect_equal(result$cumhaz, -log(expected), tolerance = 1e-12)
    expect_equal(result$std.err, std / expected, tolerance = 1e-12)
    expect_equal(result$std.chaz, result$std.err)
    expect_equal(nrow(result$surv), length(result$time))
    expect_equal(ncol(result$surv), nlevel)
    expect_equal(setup$summary(mean, variance), result)
  }
})

.yb_data <- function() {
  set.seed(825)
  data.frame(time = sample(2:45, 64, TRUE), event = rbinom(64, 1, .65),
    start = runif(64, 0, 1), x = rnorm(64), z = factor(rep(letters[1:4], 16)),
    off = runif(64, -1, 1), w = runif(64, .5, 2), id = rep(1:16, each = 4))
}

.yb_compare <- function(fit) {
  actual <- attr(yates_setup(fit, "survival"), "survivalr_prediction")$baseline
  expected <- suppressWarnings(survival::survfit(fit, censor = FALSE))
  actual$call <- expected$call <- NULL
  expect_equal(actual, expected, tolerance = 1e-10)
}

test_that("Yates baseline fields match shared Cox curves for R fits", {
  d <- .yb_data()
  for (counting in c(FALSE, TRUE)) for (ties in c("breslow", "efron")) {
    response <- if (counting) "Surv(start,time,event)" else "Surv(time,event)"
    for (rhs in c("1", "x+z", "poly(x,2)", "x+offset(off)", "x+z+cluster(id)", "x+I(2*x)")) {
      formula <- as.formula(paste(response, "~", rhs))
      for (retain in c(FALSE, TRUE)) {
        fit <- survival::coxph(formula, d, ties=ties, weights=w, model=retain, x=retain, y=retain)
        .yb_compare(fit)
      }
    }
  }
  for (counting in c(FALSE,TRUE)) {
    formula <- if (counting) Surv(start,time,event)~x else Surv(time,event)~x
    fit <- survival::coxph(formula,d,ties="exact",model=TRUE)
    # R's exact counting fitter leaves numeric method metadata and no class.
    if (counting) { class(fit) <- "coxph"; fit$method <- "exact" }
    .yb_compare(fit)
  }
})

test_that("Yates baselines preserve penalized and frailty curve conventions", {
  d <- .yb_data()
  for (formula in list(Surv(time,event)~ridge(x,theta=1),
                       Surv(time,event)~pspline(x,df=3),
                       Surv(time,event)~x+frailty(z,theta=.3,sparse=TRUE),
                       Surv(time,event)~x+frailty(z,distribution="gaussian",theta=.3,sparse=TRUE))) {
    fit <- survival::coxph(formula,d,model=TRUE,control=survival::coxph.control(iter.max=100))
    .yb_compare(fit)
  }
})

test_that("Yates setup retains curve data without retaining the fitted model", {
  d <- .yb_data()
  fit <- survival::coxph(Surv(time,event)~x,d,model=TRUE,x=TRUE)
  before <- fit[c("coefficients", "means", "var", "linear.predictors", "x", "y", "model")]
  setup <- yates_setup(fit,"survival")
  expect_false("fit" %in% ls(environment(setup$predict),all.names=TRUE))
  expect_false("fit" %in% ls(environment(setup$summary),all.names=TRUE))
  expect_equal(fit[names(before)],before)
  estimate <- setup$predict(c(a=-.5,b=.5))
  variance <- matrix(.001,nrow(estimate),ncol(estimate))
  expected <- setup$summary(estimate,variance)
  changed <- setup$summary(estimate,variance)
  changed$surv[1,1] <- -7
  expect_equal(setup$summary(estimate,variance),expected)
  expect_error(yates_setup(survival::coxph(Surv(time,event)~x+strata(z),d),"survival"),"stratified")
})

test_that("Yates survival uses no reference numerical functions", {
  d <- .yb_data()
  fit <- survival::coxph(Surv(time,event)~x+z,d,model=TRUE)
  bridge <- yates_setup
  expected <- bridge(fit,"survival")$predict(c(-1,0,1))
  local_mocked_bindings(survfit=function(...) stop("reference curve called"),
    survfit.coxph=function(...) stop("reference Cox curve called"),
    coxsurv.fit=function(...) stop("reference assembly called"),
    agsurv=function(...) stop("reference baseline called"),.package="survival")
  local_mocked_bindings(.call_r_api=function(...) stop("Python formula called"),.package="survivalr")
  expect_equal(bridge(fit,"survival")$predict(c(-1,0,1)),expected)
})

test_that("Yates reconstruction preserves time fixing, subsets and offsets", {
  d <- .yb_data()
  d$time[1:3] <- c(2, 2 + 1e-12, 2 + 2e-12)
  d$x[c(5, 8)] <- NA_real_
  for (timefix in c(FALSE, TRUE)) for (off_shift in c(0, 100)) {
    d$off <- d$off + off_shift
    fit <- survival::coxph(Surv(time,event) ~ x + z + offset(off), d,
      subset = time < 40, weights = w, na.action = na.exclude,
      x = FALSE, y = FALSE, model = FALSE,
      control = survival::coxph.control(timefix = timefix))
    .yb_compare(fit)
  }
  fit <- survival::coxph(Surv(time,event) ~ x, d, y = FALSE)
  d <- d[-1, , drop = FALSE]
  expect_error(yates_setup(fit, "survival"), "original data set")
})

test_that("Yates baselines keep sample counts with only one observed time", {
  for (event in c(0, 1)) {
    fit <- survival::coxph(Surv(time,event) ~ 1,
      data.frame(time = c(2, 2), event = rep(event, 2)))
    baseline <- attr(yates_setup(fit, "survival"), "survivalr_prediction")$baseline
    expect_equal(baseline$n, 2L)
    expect_equal(baseline$time, if (event) 2 else numeric())
    expect_equal(baseline$cumhaz, if (event) 1.5 else numeric())
  }
  d <- .yb_data()
  d$time <- d$time - 50
  .yb_compare(survival::coxph(Surv(time,event) ~ x, d))
})

test_that("Yates can use retained design and response after source data removal", {
  d <- .yb_data()
  fit <- survival::coxph(Surv(time,event) ~ x + z, d, x = TRUE, y = TRUE)
  rm(d)
  .yb_compare(fit)
})
