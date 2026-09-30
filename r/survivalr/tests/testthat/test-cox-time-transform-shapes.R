.tt_shape_data <- function() {
  d <- survival::lung[1:60, c("time", "status", "age", "sex")]
  d$time <- floor(d$time / 50) + 1
  d$status <- as.integer(d$status == 2)
  d$x <- (d$age - 60) / 10
  d$z <- sin(seq_len(nrow(d)))
  d$g <- factor(rep(c("a", "b", "c"), 20), levels = c("c", "b", "a"))
  d$w <- rep(c(.7, 1, 1.4), 20)
  d$o <- (seq_len(nrow(d)) %% 5 - 2) / 20
  d$id <- (seq_len(nrow(d)) - 1L) %/% 3L
  d
}

.tt_shape_compare <- function(actual, expected, residual_reference = expected) {
  expect_identical(names(coef(actual)), names(coef(expected)))
  expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 3e-7)
  expect_equal(vcov(actual), vcov(expected), tolerance = 3e-7)
  expect_equal(as.numeric(actual$loglik), expected$loglik, tolerance = 3e-7)
  expect_equal(unname(model.matrix(actual)), unname(expected$x), ignore_attr = TRUE)
  expect_equal(unname(residuals(actual)), unname(residuals(residual_reference)), tolerance = 3e-7)
  expect_equal(summary(actual)$n, expected$n)
}

test_that("matrix and factor callbacks rebuild Cox names and contrasts", {
  d <- .tt_shape_data()
  transforms <- list(
    named = function(x,t,...) cbind(log=x*log(t),root=x*sqrt(t)),
    unnamed = function(x,t,...) cbind(x*log(t),x*sqrt(t)),
    one = function(x,t,...) cbind(ignored=x*log(t)),
    logical = function(x,t,...) x>median(x),
    factor = function(x,t,...) factor(x>median(x), levels=c(TRUE,FALSE,"unused")),
    ordered = function(x,t,...) ordered(ifelse(x < -.5, "low", ifelse(x > .5,"high","middle")),
                                       levels=c("low","middle","high")),
    custom = function(x,t,...) {
      value <- factor(ifelse(x < -.5,"low",ifelse(x > .5,"high","middle")),
                      levels=c("low","middle","high"))
      contrasts(value) <- contr.sum(3)
      value
    })
  for (name in names(transforms)) for (method in c("efron","breslow")) {
    fun <- transforms[[name]]
    for (form in list(Surv(time,status)~x+tt(x), Surv(time,status)~tt(x)+tt(x):z)) {
      actual <- coxph(form,d,tt=fun,ties=method,x=TRUE,robust=FALSE)
      expected <- survival::coxph(form,d,tt=fun,ties=method,x=TRUE,robust=FALSE)
      .tt_shape_compare(actual,expected)
    }
  }
})

test_that("transforms preserve weights, strata, counting rows, and robust groups", {
  d <- .tt_shape_data()
  for (counting in c(FALSE,TRUE)) for (robust in c(FALSE,TRUE)) {
    frame <- d
    if (counting) frame$time <- frame$time + seq_len(nrow(frame))/1000
    frame$start <- pmin(seq_len(nrow(frame))%%3, frame$time-.5)
    form <- if (counting) Surv(start,time,status)~x+tt(x)+strata(g)+offset(o)
            else Surv(time,status)~x+tt(x)+strata(g)+offset(o)
    fun <- function(x,t,riskset,weights) cbind(log=x*log(t),root=x*sqrt(t))
    actual <- coxph(form,frame,tt=fun,weights=w,cluster=if (robust) frame$id else NULL,robust=robust,x=TRUE)
    expected <- survival::coxph(form,frame,tt=fun,weights=w,cluster=if (robust) frame$id else NULL,robust=robust,x=TRUE)
    .tt_shape_compare(actual,expected)
    detail <- coxph.detail(actual)
    if (!anyNA(coef(expected))) {
      reference <- tryCatch(survival::coxph.detail(expected), error = identity)
      if (inherits(reference,"error")) {
        expect_match(conditionMessage(reference),"NA/NaN/Inf in foreign function call")
      } else for (field in c("time","nrisk","nevent","hazard","score","means")) {
        expect_equal(detail[[field]],reference[[field]],ignore_attr=TRUE,tolerance=3e-7,info=field)
      }
    }
  }
})

test_that("transform callbacks run once in variable order including interactions", {
  d <- .tt_shape_data()
  seen <- character()
  first <- function(x,t,riskset,weights) {
    seen <<- c(seen,"x")
    expect_identical(min(riskset),1L)
    expect_null(weights)
    cbind(log=x*log(t),root=x*sqrt(t))
  }
  second <- function(x,t,riskset,weights) {
    seen <<- c(seen,"z")
    x*sqrt(t)
  }
  form <- Surv(time,status)~tt(x):z+tt(z)
  actual <- coxph(form,d,tt=list(first,second),x=TRUE)
  expect_identical(seen,c("x","z"))
  source <- deparse(survival::coxph)
  corrected <- eval(parse(text=sub("ntrans <- length(timetrans$terms)",
    "ntrans <- length(timetrans$vars)",source,fixed=TRUE)),envir=asNamespace("survival"))
  expected <- corrected(form,d,tt=list(first,second),x=TRUE)
  .tt_shape_compare(actual,expected)
  fun <- function(x,t,...) cbind(log=as.integer(x)*log(t),root=as.integer(x)*sqrt(t))
  .tt_shape_compare(coxph(Surv(time,status)~x+tt(g),d,tt=fun,x=TRUE),
                    survival::coxph(Surv(time,status)~x+tt(g),d,tt=fun,x=TRUE))
})

test_that("transformed penalties keep their controller, summary, and saved state", {
  d <- survival::ovarian
  for (fun in list(function(x,t,...) survival::pspline(x+log(t),df=3),
                   function(x,t,...) survivalr::pspline(x+log(t),df=3),
                   function(x,t,...) survival::ridge(cbind(x*log(t),x*sqrt(t)),theta=2))) {
    actual <- coxph(Surv(futime,fustat)~age+tt(age),d,tt=fun,x=TRUE)
    expected <- survival::coxph(Surv(futime,fustat)~age+tt(age),d,tt=fun,x=TRUE)
    # Stock coxpenal.fit's cleanup residual kernel gives nonzero sums within
    # these one-event strata. Recompute with the ordinary R Cox kernel at the
    # same fitted coefficients, replacing only aliased coefficients by zero.
    expanded <- data.frame(t=expected$y[,1],event=expected$y[,2],group=expected$strata)
    expanded$basis <- expected$x
    residual_reference <- survival::coxph(survival::Surv(t,event)~basis+strata(group),
      expanded,init=ifelse(is.na(coef(expected)),0,coef(expected)),
      control=survival::coxph.control(iter.max=0))
    .tt_shape_compare(actual,expected,residual_reference)
    expect_equal(summary(actual)$coefficients,summary(expected)$coefficients,tolerance=3e-7)
    expect_equal(summary(actual)$print2,summary(expected)$print2)
    expect_equal(as.numeric(actual$df),expected$df,tolerance=3e-7)
    expect_equal(actual$history[[1]]$theta,expected$history[[1]]$theta,tolerance=3e-7)
    restored <- unserialize(serialize(actual,NULL))
    expect_equal(summary(restored)$coefficients,summary(actual)$coefficients,tolerance=3e-7)
    expect_equal(summary(restored)$print2,summary(actual)$print2)
    expect_equal(vcov(restored),vcov(actual))
  }
})

test_that("malformed transform results fail before reaching the fitter", {
  d <- .tt_shape_data()
  expect_error(coxph(Surv(time,status)~tt(x),d,tt=function(x,t,...) matrix(0,1,2)),
               "one value per expanded row")
  expect_error(coxph(Surv(time,status)~tt(x),d,tt=function(x,t,...) rep(Inf,length(x))),
               "infinite predictor")
  expect_error(coxph(Surv(time,status)~tt(x),d,tt=function(x,t,...) factor(rep("same",length(x)))),
               "2 or more levels")
  expect_error(coxph(Surv(time,status)~tt(x):z,d,
                    tt=function(x,t,...) survival::ridge(x*log(t),theta=1)),
               "Penalty terms cannot be in an interaction")
  expect_error(coxph(Surv(time,status)~tt(x),d,
                    tt=function(x,t,...) survival::frailty(as.integer(x>0))),
               "factor-valued time-transform penalties")
})
