.dpqr_warning_capture <- function(expr) {
  messages <- character()
  error <- NULL
  value <- tryCatch(withCallingHandlers(force(expr),warning=function(condition) {
    messages <<- c(messages,conditionMessage(condition))
    invokeRestart("muffleWarning")
  }), error=function(condition) { error <<- conditionMessage(condition); NULL })
  list(value=value,warnings=messages,error=error)
}

test_that("DPQR numeric transport preserves empty operands and missing kinds", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (kind in c("dsurvreg","psurvreg","qsurvreg")) {
    actual <- get(kind,asNamespace("survivalr"))
    stock <- get(kind,asNamespace("survival"))
    for (distribution in c("gaussian","lognormal","weibull","exponential","rayleigh",
      "loglogistic","extreme","logistic","t")) {
      parms <- if(distribution=="t")4 else NULL
      for (values in list(c(NA_real_,NaN),c(NaN,NA_real_),c(.25,NA_real_,NaN),NULL,numeric()))
        for (mean in list(0,c(NA_real_,NaN),c(NaN,NA_real_),NULL))
          for (scale in list(1,c(NA_real_,NaN),c(NaN,NA_real_),NULL)) {
            args <- list(values,mean=mean,scale=scale,distribution=distribution)
            if (!is.null(parms)) args$parms <- parms
            expected <- .dpqr_warning_capture(do.call(stock,args))
            observed <- .dpqr_warning_capture(do.call(actual,args))
            label <- paste(kind,distribution,length(values),length(mean),length(scale))
            expect_identical(observed$error,expected$error,info=label)
            expect_equal(observed$value,expected$value,tolerance=2e-12,info=label)
            expect_identical(is.na(observed$value),is.na(expected$value),info=label)
            expect_identical(is.nan(observed$value),is.nan(expected$value),info=label)
            expect_identical(observed$warnings,expected$warnings,info=label)
          }
    }
  }
})

test_that("DPQR warn=2 stops before later custom callbacks", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  previous <- getOption("warn")
  on.exit(options(warn=previous),add=TRUE)
  options(warn=2)
  calls <- character()
  distribution <- list(name="DPQR warning transport sentinel",
    init=function(x,wt,...)c(0,1),deviance=function(...)list(center=0,loglik=0),
    density=function(z,...) {
      calls <<- c(calls,"density")
      cbind(pnorm(z),pnorm(-z),dnorm(z),-z,z^2-1)
    },quantile=function(p,...) { calls <<- c(calls,"quantile");qnorm(p) })
  expect_error(dsurvreg(1:3,mean=1:2,distribution=distribution),
    "longer object length is not a multiple")
  expect_identical(calls,character())
  expect_error(qsurvreg(c(.2,.4,.6),mean=1:2,distribution=distribution),
    "longer object length is not a multiple")
  expect_identical(calls,"quantile")
})

test_that("Python DPQR UserWarning callbacks become immediate R warnings", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  reticulate::py_run_string(paste(
    "import warnings, numpy as np",
    "from survival import _survival as _dpqr_core",
    "_dpqr_user_warning_events = []",
    "def _dpqr_user_density(z):",
    "    _dpqr_user_warning_events.append('before')",
    "    warnings.warn('DPQR callback UserWarning', UserWarning)",
    "    _dpqr_user_warning_events.append('after')",
    "    return np.column_stack((z*0+.5,z*0+.5,z*0+.2,-z,z*z-1))",
    "_dpqr_user_warning_distribution = _dpqr_core.SurvregDistribution.from_callbacks(",
    "    name='UserWarning transport', init=lambda y,w: [0,1],",
    "    density=_dpqr_user_density, deviance=lambda y,s: (np.zeros(len(y)),np.zeros(len(y))),",
    "    quantile=lambda p:p)",sep="\n"))
  distribution <- reticulate::py$`_dpqr_user_warning_distribution`
  value <- .dpqr_warning_capture(dsurvreg(1,mean=0,distribution=distribution))
  expect_null(value$error)
  expect_identical(value$warnings,"DPQR callback UserWarning")
  expect_equal(value$value,.2)
  previous <- getOption("warn")
  on.exit(options(warn=previous),add=TRUE)
  options(warn=2)
  reticulate::py_run_string("_dpqr_user_warning_events.clear()")
  expect_error(dsurvreg(1,mean=0,distribution=distribution),"DPQR callback UserWarning")
  expect_identical(unlist(reticulate::py$`_dpqr_user_warning_events`,use.names=FALSE),"before")
})

test_that("named Student-t DPQR distinguishes omitted NULL and empty parameters at the stock stage", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  parameters <- list(omitted=NULL,null=NULL,empty=numeric(),ordinary=4,
    missing=NA_real_,nan=NaN,recycled=c(3,5),list=list(4))
  for (kind in c("dsurvreg","psurvreg","qsurvreg","rsurvreg")) {
    actual <- get(kind,asNamespace("survivalr"))
    stock <- get(kind,asNamespace("survival"))
    for (mode in names(parameters)) for (recycle in c(FALSE,TRUE)) {
      args <- list(if(kind=="rsurvreg")3L else c(.2,.4),
        mean=if(recycle)1:3 else 0,scale=if(recycle)1:5 else 1,distribution="t")
      if(mode!="omitted") args["parms"] <- list(parameters[[mode]])
      set.seed(739)
      expected <- .dpqr_warning_capture(do.call(stock,args))
      stock_next <- runif(1)
      set.seed(739)
      observed <- .dpqr_warning_capture(do.call(actual,args))
      observed_next <- runif(1)
      label <- paste(kind,mode,recycle)
      if(is.null(expected$error)) {
        expect_null(observed$error,info=label)
        expect_equal(observed$value,expected$value,tolerance=2e-12,info=label)
        expect_identical(is.nan(observed$value),is.nan(expected$value),info=label)
      } else {
        expect_true(!is.null(observed$error) && grepl(expected$error,observed$error,fixed=TRUE),info=label)
      }
      expect_identical(observed$warnings,expected$warnings,info=label)
      expect_identical(observed_next,stock_next,info=label)
    }
  }
})

test_that("named non-Student-t DPQR keeps unused parameters lazy", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  for (kind in c("dsurvreg","psurvreg","qsurvreg","rsurvreg"))
    for (distribution in c("weibull","exponential","rayleigh","lognormal",
      "loglogistic","gaussian","logistic","extreme")) {
      actual <- get(kind,asNamespace("survivalr"))
      stock <- get(kind,asNamespace("survival"))
      query <- if(kind=="rsurvreg")2L else .5
      set.seed(17)
      expected <- stock(query,mean=0,distribution=distribution,parms=stop("must stay unused"))
      set.seed(17)
      observed <- actual(query,mean=0,distribution=distribution,parms=stop("must stay unused"))
      # The numeric adapter drops stock's empty scalar row name; this check
      # concerns unused promises and values rather than that existing metadata.
      expect_equal(unname(observed),unname(expected),tolerance=2e-12,info=paste(kind,distribution))
    }
})
