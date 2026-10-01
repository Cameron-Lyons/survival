.se_data <- function() {
  set.seed(715)
  data.frame(time = sample(2:20, 80, TRUE), status = rbinom(80, 1, .7),
    start = runif(80, 0, 1), score = rnorm(80), z = factor(rep(letters[1:2], 40)),
    group = factor(rep(c("b", "a"), each = 40)), wt = runif(80, .5, 2), off = rnorm(80),
    age = runif(80, 40, 70)*365.25, sex = rep(1:2, 40), year = as.Date("2000-01-01"),
    row.names = paste0("person", 1:80))
}

.se_equal <- function(actual, expected) {
  if (is.list(actual)) {
    actual$call <- expected$call <- NULL
    if (!is.null(actual$model)) {
      a <- attr(actual$model, "terms"); b <- attr(expected$model, "terms")
      environment(a) <- environment(b) <- baseenv()
      attr(actual$model, "terms") <- a; attr(expected$model, "terms") <- b
    }
  }
  expect_equal(actual, expected, tolerance = 1e-11)
}

test_that("Cox expected survival reconstructs an omitted training response", {
  d <- .se_data()
  d$time[1:3] <- c(2, 2 + 1e-12, 2 + 2e-12)
  for (timefix in c(FALSE, TRUE)) {
    fit <- survival::coxph(survival::Surv(time,status) ~ score + z, d,
      x = FALSE, y = FALSE, model = FALSE,
      control = survival::coxph.control(timefix = timefix))
    for (method in c("ederer", "conditional", "individual.h")) {
      .se_equal(survexp(Surv(time,status) ~ group, d, ratetable = fit, method = method),
        survival::survexp(Surv(time,status) ~ group, d, ratetable = fit, method = method))
    }
  }
})

test_that("Cox cohorts and individual predictions share numerical kernels", {
  d <- .se_data()
  for (counting in c(FALSE, TRUE)) for (ties in c("efron", "breslow", "exact")) {
    form <- if (counting) survival::Surv(start,time,status) ~ score + z else survival::Surv(time,status) ~ score + z
    fit <- survival::coxph(form, d, ties = ties, model = TRUE)
    # R 3.8-12's exact counting fitter omits the class and retains a numeric
    # method code. Repair only that metadata before checking its curve kernels.
    if (counting && ties == "exact") { class(fit) <- "coxph"; fit$method <- "exact" }
    for (method in c("ederer", "hakulinen", "conditional", "individual.h", "individual.s")) {
      for (requested in list(NULL, c(0, .5, 2, 2, 5, 12, 30))) {
        args <- list(formula = Surv(time,status) ~ group, data = d, ratetable = fit,
          weights = d$wt, method = method, x = TRUE, y = TRUE)
        if (!is.null(requested)) args$times <- requested
        .se_equal(do.call(survexp, args), do.call(survival::survexp, args))
      }
    }
  }
})

test_that("rate tables retain formula environments, metadata and numeric matrices", {
  d <- .se_data()
  for (method in c("ederer", "hakulinen", "conditional", "individual.h", "individual.s")) {
    .se_equal(survexp(Surv(time,status) ~ group + factor(sex), d, method = method, model = TRUE),
      survival::survexp(Surv(time,status) ~ group + factor(sex), d, method = method, model = TRUE))
  }
  env <- list2env(as.list(d), parent = environment())
  f <- as.formula("Surv(time,status) ~ group", env)
  .se_equal(survexp(f, times = c(2, 5)), survival::survexp(f, times = c(2, 5)))
})

test_that("Cox offsets use multiplicative relative risk for individual predictions", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status) ~ score + offset(off), d, weights = wt, model = TRUE)
  .se_equal(survexp(Surv(time,status) ~ group, d, ratetable = fit, method = "ederer", model = TRUE),
    survival::survexp(Surv(time,status) ~ group, d, ratetable = fit, method = "ederer", model = TRUE))
  # predict.coxph adds the new offset after exp; use the per-row survfit hazard
  # as an independent check of the intended multiplicative offset model.
  curves <- survival::survfit(fit, newdata = d, se.fit = FALSE)
  expected <- vapply(seq_len(nrow(d)), function(i) curves$cumhaz[max(which(curves$time <= d$time[i])), i], numeric(1))
  actual <- survexp(Surv(time,status) ~ 1, d, ratetable = fit, method = "individual.h")
  expect_equal(unname(actual), expected, tolerance = 1e-11)
})

test_that("Cox stratified cohorts use each row's baseline", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status) ~ score + strata(z), d, model = TRUE)
  .se_equal(survexp(~ group, d, ratetable = fit, method = "ederer", times = c(0, 3, 8)),
    survival::survexp(~ group, d, ratetable = fit, method = "ederer", times = c(0, 3, 8)))
  for (method in c("conditional", "hakulinen")) {
    actual <- survexp(Surv(time,status) ~ z, d, ratetable = fit, method = method)
    # The stock stratified conditional/Hakulinen branch errors in hazard[,].
    # Compare separate predictions using the same fitted coefficients/baselines.
    curves <- lapply(seq_len(nrow(d)), function(i) survival::survfit(fit, newdata = d[i,,drop=FALSE], se.fit=FALSE, censor=FALSE))
    grid <- sort(unique(unlist(lapply(curves, `[[`, "time"))))
    hazard <- vapply(curves, function(c) c(0,c$cumhaz)[findInterval(grid,c$time)+1L], numeric(length(grid)))
    survival <- exp(-hazard)
    delta <- apply(rbind(0,hazard),2,diff)
    expected <- sapply(levels(d$z), function(z) {
      keep <- d$z == z
      weight <- outer(grid, d$time, `<=`) * rep(keep, each=length(grid))
      if (method == "hakulinen") weight <- weight * rbind(1,survival[-nrow(survival),,drop=FALSE])
      exp(-cumsum(rowSums(delta*weight)/rowSums(weight)))
    })
    expect_equal(actual$time, grid)
    expect_equal(unname(actual$surv), unname(expected), tolerance=1e-11)
  }
})

test_that("Cox population mapping and exclusions keep source rows aligned", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status) ~ score + z, d, model = TRUE)
  input <- transform(d, changed = score/2, follow = time + 1)
  input$changed[c(2,4)] <- NA
  args <- list(formula = follow ~ group, data = input, ratetable = fit,
    rmap = quote(list(score=changed*2)), subset = quote(seq_len(nrow(input)) <= 30),
    na.action = na.exclude, times = c(0,3,8), model = TRUE)
  # do.call must retain the unevaluated mapping/subset expressions.
  .se_equal(do.call(survexp,args), do.call(survival::survexp,args))
  actual <- survexp(follow ~ group, input, ratetable = fit, rmap = list(score=changed*2),
    subset = seq_len(nrow(input)) <= 30, na.action=na.exclude, method="individual.h")
  selected <- input[1:30,,drop=FALSE]
  selected$score <- selected$changed*2
  expected <- predict(fit, newdata=selected, type="expected", na.action=na.pass)
  expect_equal(unname(actual), unname(expected), tolerance=1e-11)
  expect_true(all(is.na(actual[c(2,4)])))
  # The fitted response controls individual predictions, rather than follow.
  no_map <- survexp(follow ~ 1, d <- transform(d,follow=time+100), ratetable=fit, method="individual.h")
  expect_equal(unname(no_map), unname(predict(fit,newdata=d,type="expected")), tolerance=1e-11)
})

test_that("Python-backed rate models use the prepared population path", {
  d <- .se_data()
  for (stratified in c(FALSE, TRUE)) {
    form <- if (stratified) Surv(time,status) ~ score + strata(z) else Surv(time,status) ~ score + z
    fit <- coxph(form, d)
    stock <- survival::coxph(form, d, model=TRUE)
    for (method in c("ederer", "individual.h", "individual.s")) {
      .se_equal(survexp(Surv(time,status) ~ group, d, ratetable=fit, method=method, times=c(0,3,8)),
        survival::survexp(Surv(time,status) ~ group, d, ratetable=stock, method=method, times=c(0,3,8)))
    }
  }
})

test_that("formula calls evaluate responses and grouping transforms once", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status) ~ score, d, model=TRUE)
  nr <- ng <- 0L
  response <- function() { nr <<- nr+1L; Surv(d$time,d$status) }
  grouping <- function(x) { ng <<- ng+1L; factor(x) }
  out <- survexp(response() ~ grouping(z), d, ratetable=fit, model=TRUE)
  expect_equal(nr,1L); expect_equal(ng,1L)
  expect_false(is.function(attr(attr(out$model,"terms"),"predvars")[[2L]][[1L]]))
  env <- list2env(as.list(d),parent=environment())
  f <- as.formula("Surv(time,status) ~ z", env)
  .se_equal(survexp(f,ratetable=fit,times=c(0,3,8)), survival::survexp(f,ratetable=fit,times=c(0,3,8)))
})

test_that("Cox singleton populations and empty event baselines are well defined", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status) ~ score, d, model=TRUE)
  actual <- survexp(~1,d[1,,drop=FALSE],ratetable=fit,times=c(0,3,8))
  curve <- survival::survfit(fit,newdata=d[1,,drop=FALSE],se.fit=FALSE,censor=FALSE)
  expect_equal(unname(actual$surv), c(1,curve$surv)[findInterval(c(0,3,8),curve$time)+1L], tolerance=1e-11)
  expect_equal(actual$n.risk,c(1,1,1))
  none <- transform(d,status=0)
  empty <- survival::coxph(survival::Surv(time,status)~score,none,model=TRUE)
  out <- survexp(~group,none,ratetable=empty,times=c(0,3,8))
  expect_equal(unname(out$surv),matrix(1,3,2))
})

test_that("expected survival runs with reference numerical functions disabled", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status)~score,d,model=TRUE)
  expected <- survexp(~group,d,ratetable=fit,times=c(0,3,8))
  bridge <- survexp
  testthat::local_mocked_bindings(
    survexp=function(...) stop("reference expected survival"),
    survfit.coxph=function(...) stop("reference curves"),
    predict.coxph=function(...) stop("reference prediction"),
    .package="survival")
  .se_equal(bridge(~group,d,ratetable=fit,times=c(0,3,8)),expected)
  expect_length(bridge(time~1,d,ratetable=fit,method="individual.h"),nrow(d))
})

test_that("invalid expected-survival populations fail explicitly", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status)~score,d,model=TRUE)
  expect_error(survexp(~1,d,ratetable=list()),"Invalid rate table")
  expect_error(survexp(~1,d,ratetable=fit,method="conditional"),"response is required")
  expect_error(survexp(time~z*group,d,ratetable=fit),"interaction")
  expect_error(survexp(time~1,d,ratetable=fit,rmap=list(wrong=score)),"Variable not found")
  expect_error(survexp(time~1,d,ratetable=fit,times=c(3,1)),"increasing")
  expect_error(survexp(time~1,d,ratetable=fit,weights=rep(0,nrow(d))),"positive")
  expect_warning(survexp(time~1,d,ratetable=fit,se.fit=TRUE),"ignored")
})

test_that("penalized Cox rate models preserve fitted risk sets", {
  d <- .se_data()
  fit <- survival::coxph(survival::Surv(time,status)~ridge(score,theta=1),d,model=TRUE)
  for (method in c("ederer","hakulinen","conditional","individual.h")) {
    .se_equal(survexp(time~group,d,ratetable=fit,method=method),
      survival::survexp(time~group,d,ratetable=fit,method=method))
  }
  sparse <- survival::coxph(survival::Surv(time,status)~score+frailty(z,theta=.5,sparse=TRUE),d,
    model=TRUE,control=survival::coxph.control(iter.max=100))
  .se_equal(survexp(time~1,d,ratetable=sparse,method="individual.h"),
    survival::survexp(time~1,d,ratetable=sparse,method="individual.h"))
  expect_error(survexp(~1,d,ratetable=sparse),"frailty")
})

test_that("rate mappings preserve non-syntactic variable names", {
  d <- .se_data()
  d[["raw score"]] <- d$score
  fit <- survival::coxph(survival::Surv(time,status)~`raw score`,d,model=TRUE)
  expected <- survexp(~group,d,ratetable=fit)
  d$mapped <- d[["raw score"]]
  d[["raw score"]] <- NULL
  actual <- survexp(~group,d,ratetable=fit,rmap=list(`raw score`=mapped))
  .se_equal(actual,expected)
})

test_that("Cox design preparation handles null, transformed and interaction models", {
  d <- .se_data()
  for (form in list(survival::Surv(time,status)~poly(score,2),
                    survival::Surv(time,status)~score*z)) {
    # Reconstruct the training frame from the fit when it was not retained.
    fit <- survival::coxph(form,d)
    for (method in c("ederer","hakulinen","individual.h")) {
      .se_equal(survexp(time~group,d,ratetable=fit,method=method),
        survival::survexp(time~group,d,ratetable=fit,method=method))
    }
  }
  null <- survival::coxph(survival::Surv(time,status)~1,d)
  py_null <- coxph(Surv(time,status)~1,d)
  baseline <- survival::survfit(null,se.fit=FALSE,censor=FALSE)
  # R's survexp drops all rows while constructing its empty rate data frame.
  # A null model gives each retained population row the same baseline curve.
  for (fit in list(null,py_null)) {
    out <- survexp(~group,d,ratetable=fit)
    expect_equal(out$time,baseline$time)
    expect_equal(unname(out$surv),matrix(rep(baseline$surv,2),ncol=2),tolerance=1e-11)
    expect_equal(unname(out$n.risk),matrix(40,length(baseline$time),2))
  }
})
