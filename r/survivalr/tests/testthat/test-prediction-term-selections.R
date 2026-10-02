.term_prediction_capture <- function(fun) {
  messages <- character()
  value <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = messages)
}

.term_prediction_compare <- function(actual, expected) {
  expect_identical(dim(actual), dim(expected))
  expect_identical(colnames(actual), colnames(expected))
  expect_identical(is.na(actual), is.na(expected), ignore_attr = TRUE)
  expect_identical(is.nan(actual), is.nan(expected), ignore_attr = TRUE)
  expect_equal(unname(actual), unname(expected), ignore_attr = TRUE, tolerance = 3e-7)
}

test_that("term prediction subscripts retain R shapes, labels, values and warning conditions", {
  d <- survival::ovarian
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
  nd <- d[c(3L,4L,5L,6L,7L,8L), ]; nd$age[2L] <- NA; nd$rx[4L] <- NA
  d$age[c(2L,9L)] <- NA
  specs <- list(
    aft_plain = c("survreg", "age + rx"),
    aft_strata_first = c("survreg", "strata(cl) + age + rx"),
    aft_strata_last = c("survreg", "age + rx + strata(cl)"),
    aft_ridge = c("survreg", "age + ridge(rx, theta = 2)"),
    aft_spline = c("survreg", "pspline(age, df = 3) + rx"),
    cox_strata_first = c("coxph", "strata(cl) + age + rx"),
    cox_strata_last = c("coxph", "age + rx + strata(cl)")
  )
  for (name in names(specs)) {
    spec <- specs[[name]]
    options <- list(as.formula(paste("Surv(futime,fustat)~", spec[[2L]])), data = d,
                    x = TRUE, na.action = na.exclude)
    actual <- do.call(get(spec[[1L]]), options)
    reference <- do.call(get(spec[[1L]], envir = asNamespace("survival")), options)
    intended <- colnames(predict(reference, type = "terms"))
    if (spec[[1L]] == "survreg") {
      tt <- reference$terms
      special <- attr(tt,"specials")$strata
      if (length(special)) tt <- survival:::drop.special(tt, special)
      groups <- survival::attrassign(model.matrix(reference), tt)
      groups$"(Intercept)" <- NULL
      intended <- names(groups)
    }
    selectors <- list(null = NULL, empty = integer(), zero = 0, negative = -1,
      repeated = c(2L,1L,2L), fractional = c(1.9,2.7), true = TRUE, false = FALSE,
      logical = c(TRUE,FALSE), numeric_missing = c(1,NA_real_), logical_missing = c(TRUE,NA),
      missing_numeric_only = NA_real_, missing_logical_only = NA, mixed_sign = c(1,-2),
      missing_numeric_and_zero = c(NA_real_,0), missing_numeric_invalid = c(NA_real_,999),
      negative_missing = c(-1,NA_real_), long_logical = c(TRUE,FALSE,FALSE),
      overflow = c(1,Inf), named = intended[c(2L,1L)],
      factor = factor(intended[c(2L,1L)], levels = rev(intended)))
    for (new in list(NULL, nd)) for (se in c(FALSE,TRUE)) {
      args <- list(object = reference, type = "terms", se.fit = se)
      if (!is.null(new)) args$newdata <- new
      full <- do.call(predict, args)
      rename <- function(value) { colnames(value) <- intended; value }
      if (se) { full$fit <- rename(full$fit); full$se.fit <- rename(full$se.fit) }
      else full <- rename(full)
      for (choice in names(selectors)) {
        selection <- selectors[[choice]]
        args$object <- actual; args["terms"] <- list(selection)
        result <- .term_prediction_capture(function() do.call(predict, args))
        if (spec[[1L]] == "survreg") {
          expected <- .term_prediction_capture(function() {
            select <- function(value) if (is.null(selection)) value else value[,selection,drop=FALSE]
            if (se) list(fit = select(full$fit), se.fit = select(full$se.fit)) else select(full)
          })
        } else {
          args$object <- reference
          expected <- .term_prediction_capture(function() do.call(predict, args))
        }
        expect_identical(result$warnings, expected$warnings, info = paste(name,choice,se))
        if (is.list(expected$value) && !is.null(expected$value$error)) {
          expect_match(result$value$error, expected$value$error, fixed = TRUE)
        } else if (se) {
          .term_prediction_compare(result$value$fit, expected$value$fit)
          .term_prediction_compare(result$value$se.fit, expected$value$se.fit)
        } else .term_prediction_compare(result$value, expected$value)
      }
    }
  }
})

test_that("omitted new-data rows and empty selections retain prediction widths", {
  d <- survival::ovarian
  actual <- survreg(Surv(futime,fustat)~age+rx,d)
  reference <- survival::survreg(Surv(futime,fustat)~age+rx,d)
  nd <- d[seq_len(6L), ]; nd$age[c(2L,4L)] <- NA
  for (action in c("na.pass","na.omit","na.exclude")) for (selection in list(0,-1,c(TRUE,NA),NA_real_)) {
    options <- list(newdata = nd, type = "terms", terms = selection, se.fit = TRUE, na.action = action)
    result <- do.call(predict, c(list(actual),options))
    expected <- do.call(predict, c(list(reference),options))
    .term_prediction_compare(result$fit, expected$fit)
    .term_prediction_compare(result$se.fit, expected$se.fit)
  }
  nd$age[] <- NA
  result <- predict(actual, nd, type = "terms", terms = c(TRUE,NA), se.fit = TRUE, na.action = na.omit)
  expect_identical(dim(result$fit), c(0L,2L))
  expect_identical(colnames(result$fit), c("age",NA_character_))
  expect_identical(dim(result$se.fit), c(0L,2L))
  fitted_terms <- fitted(actual, type = "terms", terms = -1, se.fit = TRUE)
  selected <- predict(actual, type = "terms", terms = -1, se.fit = TRUE)
  expect_equal(fitted_terms, selected)
})
