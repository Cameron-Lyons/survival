.prediction_rows_compare <- function(actual, expected, info) {
  if (is.list(expected)) {
    expect_named(actual, names(expected), info = info)
    for (field in names(expected)) .prediction_rows_compare(actual[[field]], expected[[field]], paste(info, field))
    return(invisible(NULL))
  }
  expect_identical(dim(actual), dim(expected), info = info)
  expect_identical(names(actual), names(expected), info = info)
  expect_identical(dimnames(actual), dimnames(expected), info = info)
  expect_identical(is.na(actual), is.na(expected), ignore_attr = TRUE, info = info)
  expect_identical(is.nan(actual), is.nan(expected), ignore_attr = TRUE, info = info)
  expect_equal(as.numeric(actual), as.numeric(expected), tolerance = 3e-7, info = info)
}

test_that("predictions preserve stock row labels and intentionally unnamed results", {
  d <- survival::ovarian
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
  row.names(d) <- c("case:1", "case:1.1", paste0("patient / ", seq.int(3L, nrow(d))))
  nd <- d[c(3L,4L,5L,6L), ]; row.names(nd) <- c("四", "two / 2", "six:6", "one")
  nd$age[c(2L,4L)] <- NA
  d$age[c(2L,9L)] <- NA
  subset <- c(12L,2L,1L,9L,1L,setdiff(seq_len(nrow(d)), c(12L,2L,1L,9L)))
  specs <- list(
    cox = c("coxph", "age + rx"),
    cox_strata = c("coxph", "age + rx + strata(cl)"),
    cox_ridge = c("coxph", "age + ridge(rx, theta = 2)"),
    cox_sparse = c("coxph", "age + frailty(cl, theta = 0.5, sparse = TRUE)"),
    cox_sparse_only = c("coxph", "frailty(cl, theta = 0.5, sparse = TRUE)"),
    aft = c("survreg", "age + rx"),
    aft_strata = c("survreg", "age + rx + strata(cl)"),
    aft_ridge = c("survreg", "age + ridge(rx, theta = 2)"),
    aft_spline = c("survreg", "pspline(age, df = 3) + rx"),
    aft_fixed = c("survreg", "age + rx")
  )
  for (name in names(specs)) for (action in c("na.omit", "na.exclude")) {
    spec <- specs[[name]]
    options <- list(as.formula(paste("Surv(futime,fustat)~", spec[[2L]])), data = d,
                    subset = subset, na.action = action, x = TRUE)
    if (name == "aft_fixed") options$scale <- 1
    actual <- do.call(get(spec[[1L]]), options)
    reference <- do.call(get(spec[[1L]], envir = asNamespace("survival")), options)
    types <- if (grepl("sparse", name)) c("lp", "risk", "terms") else
      if (spec[[1L]] == "coxph") c("lp", "risk", "expected", "terms", "survival") else
      c("response", "link", "terms", "quantile", "uquantile")
    for (type in types) for (se in c(FALSE,TRUE)) {
      args <- list(type = type, se.fit = se)
      for (data in list(NULL,nd,nd[1L,,drop=FALSE])) for (new_action in c("na.pass","na.omit","na.exclude")) {
        if (is.null(data) && new_action != "na.pass") next
        if (!is.null(data)) { args$newdata <- data; args$na.action <- new_action }
        else { args$newdata <- NULL; args$na.action <- NULL }
        info <- paste(name, action, type, se, if (is.null(data)) "stored" else nrow(data), new_action)
        result <- do.call(predict, c(list(actual),args))
        prediction_reference <- reference
        # The penalized wrapper refuses survival before the supported ordinary
        # prediction method; compare that method on the same fitted state.
        if (type == "survival" && inherits(reference, "coxph.penal")) class(prediction_reference) <- "coxph"
        if (name == "cox_sparse" && type == "terms") {
          prediction_reference$x <- reference$x[, "age", drop = FALSE]
          prediction_reference$terms <- terms(Surv(futime,fustat)~age)
          prediction_reference$assign <- list(age = 1L)
          prediction_reference$pterms <- c(age = 0)
          class(prediction_reference) <- "coxph"
        }
        expected <- do.call(predict, c(list(prediction_reference),args))
        if (name == "cox_sparse" && type == "terms") {
          base <- if (se) expected$fit else expected
          if (is.null(data)) {
            index <- as.integer(factor(reference$x[,2L]))
            frail <- naresid(reference$na.action, reference$frail[index])
            errors <- naresid(reference$na.action, sqrt(reference$fvar[index]))
          } else {
            frail <- errors <- numeric(nrow(base))
            if (new_action == "na.exclude") frail[is.na(base[,1L])] <- errors[is.na(base[,1L])] <- NA
          }
          add <- function(x, values) {
            result <- cbind(x, values)
            colnames(result) <- names(reference$pterms)
            result
          }
          expected <- if (se) list(fit = add(expected$fit, frail), se.fit = add(expected$se.fit, errors)) else add(expected, frail)
        }
        .prediction_rows_compare(result, expected, info)
      }
      if (spec[[1L]] == "survreg") {
        .prediction_rows_compare(do.call(fitted, c(list(actual),list(type=type,se.fit=se))),
                                 do.call(predict,c(list(reference),list(type=type,se.fit=se))),
                                 paste(name,action,type,se,"fitted"))
      }
    }
    if (spec[[1L]] == "coxph") .prediction_rows_compare(fitted(actual), fitted(reference), paste(name,action,"fitted"))
  }
})

test_that("quantile labels retain scale names, scalar errors and fixed-scale repetitions", {
  d <- survival::ovarian
  row.names(d) <- paste0("subject:",seq_len(nrow(d)))
  d$cl <- factor(rep(c("a","b","c"),length.out=nrow(d)),levels=c("c","b","a"))
  nd <- d[1:4,]; nd$age[2L] <- NA
  d$age[9L] <- NA
  for (scale in c(0,1)) for (stratified in c(FALSE,TRUE)) {
    if (scale==1 && stratified) next
    formula <- if(stratified) Surv(futime,fustat)~age+rx+strata(cl) else Surv(futime,fustat)~age+rx
    actual <- survreg(formula,d,scale=scale,x=TRUE,na.action=na.exclude)
    expected <- survival::survreg(formula,d,scale=scale,x=TRUE,na.action=na.exclude)
    for(type in c("quantile","uquantile")) for(p in list(.5,c(.1,.5,.9))) for(data in list(NULL,nd,nd[1L,,drop=FALSE])) {
      args <- list(type=type,p=p,se.fit=TRUE)
      if(!is.null(data)) {args$newdata<-data;args$na.action<-na.exclude}
      .prediction_rows_compare(do.call(predict,c(list(actual),args)),do.call(predict,c(list(expected),args)),
                               paste(scale,stratified,type,length(p),if(is.null(data))"stored" else nrow(data)))
    }
  }
})

test_that("selected and entirely omitted term matrices retain row names and widths", {
  d <- survival::ovarian; row.names(d) <- paste0("subject:",seq_len(nrow(d)))
  nd <- d[1:4,]; nd$age[c(2L,4L)] <- NA
  for(kind in c("coxph","survreg")) {
    actual <- do.call(get(kind),list(Surv(futime,fustat)~age+rx,data=d))
    expected <- do.call(get(kind,envir=asNamespace("survival")),list(Surv(futime,fustat)~age+rx,data=d))
    for(na in c("na.pass","na.omit","na.exclude")) for(selection in list(integer(),c(2L,1L,2L))) {
      args <- list(newdata=nd,type="terms",terms=selection,se.fit=TRUE,na.action=na)
      .prediction_rows_compare(do.call(predict,c(list(actual),args)),do.call(predict,c(list(expected),args)),paste(kind,na,length(selection)))
    }
    all_missing <- nd; all_missing$age[] <- NA
    for(selection in list(integer(),c(2L,1L,2L))) {
      args <- list(newdata=all_missing,type="terms",terms=selection,se.fit=TRUE,na.action=na.omit)
      if (kind == "survreg") {
        expect_error(do.call(predict,c(list(expected),args)), "argument is not really a model matrix", fixed=TRUE)
        empty <- matrix(numeric(), 0L, length(selection), dimnames=list(NULL,c("age","rx")[selection]))
        reference <- list(fit=empty,se.fit=empty)
      } else reference <- do.call(predict,c(list(expected),args))
      .prediction_rows_compare(do.call(predict,c(list(actual),args)),reference,paste(kind,"omitted",length(selection)))
    }
  }
})
