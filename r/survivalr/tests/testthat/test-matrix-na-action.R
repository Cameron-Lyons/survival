test_that("fresh model matrices honor global omission options", {
  original <- options(na.action="na.omit")
  on.exit(options(original),add=TRUE)
  setup <- .matrix_na_setup()
  for(kind in c("coxph","survreg")) for(rhs in setup$formulas) for(named in c(FALSE,TRUE)) {
    d <- .aft_matrix_input_data(setup,"complete",named)
    fit <- function(namespace) get(kind,asNamespace(namespace))(
      as.formula(paste("Surv(futime,fustat)~",rhs)),d,x=TRUE,model=TRUE)
    actual <- .logical_matrix_capture(function() fit("survivalr"))
    expected <- .logical_matrix_capture(function() fit("survival"))
    expect_identical(gsub("[[:space:]]+","",actual$warnings),gsub("[[:space:]]+","",expected$warnings))
    for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
      options(na.action=action)
      for(input in setup$inputs) {
        nd <- .matrix_na_newdata(d,input)
        call <- function(fit) if(is.null(nd)) model.matrix(fit) else model.matrix(fit,nd)
        stock <- .logical_matrix_capture(function() call(expected$value))
        result <- .logical_matrix_capture(function() call(actual$value))
        info <- paste(kind,rhs,named,action,input)
        if(is.list(stock$value) && !is.null(stock$value$error)) {
          expect_match(result$value$error,"missing values",info=info)
        } else {
          .logical_matrix_compare(result$value,stock$value,info)
          expect_identical(is.na(result$value),is.na(stock$value),info=info)
          expect_identical(is.nan(result$value),is.nan(stock$value),info=info)
          # Python domain warnings retain the expression context.
          expect_identical(sub(" in (log|sqrt)\\(.*\\)$","",result$warnings),stock$warnings,info=info)
        }
      }
    }
    options(na.action="na.omit")
  }
})

test_that("new-data matrix calls ignore explicit na.action as stock does", {
  original <- options(na.action="na.pass")
  on.exit(options(original),add=TRUE)
  d <- .matrix_na_setup()$data
  for(kind in c("coxph","survreg")) {
    fit <- get(kind,asNamespace("survivalr"))(Surv(futime,fustat)~age,d,x=TRUE)
    nd <- d[1:3,];nd$age[2L]<-NA_real_
    expect_equal(model.matrix(fit,nd,na.action=na.fail),model.matrix(fit,nd))
    options(na.action="na.fail")
    expect_error(model.matrix(fit,nd,na.action=na.pass),"missing values")
    expect_equal(nrow(model.matrix(fit)),nrow(d))
    options(na.action="na.pass")
  }
})

test_that("stored omission masks survive changed global options without cached matrices", {
  original <- options(na.action="na.omit")
  on.exit(options(original),add=TRUE)
  d <- .matrix_na_setup()$data
  d$age[2L] <- NA_real_
  for(kind in c("coxph","survreg")) for(cached_frame in c(FALSE,TRUE)) {
    actual <- get(kind,asNamespace("survivalr"))(Surv(futime,fustat)~age,d,
      x=FALSE,model=cached_frame,na.action=na.exclude)
    stock <- get(kind,asNamespace("survival"))(Surv(futime,fustat)~age,d,
      x=FALSE,model=cached_frame,na.action=na.exclude)
    for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
      options(na.action=action)
      .logical_matrix_compare(model.matrix(actual),model.matrix(stock),paste(kind,cached_frame,action))
    }
    options(na.action="na.omit")
  }
})
