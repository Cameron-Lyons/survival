test_that("transforms precede model-matrix row omission and signal warnings on failure", {
  original <- options(na.action="na.omit")
  on.exit(options(original),add=TRUE)
  setup <- .formula_warning_setup()
  for(kind in c("coxph","survreg"))for(rhs in setup$formulas)for(named in c(FALSE,TRUE)) {
    d <- .aft_matrix_input_data(setup,"complete",named)
    fit <- function(namespace)get(kind,asNamespace(namespace))(
      as.formula(paste("Surv(futime,fustat)~",rhs)),d,x=TRUE,model=TRUE)
    actual <- .logical_matrix_capture(function()fit("survivalr"))
    stock <- .logical_matrix_capture(function()fit("survival"))
    expect_identical(gsub("[[:space:]]+","",actual$warnings),gsub("[[:space:]]+","",stock$warnings))
    for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
      options(na.action=action)
      for(input in setup$inputs) {
        nd <- .formula_warning_newdata(d,input)
        result <- .logical_matrix_capture(function()model.matrix(actual$value,nd))
        expected <- .logical_matrix_capture(function()model.matrix(stock$value,nd))
        info <- paste(kind,rhs,named,action,input)
        if(is.list(expected$value)&&!is.null(expected$value$error)) {
          expect_match(result$value$error,"missing values",info=info)
        } else {
          .logical_matrix_compare(result$value,expected$value,info)
          expect_identical(is.na(result$value),is.na(expected$value),info=info)
          expect_identical(is.nan(result$value),is.nan(expected$value),info=info)
        }
        expect_identical(gsub("[[:space:]]+","",.formula_warning_messages(result$warnings)),
          gsub("[[:space:]]+","",expected$warnings),info=info)
      }
    }
    options(na.action="na.omit")
  }
})

test_that("training transforms precede subsets, weights and omission actions", {
  d <- .formula_warning_setup()$data
  for(kind in c("coxph","survreg"))for(rhs in c("age + log(z)","age + sqrt(z)","age + sqrt(z) - sqrt(z)")) {
    for(variant in c("covariate","weights","subset","pass_unused")) {
      if(variant=="pass_unused"&&rhs!="age + sqrt(z) - sqrt(z)")next
      training <- .formula_warning_training(d,variant)
      for(action in if(variant=="pass_unused")"na.pass"else c("na.omit","na.exclude","na.fail")) {
        fit <- function(namespace)do.call(get(kind,asNamespace(namespace)),
          c(list(as.formula(paste("Surv(futime,fustat)~",rhs)),training$data,na.action=action),training$args))
        actual <- .logical_matrix_capture(function()fit("survivalr"))
        stock <- .logical_matrix_capture(function()fit("survival"))
        info <- paste(kind,rhs,variant,action)
        if(is.list(stock$value)&&!is.null(stock$value$error)) {
          expect_match(actual$value$error,"missing values",info=info)
        } else {
          expect_equal(coef(actual$value),coef(stock$value),tolerance=3e-7,info=info)
          expect_equal(vcov(actual$value),vcov(stock$value),tolerance=3e-7,info=info)
          .logical_matrix_compare(model.matrix(actual$value),model.matrix(stock$value),info)
        }
        expect_identical(gsub("[[:space:]]+","",.formula_warning_messages(actual$warnings)),
          gsub("[[:space:]]+","",stock$warnings),info=info)
      }
    }
  }
})

test_that("prediction warnings and whole results follow evaluated variables", {
  d <- .formula_warning_setup()$data
  for(kind in c("coxph","survreg"))for(rhs in c("age + log(z)","age + I(log(z) + sqrt(w))", "age + sqrt(z) - sqrt(z)")) {
    fit <- function(namespace)get(kind,asNamespace(namespace))(
      as.formula(paste("Surv(futime,fustat)~",rhs)),d,x=TRUE,model=TRUE)
    actual <- fit("survivalr"); stock <- fit("survival")
    for(input in c("domain","overlap_covariate","all_omitted"))for(action in c("na.omit","na.exclude","na.pass","na.fail")) {
      nd <- .formula_warning_newdata(d,input)
      for(type in if(kind=="coxph")c("lp","terms")else c("link","terms")) {
        call <- function(fit)predict(fit,newdata=nd,type=type,se.fit=TRUE,na.action=action)
        result <- .logical_matrix_capture(function()call(actual))
        expected <- .logical_matrix_capture(function()call(stock))
        info <- paste(kind,rhs,input,action,type)
        if(!is.null(expected$value$error)) {
          if(grepl("missing values",expected$value$error)) {
            expect_match(result$value$error,"missing values",info=info)
          } else {
            # Stock AFT predict.terms loses model-matrix attributes at zero rows.
            expect_identical(expected$value$error,"argument is not really a model matrix",info=info)
            expect_identical(kind,"survreg",info=info)
            expect_identical(type,"terms",info=info)
            expect_identical(input,"all_omitted",info=info)
            rows <- if(action=="na.exclude")nrow(nd)else 0L
            columns <- attr(terms(stock),"term.labels")
            for(field in c("fit","se.fit")) {
              expect_identical(dim(result$value[[field]]),c(rows,length(columns)),info=info)
              expect_identical(colnames(result$value[[field]]),columns,info=info)
              expect_true(all(is.na(result$value[[field]])),info=info)
            }
          }
        } else {
          expect_named(result$value,names(expected$value),info=info)
          for(field in names(expected$value)) {
            expect_identical(dim(result$value[[field]]),dim(expected$value[[field]]),info=info)
            expect_identical(names(result$value[[field]]),names(expected$value[[field]]),info=info)
            expect_identical(dimnames(result$value[[field]]),dimnames(expected$value[[field]]),info=info)
            expect_equal(as.numeric(result$value[[field]]),as.numeric(expected$value[[field]]),tolerance=3e-7,info=info)
            expect_identical(is.na(result$value[[field]]),is.na(expected$value[[field]]),info=info)
            expect_identical(is.nan(result$value[[field]]),is.nan(expected$value[[field]]),info=info)
          }
        }
        expect_identical(.formula_warning_messages(result$warnings),expected$warnings,info=info)
      }
    }
  }
})

test_that("other formula APIs signal domain warnings before subset and errors", {
  d <- .formula_warning_setup()$data;d$z[2]<- -1
  for(subset in list(NULL,setdiff(seq_len(nrow(d)),2L)))for(action in c("na.omit","na.fail")) {
    for(api in c("survfit","survdiff","concordance","rttright")) {
      form <- if(api=="concordance")Surv(futime,fustat)~log(z) else Surv(futime,fustat)~I(log(z)>1.8)
      call <- function(namespace) {
        function_name <- if(api %in% c("survfit","concordance")&&namespace=="survival")paste0(api,".formula")else api
        get(function_name,asNamespace(namespace))(form,d,subset=subset,na.action=action)
      }
      actual <- .logical_matrix_capture(function()call("survivalr"))
      stock <- .logical_matrix_capture(function()call("survival"))
      info <- paste(api,length(subset),action)
      if(is.list(stock$value)&&!is.null(stock$value$error)) {
        expect_match(actual$value$error,"missing values",info=info)
      } else if(api=="survfit") {
        frame <- as.data.frame(actual$value)
        for(field in c("time","n.event","n.risk","surv"))expect_equal(frame[[field]],stock$value[[field]],info=info)
      } else if(api=="survdiff") {
        expect_equal(as.numeric(actual$value$chisq),stock$value$chisq,info=info)
      } else if(api=="concordance") {
        for(field in c("concordance","var","count"))expect_equal(as.numeric(actual$value[[field]]),as.numeric(stock$value[[field]]),info=info)
      } else {
        expect_equal(as.numeric(actual$value),as.numeric(stock$value),info=info)
      }
      expect_identical(.formula_warning_messages(actual$warnings),stock$warnings,info=info)
    }
  }
})
