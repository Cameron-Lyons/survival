#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/prediction_row_reference.json"
d <- ovarian
# A reserved suffix, repeated subset and non-syntactic names distinguish source labels
# from row positions and exercise data.frame's make.unique before NA removal.
row.names(d) <- c("case:1", "case:1.1", paste0("patient / ", seq.int(3L,nrow(d))))
d$cl <- factor(rep(c("a","b","c"),length.out=nrow(d)),levels=c("c","b","a"))
nd <- d[c(3L,4L,5L,6L),]; row.names(nd) <- c("四","two / 2","six:6","one")
nd$age[c(2L,4L)] <- NA
single <- nd[1L,,drop=FALSE]
d$age[c(2L,9L)] <- NA
subset <- c(12L,2L,1L,9L,1L,setdiff(seq_len(nrow(d)),c(12L,2L,1L,9L)))
specs <- list(cox=c("coxph","age + rx"),cox_strata=c("coxph","age + rx + strata(cl)"),
  cox_ridge=c("coxph","age + ridge(rx, theta = 2)"),
  cox_sparse_only=c("coxph","frailty(cl, theta = 0.5, sparse = TRUE)"),
  aft=c("survreg","age + rx"),aft_strata=c("survreg","age + rx + strata(cl)"),
  aft_ridge=c("survreg","age + ridge(rx, theta = 2)"),
  aft_spline=c("survreg","pspline(age, df = 3) + rx"),aft_fixed=c("survreg","age + rx"),
  aft_intercept=c("survreg","1"))
capture <- function(fun) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(fun(),warning=function(w) {
    warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")
  }),error=function(e) list(error=conditionMessage(e)))
  list(value=value,warnings=I(warnings))
}
encode <- function(value) {
  if (is.list(value)) return(lapply(value,encode))
  list(values=I(as.numeric(value)),dim=if(is.null(dim(value)))NULL else I(dim(value)),
       names=if(is.null(names(value)))NULL else I(names(value)),
       rows=if(is.null(rownames(value)))NULL else I(rownames(value)),
       columns=if(is.null(colnames(value)))NULL else I(colnames(value)))
}
cases <- list()
for (name in names(specs)) for (action in c("na.omit","na.exclude")) {
  spec <- specs[[name]]
  opts <- list(formula=as.formula(paste("Surv(futime,fustat)~",spec[[2L]])),data=d,
               subset=subset,x=TRUE,na.action=action)
  if (name=="aft_fixed") opts$scale <- 1
  fit <- do.call(get(spec[[1L]]),opts)
  types <- if (name=="cox_sparse_only") c("lp","risk","terms") else
    if(spec[[1L]]=="coxph") c("lp","risk","terms","expected","survival") else
      c("response","link","terms","quantile","uquantile")
  for(type in types) for(se in c(FALSE,TRUE)) {
    probabilities <- if(type %in% c("quantile","uquantile")) list(.5,c(.1,.5,.9)) else list(NULL)
    for(p in probabilities) for(source in c("stored","new","single")) for(na in c("na.pass","na.omit","na.exclude")) {
      if(source=="stored" && na!="na.pass") next
      args <- list(object=fit,type=type,se.fit=se)
      if(!is.null(p)) args$p <- p
      if(source!="stored") {args$newdata<-if(source=="new") nd else single; args$na.action<-na}
      raw <- capture(function() do.call(predict,args))
      # The penalized wrapper refuses survival; retain it and check the ordinary
      # prediction method on the same fitted state as in the preceding reference.
      if(type=="survival" && inherits(fit,"coxph.penal")) class(args$object) <- "coxph"
      expected <- capture(function() do.call(predict,args))
      if(name=="aft_intercept" && type=="terms") {
        expected <- capture(function() {
          frame <- if(source=="stored") model.frame(fit) else
            model.frame(delete.response(fit$terms),data=args$newdata,na.action=na)
          value <- matrix(numeric(),nrow=nrow(frame),ncol=0L,
                          dimnames=list(row.names(frame),NULL))
          if(se) list(fit=value,se.fit=value) else value
        })
      }
      encode_capture <- function(value) list(result=if(is.list(value$value)&&!is.null(value$value$error))
        value$value else encode(value$value),warnings=value$warnings)
      cases[[length(cases)+1L]] <- list(name=paste(name,action,type,se,length(p),source,na,sep="/"),
        model=name,fit_action=action,type=type,se_fit=se,p=if(is.null(p))NULL else I(p),
        source=source,na_action=na,raw=encode_capture(raw),expected=encode_capture(expected))
    }
  }
}
# Compact case records avoid thousands of one-value-per-line label copies.
header <- list(metadata=list(r=as.character(getRversion()),survival=as.character(packageVersion("survival")),
  reference="complete stock predictions and metadata; raw penalized survival refusal retained beside ordinary-method reference"),
  data=d,newdata=nd,levels=I(levels(d$cl)),row_names=I(row.names(d)),new_names=I(row.names(nd)),
  subset=I(subset),specs=specs)
serialize <- function(x) jsonlite::toJSON(x,auto_unbox=TRUE,digits=17,na="null",null="null",dataframe="columns")
first <- serialize(header)
lines <- c(substring(first,1L,nchar(first)-1L),',"cases":[',
           vapply(seq_along(cases),function(i) paste0(serialize(cases[[i]]),if(i<length(cases))"," else ""),""),"]}")
writeLines(lines,output,useBytes=TRUE)
cat(length(cases),"prediction row references written\n")
