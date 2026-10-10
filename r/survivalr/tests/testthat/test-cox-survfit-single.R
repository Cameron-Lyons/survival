testthat::test_that("nondimensional ordinary Cox selectors agree with stock", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
d <- data.frame(time=1:12, status=rep(c(1,1,0),4), x=rep(c(0,1,2),4),
  z=c(0,1,.5,-1,2,.7,1,0,-1,.5,2,1), g=factor(rep(c("a","b"),each=6)))
nd <- data.frame(x=c(-1,0,2),z=c(1,0,-1),tag=factor(c("b","a","b"),levels=c("a","unused","b")),
  row.names=c("two","one","three"))
pair <- function(grouped=FALSE) {
  form <- if(grouped) Surv(time,status)~x+z+strata(g) else Surv(time,status)~x+z
  list(actual=survivalr::coxph(form,d,init=c(.25,-.125),iter.max=0),
    stock=survival::coxph(form,d,init=c(.25,-.125),iter.max=0))
}
fit <- pair(); fs <- pair(TRUE)
sources <- list(
  default=list(actual=survivalr::survfit(fit$actual),stock=survival::survfit(fit$stock)),
  plain_one=list(actual=survivalr::survfit(fit$actual,newdata=nd[1,,drop=FALSE]),
    stock=survival::survfit(fit$stock,newdata=nd[1,,drop=FALSE])),
  selected_one=list(actual=survivalr::survfit(fs$actual,newdata=nd)[1,1],
    stock=survival::survfit(fs$stock,newdata=nd)[1,1]),
  aggregated=list(actual=aggregate(survivalr::survfit(fit$actual,newdata=nd)),
    stock=aggregate(survival::survfit(fit$stock,newdata=nd))),
  starts_late=list(actual=survivalr::survfit(fit$actual,newdata=nd[1,,drop=FALSE],start.time=4.5),
    stock=survival::survfit(fit$stock,newdata=nd[1,,drop=FALSE],start.time=4.5)))
selectors <- list(missing=quote(x[]),null=quote(x[NULL]),one=quote(x[1]),
  repeated=quote(x[c(1,1)]),empty=quote(x[integer()]),zero=quote(x[0]),
  negative=quote(x[-1]),true=quote(x[TRUE]),false=quote(x[FALSE]),
  repeat_true=quote(x[c(TRUE,TRUE)]),one_string=quote(x["1"]),
  bad_name=quote(x["two"]),fraction=quote(x[1.5]),missing_value=quote(x[NA]),
  matrix_ones=quote(x[matrix(1,2,2)]),one_list=quote(x[list(1)]),
  extra_dimension=quote(x[1,1]),drop_false=quote(x[1,drop=FALSE]),drop_null=quote(x[1,drop=NULL]))
capture <- function(expr) tryCatch(list(value=force(expr)),error=function(e)list(error=conditionMessage(e)))
  for (source in names(sources)) for(selector in names(selectors)) {
    x <- sources[[source]]$actual; actual <- capture(eval(selectors[[selector]]))
    x <- sources[[source]]$stock; expected <- capture(eval(selectors[[selector]]))
    info <- paste(source,selector)
    if (!is.null(expected$error)) testthat::expect_identical(actual$error,expected$error,info=info)
    else {
      testthat::expect_null(actual$error,info=info)
      for (object in list(actual$value,unserialize(serialize(actual$value,NULL)))) {
        fields <- as.list(object)
        testthat::expect_identical(dim(object),dim(expected$value),info=info)
        names <- c("time","n","n.risk","n.event","n.censor","surv","cumhaz","std.err",
          "std.chaz","lower","upper","newdata","start.time","strata")
        # Stock aggregate retains stale cumulative-hazard uncertainty columns;
        # the existing Python aggregate clears them deliberately.
        if (source == "aggregated") names <- setdiff(names,"std.chaz")
        for(name in names)
          testthat::expect_equal(fields[[name]],expected$value[[name]],tolerance=2e-12,
            info=paste(info,name))
        testthat::expect_equal(quantile(object,probs=.5),quantile(expected$value,probs=.5),
          tolerance=2e-12,info=info)
        for (timed in c(FALSE,TRUE)) {
          a <- if(timed) summary(object,times=c(0,4,8,12),extend=TRUE) else summary(object,censored=TRUE)
          e <- if(timed) summary(expected$value,times=c(0,4,8,12),extend=TRUE,data.frame=TRUE)
            else summary(expected$value,censored=TRUE,data.frame=TRUE)
          for(name in intersect(names(a),names(e)))
            testthat::expect_equal(a[[name]],e[[name]],tolerance=2e-12,info=paste(info,name))
        }
        again <- object[c(1,1)]
        testthat::expect_null(again$newdata,info=info)
      }
    }
  }
})

testthat::test_that("start.time is dropped for selected Cox curves except exact linear/full no-ops", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
d <- data.frame(time=1:12,status=rep(c(1,1,0),4),x=rep(c(0,1,2),4),
  z=c(0,1,.5,-1,2,.7,1,0,-1,.5,2,1),g=factor(rep(c("a","b"),each=6)))
nd <- data.frame(x=c(-1,0,2),z=c(1,0,-1))
form <- Surv(time,status)~x+z+strata(g)
a <- survivalr::survfit(survivalr::coxph(form,d,init=c(.25,-.125),iter.max=0),newdata=nd,start.time=4.5)
e <- survival::survfit(survival::coxph(form,d,init=c(.25,-.125),iter.max=0),newdata=nd,start.time=4.5)
selections <- list(missing=function(x)x[],null=function(x)x[NULL],both_missing=function(x)x[,],
  full_margins=function(x)x[1:2,1:3,drop=FALSE],reorder_margins=function(x)x[2:1,3:1,drop=FALSE],
  repeat_margins=function(x)x[c(2,1,2),c(3,1),drop=FALSE],
  single_margin=function(x)x[1,,drop=FALSE],data_margin=function(x)x[,1,drop=FALSE],
  single_both=function(x)x[1,1],linear_full=function(x)x[1:6],
  linear_reorder=function(x)x[6:1],linear_repeat=function(x)x[c(4,1,4)],
  linear_single=function(x)x[2])
  for(name in names(selections)) {
    actual <- selections[[name]](a);expected <- selections[[name]](e)
    testthat::expect_equal(actual$start.time,expected$start.time,info=name)
    testthat::expect_equal(dim(actual),dim(expected),info=name)
    testthat::expect_equal(actual$surv,expected$surv,tolerance=2e-12,info=name)
    testthat::expect_equal(actual$newdata,expected$newdata,info=name)
    testthat::expect_equal(survivalr::survfit0(actual)$time,survival::survfit0(expected)$time,info=name)
    testthat::expect_equal(quantile(actual,.5),quantile(expected,.5),tolerance=2e-12,info=name)
    if (name %in% c("missing","null","both_missing","linear_full")) {
      testthat::expect_equal(actual$start.time,a$start.time,info=name)
      testthat::expect_identical(expected,e,info=name)
    } else testthat::expect_null(actual$start.time,info=name)
  }
})
