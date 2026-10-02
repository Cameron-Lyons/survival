test_that("entirely omitted AFT predictions preserve stock dimensions and absent labels", {
  d <- survival::ovarian
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
  nd <- d[3:6, ]
  row.names(nd) <- c("四", "two / 2", "six:6", "one")
  nd$age[] <- NA_real_
  for (model in c("aft", "aft_strata", "aft_fixed")) {
    rhs <- if (model == "aft_strata") "age + rx + strata(cl)" else "age + rx"
    form <- as.formula(paste("Surv(futime, fustat) ~", rhs))
    scale <- if (model == "aft_fixed") 1 else 0
    fit <- survreg(form, d, scale = scale)
    stock <- survival::survreg(form, d, scale = scale)
    for (type in c("response", "link", "quantile", "uquantile")) {
      probabilities <- if (type %in% c("quantile", "uquantile")) list(.5, c(.1,.5,.9)) else list(.5)
      for (p in probabilities) for (action in c("na.omit", "na.exclude")) for (se in c(FALSE, TRUE)) {
        args <- list(newdata = nd, type = type, p = p, na.action = action, se.fit = se)
        actual <- do.call(predict, c(list(fit), args))
        expected <- do.call(predict, c(list(stock), args))
        info <- paste(model, type, length(p), action, se)
        if (!se) {actual <- list(fit = actual); expected <- list(fit = expected)}
        expect_named(actual, names(expected), info = info)
        for (name in names(expected)) {
          expect_identical(dim(actual[[name]]), dim(expected[[name]]), info = info)
          expect_identical(names(actual[[name]]), names(expected[[name]]), info = info)
          expect_identical(dimnames(actual[[name]]), dimnames(expected[[name]]), info = info)
          expect_identical(is.na(actual[[name]]), is.na(expected[[name]]), info = info)
        }
      }
    }
  }
})
