.grouped_cox_compare <- function(actual, expected) {
  if (is.null(dim(actual))) actual <- matrix(actual, ncol = 1L, dimnames = list(names(actual), NULL))
  expect_identical(dim(actual), dim(expected))
  expect_identical(rownames(actual), rownames(expected))
  expect_identical(is.nan(actual), is.nan(expected), ignore_attr = TRUE)
  expect_equal(unname(actual), unname(expected), ignore_attr = TRUE, tolerance = 2e-7)
}

test_that("one, two and three grouped rows retain selected term matrix shapes", {
  d <- survival::ovarian
  actual <- coxph(Surv(futime,fustat) ~ age + rx, d)
  expected <- survival::coxph(Surv(futime,fustat) ~ age + rx, d)
  reference <- predict(expected, type = "terms", se.fit = TRUE)
  for (n in c(1L, 2L, 3L)) {
    group <- seq_len(nrow(d)) %% n
    for (selection in list(integer(), 1L, c(2L, 1L), c(2L, 1L, 2L))) {
      fitted <- rowsum(reference$fit[, selection, drop = FALSE], group)
      errors <- sqrt(rowsum(reference$se.fit[, selection, drop = FALSE]^2, group))
      for (se in c(FALSE, TRUE)) {
        result <- predict(actual, type = "terms", terms = selection, se.fit = se, collapse = group)
        if (se) {
          .grouped_cox_compare(result$fit, fitted)
          .grouped_cox_compare(result$se.fit, errors)
          expect_identical(colnames(result$fit), colnames(fitted))
          expect_identical(colnames(result$se.fit), colnames(errors))
        } else {
          .grouped_cox_compare(result, fitted)
          expect_identical(colnames(result), colnames(fitted))
        }
      }
    }
  }
})

test_that("native grouped predictions restore groups containing only excluded rows", {
  d <- survival::ovarian
  d$x <- (d$age - 60)/10
  d$cl <- factor(rep(c("a", "b", "c"), length.out = nrow(d)), levels = c("c", "b", "a"))
  group <- factor(replace(rep(c("kept-a", "kept-c"), length.out = nrow(d)), c(2,9), "only-omitted"),
                  levels = c("kept-c", "only-omitted", "kept-a", "unused"))
  specs <- list(plain = "x + rx", ridge = "x + ridge(rx, theta = 2)",
                multi = "ridge(x, rx, theta = 2)",
                sparse = "x + frailty(cl, sparse = TRUE, theta = .4)",
                sparse_only = "frailty(cl, sparse = TRUE, theta = .4)")
  for (name in names(specs)) {
    data <- d
    if (name == "sparse_only") data$cl[c(2,9)] <- NA else data$x[c(2,9)] <- NA
    options <- list(formula = as.formula(paste("Surv(futime,fustat) ~", specs[[name]])),
                    data = data, robust = FALSE, x = TRUE, na.action = na.exclude)
    actual <- do.call(coxph, options)
    expected <- do.call(survival::coxph, options)
    dense <- expected
    if (name == "sparse") {
      dense$x <- expected$x[, "x", drop = FALSE]
      dense$terms <- terms(Surv(futime,fustat) ~ x)
      dense$assign <- list(x = 1L)
      dense$pterms <- c(x = 0)
      class(dense) <- "coxph"
    }
    types <- if (name == "sparse_only") c("lp", "risk", "terms") else
      c("lp", "risk", "terms", "expected", "survival")
    for (type in types) for (se in c(FALSE, TRUE)) {
      source <- if (name == "sparse" && type == "terms") dense else expected
      value <- predict(source, type = if (type == "survival") "expected" else type, se.fit = se)
      if (name == "sparse" && type == "terms") {
        index <- as.integer(factor(expected$x[,2L]))
        frail <- naresid(expected$na.action, expected$frail[index])
        fvar <- naresid(expected$na.action, sqrt(expected$fvar[index]))
        if (se) { value$fit <- cbind(value$fit, frail); value$se.fit <- cbind(value$se.fit, fvar) }
        else value <- cbind(value, frail)
      }
      if (type == "survival") {
        if (se) { value$fit <- exp(-value$fit); value$se.fit <- value$se.fit * value$fit }
        else value <- exp(-value)
      }
      reference <- if (se) list(fit = rowsum(value$fit, group), se.fit = sqrt(rowsum(value$se.fit^2, group)))
                   else rowsum(value, group)
      result <- predict(actual, type = type, se.fit = se, collapse = group)
      if (se) {
        .grouped_cox_compare(result$fit, reference$fit)
        .grouped_cox_compare(result$se.fit, reference$se.fit)
        expect_true(all(is.na(result$fit[2L, drop = TRUE])))
      } else .grouped_cox_compare(result, reference)
    }
    if (!name %in% c("sparse", "sparse_only")) {
      for (selection in list(integer(), if (name == "multi") c(1L,1L) else c(2L,1L,2L))) {
        reference <- predict(expected, type = "terms", se.fit = TRUE)
        pick <- function(x) {
          if (!is.matrix(x)) x <- matrix(x, ncol = 1L)
          x[, selection, drop = FALSE]
        }
        reference <- list(fit = rowsum(pick(reference$fit), group),
                          se.fit = sqrt(rowsum(pick(reference$se.fit)^2, group)))
        result <- predict(actual, type = "terms", terms = selection, se.fit = TRUE, collapse = group)
        .grouped_cox_compare(result$fit, reference$fit)
        .grouped_cox_compare(result$se.fit, reference$se.fit)
      }
    }
  }
})

test_that("new-data grouping pads omitted-only groups and can omit all rows", {
  d <- survival::ovarian; d$x <- (d$age - 60)/10
  actual <- coxph(Surv(futime,fustat) ~ x + rx, d)
  expected <- survival::coxph(Surv(futime,fustat) ~ x + rx, d)
  nd <- d[seq_len(8), ]; nd$x[c(2,7)] <- NA
  group <- factor(c("kept-a", "only-omitted", NA, "kept-c", "kept-a", "kept-c", "only-omitted", "kept-a"),
                  levels = c("kept-c", "only-omitted", "kept-a", "unused"))
  for (action in c("na.pass", "na.omit", "na.exclude")) for (type in c("lp", "risk", "terms", "expected", "survival")) {
    used <- nd
    if (action != "na.pass") used$x[is.na(group)] <- NA
    value <- predict(expected, used, type = type, se.fit = TRUE, na.action = action)
    labels <- if (action == "na.omit") group[complete.cases(used[, c("x", "rx")])] else group
    reference <- suppressWarnings(list(fit = rowsum(value$fit, labels),
                                        se.fit = sqrt(rowsum(value$se.fit^2, labels))))
    result <- suppressWarnings(predict(actual, nd, type = type, se.fit = TRUE, collapse = group,
                                       na.action = action))
    .grouped_cox_compare(result$fit, reference$fit)
    .grouped_cox_compare(result$se.fit, reference$se.fit)
  }
  result <- predict(actual, nd, type = "terms", se.fit = TRUE, collapse = rep(NA_character_,8),
                    na.action = "na.omit")
  expect_identical(dim(result$fit), c(0L,2L))
  expect_identical(dim(result$se.fit), c(0L,2L))
})
