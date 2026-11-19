test_that("nonbinary constant aliases retain stock identified AFT fits", {
  i <- seq_len(80L)
  d <- data.frame(time = 10 + (i * 37) %% 97 + i / 7,
    status = as.integer((i * 17 + 3) %% 11 > 2),
    w = rep(c(.7, 1, 1.4, 1.1), length.out = length(i)),
    id = (i - 1L) %/% 3L, o = (i %% 5 - 2) / 20)
  d$g <- ordered(c("z", "m", "a")[i %% 3L + 1L],
    levels = c("z", "m", "a", "unused"))
  contrasts(d$g) <- contr.helmert(4L)
  h <- contr.helmert(4L)[as.integer(d$g), , drop = FALSE]
  d$h1 <- h[, 1L]
  d$h2 <- h[, 2L]
  for (dist in c("weibull", "lognormal", "gaussian", "exponential"))
    for (robust in c(FALSE, TRUE)) {
    arguments <- list(data = d, dist = dist, weights = d$w,
      x = TRUE, model = TRUE, score = TRUE)
    if (robust) arguments$cluster <- d$id
    reduced <- do.call(survival::survreg,
      c(list(formula = Surv(time, status) ~ h1 + h2 + offset(o)), arguments))
    initial <- c(coef(reduced), 0, if (dist != "exponential") log(reduced$scale))
    form <- Surv(time, status) ~ identity(g) + offset(o)
    expected <- do.call(survival::survreg, c(list(formula = form, init = initial), arguments))
    actual <- do.call(survivalr::survreg, c(list(formula = form), arguments))
    expect_length(coef(actual), 4L)
    expect_true(is.na(coef(actual)[[4L]]))
    expect_identical(names(coef(actual)), names(coef(expected)))
    expect_equal(coef(actual), coef(expected), tolerance = 3e-7)
    expect_equal(vcov(actual), vcov(expected), tolerance = 3e-7)
    expect_equal(actual$scale, expected$scale, tolerance = 3e-7)
    expect_equal(actual$loglik, expected$loglik, tolerance = 3e-7)
    expect_equal(unname(predict(actual, type = "lp")),
      unname(reduced$linear.predictors), tolerance = 3e-7)
    expect_equal(predict(actual, type = "lp", se.fit = TRUE),
      predict(expected, type = "lp", se.fit = TRUE), tolerance = 3e-7)
    expect_equal(predict(actual, type = "quantile", p = c(.25, .75), se.fit = TRUE),
      predict(expected, type = "quantile", p = c(.25, .75), se.fit = TRUE), tolerance = 3e-7)
    if (robust) {
      naive <- do.call(rbind, lapply(actual$naive_var, unlist))
      expect_equal(unname(naive), unname(expected$naive.var), tolerance = 3e-7)
    } else expect_null(actual$naive_var)
    expect_true(all(vcov(actual)[4L, ] == 0))
    expect_true(all(is.na(suppressWarnings(predict(actual, d[1:3, ], type = "lp")))))
  }
})

test_that("aliased predictions preserve independent numeric NaNs", {
  d <- data.frame(
    time = c(2, 4, 3, 5, 6, 1, 4, 2, 8, 3),
    status = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    x = c(.4, .2, .6, .1, .5, .3, .8, 1.2, -.2, .7),
    z = c(1, -.3, .4, .1, 1.4, 1.2, -.8, .3, 1.4, -.3)
  )
  d$dup <- d$x
  formula <- Surv(time, status) ~ x + dup + z
  expected <- survival::survreg(formula, d)
  actual <- survivalr::survreg(formula, d)
  nd <- data.frame(x = c(1, 2, 3, NaN, NA, 2, NaN, NA),
    dup = c(1, 1, NaN, 1, 1, NA, NaN, 1), z = c(0, NaN, 0, 2, NaN, NaN, NaN, 0))
  design <- as.matrix(nd)
  has_nan <- rowSums(is.nan(design)) > 0
  has_na <- rowSums(is.na(design) & !is.nan(design)) > 0
  expect_identical(names(coef(expected))[is.na(coef(expected))], "dup")
  for (type in c("terms", "lp", "response", "quantile", "uquantile")) {
    stock <- predict(expected, nd, type = type, p = .4, se.fit = TRUE)
    bridged <- predict(actual, nd, type = type, p = .4, se.fit = TRUE)
    for (part in names(stock)) {
      # R's ?NA allows either missing kind when arithmetic combines NA and NaN.
      mixed <- if (type == "terms") {
        cbind(x = FALSE, dup = is.nan(nd$dup), z = FALSE)
      } else if (part == "se.fit" && type %in% c("lp", "uquantile")) {
        # These errors use the design and covariance independently of the alias.
        has_nan & has_na
      } else {
        # The fitted location includes the aliased coefficient's NA.
        has_nan
      }
      label <- paste(type, part)
      expect_equal(bridged[[part]], stock[[part]], tolerance = 3e-7, info = label)
      expect_identical(is.na(bridged[[part]]), is.na(stock[[part]]), info = label)
      expect_identical(is.nan(bridged[[part]])[!mixed], is.nan(stock[[part]])[!mixed],
        info = label)
    }
  }
})
