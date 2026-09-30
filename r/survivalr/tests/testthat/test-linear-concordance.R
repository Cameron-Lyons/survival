.linear_concordance_data <- function() {
  i <- seq_len(36)
  data.frame(y = 3 + sin(i) + i/8, x = cos(i/2), z = (i*7) %% 17,
             a = factor(rep(c("c", "a", "b"), 12), levels = c("c", "b", "a")),
             o = sin(i/3), w = 1 + i %% 4, id = i %% 7)
}

.linear_concordance_reference <- get("concordance.lm", asNamespace("survival"))

.expect_linear_concordance <- function(fit, ..., tolerance = 1e-10) {
  expected <- .linear_concordance_reference(fit, ...)
  actual <- concordance(fit, ...)
  actual$call <- expected$call
  expect_equal(actual, expected, tolerance = tolerance)
  invisible(actual)
}

test_that("external linear models share concordance for training and new rows", {
  data <- .linear_concordance_data()
  for (formula in list(y ~ x + z, y ~ 0 + a + x, y ~ a*x + z,
                       log(y) ~ x + I(z*z), y ~ x + offset(o), y ~ splines::ns(x, 3))) {
    for (weighted in c(FALSE, TRUE)) {
      fit <- lm(formula, data, weights = if (weighted) w else NULL)
      for (influence in 0:3) {
        .expect_linear_concordance(fit, influence = influence, ranks = TRUE)
        .expect_linear_concordance(fit, influence = influence, cluster = data$id)
        .expect_linear_concordance(fit, newdata = data[c(2, 7, 3, 15, 23, 27), ],
                                   influence = influence, ranks = TRUE)
      }
    }
  }
  fit <- lm(y ~ x, data, model = FALSE, y = FALSE, na.action = na.exclude)
  .expect_linear_concordance(fit, influence = 3)
  .expect_linear_concordance(lm(y ~ 1, data), influence = 3)
  .expect_linear_concordance(lm(y ~ x, data, offset = o), influence = 3)
  .expect_linear_concordance(lm(y ~ x, data), newdata = data, ymin = 4, ymax = 6,
                             ranks = TRUE, influence = 3)
})

test_that("GLM concordance retains training linear predictors and prior weights", {
  data <- .linear_concordance_data()
  for (family in list(poisson(), gaussian("log"), Gamma("log"))) {
    data$y <- 1 + seq_len(nrow(data)) %% 8
    fit <- glm(y ~ a + x + offset(o), data, family = family, weights = w)
    .expect_linear_concordance(fit, influence = 3, ranks = TRUE)
    .expect_linear_concordance(fit, newdata = data[c(1, 4, 7, 21, 22), ], influence = 3)
  }
  data$y <- as.numeric(seq_len(nrow(data)) %% 5 < 2)
  fit <- glm(y ~ x + z, data, family = binomial("probit"))
  .expect_linear_concordance(fit, influence = 3, ranks = TRUE)
})

test_that("multiple linear fits keep joint influences and correct weighted covariance", {
  data <- .linear_concordance_data()
  first <- lm(y ~ x, data)
  second <- lm(y ~ x + z, data)
  for (influence in 0:3) .expect_linear_concordance(first, second,
                                                  influence = influence, ranks = TRUE)
  first <- lm(y ~ x, data, weights = w)
  second <- lm(y ~ x + z, data, weights = w)
  for (cluster in list(NULL, data$id)) {
    one <- concordance(first, cluster = cluster, influence = 1)
    two <- concordance(second, cluster = cluster, influence = 1)
    joint <- concordance(first, second, cluster = cluster, influence = 1)
    expected <- cbind(one$dfbeta, two$dfbeta)
    expect_equal(unname(joint$dfbeta), unname(expected))
    expect_equal(unname(joint$var), unname(crossprod(expected)))
    expect_equal(unname(diag(joint$var)), c(one$var, two$var))
    expect_named(joint$concordance, c("first", "second"))
    expect_equal(joint$count, rbind(first = one$count, second = two$count))
  }
})

test_that("newdata omission aligns response and predictions", {
  data <- .linear_concordance_data()
  fit <- lm(y ~ x + offset(o), data, weights = w)
  new <- data[1:9, ]
  new$y[2] <- NA; new$x[4] <- NA; new$o[7] <- NA
  kept <- setdiff(seq_len(nrow(new)), c(2, 4, 7))
  expected <- survival::concordancefit(survival::Surv(new$y[kept]),
    predict(fit, new[kept, ]), cluster = new$id[kept], influence = 3)
  class(expected) <- "concordance"
  actual <- concordance(fit, newdata = new, cluster = new$id[kept], influence = 3)
  actual$call <- NULL
  expect_equal(actual, expected, tolerance = 1e-10)
  expect_error(concordance(fit, newdata = new, cluster = new$id), "cluster")
  bad <- lm(I(y > 5) ~ x, data)
  expect_error(concordance(bad), "numeric vector")
  expect_error(concordance(fit, 1), "appropriate fit")
  expect_error(concordance(fit, lm(y ~ x, data)), "weight vector")
})

test_that("external-model concordance does not call reference numerical routines", {
  data <- .linear_concordance_data()
  fit <- lm(y ~ x + z, data)
  expected <- concordance(fit, influence = 3)
  ns <- asNamespace("survival")
  original <- get("concordancefit", ns)
  unlockBinding("concordancefit", ns)
  assign("concordancefit", function(...) stop("reference called"), ns)
  lockBinding("concordancefit", ns)
  on.exit({
    unlockBinding("concordancefit", ns); assign("concordancefit", original, ns)
    lockBinding("concordancefit", ns)
  }, add = TRUE)
  actual <- concordance(fit, influence = 3)
  actual$call <- expected$call
  expect_equal(actual, expected)
})

test_that("numeric training responses keep near ties and newdata honors timefix", {
  data <- data.frame(y = c(1, 1 + 1e-12, 2, 3), x = c(2, 1, 3, 4))
  fit <- lm(y ~ x, data)
  for (timefix in c(TRUE, FALSE)) .expect_linear_concordance(fit, timefix = timefix)
  .expect_linear_concordance(fit, newdata = data)
  expected <- survival::concordancefit(survival::Surv(data$y), predict(fit), timefix = FALSE)
  actual <- concordance(fit, newdata = data, timefix = FALSE)
  expect_equal(actual$count, expected$count)
  expect_equal(unname(actual$count), c(5, 1, 0, 0, 0))
  for (cluster in list(c("1", "10", "2", "1"), factor(c("z", "a", "b", "a"), levels = c("z", "b", "a")))) {
    .expect_linear_concordance(fit, cluster = cluster, influence = 1)
  }
})

test_that("direct concordance accepts numeric and ordered responses with bare controls", {
  y <- c(1, 1 + 1e-12, 2, 3)
  x <- c(2, 1, 3, 4)
  responses <- list(y, ordered(c("a", "a", "b", "c")), factor(c("b", "a", "b", "a")),
                    survival::Surv(y), survival::Surv(rep(0, 4), y, c(1, 1, 1, 0)))
  for (response in responses) for (std in c(TRUE, FALSE)) for (timefix in c(TRUE, FALSE)) {
    expected <- survival::concordancefit(response, x, std.err = std, timefix = timefix,
                                        influence = 3, ranks = TRUE)
    actual <- concordancefit(response, x, std.err = std, timefix = timefix,
                             influence = 3, ranks = TRUE)
    expect_equal(actual, expected, tolerance = 1e-10)
  }
  expect_error(concordancefit(y > 1, x), "numeric vector")
  expect_error(concordancefit(factor(c("a", "b", "c", "a")), x), "orderable factor")
  expect_null(concordancefit(c(1, NA, 2, 3), x))
})

test_that("joint response diagnostics use R conditions on every call", {
  data <- .linear_concordance_data()
  fit <- lm(y ~ x, data)
  other <- lm(I(y + 1) ~ x, data)
  for (i in 1:2) expect_warning(concordance(fit, other), "same response vector")
  expect_error(concordance(fit, lm(y ~ x, data[-1, ])), "same sample size")
})
