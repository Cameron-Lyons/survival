.model_concordance_data <- function() {
  data.frame(time = c(1, 3, 2, 5, 4, 6, 2, 4, 1, 7, 5, 8),
             status = rep(1, 12), x = c(0, 1, 3, 2, 5, 4, 3, 0, 2, 1, 4, 5),
             g = rep(c("a", "b"), each = 6), id = rep(1:3, 4))
}

test_that("fitted Cox concordance retains strata and model clustering", {
  data <- .model_concordance_data()
  reference_concordance <- get("concordance.coxph", asNamespace("survival"))
  for (formula in list(Surv(time, status) ~ x + strata(g),
                       Surv(time, status) ~ x + cluster(id))) {
    fit <- coxph(formula, data)
    reference <- survival::coxph(formula, data)
    actual <- concordance(fit, influence = 3, ranks = TRUE)
    expected <- reference_concordance(reference, influence = 3, ranks = TRUE)
    # R carries model-frame row names on a single unstratified rank table.
    # The shared result exposes event rows without those source labels.
    rownames(expected$ranks) <- NULL
    actual$call <- expected$call
    expect_equal(actual, expected, tolerance = 1e-10)
  }
})

.model_concordance_censored_data <- function() {
  i <- seq_len(48)
  data.frame(time = 2 + (i*13) %% 31, start = (i %% 3)/4,
             status = as.integer(i %% 4 != 0), x = sin(i), z = cos(i/3),
             g = factor(rep(c("b", "a", "c"), 16), levels = c("c", "b", "a")),
             id = rep(c("1", "10", "2", "3"), 12), w = .5 + i %% 5,
             o = sin(i/5)/3)
}

.expect_model_concordance_result <- function(actual, expected) {
  expected$call <- actual$call
  if (length(actual$concordance) > 1L) {
    names(expected$concordance) <- names(actual$concordance)
    rownames(expected$count) <- rownames(actual$count)
    if (!is.null(expected$ranks)) {
      expected$ranks$fit <- names(actual$concordance)[match(expected$ranks$fit,
                                                           unique(expected$ranks$fit))]
    }
  }
  if (!is.null(expected$ranks)) rownames(expected$ranks) <- NULL
  expect_equal(actual, expected, tolerance = 1e-9)
}

test_that("Cox and AFT fitted concordance retain weights, strata and controls", {
  data <- .model_concordance_censored_data()
  for (family in c("coxph", "survreg")) {
    fitter <- get(family, asNamespace("survivalr"))
    reference_fitter <- get(family, asNamespace("survival"))
    reference_concordance <- get(paste0("concordance.", family), asNamespace("survival"))
    formulas <- list(Surv(time, status) ~ x + z + offset(o),
                     Surv(time, status) ~ x + strata(g) + offset(o),
                     Surv(time, status) ~ ridge(x, theta = 2) + strata(g))
    if (family == "coxph") {
      formulas <- c(formulas, list(Surv(start, time, status) ~ x + strata(g)))
    }
    for (formula in formulas) for (weighted in c(FALSE, TRUE)) {
      fit <- fitter(formula, data, weights = if (weighted) data$w else NULL, robust = FALSE)
      reference <- reference_fitter(formula, data, weights = if (weighted) data$w else NULL,
                                    robust = FALSE)
      counting <- grepl("start", deparse(formula)[[1L]])
      timewts <- if (counting) c("n", "S", "I") else c("n", "S", "S/G", "n/G2", "I")
      for (timewt in timewts) for (influence in 0:3) {
        .expect_model_concordance_result(
          concordance(fit, timewt = timewt, influence = influence, ymin = 3, ymax = 25),
          reference_concordance(reference, timewt = timewt, influence = influence,
                                ymin = 3, ymax = 25))
      }
      for (keep in list(FALSE, TRUE, 2L, 3L)) {
        # R's pooled-strata path calls colSums on an already pooled vector.
        expected <- reference_concordance(reference, keepstrata = TRUE)
        if (is.matrix(expected$count) &&
            (identical(keep, FALSE) || (is.numeric(keep) && nrow(expected$count) > keep))) {
          expected$count <- colSums(expected$count)
        }
        .expect_model_concordance_result(concordance(fit, keepstrata = keep),
                                         expected)
      }
    }
  }
})

test_that("explicit clusters replace fitted clusters in R ordering", {
  data <- .model_concordance_censored_data()
  fit <- coxph(Surv(time, status) ~ x + strata(g) + cluster(id), data)
  reference <- survival::coxph(Surv(time, status) ~ x + strata(g) + cluster(id), data)
  reference_concordance <- get("concordance.coxph", asNamespace("survival"))
  for (cluster in list(NULL, data$id, factor(data$id, levels = c("2", "1", "3", "10")),
                       rep(1:6, 8))) {
    for (influence in 0:3) {
      .expect_model_concordance_result(concordance(fit, cluster = cluster, influence = influence),
        reference_concordance(reference, cluster = cluster, influence = influence))
    }
  }
})

test_that("joint models use each response and stratum partition", {
  data <- .model_concordance_censored_data()
  for (family in c("coxph", "survreg")) {
    fitter <- get(family, asNamespace("survivalr"))
    reference_fitter <- get(family, asNamespace("survival"))
    reference_concordance <- get(paste0("concordance.", family), asNamespace("survival"))
    first <- fitter(Surv(time, status) ~ x + strata(g), data)
    second <- fitter(Surv(time, status) ~ z, data)
    ref_first <- reference_fitter(Surv(time, status) ~ x + strata(g), data)
    ref_second <- reference_fitter(Surv(time, status) ~ z, data)
    for (influence in 0:3) for (cluster in list(NULL, data$id)) {
      .expect_model_concordance_result(
        concordance(first, second, influence = influence, cluster = cluster),
        reference_concordance(ref_first, ref_second, influence = influence, cluster = cluster))
    }
    other <- fitter(Surv(35 - time, status) ~ x + strata(g), data)
    for (i in 1:2) expect_warning(joint <- concordance(first, other, influence = 1),
                                  "same response vector")
    one <- concordance(first, influence = 1)
    two <- concordance(other, influence = 1)
    expect_equal(unname(joint$concordance), c(one$concordance, two$concordance))
    expect_equal(joint$var, crossprod(cbind(one$dfbeta, two$dfbeta)))
    expect_equal(unname(joint$count), unname(rbind(colSums(one$count), colSums(two$count))))
  }
})

test_that("joint weighted clustered covariance applies weights once", {
  data <- .model_concordance_censored_data()
  first <- coxph(Surv(time, status) ~ x + strata(g) + cluster(id), data, weights = w)
  second <- coxph(Surv(time, status) ~ z + cluster(id), data, weights = w)
  one <- concordance(first, influence = 1)
  two <- concordance(second, influence = 1)
  joint <- concordance(first, second, influence = 1)
  expect_equal(joint$dfbeta, cbind(one$dfbeta, two$dfbeta))
  expect_equal(joint$var, crossprod(joint$dfbeta))
  expect_equal(diag(joint$var), c(one$var, two$var))
  unclustered <- coxph(Surv(time, status) ~ z, data, weights = w)
  expect_error(concordance(first, unclustered), "identical clustering")
  expect_error(concordance(unclustered, first), "identical clustering")
  expect_equal(concordance(first, unclustered, cluster = data$id)$var, joint$var)
})

test_that("stored fit rows and newdata omission stay aligned", {
  data <- .model_concordance_censored_data()
  formula <- Surv(time, status) ~ x + strata(g) + offset(o)
  for (family in c("coxph", "survreg")) {
    fitter <- get(family, asNamespace("survivalr"))
    reference_fitter <- get(family, asNamespace("survival"))
    reference_concordance <- get(paste0("concordance.", family), asNamespace("survival"))
    train <- data
    train$x[c(3, 19)] <- NA
    fit <- fitter(formula, train, na.action = na.exclude, weights = train$w)
    reference <- reference_fitter(formula, train, na.action = na.exclude, weights = train$w)
    .expect_model_concordance_result(concordance(fit, influence = 3),
                                     reference_concordance(reference, influence = 3))
    new <- data[c(30:48, 1:20), ]
    new$time[2] <- NA; new$x[4] <- NA; new$o[7] <- NA; new$g[9] <- NA
    kept <- new[complete.cases(new), ]
    actual <- concordance(fit, newdata = new, cluster = kept$id, influence = 3)
    # R's survreg new-data lp path drops formula offsets. Include the offset
    # explicitly in the reference scores, retaining the existing correction.
    score <- predict(reference, kept, type = "lp")
    if (family == "survreg") score <- score + kept$o
    expected <- survival::concordancefit(survival::Surv(kept$time, kept$status), score,
      strata = kept$g, cluster = kept$id, influence = 3, reverse = family == "coxph")
    class(expected) <- "concordance"
    .expect_model_concordance_result(actual, expected)
    expect_equal(actual$n, nrow(kept))
    expect_error(concordance(fit, newdata = new, cluster = new$id), "cluster")
    expect_equal(concordance(fit, newdata = NULL)$concordance, concordance(fit)$concordance)
  }
})

test_that("unstratified ranks and joint influence shapes match fitted methods", {
  data <- .model_concordance_censored_data()
  first <- coxph(Surv(time, status) ~ x, data)
  second <- coxph(Surv(time, status) ~ z, data)
  ref_first <- survival::coxph(Surv(time, status) ~ x, data)
  ref_second <- survival::coxph(Surv(time, status) ~ z, data)
  reference_concordance <- get("concordance.coxph", asNamespace("survival"))
  for (influence in 0:3) {
    .expect_model_concordance_result(concordance(first, influence = influence, ranks = TRUE),
      reference_concordance(ref_first, influence = influence, ranks = TRUE))
    .expect_model_concordance_result(concordance(first, second, influence = influence, ranks = TRUE),
      reference_concordance(ref_first, ref_second, influence = influence, ranks = TRUE))
  }
})

test_that("stratified event ranks equal independent per-stratum calculations", {
  data <- .model_concordance_censored_data()
  fit <- coxph(Surv(time, status) ~ x + strata(g), data)
  reference <- survival::coxph(Surv(time, status) ~ x + strata(g), data)
  for (timewt in c("n", "S", "S/G", "n/G2", "I")) {
    expected <- lapply(levels(data$g), function(level) {
      keep <- data$g == level
      survival::concordancefit(reference$y[keep, ], reference$linear.predictors[keep],
        timewt = timewt, ranks = TRUE, reverse = TRUE, ymax = 25)$ranks
    })
    expected <- do.call(rbind, expected)
    rownames(expected) <- NULL
    expect_equal(concordance(fit, timewt = timewt, ranks = TRUE, ymax = 25)$ranks,
                  expected, tolerance = 1e-10)
  }
})

test_that("fitted-model scoring uses the shared computation without R numerical calls", {
  data <- .model_concordance_censored_data()
  first <- coxph(Surv(time, status) ~ x + strata(g) + cluster(id), data)
  second <- coxph(Surv(time, status) ~ z + cluster(id), data)
  expected <- concordance(first, second, influence = 2)
  ns <- asNamespace("survival")
  original <- get("concordancefit", ns)
  unlockBinding("concordancefit", ns)
  assign("concordancefit", function(...) stop("reference called"), ns)
  lockBinding("concordancefit", ns)
  on.exit({
    unlockBinding("concordancefit", ns); assign("concordancefit", original, ns)
    lockBinding("concordancefit", ns)
  }, add = TRUE)
  actual <- concordance(first, second, influence = 2)
  actual$call <- expected$call
  expect_equal(actual, expected)
})

test_that("newdata requires every fitted stratum variable", {
  data <- .model_concordance_censored_data()
  fit <- coxph(Surv(time, status) ~ x + strata(g), data)
  expect_error(concordance(fit, newdata = data[, names(data) != "g"]), "strata")
})

test_that("fitted concordance honors timefix and rejects time-transform fits", {
  data <- data.frame(time = c(1, 1 + 1e-12, 2, 3, 4, 5), status = 1,
                     x = c(2, 1, 3, 5, 4, 6))
  fit <- coxph(Surv(time, status) ~ x, data, timefix = FALSE)
  for (timefix in c(FALSE, TRUE)) {
    expected <- survival::concordancefit(survival::Surv(data$time, data$status),
      as.numeric(predict(fit, type = "lp")), reverse = TRUE, timefix = timefix)
    class(expected) <- "concordance"
    .expect_model_concordance_result(concordance(fit, timefix = timefix), expected)
  }
  fit <- coxph(Surv(time, status) ~ tt(x), data,
                tt = function(x, time, riskset, weights) x * log(time + 1))
  expect_error(concordance(fit), "tt terms")
})
