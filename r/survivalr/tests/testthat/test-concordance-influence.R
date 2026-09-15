.concordance_bridge_data <- function() {
  data.frame(
    start = c(0, 0, 0, 1, 0, 2),
    time = c(1, 1, 1, 2, 2, 3),
    status = c(1, 0, 1, 1, 0, 1),
    score = c(3, 3, 2, 1, 2, 0),
    score2 = c(3, 2, 2, 1, 3, 0),
    weight = c(2, 1, 1.25, 1.5, 3, 0.5),
    cluster = c(2, 2, 1, 1, 3, 3)
  )
}

test_that("concordancefit keeps native R influence scales with tied case weights", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_bridge_data()

  for (counting in c(FALSE, TRUE)) {
    y <- if (counting) {
      Surv(data$start, data$time, data$status)
    } else {
      Surv(data$time, data$status)
    }
    reference_y <- if (counting) {
      survival::Surv(data$start, data$time, data$status)
    } else {
      survival::Surv(data$time, data$status)
    }
    timeweights <- if (counting) c("n", "S", "I") else c("n", "S", "S/G", "n/G2", "I")
    for (timewt in timeweights) {
      for (mode in 0:3) {
        actual <- concordancefit(
          y, data$score, weights = data$weight, timewt = timewt,
          influence = mode, timefix = FALSE
        )
        expected <- survival::concordancefit(
          reference_y, data$score, weights = data$weight, timewt = timewt,
          influence = mode, timefix = FALSE
        )
        for (field in c("concordance", "count", "var", "cvar", "dfbeta", "influence")) {
          expect_equal(actual[[field]], expected[[field]], tolerance = 1e-12)
        }
        expect_identical(is.null(actual$dfbeta), !(mode %in% c(1L, 3L)))
        expect_identical(is.null(actual$influence), !(mode %in% c(2L, 3L)))
      }
    }
  }
})

test_that("concordancefit retains correlated score covariance in every influence mode", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_bridge_data()
  scores <- as.matrix(data[c("score", "score2")])

  for (counting in c(FALSE, TRUE)) {
    y <- if (counting) Surv(data$start, data$time, data$status) else Surv(data$time, data$status)
    reference_y <- if (counting) {
      survival::Surv(data$start, data$time, data$status)
    } else {
      survival::Surv(data$time, data$status)
    }
    for (cluster in list(NULL, data$cluster)) {
      for (mode in 0:3) {
        actual <- concordancefit(
          y, scores, weights = data$weight, cluster = cluster,
          influence = mode, timefix = FALSE
        )
        expected <- survival::concordancefit(
          reference_y, scores, weights = data$weight, cluster = cluster,
          influence = mode, timefix = FALSE
        )
        expect_true(all(diag(expected$var) > 0))
        expect_gt(abs(expected$var[1L, 2L]), 1e-6)
        for (field in c("concordance", "count", "var", "dfbeta", "influence")) {
          expect_equal(actual[[field]], expected[[field]], tolerance = 1e-12)
        }
        expect_identical(is.null(actual$dfbeta), !(mode %in% c(1L, 3L)))
        expect_identical(is.null(actual$influence), !(mode %in% c(2L, 3L)))
      }
    }
  }
})

test_that("formula covariance pools strata and clusters without exposing hidden diagnostics", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_bridge_data()
  data <- rbind(data, transform(data, score = score2, score2 = score))
  data$stratum <- rep(c("a", "b"), each = 6L)
  data$cluster <- c(1, 1, 2, 2, 3, 3, 1, 2, 1, 3, 2, 3)

  for (clustered in c(FALSE, TRUE)) {
    for (mode in 0:3) {
      actual <- concordance(
        Surv(time, status) ~ score + score2 + strata(stratum),
        data = data, weights = weight,
        cluster = if (clustered) data$cluster else NULL,
        influence = mode, timefix = FALSE
      )
      # R 3.8.11 cannot assemble the count dimensions for multiple scores and
      # strata together. Each single-score call still supplies pooled dfbeta,
      # whose cross-products give the joint covariance independently.
      reference <- lapply(c("score", "score2"), function(score) {
        formula <- stats::reformulate(
          c(score, "strata(stratum)"), response = "survival::Surv(time, status)"
        )
        survival::concordance(
          formula, data = data, weights = weight,
          cluster = if (clustered) data$cluster else NULL,
          influence = 1, timefix = FALSE
        )
      })
      expected_covariance <- crossprod(do.call(cbind, lapply(reference, `[[`, "dfbeta")))
      expected_concordance <- setNames(
        vapply(reference, `[[`, numeric(1), "concordance"), c("score", "score2")
      )
      expect_true(all(diag(expected_covariance) > 0))
      expect_gt(abs(expected_covariance[1L, 2L]), 1e-6)
      expect_equal(coef(actual), expected_concordance, tolerance = 1e-12)
      expect_equal(vcov(actual), expected_covariance, tolerance = 1e-12)
      expect_identical(is.null(.result_field(actual, "dfbeta")), !(mode %in% c(1L, 3L)))
      expect_identical(is.null(.result_field(actual, "influence")), !(mode %in% c(2L, 3L)))
    }
  }
})
