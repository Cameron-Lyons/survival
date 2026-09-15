.concordance_explicit_strata_data <- function() {
  first <- data.frame(
    start = rep(0, 12),
    time = c(1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5),
    status = c(1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1),
    score = c(2, 2, 1, 2, 1, 3, 2, 3, 2, 1, 2, 3),
    weight = c(.5, 1.25, 2, .75, 1.5, 0, 2.25, .5, 1.75, 1, 2, 1.25)
  )
  second <- transform(first, score = 5 - score, weight = (rev(weight) + .25) * .75)
  data <- rbind(first, second)
  data$group <- rep(c("A", "B"), each = 12)
  data$score2 <- data$score + .75 * rep(c(1, -1, 0, 1, 0, -1, 1, -1, 0, 1, -1, 0), 2)
  data$cluster <- rep(c("a", "b", "a", "c", "d", "e"), 4)
  data
}

.concordance_strata_reference <- function(data, counting, columns, clustered, timewt = "n") {
  y <- if (counting) {
    survival::Surv(data$start, data$time, data$status)
  } else {
    survival::Surv(data$time, data$status)
  }
  # R 3.8.11 fails both pooled keepstrata=FALSE and joint-score strata output
  # assembly. Single-score retained fits provide independent pooled dfbeta and
  # per-stratum counts; sum counts and crossprod dfbeta for the intended output.
  lapply(columns, function(column) {
    survival::concordancefit(
      y, data[[column]], strata = data$group, weights = data$weight,
      cluster = if (clustered) data$cluster else NULL,
      timewt = timewt, influence = 3, timefix = FALSE, keepstrata = TRUE
    )
  })
}

test_that("explicit strata preserve pooled score covariance and shared clusters", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_explicit_strata_data()
  data <- data[c(13:24, 1:12), ]
  for (counting in c(FALSE, TRUE)) for (clustered in c(FALSE, TRUE)) {
    y <- if (counting) Surv(data$start, data$time, data$status) else Surv(data$time, data$status)
    for (columns in list("score", c("score", "score2"))) {
      reference <- .concordance_strata_reference(data, counting, columns, clustered)
      multi <- length(columns) > 1L
      scores <- if (multi) as.matrix(data[columns]) else data[[columns]]
      expected_count <- do.call(rbind, lapply(reference, function(f) colSums(f$count)))
      expected_dfbeta <- do.call(cbind, lapply(reference, `[[`, "dfbeta"))
      expected_covariance <- crossprod(expected_dfbeta)
      if (multi) {
        rownames(expected_count) <- columns
        expect_true(all(diag(expected_covariance) > 0))
        expect_gt(abs(expected_covariance[1, 2]), 1e-8)
      }
      for (keepstrata in list(TRUE, FALSE, 0, 1, 2, 10)) for (mode in 0:3) {
        result <- concordancefit(
          y, scores, strata = data$group, weights = data$weight,
          cluster = if (clustered) data$cluster else NULL,
          influence = mode, timefix = FALSE, keepstrata = keepstrata
        )
        expect_equal(unname(result$concordance), vapply(reference, `[[`, numeric(1), "concordance"))
        if (multi) {
          expect_equal(result$count, expected_count, tolerance = 1e-12)
          expect_equal(result$var, expected_covariance, tolerance = 1e-12)
        } else {
          retain <- isTRUE(keepstrata) || (is.numeric(keepstrata) && keepstrata >= 2)
          expected <- if (retain) reference[[1]]$count else colSums(reference[[1]]$count)
          expect_equal(result$count, expected, tolerance = 1e-12)
          expect_equal(result$var, reference[[1]]$var, tolerance = 1e-12)
        }
        expect_equal(result$cvar, vapply(reference, `[[`, numeric(1), "cvar"), tolerance = 1e-12)
        expect_identical(is.null(result$dfbeta), !(mode %in% c(1L, 3L)))
        expect_identical(is.null(result$influence), !(mode %in% c(2L, 3L)))
        if (mode %in% c(1L, 3L)) {
          expected <- if (multi) expected_dfbeta else reference[[1]]$dfbeta
          expect_equal(result$dfbeta, expected, tolerance = 1e-12)
        }
        if (mode %in% c(2L, 3L)) {
          if (multi) {
            for (j in seq_along(columns)) {
              expect_equal(result$influence[, , j], reference[[j]]$influence, tolerance = 1e-12)
            }
          } else {
            expect_equal(result$influence, reference[[1]]$influence, tolerance = 1e-12)
          }
        }
      }
    }
  }
})

test_that("retained counts use R factor and numeric stratum order", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_explicit_strata_data()
  data <- data[c(13:24, 1:12), ]
  labels <- list(data$group, factor(data$group, levels = c("B", "A")), ifelse(data$group == "A", 10, 2))
  for (group in labels) {
    data$group <- group
    reference <- .concordance_strata_reference(data, FALSE, "score", TRUE)[[1]]
    actual <- concordancefit(
      Surv(data$time, data$status), data$score, strata = group,
      weights = data$weight, cluster = data$cluster
    )
    expect_equal(actual$count, reference$count, tolerance = 1e-12)
    expect_equal(actual$var, reference$var, tolerance = 1e-12)
    no_se <- concordancefit(
      Surv(data$time, data$status), data$score, strata = group,
      weights = data$weight, std.err = FALSE
    )
    expect_equal(no_se$count, reference$count, tolerance = 1e-12)
    expect_null(no_se$var)
    expect_null(no_se$cvar)
    expect_null(no_se$dfbeta)
    expect_null(no_se$influence)
  }
})

test_that("formula strata align subset and missing data before clustered covariance", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_explicit_strata_data()
  data$group[15] <- NA_character_
  data$weight[5] <- NA_real_
  data$score2[20] <- NA_real_
  data$cluster[24] <- NA_character_
  subset <- c(24, 3, 15, 1, 9, 20, 5, 17, 11, 22, 7, 13, 2, 18, 8, 21, 12, 16)
  selected <- subset[!subset %in% c(15, 5, 20, 24)]
  for (counting in c(FALSE, TRUE)) {
    lhs <- if (counting) "Surv(start,time,status)" else "Surv(time,status)"
    formula <- stats::as.formula(paste(lhs, "~ score + score2 + strata(group)"))
    result <- concordance(
      formula, data = data, weights = weight, cluster = cluster,
      subset = seq_len(nrow(data)) %in% subset, na.action = "omit", timewt = "S", influence = 3,
      timefix = FALSE
    )
    # Logical subsetting preserves input order. Align the explicit R reference
    # independently; shared clusters cross stratum boundaries after filtering.
    clean <- data[sort(selected), ]
    reference <- .concordance_strata_reference(clean, counting, c("score", "score2"), TRUE, "S")
    expected_dfbeta <- do.call(cbind, lapply(reference, `[[`, "dfbeta"))
    expect_equal(as.numeric(result$concordance), vapply(reference, `[[`, numeric(1), "concordance"))
    expect_equal(vcov(result), crossprod(expected_dfbeta), tolerance = 1e-12)
    actual_dfbeta <- lapply(.result_field(result, "dfbeta"), .as_numeric_vector)
    for (j in 1:2) {
      # Formula wrappers expose Python's first-seen cluster order; concordancefit
      # above separately checks its R-sorted named cluster matrix convention.
      expect_equal(
        actual_dfbeta[[j]], as.numeric(reference[[j]]$dfbeta[unique(clean$cluster)]),
        tolerance = 1e-12
      )
    }
  }
})

test_that("one stratum and invalid explicit strata have clear behavior", {
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")
  data <- .concordance_explicit_strata_data()
  y <- Surv(data$time, data$status)
  one <- concordancefit(y, data$score, strata = rep("A", nrow(data)), keepstrata = TRUE)
  plain <- concordancefit(y, data$score)
  expect_equal(one$count, plain$count)
  expect_equal(concordancefit(y, data$score, strata = integer())$count, plain$count)
  expect_null(dim(one$count))
  expect_error(concordancefit(y, data$score, strata = c("A", "B")), "length")
})
