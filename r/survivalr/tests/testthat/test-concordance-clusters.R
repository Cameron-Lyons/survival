.cluster_concordance_data <- function() {
  i <- seq_len(24)
  data.frame(time = 2+(i*13)%%19, status = as.integer(i%%4 != 0),
             x = sin(i), z = cos(i/3), id = rep(c("1", "10", "2"), 8))
}

test_that("fitted categorical clusters keep R levels across omission and subsets", {
  data <- .cluster_concordance_data()
  data$id <- factor(data$id, levels = c("unused", "2", "10", "1", "last"))
  data$x[4] <- NA; data$id[8] <- NA
  data <- data[c(21, 19, 8, 4, 2, 6, 14, 13, 12, 9), ]
  rows <- seq_len(nrow(data)) != 2L
  reference_concordance <- get("concordance.coxph", asNamespace("survival"))
  for (source in c("formula", "vector")) for (model in c(FALSE, TRUE)) {
    if (source == "formula") {
      fit <- coxph(Surv(time, status) ~ x + cluster(id), data,
                    subset = rows, na.action = na.exclude, model = model)
      reference <- survival::coxph(Surv(time, status) ~ x + cluster(id), data,
                                    subset = rows, na.action = na.exclude, model = model)
    } else {
      fit <- coxph(Surv(time, status) ~ x, data, cluster = id,
                    subset = rows, na.action = na.exclude, model = model)
      reference <- survival::coxph(Surv(time, status) ~ x, data, cluster = id,
                                    subset = rows, na.action = na.exclude, model = model)
    }
    for (influence in 0:3) {
      actual <- concordance(fit, influence = influence)
      expected <- reference_concordance(reference, influence = influence)
      actual$call <- expected$call
      expect_equal(actual, expected, tolerance = 1e-10)
    }
    expect_named(concordance(fit, influence = 1)$dfbeta, c("2", "10", "1"))
  }
})

test_that("renaming or reordering clusters preserves joint covariance", {
  data <- .cluster_concordance_data()
  first <- coxph(Surv(time, status) ~ x + cluster(id), data)
  original <- coxph(Surv(time, status) ~ z + cluster(id), data)
  for (group in list(factor(data$id, levels = c("2", "1", "10")),
                     c("z", "a", "b")[match(data$id, c("1", "10", "2"))])) {
    other <- data
    other$id <- group
    second <- coxph(Surv(time, status) ~ z + cluster(id), other)
    for (influence in 0:3) {
      actual <- concordance(first, second, influence = influence)
      expected <- concordance(first, original, influence = influence)
      names(expected$concordance) <- names(actual$concordance)
      rownames(expected$count) <- rownames(actual$count)
      expected$call <- actual$call
      expect_equal(actual, expected, tolerance = 1e-12)
      reverse <- concordance(second, first, influence = influence)
      expect_equal(reverse$var, actual$var[2:1, 2:1])
    }
  }
})

test_that("joint covariance rejects incompatible cluster memberships", {
  data <- .cluster_concordance_data()
  first <- coxph(Surv(time, status) ~ x + cluster(id), data)
  second <- coxph(Surv(time, status) ~ z, data, cluster = rep(1:3, each = 8))
  expect_error(concordance(first, second), "identical clustering")
  expect_error(concordance(second, first), "identical clustering")
  # One explicit grouping replaces both fitted partitions.
  actual <- concordance(first, second, cluster = data$id, influence = 1)
  influences <- cbind(concordance(first, cluster = data$id, influence = 1)$dfbeta,
                      concordance(second, cluster = data$id, influence = 1)$dfbeta)
  expect_equal(actual$dfbeta, influences)
  expect_equal(actual$var, crossprod(influences))
})

test_that("singleton groups align with unclustered models in either order", {
  data <- .cluster_concordance_data()
  first <- coxph(Surv(time, status) ~ x, data, cluster = rev(seq_len(nrow(data))))
  second <- coxph(Surv(time, status) ~ z, data)
  actual <- concordance(first, second, influence = 1)
  expected <- concordance(first, second, influence = 1, cluster = rev(seq_len(nrow(data))))
  actual$call <- expected$call
  expect_equal(actual, expected)
  actual <- concordance(second, first, influence = 1)
  expected <- concordance(second, first, influence = 1, cluster = seq_len(nrow(data)))
  expect_equal(unname(actual$dfbeta), unname(expected$dfbeta))
  expect_equal(actual$var, expected$var)
})

test_that("joint covariance corrects the reference row-position mismatch", {
  data <- .cluster_concordance_data()
  data$id <- rep(0:2, 8)
  renamed <- data
  renamed$id <- (data$id + 1) %% 3
  first <- coxph(Surv(time, status) ~ x + cluster(id), data)
  second <- coxph(Surv(time, status) ~ z + cluster(id), renamed)
  ref_first <- survival::coxph(Surv(time, status) ~ x + cluster(id), data)
  ref_second <- survival::coxph(Surv(time, status) ~ z + cluster(id), renamed)
  reference_concordance <- get("concordance.coxph", asNamespace("survival"))
  raw <- reference_concordance(ref_first, ref_second)
  one <- reference_concordance(ref_first, influence = 1)$dfbeta
  two <- reference_concordance(ref_second, influence = 1)$dfbeta
  aligned <- cbind(one, two[as.character((as.numeric(names(one))+1)%%3)])
  colnames(aligned) <- NULL
  actual <- concordance(first, second, influence = 1)
  expect_equal(actual$dfbeta, aligned, tolerance = 1e-12)
  expect_equal(actual$var, crossprod(aligned), tolerance = 1e-12)
  expect_gt(actual$var[1, 2], 0)
  expect_lt(raw$var[1, 2], 0)
  expect_equal(diag(actual$var), diag(raw$var), tolerance = 1e-12)
})
