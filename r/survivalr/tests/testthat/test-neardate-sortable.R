test_that("neardate retains stock character ordering and match direction", {
  cases <- list(
    list(id1 = rep(1, 6), id2 = rep(1, 6),
      y1 = c("2", "10", "3", NA, "11", "10"),
      y2 = c("1", "2", "10", "10", "12", NA)),
    list(id1 = c("a", "a", "b", "b", "c"),
      id2 = c("a", "b", "a", "b", "d"),
      y1 = c("2020-01-02", "2020-01-10", "2020-01-03", NA, "2020-01-01"),
      y2 = c("2020-01-01", "2020-01-03", "2020-01-12", "2020-01-03", "2020-01-01")),
    list(id1 = rep(1, 3), id2 = rep(1, 3),
      y1 = c(2, 10, 3), y2 = c("1", "2", "10")),
    list(id1 = c(NA, 1, 1), id2 = c(NA, 1, 1),
      y1 = c("a", NA, "z"), y2 = c("a", "b", NA)),
    list(id1 = rep(1, 6), id2 = rep(1, 6),
      y1 = c("a", "A", "b", "B", "z", "Z"),
      y2 = c("B", "b", "A", "a", "Z", "z"))
  )
  for (case in cases) for (best in c("after", "prior", "a", "p"))
    for (nomatch in c(NA_integer_, -1L, 0L)) {
    args <- c(case, list(best = best, nomatch = nomatch))
    expect_identical(do.call(neardate, args), do.call(survival::neardate, args))
  }
})

test_that("neardate calendar classes use common stock ranks", {
  query <- c("2020-01-02", "2020-01-10", "2020-01-03", NA, "1960-01-01", "2020-01-03")
  reference <- c("2020-01-01", "2020-01-03", "2020-01-12", "2020-01-03", NA, "1970-01-01")
  cases <- list(
    list(as.Date(query), as.Date(reference)),
    list(as.Date(query), reference),
    list(as.POSIXct(query, tz = "UTC"), as.POSIXct(reference, tz = "America/Chicago")),
    list(as.POSIXlt(query, tz = "UTC"), as.POSIXct(reference, tz = "UTC")),
    list(as.POSIXct(query, tz = "UTC"), as.POSIXlt(reference, tz = "UTC")),
    list(as.POSIXct(query, tz = "UTC"), reference)
  )
  for (case in cases) for (best in c("after", "prior")) {
    args <- list(id1 = rep(1, 6), id2 = rep(1, 6), y1 = case[[1]], y2 = case[[2]],
      best = best, nomatch = -1L)
    if (inherits(case[[1]], "POSIXt")) {
      # Stock passes the two-element POSIXt class vector to methods::as.
      # Check its failure separately and compare intended matching through
      # independently prepared epoch seconds instead.
      expect_error(do.call(survival::neardate, args), "length\\(class2\\) == 1L")
      numeric_args <- args
      numeric_args$y1 <- as.numeric(as.POSIXct(case[[1]]))
      numeric_args$y2 <- as.numeric(as.POSIXct(case[[2]]))
      expect_identical(do.call(neardate, args), do.call(survival::neardate, numeric_args))
    } else {
      expect_identical(do.call(neardate, args), do.call(survival::neardate, args))
    }
  }
})

test_that("neardate numeric paths and validation remain stock compatible", {
  cases <- list(
    list(c(NA, 2, Inf, -Inf), c(1, NA, Inf, -Inf)),
    list(c(TRUE, FALSE, NA), c(FALSE, TRUE, TRUE)),
    list(c(0L, 1L, 2L), c(1L, 1L, 2L))
  )
  for (case in cases) for (best in c("after", "prior")) {
    args <- list(id1 = rep(1, length(case[[1]])), id2 = rep(1, length(case[[2]])),
      y1 = case[[1]], y2 = case[[2]], best = best, nomatch = 0L)
    expect_identical(do.call(neardate, args), do.call(survival::neardate, args))
  }
  for (which in c(1L, 2L)) {
    dates <- list(1:2, 1:2)
    dates[[which]] <- factor(dates[[which]])
    expect_error(neardate(c(1, 1), c(1, 1), dates[[1]], dates[[2]]), "must be sortable")
  }
  expect_error(neardate(1, 1, "a", NA_character_), "No valid entries in data set 2")
  expect_error(neardate(1, 1, "a", "b", best = "closest"), "'arg' should be one of")
})
