test_that("tmerge subsetting uses normal data-frame dispatch", {
  data <- data.frame(id = 1:3, stop = c(3, 5, 7), status = c(1, 0, 1))
  actual <- tmerge(data, data, id = id, event = event(stop, status))
  stock <- survival::tmerge(data, data, id = id, event = event(stop, status))
  expected_counts <- attr(stock, "tcount")
  class(stock) <- "data.frame"
  for (field in c("tm.retain", "tcount", "call")) attr(stock, field) <- NULL
  subsets <- list(
    function(x) x[],
    function(x) x[, ],
    function(x) x[1:2],
    function(x) x["event"],
    function(x) x[c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE)],
    function(x) x[FALSE],
    function(x) x[2:1, ],
    function(x) x[, "event"],
    function(x) x[2:1, "event"],
    function(x) x[2:1, "event", drop = FALSE]
  )
  for (subset in subsets) {
    result <- subset(actual)
    expected <- subset(stock)
    expect_equal(result, expected)
    expect_identical(class(result), class(expected))
    for (field in c("tm.retain", "tcount", "call")) expect_null(attr(result, field))
  }
  expect_s3_class(actual, "tmerge")
  expect_equal(attr(actual, "tcount"), expected_counts)
})
