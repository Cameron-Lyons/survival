# survSplit names, keeps and orders its columns as survival's R code does.  The stock
# calls see survival's own Surv and Surv2, as they would without survivalr.
stock_split <- function(expr) {
  eval(
    substitute(expr),
    list(survSplit = survival::survSplit, Surv = survival::Surv, Surv2 = survival::Surv2),
    parent.frame()
  )
}

test_that("survSplit names the new columns after Surv()'s matched arguments", {
  l <- survival::lung[1:2, c("time", "status", "age")]
  l$status <- l$status - 1
  heart <- survival::heart[1:6, ]
  expect_equal(
    survSplit(Surv(event = status, time = time) ~ age, data = l, cut = 200),
    stock_split(survSplit(Surv(event = status, time = time) ~ age, data = l, cut = 200))
  )
  # An argument that is not a variable name keeps the default name.
  expect_equal(
    survSplit(Surv(time, status == 1) ~ age, data = l, cut = 200),
    stock_split(survSplit(Surv(time, status == 1) ~ age, data = l, cut = 200))
  )
  expect_named(
    survSplit(Surv(time / 30, status) ~ age, data = l, cut = 10),
    c("age", "tstart", "tstop", "status")
  )
  expect_equal(
    survSplit(Surv(start / 7, stop / 7, event) ~ age, data = heart, cut = 3),
    stock_split(survSplit(Surv(start / 7, stop / 7, event) ~ age, data = heart, cut = 3))
  )
  expect_equal(
    survSplit(Surv(start, stop, event) ~ age, data = heart, cut = 30, start = "a", end = "b"),
    stock_split(survSplit(Surv(start, stop, event) ~ age, data = heart, cut = 30,
      start = "a", end = "b"))
  )
})

test_that("survSplit takes the old data, end and event form", {
  v <- survival::veteran[1:2, c("time", "status", "age")]
  reference <- stock_split(survSplit(v, cut = 100, end = "time", event = "status"))
  expect_equal(survSplit(v, cut = 100, end = "time", event = "status"), reference)
  expect_equal(survSplit(data = v, cut = 100, end = "time", event = "status"), reference)
  h <- survival::heart[1:6, c("start", "stop", "event", "age")]
  expect_equal(
    survSplit(h, cut = 30, start = "start", end = "stop", event = "event"),
    stock_split(survSplit(h, cut = 30, start = "start", end = "stop", event = "event"))
  )
  expect_error(
    survSplit(v, cut = 100, event = "status"),
    "either a formula or the end and event arguments are required"
  )
})

test_that("survSplit splits Surv2 timelines", {
  d <- data.frame(
    id = c(1, 1, 1, 2, 2), t = c(0, 5, 10, 0, 8), st = c(0, 0, 1, 0, 0),
    x = c(1, 2, 2, 3, 3)
  )
  expect_equal(
    survSplit(Surv2(t, st) ~ x, data = d, id = id, cut = c(3, 7)),
    stock_split(survSplit(Surv2(t, st) ~ x, data = d, id = id, cut = c(3, 7)))
  )
  expect_equal(
    survSplit(Surv2(t, st) ~ ., data = d, id = id, cut = c(3, 7)),
    stock_split(survSplit(Surv2(t, st) ~ ., data = d, id = id, cut = c(3, 7)))
  )
  d$st <- factor(c("none", "a", "b", "none", "a"), levels = c("none", "a", "b"))
  expect_equal(
    survSplit(Surv2(t, st) ~ x, data = d, id = id, cut = c(3, 7)),
    stock_split(survSplit(Surv2(t, st) ~ x, data = d, id = id, cut = c(3, 7)))
  )
  expect_error(survSplit(Surv2(t, st) ~ x, data = d, cut = 3), "an id statement is required")
})

test_that("survSplit keeps the data's columns for a '.' right-hand side with all rows", {
  v <- survival::veteran[1:3, c("time", "status", "age")]
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = v, cut = 100, end = "stop", event = "dead"),
    stock_split(survSplit(Surv(time, status) ~ ., data = v, cut = 100,
      end = "stop", event = "dead"))
  )
  # With rows dropped, the model frame's columns are kept instead.
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = v, subset = age > 63, cut = 100),
    stock_split(survSplit(Surv(time, status) ~ ., data = v, subset = age > 63, cut = 100))
  )
  lu <- survival::lung[1:8, c("time", "status", "age", "ph.ecog", "sex")]
  lu$ph.ecog[2] <- NA
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = lu, na.action = na.omit, cut = 200),
    stock_split(survSplit(Surv(time, status) ~ ., data = lu, na.action = na.omit, cut = 200))
  )
})

test_that("survSplit's episode and added names go through make.names", {
  v <- survival::veteran[1:2, c("time", "status", "age")]
  split <- survSplit(Surv(time, status) ~ age, data = v, cut = 100,
    episode = "my ep", added = "was added")
  expect_equal(
    split,
    stock_split(survSplit(Surv(time, status) ~ age, data = v, cut = 100,
      episode = "my ep", added = "was added"))
  )
  expect_named(split, c("age", "tstart", "time", "status", "my.ep", "was.added"))
  expect_type(split$my.ep, "double")
})

test_that("survSplit's multi-state status has R's censor level", {
  mg <- data.frame(t = c(4, 9, 6), st = factor(c("a", "b", "none"), levels = c("none", "a", "b")))
  split <- survSplit(Surv(t, st) ~ 1, data = mg, cut = 5)
  expect_equal(split, stock_split(survSplit(Surv(t, st) ~ 1, data = mg, cut = 5)))
  expect_equal(levels(split$st), c("censor", "a", "b"))
  expect_equal(
    survSplit(Surv(t, st) ~ 1, data = mg, cut = 5, id = "id"),
    stock_split(survSplit(Surv(t, st) ~ 1, data = mg, cut = 5, id = "id"))
  )
})

test_that("survSplit adds only an id it creates", {
  v <- survival::veteran[1:3, c("time", "status", "age")]
  expect_equal(
    survSplit(Surv(time, status) ~ age, data = v, id = "newid", cut = 100),
    stock_split(survSplit(Surv(time, status) ~ age, data = v, id = "newid", cut = 100))
  )
  v$sid <- c(5, 6, 7)
  expect_equal(
    survSplit(Surv(time, status) ~ age, data = v, id = "sid", cut = 100),
    stock_split(survSplit(Surv(time, status) ~ age, data = v, id = "sid", cut = 100))
  )
  # R fails here (it assigns 1:nrow(data) to the smaller model frame); each kept row
  # takes its data row's number.
  expect_equal(
    survSplit(Surv(time, status) ~ age, data = v, subset = age > 63, id = "newid", cut = 100)$newid,
    c(1, 2, 2)
  )
})

test_that("survSplit stores a split Surv response under the response's name", {
  h <- survival::heart[1:4, c("start", "stop", "event", "age")]
  h$y <- survival::Surv(h$start, h$stop, h$event)
  split <- survSplit(y ~ age, data = h, cut = 30)
  reference <- stock_split(survSplit(y ~ age, data = h, cut = 30))
  expect_equal(split, reference, ignore_attr = "inputAttributes")
  # R's (time, status) branch stores Surv2(tstart, status); the split is
  # Surv(tstart, tstop, status).
  v <- survival::veteran[1:2, c("time", "status", "age")]
  v$y <- survival::Surv(v$time, v$status)
  split <- survSplit(y ~ age, data = v, cut = 100)
  expect_s3_class(split$y, "Surv")
  expect_equal(attr(split$y, "type"), "counting")
  expect_equal(unname(unclass(split$y)[, 1:3]), cbind(c(0, 0, 100), c(72, 100, 411), c(1, 0, 1)))
})

test_that("survSplit splits a single row and Surv2 subjects with any id", {
  d <- data.frame(time = c(5, 10, 3), status = c(1, 0, 1), age = c(50, 60, 70))
  expect_equal(
    survSplit(Surv(time, status) ~ age, data = d[1, ], cut = 4),
    stock_split(survSplit(Surv(time, status) ~ age, data = d[1, ], cut = 4))
  )
  expect_equal(
    survSplit(Surv(time, status) ~ age, data = d, subset = age == 50, cut = 4),
    stock_split(survSplit(Surv(time, status) ~ age, data = d, subset = age == 50, cut = 4))
  )
  one <- data.frame(t1 = 0, t2 = 5, s = 1, w = 1)
  expect_equal(
    survSplit(Surv(t1, t2, s) ~ w, data = one, cut = 3),
    stock_split(survSplit(Surv(t1, t2, s) ~ w, data = one, cut = 3))
  )
  timeline <- data.frame(pid = c(1, 1, 2), day = c(0, 5, 0), ev = c(0, 1, 1), z = 1:3)
  expect_equal(
    survSplit(Surv2(day, ev) ~ z, data = timeline[1, ], id = pid, cut = 3),
    stock_split(survSplit(Surv2(day, ev) ~ z, data = timeline[1, ], id = pid, cut = 3))
  )
  # R groups a missing id as one subject, and orders character ids as sort() does.
  for (pid in list(c(1, NA, 2), c("b", "b", "a"))) {
    timeline$pid <- pid
    expect_equal(
      survSplit(Surv2(day, ev) ~ z, data = timeline, id = pid, cut = 3),
      stock_split(survSplit(Surv2(day, ev) ~ z, data = timeline, id = pid, cut = 3))
    )
  }
})

test_that("survSplit builds the multi-state status as R's factor(labels =) does", {
  # A state named "censor" merges with the censored rows.
  d <- data.frame(time = c(5, 10, 3), age = c(50, 60, 70))
  d$fs <- factor(c("none", "censor", "b"), levels = c("none", "censor", "b"))
  split <- survSplit(Surv(time, fs) ~ age, data = d, cut = 4)
  expect_equal(split, stock_split(survSplit(Surv(time, fs) ~ age, data = d, cut = 4)))
  expect_equal(levels(split$fs), c("censor", "b"))
  # A factor with one level is a multi-state status with no states.
  one_level <- data.frame(time = c(30, 80), x = 1:2, st = factor(c("alive", "alive")))
  split <- survSplit(Surv(time, st) ~ x, data = one_level, cut = 50)
  expect_equal(split, stock_split(survSplit(Surv(time, st) ~ x, data = one_level, cut = 50)))
  expect_equal(levels(split$st), "censor")
})

test_that("survSplit reads zero and timefix as R does", {
  d <- data.frame(time = c(10, 20), status = c(1, 0))
  # timefix runs on the cut and event times before zero becomes the start.
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = d, cut = 2 + 1e-12, zero = c(2, 0)),
    stock_split(survSplit(Surv(time, status) ~ ., data = d, cut = 2 + 1e-12, zero = c(2, 0)))
  )
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = d, cut = 15, zero = c(10 - 1e-12, 0)),
    stock_split(survSplit(Surv(time, status) ~ ., data = d, cut = 15, zero = c(10 - 1e-12, 0)))
  )
  expect_equal(
    survSplit(Surv(time, status) ~ ., data = d, cut = 5, zero = -Inf),
    stock_split(survSplit(Surv(time, status) ~ ., data = d, cut = 5, zero = -Inf))
  )
  # (start, stop] data never reads zero.
  heart <- survival::heart[1:20, ]
  expect_equal(
    survSplit(Surv(start, stop, event) ~ age, data = heart, cut = 30, zero = NA),
    stock_split(survSplit(Surv(start, stop, event) ~ age, data = heart, cut = 30, zero = NA))
  )
  expect_error(survSplit(Surv(time, status) ~ ., data = d, cut = NA),
    "cut must be a vector of finite numbers")
  expect_error(survSplit(Surv(time, status) ~ ., data = d, cut = list(5, 15)),
    "cut must be a vector of finite numbers")
  expect_error(survSplit(Surv(time, status) ~ ., data = d, cut = 5, zero = NA),
    "missing value where TRUE/FALSE needed")
  expect_error(survSplit(Surv(time, status) ~ ., data = d, cut = 5, timefix = "a"),
    "invalid value for timefix option")
})

test_that("survSplit numbers a created id by model-frame row when no row is dropped", {
  v <- survival::veteran[1:10, ]
  expect_equal(
    survSplit(Surv(time, status) ~ trt, data = v, cut = 50, id = "nid", subset = 10:1),
    stock_split(survSplit(Surv(time, status) ~ trt, data = v, cut = 50, id = "nid",
      subset = 10:1))
  )
})
