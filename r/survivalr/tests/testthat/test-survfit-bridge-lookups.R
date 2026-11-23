# survival::survfit(<formula>) called here would dispatch to survivalr's own
# survfit.formula, so stock fits call survival's method directly.
stock_survfit <- function(...) get("survfit.formula", asNamespace("survival"))(...)

mgus2_states <- function() {
  mg <- survival::mgus2
  mg$etime <- with(mg, ifelse(pstat == 0, futime, ptime))
  mg$event <- factor(with(mg, ifelse(pstat == 0, 2 * death, 1)), 0:2,
    labels = c("censor", "pcm", "death"))
  mg
}

test_that("survfit(coxfit, newdata, id = name) reads the id from newdata", {
  heart <- survival::heart
  fit <- coxph(Surv(start, stop, event) ~ age + transplant, data = heart)
  stock <- survival::coxph(survival::Surv(start, stop, event) ~ age + transplant, data = heart)
  nd <- data.frame(
    start = c(0, 30, 0), stop = c(30, 300, 200), event = c(0, 1, 0),
    age = c(-5, -5, 5), transplant = factor(c(0, 1, 0)), pid = c(1, 1, 2)
  )
  expected <- summary(survival::survfit(stock, newdata = nd, id = pid), times = c(100, 250))$surv
  expect_equal(summary(survfit(fit, newdata = nd, id = pid), times = c(100, 250))$surv, expected)
  # A calling-environment variable of the same name does not override newdata's.
  pid <- c(1, 2, 2)
  expect_equal(summary(survfit(fit, newdata = nd, id = pid), times = c(100, 250))$surv, expected)
  expect_equal(summary(survfit(fit, newdata = nd, id = nd$pid), times = c(100, 250))$surv, expected)
})

test_that("a selection of several strata keeps working with the whole-fit methods", {
  lung <- survival::lung
  fit <- survfit(Surv(time, status) ~ ph.ecog, data = lung)
  stock <- stock_survfit(survival::Surv(time, status) ~ ph.ecog, data = lung)
  selections <- list(1:2, c(3, 1), -1, c(TRUE, FALSE, TRUE, FALSE))
  for (selection in selections) {
    subset <- fit[selection]
    reference <- stock[selection]
    expect_equal(c(summary(subset, times = c(100, 300))$surv),
      c(summary(reference, times = c(100, 300))$surv))
    expect_equal(c(summary(survfit0(subset), times = 50)$surv),
      c(summary(survival::survfit0(reference), times = 50)$surv))
    # Rows per curve, in order; the bridge labels curves by their bare levels.
    strata <- as.character(summary(subset)$strata)
    expect_false(anyNA(strata))
    expect_equal(rle(strata)$lengths, rle(as.character(summary(reference)$strata))$lengths)
  }
  expect_equal(c(summary(fit[c("0", "1")], times = 100)$surv),
    c(summary(stock[c("ph.ecog=0", "ph.ecog=1")], times = 100)$surv))
})

test_that("a selection of several multi-state strata keeps its states", {
  mg <- mgus2_states()
  fit <- survfit(Surv(etime, event) ~ cut(age, c(0, 60, 70, 80, 120)), data = mg)
  stock <- stock_survfit(survival::Surv(etime, event) ~ cut(age, c(0, 60, 70, 80, 120)),
    data = mg)
  columns <- c("time", "n.risk", "n.event", "pstate", "state")
  expect_equal(
    summary(fit[c(1, 3), ], times = c(50, 100), data.frame = TRUE)[columns],
    summary(stock[c(1, 3), ], times = c(50, 100), data.frame = TRUE)[columns],
    ignore_attr = TRUE
  )
  # Stock R's data.frame = TRUE fails for selected states (its columns have different
  # lengths), so the states' probabilities are compared with its list form.
  expect_equal(
    summary(fit[c(1, 3), 2:3], times = c(50, 100), data.frame = TRUE)$pstate,
    c(summary(stock[c(1, 3), 2:3], times = c(50, 100))$pstate)
  )
})

test_that("multi-state summary(times =) sorts the times and accumulates the counts", {
  mg <- mgus2_states()
  fit <- survfit(Surv(etime, event) ~ 1, data = mg)
  stock <- stock_survfit(survival::Surv(etime, event) ~ 1, data = mg)
  columns <- c("time", "n.event", "n.censor", "pstate", "state")
  for (times in list(c(60, 12), c(12, 12, 60))) {
    expect_equal(summary(fit, times = times, data.frame = TRUE)[columns],
      summary(stock, times = times, data.frame = TRUE)[columns], ignore_attr = TRUE)
  }
  expect_equal(summary(fit, times = c(60, 12), dosum = TRUE, data.frame = TRUE)[columns],
    summary(stock, times = c(60, 12), data.frame = TRUE)[columns], ignore_attr = TRUE)
  expect_equal(summary(fit, times = c(60, 12), dosum = FALSE, data.frame = TRUE)[columns],
    summary(stock, times = c(60, 12), dosum = FALSE, data.frame = TRUE)[columns],
    ignore_attr = TRUE)
  grouped <- survfit(Surv(etime, event) ~ sex, data = mg)
  grouped_stock <- stock_survfit(survival::Surv(etime, event) ~ sex, data = mg)
  expect_equal(summary(grouped, times = c(60, 12), data.frame = TRUE)[columns],
    summary(grouped_stock, times = c(60, 12), data.frame = TRUE)[columns], ignore_attr = TRUE)
  expect_error(summary(fit, times = c(1, Inf)), "times contains missing or infinite values")
})
