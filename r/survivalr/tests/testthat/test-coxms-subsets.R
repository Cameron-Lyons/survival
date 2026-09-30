test_that("selected Cox states retain counts for summaries and initial rows", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  data <- survival::mgus2
  data$etime <- with(data, ifelse(pstat == 1, ptime, futime))
  data$event <- factor(with(data, ifelse(pstat == 1, "pcm",
                                        ifelse(death == 1, "death", "censor"))),
                       levels = c("censor", "pcm", "death"))
  for (stratified in c(FALSE, TRUE)) {
    formula <- if (stratified) Surv(etime, event) ~ age + strata(sex) else
      Surv(etime, event) ~ age + sex
    actual_fit <- coxph(formula, data, id = id)
    reference_fit <- survival::coxph(formula, data, id = id)
    newdata <- data.frame(age = c(60, 80), sex = c("F", "M"))
    actual <- suppressWarnings(survfit(actual_fit, newdata = newdata))
    reference <- survival::survfit(reference_fit, newdata = newdata)
    groups <- if (stratified) 2:1 else 1L
    rows <- if (stratified) unlist(lapply(groups, function(g) {
      seq_len(reference$strata[g]) + sum(reference$strata[seq_len(g - 1L)])
    })) else seq_along(reference$time)
    for (states in list("death", c("death", "pcm"), c("death", "(s0)", "death"))) {
      selected <- actual[groups, 2:1, states, drop = FALSE]
      expected <- reference[groups, 2:1, states, drop = FALSE]
      # R leaves censor counts in the original state order. Correct only the
      # metadata before asking its summary and survfit0 methods for references.
      expected$n.censor <- reference$n.censor[rows, states, drop = FALSE]
      expected$n.id <- reference$n.id[groups]
      for (initial in c(FALSE, TRUE)) {
        current <- if (initial) survfit0(selected) else selected
        ref <- if (initial) survival::survfit0(expected) else expected
        summary <- summary(current, times = c(0, 100, 200), extend = TRUE)
        ref_summary <- summary(ref, times = c(0, 100, 200), extend = TRUE)
        expect_equal(summary$time, ref_summary$time)
        expect_equal(summary$pstate, ref_summary$pstate, tolerance = 1e-8,
                     ignore_attr = TRUE)
        for (field in c("n.risk", "n.event", "n.censor")) {
          expect_equal(summary[[field]], ref_summary[[field]], ignore_attr = TRUE)
        }
        expect_null(summary$cumhaz)
        expect_null(summary$n.transition)
        expect_equal(summary$table[, c("n", "rmean")],
                     ref_summary$table[, c("n", "rmean")], tolerance = 1e-8,
                     ignore_attr = TRUE)
      }
      frame <- as.data.frame(selected)
      expect_equal(unique(as.character(frame$state)), unique(states))
      expect_equal(nrow(frame), length(expected$time) * 2 * length(states))
      expect_equal(frame$pstate, as.numeric(expected$pstate), tolerance = 1e-8)
      expect_equal(frame$n.censor,
                   unlist(lapply(seq_along(states), function(s) rep(expected$n.censor[, s], 2))))
      expect_equal(summary(selected[, , 1, drop = FALSE], times = 100)$pstate,
                   summary(selected, times = 100)$pstate[, , 1, drop = FALSE])
    }
  }
})
