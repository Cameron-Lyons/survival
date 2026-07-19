test_that("R formula wrappers delegate to the Python survival package", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 1, 0, 1),
    group = c("control", "control", "treated", "treated"),
    x = c(0.2, 0.4, 0.8, 1.0),
    wt = c(1.0, 2.0, 1.5, 0.5)
  )

  model_frame_probe <- stats::model.frame(Surv(time, status) ~ group + x, data = data)
  model_frame_response <- stats::model.response(model_frame_probe)
  expect_s3_class(model_frame_response, "Surv")
  expect_equal(unname(model_frame_response[, "time"]), data$time)
  expect_equal(unname(model_frame_response[, "status"]), data$status)
  expect_equal(model_frame_probe$group, data$group)
  expect_equal(model_frame_probe$x, data$x)
  subset_model_frame_probe <- stats::model.frame(
    Surv(time, status) ~ group,
    data = data,
    subset = c(TRUE, FALSE, TRUE, FALSE)
  )
  subset_model_frame_response <- stats::model.response(subset_model_frame_probe)
  expect_s3_class(subset_model_frame_response, "Surv")
  expect_equal(unname(subset_model_frame_response[, "time"]), data$time[c(1L, 3L)])
  expect_equal(unname(subset_model_frame_response[, "status"]), data$status[c(1L, 3L)])
  expect_equal(subset_model_frame_probe$group, data$group[c(1L, 3L)])

  response <- Surv(data$time, data$status)
  expect_true(is.Surv(response))
  expected_response_matrix <- cbind(time = data$time, status = data$status)
  expect_equal(length(response), nrow(data))
  expect_null(names(response))
  expect_equal(as.matrix(response), expected_response_matrix)
  expect_error(as.logical(response), "invalid operation on a survival time")
  expect_equal(c(response, response[1L]), response[c(seq_len(nrow(data)), 1L)])
  expect_equal(c(response[1L], response[2:3]), response[1:3])
  expect_equal(rev(response), response[rev(seq_len(nrow(data)))])
  expect_equal(rep(response[1:2], 2), response[c(1L, 2L, 1L, 2L)])
  expect_equal(rep.int(response[1:2], 2), response[c(1L, 2L, 1L, 2L)])
  expect_equal(rep_len(response, 6), response[c(1L, 2L, 3L, 4L, 1L, 2L)])
  expect_equal(t(response), t(expected_response_matrix))
  expect_equal(head(response, 2), response[1:2])
  expect_equal(tail(response, 2), response[3:4])
  expect_equal(
    quantile(response, probs = c(0.25, 0.5, 0.75), conf.int = FALSE),
    c(`25` = 1.5, `50` = 3.0, `75` = 4.0)
  )
  response_quantile <- quantile(response, probs = c(0.25, 0.5, 0.75))
  expect_named(response_quantile, c("quantile", "lower", "upper"))
  expect_equal(response_quantile$quantile, c(`25` = 1.5, `50` = 3.0, `75` = 4.0))
  expect_equal(response_quantile$lower, c(`25` = 1.0, `50` = 1.0, `75` = 2.0))
  expect_true(all(is.na(response_quantile$upper)))
  response_median <- median(response)
  expect_named(response_median, c("quantile", "lower", "upper"))
  expect_equal(response_median$quantile, c(`50` = 3.0))
  duplicate_response <- Surv(c(1, 2, 1, NA), c(1, 0, 1, 0))
  expect_equal(as.vector(duplicated(duplicate_response)), c(FALSE, FALSE, TRUE, FALSE))
  expect_equal(anyDuplicated(duplicate_response), 3L)
  expect_equal(unique(duplicate_response), duplicate_response[c(1L, 2L, 4L)])
  expect_equal(xtfrm(duplicate_response), c(1L, 3L, 2L, NA_integer_))
  expect_error(sum(response), "Invalid operation on a survival time")
  expect_error(response + 1, "Invalid operation on a survival time")
  expect_error(log(response), "Invalid operation on a survival time")
  expect_equal(capture.output(print(response)), "[1] 1  2  3+ 4 ")
  native_response <- .as_native_surv(response)
  expect_equal(capture.output(print(native_response)), "[1] 1  2  3+ 4 ")
  native_survfit <- survfit(native_response)
  python_survfit <- survfit(response)
  expect_s3_class(native_survfit, "survival_py_survfit")
  expect_equal(as.data.frame(native_survfit), as.data.frame(python_survfit))
  grouped_native_survfit <- survfit(native_response, group = data$group, se.fit = FALSE)
  grouped_python_survfit <- survfit(response, group = data$group, se.fit = FALSE)
  expect_s3_class(grouped_native_survfit, "survival_py_survfit")
  expect_equal(as.data.frame(grouped_native_survfit), as.data.frame(grouped_python_survfit))
  renamed_response <- response
  names(renamed_response) <- letters[seq_len(nrow(data))]
  expect_s3_class(renamed_response, "Surv")
  expect_equal(names(renamed_response), letters[seq_len(nrow(data))])
  expected_renamed_response <- response[seq_len(nrow(data))]
  rownames(expected_renamed_response) <- letters[seq_len(nrow(data))]
  expect_equal(renamed_response, expected_renamed_response)
  response_frame <- as.data.frame(response)
  expect_s3_class(response_frame, "data.frame")
  expect_equal(names(response_frame), "x")
  expect_s3_class(response_frame[[1L]], "Surv")
  expect_equal(response_frame[[1L]][, "time"], data$time)
  expect_equal(response_frame[[1L]][, "status"], data$status)
  counting_response <- Surv(c(0, 1), c(2, 3), c(1, 0))
  counting_frame <- as.data.frame(counting_response)
  expect_equal(names(counting_frame), "x")
  expect_s3_class(counting_frame[[1L]], "Surv")
  expect_equal(counting_frame[[1L]][, "stop"], c(2, 3))
  named_response <- Surv(time = data$time, status = data$status)
  named_frame <- as.data.frame(named_response)
  expect_equal(named_frame[[1L]][, "time"], data$time)
  expect_equal(named_frame[[1L]][, "status"], data$status)
  named_counting <- Surv(start = c(0, 1), stop = c(2, 3), status = c(1, 0))
  named_counting_frame <- as.data.frame(named_counting)
  expect_equal(named_counting_frame[[1L]][, "start"], c(0, 1))
  expect_equal(named_counting_frame[[1L]][, "stop"], c(2, 3))
  named_interval2 <- Surv(time1 = c(-Inf, 2), stop = c(1, Inf), type = "interval2")
  named_interval2_frame <- as.data.frame(named_interval2)
  expect_equal(named_interval2_frame[[1L]][, "status"], c(2, 0))
  expect_error(
    Surv(time = data$time, start = data$time, status = data$status),
    "multiple time"
  )
  expect_equal(as.character(response), c("1", "2", "3+", "4"))
  expect_equal(is.na(response), c(FALSE, FALSE, FALSE, FALSE))
  expect_equal(format(response), c("1 ", "2 ", "3+", "4 "))
  missing_response <- Surv(c(1, NA, 3), c(1, 0, 1))
  expect_equal(is.na(missing_response), c(FALSE, TRUE, FALSE))
  expect_equal(as.character(missing_response), c(" 1", "NA+", " 3"))
  expect_equal(format(missing_response), c(" 1 ", "NA+", " 3 "))
  expect_error(quantile(missing_response, probs = 0.5), "missing values")
  expect_equal(quantile(missing_response, probs = 0.5, na.rm = TRUE, conf.int = FALSE), c(`50` = 2))
  expect_equal(trimws(format(counting_response)), c("(0, 2]", "(1, 3+]"))
  expect_equal(format.Surv(response), format(response))
  expect_equal(is.na.Surv(response), is.na(response))
  factor_response <- Surv(c(1, 2, NA), factor(c("censor", "relapse", "death")))
  expect_equal(levels(factor_response), c("death", "relapse"))
  factor_response_matrix <- as.matrix(factor_response)
  expect_false(inherits(factor_response_matrix, "Surv"))
  expect_equal(dim(factor_response_matrix), c(3L, 2L))
  expect_equal(colnames(factor_response_matrix), c("time", "status"))
  expect_error(as.logical(factor_response), "invalid operation on a survival time")
  factor_response_frame <- as.data.frame(factor_response)
  expect_s3_class(factor_response_frame, "data.frame")
  expect_equal(names(factor_response_frame), "x")
  expect_equal(nrow(factor_response_frame), 3L)
  expect_equal(factor_response_frame[[1L]], factor_response)
  expect_s3_class(factor_response[1:2], "Surv")
  expect_equal(attr(factor_response[1:2], "states"), attr(factor_response, "states"))
  surv2_frame_response <- Surv2(c(1, 2, NA), c("a", "b", NA))
  surv2_response_matrix <- as.matrix(surv2_frame_response)
  expect_false(inherits(surv2_response_matrix, "Surv2"))
  expect_equal(dim(surv2_response_matrix), c(3L, 2L))
  expect_equal(colnames(surv2_response_matrix), c("time", "status"))
  expect_error(as.logical(surv2_frame_response), "invalid operation on a survival time")
  expect_equal(c(surv2_frame_response[1L], surv2_frame_response[2:3]), surv2_frame_response[1:3])
  expect_equal(rev(surv2_frame_response), surv2_frame_response[3:1])
  expect_equal(rep(surv2_frame_response[1:2], 2), surv2_frame_response[c(1L, 2L, 1L, 2L)])
  expect_equal(rep.int(surv2_frame_response[1:2], 2), surv2_frame_response[c(1L, 2L, 1L, 2L)])
  expect_equal(rep_len(surv2_frame_response, 4), surv2_frame_response[c(1L, 2L, 3L, 1L)])
  expect_equal(t(surv2_frame_response), t(surv2_response_matrix))
  expect_equal(tail(surv2_frame_response, 2), surv2_frame_response[2:3])
  duplicate_surv2_response <- Surv2(c(1, 2, 1, NA), c("a", "b", "a", NA))
  expect_equal(as.vector(duplicated(duplicate_surv2_response)), c(FALSE, FALSE, TRUE, FALSE))
  expect_equal(anyDuplicated(duplicate_surv2_response), 3L)
  expect_error(sum(surv2_frame_response), "Invalid operation on a survival time")
  expect_error(surv2_frame_response + 1, "Invalid operation on a survival time")
  expect_error(log(surv2_frame_response), "Invalid operation on a survival time")
  expect_equal(capture.output(print(surv2_frame_response)), "[1]  1+   2:b NA? ")
  surv2_response_frame <- as.data.frame(surv2_frame_response)
  expect_s3_class(surv2_response_frame, "data.frame")
  expect_equal(names(surv2_response_frame), "x")
  expect_equal(nrow(surv2_response_frame), 3L)
  expect_equal(surv2_response_frame[[1L]], surv2_frame_response)
  expect_s3_class(surv2_frame_response[1:2], "Surv2")
  expect_equal(attr(surv2_frame_response[1:2], "states"), attr(surv2_frame_response, "states"))
  surv2data_probe <- data.frame(
    id = c(1, 1, 1, 2, 2),
    time = c(0, 2, 5, 0, 3),
    state = factor(
      c("entry", "ill", "death", "entry", "censor"),
      levels = c("censor", "entry", "ill", "death")
    ),
    z = c("A", "A", "A", "B", "B"),
    x = c(10, 11, 12, 20, 21)
  )
  bridged_surv2data_probe <- Surv2data(Surv2(time, state) ~ z + x, data = surv2data_probe, id = id)
  expect_equal(names(bridged_surv2data_probe)[[1L]], "Surv2(time, state)")
  expect_s3_class(bridged_surv2data_probe[[1L]], "Surv2")
  expect_equal(nrow(bridged_surv2data_probe), 3L)
  expect_false(any(grepl("Surv2\\(time, state\\)\\.time", capture.output(print(bridged_surv2data_probe)))))
  reference_factor_response <- survival::Surv(c(1, 2, NA), factor(c("censor", "relapse", "death")))
  expect_true(is.Surv(factor_response))
  expect_true(is.Surv(reference_factor_response))
  expect_equal(unclass(factor_response), unclass(reference_factor_response))
  expect_equal(attr(factor_response, "type"), attr(reference_factor_response, "type"))
  expect_equal(attr(factor_response, "states"), attr(reference_factor_response, "states"))
  expect_equal(attr(factor_response, "inputAttributes"), attr(reference_factor_response, "inputAttributes"))
  expect_equal(format(factor_response), format(reference_factor_response))
  expect_equal(is.na(factor_response), is.na(reference_factor_response))
  factor_counting_response <- Surv(c(0, 0), c(1, 2), factor(c("a", "b")), type = "counting")
  reference_factor_counting_response <- survival::Surv(
    c(0, 0),
    c(1, 2),
    factor(c("a", "b")),
    type = "counting"
  )
  expect_true(is.Surv(factor_counting_response))
  expect_equal(unclass(factor_counting_response), unclass(reference_factor_counting_response))
  expect_equal(attr(factor_counting_response, "type"), attr(reference_factor_counting_response, "type"))
  expect_equal(attr(factor_counting_response, "states"), attr(reference_factor_counting_response, "states"))
  expect_equal(attr(factor_counting_response, "inputAttributes"), attr(reference_factor_counting_response, "inputAttributes"))
  expect_equal(format(factor_counting_response), format(reference_factor_counting_response))
  expect_equal(is.na(factor_counting_response), is.na(reference_factor_counting_response))
  reference_model_frame_formula <- Surv(time, status) ~ group + x
  reference_model_frame_env <- list2env(list(Surv = survival::Surv), parent = parent.frame())
  environment(reference_model_frame_formula) <- reference_model_frame_env
  actual_model_frame <- stats::model.frame(Surv(time, status) ~ group + x, data = data)
  reference_model_frame <- stats::model.frame.default(reference_model_frame_formula, data = data)
  expect_equal(names(actual_model_frame), names(reference_model_frame))
  expect_equal(stats::model.response(actual_model_frame), stats::model.response(reference_model_frame))
  expect_equal(actual_model_frame$group, reference_model_frame$group)
  expect_equal(actual_model_frame$x, reference_model_frame$x)
  expect_equal(as.data.frame(response), as.data.frame(survival::Surv(data$time, data$status)))
  expect_equal(as.matrix(response), as.matrix(survival::Surv(data$time, data$status)))
  expect_equal(c(response, response[1L]), c(survival::Surv(data$time, data$status), survival::Surv(data$time, data$status)[1L]))
  expect_equal(rep(response[1:2], 2), rep(survival::Surv(data$time, data$status)[1:2], 2))
  expect_equal(rep.int(response[1:2], 2), rep.int(survival::Surv(data$time, data$status)[1:2], 2))
  expect_equal(rep_len(response, 6), rep_len(survival::Surv(data$time, data$status), 6))
  expect_equal(rev(response), rev(survival::Surv(data$time, data$status)))
  expect_equal(t(response), t(survival::Surv(data$time, data$status)))
  expect_equal(head(response, 2), head(survival::Surv(data$time, data$status), 2))
  expect_equal(tail(response, 2), tail(survival::Surv(data$time, data$status), 2))
  expect_equal(duplicated(response), duplicated(survival::Surv(data$time, data$status)))
  expect_equal(anyDuplicated(response), anyDuplicated(survival::Surv(data$time, data$status)))
  expect_equal(unique(response), unique(survival::Surv(data$time, data$status)))
  expect_equal(xtfrm(response), xtfrm(survival::Surv(data$time, data$status)))
  expect_equal(
    quantile(response, probs = c(0.25, 0.5, 0.75), conf.int = FALSE),
    quantile(survival::Surv(data$time, data$status), probs = c(0.25, 0.5, 0.75), conf.int = FALSE)
  )
  expect_equal(
    quantile(response, probs = c(0.25, 0.5, 0.75)),
    quantile(survival::Surv(data$time, data$status), probs = c(0.25, 0.5, 0.75))
  )
  expect_equal(median(response), median(survival::Surv(data$time, data$status)))
  expect_equal(
    as.data.frame(counting_response),
    as.data.frame(survival::Surv(c(0, 1), c(2, 3), c(1, 0)))
  )
  expect_equal(
    capture.output(print(response)),
    capture.output(print(survival::Surv(data$time, data$status)))
  )
  expect_equal(
    capture.output(print(surv2_frame_response)),
    capture.output(print(survival::Surv2(c(1, 2, NA), c("a", "b", NA))))
  )
  expect_s3_class(response[1:2], "Surv")
  expect_equal(response[1:2], survival::Surv(data$time, data$status)[1:2])
  expect_equal(response[, 1], survival::Surv(data$time, data$status)[, 1])
  expect_equal(response[FALSE], survival::Surv(data$time, data$status)[FALSE])
  expect_equal(
    Surv(c(0, 1, 2), c(1, 2, 3), c(1, 0, 1))[1:2],
    survival::Surv(c(0, 1, 2), c(1, 2, 3), c(1, 0, 1))[1:2]
  )
  expect_equal(
    Surv(c(1, 2, 3), c(1, 0, 1), type = "left")[c(TRUE, FALSE, TRUE)],
    survival::Surv(c(1, 2, 3), c(1, 0, 1), type = "left")[c(TRUE, FALSE, TRUE)]
  )
  expect_equal(
    Surv(c(1, 2, 3), c(2, 3, 4), c(0, 2, 3), type = "interval")[1:2],
    survival::Surv(c(1, 2, 3), c(2, 3, 4), c(0, 2, 3), type = "interval")[1:2]
  )
  expect_equal(
    Surv(c(-Inf, 2, 3), c(1, 3, Inf), type = "interval2")[1:2],
    survival::Surv(c(-Inf, 2, 3), c(1, 3, Inf), type = "interval2")[1:2]
  )
  native_surv_examples <- list(
    survival::Surv(c(1, NA, 3), c(1, 0, 1)),
    survival::Surv(c(1, NA, 3), c(1, 0, 1), type = "left"),
    survival::Surv(c(1, NA, 3), c(2, 3, 4), c(1, 0, NA)),
    survival::Surv(c(1, NA, 3), c(2, 3, 4), c(1, 3, 0), type = "interval"),
    survival::Surv(c(-Inf, 1, 2), c(1, 2, Inf), type = "interval2"),
    survival::Surv(c(1, 2, NA), factor(c("censor", "relapse", "death")))
  )
  for (native_surv in native_surv_examples) {
    expect_equal(format.Surv(native_surv), survival::format.Surv(native_surv))
    expect_equal(is.na.Surv(native_surv), survival::is.na.Surv(native_surv))
  }
  surv2_response <- Surv2(c(1, 2, 3), c("a", "b", "c"))
  reference_surv2 <- survival::Surv2(c(1, 2, 3), c("a", "b", "c"))
  expect_equal(unclass(surv2_response), unclass(reference_surv2))
  expect_equal(attr(surv2_response, "states"), attr(reference_surv2, "states"))
  expect_equal(attr(surv2_response, "repeated"), attr(reference_surv2, "repeated"))
  expect_equal(format(surv2_response), format(reference_surv2))
  missing_surv2 <- Surv2(c(1, NA, 3), c(NA, "b", "c"), repeated = TRUE)
  expect_equal(is.na(missing_surv2), c(TRUE, TRUE, FALSE))
  expect_true(attr(missing_surv2, "repeated"))
  expect_error(Surv2(c(1, 2), c("a")), "different lengths")
  expect_error(Surv2(c(1, 2), c("a", "b"), repeated = c(TRUE, FALSE)), "repeated")

  timeline_data <- data.frame(
    id = c(1, 1, 1, 2, 2),
    time = c(0, 2, 5, 0, 3),
    state = factor(
      c("entry", "ill", "death", "entry", "censor"),
      levels = c("censor", "entry", "ill", "death")
    ),
    z = c("A", "A", "A", "B", "B"),
    x = c(10, 11, 12, 20, 21)
  )
  expect_equal(
    Surv2data(survival::Surv2(time, state) ~ z + x, data = timeline_data, id = id),
    survival::Surv2data(survival::Surv2(time, state) ~ z + x, data = timeline_data, id = id)
  )

  counting_data <- data.frame(
    id = c(1, 1, 2),
    start = c(0, 2, 0),
    stop = c(2, 5, 3),
    state = factor(c("ill", "death", "censor"), levels = c("censor", "ill", "death")),
    istate = factor(c("entry", "ill", "entry"), levels = c("entry", "ill", "death")),
    z = c("A", "A", "B"),
    x = c(10, 10, 20)
  )
  expect_equal(
    totimeline(Surv(start, stop, state) ~ z + x, data = counting_data, id = id, istate = istate),
    survival::totimeline(survival::Surv(start, stop, state) ~ z + x, data = counting_data, id = id, istate = istate)
  )
  counting_no_istate <- counting_data[c("id", "start", "stop", "state", "z", "x")]
  expect_equal(
    totimeline(Surv(start, stop, state) ~ z + x, data = counting_no_istate, id = id),
    survival::totimeline(survival::Surv(start, stop, state) ~ z + x, data = counting_no_istate, id = id)
  )

  timeline_right <- data.frame(
    id = c(1, 1, 1, 2, 2, 2),
    time = c(0, 2, 5, 0, 3, 6),
    status = c(1, 1, 1, 1, 1, 0),
    z = c("A", "A", "A", "B", "B", "B"),
    x = c(10, 11, 12, 20, 21, 22)
  )
  expect_equal(
    fromtimeline(Surv(time, status) ~ z + x, data = timeline_right, id = id),
    survival::fromtimeline(survival::Surv(time, status) ~ z + x, data = timeline_right, id = id)
  )
  timeline_multistate <- data.frame(
    id = c(1, 1, 1, 2, 2, 2),
    time = c(0, 2, 5, 0, 3, 6),
    state = factor(
      c("entry", "ill", "death", "entry", "ill", "censor"),
      levels = c("censor", "entry", "ill", "death")
    ),
    z = c("A", "A", "A", "B", "B", "B"),
    x = c(10, 11, 12, 20, 21, 22)
  )
  expect_equal(
    fromtimeline(Surv(time, state) ~ z + x, data = timeline_multistate, id = id),
    survival::fromtimeline(survival::Surv(time, state) ~ z + x, data = timeline_multistate, id = id)
  )

  yates_data <- data.frame(
    y = c(1, 2, 3, 4, 5, 6),
    group = factor(c("A", "A", "B", "B", "C", "C"))
  )
  yates_fit <- stats::lm(y ~ group, data = yates_data, model = TRUE)
  expect_equal(yates(yates_fit, "group"), survival::yates(yates_fit, "group"))

  yates_cox_data <- data.frame(
    time = c(5, 8, 6, 9, 7, 10, 4, 11, 12, 13),
    status = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 0),
    x = c(0.2, -0.1, 0.8, 0.4, -0.3, 0.5, 1.2, -0.7, 0.1, 0.9)
  )
  yates_cox_fit <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = yates_cox_data,
    model = TRUE,
    x = TRUE
  )
  expect_equal(yates_setup(yates_cox_fit, predict = "linear"), survival::yates_setup(yates_cox_fit, predict = "linear"))
  expect_equal(
    yates_setup(yates_cox_fit, predict = "risk")(c(-1, 0, 1), NULL),
    survival::yates_setup(yates_cox_fit, predict = "risk")(c(-1, 0, 1), NULL)
  )
  expect_error(yates_setup(yates_cox_fit, predict = "terms"), "type expected is not supported")
  yates_glm_fit <- stats::glm(
    c(0, 1, 1, 0) ~ c(0, 0, 1, 1),
    family = stats::binomial()
  )
  expect_equal(yates_setup(yates_glm_fit, predict = "link"), survival::yates_setup(yates_glm_fit, predict = "link"))
  expect_equal(
    yates_setup(yates_glm_fit, predict = "response")(c(-1, 0, 1), NULL),
    survival::yates_setup(yates_glm_fit, predict = "response")(c(-1, 0, 1), NULL)
  )
  expect_warning(
    expect_null(yates_setup(yates_fit, type = "risk")),
    "no yates_setup method exists"
  )
  yates_py_cox_fit <- coxph(
    Surv(time, status) ~ x,
    data = yates_cox_data,
    max_iter = 0
  )
  expect_null(yates_setup(yates_py_cox_fit, predict = "lp"))
  expect_equal(yates_setup(yates_py_cox_fit, predict = "risk")(c(-1, 0, 1), NULL), exp(c(-1, 0, 1)))

  aareg_data <- data.frame(
    time = c(1, 2, 3, 4, 5),
    status = c(1, 0, 1, 1, 0),
    x = c(0, 1, 0.5, 1.5, 0.2)
  )
  expect_equal(
    aareg(survival::Surv(time, status) ~ x, data = aareg_data),
    survival::aareg(survival::Surv(time, status) ~ x, data = aareg_data)
  )
  tmerge_data <- data.frame(id = 1:2, tstop = c(5, 6))
  expect_equal(
    tmerge(tmerge_data, tmerge_data, id = id, tstop = tstop),
    survival::tmerge(tmerge_data, tmerge_data, id = id, tstop = tstop)
  )
  clogit_data <- data.frame(
    case = c(1, 0, 1, 0, 0, 1, 0, 1),
    set = factor(c(1, 1, 2, 2, 3, 3, 4, 4)),
    x = c(0.2, 0.4, 0.3, 0.1, 0.5, 0.2, 0.3, 0.7)
  )
  bridged_clogit <- clogit(case ~ x + strata(set), data = clogit_data, method = "breslow")
  reference_clogit <- survival::coxph(
    survival::Surv(rep(1, nrow(clogit_data)), case) ~ x + survival::strata(set),
    data = clogit_data,
    method = "breslow"
  )
  expect_equal(coef(bridged_clogit), coef(reference_clogit), tolerance = 1e-6)
  expect_equal(as.numeric(logLik(bridged_clogit)), as.numeric(logLik(reference_clogit)), tolerance = 1e-8)
  expect_s3_class(bridged_clogit, "clogit")
  expect_s3_class(bridged_clogit, "survival_py_model")
  expect_s3_class(bridged_clogit, "coxph")
  expect_equal(attr(bridged_clogit, "userCall")[[1L]], quote(clogit))

  adjusted_response <- aeqSurv(survival::Surv(c(1, 1 + 1e-8, 2), c(1, 0, 1)), tolerance = 1e-7)
  adjusted_frame <- as.data.frame(adjusted_response)
  expect_equal(adjusted_frame[[1L]][, "time"], c(1, 1, 2), tolerance = 1e-10)
  expect_equal(adjusted_frame[[1L]][, "status"], c(1, 0, 1))
  adjusted_multistate <- aeqSurv(
    Surv(c(1, 1 + 1e-8, 2), factor(c("censor", "ill", "death"))),
    tolerance = 1e-7
  )
  reference_adjusted_multistate <- survival::aeqSurv(
    survival::Surv(c(1, 1 + 1e-8, 2), factor(c("censor", "ill", "death"))),
    tolerance = 1e-7
  )
  expect_equal(unclass(adjusted_multistate), unclass(reference_adjusted_multistate))
  expect_equal(attributes(adjusted_multistate), attributes(reference_adjusted_multistate))
  adjusted_counting_multistate <- aeqSurv(
    Surv(c(0, 0), c(1, 1 + 1e-8), factor(c("censor", "ill")), type = "counting"),
    tolerance = 1e-7
  )
  reference_adjusted_counting_multistate <- survival::aeqSurv(
    survival::Surv(c(0, 0), c(1, 1 + 1e-8), factor(c("censor", "ill")), type = "counting"),
    tolerance = 1e-7
  )
  expect_equal(unclass(adjusted_counting_multistate), unclass(reference_adjusted_counting_multistate))
  expect_equal(attributes(adjusted_counting_multistate), attributes(reference_adjusted_counting_multistate))
  expect_error(
    aeqSurv(
      Surv(c(0, 1), c(1, 1 + 1e-8), factor(c("censor", "ill")), type = "counting"),
      tolerance = 1e-7
    ),
    "effective length 0"
  )
  expect_error(aeqSurv(c(1, 2)), "Surv object")
  expect_error(aeqSurv(response, tolerance = Inf), "tolerance")

  split_data <- data.frame(
    time = c(5, 8),
    status = c(1, 0),
    group = c("a", "b"),
    x = c(10, 20)
  )
  right_split <- survSplit(
    Surv(time, status) ~ group + x,
    data = split_data,
    cut = c(3, 6),
    episode = "episode",
    id = "rowid"
  )
  expect_s3_class(right_split, "data.frame")
  expect_equal(names(right_split), c("group", "x", "rowid", "tstart", "time", "status", "episode"))
  expect_equal(right_split$group, c("a", "a", "b", "b", "b"))
  expect_equal(as.numeric(right_split$x), c(10, 10, 20, 20, 20))
  expect_equal(as.integer(right_split$rowid), c(1L, 1L, 2L, 2L, 2L))
  expect_equal(as.numeric(right_split$tstart), c(0, 3, 0, 3, 6))
  expect_equal(as.numeric(right_split$time), c(3, 5, 3, 6, 8))
  expect_equal(as.integer(right_split$status), c(0L, 1L, 0L, 0L, 0L))
  expect_equal(as.integer(right_split$episode), c(1L, 2L, 1L, 2L, 3L))
  split_factor_data <- transform(
    split_data,
    group = factor(group, levels = c("a", "b", "c")),
    ord = ordered(c("low", "high"), levels = c("low", "high")),
    visit = as.Date(c("2020-01-01", "2020-02-01")),
    stamp = as.POSIXct(c("2020-01-01 01:02:03", "2020-02-01 04:05:06"), tz = "UTC")
  )
  factor_split <- survSplit(
    Surv(time, status) ~ group + ord + visit + stamp,
    data = split_factor_data,
    cut = 3,
    episode = "episode",
    id = "rowid"
  )
  factor_split_formula <- Surv(time, status) ~ group + ord + visit + stamp
  environment(factor_split_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  reference_factor_split <- survival::survSplit(
    factor_split_formula,
    data = split_factor_data,
    cut = 3,
    episode = "episode",
    id = "rowid"
  )
  expect_equal(factor_split, reference_factor_split)

  split_counting <- data.frame(
    start = c(0, 2),
    stop = c(5, 8),
    status = c(1, 0),
    group = c("a", "b")
  )
  counting_split <- survSplit(
    Surv(start, stop, status) ~ group,
    data = split_counting,
    cut = c(3, 6),
    episode = "episode",
    id = "rowid"
  )
  expect_equal(names(counting_split), c("group", "rowid", "start", "stop", "status", "episode"))
  expect_equal(counting_split$group, c("a", "a", "b", "b", "b"))
  expect_equal(as.integer(counting_split$rowid), c(1L, 1L, 2L, 2L, 2L))
  expect_equal(as.numeric(counting_split$start), c(0, 3, 2, 3, 6))
  expect_equal(as.numeric(counting_split$stop), c(3, 5, 3, 6, 8))
  expect_equal(as.integer(counting_split$status), c(0L, 1L, 0L, 0L, 0L))
  expect_equal(as.integer(counting_split$episode), c(1L, 2L, 1L, 2L, 3L))

  check_data <- data.frame(
    id = c(1, 1, 2),
    start = c(0, 1, 0),
    stop = c(1, 2, 2),
    status = c(0, 1, 1)
  )
  checked <- survcheck(Surv(start, stop, status) ~ 1, data = check_data, id = id)
  expect_s3_class(checked, "survival_py_survcheck")
  expect_equal(checked$n_subjects, 2L)
  expect_equal(checked$n_transitions, 3L)
  expect_true(checked$is_valid)
  subset_check_data <- transform(check_data, keep = c(TRUE, TRUE, FALSE))
  subset_checked <- survcheck(Surv(start, stop, status) ~ 1, data = subset_check_data, id = id, subset = keep)
  expect_equal(subset_checked$n_subjects, 1L)
  expect_equal(subset_checked$n_transitions, 2L)

  overlap_data <- data.frame(
    id = c("a", "a"),
    start = c(0, 0.5),
    stop = c(1, 2),
    status = c(0, 1)
  )
  overlap_check <- survcheck(Surv(start, stop, status) ~ 1, data = overlap_data, id = id)
  expect_false(overlap_check$is_valid)
  expect_equal(overlap_check$overlap_ids, 1L)

  rtt_data <- data.frame(
    time = c(3, 1, 2),
    status = c(1, 0, 1),
    wt = c(1, 1, 1),
    id = c("c", "a", "b")
  )
  expect_equal(rttright(Surv(time, status) ~ 1, data = rtt_data), c(0.5, 0, 0.5))
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = rtt_data, renorm = FALSE),
    c(1.5, 0, 1.5)
  )
  expect_equal(rttright(Surv(time, status) ~ 1, data = rtt_data, weights = wt), c(0.5, 0, 0.5))
  expect_equal(rttright(Surv(time, status) ~ 1, data = rtt_data, id = id), c(0.5, 0, 0.5))
  subset_rtt_data <- transform(rtt_data, keep = c(TRUE, TRUE, FALSE))
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = subset_rtt_data, weights = wt, subset = keep),
    survival::rttright(survival::Surv(time, status) ~ 1, data = subset_rtt_data, weights = wt, subset = keep)
  )
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = rtt_data, times = 2),
    c(0.5, 0, 0.5)
  )
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = rtt_data, times = c(1, 2, 3)),
    matrix(
      c(1 / 3, 0.5, 0.5, 1 / 3, 0, 0, 1 / 3, 0.5, 0.5),
      nrow = 3,
      byrow = TRUE,
      dimnames = list(NULL, c("1", "2", "3"))
    )
  )
  repeated_id_rtt <- data.frame(time = c(1, 2, 3), status = c(0, 0, 1), id = c("a", "a", "b"))
  expect_error(rttright(Surv(time, status) ~ 1, data = repeated_id_rtt, id = id), "survcheck")

  grouped_rtt <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(0, 1, 0, 1),
    group = c("A", "A", "B", "B")
  )
  expect_equal(rttright(Surv(time, status) ~ group, data = grouped_rtt), c(0, 1, 0, 1))
  offset_grouped_rtt <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    group = c("treated", "treated", "control", "control"),
    off = c(1, 2, 3, 4)
  )
  expect_warning(
    bridged_offset_rtt <- rttright(Surv(time, status) ~ group + offset(off), data = offset_grouped_rtt),
    "Offset term ignored"
  )
  expect_warning(
    reference_offset_rtt <- survival::rttright(
      survival::Surv(time, status) ~ group + offset(off),
      data = offset_grouped_rtt
    ),
    "Offset term ignored"
  )
  expect_equal(bridged_offset_rtt, reference_offset_rtt)
  expect_equal(
    rttright(Surv(time, status) ~ group, data = grouped_rtt, times = c(1, 2, 3, 4)),
    matrix(
      c(0.5, 0, 0, 0, 0.5, 1, 1, 1, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 1),
      nrow = 4,
      byrow = TRUE,
      dimnames = list(NULL, c("1", "2", "3", "4"))
    )
  )

  counting_rtt <- data.frame(
    id = c("a", "a", "b", "b"),
    start = c(0, 1, 0, 2),
    stop = c(1, 3, 2, 4),
    status = c(0, 1, 0, 1)
  )
  expect_equal(
    rttright(Surv(start, stop, status) ~ 1, data = counting_rtt, id = id),
    survival::rttright(survival::Surv(start, stop, status) ~ 1, data = counting_rtt, id = id)
  )
  expect_equal(
    rttright(
      Surv(start, stop, status) ~ 1,
      data = counting_rtt,
      id = id,
      times = c(1, 2, 3, 4)
    ),
    survival::rttright(
      survival::Surv(start, stop, status) ~ 1,
      data = counting_rtt,
      id = id,
      times = c(1, 2, 3, 4)
    )
  )

  weighted_counting_rtt <- data.frame(
    id = c("a", "a", "b", "b", "c", "c"),
    start = c(0, 1, 0, 2, 0, 1.5),
    stop = c(1, 3, 2, 4, 1.5, 2.5),
    status = c(0, 1, 0, 1, 0, 0),
    wt = c(2, 2, 1, 1, 3, 3),
    group = c("x", "x", "y", "y", "x", "x")
  )
  expect_equal(
    rttright(
      Surv(start, stop, status) ~ group,
      data = weighted_counting_rtt,
      id = id,
      weights = wt,
      times = c(1, 2, 3, 4)
    ),
    survival::rttright(
      survival::Surv(start, stop, status) ~ group,
      data = weighted_counting_rtt,
      id = id,
      weights = wt,
      times = c(1, 2, 3, 4)
    )
  )

  state_connect <- matrix(
    c(0, 1, 0, 0),
    nrow = 2,
    byrow = TRUE,
    dimnames = list(c("a", "b"), c("a", "b"))
  )
  grDevices::pdf(NULL)
  reference_statefig <- survival::statefig(c(1, 1), state_connect)
  reference_statefig_coords <- survival::statefig(
    matrix(c(0.2, 0.7, 0.8, 0.3), nrow = 2, byrow = TRUE),
    state_connect,
    box = FALSE
  )
  grDevices::dev.off()
  expect_equal(statefig(c(1, 1), state_connect), reference_statefig)
  expect_equal(
    statefig(matrix(c(0.2, 0.7, 0.8, 0.3), nrow = 2, byrow = TRUE), state_connect, box = FALSE),
    reference_statefig_coords
  )
  expect_error(statefig("bad", state_connect), "layout")
  expect_error(statefig(c(1, 1), matrix(0, nrow = 1, ncol = 2)), "square")

  ridge_x <- c(1, 2, NA, 4)
  bridged_ridge <- ridge(ridge_x, theta = 2)
  reference_ridge <- survival::ridge(ridge_x, theta = 2)
  expect_equal(as.vector(bridged_ridge), as.vector(reference_ridge))
  expect_equal(dim(bridged_ridge), dim(reference_ridge))
  expect_equal(dimnames(bridged_ridge), dimnames(reference_ridge))
  expect_equal(class(bridged_ridge), class(reference_ridge))
  expect_equal(attr(bridged_ridge, "diag"), attr(reference_ridge, "diag"))
  expect_equal(attr(bridged_ridge, "cparm"), attr(reference_ridge, "cparm"))
  expect_equal(attr(bridged_ridge, "pparm"), attr(reference_ridge, "pparm"))
  expect_equal(attr(bridged_ridge, "varname"), attr(reference_ridge, "varname"))
  expect_equal(
    attr(bridged_ridge, "pfun")(c(0.2), 2, 3, attr(bridged_ridge, "pparm")),
    attr(reference_ridge, "pfun")(c(0.2), 2, 3, attr(reference_ridge, "pparm"))
  )

  bridged_unscaled_ridge <- ridge(c(1, 2, 3), theta = 2, scale = FALSE)
  reference_unscaled_ridge <- survival::ridge(c(1, 2, 3), theta = 2, scale = FALSE)
  expect_equal(
    attr(bridged_unscaled_ridge, "pfun")(c(0.2), 2, 3, attr(bridged_unscaled_ridge, "pparm")),
    attr(reference_unscaled_ridge, "pfun")(c(0.2), 2, 3, attr(reference_unscaled_ridge, "pparm"))
  )

  bridged_df_ridge <- ridge(ridge_x, df = 1.5, eps = 0.05)
  reference_df_ridge <- survival::ridge(ridge_x, df = 1.5, eps = 0.05)
  expect_equal(as.vector(bridged_df_ridge), as.vector(reference_df_ridge))
  expect_equal(dim(bridged_df_ridge), dim(reference_df_ridge))
  expect_equal(dimnames(bridged_df_ridge), dimnames(reference_df_ridge))
  expect_equal(class(bridged_df_ridge), class(reference_df_ridge))
  expect_equal(attr(bridged_df_ridge, "cargs"), attr(reference_df_ridge, "cargs"))
  expect_equal(attr(bridged_df_ridge, "cparm"), attr(reference_df_ridge, "cparm"))
  expect_equal(attr(bridged_df_ridge, "pparm"), attr(reference_df_ridge, "pparm"))
  expect_equal(attr(bridged_df_ridge, "varname"), attr(reference_df_ridge, "varname"))
  expect_error(ridge(c(1, 2, 3), theta = 1, df = 1), "Only one of df or theta")
  frailty_x <- factor(c("a", "a", "b", "c"))
  expect_frailty_equal <- function(bridged, reference) {
    expect_equal(class(bridged), class(reference))
    expect_equal(as.vector(bridged), as.vector(reference))
    expect_equal(levels(bridged), levels(reference))
    expect_equal(attr(bridged, "contrasts"), attr(reference, "contrasts"))
    for (name in c("diag", "sparse", "cargs", "cparm", "pparm", "varname")) {
      expect_equal(attr(bridged, name), attr(reference, name))
    }
    expect_true(is.function(attr(bridged, "pfun")))
    expect_true(is.function(attr(bridged, "printfun")))
    if (!is.null(attr(reference, "cfun"))) {
      expect_true(is.function(attr(bridged, "cfun")))
    }

    ncoef <- if (is.factor(reference)) {
      length(levels(reference))
    } else {
      max(as.integer(reference), na.rm = TRUE)
    }
    coef <- seq_len(ncoef) / 10
    if (is.null(attr(reference, "pparm"))) {
      expect_equal(
        attr(bridged, "pfun")(coef, 0.5, 3),
        attr(reference, "pfun")(coef, 0.5, 3)
      )
    } else {
      expect_equal(
        attr(bridged, "pfun")(coef, 0.5, 3, attr(bridged, "pparm")),
        attr(reference, "pfun")(coef, 0.5, 3, attr(reference, "pparm"))
      )
    }
    history <- list(theta = 0.5, c.loglik = -2)
    expect_equal(
      attr(bridged, "printfun")(coef, diag(ncoef), diag(ncoef), 2, history),
      attr(reference, "printfun")(coef, diag(ncoef), diag(ncoef), 2, history)
    )
  }
  for (name in c("frailty", "frailty.gamma", "frailty.gaussian", "frailty.t")) {
    expect_frailty_equal(
      do.call(get(name), list(frailty_x, theta = 0.5)),
      do.call(getExportedValue("survival", name), list(frailty_x, theta = 0.5))
    )
  }
  expect_frailty_equal(
    frailty.gamma(factor(letters[1:6]), theta = 0.5),
    survival::frailty.gamma(factor(letters[1:6]), theta = 0.5)
  )
  expect_frailty_equal(
    frailty.gamma(frailty_x, df = 2),
    survival::frailty.gamma(frailty_x, df = 2)
  )
  expect_frailty_equal(
    frailty.gaussian(frailty_x),
    survival::frailty.gaussian(frailty_x)
  )
  expect_frailty_equal(
    frailty.t(frailty_x, df = 2),
    survival::frailty.t(frailty_x, df = 2)
  )
  expect_pspline_equal <- function(bridged, reference) {
    expect_equal(matrix(as.numeric(bridged), nrow = nrow(bridged)), matrix(as.numeric(reference), nrow = nrow(reference)))
    expect_equal(class(bridged), class(reference))
    for (name in c("diag", "cargs", "cparm", "pparm", "varname", "intercept", "nterm", "degree", "df", "Boundary.knots", "combine")) {
      reference_attr <- attr(reference, name)
      if (!is.null(reference_attr)) {
        expect_equal(attr(bridged, name), reference_attr)
      }
    }
    if (!is.null(attr(reference, "pfun"))) {
      coef <- seq_len(ncol(bridged)) / 10
      expect_equal(
        attr(bridged, "pfun")(coef, 0.5, nrow(bridged), attr(bridged, "pparm")),
        attr(reference, "pfun")(coef, 0.5, nrow(reference), attr(reference, "pparm"))
      )
      expect_true(is.function(attr(bridged, "printfun")))
      expect_equal(
        formals(attr(bridged, "printfun"))$cbase,
        formals(attr(reference, "printfun"))$cbase
      )
    }
  }
  expect_pspline_equal(pspline(1:5, df = 3), survival::pspline(1:5, df = 3))
  expect_pspline_equal(pspline(1:5, theta = 0.5), survival::pspline(1:5, theta = 0.5))
  expect_pspline_equal(pspline(1:5, df = 0), survival::pspline(1:5, df = 0))
  expect_pspline_equal(
    pspline(c(0, 1, 5, 6), df = 3, Boundary.knots = c(1, 5), penalty = FALSE),
    survival::pspline(c(0, 1, 5, 6), df = 3, Boundary.knots = c(1, 5), penalty = FALSE)
  )
  expect_pspline_equal(
    pspline(1:5, df = 3, intercept = TRUE, penalty = FALSE),
    survival::pspline(1:5, df = 3, intercept = TRUE, penalty = FALSE)
  )

  strata_factor <- strata(c("b", "a", "b", NA), c(2, 1, 1, 1), shortlabel = TRUE)
  expect_s3_class(strata_factor, "factor")
  expect_equal(as.integer(strata_factor), c(3L, 1L, 2L, NA))
  expect_equal(levels(strata_factor), c("a, 1", "b, 1", "b, 2"))
  named_strata <- strata(x = c("b", "a", "b", NA), y = c(2, 1, 1, 1))
  expect_equal(levels(named_strata), c("x=a, y=1", "x=b, y=1", "x=b, y=2"))
  expect_equal(
    as.integer(strata(c("b", "a", "b", NA), c(2, 1, 1, 1), na.group = TRUE)),
    c(3L, 1L, 2L, 4L)
  )
  expect_equal(cluster(data$group), survival::cluster(data$group))
  expect_equal(
    levels(strata(x = c("b", "a", "b", NA), y = c(2, 1, 1, 1), sep = "|")),
    c("x=a|y=1", "x=b|y=1", "x=b|y=2")
  )
  specials_terms <- terms(
    y ~ x + strata(group) + cluster(id) + x:strata(group),
    specials = c("strata", "cluster")
  )
  expect_equal(
    untangle.specials(specials_terms, "strata"),
    survival::untangle.specials(specials_terms, "strata")
  )
  expect_equal(
    untangle.specials(specials_terms, "cluster"),
    survival::untangle.specials(specials_terms, "cluster")
  )
  expect_equal(
    untangle.specials(specials_terms, "strata", order = 2),
    survival::untangle.specials(specials_terms, "strata", order = 2)
  )
  assign_data <- data.frame(x = c(1, 2, 3), group = factor(c("a", "b", "a")))
  assign_terms <- terms(~ x + group, data = assign_data)
  assign_matrix <- model.matrix(assign_terms, assign_data)
  expect_equal(attrassign(assign_matrix, assign_terms), survival::attrassign(assign_matrix, assign_terms))
  assign_fit <- stats::lm(x ~ group, data = assign_data)
  expect_equal(attrassign(assign_fit), survival::attrassign(assign_fit))

  cox_control <- coxph.control(iter.max = 0, eps = 1e-05, toler.chol = 1e-08, timefix = FALSE)
  expect_named(cox_control, c("eps", "toler.chol", "iter.max", "toler.inf", "outer.max", "timefix"))
  expect_equal(cox_control[["iter.max"]], 0L)
  expect_false(cox_control[["timefix"]])
  expect_error(coxph.control(iter.max = -1), "iter.max")
  cox_fit_data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6),
    status = c(1, 1, 0, 1, 0, 1),
    x = c(0.2, 0.4, 0.1, 0.8, 1.0, 1.2),
    group = c(1, 1, 1, 2, 2, 2)
  )
  cox_fit_x <- stats::model.matrix(~ x, data = cox_fit_data)[, -1, drop = FALSE]
  cox_fit_y <- survival::Surv(cox_fit_data$time, cox_fit_data$status)
  bridged_cox_fit <- coxph.fit(
    cox_fit_x,
    cox_fit_y,
    strata = NULL,
    offset = rep(0, nrow(cox_fit_x)),
    init = NULL,
    control = coxph.control(iter.max = 20, eps = 1e-09),
    weights = rep(1, nrow(cox_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(cox_fit_x)))
  )
  reference_cox_fit <- survival::coxph.fit(
    cox_fit_x,
    cox_fit_y,
    strata = NULL,
    offset = rep(0, nrow(cox_fit_x)),
    init = NULL,
    control = survival::coxph.control(iter.max = 20, eps = 1e-09),
    weights = rep(1, nrow(cox_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(cox_fit_x)))
  )
  expect_equal(names(bridged_cox_fit), names(reference_cox_fit))
  expect_equal(bridged_cox_fit$coefficients, reference_cox_fit$coefficients, tolerance = 1e-5)
  expect_equal(bridged_cox_fit$var, reference_cox_fit$var, tolerance = 1e-5)
  expect_equal(bridged_cox_fit$loglik, reference_cox_fit$loglik, tolerance = 1e-6)
  expect_equal(bridged_cox_fit$score, reference_cox_fit$score, tolerance = 1e-6)
  expect_equal(bridged_cox_fit$linear.predictors, reference_cox_fit$linear.predictors, tolerance = 1e-5)
  expect_equal(bridged_cox_fit$residuals, reference_cox_fit$residuals, tolerance = 1e-6)
  expect_equal(bridged_cox_fit$means, reference_cox_fit$means, tolerance = 1e-12)
  expect_equal(bridged_cox_fit$method, reference_cox_fit$method)

  bridged_stratified_cox_fit <- coxph.fit(
    cox_fit_x,
    cox_fit_y,
    strata = cox_fit_data$group,
    offset = rep(0, nrow(cox_fit_x)),
    init = c(0),
    control = coxph.control(iter.max = 0),
    weights = rep(1, nrow(cox_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(cox_fit_x))),
    resid = FALSE
  )
  reference_stratified_cox_fit <- survival::coxph.fit(
    cox_fit_x,
    cox_fit_y,
    strata = cox_fit_data$group,
    offset = rep(0, nrow(cox_fit_x)),
    init = c(0),
    control = survival::coxph.control(iter.max = 0),
    weights = rep(1, nrow(cox_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(cox_fit_x))),
    resid = FALSE
  )
  expect_false("residuals" %in% names(bridged_stratified_cox_fit))
  expect_equal(bridged_stratified_cox_fit$loglik, reference_stratified_cox_fit$loglik, tolerance = 1e-12)
  expect_equal(bridged_stratified_cox_fit$score, reference_stratified_cox_fit$score, tolerance = 1e-6)

  agreg_fit_data <- data.frame(
    start = c(0, 0, 1, 2, 3, 4),
    stop = c(2, 3, 4, 5, 6, 7),
    status = c(1, 0, 1, 1, 0, 1),
    x = c(0.2, 0.4, 0.1, 0.8, 1.0, 1.2)
  )
  agreg_fit_x <- stats::model.matrix(~ x, data = agreg_fit_data)[, -1, drop = FALSE]
  agreg_fit_y <- survival::Surv(agreg_fit_data$start, agreg_fit_data$stop, agreg_fit_data$status)
  bridged_agreg_fit <- agreg.fit(
    agreg_fit_x,
    agreg_fit_y,
    strata = NULL,
    offset = rep(0, nrow(agreg_fit_x)),
    init = c(0),
    control = coxph.control(iter.max = 0),
    weights = rep(1, nrow(agreg_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(agreg_fit_x)))
  )
  reference_agreg_fit <- survival::agreg.fit(
    agreg_fit_x,
    agreg_fit_y,
    strata = NULL,
    offset = rep(0, nrow(agreg_fit_x)),
    init = c(0),
    control = survival::coxph.control(iter.max = 0),
    weights = rep(1, nrow(agreg_fit_x)),
    method = "breslow",
    rownames = as.character(seq_len(nrow(agreg_fit_x)))
  )
  expect_equal(names(bridged_agreg_fit), names(reference_agreg_fit))
  expect_equal(bridged_agreg_fit$coefficients, reference_agreg_fit$coefficients, tolerance = 1e-12)
  expect_equal(bridged_agreg_fit$var, reference_agreg_fit$var, tolerance = 1e-6)
  expect_equal(bridged_agreg_fit$loglik, reference_agreg_fit$loglik, tolerance = 1e-12)
  expect_equal(bridged_agreg_fit$score, reference_agreg_fit$score, tolerance = 1e-6)
  expect_equal(bridged_agreg_fit$residuals, reference_agreg_fit$residuals, tolerance = 1e-12)
  expect_equal(bridged_agreg_fit$means, reference_agreg_fit$means, tolerance = 1e-12)
  expect_equal(bridged_agreg_fit$first, reference_agreg_fit$first, tolerance = 1e-12)
  expect_equal(bridged_agreg_fit$info, reference_agreg_fit$info)

  bridged_agexact_fit <- agexact.fit(
    agreg_fit_x,
    agreg_fit_y,
    strata = NULL,
    offset = rep(0, nrow(agreg_fit_x)),
    init = NULL,
    control = coxph.control(iter.max = 20, eps = 1e-09),
    weights = rep(1, nrow(agreg_fit_x)),
    method = "exact",
    rownames = as.character(seq_len(nrow(agreg_fit_x)))
  )
  reference_agexact_fit <- survival::agexact.fit(
    agreg_fit_x,
    agreg_fit_y,
    strata = NULL,
    offset = rep(0, nrow(agreg_fit_x)),
    init = NULL,
    control = survival::coxph.control(iter.max = 20, eps = 1e-09),
    weights = rep(1, nrow(agreg_fit_x)),
    method = "exact",
    rownames = as.character(seq_len(nrow(agreg_fit_x)))
  )
  expect_equal(names(bridged_agexact_fit), names(reference_agexact_fit))
  expect_equal(bridged_agexact_fit$coefficients, reference_agexact_fit$coefficients, tolerance = 1e-4)
  expect_equal(bridged_agexact_fit$var, reference_agexact_fit$var, tolerance = 1e-5)
  expect_equal(bridged_agexact_fit$loglik, reference_agexact_fit$loglik, tolerance = 1e-6)
  expect_equal(bridged_agexact_fit$score, reference_agexact_fit$score, tolerance = 1e-6)
  expect_equal(bridged_agexact_fit$linear.predictors, reference_agexact_fit$linear.predictors, tolerance = 1e-4)
  expect_equal(bridged_agexact_fit$residuals, reference_agexact_fit$residuals, tolerance = 1e-5)
  expect_equal(bridged_agexact_fit$means, reference_agexact_fit$means, tolerance = 1e-12)
  expect_equal(bridged_agexact_fit$method, reference_agexact_fit$method)
  expect_error(
    agexact.fit(
      agreg_fit_x,
      agreg_fit_y,
      strata = NULL,
      offset = rep(0, nrow(agreg_fit_x)),
      init = NULL,
      control = coxph.control(iter.max = 0),
      weights = c(1, rep(2, nrow(agreg_fit_x) - 1L)),
      method = "exact",
      rownames = as.character(seq_len(nrow(agreg_fit_x)))
    ),
    "Case weights are not supported"
  )
  expect_equal(coxph.wtest(diag(2), c(1, 2)), survival::coxph.wtest(diag(2), c(1, 2)))
  expect_equal(
    coxph.wtest(matrix(c(2, 0.5, 0.5, 1), 2), c(1, 2)),
    survival::coxph.wtest(matrix(c(2, 0.5, 0.5, 1), 2), c(1, 2)),
    tolerance = 1e-12
  )
  expect_equal(
    coxph.wtest(matrix(c(1, 2, 2, 4), 2), c(1, 2)),
    survival::coxph.wtest(matrix(c(1, 2, 2, 4), 2), c(1, 2)),
    tolerance = 1e-12
  )
  expect_equal(
    coxph.wtest(diag(2), matrix(c(1, 2, 3, 4), nrow = 2)),
    survival::coxph.wtest(diag(2), matrix(c(1, 2, 3, 4), nrow = 2))
  )
  expect_equal(coxph.wtest(diag(2), c(NA, 2)), survival::coxph.wtest(diag(2), c(NA, 2)))

  survreg_control <- survreg.control(maxiter = 1, rel.tolerance = 1e-05, toler.chol = 1e-08)
  expect_named(survreg_control, c("iter.max", "rel.tolerance", "toler.chol", "debug", "maxiter", "outer.max"))
  expect_equal(survreg_control[["iter.max"]], 1L)
  expect_equal(survreg_control[["maxiter"]], 1L)
  expect_error(survreg.control(rel.tolerance = 0), "rel.tolerance")

  expect_equal(
    dsurvreg(c(1, 2), mean = 0, scale = 1, distribution = "t", parms = 5),
    c(0.2196798, 0.06509031),
    tolerance = 1e-7
  )
  expect_equal(
    psurvreg(c(1, 2), mean = 0, scale = 1, distribution = "t", parms = 5),
    c(0.8183913, 0.9490303),
    tolerance = 1e-7
  )
  expect_equal(
    qsurvreg(c(0.25, 0.5), mean = 0, scale = 1, distribution = "t", parms = 5),
    c(-0.7266868, 0),
    tolerance = 1e-7
  )
  expect_equal(
    dsurvreg(1, mean = 0, distribution = "gaussian", parms = 5),
    dsurvreg(1, mean = 0, distribution = "gaussian"),
    tolerance = 1e-12
  )

  km <- survfit(Surv(time, status) ~ group, data = data)
  expect_s3_class(km, "survival_py_survfit")
  km_direct <- survfit.formula(Surv(time, status) ~ group, data = data)
  expect_equal(as.data.frame(km_direct), as.data.frame(km))
  km_from_string <- survfit("Surv(time, status) ~ group", data = data)
  expect_s3_class(km_from_string, "survival_py_survfit")
  km_from_string_id <- survfit("Surv(time, status) ~ 1", data = data, id = seq_along(time), model = TRUE)
  reference_km_id <- getFromNamespace("survfit.formula", "survival")(
    survival::Surv(time, status) ~ 1,
    data = data,
    id = seq_along(time),
    model = TRUE
  )
  expect_equal(
    pseudo(km_from_string_id, times = 2),
    survival::pseudo(reference_km_id, times = 2),
    tolerance = 1e-8
  )
  km_model_frame <- model.frame(km_from_string_id)
  expect_s3_class(km_model_frame, "data.frame")
  expect_equal(km_model_frame$time, data$time)
  expect_equal(km_model_frame$status, data$status)
  grouped_model_fit <- survfit("Surv(time, status) ~ group", data = data, model = TRUE)
  reference_grouped_model_fit <- getFromNamespace("survfit.formula", "survival")(
    survival::Surv(time, status) ~ group,
    data = data,
    model = TRUE
  )
  grouped_model_frame <- model.frame(grouped_model_fit)
  expect_equal(grouped_model_frame$time, data$time)
  expect_equal(grouped_model_frame$status, data$status)
  expect_equal(grouped_model_frame$group, data$group)
  direct_model_fit <- survfit(response, model = TRUE)
  reference_direct_model_fit <- getFromNamespace("survfit.formula", "survival")(
    survival::Surv(time, status) ~ 1,
    data = data,
    model = TRUE
  )
  direct_model_frame <- model.frame(direct_model_fit)
  expect_equal(direct_model_frame$time, data$time)
  expect_equal(direct_model_frame$status, data$status)
  km_from_string_frame <- model.frame(km_from_string)
  expect_equal(km_from_string_frame$time, data$time)
  expect_equal(km_from_string_frame$status, data$status)
  expect_equal(km_from_string_frame$group, data$group)
  km_from_response <- survfit(response)
  expect_s3_class(km_from_response, "survival_py_survfit")
  expect_equal(
    quantile(km_from_response, probs = c(0.25, 0.5), conf.int = FALSE),
    c(`25` = 1.5, `50` = 3.0)
  )
  expect_equal(median(km_from_response), c(`50` = 3.0))
  expect_error(coef(km_from_response), "coef method not applicable")
  expect_error(vcov(km_from_response), "vcov method not applicable")
  expect_error(confint(km_from_response), "confint method not defined")
  expect_error(residuals(km_from_response), "times argument")
  expect_equal(
    residuals(km_from_response, times = 2),
    stats::residuals(reference_direct_model_fit, times = 2),
    tolerance = 1e-12
  )
  expect_equal(
    residuals(direct_model_fit, times = c(1, 3), type = "survival"),
    stats::residuals(reference_direct_model_fit, times = c(1, 3), type = "survival"),
    tolerance = 1e-12
  )
  expect_equal(
    residuals(direct_model_fit, times = c(1, 3), type = "cumhaz", data.frame = TRUE),
    stats::residuals(reference_direct_model_fit, times = c(1, 3), type = "cumhaz", data.frame = TRUE),
    tolerance = 1e-12
  )
  expect_equal(
    residuals(direct_model_fit, times = c(1, 3), type = "auc"),
    stats::residuals(reference_direct_model_fit, times = c(1, 3), type = "auc"),
    tolerance = 1e-12
  )
  expect_equal(
    residuals(grouped_model_fit, times = c(1, 3), type = "survival"),
    stats::residuals(reference_grouped_model_fit, times = c(1, 3), type = "survival"),
    tolerance = 1e-12
  )
  grouped_residual_extra <- residuals(grouped_model_fit, times = c(1, 3), type = "survival", extra = TRUE)
  reference_grouped_residual_extra <- stats::residuals(
    reference_grouped_model_fit,
    times = c(1, 3),
    type = "survival",
    extra = TRUE
  )
  expect_equal(grouped_residual_extra$resid, reference_grouped_residual_extra$resid, tolerance = 1e-12)
  expect_equal(grouped_residual_extra$curve, reference_grouped_residual_extra$curve)
  expect_equal(quantile(km_from_response, probs = c(0.25, 0.5), scale = 2)$quantile, c(`25` = 0.75, `50` = 1.5))
  km_from_response_frame <- as.data.frame(km_from_response)
  expect_equal(names(km_from_response), names(km_from_response_frame))
  expect_equal(length(km_from_response), ncol(km_from_response_frame))
  expect_null(dim(km_from_response))
  expect_equal(as.list(km_from_response), as.list(km_from_response_frame))
  expect_s3_class(km_from_response[1], "survival_py_survfit")
  expect_equal(as.data.frame(km_from_response[1]), km_from_response_frame)
  expect_error(km_from_response[2], "subscript out of bounds")
  expect_equal(km_from_response[[1L]], km_from_response_frame[[1L]])
  expect_equal(km_from_response[["n.risk"]], km_from_response_frame[["n.risk"]])
  expect_equal(km_from_response[["n_risk"]], km_from_response_frame[["n.risk"]])
  expect_equal(km_from_response$n.risk, km_from_response_frame[["n.risk"]])
  expect_equal(km_from_response$conf_upper, km_from_response_frame[["upper"]])
  grouped_from_response <- survfit(response, group = data$group, se.fit = FALSE)
  expect_s3_class(grouped_from_response, "survival_py_survfit")
  expect_equal(names(grouped_from_response), c("control", "treated"))
  expect_equal(length(grouped_from_response), 2L)
  expect_equal(dim(grouped_from_response), c(strata = 2L))
  grouped_quantile <- quantile(grouped_from_response, probs = c(0.25, 0.5))
  expect_true(is.matrix(grouped_quantile))
  expect_equal(rownames(grouped_quantile), c("control", "treated"))
  expect_equal(colnames(grouped_quantile), c("25", "50"))
  expect_equal(grouped_quantile, matrix(
    c(1, 4, 1.5, 4),
    nrow = 2,
    dimnames = list(c("control", "treated"), c("25", "50"))
  ))
  expect_equal(median(grouped_from_response), grouped_quantile[, "50", drop = FALSE])
  grouped_no_se_frame <- as.data.frame(grouped_from_response)
  expect_s3_class(grouped_no_se_frame, "data.frame")
  expect_false(any(c("std.err", "lower", "upper", "std.chaz") %in% names(grouped_no_se_frame)))
  expect_true(all(vapply(grouped_no_se_frame, length, integer(1)) == nrow(grouped_no_se_frame)))
  control_curve <- grouped_from_response[1]
  expect_s3_class(control_curve, "survival_py_survfit")
  expect_null(dim(control_curve))
  control_frame <- grouped_no_se_frame[grouped_no_se_frame$strata == "control", setdiff(names(grouped_no_se_frame), "strata")]
  rownames(control_frame) <- NULL
  expect_equal(as.data.frame(control_curve), control_frame)
  control_curve_list <- grouped_from_response[1, drop = FALSE]
  expect_s3_class(control_curve_list, "survival_py_survfit")
  expect_equal(names(control_curve_list), "control")
  expect_equal(dim(control_curve_list), c(strata = 1L))
  treated_curve <- grouped_from_response["treated"]
  expect_s3_class(treated_curve, "survival_py_survfit")
  treated_frame <- grouped_no_se_frame[grouped_no_se_frame$strata == "treated", setdiff(names(grouped_no_se_frame), "strata")]
  rownames(treated_frame) <- NULL
  expect_equal(as.data.frame(treated_curve), treated_frame)
  expect_error(grouped_from_response["missing"], "strata missing not matched")
  omitted_direct <- survfit(
    response,
    group = c("control", NA, "treated", "treated"),
    subset = c(TRUE, TRUE, TRUE, FALSE),
    na.action = stats::na.omit
  )
  omitted_manual <- survfit(
    Surv(data$time[c(1, 3)], data$status[c(1, 3)]),
    group = data$group[c(1, 3)]
  )
  expect_equal(as.data.frame(omitted_direct), as.data.frame(omitted_manual))
  km_frame <- as.data.frame(km)
  expect_s3_class(km_frame, "data.frame")
  expect_true(all(c("strata", "time", "surv") %in% names(km_frame)))
  response_frame <- as.data.frame(km_from_response)
  expect_true(all(c("time", "surv") %in% names(response_frame)))
  km_summary <- summary(km)
  expect_s3_class(km_summary, "summary.survival_py_survfit")
  expect_s3_class(km_summary, "data.frame")
  expect_true(all(c("strata", "time", "surv") %in% names(km_summary)))
  expect_false(any(km_summary$n.event == 0))
  expect_equal(as.data.frame(summary(km, censored = TRUE)), km_frame)
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  summary_frame <- function(value) {
    frame <- data.frame(time = value$time)
    if (!is.null(value$strata)) {
      frame$strata <- sub("^group=", "", as.character(value$strata))
    }
    for (column in c("n.risk", "n.event", "n.censor", "surv", "cumhaz", "std.err", "lower", "upper", "std.chaz")) {
      if (!is.null(value[[column]])) {
        frame[[column]] <- value[[column]]
      }
    }
    frame[c(intersect(c("strata", "time"), names(frame)), setdiff(names(frame), c("strata", "time")))]
  }
  reference_km <- reference_survfit(survival::Surv(time, status) ~ group, data = data)
  grouped_summary_times <- c(1, 3, 6)
  expect_equal(
    as.data.frame(summary(km, times = grouped_summary_times, extend = TRUE))[
      names(summary_frame(summary(reference_km, times = grouped_summary_times, extend = TRUE)))
    ],
    summary_frame(summary(reference_km, times = grouped_summary_times, extend = TRUE)),
    tolerance = 1e-8
  )
  reference_direct_km <- reference_survfit(survival::Surv(time, status) ~ 1, data = data)
  direct_summary_times <- c(0, 1.5, 3, 6)
  expect_equal(
    as.data.frame(summary(km_from_response, times = direct_summary_times, extend = TRUE, scale = 2))[
      names(summary_frame(summary(reference_direct_km, times = direct_summary_times, extend = TRUE, scale = 2)))
    ],
    summary_frame(summary(reference_direct_km, times = direct_summary_times, extend = TRUE, scale = 2)),
    tolerance = 1e-8
  )
  expect_true(any(grepl("time", capture.output(print(km)), fixed = TRUE)))
  survfitkm_response <- Surv(data$time, data$status)
  reference_survfitkm_response <- survival::Surv(data$time, data$status)
  expect_survfitkm_equal <- function(bridged, reference, tolerance = 1e-8) {
    for (name in setdiff(intersect(names(reference), names(bridged)), c("lower", "upper"))) {
      expect_equal(bridged[[name]], reference[[name]], tolerance = tolerance)
    }
    for (name in intersect(c("lower", "upper"), intersect(names(reference), names(bridged)))) {
      actual <- bridged[[name]]
      expected <- reference[[name]]
      common <- seq_len(min(length(actual), length(expected)))
      comparable <- common[!is.na(actual[common]) & !is.na(expected[common])]
      expect_equal(actual[comparable], expected[comparable], tolerance = tolerance)
      if (any(is.na(expected[common]))) {
        expected_na <- common[is.na(expected[common])]
        expect_true(all(is.na(actual[expected_na]) | abs(actual[expected_na]) <= tolerance))
      }
      if (any(is.na(actual[common]))) {
        actual_na <- common[is.na(actual[common])]
        expect_true(all(is.na(expected[actual_na]) | abs(expected[actual_na]) <= tolerance))
      }
      if (length(actual) > length(expected)) {
        extra <- actual[(length(expected) + 1L):length(actual)]
        expect_true(all(is.na(extra) | abs(extra) <= tolerance))
      }
      if (length(expected) > length(actual)) {
        extra <- expected[(length(actual) + 1L):length(expected)]
        expect_true(all(is.na(extra) | abs(extra) <= tolerance))
      }
    }
  }
  survfitkm_cases <- list(
    default = list(x = factor(rep(1L, nrow(data)))),
    grouped = list(x = factor(data$group, levels = c("control", "treated", "empty"))),
    no_se = list(x = factor(rep(1L, nrow(data))), se.fit = FALSE),
    fh2 = list(x = factor(rep(1L, nrow(data))), type = "fh2"),
    peto = list(x = factor(rep(1L, nrow(data))), conf.lower = "peto"),
    modified = list(x = factor(rep(1L, nrow(data))), conf.lower = "modified"),
    weighted = list(
      x = factor(rep(1L, nrow(data))),
      weights = c(1, 2, 1.5, 0.5)
    ),
    grouped_weighted = list(
      x = factor(data$group, levels = c("control", "treated", "empty")),
      weights = c(1, 2, 1.5, 0.5)
    ),
    grouped_modified = list(
      x = factor(data$group, levels = c("control", "treated", "empty")),
      conf.lower = "modified"
    )
  )
  for (case in survfitkm_cases) {
    bridged_survfitkm <- do.call(survfitKM, c(list(y = survfitkm_response), case))
    reference_call <- list(y = reference_survfitkm_response)
    reference_call <- c(reference_call, case)
    reference_survfitkm <- if (identical(case$conf.lower, "modified")) {
      suppressWarnings(do.call(survival::survfitKM, reference_call))
    } else {
      do.call(survival::survfitKM, reference_call)
    }
    expect_survfitkm_equal(bridged_survfitkm, reference_survfitkm)
  }
  influence_survfitkm_response <- Surv(c(1, 2, 3, 4), c(1, 0, 1, 0))
  reference_influence_survfitkm_response <- survival::Surv(c(1, 2, 3, 4), c(1, 0, 1, 0))
  for (influence_value in list(1L, 2L, 3L, TRUE)) {
    expect_survfitkm_equal(
      survfitKM(
        factor(rep(1L, 4)),
        influence_survfitkm_response,
        influence = influence_value
      ),
      survival::survfitKM(
        factor(rep(1L, 4)),
        reference_influence_survfitkm_response,
        influence = influence_value
      )
    )
  }
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 4)),
      influence_survfitkm_response,
      type = "fleming-harrington",
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 4)),
      reference_influence_survfitkm_response,
      type = "fleming-harrington",
      influence = 3L
    )
  )
  expect_survfitkm_equal(
    survfitKM(
      factor(c("a", "a", "b", "b")),
      influence_survfitkm_response,
      influence = 3L
    ),
    survival::survfitKM(
      factor(c("a", "a", "b", "b")),
      reference_influence_survfitkm_response,
      influence = 3L
    )
  )
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 4)),
      influence_survfitkm_response,
      cluster = c("z", "z", "a", "b"),
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 4)),
      reference_influence_survfitkm_response,
      cluster = c("z", "z", "a", "b"),
      influence = 3L
    )
  )
  expect_warning(
    expect_survfitkm_equal(
      survfitKM(
        factor(rep(1L, 4)),
        influence_survfitkm_response,
        influence = 3L,
        robust = FALSE
      ),
      suppressWarnings(survival::survfitKM(
        factor(rep(1L, 4)),
        reference_influence_survfitkm_response,
        influence = 3L,
        robust = FALSE
      ))
    ),
    "robust=FALSE implies influence=FALSE"
  )
  tied_influence_survfitkm_response <- Surv(
    c(1, 1, 1, 2, 2, 3),
    c(1, 1, 0, 1, 1, 0)
  )
  reference_tied_influence_survfitkm_response <- survival::Surv(
    c(1, 1, 1, 2, 2, 3),
    c(1, 1, 0, 1, 1, 0)
  )
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 6)),
      tied_influence_survfitkm_response,
      stype = 1L,
      ctype = 2L,
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 6)),
      reference_tied_influence_survfitkm_response,
      stype = 1L,
      ctype = 2L,
      influence = 3L
    )
  )
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 6)),
      tied_influence_survfitkm_response,
      type = "fh2",
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 6)),
      reference_tied_influence_survfitkm_response,
      type = "fh2",
      influence = 3L
    )
  )
  counting_survfitkm_response <- Surv(
    c(0, 10, 25, 0, 5),
    c(10, 20, 30, 15, 25),
    c(0, 0, 1, 1, 0)
  )
  reference_counting_survfitkm_response <- survival::Surv(
    c(0, 10, 25, 0, 5),
    c(10, 20, 30, 15, 25),
    c(0, 0, 1, 1, 0)
  )
  counting_survfitkm_id <- c("a", "a", "a", "b", "c")
  for (counting_x in list(
    factor(rep(1L, 5)),
    factor(c("A", "A", "A", "A", "B"), levels = c("A", "B", "empty"))
  )) {
    expect_survfitkm_equal(
      survfitKM(
        counting_x,
        counting_survfitkm_response,
        id = counting_survfitkm_id,
        entry = TRUE
      ),
      survival::survfitKM(
        counting_x,
        reference_counting_survfitkm_response,
        id = counting_survfitkm_id,
        entry = TRUE
      )
    )
  }
  for (influence_value in list(1L, 2L, 3L, TRUE)) {
    expect_survfitkm_equal(
      survfitKM(
        factor(rep(1L, 5)),
        counting_survfitkm_response,
        id = counting_survfitkm_id,
        entry = TRUE,
        influence = influence_value
      ),
      survival::survfitKM(
        factor(rep(1L, 5)),
        reference_counting_survfitkm_response,
        id = counting_survfitkm_id,
        entry = TRUE,
        influence = influence_value
      )
    )
  }
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 5)),
      counting_survfitkm_response,
      id = counting_survfitkm_id,
      entry = TRUE,
      type = "fleming-harrington",
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 5)),
      reference_counting_survfitkm_response,
      id = counting_survfitkm_id,
      entry = TRUE,
      type = "fleming-harrington",
      influence = 3L
    )
  )
  expect_survfitkm_equal(
    survfitKM(
      factor(rep(1L, 5)),
      counting_survfitkm_response,
      id = counting_survfitkm_id,
      entry = TRUE,
      stype = 1L,
      ctype = 2L,
      influence = 3L
    ),
    survival::survfitKM(
      factor(rep(1L, 5)),
      reference_counting_survfitkm_response,
      id = counting_survfitkm_id,
      entry = TRUE,
      stype = 1L,
      ctype = 2L,
      influence = 3L
    )
  )
  expect_error(survfitKM(data$x, survfitkm_response), "x must be a factor")
  reference_survfit_confint_impl <- get("survfit_confint", envir = asNamespace("survival"))
  reference_survfit_confint <- function(...) {
    args <- list(...)
    if (is.null(args$conf.int)) {
      args$conf.int <- 0.95
    }
    do.call(reference_survfit_confint_impl, args)
  }
  for (conf_type in c("plain", "log", "log-log", "logit", "arcsin")) {
    actual_confint <- survfit_confint(c(0.2, 0.5, 0.9), 0.1, conf.type = conf_type)
    expected_confint <- reference_survfit_confint(c(0.2, 0.5, 0.9), 0.1, conf.type = conf_type)
    # survival 3.8-3 vectorized these variants; current survival and the bridge return the first interval.
    if (conf_type %in% c("log", "log-log", "logit") &&
      length(actual_confint$lower) == 1L &&
      length(expected_confint$lower) > 1L) {
      expected_confint <- lapply(expected_confint, function(value) value[seq_along(actual_confint$lower)])
    }
    expect_equal(actual_confint, expected_confint, tolerance = 1e-12)
  }
  expect_equal(
    survfit_confint(c(0.2, 0.5), c(0.1, 0.2, 0.3), conf.type = "plain"),
    suppressWarnings(reference_survfit_confint(c(0.2, 0.5), c(0.1, 0.2, 0.3), conf.type = "plain")),
    tolerance = 1e-12
  )
  expect_equal(
    survfit_confint(0.5, 0.1, logse = FALSE, conf.type = "plain", selow = 0.05, ulimit = FALSE),
    reference_survfit_confint(0.5, 0.1, logse = FALSE, conf.type = "plain", selow = 0.05, ulimit = FALSE),
    tolerance = 1e-12
  )
  expect_error(survfit_confint(0.5, 0.1, conf.type = "p"), "invalid conf.int type")
  pseudo_data <- data.frame(time = c(1, 2, 3, 4), status = c(1, 0, 1, 1))
  pseudo_fit <- survfit(Surv(time, status) ~ 1, data = pseudo_data, model = TRUE)
  expect_equal(
    pseudo(pseudo_fit, times = c(1, 2, 3)),
    matrix(
      c(0, 0, 0, 1, 1, 0.5, 1, 1, -0.25, 1, 1, 1.25),
      nrow = 4,
      byrow = TRUE,
      dimnames = list(NULL, c("1", "2", "3"))
    )
  )
  expect_equal(pseudo(pseudo_fit, times = 2), c(0, 1, 1, 1))
  expect_equal(
    pseudo(pseudo_fit, times = c(1, 2, 3), collapse = FALSE),
    matrix(
      c(0, 0, 0, 1, 1, 0.5, 1, 1, -0.25, 1, 1, 1.25),
      nrow = 4,
      byrow = TRUE,
      dimnames = list(NULL, c("1", "2", "3"))
    )
  )
  pseudo_frame <- pseudo(pseudo_fit, times = 2, data.frame = TRUE)
  expect_s3_class(pseudo_frame, "data.frame")
  expect_equal(names(pseudo_frame), c("id", "time", "pseudo"))
  grouped_pseudo_data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    group = c("A", "B", "A", "B")
  )
  grouped_pseudo_fit <- survfit(
    Surv(time, status) ~ group,
    data = grouped_pseudo_data,
    model = TRUE
  )
  expect_equal(
    pseudo(grouped_pseudo_fit, times = c(1, 2, 3)),
    matrix(
      c(0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1),
      nrow = 4,
      byrow = TRUE,
      dimnames = list(NULL, c("1", "2", "3"))
    )
  )
  expect_equal(pseudo(grouped_pseudo_fit, times = 2), c(0, 1, 1, 1))
  expect_equal(
    pseudo(grouped_pseudo_fit, times = c(1, 2, 3), collapse = FALSE),
    pseudo(grouped_pseudo_fit, times = c(1, 2, 3))
  )
  grouped_pseudo_frame <- pseudo(grouped_pseudo_fit, times = 2, data.frame = TRUE)
  expect_s3_class(grouped_pseudo_frame, "data.frame")
  expect_equal(names(grouped_pseudo_frame), c("strata", "id", "time", "pseudo"))
  expect_equal(grouped_pseudo_frame$strata, grouped_pseudo_data$group)
  expect_equal(grouped_pseudo_frame$pseudo, c(0, 1, 1, 1))

  counting_pseudo_data <- data.frame(
    start = c(0, 2, 0, 3, 0, 4),
    stop = c(2, 5, 3, 6, 4, 7),
    status = c(0, 1, 1, 0, 0, 1),
    id = c(1, 1, 2, 2, 3, 3)
  )
  counting_pseudo_fit <- survfit(
    Surv(start, stop, status) ~ 1,
    data = counting_pseudo_data,
    id = id,
    model = TRUE
  )
  reference_counting_pseudo_fit <- getFromNamespace("survfit.formula", "survival")(
    survival::Surv(start, stop, status) ~ 1,
    data = counting_pseudo_data,
    id = counting_pseudo_data$id,
    model = TRUE
  )
  for (pseudo_type in c("survival", "cumhaz", "rmst")) {
    expect_equal(
      pseudo(counting_pseudo_fit, times = c(3, 5, 7), type = pseudo_type),
      survival::pseudo(reference_counting_pseudo_fit, times = c(3, 5, 7), type = pseudo_type),
      tolerance = 1e-8
    )
  }
  expect_equal(
    pseudo(counting_pseudo_fit, times = c(3, 5, 7), collapse = FALSE),
    survival::pseudo(reference_counting_pseudo_fit, times = c(3, 5, 7), collapse = FALSE),
    tolerance = 1e-8
  )
  expect_equal(
    pseudo(counting_pseudo_fit, times = 5),
    survival::pseudo(reference_counting_pseudo_fit, times = 5),
    tolerance = 1e-8
  )
  counting_pseudo_frame <- pseudo(counting_pseudo_fit, times = 5, data.frame = TRUE)
  reference_counting_pseudo_frame <- survival::pseudo(
    reference_counting_pseudo_fit,
    times = 5,
    data.frame = TRUE
  )
  expect_equal(counting_pseudo_frame[["(id)"]], reference_counting_pseudo_frame[["(id)"]])
  expect_equal(counting_pseudo_frame$time, reference_counting_pseudo_frame$time)
  expect_equal(counting_pseudo_frame$pseudo, reference_counting_pseudo_frame$pseudo, tolerance = 1e-8)
  for (residual_type in c("survival", "cumhaz", "auc")) {
    expect_equal(
      residuals(counting_pseudo_fit, times = c(3, 5, 7), type = residual_type),
      stats::residuals(reference_counting_pseudo_fit, times = c(3, 5, 7), type = residual_type),
      tolerance = 1e-8
    )
  }
  expect_equal(
    residuals(counting_pseudo_fit, times = c(3, 5, 7), collapse = TRUE, weighted = TRUE),
    stats::residuals(reference_counting_pseudo_fit, times = c(3, 5, 7), collapse = TRUE, weighted = TRUE),
    tolerance = 1e-8
  )

  grouped_counting_pseudo_data <- data.frame(
    start = c(0, 2, 0, 3, 0, 4, 0, 5),
    stop = c(2, 5, 3, 6, 4, 7, 5, 8),
    status = c(0, 1, 1, 0, 0, 1, 1, 0),
    id = c(1, 1, 2, 2, 3, 3, 4, 4),
    group = c("A", "A", "A", "A", "B", "B", "B", "B")
  )
  grouped_counting_pseudo_fit <- survfit(
    Surv(start, stop, status) ~ group,
    data = grouped_counting_pseudo_data,
    id = id,
    model = TRUE
  )
  reference_grouped_counting_pseudo_fit <- getFromNamespace("survfit.formula", "survival")(
    survival::Surv(start, stop, status) ~ group,
    data = grouped_counting_pseudo_data,
    id = grouped_counting_pseudo_data$id,
    model = TRUE
  )
  for (pseudo_type in c("survival", "cumhaz", "rmst")) {
    expect_equal(
      pseudo(grouped_counting_pseudo_fit, times = c(3, 5), type = pseudo_type),
      survival::pseudo(
        reference_grouped_counting_pseudo_fit,
        times = c(3, 5),
        type = pseudo_type
      ),
      tolerance = 1e-8
    )
  }
  expect_equal(
    pseudo(grouped_counting_pseudo_fit, times = c(3, 5), collapse = FALSE),
    survival::pseudo(reference_grouped_counting_pseudo_fit, times = c(3, 5), collapse = FALSE),
    tolerance = 1e-8
  )
  expect_equal(
    pseudo(grouped_counting_pseudo_fit, times = 5),
    survival::pseudo(reference_grouped_counting_pseudo_fit, times = 5),
    tolerance = 1e-8
  )
  grouped_counting_residual_extra <- residuals(
    grouped_counting_pseudo_fit,
    times = c(3, 5),
    type = "survival",
    extra = TRUE
  )
  reference_grouped_counting_residual_extra <- stats::residuals(
    reference_grouped_counting_pseudo_fit,
    times = c(3, 5),
    type = "survival",
    extra = TRUE
  )
  expect_equal(
    grouped_counting_residual_extra$resid,
    reference_grouped_counting_residual_extra$resid,
    tolerance = 1e-8
  )
  expect_equal(grouped_counting_residual_extra$curve, reference_grouped_counting_residual_extra$curve)
  expect_equal(
    residuals(grouped_counting_pseudo_fit, times = c(3, 5), type = "cumhaz"),
    stats::residuals(reference_grouped_counting_pseudo_fit, times = c(3, 5), type = "cumhaz"),
    tolerance = 1e-8
  )

  fit <- coxph(Surv(time, status) ~ x, data = data, max_iter = 0, model = TRUE)
  controlled_fit <- coxph(Surv(time, status) ~ x, data = data, control = cox_control)
  expect_equal(coef(controlled_fit), coef(fit))
  aft_fit <- survreg(Surv(time, status) ~ x, data = data, control = survreg_control)
  expect_s3_class(aft_fit, "survival_py_survreg")
  expect_s3_class(aft_fit, "survival_py_model")
  expect_equal(df.residual(aft_fit), nobs(aft_fit) - attr(logLik(aft_fit), "df"))
  direct_cox_fit <- coxph(response, x = data.frame(x = data$x), max_iter = 0)
  direct_aft_fit <- survreg(response, x = data.frame(x = data$x), control = survreg_control)
  expect_equal(names(coef(direct_cox_fit)), "x")
  expect_equal(names(coef(direct_aft_fit)), "x")
  direct_prediction <- predict(direct_cox_fit, data.frame(x = c(0.5, 0.7)))
  expect_type(direct_prediction, "double")
  expect_length(direct_prediction, 2L)
  direct_terms <- predict(direct_cox_fit, data.frame(x = c(0.5, 0.7)), type = "terms", terms = "x")
  expect_true(is.matrix(direct_terms))
  expect_equal(colnames(direct_terms), "x")
  direct_aft_terms <- predict(direct_aft_fit, data.frame(x = c(0.5, 0.7)), type = "terms", terms = "x")
  expect_true(is.matrix(direct_aft_terms))
  expect_equal(colnames(direct_aft_terms), "x")
  direct_curves <- survfit(direct_cox_fit, newdata = data.frame(x = c(0.5, 0.7)), se.fit = FALSE)
  direct_curves_frame <- as.data.frame(direct_curves)
  expect_false("strata" %in% names(direct_curves_frame))
  expect_true(all(c("curve", "time", "surv") %in% names(direct_curves_frame)))
  aft_print <- capture.output(print(aft_fit))
  expect_true(any(grepl("Call:", aft_print, fixed = TRUE)))
  expect_true(any(grepl("Coefficients:", aft_print, fixed = TRUE)))
  expect_true(any(grepl("Surv(time, status) ~ x", aft_print, fixed = TRUE)))
  expect_true(any(grepl("logLik=", aft_print, fixed = TRUE)))
  expect_true(any(grepl("n=4", aft_print, fixed = TRUE)))
  expect_false(any(grepl("events=", aft_print, fixed = TRUE)))
  expect_false(any(grepl("survival.r_api", aft_print, fixed = TRUE)))
  expect_s3_class(fit, "survival_py_model")
  fit_print <- capture.output(print(fit))
  expect_true(any(grepl("Call:", fit_print, fixed = TRUE)))
  expect_true(any(grepl("Coefficients:", fit_print, fixed = TRUE)))
  expect_true(any(grepl("Surv(time, status) ~ x", fit_print, fixed = TRUE)))
  expect_true(any(grepl("logLik=", fit_print, fixed = TRUE)))
  expect_true(any(grepl("n=4", fit_print, fixed = TRUE)))
  expect_true(any(grepl("events=3", fit_print, fixed = TRUE)))
  expect_false(any(grepl("survival.r_api", fit_print, fixed = TRUE)))
  expect_length(coef(fit), 1)
  expect_named(coef(fit), "x")
  design <- model.matrix(fit)
  expect_equal(dim(design), c(nrow(data), 1L))
  expect_equal(colnames(design), "x")
  frame <- model.frame(fit)
  expect_s3_class(frame, "data.frame")
  expect_true(all(c("time", "status", "x") %in% names(frame)))
  expect_equal(dim(vcov(fit)), c(1L, 1L))
  expect_equal(dimnames(vcov(fit)), list("x", "x"))
  expect_equal(dim(confint(fit)), c(1L, 2L))
  expect_equal(rownames(confint(fit)), "x")
  expect_equal(rownames(confint(fit, parm = "x")), "x")
  expect_s3_class(logLik(fit), "logLik")
  expect_null(deviance(fit))
  expect_equal(attr(logLik(fit), "df"), 1L)
  expect_equal(nobs(fit), sum(data$status))
  expect_equal(attr(logLik(fit), "nobs"), sum(data$status))
  expect_equal(
    BIC(fit),
    -2 * as.numeric(logLik(fit)) + log(sum(data$status)) * attr(logLik(fit), "df")
  )
  expect_equal(nobs(aft_fit), nrow(data))
  expect_null(attr(logLik(aft_fit), "nobs"))
  fit_aic <- extractAIC(fit)
  expect_named(fit_aic, c("df", "AIC"))
  expect_equal(fit_aic[["df"]], 1)
  expect_equal(fit_aic[["AIC"]], as.numeric(AIC(fit)))
  expect_equal(deparse(formula(fit)), "Surv(time, status) ~ x")
  expect_s3_class(terms(fit), "terms")
  expect_null(weights(fit))
  weighted_fit <- coxph(Surv(time, status) ~ x, data = data, weights = wt, max_iter = 0)
  expect_equal(weights(weighted_fit), data$wt)
  fitted_values <- fitted(fit)
  expect_true(is.numeric(unlist(fitted_values, use.names = FALSE)))
  expect_equal(length(unlist(fitted_values, use.names = FALSE)), nrow(data))
  fit_summary <- summary(fit)
  expect_s3_class(fit_summary, "summary.survival_py_model")
  expect_equal(rownames(fit_summary$coefficients), "x")
  expect_true(all(c("coef", "se(coef)", "z", "Pr(>|z|)") %in% colnames(fit_summary$coefficients)))
  expect_equal(fit_summary$n, nrow(data))
  fit_summary_print <- capture.output(print(fit_summary))
  expect_true(any(grepl("n=4", fit_summary_print, fixed = TRUE)))
  expect_true(any(grepl("events=3", fit_summary_print, fixed = TRUE)))
  prediction <- predict(fit, data.frame(x = c(0.5, 0.7)))
  expect_true(is.numeric(unlist(prediction, use.names = FALSE)))
  prediction_with_se <- predict(fit, data.frame(x = c(0.5, 0.7)), se.fit = TRUE)
  expect_named(prediction_with_se, c("fit", "se.fit"))
  expect_type(prediction_with_se$fit, "double")
  expect_type(prediction_with_se$se.fit, "double")
  expect_equal(length(prediction_with_se$fit), 2L)
  term_prediction <- predict(fit, data.frame(x = c(0.5, 0.7)), type = "terms")
  expect_true(is.matrix(term_prediction))
  expect_equal(dim(term_prediction), c(2L, 1L))
  expect_equal(colnames(term_prediction), "x")
  term_prediction_with_se <- predict(fit, data.frame(x = c(0.5, 0.7)), type = "terms", se.fit = TRUE)
  expect_named(term_prediction_with_se, c("fit", "se.fit"))
  expect_true(is.matrix(term_prediction_with_se$fit))
  expect_true(is.matrix(term_prediction_with_se$se.fit))
  expect_equal(dim(term_prediction_with_se$fit), c(2L, 1L))
  expect_equal(colnames(term_prediction_with_se$fit), "x")
  expect_type(residuals(fit, type = "score"), "double")
  partial_residuals <- residuals(fit, type = "partial")
  expect_true(is.matrix(partial_residuals))
  expect_equal(dim(partial_residuals), c(nrow(data), 1L))
  expect_equal(colnames(partial_residuals), "x")
  multi_fit <- coxph(Surv(time, status) ~ x + wt, data = data, max_iter = 0)
  score_residuals <- residuals(multi_fit, type = "score")
  expect_true(is.matrix(score_residuals))
  expect_equal(dim(score_residuals), c(nrow(data), 2L))
  expect_equal(colnames(score_residuals), c("x", "wt"))
  cox_curves <- survfit(fit, newdata = data.frame(x = c(0.5, 0.7)), se.fit = FALSE)
  expect_s3_class(cox_curves, "survival_py_survfit")
  cox_curve_frame <- as.data.frame(cox_curves)
  expect_s3_class(cox_curve_frame, "data.frame")
  expect_true(all(c("curve", "time", "surv", "cumhaz", "linear.predictor") %in% names(cox_curve_frame)))
  expect_equal(length(unique(cox_curve_frame$curve)), 2L)
  expect_equal(dim(cox_curves), c(data = 2L))
  expect_error(residuals(cox_curves), "coxph survival curve")
  grouped_plot_fit <- survfit(Surv(time, status) ~ group, data = data)
  cox_curves_with_ci <- survfit(fit, newdata = data.frame(x = c(0.5, 0.7)), se.fit = TRUE)
  cox_curve_ci_frame <- as.data.frame(cox_curves_with_ci)
  plot_file <- tempfile(fileext = ".png")
  grDevices::png(plot_file)
  grouped_plot_end <- expect_warning(plot(grouped_plot_fit, conf.int = TRUE, mark.time = TRUE), NA)
  cox_lines_end <- expect_warning(lines(cox_curves_with_ci, conf.int = TRUE, col = 2), NA)
  expect_null(points(grouped_plot_fit))
  cox_cumhaz_end <- expect_warning(plot(cox_curves_with_ci, fun = "cumhaz", conf.int = TRUE), NA)
  grDevices::dev.off()
  expect_true(file.exists(plot_file))
  expect_gt(file.info(plot_file)$size, 0)
  expect_named(grouped_plot_end, c("x", "y"))
  expect_named(cox_lines_end, c("x", "y"))
  expect_named(cox_cumhaz_end, c("x", "y"))
  expect_length(grouped_plot_end$x, 2L)
  expect_length(cox_lines_end$x, 2L)
  expect_true(all(c("lower", "upper") %in% names(cox_curve_ci_frame)))
  expected_cumhaz_end <- vapply(split(cox_curve_ci_frame$cumhaz, cox_curve_ci_frame$curve), tail, numeric(1), 1L)
  expect_equal(cox_cumhaz_end$y, unname(expected_cumhaz_end))
  cox_aggregate_curves <- survfit(fit, newdata = data.frame(x = c(0.2, 0.7, 1.1)), se.fit = FALSE)
  cox_aggregate_frame <- as.data.frame(cox_aggregate_curves)
  cox_surv_by_curve <- split(cox_aggregate_frame$surv, cox_aggregate_frame$curve)
  cox_default_aggregate <- aggregate(cox_aggregate_curves)
  cox_default_frame <- as.data.frame(cox_default_aggregate)
  expected_default_surv <- rowMeans(do.call(cbind, cox_surv_by_curve))
  expect_s3_class(cox_default_aggregate, "survival_py_survfit")
  expect_null(dim(cox_default_aggregate))
  expect_equal(cox_default_frame$surv, expected_default_surv, tolerance = 1e-8)
  expect_equal(cox_default_frame$cumhaz, -log(expected_default_surv), tolerance = 1e-8)
  cox_group_aggregate <- aggregate(cox_aggregate_curves, by = c("lo", "hi", "lo"))
  cox_group_frame <- as.data.frame(cox_group_aggregate)
  expect_s3_class(cox_group_aggregate, "survival_py_survfit")
  expect_equal(dim(cox_group_aggregate), c(data = 2L))
  expect_equal(unique(cox_group_frame$curve), c(1L, 2L))
  expect_equal(cox_group_frame$surv[cox_group_frame$curve == 1L], cox_surv_by_curve[["2"]], tolerance = 1e-8)
  expect_equal(
    cox_group_frame$surv[cox_group_frame$curve == 2L],
    rowMeans(cbind(cox_surv_by_curve[["1"]], cox_surv_by_curve[["3"]])),
    tolerance = 1e-8
  )
  expect_error(aggregate(survfit(response, se.fit = FALSE)), "data.*margin")
  expect_error(aggregate(cox_aggregate_curves, by = "lo"), "same length")
  expect_error(aggregate(cox_aggregate_curves, FUN = max), "FUN must be mean")
  stratified_curves <- survfit(
    coxph(Surv(time, status) ~ x + strata(group), data = data, max_iter = 0),
    newdata = data.frame(x = c(0.5, 0.7), group = c("control", "treated")),
    se.fit = FALSE
  )
  stratified_curve_frame <- as.data.frame(stratified_curves)
  expect_equal(unique(stratified_curve_frame$strata), c(1L, 2L))
  expect_equal(dim(stratified_curves), c(strata = 2L))

  hazard_frame <- as.data.frame(basehaz(fit))
  expect_s3_class(hazard_frame, "data.frame")
  expect_true(all(c("time", "cumhaz") %in% names(hazard_frame)))
  stratified_fit <- coxph(Surv(time, status) ~ x + strata(group), data = data, max_iter = 0)
  stratified_hazard_frame <- as.data.frame(basehaz(stratified_fit, centered = FALSE))
  expect_equal(unique(stratified_hazard_frame$strata), c("control", "treated"))
  hazard_summary <- summary(basehaz(fit))
  expect_s3_class(hazard_summary, "summary.survival_py_basehaz")
  expect_true(all(c("time", "cumhaz") %in% names(hazard_summary)))

  zph_frame <- as.data.frame(cox.zph(fit))
  expect_s3_class(zph_frame, "data.frame")
  expect_true(all(c("name", "chisq", "p") %in% names(zph_frame)))
  zph_summary <- summary(cox.zph(fit))
  expect_s3_class(zph_summary, "summary.survival_py_cox_zph")
  expect_true(all(c("name", "chisq", "p") %in% names(zph_summary)))
  royston_data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6),
    status = c(1, 1, 0, 1, 1, 0),
    x = c(0.1, 0.5, 0.2, 1.0, 0.7, 0.3)
  )
  royston_fit <- coxph(Surv(time, status) ~ x, data = royston_data, max_iter = 50, model = TRUE)
  reference_royston_fit <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = royston_data,
    iter.max = 50,
    model = TRUE,
    y = TRUE
  )
  expect_equal(names(royston(royston_fit)), names(survival::royston(reference_royston_fit)))
  expect_equal(royston(royston_fit), survival::royston(reference_royston_fit), tolerance = 2e-3)

  brier_data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6, 7, 8),
    status = c(1, 1, 0, 1, 0, 1, 1, 0),
    x = c(0.2, 0.4, 0.1, 0.8, 1.0, 1.2, 0.6, 1.4)
  )
  brier_fit <- coxph(Surv(time, status) ~ x, data = brier_data, max_iter = 50, model = TRUE)
  reference_brier_fit <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = brier_data,
    iter.max = 50,
    model = TRUE,
    y = TRUE
  )
  bridged_brier <- brier(brier_fit, times = c(2, 4, 6), newdata = brier_data, detail = TRUE)
  reference_brier <- survival::brier(
    reference_brier_fit,
    times = c(2, 4, 6),
    newdata = brier_data,
    detail = TRUE
  )
  expect_equal(names(bridged_brier), names(reference_brier))
  expect_equal(bridged_brier$times, reference_brier$times)
  expect_equal(bridged_brier$p0, reference_brier$p0, tolerance = 1e-12)
  expect_equal(bridged_brier$eff.n, reference_brier$eff.n, tolerance = 1e-12)
  expect_lt(max(abs(bridged_brier$brier - reference_brier$brier)), 3e-3)
  expect_lt(max(abs(bridged_brier$rsquared - reference_brier$rsquared)), 3e-3)
  expect_lt(max(abs(bridged_brier$phat - reference_brier$phat)), 3e-3)

  brier_counting_data <- data.frame(
    start = rep(0, nrow(brier_data)),
    stop = brier_data$time,
    status = brier_data$status,
    x = brier_data$x,
    id = seq_len(nrow(brier_data))
  )
  brier_counting_fit <- coxph(
    Surv(start, stop, status) ~ x,
    data = brier_counting_data,
    id = id,
    max_iter = 50,
    model = TRUE
  )
  reference_brier_counting_fit <- survival::coxph(
    survival::Surv(start, stop, status) ~ x,
    data = brier_counting_data,
    id = id,
    iter.max = 50,
    model = TRUE,
    y = TRUE
  )
  bridged_brier_counting <- brier(
    brier_counting_fit,
    times = c(2, 4, 6),
    newdata = brier_counting_data,
    detail = TRUE
  )
  reference_brier_counting <- survival::brier(
    reference_brier_counting_fit,
    times = c(2, 4, 6),
    newdata = brier_counting_data,
    detail = TRUE
  )
  expect_equal(bridged_brier_counting$p0, reference_brier_counting$p0, tolerance = 1e-12)
  expect_equal(bridged_brier_counting$eff.n, reference_brier_counting$eff.n, tolerance = 1e-12)
  expect_lt(max(abs(bridged_brier_counting$brier - reference_brier_counting$brier)), 3e-3)
  expect_lt(max(abs(bridged_brier_counting$rsquared - reference_brier_counting$rsquared)), 3e-3)
  expect_lt(max(abs(bridged_brier_counting$phat - reference_brier_counting$phat)), 3e-3)

  brier_common_start_data <- data.frame(
    start = c(0, 2, 0, 3, 0, 4),
    stop = c(2, 5, 3, 6, 4, 7),
    status = c(0, 1, 1, 0, 0, 1),
    x = c(0.2, 0.2, 0.6, 0.6, 1.0, 1.0),
    id = c(1, 1, 2, 2, 3, 3)
  )
  brier_common_start_fit <- coxph(
    Surv(start, stop, status) ~ x,
    data = brier_common_start_data,
    id = id,
    max_iter = 0,
    model = TRUE
  )
  reference_brier_common_start_fit <- survival::coxph(
    survival::Surv(start, stop, status) ~ x,
    data = brier_common_start_data,
    id = id,
    iter.max = 0,
    model = TRUE,
    y = TRUE
  )
  bridged_brier_common_start <- brier(
    brier_common_start_fit,
    times = c(3, 5, 7),
    newdata = brier_common_start_data,
    detail = TRUE
  )
  reference_brier_common_start <- survival::brier(
    reference_brier_common_start_fit,
    times = c(3, 5, 7),
    newdata = brier_common_start_data,
    detail = TRUE
  )
  expect_equal(bridged_brier_common_start$p0, reference_brier_common_start$p0, tolerance = 1e-12)
  expect_equal(bridged_brier_common_start$eff.n, reference_brier_common_start$eff.n, tolerance = 1e-12)
  expect_lt(max(abs(bridged_brier_common_start$brier - reference_brier_common_start$brier)), 3e-3)
  expect_equal(bridged_brier_common_start$rsquared, reference_brier_common_start$rsquared, tolerance = 3e-3)
  expect_lt(max(abs(bridged_brier_common_start$phat - reference_brier_common_start$phat)), 3e-3)

  brier_gap_data <- transform(brier_common_start_data, start = c(0, 3, 0, 3, 0, 4))
  brier_gap_fit <- coxph(
    Surv(start, stop, status) ~ x,
    data = brier_gap_data,
    id = id,
    max_iter = 0,
    model = TRUE
  )
  expect_error(brier(brier_gap_fit, times = c(3, 5, 7)), "survcheck")

  direct_concordance <- concordance(
    response,
    scores = data$x,
    weights = data$wt,
    cluster = c("a", NA, "b", "b"),
    subset = c(TRUE, TRUE, TRUE, FALSE),
    na.action = stats::na.omit,
    influence = 1
  )
  formula_concordance <- concordance(
    "Surv(time, status) ~ x",
    data = data[c(1, 3), ],
    weights = "wt",
    cluster = c("a", "b"),
    influence = 1
  )
  named_formula_concordance <- concordance(
    formula = Surv(time, status) ~ x,
    data = data[c(1, 3), ],
    weights = wt,
    cluster = c("a", "b"),
    influence = 1
  )
  reference_formula_concordance <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = data[c(1, 3), ],
    weights = wt,
    cluster = c("a", "b"),
    influence = 1
  )
  multi_formula_concordance <- concordance(
    Surv(time, status) ~ x + wt,
    data = data,
    influence = 1
  )
  reference_multi_formula_concordance <- survival::concordance(
    survival::Surv(time, status) ~ x + wt,
    data = data,
    influence = 1
  )
  expect_error(
    concordance(Surv(time, status) ~ x + offset(x), data = data),
    "Offset terms not allowed"
  )
  string_column_concordance <- concordance(
    "Surv(time, status) ~ x",
    data = data,
    weights = "wt",
    cluster = "group",
    influence = 1
  )
  symbol_concordance <- concordance(
    Surv(time, status) ~ x,
    data = data,
    weights = wt,
    cluster = group,
    influence = 1
  )
  concordance_subset_data <- transform(data, keep = c(TRUE, FALSE, TRUE, TRUE))
  subset_symbol_concordance <- concordance(
    Surv(time, status) ~ x,
    data = concordance_subset_data,
    weights = wt,
    subset = keep,
    influence = 1
  )
  reference_subset_symbol_concordance <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = concordance_subset_data,
    weights = wt,
    subset = keep,
    influence = 1
  )
  direct_concordance_frame <- as.data.frame(direct_concordance)
  formula_concordance_frame <- as.data.frame(formula_concordance)
  named_formula_concordance_frame <- as.data.frame(named_formula_concordance)
  string_column_concordance_frame <- as.data.frame(string_column_concordance)
  symbol_concordance_frame <- as.data.frame(symbol_concordance)
  subset_symbol_concordance_frame <- as.data.frame(subset_symbol_concordance)
  expect_s3_class(direct_concordance_frame, "data.frame")
  expect_equal(formula_concordance_frame$concordance, as.numeric(reference_formula_concordance$concordance))
  expect_equal(formula_concordance_frame$variance, as.numeric(reference_formula_concordance$var))
  expect_equal(coef(formula_concordance), coef(reference_formula_concordance))
  expect_equal(vcov(formula_concordance), vcov(reference_formula_concordance))
  expect_equal(coef(multi_formula_concordance), coef(reference_multi_formula_concordance), tolerance = 1e-12)
  expect_equal(vcov(multi_formula_concordance), vcov(reference_multi_formula_concordance), tolerance = 1e-12)
  expect_equal(named_formula_concordance_frame$concordance, formula_concordance_frame$concordance)
  expect_equal(named_formula_concordance_frame$variance, formula_concordance_frame$variance)
  expect_equal(direct_concordance_frame$concordance, 1 - formula_concordance_frame$concordance)
  expect_equal(symbol_concordance_frame$concordance, string_column_concordance_frame$concordance)
  expect_equal(symbol_concordance_frame$variance, string_column_concordance_frame$variance)
  expect_equal(subset_symbol_concordance_frame$concordance, as.numeric(reference_subset_symbol_concordance$concordance))
  expect_equal(subset_symbol_concordance_frame$variance, as.numeric(reference_subset_symbol_concordance$var))
  old_concordance <- suppressWarnings(survConcordance(
    Surv(time, status) ~ x,
    data = data
  ))
  reference_old_concordance <- suppressWarnings(survival::survConcordance(
    survival::Surv(time, status) ~ x,
    data = data
  ))
  old_fit_stats <- suppressWarnings(survConcordance.fit(
    Surv(data$time, data$status),
    data$x
  ))
  reference_old_fit_stats <- suppressWarnings(survival::survConcordance.fit(
    survival::Surv(data$time, data$status),
    data$x
  ))
  old_subset_concordance <- suppressWarnings(survConcordance(
    Surv(time, status) ~ x,
    data = concordance_subset_data,
    weights = wt,
    subset = keep
  ))
  reference_old_subset_concordance <- suppressWarnings(survival::survConcordance(
    survival::Surv(time, status) ~ x,
    data = concordance_subset_data,
    weights = wt,
    subset = keep
  ))
  expect_equal(as.numeric(old_concordance$concordance), as.numeric(reference_old_concordance$concordance))
  expect_equal(names(old_fit_stats), c("concordant", "discordant", "tied.risk", "tied.time", "std(c-d)"))
  expect_equal(unname(old_fit_stats[["concordant"]]), unname(reference_old_fit_stats[["concordant"]]))
  expect_equal(as.numeric(old_subset_concordance$concordance), as.numeric(reference_old_subset_concordance$concordance))
  bridged_concordancefit <- concordancefit(
    Surv(data$time, data$status),
    data$x,
    influence = 3,
    ranks = TRUE
  )
  reference_concordancefit <- survival::concordancefit(
    survival::Surv(data$time, data$status),
    data$x,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(names(bridged_concordancefit), names(reference_concordancefit))
  expect_equal(bridged_concordancefit$concordance, reference_concordancefit$concordance, tolerance = 1e-12)
  expect_equal(bridged_concordancefit$count, reference_concordancefit$count, tolerance = 1e-12)
  expect_equal(bridged_concordancefit$n, reference_concordancefit$n)
  expect_equal(bridged_concordancefit$var, reference_concordancefit$var, tolerance = 1e-12)
  expect_true(is.finite(bridged_concordancefit$cvar))
  expect_equal(bridged_concordancefit$dfbeta, reference_concordancefit$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_concordancefit$influence, reference_concordancefit$influence, tolerance = 1e-12)
  expect_equal(bridged_concordancefit$ranks, reference_concordancefit$ranks, tolerance = 1e-12)
  tied_data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 1, 0, 1),
    x = c(0.2, 0.4, 0.4, 1.0)
  )
  expect_equal(
    concordancefit(Surv(tied_data$time, tied_data$status), tied_data$x)$count,
    survival::concordancefit(
      survival::Surv(tied_data$time, tied_data$status),
      tied_data$x
    )$count,
    tolerance = 1e-12
  )
  tied_time_data <- data.frame(
    time = c(1, 2, 2, 3, 4),
    status = c(1, 1, 1, 0, 1),
    x = c(0.2, 0.4, 0.4, 0.8, 1.0)
  )
  expect_equal(
    concordancefit(Surv(tied_time_data$time, tied_time_data$status), tied_time_data$x)$count,
    survival::concordancefit(
      survival::Surv(tied_time_data$time, tied_time_data$status),
      tied_time_data$x
    )$count,
    tolerance = 1e-12
  )
  expect_equal(
    concordancefit(Surv(data$time, data$status), data$x, reverse = TRUE)$concordance,
    survival::concordancefit(survival::Surv(data$time, data$status), data$x, reverse = TRUE)$concordance,
    tolerance = 1e-12
  )

  aft_terms <- predict(aft_fit, data.frame(x = c(0.5, 0.7)), type = "terms")
  expect_true(is.matrix(aft_terms))
  expect_equal(dim(aft_terms), c(2L, 1L))
  expect_equal(colnames(aft_terms), "x")
  aft_quantiles <- predict(aft_fit, data.frame(x = c(0.5, 0.7)), type = "quantile")
  expect_true(is.matrix(aft_quantiles))
  expect_equal(dim(aft_quantiles), c(2L, 2L))
  aft_quantiles_with_se <- predict(aft_fit, data.frame(x = c(0.5, 0.7)), type = "quantile", se.fit = TRUE)
  expect_named(aft_quantiles_with_se, c("fit", "se.fit"))
  expect_true(is.matrix(aft_quantiles_with_se$fit))
  expect_true(is.matrix(aft_quantiles_with_se$se.fit))
  aft_matrix_residuals <- residuals(aft_fit, type = "matrix")
  expect_true(is.matrix(aft_matrix_residuals))
  expect_equal(dim(aft_matrix_residuals), c(nrow(data), 6L))
  expect_equal(colnames(aft_matrix_residuals), c("g", "dg", "ddg", "ds", "dds", "dsg"))
  aft_dfbeta <- residuals(aft_fit, type = "dfbeta")
  expect_true(is.matrix(aft_dfbeta))
  expect_equal(nrow(aft_dfbeta), nrow(data))
})

test_that("data-prep helpers match R survival shapes", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  cut_value <- c(5, 15, 30)
  cut_breaks <- c(0, 10, 20, 30)
  bridged_cut <- tcut(cut_value, cut_breaks)
  reference_cut <- survival::tcut(cut_value, cut_breaks)

  expect_equal(unclass(bridged_cut), unclass(reference_cut))
  expect_equal(attr(bridged_cut, "cutpoints"), attr(reference_cut, "cutpoints"))
  expect_equal(attr(bridged_cut, "labels"), attr(reference_cut, "labels"))
  expect_s3_class(bridged_cut, "tcut")

  bridged_scaled <- tcut(cut_value, cut_breaks, labels = c("a", "b", "c"), scale = 365.25)
  reference_scaled <- survival::tcut(
    cut_value,
    cut_breaks,
    labels = c("a", "b", "c"),
    scale = 365.25
  )
  expect_equal(unclass(bridged_scaled), unclass(reference_scaled))
  expect_equal(attr(bridged_scaled, "cutpoints"), attr(reference_scaled, "cutpoints"))
  expect_equal(attr(bridged_scaled, "labels"), attr(reference_scaled, "labels"))

  expect_equal(
    neardate(c(1, 1, 2), c(1, 1, 2), c(4, 12, 7), c(5, 10, 9)),
    survival::neardate(c(1, 1, 2), c(1, 1, 2), c(4, 12, 7), c(5, 10, 9))
  )
  expect_equal(
    neardate(c(1, 1, 2), c(1, 1, 2), c(4, 12, 7), c(5, 10, 9), best = "prior"),
    survival::neardate(c(1, 1, 2), c(1, 1, 2), c(4, 12, 7), c(5, 10, 9), best = "prior")
  )
  expect_equal(
    neardate(c("a", "b"), c("a", "b"), c(4, 12), c(5, 10), nomatch = 0L),
    survival::neardate(c("a", "b"), c("a", "b"), c(4, 12), c(5, 10), nomatch = 0L)
  )
  reference_lvcf <- get0("lvcf", envir = asNamespace("survival"), inherits = FALSE)
  expect_equal(
    lvcf(c(1, 1, 1, 2, 2), c(10, NA, 12, NA, 20)),
    if (is.null(reference_lvcf)) {
      c(10, 10, 12, NA, 20)
    } else {
      reference_lvcf(c(1, 1, 1, 2, 2), c(10, NA, 12, NA, 20))
    }
  )
  expect_equal(
    lvcf(c(1, 1, 1), c(NA, 10, NA), c(2, 1, 3)),
    if (is.null(reference_lvcf)) {
      c(10, 10, 10)
    } else {
      reference_lvcf(c(1, 1, 1), c(NA, 10, NA), c(2, 1, 3))
    }
  )
  lvcf_factor <- factor(c("a", NA, "b", NA), levels = c("a", "b"))
  expect_equal(
    lvcf(c(1, 1, 1, 2), lvcf_factor),
    if (is.null(reference_lvcf)) {
      factor(c("a", "a", "b", NA), levels = c("a", "b"))
    } else {
      reference_lvcf(c(1, 1, 1, 2), lvcf_factor)
    }
  )
  reference_nostutter <- get0("nostutter", envir = asNamespace("survival"), inherits = FALSE)
  expect_equal(
    nostutter(c(1, 1, 1, 2, 2), c(0, 1, 1, 1, 1)),
    if (is.null(reference_nostutter)) {
      factor(c(0, 1, 0, 1, 0), levels = c(0, 1))
    } else {
      reference_nostutter(c(1, 1, 1, 2, 2), c(0, 1, 1, 1, 1))
    }
  )
  expect_equal(
    nostutter(c(1, 1, 1, 2, 2), c("censor", "a", "a", "b", "b"), censor = "censor"),
    if (is.null(reference_nostutter)) {
      factor(c("censor", "a", "censor", "b", "censor"), levels = c("censor", "a", "b"))
    } else {
      reference_nostutter(c(1, 1, 1, 2, 2), c("censor", "a", "a", "b", "b"), censor = "censor")
    }
  )
  single_numeric <- nostutter(
    c(1, 1, 1, 1, 2, 2, 2),
    c(1, 2, 1, 3, 1, 1, 2),
    single = TRUE
  )
  expect_equal(as.character(single_numeric), c("1", "2", "0", "3", "1", "0", "2"))
  expect_equal(levels(single_numeric), c("0", "1", "2", "3"))

  single_character <- nostutter(
    c(1, 1, 1, 1),
    c("censor", "a", "b", "a"),
    censor = "censor",
    single = TRUE
  )
  expect_equal(as.character(single_character), c("censor", "a", "b", "censor"))
  expect_equal(levels(single_character), c("censor", "a", "b"))

  expect_equal(nsk(1:5), survival::nsk(1:5), tolerance = 1e-10)
  expect_equal(nsk(1:5, df = 3), survival::nsk(1:5, df = 3), tolerance = 1e-10)
  expect_equal(
    nsk(1:5, knots = c(2, 4), Boundary.knots = c(1, 5)),
    survival::nsk(1:5, knots = c(2, 4), Boundary.knots = c(1, 5)),
    tolerance = 1e-10
  )
  expect_equal(
    nsk(1:5, df = 4, intercept = TRUE),
    survival::nsk(1:5, df = 4, intercept = TRUE),
    tolerance = 1e-10
  )
  expect_equal(
    nsk(c(1, NA, 3, 4, 5), df = 3),
    survival::nsk(c(1, NA, 3, 4, 5), df = 3),
    tolerance = 1e-10
  )

  expect_true(is.ratetable(survival::survexp.us))
  expect_equal(
    is.ratetable(survival::survexp.us, verbose = TRUE),
    survival::is.ratetable(survival::survexp.us, verbose = TRUE)
  )
  expect_false(is.ratetable(1))
  bridged_date <- ratetableDate(as.Date(c("1940-01-01", "2000-02-29", "2001-01-01")))
  reference_date <- survival::ratetableDate(as.Date(c("1940-01-01", "2000-02-29", "2001-01-01")))
  expect_equal(unclass(bridged_date), unclass(reference_date))
  expect_equal(class(bridged_date), class(reference_date))
  bridged_rtable <- ratetable(
    age = c(50, 60) * 365.25,
    sex = factor(c("male", "female")),
    year = as.Date(c("2000-01-01", "2001-01-01"))
  )
  reference_rtable <- survival::ratetable(
    age = c(50, 60) * 365.25,
    sex = factor(c("male", "female")),
    year = as.Date(c("2000-01-01", "2001-01-01"))
  )
  expect_equal(bridged_rtable, reference_rtable)
  bridged_match <- match.ratetable(bridged_rtable, survival::survexp.us)
  reference_match <- survival::match.ratetable(reference_rtable, survival::survexp.us)
  expect_equal(bridged_match, reference_match)
  rtable_frame <- data.frame(
    age = c(50, 60) * 365.25,
    sex = factor(c("male", "female")),
    year = as.Date(c("2000-01-01", "2001-01-01"))
  )
  expect_equal(
    match.ratetable(rtable_frame, survival::survexp.us),
    survival::match.ratetable(rtable_frame, survival::survexp.us)
  )
  expect_error(
    match.ratetable(rtable_frame[c("age", "year")], survival::survexp.us),
    "sex"
  )

  bridged_ci <- cipoisson(5, time = 10)
  reference_ci <- survival::cipoisson(5, time = 10)
  expect_equal(bridged_ci, reference_ci, tolerance = 1e-6)
  bridged_ci_matrix <- cipoisson(c(0, 5, 20), time = c(1, 10, 4))
  reference_ci_matrix <- survival::cipoisson(c(0, 5, 20), time = c(1, 10, 4))
  expect_equal(bridged_ci_matrix, reference_ci_matrix, tolerance = 1e-6)
  bridged_ci_recycled <- cipoisson(c(1, 2), time = c(1, 2, 3))
  reference_ci_recycled <- survival::cipoisson(c(1, 2), time = c(1, 2, 3))
  expect_equal(bridged_ci_recycled, reference_ci_recycled, tolerance = 1e-6)
  expect_equal(
    cipoisson(c(1, 2), time = c(0, 2)),
    survival::cipoisson(c(1, 2), time = c(0, 2)),
    tolerance = 1e-6
  )
  expect_equal(
    cipoisson(5, time = 10, method = "anscombe"),
    survival::cipoisson(5, time = 10, method = "anscombe"),
    tolerance = 1e-6
  )

  link_x <- c(0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)
  for (link_name in c("blogit", "bprobit", "bcloglog", "blog")) {
    bridged_link <- get(link_name)(0.05)
    reference_link <- get(link_name, asNamespace("survival"))(0.05)
    expect_s3_class(bridged_link, "link-glm")
    expect_equal(bridged_link$name, reference_link$name)
    expect_equal(bridged_link$linkfun(link_x), reference_link$linkfun(link_x), tolerance = 1e-6)
    expect_equal(
      bridged_link$linkinv(c(-2, 0, 2)),
      reference_link$linkinv(c(-2, 0, 2)),
      tolerance = 1e-12
    )
    expect_equal(
      bridged_link$mu.eta(c(-2, 0, 2)),
      reference_link$mu.eta(c(-2, 0, 2)),
      tolerance = 1e-12
    )
    expect_true(bridged_link$valideta(c(-Inf, 0, Inf)))
  }
  expect_equal(
    blogit(0.6)$linkfun(c(0, 0.25, 0.5, 0.75, 1)),
    survival::blogit(0.6)$linkfun(c(0, 0.25, 0.5, 0.75, 1)),
    tolerance = 1e-6
  )

  bridged_survexp <- survexp(
    c(365.25, 730.5),
    age = c(18262.5, 21915.0),
    year = c(2000, 2000),
    sex = c(0, 1),
    times = c(365.25, 730.5),
    method = "ederer",
    scale = 365.25
  )
  expect_s3_class(bridged_survexp, "survexp")
  expect_equal(bridged_survexp$time, c(1, 2))
  expect_equal(length(bridged_survexp$surv), 2L)
  bridged_individual <- survexp(
    c(365.25, 730.5),
    age = c(18262.5, 21915.0),
    year = c(2000, 2000),
    sex = c(0, 1),
    method = "individual.s"
  )
  expect_type(bridged_individual, "double")
  expect_equal(length(bridged_individual), 2L)
  fallback_data <- data.frame(
    time = c(10, 20),
    status = c(1, 0),
    age = c(50, 60) * 365.25,
    sex = factor(c("male", "female")),
    year = as.Date(c("2000-01-01", "2000-01-01"))
  )
  bridged_fallback <- survexp(
    survival::Surv(time, status) ~ 1,
    data = fallback_data,
    times = c(5, 10)
  )
  reference_fallback <- survival::survexp(
    survival::Surv(time, status) ~ 1,
    data = fallback_data,
    times = c(5, 10)
  )
  expect_equal(bridged_fallback$time, reference_fallback$time)
  expect_equal(bridged_fallback$surv, reference_fallback$surv)

  bridged_pyears <- pyears(
    c(10, 20, 30),
    event = c(1, 0, 1),
    group = c("a", "a", "b"),
    scale = 1
  )
  expect_s3_class(bridged_pyears, "pyears")
  expect_equal(unname(bridged_pyears$pyears), c(30, 30))
  expect_equal(names(bridged_pyears$pyears), c("a", "b"))
  expect_equal(unname(bridged_pyears$event), c(1, 1))
  bridged_pyears_frame <- pyears(
    c(10, 20, 30),
    event = c(1, 0, 1),
    group = c("a", "a", "b"),
    scale = 1,
    data.frame = TRUE
  )
  expect_s3_class(bridged_pyears_frame, "pyears")
  expect_s3_class(bridged_pyears_frame$data, "data.frame")
  expect_equal(bridged_pyears_frame$data$pyears, c(30, 30))
  pyears_formula_data <- data.frame(
    time = c(10, 20, 30),
    status = c(1, 0, 1),
    group = c("a", "a", "b"),
    wt = c(1, 2, 3)
  )
  bridged_pyears_formula <- pyears(
    Surv(time, status) ~ group,
    data = pyears_formula_data,
    scale = 1
  )
  reference_pyears_formula <- survival::pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_formula_data,
    scale = 1
  )
  expect_equal(bridged_pyears_formula$pyears, reference_pyears_formula$pyears)
  expect_equal(bridged_pyears_formula$n, reference_pyears_formula$n)
  expect_equal(bridged_pyears_formula$event, reference_pyears_formula$event)
  expect_s3_class(bridged_pyears_formula$terms, "terms")
  pyears_order_data <- data.frame(
    time = c(10, 20, 30, 40),
    status = c(1, 0, 1, 1),
    group = c("treated", "treated", "control", "control"),
    id = 1:4,
    off = c(0.1, 0.2, 0.3, 0.4)
  )
  bridged_pyears_order <- pyears(
    Surv(time, status) ~ group,
    data = pyears_order_data,
    scale = 1
  )
  reference_pyears_order <- survival::pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_order_data,
    scale = 1
  )
  expect_equal(bridged_pyears_order$pyears, reference_pyears_order$pyears)
  expect_equal(bridged_pyears_order$event, reference_pyears_order$event)
  expect_equal(
    pyears(Surv(time, status) ~ offset(off), data = pyears_order_data, scale = 1)$pyears,
    survival::pyears(
      survival::Surv(time, status) ~ offset(off),
      data = pyears_order_data,
      scale = 1
    )$pyears
  )
  expect_equal(
    pyears(Surv(time, status) ~ group + offset(off), data = pyears_order_data, scale = 1)$pyears,
    survival::pyears(
      survival::Surv(time, status) ~ group + offset(off),
      data = pyears_order_data,
      scale = 1
    )$pyears
  )
  bridged_pyears_cluster <- pyears(
    Surv(time, status) ~ group + cluster(id),
    data = pyears_order_data,
    scale = 1
  )
  reference_pyears_cluster <- survival::pyears(
    survival::Surv(time, status) ~ group + cluster(id),
    data = pyears_order_data,
    scale = 1
  )
  expect_equal(bridged_pyears_cluster$pyears, reference_pyears_cluster$pyears)
  expect_equal(bridged_pyears_cluster$n, reference_pyears_cluster$n)
  expect_equal(bridged_pyears_cluster$event, reference_pyears_cluster$event)
  bridged_pyears_intercept <- pyears(
    survival::Surv(time, status) ~ 1,
    data = pyears_formula_data,
    scale = 1
  )
  reference_pyears_intercept <- survival::pyears(
    survival::Surv(time, status) ~ 1,
    data = pyears_formula_data,
    scale = 1
  )
  expect_equal(bridged_pyears_intercept$pyears, reference_pyears_intercept$pyears)
  expect_equal(bridged_pyears_intercept$event, reference_pyears_intercept$event)
  expect_equal(
    pyears(Surv(time, status) ~ group, data = pyears_formula_data, weights = wt, scale = 1)$pyears,
    survival::pyears(
      survival::Surv(time, status) ~ group,
      data = pyears_formula_data,
      weights = wt,
      scale = 1
    )$pyears
  )
  pyears_counting_data <- data.frame(
    start = c(0, 5, 10),
    stop = c(10, 20, 30),
    event = c(1, 0, 1),
    group = c("a", "a", "b")
  )
  expect_equal(
    pyears(Surv(start, stop, event) ~ group, data = pyears_counting_data, scale = 1)$pyears,
    survival::pyears(
      survival::Surv(start, stop, event) ~ group,
      data = pyears_counting_data,
      scale = 1
    )$pyears
  )
  pyears_multi_data <- data.frame(
    time = c(10, 20, 30),
    status = c(1, 0, 1),
    group = factor(c("a", "a", "b"), levels = c("a", "b", "c")),
    sex = factor(c("m", "f", "m"), levels = c("f", "m"))
  )
  bridged_pyears_multi <- pyears(
    Surv(time, status) ~ group + sex,
    data = pyears_multi_data,
    scale = 1
  )
  reference_pyears_multi <- survival::pyears(
    survival::Surv(time, status) ~ group + sex,
    data = pyears_multi_data,
    scale = 1
  )
  expect_equal(bridged_pyears_multi$pyears, reference_pyears_multi$pyears)
  expect_equal(bridged_pyears_multi$n, reference_pyears_multi$n)
  expect_equal(bridged_pyears_multi$event, reference_pyears_multi$event)
  bridged_pyears_formula_frame <- pyears(
    Surv(time, status) ~ group,
    data = pyears_formula_data,
    scale = 1,
    data.frame = TRUE
  )
  reference_pyears_formula_frame <- survival::pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_formula_data,
    scale = 1,
    data.frame = TRUE
  )
  expect_equal(
    bridged_pyears_formula_frame$data[c("group", "pyears", "n", "event")],
    reference_pyears_formula_frame$data[c("group", "pyears", "n", "event")]
  )
  pyears_multi_frame_data <- data.frame(
    time = c(10, 20, 30),
    status = c(1, 0, 1),
    group = c("a", "a", "b"),
    sex = c("m", "f", "m")
  )
  expect_equal(
    pyears(
      Surv(time, status) ~ group + sex,
      data = pyears_multi_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data,
    survival::pyears(
      survival::Surv(time, status) ~ group + sex,
      data = pyears_multi_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data
  )
  pyears_factor_frame_data <- data.frame(
    time = c(10, 20, 30, 15),
    status = c(1, 0, 1, 1),
    group = factor(c("a", "a", "b", "b"), levels = c("a", "b", "c")),
    sex = ordered(c("m", "f", "m", "f"), levels = c("f", "m"))
  )
  bridged_pyears_factor_frame <- pyears(
    Surv(time, status) ~ group + sex,
    data = pyears_factor_frame_data,
    scale = 1,
    data.frame = TRUE
  )$data
  reference_pyears_factor_frame <- survival::pyears(
    survival::Surv(time, status) ~ group + sex,
    data = pyears_factor_frame_data,
    scale = 1,
    data.frame = TRUE
  )$data
  expect_equal(as.character(bridged_pyears_factor_frame$group), as.character(reference_pyears_factor_frame$group))
  expect_equal(levels(bridged_pyears_factor_frame$group), levels(reference_pyears_factor_frame$group))
  expect_equal(is.ordered(bridged_pyears_factor_frame$group), is.ordered(reference_pyears_factor_frame$group))
  expect_equal(as.character(bridged_pyears_factor_frame$sex), as.character(reference_pyears_factor_frame$sex))
  expect_equal(levels(bridged_pyears_factor_frame$sex), levels(reference_pyears_factor_frame$sex))
  expect_equal(is.ordered(bridged_pyears_factor_frame$sex), is.ordered(reference_pyears_factor_frame$sex))
  expect_equal(
    bridged_pyears_factor_frame[c("pyears", "n", "event")],
    reference_pyears_factor_frame[c("pyears", "n", "event")]
  )
  expect_equal(attr(bridged_pyears_factor_frame, "out.attrs"), attr(reference_pyears_factor_frame, "out.attrs"))
  pyears_date_frame_data <- data.frame(
    time = c(10, 20, 30, 15),
    status = c(1, 0, 1, 1),
    visit = as.Date(c("2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"))
  )
  expect_equal(
    pyears(
      Surv(time, status) ~ visit,
      data = pyears_date_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data,
    survival::pyears(
      survival::Surv(time, status) ~ visit,
      data = pyears_date_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data
  )
  pyears_posix_frame_data <- data.frame(
    time = c(10, 20, 30, 15),
    status = c(1, 0, 1, 1),
    stamp = as.POSIXct(
      c("2020-01-01 01:00:00", "2020-01-01 01:00:00", "2020-02-01 02:00:00", "2020-02-01 02:00:00"),
      tz = "UTC"
    )
  )
  expect_equal(
    pyears(
      Surv(time, status) ~ stamp,
      data = pyears_posix_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data,
    survival::pyears(
      survival::Surv(time, status) ~ stamp,
      data = pyears_posix_frame_data,
      scale = 1,
      data.frame = TRUE
    )$data
  )
  pyears_collision_data <- data.frame(
    time = c(5, 7, 11, 13),
    status = c(1, 1, 0, 1),
    a = c("x\ry", "x", "x\ry", "x"),
    b = c("z", "y\rz", "z", "y\rz")
  )
  bridged_pyears_collision <- pyears(
    Surv(time, status) ~ a + b,
    data = pyears_collision_data,
    scale = 1
  )
  reference_pyears_collision <- survival::pyears(
    survival::Surv(time, status) ~ a + b,
    data = pyears_collision_data,
    scale = 1
  )
  expect_equal(bridged_pyears_collision$pyears, reference_pyears_collision$pyears)
  expect_equal(bridged_pyears_collision$event, reference_pyears_collision$event)
  pyears_transform_data <- data.frame(
    time = c(10, 20, 30, 15),
    status = c(1, 0, 1, 1),
    group = c("a", "a", "b", "b"),
    wt = c(1, 2, 3, 4),
    keep = c(TRUE, TRUE, TRUE, FALSE)
  )
  bridged_pyears_factor <- pyears(
    Surv(time, status) ~ factor(group),
    data = pyears_transform_data,
    scale = 1
  )
  reference_pyears_factor <- survival::pyears(
    survival::Surv(time, status) ~ factor(group),
    data = pyears_transform_data,
    scale = 1
  )
  expect_equal(bridged_pyears_factor$pyears, reference_pyears_factor$pyears)
  expect_equal(bridged_pyears_factor$n, reference_pyears_factor$n)
  expect_equal(bridged_pyears_factor$event, reference_pyears_factor$event)
  bridged_pyears_paste <- pyears(
    Surv(time, status) ~ paste0(group, status),
    data = pyears_transform_data,
    weights = wt,
    subset = keep,
    scale = 1
  )
  reference_pyears_paste <- survival::pyears(
    survival::Surv(time, status) ~ paste0(group, status),
    data = pyears_transform_data,
    weights = wt,
    subset = keep,
    scale = 1
  )
  expect_equal(bridged_pyears_paste$pyears, reference_pyears_paste$pyears)
  expect_equal(bridged_pyears_paste$n, reference_pyears_paste$n)
  expect_equal(bridged_pyears_paste$event, reference_pyears_paste$event)
  pyears_counting_transform_data <- data.frame(
    start = c(0, 5, 0),
    stop = c(10, 20, 30),
    event = c(1, 0, 1),
    group = c("a", "a", "b")
  )
  expect_equal(
    pyears(
      Surv(start, stop, event) ~ factor(group),
      data = pyears_counting_transform_data,
      scale = 1
    )$pyears,
    survival::pyears(
      survival::Surv(start, stop, event) ~ factor(group),
      data = pyears_counting_transform_data,
      scale = 1
    )$pyears
  )

  bridged_finegray <- finegray(
    c(0, 0, 0, 0),
    tstop = c(1, 2, 3, 4),
    ctime = c(0.5, 1.5, 2.5, 3.5),
    cprob = c(0.1, 0.2, 0.3, 0.4),
    extend = c(TRUE, TRUE, FALSE, FALSE),
    keep = c(TRUE, TRUE, TRUE, TRUE)
  )
  expect_s3_class(bridged_finegray, "data.frame")
  expect_equal(names(bridged_finegray), c("row", "start", "end", "wt", "add"))
  expect_equal(nrow(bridged_finegray), 7L)
  finegray_data <- data.frame(
    time = c(5, 8, 10, 12),
    status = factor(c("a", "b", "censored", "a"), levels = c("censored", "a", "b")),
    x = c("a", "b", "a", "b")
  )
  bridged_finegray_fallback <- suppressWarnings(finegray(
    survival::Surv(time, status) ~ x,
    data = finegray_data,
    etype = "a"
  ))
  reference_finegray_fallback <- suppressWarnings(survival::finegray(
    survival::Surv(time, status) ~ x,
    data = finegray_data,
    etype = "a"
  ))
  expect_equal(bridged_finegray_fallback, reference_finegray_fallback)
  finegray_extended_data <- data.frame(
    time = c(5, 8, 10, 12, 7, 11),
    status = factor(
      c("a", "b", "censored", "a", "censored", "b"),
      levels = c("censored", "a", "b")
    ),
    x = c("a", "b", "a", "b", "a", "b"),
    group = c("one", "one", "one", "one", "two", "two"),
    wt = c(1, 2, 1, 3, 2, 1)
  )
  expect_equal(
    finegray(Surv(time, status) ~ x, data = finegray_extended_data, etype = "a"),
    survival::finegray(
      survival::Surv(time, status) ~ x,
      data = finegray_extended_data,
      etype = "a"
    )
  )
  expect_equal(
    finegray(
      Surv(time, status) ~ x,
      data = finegray_extended_data,
      weights = wt,
      etype = "a"
    ),
    survival::finegray(
      survival::Surv(time, status) ~ x,
      data = finegray_extended_data,
      weights = wt,
      etype = "a"
    )
  )
  expect_equal(
    finegray(
      Surv(time, status) ~ x,
      data = finegray_extended_data,
      etype = "b",
      prefix = "cr",
      count = "added"
    ),
    survival::finegray(
      survival::Surv(time, status) ~ x,
      data = finegray_extended_data,
      etype = "b",
      prefix = "cr",
      count = "added"
    )
  )
  expect_equal(
    finegray(
      Surv(time, status) ~ x + strata(group),
      data = finegray_extended_data,
      etype = "a"
    ),
    survival::finegray(
      survival::Surv(time, status) ~ x + strata(group),
      data = finegray_extended_data,
      etype = "a"
    )
  )
  finegray_counting_data <- data.frame(
    id = c(1, 1, 2, 2, 3, 3),
    start = c(0, 5, 0, 4, 0, 6),
    stop = c(5, 8, 4, 9, 6, 10),
    status = factor(
      c("censored", "a", "censored", "b", "censored", "a"),
      levels = c("censored", "a", "b")
    ),
    x = c("a", "a", "b", "b", "a", "a")
  )
  expect_equal(
    finegray(
      Surv(start, stop, status) ~ x,
      data = finegray_counting_data,
      id = id,
      etype = "a"
    ),
    survival::finegray(
      survival::Surv(start, stop, status) ~ x,
      data = finegray_counting_data,
      id = id,
      etype = "a"
    )
  )
  finegray_delayed_data <- data.frame(
    id = c(1, 1, 2, 2, 3, 3, 4, 4),
    start = c(0, 5, 0, 4, 2, 6, 6, 9),
    stop = c(5, 8, 4, 9, 6, 10, 9, 12),
    status = factor(
      c("censored", "a", "censored", "b", "censored", "a", "censored", "b"),
      levels = c("censored", "a", "b")
    ),
    x = c("a", "a", "b", "b", "a", "a", "b", "b")
  )
  expect_equal(
    finegray(
      Surv(start, stop, status) ~ x,
      data = finegray_delayed_data,
      id = id,
      etype = "a",
      count = "extra"
    ),
    survival::finegray(
      survival::Surv(start, stop, status) ~ x,
      data = finegray_delayed_data,
      id = id,
      etype = "a",
      count = "extra"
    )
  )
  expect_error(
    finegray(
      Surv(start, stop, status) ~ x,
      data = finegray_counting_data,
      etype = "a"
    ),
    "requires a subject id"
  )

  bridged_obrien <- survobrien(
    c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    covariate = c(0.1, 0.4, 0.2, 0.8),
    strata = c(1, 1, 2, 2)
  )
  expect_true(is.list(bridged_obrien))
  expect_equal(names(bridged_obrien), c(
    "statistic", "p.value", "df", "scores", "score.sum", "expected", "variance"
  ))
  expect_equal(length(bridged_obrien$scores), 4L)
  expect_equal(bridged_obrien$df, 1L)
  expect_true(is.finite(bridged_obrien$statistic))
  labeled_obrien <- survobrien(
    c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    covariate = c(0.1, 0.4, 0.2, 0.8),
    strata = c("a", "a", "b", "b")
  )
  expect_equal(labeled_obrien$statistic, bridged_obrien$statistic)
  expect_equal(labeled_obrien$p.value, bridged_obrien$p.value)
  expect_equal(labeled_obrien$scores, bridged_obrien$scores)
  obrien_data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    x = c(0.1, 0.4, 0.2, 0.8),
    group = c("a", "a", "b", "b"),
    id = c(10, 11, 12, 13),
    off = c(0.1, 0.2, 0.3, 0.4)
  )
  bridged_obrien_fallback <- survobrien(
    survival::Surv(time, status) ~ x + strata(group),
    data = obrien_data
  )
  reference_obrien_fallback <- survival::survobrien(
    survival::Surv(time, status) ~ x + strata(group),
    data = obrien_data
  )
  expect_equal(bridged_obrien_fallback, reference_obrien_fallback)
  obrien_transform <- function(x) x * 2
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ x,
      data = obrien_data,
      transform = obrien_transform
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ x,
      data = obrien_data,
      transform = obrien_transform
    )
  )
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ log(x),
      data = obrien_data
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ log(x),
      data = obrien_data
    )
  )
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ x + offset(off),
      data = obrien_data
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ x + offset(off),
      data = obrien_data
    )
  )
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ sqrt(x) + strata(group),
      data = obrien_data
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ sqrt(x) + strata(group),
      data = obrien_data
    )
  )
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ identity(x),
      data = obrien_data
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ identity(x),
      data = obrien_data
    )
  )
  expect_equal(
    survobrien(
      survival::Surv(time, status) ~ x + cluster(id),
      data = obrien_data
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ x + cluster(id),
      data = obrien_data
    )
  )

  obrien_counting_data <- data.frame(
    start = c(0, 0, 1, 2),
    stop = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    x = c(0.1, 0.4, 0.2, 0.8)
  )
  expect_equal(
    survobrien(survival::Surv(start, stop, status) ~ x, data = obrien_counting_data),
    survival::survobrien(survival::Surv(start, stop, status) ~ x, data = obrien_counting_data)
  )
  obrien_counting_strata_data <- data.frame(
    start = c(0, 0, 1, 2, 0, 3),
    stop = c(1, 2, 3, 4, 2, 5),
    status = c(1, 0, 1, 1, 1, 0),
    x = c(0.1, 0.4, 0.2, 0.8, 0.5, 0.7),
    group = c("a", "a", "b", "b", "a", "b")
  )
  expect_equal(
    survobrien(
      survival::Surv(start, stop, status) ~ x + strata(group),
      data = obrien_counting_strata_data
    ),
    survival::survobrien(
      survival::Surv(start, stop, status) ~ x + strata(group),
      data = obrien_counting_strata_data
    )
  )

  condense_data <- data.frame(
    id = c(2, 1, 1, 2),
    tstart = c(0, 0, 5, 3),
    tstop = c(3, 5, 8, 5),
    event = c(0, 0, 0, 1),
    x = c("a", "b", "b", "a"),
    wt = c(1, 2, 2, 1)
  )
  bridged_condense <- survcondense(
    Surv(tstart, tstop, event) ~ x,
    data = condense_data,
    id = id
  )
  reference_formula <- Surv(tstart, tstop, event) ~ x
  environment(reference_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  reference_condense <- survival::survcondense(
    reference_formula,
    data = condense_data,
    id = id
  )
  expect_equal(bridged_condense, reference_condense)
  bridged_condense_weighted <- survcondense(
    Surv(tstart, tstop, event) ~ x,
    data = condense_data,
    id = id,
    weights = wt
  )
  reference_condense_weighted <- survival::survcondense(
    reference_formula,
    data = condense_data,
    id = id,
    weights = wt
  )
  expect_equal(bridged_condense_weighted, reference_condense_weighted)
  condense_subset_data <- rbind(
    transform(condense_data, keep = TRUE),
    data.frame(id = 3, tstart = 0, tstop = 1, event = 0, x = "c", wt = 5, keep = FALSE)
  )
  bridged_condense_subset <- survcondense(
    Surv(tstart, tstop, event) ~ x,
    data = condense_subset_data,
    id = id,
    subset = keep
  )
  reference_condense_subset <- survival::survcondense(
    reference_formula,
    data = condense_subset_data,
    id = id,
    subset = keep
  )
  expect_equal(bridged_condense_subset, reference_condense_subset)
  bridged_condense_weighted_subset <- survcondense(
    Surv(tstart, tstop, event) ~ x,
    data = condense_subset_data,
    id = id,
    weights = wt,
    subset = keep
  )
  reference_condense_weighted_subset <- survival::survcondense(
    reference_formula,
    data = condense_subset_data,
    id = id,
    weights = wt,
    subset = keep
  )
  expect_equal(bridged_condense_weighted_subset, reference_condense_weighted_subset)
  condense_factor_data <- transform(
    condense_data,
    x = factor(x, levels = c("a", "b", "c")),
    y = ordered(c("late", "early", "early", "late"), levels = c("early", "late")),
    visit = as.Date(c("2020-02-01", "2020-01-01", "2020-01-01", "2020-02-01")),
    stamp = as.POSIXct(
      c("2020-02-01 04:05:06", "2020-01-01 01:02:03", "2020-01-01 01:02:03", "2020-02-01 04:05:06"),
      tz = "UTC"
    )
  )
  condense_factor_formula <- Surv(tstart, tstop, event) ~ x + y + visit + stamp
  environment(condense_factor_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(condense_factor_formula, data = condense_factor_data, id = id),
    survival::survcondense(condense_factor_formula, data = condense_factor_data, id = id)
  )
  condense_factor_call_formula <- Surv(tstart, tstop, event) ~ factor(x)
  environment(condense_factor_call_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(condense_factor_call_formula, data = condense_factor_data, id = id),
    survival::survcondense(condense_factor_call_formula, data = condense_factor_data, id = id)
  )
  condense_special_data <- transform(
    condense_data,
    site = c("south", "north", "north", "south"),
    phase = c("late", "early", "early", "late"),
    off = c(1, 2, 2, 1)
  )
  condense_strata_formula <- Surv(tstart, tstop, event) ~ strata(site)
  environment(condense_strata_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(condense_strata_formula, data = condense_special_data, id = id),
    survival::survcondense(condense_strata_formula, data = condense_special_data, id = id)
  )
  condense_multi_strata_formula <- Surv(tstart, tstop, event) ~ strata(site, phase)
  environment(condense_multi_strata_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(condense_multi_strata_formula, data = condense_special_data, id = id),
    survival::survcondense(condense_multi_strata_formula, data = condense_special_data, id = id)
  )
  condense_offset_formula <- Surv(tstart, tstop, event) ~ offset(off) + x
  environment(condense_offset_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(condense_offset_formula, data = condense_special_data, id = id),
    survival::survcondense(condense_offset_formula, data = condense_special_data, id = id)
  )
})

test_that("Cox bridge agrees with R survival on a small right-censored fixture", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 2, 3, 4, 5, 6, 7),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    x = c(0.1, 0.4, 0.2, 0.8, 1.1, 0.7, 1.5, 1.2),
    z = c(1, 0, 1, 0, 1, 1, 0, 0)
  )
  newdata <- data.frame(x = c(0.3, 0.9), z = c(0, 1))

  bridged <- coxph(Surv(time, status) ~ x + z, data = data, eps = 1e-10, max_iter = 50)
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + z,
    data = data,
    eps = 1e-10,
    iter.max = 50
  )

  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 1e-05)
  expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 1e-04)
  expect_equal(
    unname(predict(bridged, newdata, type = "lp")),
    unname(stats::predict(reference, newdata, type = "lp")),
    tolerance = 1e-05
  )
  expect_equal(
    unname(predict(bridged, newdata, type = "risk")),
    unname(stats::predict(reference, newdata, type = "risk")),
    tolerance = 1e-05
  )

  bridged_hazard <- as.data.frame(basehaz(bridged, centered = FALSE))
  reference_hazard <- survival::basehaz(reference, centered = FALSE)
  expect_equal(bridged_hazard$time, reference_hazard$time)
  expect_equal(bridged_hazard$cumhaz, reference_hazard$hazard, tolerance = 1e-04)
  expect_equal(as.numeric(logLik(bridged)), reference$loglik[[2L]], tolerance = 1e-05)
  expect_equal(nobs(bridged), nobs(reference))
  expect_equal(attr(logLik(bridged), "nobs"), attr(logLik(reference), "nobs"))
  expect_equal(BIC(bridged), BIC(reference), tolerance = 1e-05)
  bridged_summary <- summary(bridged)
  reference_summary <- summary(reference)
  expect_equal(bridged_summary$n, reference_summary$n)
  expect_equal(bridged_summary$n_event, reference_summary$nevent)
  expect_equal(deviance(bridged), deviance(reference))
  expect_equal(labels(bridged), attr(reference$terms, "term.labels"))
  bridged_concordance <- concordance(bridged)
  direct_concordance <- concordancefit(Surv(data$time, data$status), predict(bridged, type = "lp"), reverse = TRUE)
  reference_concordance <- survival::concordance(reference)
  expect_s3_class(bridged_concordance, "concordance")
  expect_equal(coef(bridged_concordance), bridged_concordance$concordance)
  expect_equal(vcov(bridged_concordance), bridged_concordance$var)
  bridged_concordance_print <- capture.output(print(bridged_concordance))
  expect_true(any(grepl("Call:", bridged_concordance_print, fixed = TRUE)))
  expect_true(any(grepl("Concordance=", bridged_concordance_print, fixed = TRUE)))
  expect_false(any(grepl("$concordance", bridged_concordance_print, fixed = TRUE)))
  expect_equal(bridged_concordance$concordance, direct_concordance$concordance, tolerance = 1e-12)
  expect_equal(bridged_concordance$count, direct_concordance$count, tolerance = 1e-12)
  expect_equal(bridged_concordance$concordance, reference_concordance$concordance, tolerance = 1e-02)
  expect_equal(bridged_concordance$n, reference_concordance$n)
})

test_that("Cox likelihood metadata counts weighted and recurrent event rows", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  right <- data.frame(
    time = 1:6,
    status = c(1, 0, 1, 0, 1, 0),
    x = c(0.2, 0.4, 0.1, 0.8, 0.5, 0.3),
    weight = c(0.5, 2, 1.5, 0.75, 3, 4)
  )
  weighted <- coxph(
    Surv(time, status) ~ x,
    data = right,
    weights = weight,
    max_iter = 0
  )
  reference_weighted <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = right,
    weights = weight,
    control = survival::coxph.control(iter.max = 0)
  )

  expect_equal(nobs(weighted), sum(right$status))
  expect_equal(nobs(weighted), nobs(reference_weighted))
  expect_equal(attr(logLik(weighted), "nobs"), attr(logLik(reference_weighted), "nobs"))
  expect_equal(BIC(weighted), BIC(reference_weighted), tolerance = 1e-12)
  expect_equal(summary(weighted)$n, nrow(right))

  no_events <- coxph(
    Surv(time, rep(0, nrow(right))) ~ x,
    data = right
  )
  reference_no_events <- survival::coxph(
    survival::Surv(time, rep(0, nrow(right))) ~ x,
    data = right
  )
  expect_equal(nobs(no_events), 0L)
  expect_equal(attr(logLik(no_events), "df"), 0L)
  expect_equal(attr(logLik(no_events), "nobs"), 0L)
  expect_true(is.nan(BIC(no_events)))
  expect_true(is.nan(BIC(reference_no_events)))
  expect_equal(summary(no_events)$n, nrow(right))
  expect_equal(summary(no_events)$n_event, 0L)

  recurrent <- data.frame(
    start = c(0, 1, 0, 2, 0, 1, 2, 0),
    stop = c(1, 3, 2, 4, 5, 2, 4, 6),
    status = c(0, 1, 1, 0, 1, 1, 0, 1),
    x = c(0.2, 0.4, 0.1, 0.8, 0.5, 0.3, 0.9, 0.6),
    id = c(1, 1, 2, 2, 3, 3, 4, 4)
  )
  counting <- coxph(
    Surv(start, stop, status) ~ x,
    data = recurrent,
    id = id,
    max_iter = 0
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x,
    data = recurrent,
    id = id,
    control = survival::coxph.control(iter.max = 0)
  )

  expect_equal(nobs(counting), sum(recurrent$status))
  expect_equal(nobs(counting), nobs(reference_counting))
  expect_equal(attr(logLik(counting), "nobs"), attr(logLik(reference_counting), "nobs"))
  expect_equal(BIC(counting), BIC(reference_counting), tolerance = 1e-12)
  expect_equal(summary(counting)$n, nrow(recurrent))
  expect_equal(summary(counting)$n_event, sum(recurrent$status))
})

test_that("survreg bridge agrees with R survival distributions", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  expect_equal(names(survreg.distributions), names(survival::survreg.distributions))
  std_x <- c(-1, -0.25, 0.5, 1.25)
  probabilities <- c(0.2, 0.5, 0.8)
  deviance_y <- matrix(
    c(1, 2, 1, 2, 4, 3, 3, 5, 0),
    ncol = 3,
    byrow = TRUE
  )
  for (dist in c("extreme", "logistic", "gaussian", "t")) {
    bridged_dist <- survreg.distributions[[dist]]
    reference_dist <- survival::survreg.distributions[[dist]]
    parms <- if (identical(dist, "t")) 4 else NULL
    dist_deviance_y <- if (identical(dist, "t")) {
      matrix(c(1, 2, 1, 2, 4, 0, 3, 5, 1), ncol = 3, byrow = TRUE)
    } else {
      deviance_y
    }
    expect_equal(bridged_dist$name, reference_dist$name)
    expect_equal(bridged_dist$variance(if (is.null(parms)) NULL else parms),
                 reference_dist$variance(if (is.null(parms)) NULL else parms))
    expect_equal(
      bridged_dist$init(std_x, rep(1, length(std_x)), df = 4),
      reference_dist$init(std_x, rep(1, length(std_x)), df = 4),
      tolerance = 1e-12
    )
    expect_equal(
      bridged_dist$deviance(dist_deviance_y, scale = 1.3, parms = parms),
      reference_dist$deviance(dist_deviance_y, scale = 1.3, parms = parms),
      tolerance = 1e-12
    )
    expect_equal(
      bridged_dist$density(std_x, parms),
      reference_dist$density(std_x, parms),
      tolerance = 1e-12
    )
    expect_equal(
      bridged_dist$quantile(probabilities, parms),
      reference_dist$quantile(probabilities, parms),
      tolerance = 1e-12
    )
  }
  for (dist in c("weibull", "exponential", "rayleigh", "loggaussian", "lognormal", "loglogistic")) {
    bridged_dist <- survreg.distributions[[dist]]
    reference_dist <- survival::survreg.distributions[[dist]]
    expect_equal(names(bridged_dist), names(reference_dist))
    expect_equal(bridged_dist$name, reference_dist$name)
    expect_equal(bridged_dist$dist, reference_dist$dist)
    expect_equal(bridged_dist$scale, reference_dist$scale)
    expect_equal(bridged_dist$trans(c(1, 2, 4)), reference_dist$trans(c(1, 2, 4)))
    expect_equal(bridged_dist$itrans(c(0, 1, 2)), reference_dist$itrans(c(0, 1, 2)))
    expect_equal(bridged_dist$dtrans(c(1, 2, 4)), reference_dist$dtrans(c(1, 2, 4)))
  }
  for (dist in names(survreg.distributions)) {
    expect_equal(survregDtest(survreg.distributions[[dist]]), survival::survregDtest(survival::survreg.distributions[[dist]]))
  }
  invalid_dist <- list(
    name = "Broken",
    dist = "missing",
    trans = identity,
    itrans = identity,
    dtrans = identity
  )
  expect_equal(survregDtest(invalid_dist), survival::survregDtest(invalid_dist))
  expect_equal(survregDtest(invalid_dist, verbose = TRUE), survival::survregDtest(invalid_dist, verbose = TRUE))

  data <- data.frame(
    time = c(1.2, 2.1, 2.8, 3.4, 4.2, 5.0, 6.3, 7.1),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    x = c(0.1, 0.3, 0.2, 0.8, 1.0, 0.7, 1.4, 1.1),
    z = c(1, 0, 1, 0, 1, 1, 0, 0)
  )
  newdata <- data.frame(x = c(0.25, 0.95), z = c(0, 1))

  for (dist in c("weibull", "lognormal", "loglogistic", "gaussian", "logistic", "exponential")) {
    bridged <- survreg(
      Surv(time, status) ~ x + z,
      data = data,
      dist = dist,
      max_iter = 150,
      eps = 1e-10
    )
    reference <- survival::survreg(
      survival::Surv(time, status) ~ x + z,
      data = data,
      dist = dist,
      control = survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)
    )

    expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-04)
    expect_equal(labels(bridged), labels(reference))
    expect_equal(as.numeric(summary(bridged)$scale), reference$scale, tolerance = 5e-05)
    expect_equal(as.numeric(logLik(bridged)), reference$loglik[[2L]], tolerance = 1e-05)
    expect_equal(deviance(bridged), deviance(reference))
    bridged_concordance <- concordance(bridged)
    direct_concordance <- concordancefit(Surv(data$time, data$status), predict(bridged, type = "lp"), reverse = FALSE)
    reference_concordance <- survival::concordance(reference)
    expect_s3_class(bridged_concordance, "concordance")
    expect_equal(coef(bridged_concordance), bridged_concordance$concordance)
    expect_equal(vcov(bridged_concordance), bridged_concordance$var)
    expect_true(any(grepl("Concordance=", capture.output(print(bridged_concordance)), fixed = TRUE)))
    expect_equal(bridged_concordance$concordance, direct_concordance$concordance, tolerance = 1e-12)
    expect_equal(bridged_concordance$count, direct_concordance$count, tolerance = 1e-12)
    expect_equal(bridged_concordance$concordance, reference_concordance$concordance, tolerance = 1e-06)
    expect_equal(bridged_concordance$n, reference_concordance$n)
    expect_equal(
      unname(predict(bridged, newdata, type = "lp")),
      unname(stats::predict(reference, newdata, type = "lp")),
      tolerance = 2e-04
    )
    expect_equal(
      unname(predict(bridged, newdata, type = "response")),
      unname(stats::predict(reference, newdata, type = "response")),
      tolerance = 5e-04
    )

    bridged_dist_list <- survreg(
      Surv(time, status) ~ x + z,
      data = data,
      dist = survreg.distributions[[dist]],
      max_iter = 150,
      eps = 1e-10
    )
    reference_dist_list <- survival::survreg(
      survival::Surv(time, status) ~ x + z,
      data = data,
      dist = survival::survreg.distributions[[dist]],
      control = survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)
    )
    expect_equal(unname(coef(bridged_dist_list)), unname(coef(reference_dist_list)), tolerance = 2e-04)
    expect_equal(as.numeric(summary(bridged_dist_list)$scale), reference_dist_list$scale, tolerance = 5e-05)
  }
  bridged_t <- survreg(
    Surv(time, status) ~ x + z,
    data = data,
    dist = survreg.distributions$t,
    max_iter = 150,
    eps = 1e-10
  )
  reference_t <- survival::survreg(
    survival::Surv(time, status) ~ x + z,
    data = data,
    dist = survival::survreg.distributions$t,
    control = survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)
  )
  expect_equal(unname(coef(bridged_t)), unname(coef(reference_t)), tolerance = 1e-03)
  expect_equal(as.numeric(summary(bridged_t)$scale), reference_t$scale, tolerance = 1e-03)
  expect_equal(as.numeric(logLik(bridged_t)), reference_t$loglik[[2L]], tolerance = 1e-05)
  expect_equal(
    unname(predict(bridged_t, newdata, type = "response")),
    unname(stats::predict(reference_t, newdata, type = "response")),
    tolerance = 1e-03
  )
  for (resid_type in c("response", "deviance", "working")) {
    expect_equal(
      unname(residuals(bridged_t, type = resid_type)),
      unname(residuals(reference_t, type = resid_type)),
      tolerance = if (identical(resid_type, "working")) 1e-02 else 5e-03
    )
  }
})

test_that("Kaplan-Meier and log-rank bridge results agree with R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  expect_survdiff_equal <- function(bridged, reference) {
    frame <- as.data.frame(bridged)
    reference_observed <- reference$obs
    reference_expected <- reference$exp
    if (is.matrix(reference_observed)) {
      reference_observed <- rowSums(reference_observed)
      reference_expected <- rowSums(reference_expected)
    }
    expect_equal(frame$observed, unname(reference_observed), tolerance = 1e-06)
    expect_equal(frame$expected, unname(reference_expected), tolerance = 1e-06)
    reference_variance <- if (is.matrix(reference$var)) {
      diag(reference$var)
    } else {
      reference$var
    }
    expect_equal(frame$variance, unname(reference_variance), tolerance = 1e-06)
    expect_equal(as.numeric(bridged$statistic), reference$chisq, tolerance = 1e-06)
    expect_equal(as.numeric(bridged$p_value), reference$pvalue, tolerance = 1e-06)
  }

  data <- data.frame(
    time = c(1, 2, 2, 3, 4, 5, 6, 7),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    group = c("A", "A", "B", "B", "A", "B", "A", "B"),
    keep = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE)
  )

  bridged_fit <- survfit(Surv(time, status) ~ group, data = data, conf.type = "log")
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))
  reference_fit <- reference_survfit(
    survival::Surv(time, status) ~ group,
    data = data,
    conf.type = "log"
  )
  reference_summary <- summary(reference_fit, censored = TRUE)
  reference_frame <- data.frame(
    strata = sub("^group=", "", as.character(reference_summary$strata)),
    time = reference_summary$time,
    n.risk = reference_summary$n.risk,
    n.event = reference_summary$n.event,
    n.censor = reference_summary$n.censor,
    surv = reference_summary$surv,
    std.err = reference_summary$std.err,
    lower = reference_summary$lower,
    upper = reference_summary$upper
  )
  bridged_frame <- as.data.frame(bridged_fit)

  expect_equal(bridged_frame$strata, reference_frame$strata)
  for (column in c("time", "n.risk", "n.event", "n.censor", "surv", "std.err")) {
    expect_equal(bridged_frame[[column]], reference_frame[[column]], tolerance = 1e-06)
  }
  for (column in c("lower", "upper")) {
    expect_equal(is.na(bridged_frame[[column]]), is.na(reference_frame[[column]]))
    finite_rows <- !is.na(reference_frame[[column]])
    expect_equal(
      bridged_frame[[column]][finite_rows],
      reference_frame[[column]][finite_rows],
      tolerance = 1e-06
    )
  }

  ordered_survfit_data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6),
    status = c(1, 0, 1, 1, 0, 1),
    group = c("treated", "treated", "control", "control", "treated", "control")
  )
  ordered_bridged_fit <- survfit(
    Surv(time, status) ~ group,
    data = ordered_survfit_data,
    se.fit = FALSE
  )
  ordered_reference_fit <- reference_survfit(
    survival::Surv(time, status) ~ group,
    data = ordered_survfit_data,
    se.fit = FALSE
  )
  ordered_reference_strata <- sub("^group=", "", names(ordered_reference_fit$strata))
  expect_equal(names(ordered_bridged_fit), ordered_reference_strata)
  expect_equal(unique(as.data.frame(ordered_bridged_fit)$strata), ordered_reference_strata)

  ordered_direct_fit <- survfit(
    Surv(ordered_survfit_data$time, ordered_survfit_data$status),
    group = ordered_survfit_data$group,
    se.fit = FALSE
  )
  expect_equal(names(ordered_direct_fit), c("treated", "control"))

  bridged_diff <- survdiff(Surv(time, status) ~ group, data = data)
  reference_diff <- survival::survdiff(survival::Surv(time, status) ~ group, data = data)
  bridged_diff_frame <- as.data.frame(bridged_diff)
  bridged_subset_diff <- survdiff(Surv(time, status) ~ group, data = data, subset = keep)
  reference_subset_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data = data,
    subset = keep
  )
  direct_group <- data$group
  direct_group[2L] <- NA
  direct_diff <- survdiff(
    Surv(data$time, data$status),
    group = direct_group,
    subset = c(rep(TRUE, 7), FALSE),
    na.action = stats::na.omit
  )
  direct_reference <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data = data[c(1, 3:7), ]
  )
  direct_diff_frame <- as.data.frame(direct_diff)

  expect_equal(bridged_diff_frame$observed, unname(reference_diff$obs), tolerance = 1e-06)
  expect_equal(bridged_diff_frame$expected, unname(reference_diff$exp), tolerance = 1e-06)
  expect_equal(direct_diff_frame$observed, unname(direct_reference$obs), tolerance = 1e-06)
  expect_equal(direct_diff_frame$expected, unname(direct_reference$exp), tolerance = 1e-06)
  expect_equal(
    bridged_diff_frame$variance,
    unname(diag(reference_diff$var)),
    tolerance = 1e-06
  )
  expect_equal(direct_diff_frame$variance, unname(diag(direct_reference$var)), tolerance = 1e-06)
  expect_equal(as.numeric(bridged_diff$statistic), reference_diff$chisq, tolerance = 1e-06)
  expect_equal(as.numeric(bridged_diff$p_value), reference_diff$pvalue, tolerance = 1e-06)
  expect_equal(as.numeric(direct_diff$statistic), direct_reference$chisq, tolerance = 1e-06)
  expect_equal(as.numeric(direct_diff$p_value), direct_reference$pvalue, tolerance = 1e-06)
  expect_survdiff_equal(bridged_subset_diff, reference_subset_diff)

  stratified_diff_data <- data.frame(
    time = c(1, 1, 2, 2, 3, 3),
    status = c(1, 0, 0, 1, 1, 1),
    group = c("treated", "treated", "control", "control", "treated", "control"),
    site = c("north", "south", "north", "south", "north", "south")
  )
  bridged_stratified_diff <- survdiff(
    Surv(time, status) ~ group + strata(site),
    data = stratified_diff_data,
    rho = 0.5
  )
  reference_stratified_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group + strata(site),
    data = stratified_diff_data,
    rho = 0.5
  )
  expect_survdiff_equal(bridged_stratified_diff, reference_stratified_diff)

  offset_diff_data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    expected = c(0.9, 0.8, 0.7, 0.6)
  )
  bridged_offset_diff <- survdiff(
    Surv(time, status) ~ offset(expected),
    data = offset_diff_data
  )
  reference_offset_diff <- survival::survdiff(
    survival::Surv(time, status) ~ offset(expected),
    data = offset_diff_data
  )
  expect_survdiff_equal(bridged_offset_diff, reference_offset_diff)
  bridged_weighted_offset_diff <- survdiff(
    Surv(time, status) ~ offset(expected),
    data = offset_diff_data,
    rho = 0.5
  )
  reference_weighted_offset_diff <- survival::survdiff(
    survival::Surv(time, status) ~ offset(expected),
    data = offset_diff_data,
    rho = 0.5
  )
  expect_survdiff_equal(bridged_weighted_offset_diff, reference_weighted_offset_diff)
  expect_error(
    survdiff(
      Surv(time, status) ~ group + offset(expected),
      data = transform(offset_diff_data, group = c("a", "a", "b", "b"))
    ),
    "Cannot have both an offset and groups"
  )
})
