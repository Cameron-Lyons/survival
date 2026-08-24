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
  plot_to_null_device <- function(x) {
    grDevices::pdf(NULL)
    on.exit(grDevices::dev.off())
    plot(x)
  }
  expect_equal(
    plot_to_null_device(response),
    plot_to_null_device(survival::Surv(data$time, data$status))
  )
  for (graphics_method in list(
    function(x) barplot(x),
    function(x) density(x),
    function(x) hist(x),
    function(x) identify(x),
    function(x) image(x),
    function(x) lines(x),
    function(x) pairs(x),
    function(x) points(x),
    function(x) text(x)
  )) {
    expect_error(graphics_method(response), "not defined for a Surv object", fixed = TRUE)
  }
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
  for (graphics_method in list(
    function(x) hist(x),
    function(x) identify(x),
    function(x) image(x),
    function(x) lines(x),
    function(x) pairs(x),
    function(x) points(x),
    function(x) text(x)
  )) {
    expect_error(
      graphics_method(surv2_frame_response),
      "not defined for a Surv2 object",
      fixed = TRUE
    )
  }
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
  expect_warning(
    explicit_multistate_response <- Surv(
      c(1, 2, 3),
      factor(c("censor", "relapse", "death")),
      type = "mstate"
    ),
    "type= 'mstate' is deprecated"
  )
  reference_explicit_multistate_response <- suppressWarnings(
    survival::Surv(
      c(1, 2, 3),
      factor(c("censor", "relapse", "death")),
      type = "mstate"
    )
  )
  expect_equal(explicit_multistate_response, reference_explicit_multistate_response)
  expect_warning(
    explicit_numeric_multistate_response <- Surv(
      c(1, 2),
      c(0, 1),
      type = "mstate"
    ),
    "type= 'mstate' is deprecated"
  )
  reference_numeric_multistate_response <- suppressWarnings(
    survival::Surv(
      c(1, 2),
      c(0, 1),
      type = "mstate"
    )
  )
  expect_equal(
    .as_native_surv(explicit_numeric_multistate_response),
    reference_numeric_multistate_response
  )
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
  expect_warning(
    explicit_multistate_counting_response <- Surv(
      c(0, 0),
      c(1, 2),
      factor(c("a", "b")),
      type = "mstate"
    ),
    "type= 'mstate' is deprecated"
  )
  reference_explicit_multistate_counting_response <- suppressWarnings(
    survival::Surv(
      c(0, 0),
      c(1, 2),
      factor(c("a", "b")),
      type = "mstate"
    )
  )
  expect_equal(
    explicit_multistate_counting_response,
    reference_explicit_multistate_counting_response
  )
  expect_warning(
    explicit_numeric_multistate_counting_response <- Surv(
      c(0, 0),
      c(1, 2),
      c(0, 1),
      type = "mstate"
    ),
    "type= 'mstate' is deprecated"
  )
  reference_numeric_multistate_counting_response <- suppressWarnings(
    survival::Surv(
      c(0, 0),
      c(1, 2),
      c(0, 1),
      type = "mstate"
    )
  )
  expect_equal(
    .as_native_surv(explicit_numeric_multistate_counting_response),
    reference_numeric_multistate_counting_response
  )
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
  expect_identical(
    attr(Surv2(c(1, 2), c("a", "a"), repeated = "first"), "repeated"),
    "first"
  )
  reference_surv2_constructor <- get("Surv2", envir = asNamespace("survival"))
  capture_surv2 <- function(fun, args) {
    captured_warnings <- character()
    result <- tryCatch(
      withCallingHandlers(
        list(kind = "value", value = do.call(fun, args)),
        warning = function(w) {
          captured_warnings <<- c(captured_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        list(kind = "error", message = conditionMessage(e), class = class(e))
      }
    )
    c(result, list(warnings = captured_warnings))
  }
  custom_surv2_time <- structure(
    c(1, 2),
    names = c("x", "y"),
    note = "kept"
  )
  blank_surv2_state <- factor(
    c("baseline", ""),
    levels = c("baseline", "")
  )
  surv2_cases <- list(
    list(time = c(1, 2, 3), event = c("a", "b", "c")),
    list(
      time = c(1, 2, 3),
      event = factor(c("a", "b", "a"), levels = c("z", "a", "b"))
    ),
    list(time = c(1, 2, 3), event = c(FALSE, TRUE, NA)),
    list(time = c(1, 2), event = c(NA_real_, NA_real_)),
    list(time = c(1, 2, 3), event = c(NA_real_, 0, 1)),
    list(time = c(1, 2), event = c(NA, NA)),
    list(time = c(1, 2, 3), event = c(NA, FALSE, TRUE)),
    list(event = c("a", "b")),
    list(time = c(1, 2), repeated = c(TRUE, FALSE)),
    list(time = c(1, 2), event = c("a", "b"), repeated = NA),
    list(time = c(1, 2), event = c("a", "b"), repeated = NA_character_),
    list(time = c(1, 2), event = c("a", "b"), repeated = 1),
    list(time = c(1, 2), event = c("a", "b"), repeated = c(TRUE, FALSE)),
    list(time = c(1, 2), event = c("a", "b"), repeated = "first"),
    list(time = numeric(), event = character()),
    list(time = custom_surv2_time, event = c("a", "b")),
    list(time = matrix(1:4, nrow = 2L), event = c("a", "b", "c", "d")),
    list(time = as.difftime(c(1, 2), units = "days"), event = c("a", "b")),
    list(time = c("1", "2"), event = c("a", "b")),
    list(time = c(1, 2), event = "a"),
    list(time = c(1, 2), event = blank_surv2_state),
    list(time = c(NA, NaN, Inf, -Inf), event = c("a", "b", "c", "d"))
  )
  for (case in surv2_cases) {
    expect_identical(
      capture_surv2(Surv2, case),
      capture_surv2(reference_surv2_constructor, case)
    )
  }
  expect_false(is.Surv(1:3))
  expect_false(is.Surv(surv2_response))

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
    fromtimeline(Surv2(time, state) ~ z + x, data = timeline_data, id = id),
    survival::fromtimeline(
      survival::Surv2(time, state) ~ z + x,
      data = timeline_data,
      id = id
    )
  )
  timeline_missing_data <- data.frame(
    id = c(1, 2, 1, 2, 1, 2),
    time = c(0, 0, 2, 3, 5, 6),
    state = factor(
      c("entry", "entry", "ill", "ill", "death", "death"),
      levels = c("censor", "entry", "ill", "death")
    ),
    x = c(10, 20, NA, NA, 12, 22),
    z = factor(c("A", "B", NA, NA, "C", "D")),
    observed = c(TRUE, FALSE, NA, NA, TRUE, TRUE),
    visit = as.Date("2026-01-01") + c(0, 1, NA, NA, 4, 5)
  )
  expect_equal(
    fromtimeline(
      Surv2(time, state) ~ x + z + observed + visit,
      data = timeline_missing_data,
      id = id
    ),
    survival::fromtimeline(
      survival::Surv2(time, state) ~ x + z + observed + visit,
      data = timeline_missing_data,
      id = id
    )
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
  legacy_timeline <- totimeline(
    Surv(start, stop, state) ~ z + x,
    data = counting_data,
    id = id,
    istate = istate
  )
  expect_identical(names(legacy_timeline), c("stop", "state", "z", "x", "id"))
  expect_equal(legacy_timeline$stop, c(0, 2, 5, 0, 3))
  expect_equal(
    as.character(legacy_timeline$state),
    c("entry", "ill", "death", "entry", "censor")
  )
  expect_equal(legacy_timeline$id, c(1, 1, 1, 2, 2))
  counting_no_istate <- counting_data[c("id", "start", "stop", "state", "z", "x")]
  legacy_timeline_without_istate <- totimeline(
    Surv(start, stop, state) ~ z + x,
    data = counting_no_istate,
    id = id
  )
  expect_s3_class(legacy_timeline_without_istate, "data.frame")
  expect_equal(nrow(legacy_timeline_without_istate), 5L)

  timeline_right <- data.frame(
    id = c(1, 2, 1, 2, 1, 2),
    time = c(0, 0, 2, 3, 5, 6),
    status = c(0, 0, 1, 1, 1, 0),
    z = c("A", "B", "A", "B", "A", "B"),
    x = c(10, 20, 11, 21, 12, 22)
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
  weighted_yates_data <- transform(
    yates_data,
    y = c(1, 2.4, 2.8, 4.7, 5.1, 6.5),
    x = c(-2, -1, 0, 1, 2, 3),
    wt = c(1, 3, 2, 1, 4, 2)
  )
  weighted_yates_fit <- stats::lm(
    y ~ group + x,
    data = weighted_yates_data,
    weights = wt,
    model = TRUE
  )
  expect_equal(
    yates(weighted_yates_fit, "group"),
    survival::yates(weighted_yates_fit, "group"),
    tolerance = 1e-12
  )
  expect_equal(
    yates(weighted_yates_fit, "group", levels = c("C", "A")),
    survival::yates(weighted_yates_fit, "group", levels = c("C", "A")),
    tolerance = 1e-12
  )
  expect_equal(
    yates(weighted_yates_fit, "group", test = "pairwise"),
    survival::yates(weighted_yates_fit, "group", test = "pairwise"),
    tolerance = 1e-12
  )
  expect_equal(
    yates(
      weighted_yates_fit,
      "group",
      levels = c("C", "A"),
      test = "pairwise"
    ),
    survival::yates(
      weighted_yates_fit,
      "group",
      levels = c("C", "A"),
      test = "pairwise"
    ),
    tolerance = 1e-12
  )
  yates_population_data <- data.frame(
    y = c(
      0.2, 1.1, -0.4, 0.8, 1.7, 0.5,
      1.3, 0.1, 1.9, 0.6, 2.1, 1.2,
      2.4, 1.5, 2.8, 1.8, 3.0, 2.2
    ),
    group = factor(rep(c("A", "B", "C"), each = 6L)),
    z = factor(rep(c("u", "v"), 9L)),
    x = seq(-2, 2, length.out = 18L),
    wt = rep(c(1, 2, 3), 6L)
  )
  factorial_yates_fit <- stats::lm(
    y ~ group * z,
    data = yates_population_data,
    weights = wt,
    model = TRUE
  )
  for (population_name in c("factorial", "yates")) {
    expect_equal(
      yates(factorial_yates_fit, "group", population = population_name),
      survival::yates(
        factorial_yates_fit,
        "group",
        population = population_name
      ),
      tolerance = 1e-12
    )
  }
  sas_yates_fit <- stats::lm(
    y ~ group * z + x,
    data = yates_population_data,
    weights = wt,
    model = TRUE
  )
  expect_equal(
    yates(sas_yates_fit, "group", population = "sas"),
    survival::yates(sas_yates_fit, "group", population = "sas"),
    tolerance = 1e-12
  )
  yates_population <- data.frame(
    z = factor(c("u", "v", "v"), levels = levels(yates_population_data$z)),
    x = c(-1, 0.5, 2)
  )
  expect_equal(
    yates(sas_yates_fit, "group", population = yates_population),
    survival::yates(
      sas_yates_fit,
      "group",
      population = yates_population
    ),
    tolerance = 1e-12
  )
  numeric_yates_fit <- stats::lm(
    y ~ x + z,
    data = yates_population_data,
    weights = wt,
    model = TRUE
  )
  for (test_name in c("global", "pairwise")) {
    expect_equal(
      yates(numeric_yates_fit, "x", levels = c(-1, 0, 1), test = test_name),
      survival::yates(
        numeric_yates_fit,
        "x",
        levels = c(-1, 0, 1),
        test = test_name
      ),
      tolerance = 1e-12
    )
  }
  glm_yates_data <- transform(
    yates_population_data,
    outcome = c(0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1),
    count = c(1, 2, 1, 3, 2, 4, 1, 3, 5, 2, 4, 6, 3, 5, 4, 7, 6, 8),
    positive = c(
      0.7, 1.4, 0.9, 1.8, 1.2, 2.1,
      1.1, 2.0, 2.8, 1.5, 2.4, 3.2,
      1.9, 3.0, 2.6, 3.8, 3.4, 4.2
    )
  )
  binomial_yates_fit <- stats::glm(
    outcome ~ group + x,
    data = glm_yates_data,
    family = stats::binomial(),
    model = TRUE
  )
  for (predict_name in c("linear", "link")) {
    for (test_name in c("global", "pairwise")) {
      expect_equal(
        yates(
          binomial_yates_fit,
          "group",
          predict = predict_name,
          test = test_name
        ),
        survival::yates(
          binomial_yates_fit,
          "group",
          predict = predict_name,
          test = test_name
        ),
        tolerance = 1e-12
      )
    }
  }
  poisson_yates_fit <- stats::glm(
    count ~ group + z + x,
    data = glm_yates_data,
    family = stats::poisson(),
    model = TRUE
  )
  for (test_name in c("global", "pairwise")) {
    expect_equal(
      yates(
        poisson_yates_fit,
        "group",
        population = "sas",
        predict = "link",
        test = test_name
      ),
      survival::yates(
        poisson_yates_fit,
        "group",
        population = "sas",
        predict = "link",
        test = test_name
      ),
      tolerance = 1e-12
    )
  }
  inverse_gaussian_n <- 90L
  inverse_gaussian_data <- data.frame(
    group = factor(rep(c("A", "B", "C"), each = inverse_gaussian_n / 3L)),
    x = rep(seq(-1, 1, length.out = inverse_gaussian_n / 3L), 3L)
  )
  inverse_gaussian_eta <- 0.7 +
    c(A = 0.05, B = 0.1, C = 0.15)[inverse_gaussian_data$group] +
    0.05 * inverse_gaussian_data$x
  inverse_gaussian_data$positive <-
    1 / sqrt(inverse_gaussian_eta) + 0.02 * sin(seq_len(inverse_gaussian_n))
  response_yates_fits <- suppressWarnings(list(
    logit = stats::glm(
      outcome ~ group + x,
      data = glm_yates_data,
      family = stats::binomial("logit"),
      model = TRUE
    ),
    probit = stats::glm(
      outcome ~ group + x,
      data = glm_yates_data,
      family = stats::binomial("probit"),
      model = TRUE
    ),
    cauchit = stats::glm(
      outcome ~ group + x,
      data = glm_yates_data,
      family = stats::binomial("cauchit"),
      model = TRUE
    ),
    cloglog = stats::glm(
      outcome ~ group + x,
      data = glm_yates_data,
      family = stats::binomial("cloglog"),
      model = TRUE
    ),
    log = stats::glm(
      count ~ group + x,
      data = glm_yates_data,
      weights = wt,
      family = stats::poisson("log"),
      model = TRUE
    ),
    sqrt = stats::glm(
      count ~ group + x,
      data = glm_yates_data,
      family = stats::poisson("sqrt"),
      model = TRUE
    ),
    identity = stats::glm(
      positive ~ group + x,
      data = glm_yates_data,
      family = stats::gaussian("identity"),
      model = TRUE
    ),
    inverse = stats::glm(
      positive ~ group + x,
      data = glm_yates_data,
      family = stats::Gamma("inverse"),
      model = TRUE
    ),
    `1/mu^2` = stats::glm(
      positive ~ group + x,
      data = inverse_gaussian_data,
      family = stats::inverse.gaussian("1/mu^2"),
      model = TRUE
    )
  ))
  set.seed(20260822)
  response_probe <- .yates_model_term(
    fit = response_yates_fits$logit,
    term = "group",
    population = "data",
    levels = NULL,
    levels_missing = TRUE,
    test = "global",
    predict = "response",
    method = "direct",
    nsim = 40
  )
  expect_type(response_probe, "list")
  expect_false("cmat" %in% names(response_probe))
  for (fit_name in names(response_yates_fits)) {
    test_names <- if (fit_name == "logit") c("global", "pairwise") else "global"
    for (test_name in test_names) {
      set.seed(20260822)
      expected <- suppressWarnings(survival::yates(
        response_yates_fits[[fit_name]],
        "group",
        predict = "response",
        test = test_name,
        nsim = 200
      ))
      set.seed(20260822)
      actual <- suppressWarnings(yates(
        response_yates_fits[[fit_name]],
        "group",
        predict = "response",
        test = test_name,
        nsim = 200
      ))
      expect_equal(
        actual,
        expected,
        tolerance = 2e-10,
        info = paste(fit_name, test_name)
      )
    }
  }

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

  compare_yates <- function(actual, expected, tolerance = 1e-9) {
    fields <- setdiff(names(expected), "call")
    expect_setequal(setdiff(names(actual), "call"), fields)
    for (field in fields) {
      expect_equal(actual[[field]], expected[[field]], tolerance = tolerance, info = field)
    }
    expect_s3_class(actual, "yates")
  }
  yates_cox_model_data <- survival::lung[
    stats::complete.cases(survival::lung[, c("time", "status", "age", "sex", "ph.ecog")]),
    c("time", "status", "age", "sex", "ph.ecog")
  ]
  yates_cox_model_data$status <- as.integer(yates_cox_model_data$status == 2L)
  yates_cox_model_data$group <- factor(
    rep(c("A", "B", "C"), length.out = nrow(yates_cox_model_data))
  )
  native_yates_cox_fit <- survival::coxph(
    survival::Surv(time, status) ~ group + age + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  local_yates_cox_fit <- coxph(
    Surv(time, status) ~ group + age + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  for (predict_name in c("linear", "lp")) {
    for (test_name in c("global", "pairwise")) {
      expected <- survival::yates(
        native_yates_cox_fit,
        "group",
        predict = predict_name,
        test = test_name
      )
      compare_yates(
        yates(
          native_yates_cox_fit,
          "group",
          predict = predict_name,
          test = test_name
        ),
        expected,
        tolerance = 1e-12
      )
      compare_yates(
        yates(
          local_yates_cox_fit,
          "group",
          predict = predict_name,
          test = test_name
        ),
        expected
      )
    }
  }
  formula_yates_cox_data <- transform(
    yates_cox_model_data,
    sex_factor = factor(sex, labels = c("male", "female"))
  )
  native_formula_yates_fit <- survival::coxph(
    survival::Surv(time, status) ~ group * sex_factor + age + ph.ecog,
    data = formula_yates_cox_data,
    model = TRUE,
    x = TRUE
  )
  local_formula_yates_fit <- coxph(
    Surv(time, status) ~ group * sex_factor + age + ph.ecog,
    data = formula_yates_cox_data,
    model = TRUE,
    x = TRUE
  )
  formula_yates_cases <- list(
    formula = list(term = ~group),
    interaction = list(term = "group:sex_factor", test = "pairwise"),
    mixed = list(term = ~group + age, levels = list(age = c(50, 70))),
    selected = list(
      term = ~group + sex_factor,
      levels = data.frame(
        group = c("C", "A"),
        sex_factor = c("female", "male")
      )
    ),
    numeric = list(term = 1L)
  )
  for (case_name in names(formula_yates_cases)) {
    arguments <- formula_yates_cases[[case_name]]
    expected <- do.call(
      survival::yates,
      c(list(fit = native_formula_yates_fit), arguments)
    )
    compare_yates(
      do.call(yates, c(list(fit = native_formula_yates_fit), arguments)),
      expected,
      tolerance = 1e-12
    )
    compare_yates(
      do.call(yates, c(list(fit = local_formula_yates_fit), arguments)),
      expected,
      tolerance = 1e-8
    )
  }
  native_spline_yates_fit <- survival::coxph(
    survival::Surv(time, status) ~
      pspline(age, theta = 0.5, nterm = 6) + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  local_spline_yates_fit <- coxph(
    Surv(time, status) ~
      pspline(age, theta = 0.5, nterm = 6) + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  spline_yates_levels <- c(50, 60, 70)
  spline_yates_cases <- list(
    variable = list(term = "age"),
    formula = list(term = ~age),
    transformed = list(term = "pspline(age)", test = "pairwise")
  )
  for (case_name in names(spline_yates_cases)) {
    arguments <- c(
      spline_yates_cases[[case_name]],
      list(levels = spline_yates_levels)
    )
    expected <- do.call(
      survival::yates,
      c(list(fit = native_spline_yates_fit), arguments)
    )
    compare_yates(
      do.call(yates, c(list(fit = local_spline_yates_fit), arguments)),
      expected,
      tolerance = 1e-7
    )
  }
  set.seed(20260822)
  spline_risk_expected <- survival::yates(
    native_spline_yates_fit,
    "age",
    levels = spline_yates_levels,
    predict = "risk",
    nsim = 100
  )
  set.seed(20260822)
  compare_yates(
    yates(
      local_spline_yates_fit,
      "age",
      levels = spline_yates_levels,
      predict = "risk",
      nsim = 100
    ),
    spline_risk_expected,
    tolerance = 1e-7
  )
  native_nsk_yates_fit <- survival::coxph(
    survival::Surv(time, status) ~ nsk(age, df = 3) + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  local_nsk_yates_fit <- coxph(
    Surv(time, status) ~ nsk(age, df = 3) + sex + ph.ecog,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  expect_equal(
    coef(local_nsk_yates_fit),
    coef(native_nsk_yates_fit),
    tolerance = 1e-8
  )
  expect_equal(
    unname(model.matrix(local_nsk_yates_fit)),
    unname(model.matrix(native_nsk_yates_fit)),
    tolerance = 1e-12
  )
  compare_yates(
    yates(local_nsk_yates_fit, "age", levels = spline_yates_levels),
    survival::yates(native_nsk_yates_fit, "age", levels = spline_yates_levels),
    tolerance = 1e-7
  )
  sas_yates_cox_expected <- survival::yates(
    native_yates_cox_fit,
    "group",
    population = "sas"
  )
  compare_yates(
    yates(native_yates_cox_fit, "group", population = "sas"),
    sas_yates_cox_expected,
    tolerance = 1e-12
  )
  compare_yates(
    yates(local_yates_cox_fit, "group", population = "sas"),
    sas_yates_cox_expected
  )
  yates_cox_model_data$sex_factor <- factor(
    yates_cox_model_data$sex,
    labels = c("male", "female")
  )
  native_factorial_cox_fit <- survival::coxph(
    survival::Surv(time, status) ~ group + sex_factor,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  local_factorial_cox_fit <- coxph(
    Surv(time, status) ~ group + sex_factor,
    data = yates_cox_model_data,
    model = TRUE,
    x = TRUE
  )
  compare_yates(
    yates(local_factorial_cox_fit, "group", population = "factorial"),
    survival::yates(native_factorial_cox_fit, "group", population = "factorial")
  )
  for (test_name in c("global", "pairwise")) {
    set.seed(20260822)
    expected <- survival::yates(
      native_yates_cox_fit,
      "group",
      predict = "risk",
      test = test_name,
      nsim = 400
    )
    set.seed(20260822)
    compare_yates(
      yates(
        native_yates_cox_fit,
        "group",
        predict = "risk",
        test = test_name,
        nsim = 400
      ),
      expected,
      tolerance = 1e-10
    )
    set.seed(20260822)
    compare_yates(
      yates(
        local_yates_cox_fit,
        "group",
        predict = "risk",
        test = test_name,
        nsim = 400
      ),
      expected,
      tolerance = 1e-8
    )
  }
  set.seed(20260822)
  selected_risk_expected <- survival::yates(
    native_yates_cox_fit,
    "group",
    levels = c("C", "A"),
    test = "pairwise",
    predict = "risk",
    nsim = 400
  )
  set.seed(20260822)
  compare_yates(
    yates(
      local_yates_cox_fit,
      "group",
      levels = c("C", "A"),
      test = "pairwise",
      predict = "risk",
      nsim = 400
    ),
    selected_risk_expected,
    tolerance = 1e-8
  )

  native_survival_setup <- yates_setup(
    yates_cox_fit,
    predict = "survival",
    options = list(rmean = 9)
  )
  reference_survival_setup <- survival::yates_setup(
    yates_cox_fit,
    predict = "survival",
    options = list(rmean = 9)
  )
  expect_equal(
    native_survival_setup$predict(c(-1, 0, 1)),
    reference_survival_setup$predict(c(-1, 0, 1)),
    tolerance = 1e-12
  )
  survival_probe <- .yates_model_term(
    fit = native_yates_cox_fit,
    term = "group",
    population = "data",
    levels = NULL,
    levels_missing = TRUE,
    test = "global",
    predict = "survival",
    method = "direct",
    nsim = 20,
    options = list(rmean = 365)
  )
  expect_false(is.null(survival_probe))

  for (test_name in c("global", "pairwise")) {
    set.seed(20260822)
    survival_expected <- survival::yates(
      native_yates_cox_fit,
      "group",
      predict = "survival",
      options = list(rmean = 365),
      test = test_name,
      nsim = 200
    )
    set.seed(20260822)
    compare_yates(
      yates(
        native_yates_cox_fit,
        "group",
        predict = "survival",
        options = list(rmean = 365),
        test = test_name,
        nsim = 200
      ),
      survival_expected,
      tolerance = 2e-10
    )
    set.seed(20260822)
    compare_yates(
      yates(
        local_yates_cox_fit,
        "group",
        predict = "survival",
        options = list(rmean = 365),
        test = test_name,
        nsim = 200
      ),
      survival_expected,
      tolerance = 2e-8
    )
  }

  weighted_survival_data <- transform(
    yates_cox_model_data,
    weight = 1 + (seq_len(nrow(yates_cox_model_data)) %% 5L) / 10
  )
  weighted_survival_fit <- survival::coxph(
    survival::Surv(time, status) ~ group + age + sex + ph.ecog,
    data = weighted_survival_data,
    weights = weight,
    model = TRUE,
    x = TRUE
  )
  set.seed(20260823)
  weighted_survival_expected <- survival::yates(
    weighted_survival_fit,
    "group",
    predict = "survival",
    options = list(rmean = 400),
    nsim = 100
  )
  set.seed(20260823)
  compare_yates(
    yates(
      weighted_survival_fit,
      "group",
      predict = "survival",
      options = list(rmean = 400),
      nsim = 100
    ),
    weighted_survival_expected,
    tolerance = 2e-10
  )

  counting_survival_data <- data.frame(
    start = c(0, 0, 1, 2, 0, 3, 4, 0),
    stop = c(2, 3, 4, 5, 6, 7, 8, 9),
    status = c(1, 0, 1, 1, 0, 1, 1, 0),
    x = c(0, 0.2, -0.1, 0.5, 0.8, -0.3, 0.1, 0.4),
    group = factor(rep(c("A", "B"), 4L))
  )
  native_counting_survival_fit <- survival::coxph(
    survival::Surv(start, stop, status) ~ group + x,
    data = counting_survival_data,
    model = TRUE,
    x = TRUE
  )
  local_counting_survival_fit <- coxph(
    Surv(start, stop, status) ~ group + x,
    data = counting_survival_data,
    model = TRUE,
    x = TRUE
  )
  set.seed(20260824)
  counting_survival_expected <- survival::yates(
    native_counting_survival_fit,
    "group",
    predict = "survival",
    options = list(rmean = 6),
    nsim = 60
  )
  set.seed(20260824)
  compare_yates(
    yates(
      local_counting_survival_fit,
      "group",
      predict = "survival",
      options = list(rmean = 6),
      nsim = 60
    ),
    counting_survival_expected,
    tolerance = 1e-7
  )

  compare_aareg <- function(actual, expected) {
    fields <- setdiff(names(expected), "call")
    expect_setequal(setdiff(names(actual), "call"), fields)
    for (field in fields) {
      expect_equal(actual[[field]], expected[[field]], tolerance = 1e-10, info = field)
    }
    expect_s3_class(actual, "aareg")
  }
  aareg_data <- data.frame(
    time = c(1, 2, 2, 3, 4, 4),
    status = c(1, 1, 1, 1, 0, 1),
    x = c(0, 1, 2, 1, 3, -1),
    z = c(1, 0, 1, 2, -1, 0),
    weight = c(1, 2, 0.5, 1.5, 1, 3),
    group = factor(c("low", "high", "low", "high", "low", "high")),
    cluster = c("a", "a", "b", "b", "c", "c")
  )
  bridged_aareg <- aareg(
    survival::Surv(time, status) ~ x + z,
    data = aareg_data,
    weights = weight,
    cluster = cluster,
    nmin = 1,
    model = TRUE,
    x = TRUE,
    y = TRUE
  )
  reference_aareg <- survival::aareg(
    survival::Surv(time, status) ~ x + z,
    data = aareg_data,
    weights = weight,
    cluster = cluster,
    nmin = 1,
    model = TRUE,
    x = TRUE,
    y = TRUE
  )
  compare_aareg(bridged_aareg, reference_aareg)
  compare_aareg(
    aareg(
      survival::Surv(time, status) ~ x + group + cluster(cluster),
      data = aareg_data,
      nmin = 1,
      model = TRUE
    ),
    survival::aareg(
      survival::Surv(time, status) ~ x + group + cluster(cluster),
      data = aareg_data,
      nmin = 1,
      model = TRUE
    )
  )
  aareg_counting <- transform(
    aareg_data,
    start = c(0, 0, 1, 0, 2, 1),
    stop = c(1, 3, 3, 4, 4, 2)
  )
  compare_aareg(
    aareg(
      survival::Surv(start, stop, status) ~ x + z + cluster(cluster),
      data = aareg_counting,
      weights = weight,
      nmin = 1,
      taper = c(1, 2),
      x = TRUE,
      y = TRUE
    ),
    survival::aareg(
      survival::Surv(start, stop, status) ~ x + z + cluster(cluster),
      data = aareg_counting,
      weights = weight,
      nmin = 1,
      taper = c(1, 2),
      x = TRUE,
      y = TRUE
    )
  )
  single_risk_aareg_data <- data.frame(
    time = 1:5,
    status = rep(1, 5),
    x = c(
      -0.626453810742332,
      0.183643324222082,
      -0.835628612410047,
      1.59528080213779,
      0.32950777181536
    ),
    cluster = letters[1:5]
  )
  compare_aareg(
    aareg(
      survival::Surv(time, status) ~ x,
      data = single_risk_aareg_data,
      cluster = cluster,
      nmin = 1
    ),
    survival::aareg(
      survival::Surv(time, status) ~ x,
      data = single_risk_aareg_data,
      cluster = cluster,
      nmin = 1
    )
  )
  counting_single_risk_aareg_data <- data.frame(
    start = c(1, 8, 7, 4, 8, 2, 6, 3, 10, 0, 5, 0, 7, 4),
    stop = c(5, 11, 9, 5, 9, 6, 11, 7, 12, 1, 6, 1, 9, 7),
    status = c(0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1),
    x = c(
      0x1.df8959eba015bp-4, 0x1.7e35d530f3d76p-1, -0x1.e45db3aa1c2c7p-1,
      -0x1.c1f630ff7d53fp-3, -0x1.c20475dbec72cp+0, 0x1.31b80f83637f2p-1,
      -0x1.60d77db8bfca2p+0, -0x1.c975026783892p-2, -0x1.f042e4171932fp-12,
      -0x1.c1628ff5cd9ap-2, -0x1.40a81c9d0f22dp-2, -0x1.82193785cfe34p+1,
      0x1.760bcc047ad82p-2, -0x1.4d055dda44427p-1
    )
  )
  compare_aareg(
    aareg(
      survival::Surv(start, stop, status) ~ x,
      data = counting_single_risk_aareg_data,
      nmin = 1,
      test = "variance"
    ),
    survival::aareg(
      survival::Surv(start, stop, status) ~ x,
      data = counting_single_risk_aareg_data,
      nmin = 1,
      test = "variance"
    )
  )
  near_singular_aareg_data <- data.frame(
    stop = c(8, 3, 3, 3, 11, 6, 7, 6, 12, 11, 2, 5, 8, 5),
    status = c(1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0),
    x = c(
      -0x1.0e4470a02be58p+0, 0x1.dcbb47d2df9d3p-5, -0x1.0396289e6d553p+1,
      -0x1.c1f5bb739a67ap-2, 0x1.ed4f3472d121ap-6, 0x1.beeb056b3e6d1p-1,
      -0x1.21b0a9bfc9fc9p-1, 0x1.c8baba26fcc35p-1, 0x1.745d3db8988c9p-1,
      -0x1.01bb5fb1c963cp+0, -0x1.50afaadf7f29cp-1, -0x1.2a5daacea2caap-3,
      -0x1.071f6811ca916p+0, 0x1.49925b191eff7p-3
    ),
    z = c(
      0x1.5a8e6842dc111p-2, 0x1.6f2cb545912e1p-4, -0x1.aec08693db5dp-2,
      -0x1.07262051539cp+1, 0x1.a53277d204f31p-2, -0x1.2b29dac0fc4c9p-1,
      0x1.4ad21257d760cp-3, 0x1.44bb6e2af625cp-1, -0x1.71537c63c8c91p-2,
      -0x1.5b5b15fb626dbp-4, 0x1.d405da565bba5p-2, -0x1.a82341f1f43c9p+0,
      -0x1.35f9fd7daef4cp+1, 0x1.c9c729b9d983cp-2
    ),
    weight = c(0.5, 2, 1, 0.5, 2, 1, 0.5, 1, 2, 1, 1.5, 1, 0.5, 2)
  )
  compare_aareg(
    aareg(
      survival::Surv(stop, status) ~ x + z,
      data = near_singular_aareg_data,
      weights = weight,
      nmin = 2,
      test = "aalen"
    ),
    survival::aareg(
      survival::Surv(stop, status) ~ x + z,
      data = near_singular_aareg_data,
      weights = weight,
      nmin = 2,
      test = "aalen"
    )
  )
  reduced_rank_aareg_data <- data.frame(
    start = c(7, 5, 0, 6, 7, 4, 1, 9, 0, 7, 4, 7, 7, 11, 0, 5),
    stop = c(8, 10, 1, 7, 11, 5, 3, 12, 1, 11, 5, 8, 9, 12, 1, 9),
    status = c(0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1),
    x = c(
      0.573908926345705, 0.557383157775316, -0.048318399946546,
      1.77926496475137, -0.994176155979485, 0.640347136457419,
      0.14103345783844, -0.54381139957941, 0.481044344724356,
      0.515125707267044, 0.273248501950473, -0.783292341519974,
      -0.236489544926223, 0.112112836045669, -1.27275904597986,
      0.990232610649446
    ),
    z = c(
      -0.998654664215969, 0.0747227936381808, 0.351874693602948,
      -0.8879259478384, 0.537219149435208, -0.837719990909545,
      -1.66434120525646, 0.227724188582013, 0.795642091749136,
      0.318483478317856, -0.189220036134785, -0.881718032267188,
      0.358906236331899, 0.963535718907729, 0.672340443408467,
      -0.091722475837281
    ),
    weight = c(2, 3, 3, 1.5, 1, 2, 2, 1, 1.5, 2, 3, 0.5, 3, 3, 3, 1)
  )
  compare_aareg(
    aareg(
      survival::Surv(start, stop, status) ~ x + z,
      data = reduced_rank_aareg_data,
      weights = weight,
      cluster = seq_len(nrow(reduced_rank_aareg_data)),
      nmin = 0
    ),
    survival::aareg(
      survival::Surv(start, stop, status) ~ x + z,
      data = reduced_rank_aareg_data,
      weights = weight,
      cluster = seq_len(nrow(reduced_rank_aareg_data)),
      nmin = 0
    )
  )
  terminal_influence_aareg_data <- data.frame(
    time = c(4, 7, 3, 10, 12, 4, 1, 8, 11),
    status = c(1, 1, 1, 1, 1, 0, 1, 0, 0),
    x = c(
      0x1.7574d82107103p-2, 0x1.237e8d13d5376p+1, -0x1.f1ba09cb1dbbep-5,
      0x1.271ab012254a3p-1, -0x1.4a30c733b1cddp-5, 0x1.5fccc3dc3cc19p-1,
      -0x1.19c20a5bcd13p-1, -0x1.5fc5542b23161p-1, -0x1.6234537bdf792p-5
    ),
    z = c(
      -0x1.01adad7011a1dp+1, 0x1.e108d739960ebp-3, -0x1.90bb6ad82f5p-3,
      -0x1.91e0d4f14e16fp-1, 0x1.9a38509226376p+0, 0x1.3fbdf2e1bedb2p-1,
      -0x1.b3f12cc82a54bp-3, 0x1.189be9b75b29fp-2, 0x1.815bcab4524bfp+0
    ),
    weight = c(1.5, 1.5, 1, 1, 3, 0.5, 1, 3, 0.5),
    cluster = c("a", "b", "c", "b", "a", "c", "c", "b", "b")
  )
  compare_aareg(
    aareg(
      survival::Surv(time, status) ~ x + z,
      data = terminal_influence_aareg_data,
      weights = weight,
      cluster = cluster,
      nmin = 0,
      test = "nrisk"
    ),
    survival::aareg(
      survival::Surv(time, status) ~ x + z,
      data = terminal_influence_aareg_data,
      weights = weight,
      cluster = cluster,
      nmin = 0,
      test = "nrisk"
    )
  )
  bridged_aareg_cluster_override <- NULL
  expect_warning(
    bridged_aareg_cluster_override <- aareg(
      survival::Surv(time, status) ~ x + cluster(cluster),
      data = aareg_data,
      cluster = group,
      nmin = 1
    ),
    "formula term ignored"
  )
  reference_aareg_cluster_override <- NULL
  expect_warning(
    reference_aareg_cluster_override <- survival::aareg(
      survival::Surv(time, status) ~ x + cluster(cluster),
      data = aareg_data,
      cluster = group,
      nmin = 1
    ),
    "formula term ignored"
  )
  compare_aareg(bridged_aareg_cluster_override, reference_aareg_cluster_override)

  reference_aareg_subset <- getFromNamespace("[.aareg", "survival")
  reference_aareg_summary <- getFromNamespace("summary.aareg", "survival")
  reference_aareg_print <- getFromNamespace("print.aareg", "survival")
  reference_aareg_summary_print <- getFromNamespace("print.summary.aareg", "survival")
  expect_equal(
    labels.aareg(bridged_aareg),
    attr(reference_aareg$terms, "term.labels")
  )
  compare_aareg(
    `[.aareg`(bridged_aareg, 1:2),
    reference_aareg_subset(reference_aareg, 1:2)
  )
  bridged_aareg_summary <- summary.aareg(bridged_aareg)
  reference_aareg_summary_result <- reference_aareg_summary(reference_aareg)
  expect_equal(bridged_aareg_summary, reference_aareg_summary_result, tolerance = 1e-10)
  expect_equal(
    summary.aareg(bridged_aareg, maxtime = 3, test = "nrisk", scale = 2),
    reference_aareg_summary(reference_aareg, maxtime = 3, test = "nrisk", scale = 2),
    tolerance = 1e-10
  )
  bridged_aareg_print <- bridged_aareg
  reference_aareg_print_result <- reference_aareg
  bridged_aareg_print$call <- reference_aareg_print_result$call <- quote(aareg(Surv(time, status) ~ x + z))
  expect_equal(
    capture.output(print.aareg(bridged_aareg_print)),
    capture.output(reference_aareg_print(reference_aareg_print_result))
  )
  expect_equal(
    capture.output(print.summary.aareg(bridged_aareg_summary)),
    capture.output(reference_aareg_summary_print(reference_aareg_summary_result))
  )
  grDevices::pdf(NULL)
  expect_silent(plot.aareg(bridged_aareg, se = TRUE))
  expect_silent(lines.aareg(bridged_aareg, se = FALSE, maxtime = 3))
  grDevices::dev.off()

  expect_error(
    aareg(
      survival::Surv(time, status) ~ x + cluster(group) + cluster(cluster),
      data = aareg_data,
      nmin = 1
    ),
    "multiple cluster terms"
  )
  expect_error(
    aareg(
      survival::Surv(time, status) ~ x:cluster(group),
      data = aareg_data,
      nmin = 1
    ),
    "cluster.*interaction"
  )
  tmerge_data <- data.frame(id = 1:2, tstop = c(5, 6))
  bridged_tmerge <- tmerge(tmerge_data, tmerge_data, id = id, tstop = tstop)
  reference_tmerge <- survival::tmerge(tmerge_data, tmerge_data, id = id, tstop = tstop)
  attr(bridged_tmerge, "call") <- attr(reference_tmerge, "call") <- quote(
    tmerge(tmerge_data, tmerge_data, id = id, tstop = tstop)
  )
  reference_tmerge_summary <- getFromNamespace("summary.tmerge", "survival")
  expect_equal(
    capture.output(summary.tmerge(bridged_tmerge)),
    capture.output(reference_tmerge_summary(reference_tmerge))
  )
  expect_equal(
    `[.tmerge`(bridged_tmerge, 1:2, c("id", "tstop"), drop = FALSE),
    reference_tmerge[1:2, c("id", "tstop"), drop = FALSE]
  )
  attr(bridged_tmerge, "call") <- NULL
  attr(reference_tmerge, "call") <- NULL
  expect_equal(
    bridged_tmerge,
    reference_tmerge
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

  capture_aeq_call <- function(fun, args) {
    observed_warnings <- character()
    result <- withCallingHandlers(
      tryCatch(
        list(value = do.call(fun, args)),
        error = function(error) list(error = conditionMessage(error))
      ),
      warning = function(warning) {
        observed_warnings <<- c(observed_warnings, conditionMessage(warning))
        invokeRestart("muffleWarning")
      }
    )
    list(result = result, warnings = observed_warnings)
  }
  aeq_nonfinite <- survival::Surv(
    c(1, 1 + 1e-8, 2, Inf, NA),
    c(1, 0, 1, 0, 1)
  )
  aeq_interval <- survival::Surv(
    c(1, 1 + 1e-8, NA, 2),
    c(1, 2, 3, Inf),
    type = "interval2"
  )
  aeq_boundary_cases <- list(
    list(x = c(1, 2)),
    list(x = c(1, 2), tolerance = 0),
    list(x = aeq_nonfinite),
    list(x = aeq_nonfinite, tolerance = 1e-7),
    list(x = aeq_nonfinite, tolerance = 0),
    list(x = aeq_nonfinite, tolerance = -1),
    list(x = aeq_nonfinite, tolerance = Inf),
    list(x = aeq_nonfinite, tolerance = numeric()),
    list(x = aeq_interval, tolerance = 1e-7),
    list(
      x = structure(
        numeric(),
        dim = c(0L, 2L),
        dimnames = list(NULL, c("time", "status")),
        type = "right",
        class = "Surv"
      )
    ),
    list(x = survival::Surv(c(-Inf, Inf), c(1, 0)), tolerance = 1e-7)
  )
  for (args in aeq_boundary_cases) {
    expect_identical(
      capture_aeq_call(aeqSurv, args),
      capture_aeq_call(survival::aeqSurv, args)
    )
  }

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
  expect_equal(names(counting_split), c("group", "start", "stop", "status", "episode"))
  expect_equal(counting_split$group, c("a", "a", "b", "b", "b"))
  expect_equal(as.numeric(counting_split$start), c(0, 3, 2, 3, 6))
  expect_equal(as.numeric(counting_split$stop), c(3, 5, 3, 6, 8))
  expect_equal(as.integer(counting_split$status), c(0L, 1L, 0L, 0L, 0L))
  expect_equal(as.integer(counting_split$episode), c(1L, 2L, 1L, 2L, 3L))
  reference_counting_split_formula <- Surv(start, stop, status) ~ group
  environment(reference_counting_split_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  expect_equal(
    counting_split,
    survival::survSplit(
      reference_counting_split_formula,
      data = split_counting,
      cut = c(3, 6),
      episode = "episode",
      id = "rowid"
    )
  )

  split_multistate <- data.frame(
    time = c(1, 3, 4),
    state = factor(c("a", "censor", "b"), levels = c("censor", "a", "b")),
    x = 11:13
  )
  multistate_split <- survSplit(
    Surv(time, state) ~ x,
    data = split_multistate,
    cut = c(2, 3.5),
    episode = "episode",
    id = "subject"
  )
  reference_multistate_formula <- Surv(time, state) ~ x
  environment(reference_multistate_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  reference_multistate_split <- survival::survSplit(
    reference_multistate_formula,
    data = split_multistate,
    cut = c(2, 3.5),
    episode = "episode",
    id = "subject"
  )
  expect_equal(multistate_split, reference_multistate_split)
  reference_multistate_dot_formula <- Surv(time, state) ~ .
  environment(reference_multistate_dot_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  expect_equal(
    survSplit(
      Surv(time, state) ~ .,
      data = split_multistate,
      cut = c(2, 3.5),
      episode = "episode",
      added = "added"
    ),
    survival::survSplit(
      reference_multistate_dot_formula,
      data = split_multistate,
      cut = c(2, 3.5),
      episode = "episode",
      added = "added"
    )
  )

  split_multistate_counting <- data.frame(
    start = c(0, 1),
    stop = c(3, 4),
    state = factor(c("a", "b"), levels = c("censor", "a", "b")),
    x = 1:2
  )
  multistate_counting_split <- survSplit(
    Surv(start, stop, state) ~ x,
    data = split_multistate_counting,
    cut = 2,
    episode = "episode",
    id = "subject"
  )
  reference_multistate_counting_formula <- Surv(start, stop, state) ~ x
  environment(reference_multistate_counting_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  reference_multistate_counting_split <- survival::survSplit(
    reference_multistate_counting_formula,
    data = split_multistate_counting,
    cut = 2,
    episode = "episode",
    id = "subject"
  )
  expect_equal(multistate_counting_split, reference_multistate_counting_split)
  reference_multistate_counting_dot_formula <- Surv(start, stop, state) ~ .
  environment(reference_multistate_counting_dot_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  expect_equal(
    survSplit(
      Surv(start, stop, state) ~ .,
      data = split_multistate_counting,
      cut = 2,
      episode = "episode",
      added = "added"
    ),
    survival::survSplit(
      reference_multistate_counting_dot_formula,
      data = split_multistate_counting,
      cut = 2,
      episode = "episode",
      added = "added"
    )
  )

  expect_identical(names(formals(survSplit)), names(formals(survival::survSplit)))
  near_tie_split_data <- data.frame(time = 1, status = 1, x = 9)
  reference_near_tie_formula <- Surv(time, status) ~ x
  environment(reference_near_tie_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  for (fix_times in c(TRUE, FALSE)) {
    bridged_near_tie <- survSplit(
      Surv(time, status) ~ x,
      data = near_tie_split_data,
      cut = 1 - 1e-9,
      added = "made",
      timefix = fix_times
    )
    reference_near_tie <- survival::survSplit(
      reference_near_tie_formula,
      data = near_tie_split_data,
      cut = 1 - 1e-9,
      added = "made",
      timefix = fix_times
    )
    expect_equal(bridged_near_tie, reference_near_tie, tolerance = 1e-12)
  }

  bridged_frame_split <- survSplit(
    near_tie_split_data,
    cut = 0.5,
    end = "time",
    event = "status",
    id = "rowid",
    added = "made"
  )
  reference_frame_split <- survival::survSplit(
    near_tie_split_data,
    cut = 0.5,
    end = "time",
    event = "status",
    id = "rowid",
    added = "made"
  )
  expect_equal(bridged_frame_split, reference_frame_split)
  sanitized_split <- survSplit(
    near_tie_split_data,
    cut = 0.5,
    end = "time",
    event = "status",
    episode = "split index",
    added = "inserted row"
  )
  expect_true(all(c("split.index", "inserted.row") %in% names(sanitized_split)))
  expect_error(
    survSplit(
      near_tie_split_data,
      cut = 0.5,
      end = "time",
      event = "status",
      episode = 1
    ),
    "episode must be a character string"
  )
  expect_error(
    survSplit(
      near_tie_split_data,
      cut = 0.5,
      end = "time",
      event = "status",
      added = 1
    ),
    "added must be a character string"
  )

  delayed_split_data <- data.frame(start = 5, stop = 10, status = 1)
  reference_delayed_split_formula <- Surv(start, stop, status) ~ 1
  environment(reference_delayed_split_formula) <- list2env(
    list(Surv = survival::Surv),
    parent = parent.frame()
  )
  bridged_delayed_split <- survSplit(
    Surv(start, stop, status) ~ 1,
    data = delayed_split_data,
    cut = c(3, 7),
    episode = "episode"
  )
  reference_delayed_split <- survival::survSplit(
    reference_delayed_split_formula,
    data = delayed_split_data,
    cut = c(3, 7),
    episode = "episode"
  )
  expect_equal(bridged_delayed_split, reference_delayed_split)

  timeline_split_data <- data.frame(
    id = c(1, 1, 1, 2, 2),
    time = c(0, 2, 5, 0, 4),
    event = factor(c("a", "b", "b", "a", "b"), levels = c("a", "b")),
    x = 1:5
  )
  reference_timeline_split_formula <- Surv2(time, event) ~ x
  environment(reference_timeline_split_formula) <- list2env(
    list(Surv2 = survival::Surv2),
    parent = parent.frame()
  )
  bridged_timeline_split <- survSplit(
    Surv2(time, event) ~ x,
    data = timeline_split_data,
    id = id,
    cut = c(1, 3),
    episode = "ep",
    added = "add"
  )
  reference_timeline_split <- survival::survSplit(
    reference_timeline_split_formula,
    data = timeline_split_data,
    id = id,
    cut = c(1, 3),
    episode = "ep",
    added = "add"
  )
  expect_equal(bridged_timeline_split, reference_timeline_split)

  expect_named_survsplit_equal <- function(y, data) {
    if (inherits(y, "Surv2")) {
      bridged <- survSplit(
        y ~ x,
        data = data,
        id = data$id,
        cut = c(1.5, 3.5),
        episode = "ep",
        added = "add"
      )
      reference <- survival::survSplit(
        y ~ x,
        data = data,
        id = data$id,
        cut = c(1.5, 3.5),
        episode = "ep",
        added = "add"
      )
    } else {
      bridged <- survSplit(
        y ~ x,
        data = data,
        cut = c(1.5, 3.5),
        episode = "ep",
        added = "add"
      )
      reference <- survival::survSplit(
        y ~ x,
        data = data,
        cut = c(1.5, 3.5),
        episode = "ep",
        added = "add"
      )
    }
    expect_equal(bridged, reference)
  }

  named_right_data <- data.frame(time = c(1, 3, 5), status = c(1, 0, 1), x = 1:3)
  expect_named_survsplit_equal(
    survival::Surv(named_right_data$time, named_right_data$status),
    named_right_data
  )
  named_mright_data <- data.frame(
    time = c(1, 3, 5),
    state = factor(c("a", "censor", "b"), levels = c("censor", "a", "b")),
    x = 1:3
  )
  expect_named_survsplit_equal(
    survival::Surv(named_mright_data$time, named_mright_data$state),
    named_mright_data
  )
  named_counting_data <- data.frame(
    start = c(0, 2, 3),
    stop = c(3, 5, 6),
    status = c(1, 0, 1),
    x = 1:3
  )
  expect_named_survsplit_equal(
    survival::Surv(
      named_counting_data$start,
      named_counting_data$stop,
      named_counting_data$status
    ),
    named_counting_data
  )
  named_mcounting_data <- data.frame(
    start = c(0, 2, 3),
    stop = c(3, 5, 6),
    state = factor(c("a", "censor", "b"), levels = c("censor", "a", "b")),
    x = 1:3
  )
  expect_named_survsplit_equal(
    survival::Surv(
      named_mcounting_data$start,
      named_mcounting_data$stop,
      named_mcounting_data$state
    ),
    named_mcounting_data
  )
  expect_named_survsplit_equal(
    survival::Surv2(timeline_split_data$time, timeline_split_data$event),
    timeline_split_data
  )

  named_na_data <- data.frame(time = c(1, 3, 5), status = c(1, 0, 1), x = c(1, NA, 3))
  named_na_response <- survival::Surv(named_na_data$time, named_na_data$status)
  expect_equal(
    survSplit(
      named_na_response ~ x,
      data = named_na_data,
      na.action = stats::na.omit,
      cut = c(1.5, 3.5)
    ),
    survival::survSplit(
      named_na_response ~ x,
      data = named_na_data,
      na.action = stats::na.omit,
      cut = c(1.5, 3.5)
    )
  )

  named_dot_data <- data.frame(x = 1:3, check.names = FALSE)
  named_dot_data[["..survsplit.source"]] <- 11:13
  named_dot_data[["..survsplit.start"]] <- 21:23
  named_dot_data[["..survsplit.end"]] <- 31:33
  named_dot_data[["..survsplit.event"]] <- 41:43
  named_dot_data$y <- survival::Surv(c(1, 3, 5), c(1, 0, 1))
  expect_equal(
    survSplit(
      y ~ .,
      data = named_dot_data,
      cut = c(1.5, 3.5),
      episode = "..survsplit.event",
      added = "..survsplit.start"
    ),
    survival::survSplit(
      y ~ .,
      data = named_dot_data,
      cut = c(1.5, 3.5),
      episode = "..survsplit.event",
      added = "..survsplit.start"
    )
  )

  check_data <- data.frame(
    id = c(1, 1, 2),
    start = c(0, 1, 0),
    stop = c(1, 2, 2),
    status = c(0, 1, 1)
  )
  compare_survcheck <- function(actual, expected) {
    fields <- c(
      "states", "transitions", "events", "flag", "istate",
      "overlap", "gap", "jump", "teleport", "n"
    )
    for (field in fields) {
      expect_equal(actual[[field]], expected[[field]], info = field)
    }
    expect_identical(actual$Y, expected$Y)
    expect_identical(actual$id, expected$id)
    expect_s3_class(actual, "survcheck")
  }
  checked <- survcheck(Surv(start, stop, status) ~ 1, data = check_data, id = id)
  reference_checked <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = check_data,
    id = id
  )
  compare_survcheck(checked, reference_checked)
  expect_equal(unname(checked$n), c(2L, 3L, 2L))
  expect_identical(checked$flag, reference_checked$flag)
  transition_origin_data <- data.frame(
    id = c("b", "a", "b", "b"),
    start = c(1, 1, 5, 5),
    stop = c(5, 5, 9, 9),
    status = c(0L, 0L, 1L, 0L)
  )
  transition_origin_checked <- survcheck(
    Surv(start, stop, status) ~ 1,
    data = transition_origin_data,
    id = id
  )
  reference_transition_origin <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = transition_origin_data,
    id = id
  )
  expect_identical(
    transition_origin_checked$transitions,
    reference_transition_origin$transitions
  )
  near_interval_data <- data.frame(
    id = rep(c("gap", "overlap"), each = 2L),
    start = c(0, 1 + 1e-15, 0, 1 - 1e-15),
    stop = c(1, 2, 1, 2),
    status = 0L
  )
  near_interval_checked <- survcheck(
    Surv(start, stop, status) ~ 1,
    data = near_interval_data,
    id = id,
    timefix = FALSE
  )
  reference_near_interval <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = near_interval_data,
    id = id,
    timefix = FALSE
  )
  compare_survcheck(near_interval_checked, reference_near_interval)
  factor_id_data <- transform(
    transition_origin_data,
    id = factor(id, levels = c("a", "b", "unused"))
  )
  factor_id_checked <- survcheck(
    Surv(start, stop, status) ~ 1,
    data = factor_id_data,
    id = id
  )
  reference_factor_id <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = factor_id_data,
    id = id
  )
  expect_identical(factor_id_checked$events, reference_factor_id$events)
  checked_print <- checked
  reference_checked_print <- reference_checked
  attr(checked_print, "call") <- attr(reference_checked_print, "call") <- NULL
  checked_print$call <- reference_checked_print$call <- quote(
    survcheck(Surv(start, stop, status) ~ 1, data = check_data, id = id)
  )
  expect_equal(
    capture.output(print.survcheck(checked_print)),
    capture.output(getFromNamespace("print.survcheck", "survival")(reference_checked_print))
  )
  subset_check_data <- transform(check_data, keep = c(TRUE, TRUE, FALSE))
  subset_checked <- survcheck(Surv(start, stop, status) ~ 1, data = subset_check_data, id = id, subset = keep)
  reference_subset_checked <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = subset_check_data,
    id = id,
    subset = keep
  )
  compare_survcheck(subset_checked, reference_subset_checked)

  overlap_data <- data.frame(
    id = c("a", "a"),
    start = c(0, 0.5),
    stop = c(1, 2),
    status = c(0, 1)
  )
  overlap_check <- survcheck(Surv(start, stop, status) ~ 1, data = overlap_data, id = id)
  reference_overlap_check <- survival::survcheck(
    survival::Surv(start, stop, status) ~ 1,
    data = overlap_data,
    id = id
  )
  compare_survcheck(overlap_check, reference_overlap_check)
  expect_equal(overlap_check$overlap, list(row = 2L, id = "a"))

  multistate_check_data <- data.frame(
    id = c("a", "a", "b", "b"),
    start = c(0, 1, 0, 2),
    stop = c(1, 2, 1, 3),
    state = factor(c("B", "C", "B", "C"), levels = c("censor", "B", "C")),
    initial = factor(c("A", "B", "A", "A"), levels = c("A", "B", "C"))
  )
  multistate_checked <- survcheck(
    Surv(start, stop, state) ~ 1,
    data = multistate_check_data,
    id = id,
    istate = initial
  )
  reference_multistate_checked <- survival::survcheck(
    survival::Surv(start, stop, state) ~ 1,
    data = multistate_check_data,
    id = id,
    istate = initial
  )
  compare_survcheck(multistate_checked, reference_multistate_checked)
  expect_equal(multistate_checked$jump, list(row = 4L, id = "b"))

  teleport_check_data <- transform(
    multistate_check_data[1:2, ],
    initial = factor(c("A", "A"), levels = c("A", "B", "C"))
  )
  teleport_checked <- survcheck(
    Surv(start, stop, state) ~ 1,
    data = teleport_check_data,
    id = id,
    istate = initial
  )
  reference_teleport_checked <- survival::survcheck(
    survival::Surv(start, stop, state) ~ 1,
    data = teleport_check_data,
    id = id,
    istate = initial
  )
  compare_survcheck(teleport_checked, reference_teleport_checked)
  expect_equal(teleport_checked$teleport, list(row = 2L, id = "a"))

  right_multistate_check_data <- data.frame(
    id = c("a", "a"),
    time = c(1, 2),
    state = factor(c("B", "C"), levels = c("censor", "B", "C"))
  )
  right_multistate_checked <- survcheck(
    Surv(time, state) ~ 1,
    data = right_multistate_check_data,
    id = id
  )
  reference_right_multistate_checked <- survival::survcheck(
    survival::Surv(time, state) ~ 1,
    data = right_multistate_check_data,
    id = id
  )
  compare_survcheck(right_multistate_checked, reference_right_multistate_checked)

  missing_check_data <- transform(check_data, x = c(1, NA, 2))
  missing_checked <- survcheck(
    Surv(start, stop, status) ~ x,
    data = missing_check_data,
    id = id,
    na.action = na.omit
  )
  reference_missing_checked <- survival::survcheck(
    survival::Surv(start, stop, status) ~ x,
    data = missing_check_data,
    id = id,
    na.action = na.omit
  )
  compare_survcheck(missing_checked, reference_missing_checked)
  expect_equal(missing_checked$na.action, reference_missing_checked$na.action)
  singleton_check_data <- data.frame(
    id = factor(c("b", "a")),
    start = c(1, 2),
    stop = c(5, 3),
    status = c(0L, 0L),
    x = c(NA, -1)
  )
  singleton_checked <- survcheck(
    Surv(start, stop, status) ~ x,
    data = singleton_check_data,
    id = id,
    na.action = na.omit
  )
  reference_singleton_checked <- survival::survcheck(
    survival::Surv(start, stop, status) ~ x,
    data = singleton_check_data,
    id = id,
    na.action = na.omit
  )
  compare_survcheck(singleton_checked, reference_singleton_checked)
  right_singleton_data <- data.frame(
    id = 1:2,
    time = 4:3,
    state = factor(c("censor", "censor"), levels = c("censor", "B", "C")),
    x = c(-1, NA)
  )
  right_singleton_checked <- survcheck(
    Surv(time, state) ~ x,
    data = right_singleton_data,
    id = id,
    na.action = na.omit
  )
  reference_right_singleton <- survival::survcheck(
    survival::Surv(time, state) ~ x,
    data = right_singleton_data,
    id = id,
    na.action = na.omit
  )
  compare_survcheck(right_singleton_checked, reference_right_singleton)

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
    c(`1` = 0.5, `2` = 0, `3` = 0.5)
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
  zero_weight_rtt <- data.frame(
    time = c(1, 2),
    status = c(0, 1),
    wt = c(1, 0),
    row.names = c("censored", "zero-event")
  )
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = zero_weight_rtt, weights = wt),
    c(0, NaN)
  )
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = zero_weight_rtt, weights = wt, times = 2),
    c(censored = 0, `zero-event` = NaN)
  )
  expect_equal(
    rttright(
      Surv(time, status) ~ 1,
      data = zero_weight_rtt,
      weights = wt,
      times = numeric()
    ),
    matrix(
      numeric(),
      nrow = 2L,
      ncol = 0L,
      dimnames = list(c("censored", "zero-event"), NULL)
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
  expect_equal(
    rttright(Surv(time, status) ~ group, data = grouped_rtt, times = 3),
    survival::rttright(survival::Surv(time, status) ~ group, data = grouped_rtt, times = 3)
  )
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

  multistate_rtt <- data.frame(
    time = c(1, 2, 3, 4, 5, 6),
    state = factor(
      c("a", "censor", "b", "a", "censor", "b"),
      levels = c("censor", "a", "b")
    ),
    group = rep(c("x", "y"), each = 3),
    wt = c(2, 1, 3, 1, 4, 2),
    id = letters[1:6]
  )
  expect_equal(
    rttright(Surv(time, state) ~ 1, data = multistate_rtt, id = id),
    survival::rttright(survival::Surv(time, state) ~ 1, data = multistate_rtt, id = id)
  )
  expect_equal(
    rttright(Surv(time, state) ~ group, data = multistate_rtt, weights = wt),
    survival::rttright(
      survival::Surv(time, state) ~ group,
      data = multistate_rtt,
      weights = wt
    )
  )
  expect_equal(
    rttright(Surv(time, state) ~ group, data = multistate_rtt, times = c(2, 4, 6)),
    survival::rttright(
      survival::Surv(time, state) ~ group,
      data = multistate_rtt,
      times = c(2, 4, 6)
    )
  )

  missing_rtt <- data.frame(
    time = c(1, NA, 3),
    status = c(0, 1, 1),
    row.names = c("early", "missing", "event")
  )
  expect_equal(
    rttright(Surv(time, status) ~ 1, data = missing_rtt, times = 2),
    survival::rttright(survival::Surv(time, status) ~ 1, data = missing_rtt, times = 2)
  )

  singleton_rtt <- data.frame(
    time = c(6, 3),
    status = c(1, 0),
    group = factor(c("south", "north"), levels = c("south", "west", "north")),
    site = c("B", "A"),
    keep = c(FALSE, TRUE)
  )
  expect_warning(
    singleton_empty_rtt <- rttright(
      Surv(time, status) ~ group + site,
      data = singleton_rtt,
      subset = keep,
      times = numeric()
    ),
    "no non-missing arguments"
  )
  expect_warning(
    reference_singleton_empty_rtt <- survival::rttright(
      survival::Surv(time, status) ~ group + site,
      data = singleton_rtt,
      subset = keep,
      times = numeric()
    ),
    "no non-missing arguments"
  )
  expect_equal(singleton_empty_rtt, reference_singleton_empty_rtt)

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

  no_event_counting_rtt <- data.frame(
    id = c("a", "a", "b"),
    start = c(0, 1, 0),
    stop = c(1, 2, 2),
    status = c(0, 0, 0)
  )
  expect_equal(
    rttright(
      Surv(start, stop, status) ~ 1,
      data = no_event_counting_rtt,
      id = id,
      times = 0
    ),
    survival::rttright(
      survival::Surv(start, stop, status) ~ 1,
      data = no_event_counting_rtt,
      id = id,
      times = 0
    )
  )

  zero_sum_counting_rtt <- data.frame(
    id = c("a", "b", "c"),
    start = c(0, 0, 0),
    stop = c(1, 2, 3),
    status = c(1, 1, 1),
    wt = c(0, 1, 1),
    group = c("zero", "positive", "positive")
  )
  expect_equal(
    rttright(
      Surv(start, stop, status) ~ group,
      data = zero_sum_counting_rtt,
      id = id,
      weights = wt,
      times = c(0, 1)
    ),
    survival::rttright(
      survival::Surv(start, stop, status) ~ group,
      data = zero_sum_counting_rtt,
      id = id,
      weights = wt,
      times = c(0, 1)
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
  graphics::frame()
  blank_statefig_plot <- grDevices::recordPlot()
  reference_statefig <- survival::statefig(c(1, 1), state_connect)
  reference_statefig_coords <- survival::statefig(
    matrix(c(0.2, 0.7, 0.8, 0.3), nrow = 2, byrow = TRUE),
    state_connect,
    box = FALSE
  )
  reference_statefig_column <- survival::statefig(matrix(c(1, 1), ncol = 1), state_connect)
  singleton_connect <- matrix(0, nrow = 1, dimnames = list("only", "only"))
  reference_statefig_singleton <- survival::statefig(1, singleton_connect)
  reference_statefig_usr <- graphics::par("usr")
  bridged_statefig <- statefig(c(1, 1), state_connect)
  bridged_statefig_coords <- statefig(
    matrix(c(0.2, 0.7, 0.8, 0.3), nrow = 2, byrow = TRUE),
    state_connect,
    box = FALSE
  )
  bridged_statefig_column <- statefig(matrix(c(1, 1), ncol = 1), state_connect)
  bridged_statefig_singleton <- statefig(1, singleton_connect)
  bridged_statefig_usr <- graphics::par("usr")
  bridged_statefig_plot <- grDevices::recordPlot()
  grDevices::dev.off()
  expect_equal(bridged_statefig, reference_statefig)
  expect_equal(bridged_statefig_usr, reference_statefig_usr)
  expect_false(identical(blank_statefig_plot[[2L]], bridged_statefig_plot[[2L]]))
  expect_equal(
    bridged_statefig_coords,
    reference_statefig_coords
  )
  expect_equal(bridged_statefig_column, reference_statefig_column)
  expect_equal(bridged_statefig_singleton, reference_statefig_singleton)
  grDevices::pdf(NULL)
  expect_error(statefig("bad", state_connect), "layout")
  expect_error(statefig(c(1, 1), matrix(0, nrow = 1, ncol = 2)), "square")
  grDevices::dev.off()
  reference_statefig_function <- get("statefig", envir = asNamespace("survival"))
  capture_statefig <- function(fun, args) {
    captured_warnings <- character()
    grDevices::pdf(NULL)
    on.exit(grDevices::dev.off())
    result <- tryCatch(
      withCallingHandlers(
        list(
          kind = "value",
          value = do.call(fun, args),
          usr = graphics::par("usr")
        ),
        warning = function(w) {
          captured_warnings <<- c(captured_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        list(
          kind = "error",
          message = conditionMessage(e),
          class = class(e),
          usr = graphics::par("usr")
        )
      }
    )
    c(result, list(warnings = captured_warnings))
  }
  state_connect_three <- matrix(
    c(0, 1, 0, 0, 0, 1, 1, 0, 0),
    nrow = 3L,
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  state_connect_column_names <- state_connect
  dimnames(state_connect_column_names) <- list(NULL, c("a", "b"))
  state_connect_missing <- state_connect
  state_connect_missing[1L, 2L] <- NA_real_
  statefig_cases <- list(
    list(layout = c(2, 1), connect = state_connect_three),
    list(layout = matrix(c(1, 1), nrow = 1L), connect = state_connect),
    list(layout = c(1, 1), connect = state_connect_column_names),
    list(layout = c(0, 2), connect = state_connect),
    list(layout = c(-1, 3), connect = state_connect),
    list(layout = c(0.5, 1.5), connect = state_connect),
    list(layout = c(NA, 2), connect = state_connect),
    list(layout = c(Inf, 1), connect = state_connect),
    list(
      layout = matrix(c(-0.1, 0.2, 0.8, 0.9), nrow = 2L, byrow = TRUE),
      connect = state_connect
    ),
    list(
      layout = matrix(c(0.1, 0.2, 1.1, 0.9), nrow = 2L, byrow = TRUE),
      connect = state_connect
    ),
    list(
      layout = matrix(c(0.1, 0.2, 0.5, 0.5, 0.8, 0.9), nrow = 3L, byrow = TRUE),
      connect = state_connect
    ),
    list(layout = c(1, 1), connect = state_connect_missing)
  )
  for (case in statefig_cases) {
    expect_identical(
      capture_statefig(statefig, case),
      capture_statefig(reference_statefig_function, case)
    )
  }

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
  expect_pspline_equal(
    pspline(rep(2, 3), df = 2, penalty = FALSE),
    survival::pspline(rep(2, 3), df = 2, penalty = FALSE)
  )
  expect_pspline_equal(
    pspline(c(1, NA, 2), df = 2, penalty = FALSE),
    survival::pspline(c(1, NA, 2), df = 2, penalty = FALSE)
  )

  bridged_nsk_basis <- nsk(1:10, df = 4)
  bridged_nsk_call_method <- get(
    "makepredictcall.nsk",
    envir = asNamespace("survivalr")
  )
  reference_nsk_call_method <- get(
    "makepredictcall.nsk",
    envir = asNamespace("survival")
  )
  original_nsk_call <- quote(nsk(value, df = 4, b = 0.2))
  expect_equal(
    bridged_nsk_call_method(bridged_nsk_basis, original_nsk_call),
    reference_nsk_call_method(bridged_nsk_basis, original_nsk_call)
  )
  namespaced_nsk_call <- bridged_nsk_call_method(
    bridged_nsk_basis,
    quote(survivalr::nsk(value, df = 4))
  )
  expect_identical(namespaced_nsk_call[[1L]], quote(survivalr::nsk))
  expect_equal(namespaced_nsk_call$knots, attr(bridged_nsk_basis, "knots"))
  expect_equal(
    namespaced_nsk_call$Boundary.knots,
    attr(bridged_nsk_basis, "Boundary.knots")
  )
  unrelated_nsk_call <- quote(splines::ns(value, df = 4))
  expect_identical(
    bridged_nsk_call_method(bridged_nsk_basis, unrelated_nsk_call),
    unrelated_nsk_call
  )

  bridged_pspline_basis <- pspline(1:10, df = 4)
  reference_pspline_basis <- survival::pspline(1:10, df = 4)
  bridged_pspline_call_method <- get(
    "makepredictcall.pspline",
    envir = asNamespace("survivalr")
  )
  reference_pspline_call_method <- get(
    "makepredictcall.pspline",
    envir = asNamespace("survival")
  )
  original_pspline_call <- quote(pspline(value, df = 4, nterm = 10))
  expect_equal(
    bridged_pspline_call_method(bridged_pspline_basis, original_pspline_call),
    reference_pspline_call_method(bridged_pspline_basis, original_pspline_call)
  )
  expect_equal(
    bridged_pspline_call_method(bridged_pspline_basis, original_pspline_call)$df,
    attr(bridged_pspline_basis, "df")
  )
  unrelated_pspline_call <- quote(stats::poly(value, degree = 3))
  expect_identical(
    bridged_pspline_call_method(bridged_pspline_basis, unrelated_pspline_call),
    unrelated_pspline_call
  )

  bridged_pspline_predict_method <- get(
    "predict.pspline",
    envir = asNamespace("survivalr")
  )
  reference_pspline_predict_method <- get(
    "predict.pspline",
    envir = asNamespace("survival")
  )
  expect_identical(
    bridged_pspline_predict_method(bridged_pspline_basis),
    bridged_pspline_basis
  )
  prediction_values <- c(0, 2.5, 5, 10, 12)
  expect_pspline_equal(
    bridged_pspline_predict_method(bridged_pspline_basis, prediction_values),
    reference_pspline_predict_method(reference_pspline_basis, prediction_values)
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
  assign_fit_matrix <- stats::model.matrix(assign_fit)
  assign_fit_terms <- stats::terms(assign_fit)
  expect_equal(
    attrassign(assign_fit_matrix, assign_fit_terms),
    survival::attrassign(assign_fit_matrix, assign_fit_terms)
  )

  cox_control <- coxph.control(iter.max = 0, eps = 1e-05, toler.chol = 1e-08, timefix = FALSE)
  expect_named(
    cox_control,
    c("eps", "toler.chol", "iter.max", "toler.inf", "outer.max", "timefix", "survcheckallow")
  )
  expect_equal(cox_control[["iter.max"]], 0L)
  expect_false(cox_control[["timefix"]])
  expect_equal(cox_control[["survcheckallow"]], "gap")
  expect_error(coxph.control(iter.max = -1), "Invalid value for iterations")
  capture_control <- function(fun, args) {
    warning_messages <- character()
    value <- tryCatch(
      withCallingHandlers(
        do.call(fun, args),
        warning = function(condition) {
          warning_messages <<- c(warning_messages, conditionMessage(condition))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(condition) structure(conditionMessage(condition), class = "captured_error")
    )
    list(value = value, warnings = warning_messages)
  }
  cox_control_cases <- list(
    list(),
    list(iter.max = 3.9, outer.max = 4.9),
    list(eps = Inf),
    list(toler.chol = Inf),
    list(toler.inf = Inf),
    list(outer.max = Inf),
    list(timefix = c(TRUE, FALSE)),
    list(timefix = NA),
    list(eps = c(1e-09, 1e-08)),
    list(iter.max = c(2, 3)),
    list(iter.max = TRUE),
    list(eps = TRUE),
    list(eps = 1e-09, toler.chol = 1e-08),
    list(survcheckallow = c("gap", "overlap"))
  )
  for (case in cox_control_cases) {
    expect_identical(
      capture_control(coxph.control, case),
      capture_control(survival::coxph.control, case)
    )
  }
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
  expect_equal(
    coxph.wtest(matrix(0, 2, 2), c(1, 2)),
    survival::coxph.wtest(matrix(0, 2, 2), c(1, 2))
  )
  expect_equal(
    coxph.wtest(matrix(c(1, 2, 2, 1), 2), c(1, 2)),
    survival::coxph.wtest(matrix(c(1, 2, 2, 1), 2), c(1, 2))
  )
  asymmetric_wtest <- matrix(c(2, 0.25, 7, 1), 2, byrow = TRUE)
  expect_equal(
    coxph.wtest(asymmetric_wtest, c(1, 2)),
    survival::coxph.wtest(asymmetric_wtest, c(1, 2))
  )
  expect_equal(coxph.wtest(diag(2), c(NA, 2)), survival::coxph.wtest(diag(2), c(NA, 2)))
  reference_coxph_wtest <- get("coxph.wtest", envir = asNamespace("survival"))
  capture_coxph_wtest <- function(fun, args) {
    captured_warnings <- character()
    result <- tryCatch(
      withCallingHandlers(
        list(kind = "value", value = do.call(fun, args)),
        warning = function(w) {
          captured_warnings <<- c(captured_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        list(kind = "error", message = conditionMessage(e), class = class(e))
      }
    )
    c(result, list(warnings = captured_warnings))
  }
  coxph_wtest_cases <- list(
    list(var = 2, b = 4),
    list(var = c(v = 2), b = c(beta = 4)),
    list(var = 0, b = 4),
    list(var = 0, b = 0),
    list(var = numeric(), b = numeric()),
    list(var = numeric(), b = 1),
    list(var = numeric(), b = matrix(numeric(), nrow = 0L, ncol = 2L)),
    list(var = c(1, 0, 0, 1), b = c(1, 2)),
    list(var = matrix(1:6, nrow = 2L), b = c(1, 2)),
    list(var = diag(2), b = 1),
    list(var = diag(2), b = matrix(numeric(), nrow = 2L, ncol = 0L)),
    list(var = matrix(numeric(), nrow = 0L, ncol = 0L), b = matrix(numeric(), nrow = 0L, ncol = 2L)),
    list(var = diag(2), b = c(1, NA)),
    list(var = diag(2), b = c(NaN, 2)),
    list(var = diag(2), b = c(Inf, 2)),
    list(var = matrix(c(1, 0, 0, NA), nrow = 2L), b = c(1, 2)),
    list(var = diag(2), b = matrix(c(1, NA, 3, 4), nrow = 2L)),
    list(var = diag(2), b = c(1, 2), toler.chol = -1),
    list(
      var = matrix(c(6, 0, 2, 0, 0, 0, 2, 0, 3), nrow = 3L),
      b = matrix(c(-1, -1, 0, -4, -2, 2, -2, -1, 1), nrow = 3L),
      toler.chol = -1
    ),
    list(
      var = matrix(c(9, -3, 3, 9, -3, 1, -1, -3, 3, -1, 1, 3, 9, -3, 3, 9), nrow = 4L),
      b = matrix(c(3, 1, -2, 4, 3, -6, 3, 2), nrow = 4L),
      toler.chol = -1e-9
    ),
    list(var = diag(2), b = c(1, 2), toler.chol = NA_real_),
    list(var = diag(2), b = c(1, 2), toler.chol = c(1e-9, 1e-8)),
    list(var = diag(2), b = c(1, 2), toler.chol = numeric()),
    list(var = diag(2), b = c(1, 2), toler.chol = c(1e-9, Inf)),
    list(var = "2", b = 4),
    list(var = 2, b = factor("4")),
    list(var = 2, b = 4 + 1i),
    list(var = diag(2), b = c(1 + 1i, 2 - 1i)),
    list(var = matrix(c(1 + 1i, 0, 0, 1), nrow = 2L), b = c(1, 2)),
    list(var = 2, b = matrix(c(1, 2), nrow = 1L))
  )
  for (case in coxph_wtest_cases) {
    expect_identical(
      capture_coxph_wtest(coxph.wtest, case),
      capture_coxph_wtest(reference_coxph_wtest, case)
    )
  }

  survreg_control <- survreg.control(maxiter = 1, rel.tolerance = 1e-05, toler.chol = 1e-08)
  expect_named(survreg_control, c("iter.max", "rel.tolerance", "toler.chol", "debug", "maxiter", "outer.max"))
  expect_equal(survreg_control[["iter.max"]], 1L)
  expect_equal(survreg_control[["maxiter"]], 1L)
  survreg_control_cases <- list(
    list(),
    list(maxiter = 3.9),
    list(maxiter = -1),
    list(rel.tolerance = 0),
    list(toler.chol = Inf),
    list(debug = c(1, 2)),
    list(outer.max = NA),
    list(maxiter = 5, iter.max = NULL),
    list(maxiter = 5, iter.max = c(2, 3)),
    list(maxiter = 5, iter.max = Inf),
    list(iter.max = TRUE)
  )
  for (case in survreg_control_cases) {
    expect_identical(
      capture_control(survreg.control, case),
      capture_control(survival::survreg.control, case)
    )
  }

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

  distributions <- c(
    "extreme", "logistic", "gaussian", "weibull", "exponential",
    "rayleigh", "loggaussian", "lognormal", "loglogistic"
  )
  values <- setNames(c(0.5, 1.2, 2), c("first", "second", "third"))
  means <- c(0.1, -0.2, 0.3)
  scales <- c(0.7, 1.1, 1.4)
  probabilities <- setNames(c(0.1, 0.5, 0.9), names(values))
  for (distribution in distributions) {
    expect_equal(
      dsurvreg(values, means, scales, distribution),
      survival::dsurvreg(values, means, scales, distribution),
      tolerance = 1e-12,
      info = paste(distribution, "density")
    )
    expect_equal(
      psurvreg(values, means, scales, distribution),
      survival::psurvreg(values, means, scales, distribution),
      tolerance = 1e-12,
      info = paste(distribution, "distribution")
    )
    expect_equal(
      qsurvreg(probabilities, means, scales, distribution),
      survival::qsurvreg(probabilities, means, scales, distribution),
      tolerance = 1e-12,
      info = paste(distribution, "quantile")
    )
  }

  expect_warning(
    recycled_density <- dsurvreg(values, mean = c(0, 1), scale = 1),
    "longer object length"
  )
  expect_equal(
    recycled_density,
    suppressWarnings(survival::dsurvreg(values, mean = c(0, 1), scale = 1)),
    tolerance = 1e-12
  )
  expect_warning(
    recycled_quantiles <- qsurvreg(
      probabilities,
      mean = c(0, 1),
      scale = c(1, 2),
      distribution = "gaussian"
    ),
    "longer object length"
  )
  expect_equal(
    recycled_quantiles,
    suppressWarnings(
      survival::qsurvreg(
        probabilities,
        mean = c(0, 1),
        scale = c(1, 2),
        distribution = "gaussian"
      )
    ),
    tolerance = 1e-12
  )

  set.seed(20260822)
  reference_draws <- suppressWarnings(
    survival::rsurvreg(5, mean = c(0, 1), scale = c(1, 2), distribution = "gaussian")
  )
  reference_seed <- .Random.seed
  set.seed(20260822)
  expect_warning(
    bridged_draws <- rsurvreg(
      5,
      mean = c(0, 1),
      scale = c(1, 2),
      distribution = "gaussian"
    ),
    "longer object length"
  )
  expect_equal(bridged_draws, reference_draws, tolerance = 1e-12)
  expect_identical(.Random.seed, reference_seed)

  set.seed(42)
  reference_t_draws <- survival::rsurvreg(
    4,
    mean = 0.5,
    scale = 1.2,
    distribution = "t",
    parms = 5
  )
  set.seed(42)
  expect_equal(
    rsurvreg(4, mean = 0.5, scale = 1.2, distribution = "t", parms = 5),
    reference_t_draws,
    tolerance = 1e-12
  )
  expect_equal(
    dsurvreg(numeric(), mean = 0, distribution = "gaussian"),
    survival::dsurvreg(numeric(), mean = 0, distribution = "gaussian")
  )
  expect_equal(
    dsurvreg(1, mean = numeric(), distribution = "gaussian"),
    survival::dsurvreg(1, mean = numeric(), distribution = "gaussian")
  )
  expect_equal(
    dsurvreg(c(-1, 0, 1), mean = 0, scale = -1, distribution = "gaussian"),
    survival::dsurvreg(c(-1, 0, 1), mean = 0, scale = -1, distribution = "gaussian")
  )
  expect_equal(
    dsurvreg(c(-1, 0, 1), mean = 0, scale = 0, distribution = "gaussian"),
    survival::dsurvreg(c(-1, 0, 1), mean = 0, scale = 0, distribution = "gaussian")
  )
  expect_warning(
    nonpositive_cdf <- psurvreg(c(-1, 0, 1), mean = 0, distribution = "weibull"),
    "NaNs produced"
  )
  expect_equal(
    nonpositive_cdf,
    suppressWarnings(survival::psurvreg(c(-1, 0, 1), mean = 0, distribution = "weibull"))
  )
  expect_warning(
    boundary_quantiles <- qsurvreg(
      c(-0.1, 0, 0.5, 1, 1.1, NA_real_),
      mean = 0,
      distribution = "gaussian"
    ),
    "NaNs produced"
  )
  expect_equal(
    boundary_quantiles,
    suppressWarnings(
      survival::qsurvreg(
        c(-0.1, 0, 0.5, 1, 1.1, NA_real_),
        mean = 0,
        distribution = "gaussian"
      )
    )
  )
  expect_error(dsurvreg(1, 0, distribution = "missing"), "Distribution not found")

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
  reference_km_id$model <- stats::model.frame.default(
    survival::Surv(time, status) ~ 1,
    data = data,
    id = seq_along(time)
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
  reference_grouped_model_fit$model <- stats::model.frame.default(
    survival::Surv(time, status) ~ group,
    data = data
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
  reference_direct_model_fit$model <- stats::model.frame.default(
    survival::Surv(time, status) ~ 1,
    data = data
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
  reference_survfit_confint <- get("survfit_confint", envir = asNamespace("survival"))
  capture_survfit_confint <- function(fun, args) {
    captured_warnings <- character()
    result <- tryCatch(
      withCallingHandlers(
        list(kind = "value", value = do.call(fun, args)),
        warning = function(w) {
          captured_warnings <<- c(captured_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) {
        list(kind = "error", message = conditionMessage(e), class = class(e))
      }
    )
    c(result, list(warnings = captured_warnings))
  }
  survfit_confint_cases <- c(
    lapply(
      c("plain", "log", "log-log", "logit", "arcsin"),
      function(conf_type) {
        list(p = c(0.2, 0.5, 0.9), se = 0.1, conf.type = conf_type)
      }
    ),
    list(
      list(p = c(a = 0.2, b = 0.5), se = c(0.1, 0.2, 0.3), conf.type = "plain"),
      list(p = matrix(c(0.2, 0.5), nrow = 1L), se = 0.1, conf.type = "plain"),
      list(p = factor(c("0.2", "0.5")), se = 0.1, conf.type = "plain"),
      list(p = c(NA, NaN, Inf, -Inf, -1, 0, 1, 2), se = c(0, 0.1), conf.type = "logit"),
      list(p = numeric(), se = 0.1, conf.type = "log"),
      list(p = 0.5, se = numeric(), conf.type = "log-log"),
      list(p = c(0.2, 0.5), se = c(0, 0.1), conf.type = "arcsin", selow = numeric()),
      list(p = 0.5, se = 0.1, logse = FALSE, conf.type = "plain", selow = 0.05, ulimit = FALSE),
      list(p = c(0.2, 0.5), se = 0.1, logse = FALSE, conf.type = "plain"),
      list(p = 0.5, se = 0.1, logse = NA, conf.type = "plain"),
      list(p = 0.5, se = 0.1, logse = 1, conf.type = "plain"),
      list(p = 0.5, se = 0.1, logse = c(TRUE, FALSE), conf.type = "plain"),
      list(p = 0.5, se = 0.1, conf.type = "plain", ulimit = NA),
      list(p = 0.5, se = 0.1, conf.type = "plain", ulimit = 0),
      list(p = 0.5, se = 0.1, conf.type = "plain", ulimit = c(TRUE, FALSE)),
      list(p = 0.5, se = 0.1, conf.type = "plain", selow = NULL),
      list(p = 0.5, se = 0.1, conf.type = "plain", conf.int = 0),
      list(p = 0.5, se = 0.1, conf.type = "plain", conf.int = 1),
      list(p = 0.5, se = 0.1, conf.type = "plain", conf.int = NA_real_),
      list(p = 0.5, se = 0.1, conf.type = "plain", conf.int = numeric()),
      list(p = 0.5, se = 0.1, conf.type = "plain", conf.int = c(0.9, 0.95)),
      list(p = 0.5, se = 0.1, conf.type = "p"),
      list(p = 0.5, se = 0.1, conf.type = NA_character_),
      list(p = 0.5, se = 0.1, conf.type = character()),
      list(p = 0.5, se = 0.1, conf.type = c("plain", "log")),
      list(p = 0.5, se = 0.1)
    )
  )
  for (case in survfit_confint_cases) {
    expect_identical(
      capture_survfit_confint(survfit_confint, case),
      capture_survfit_confint(reference_survfit_confint, case)
    )
  }
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
  reference_counting_pseudo_fit$model <- stats::model.frame.default(
    survival::Surv(start, stop, status) ~ 1,
    data = counting_pseudo_data,
    id = counting_pseudo_data$id
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
  reference_grouped_counting_pseudo_fit$model <- stats::model.frame.default(
    survival::Surv(start, stop, status) ~ group,
    data = grouped_counting_pseudo_data,
    id = grouped_counting_pseudo_data$id
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
  expect_null(names(fit_aic))
  expect_equal(fit_aic[[1L]], 1)
  expect_equal(fit_aic[[2L]], as.numeric(AIC(fit)))
  expect_equal(deparse(formula(fit)), "Surv(time, status) ~ x")
  expect_s3_class(terms(fit), "terms")
  expect_null(weights(fit))
  weighted_fit <- coxph(Surv(time, status) ~ x, data = data, weights = wt, max_iter = 0)
  expect_equal(weights(weighted_fit), data$wt)
  expect_error(
    coxph(
      Surv(time, status) ~ x,
      data = data,
      weights = wt,
      method = "exact",
      max_iter = 0
    ),
    "Case weights are not supported for the exact method"
  )
  fitted_values <- fitted(fit)
  expect_true(is.numeric(unlist(fitted_values, use.names = FALSE)))
  expect_equal(length(unlist(fitted_values, use.names = FALSE)), nrow(data))
  fit_summary <- summary(fit)
  expect_s3_class(fit_summary, "summary.survival_py_model")
  expect_equal(rownames(fit_summary$coefficients), "x")
  expect_true(all(c("coef", "se(coef)", "z", "Pr(>|z|)") %in% colnames(fit_summary$coefficients)))
  expect_equal(fit_summary$n, nrow(data))
  fit_summary_print <- capture.output(print(fit_summary))
  expect_true(any(grepl("n= 4", fit_summary_print, fixed = TRUE)))
  expect_true(any(grepl("number of events= 3", fit_summary_print, fixed = TRUE)))
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

  reference_brier_with_newdata <- function(fit, times, newdata, detail = FALSE) {
    model_env <- new.env(parent = environment(fit$terms))
    model_env$newdata <- newdata
    environment(fit$terms) <- model_env
    survival::brier(fit, times = times, newdata = newdata, detail = detail)
  }

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
  reference_brier <- reference_brier_with_newdata(
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

  brier_weighted_data <- transform(brier_data, wt = 1)
  brier_weighted_newdata <- transform(
    brier_weighted_data,
    wt = c(8, 1, 1, 1, 1, 1, 1, 1)
  )
  brier_weighted_fit <- coxph(
    Surv(time, status) ~ x,
    data = brier_weighted_data,
    weights = wt,
    max_iter = 50,
    model = TRUE
  )
  reference_brier_weighted_fit <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = brier_weighted_data,
    weights = wt,
    iter.max = 50,
    model = TRUE,
    y = TRUE
  )
  bridged_brier_weighted <- brier(
    brier_weighted_fit,
    times = c(2, 4, 6),
    newdata = brier_weighted_newdata,
    detail = TRUE
  )
  reference_brier_weighted <- reference_brier_with_newdata(
    reference_brier_weighted_fit,
    times = c(2, 4, 6),
    newdata = brier_weighted_newdata,
    detail = TRUE
  )
  expect_equal(
    bridged_brier_weighted$eff.n,
    reference_brier_weighted$eff.n,
    tolerance = 1e-12
  )
  expect_lt(
    max(abs(bridged_brier_weighted$brier - reference_brier_weighted$brier)),
    3e-3
  )
  expect_lt(
    max(abs(bridged_brier_weighted$rsquared - reference_brier_weighted$rsquared)),
    3e-3
  )

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
  reference_brier_counting <- reference_brier_with_newdata(
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
  reference_brier_common_start <- reference_brier_with_newdata(
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

  brier_custom_id_data <- transform(
    brier_common_start_data[names(brier_common_start_data) != "id"],
    subject = brier_common_start_data$id
  )
  brier_custom_id_fit <- coxph(
    Surv(start, stop, status) ~ x,
    data = brier_custom_id_data,
    id = subject,
    max_iter = 0,
    model = TRUE
  )
  reference_brier_custom_id_fit <- survival::coxph(
    survival::Surv(start, stop, status) ~ x,
    data = brier_custom_id_data,
    id = subject,
    iter.max = 0,
    model = TRUE,
    y = TRUE
  )
  brier_bad_id_newdata <- transform(
    brier_custom_id_data,
    subject = c(1, 2, 2, 1, 3, 3)
  )
  expect_error(
    brier(
      brier_custom_id_fit,
      times = c(3, 5, 7),
      newdata = brier_bad_id_newdata
    ),
    "survcheck"
  )
  expect_error(
    reference_brier_with_newdata(
      reference_brier_custom_id_fit,
      times = c(3, 5, 7),
      newdata = brier_bad_id_newdata
    ),
    "survcheck"
  )

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
  expect_identical(class(multi_formula_concordance), "concordance")
  expect_equal(
    unclass(multi_formula_concordance)[setdiff(names(multi_formula_concordance), "call")],
    unclass(reference_multi_formula_concordance)[setdiff(names(reference_multi_formula_concordance), "call")],
    tolerance = 1e-12
  )
  expect_equal(named_formula_concordance_frame$concordance, formula_concordance_frame$concordance)
  expect_equal(named_formula_concordance_frame$variance, formula_concordance_frame$variance)
  expect_equal(direct_concordance_frame$concordance, 1 - formula_concordance_frame$concordance)
  expect_equal(symbol_concordance_frame$concordance, string_column_concordance_frame$concordance)
  expect_equal(symbol_concordance_frame$variance, string_column_concordance_frame$variance)
  expect_equal(subset_symbol_concordance_frame$concordance, as.numeric(reference_subset_symbol_concordance$concordance))
  expect_equal(subset_symbol_concordance_frame$variance, as.numeric(reference_subset_symbol_concordance$var))

  default_na_data <- data
  default_na_data$x[[2L]] <- NA_real_
  old_na_action <- options(na.action = "na.omit")
  on.exit(options(old_na_action), add = TRUE)
  default_na_concordance <- concordance(
    Surv(time, status) ~ x,
    data = default_na_data,
    influence = 3,
    ranks = TRUE
  )
  reference_default_na_concordance <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = default_na_data,
    influence = 3,
    ranks = TRUE
  )
  expect_identical(class(default_na_concordance), "concordance")
  expect_identical(
    formals(survivalr:::concordance.formula),
    formals(survival:::concordance.formula)
  )
  expect_identical(names(default_na_concordance), names(reference_default_na_concordance))
  expect_equal(
    unclass(default_na_concordance)[setdiff(names(default_na_concordance), "call")],
    unclass(reference_default_na_concordance)[setdiff(names(reference_default_na_concordance), "call")],
    tolerance = 1e-12
  )
  expect_identical(
    as.character(default_na_concordance$call[[1L]]),
    "concordance.formula"
  )
  expect_identical(default_na_concordance$na.action, reference_default_na_concordance$na.action)
  default_na_print <- capture.output(print(default_na_concordance))
  expect_true(any(grepl("Call:", default_na_print, fixed = TRUE)))
  expect_true(any(grepl("observation deleted due to missingness", default_na_print, fixed = TRUE)))
  expect_true(any(grepl("Concordance=", default_na_print, fixed = TRUE)))

  excluded_concordance <- concordance(
    Surv(time, status) ~ x,
    data = default_na_data,
    na.action = stats::na.exclude
  )
  reference_excluded_concordance <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = default_na_data,
    na.action = stats::na.exclude
  )
  expect_identical(excluded_concordance$na.action, reference_excluded_concordance$na.action)
  expect_identical(class(excluded_concordance$na.action), "exclude")

  near_risk <- c(0.5, 0.5 + 5e-13, 0.1, 0.8)
  near_risk_concordance <- concordancefit(
    response,
    near_risk,
    influence = 3,
    ranks = TRUE
  )
  reference_near_risk_concordance <- survival::concordancefit(
    survival::Surv(data$time, data$status),
    near_risk,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(near_risk_concordance$concordance, reference_near_risk_concordance$concordance)
  expect_equal(near_risk_concordance$count, reference_near_risk_concordance$count)
  expect_equal(near_risk_concordance$dfbeta, reference_near_risk_concordance$dfbeta)
  expect_equal(near_risk_concordance$influence, reference_near_risk_concordance$influence)
  expect_equal(near_risk_concordance$ranks, reference_near_risk_concordance$ranks)
  counting_near_risk_response <- Surv(
    c(0, 0, 1, 1),
    c(1, 2, 3, 4),
    c(1, 0, 1, 1)
  )
  counting_near_risk_concordance <- concordancefit(
    counting_near_risk_response,
    near_risk,
    influence = 3,
    ranks = TRUE
  )
  reference_counting_near_risk_concordance <- survival::concordancefit(
    survival::Surv(c(0, 0, 1, 1), c(1, 2, 3, 4), c(1, 0, 1, 1)),
    near_risk,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(
    counting_near_risk_concordance$concordance,
    reference_counting_near_risk_concordance$concordance
  )
  expect_equal(
    counting_near_risk_concordance$count,
    reference_counting_near_risk_concordance$count
  )
  expect_equal(
    counting_near_risk_concordance$dfbeta,
    reference_counting_near_risk_concordance$dfbeta
  )
  expect_equal(
    counting_near_risk_concordance$influence,
    reference_counting_near_risk_concordance$influence
  )
  expect_equal(
    counting_near_risk_concordance$ranks,
    reference_counting_near_risk_concordance$ranks
  )
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
  expect_equal(bridged_concordancefit$cvar, reference_concordancefit$cvar, tolerance = 1e-12)
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

test_that("accelerated failure-time derivative residuals agree with survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  interval_data <- data.frame(
    left = c(1, 1, 1, 1, 2, 2, 3, 4, 5, 6),
    right = c(1, 1, 2, 1, 2, 4, Inf, 4, 7, Inf),
    status = c(1, 2, 3, 0, 1, 3, 0, 1, 3, 0),
    x = c(-1, -.7, -.4, -.1, .2, .5, .8, 1.1, 1.4, 1.7)
  )
  bridged_interval <- survreg(
    Surv(left, right, status, type = "interval") ~ x,
    data = interval_data,
    dist = "weibull"
  )
  reference_interval <- survival::survreg(
    survival::Surv(left, right, status, type = "interval") ~ x,
    data = interval_data,
    dist = "weibull"
  )
  residual_types <- c(
    "response", "deviance", "working", "ldcase", "ldresp",
    "ldshape", "dfbeta", "dfbetas", "matrix"
  )
  for (residual_type in residual_types) {
    expect_equal(
      residuals(bridged_interval, type = residual_type),
      survival:::residuals.survreg(reference_interval, type = residual_type),
      tolerance = 1e-05
    )
  }
  collapse <- rep(1:5, each = 2L)
  expect_equal(
    residuals(
      bridged_interval,
      type = "matrix",
      collapse = collapse,
      weighted = TRUE
    ),
    survival:::residuals.survreg(
      reference_interval,
      type = "matrix",
      collapse = collapse,
      weighted = TRUE
    ),
    tolerance = 1e-05
  )

  stratified_data <- data.frame(
    time = c(
      .8, 1, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2, 3.7, 4.1, 4.8,
      1, 1.4, 1.9, 2.5, 3.1, 3.8, 4.6, 5.5, 6.5, 7.6, 8.8, 10
    ),
    status = rep(c(1, 1, 0, 1), 6),
    x = seq(-1.2, 1.1, length.out = 24),
    group = factor(rep(c("a", "b"), each = 12))
  )
  bridged_stratified <- survreg(
    Surv(time, status) ~ x + strata(group),
    data = stratified_data,
    dist = "weibull"
  )
  reference_stratified <- survival::survreg(
    survival::Surv(time, status) ~ x + strata(group),
    data = stratified_data,
    dist = "weibull"
  )
  for (residual_type in residual_types) {
    expect_equal(
      residuals(bridged_stratified, type = residual_type),
      survival:::residuals.survreg(reference_stratified, type = residual_type),
      tolerance = 1e-08
    )
  }
  for (residual_type in c("dfbeta", "dfbetas", "ldcase", "ldresp", "ldshape")) {
    expect_error(
      residuals(bridged_stratified, type = residual_type, rsigma = FALSE),
      "non-conformable arguments"
    )
    expect_error(
      survival:::residuals.survreg(
        reference_stratified,
        type = residual_type,
        rsigma = FALSE
      ),
      "non-conformable arguments"
    )
  }
})

test_that("aareg drops the reference model-matrix column", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 1:6,
    status = rep(1, 6),
    x = c(0, 1, 2, 1, 3, -1),
    z = c(1, 0, 1, 2, -1, 0),
    group = factor(c("a", "b", "c", "a", "b", "c"))
  )
  length_error <- "invalid 'length' argument"
  expect_error(aareg(Surv(time, status) ~ 1, data = data, nmin = 0), length_error, fixed = TRUE)
  expect_error(
    survival::aareg(survival::Surv(time, status) ~ 1, data = data, nmin = 0),
    length_error,
    fixed = TRUE
  )
  expect_error(
    aareg(Surv(time, status) ~ 0 + x, data = data, nmin = 0),
    length_error,
    fixed = TRUE
  )
  expect_error(
    survival::aareg(survival::Surv(time, status) ~ 0 + x, data = data, nmin = 0),
    length_error,
    fixed = TRUE
  )

  bridged_numeric <- aareg(Surv(time, status) ~ 0 + x + z, data = data, nmin = 0, x = TRUE)
  reference_numeric <- survival::aareg(
    survival::Surv(time, status) ~ 0 + x + z,
    data = data,
    nmin = 0,
    x = TRUE
  )
  bridged_numeric$call <- reference_numeric$call
  expect_equal(bridged_numeric, reference_numeric, tolerance = 1e-10)

  bridged_factor <- aareg(Surv(time, status) ~ 0 + group, data = data, nmin = 0, x = TRUE)
  reference_factor <- survival::aareg(
    survival::Surv(time, status) ~ 0 + group,
    data = data,
    nmin = 0,
    x = TRUE
  )
  bridged_factor$call <- reference_factor$call
  expect_equal(bridged_factor, reference_factor, tolerance = 1e-10)
})

test_that("multi-covariate aareg variance tests preserve reference errors", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 1:6,
    status = rep(1, 6),
    x = c(0, 1, 2, 1, 3, -1),
    z = c(1, 0, 1, 2, -1, 0)
  )
  name_error <- "'names' attribute \\[3\\] must be the same length as the vector \\[2\\]"
  expect_error(
    aareg(Surv(time, status) ~ x + z, data = data, nmin = 1, test = "variance"),
    name_error
  )
  expect_error(
    survival::aareg(
      survival::Surv(time, status) ~ x + z,
      data = data,
      nmin = 1,
      test = "variance"
    ),
    name_error
  )

  expect_error(
    aareg(Surv(time, status) ~ x + z, data = data, nmin = 99, test = "variance"),
    "nmin threshold"
  )
  expect_error(
    survival::aareg(
      survival::Surv(time, status) ~ x + z,
      data = data,
      nmin = 99,
      test = "variance"
    ),
    "threshold 'nmin'",
    fixed = TRUE
  )

  bridged_single <- aareg(
    Surv(time, status) ~ x,
    data = data,
    nmin = 1,
    test = "variance"
  )
  reference_single <- survival::aareg(
    survival::Surv(time, status) ~ x,
    data = data,
    nmin = 1,
    test = "variance"
  )
  bridged_single$call <- reference_single$call
  expect_equal(bridged_single, reference_single, tolerance = 1e-10)
})

test_that("right-censored concordance includes censors at the death time", {
  data <- data.frame(
    time = c(1, 3, 2, 4, 4, 5),
    status = c(1, 0, 1, 1, 0, 1),
    x = c(0.2, 0.9, 0.4, 0.7, 0.7, 0.1),
    z = c(0.8, 0.3, 0.5, 0.2, 0.6, 0.4),
    w = c(1, 2, 0.5, 1.5, 3, 1)
  )
  bridged_formula <- concordance(
    Surv(time, status) ~ x + z,
    data = data,
    influence = 3,
    ranks = TRUE
  )
  reference_formula <- survival::concordance(
    survival::Surv(time, status) ~ x + z,
    data = data,
    influence = 3,
    ranks = TRUE
  )
  formula_fields <- setdiff(names(reference_formula), "call")
  expect_identical(names(bridged_formula), names(reference_formula))
  expect_equal(
    unclass(bridged_formula)[formula_fields],
    unclass(reference_formula)[formula_fields],
    tolerance = 1e-12
  )

  score_matrix <- as.matrix(data[c("x", "z")])
  bridged_fit <- concordancefit(
    Surv(data$time, data$status),
    score_matrix,
    weights = data$w,
    timewt = "S",
    influence = 3,
    ranks = TRUE
  )
  reference_fit <- survival::concordancefit(
    survival::Surv(data$time, data$status),
    score_matrix,
    weights = data$w,
    timewt = "S",
    influence = 3,
    ranks = TRUE
  )
  expect_identical(names(bridged_fit), names(reference_fit))
  expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)
})

test_that("concordance censoring weights use the post-death risk set", {
  data <- data.frame(
    time = c(1, 2, 2, 3, 4),
    status = c(1, 1, 0, 1, 0),
    x = c(0.1, 0.2, 0.2, 0.8, 0.4),
    w = c(1, 2, 3, 1, 2)
  )

  for (time_weight in c("S/G", "n/G2")) {
    bridged_fit <- concordancefit(
      Surv(data$time, data$status),
      data$x,
      weights = data$w,
      timewt = time_weight,
      influence = 3,
      ranks = TRUE
    )
    reference_fit <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      data$x,
      weights = data$w,
      timewt = time_weight,
      influence = 3,
      ranks = TRUE
    )

    expect_identical(names(bridged_fit), names(reference_fit))
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)
  }
})

test_that("counting concordance censoring-weight diagnostics match reference", {
  data <- data.frame(
    start = c(0, 1, 2, 0),
    stop = c(1, 3, 4, 5),
    status = c(1, 0, 1, 1),
    x = c(0.1, 0.4, 0.2, 0.8)
  )

  for (time_weight in c("S/G", "n/G2")) {
    expected_error <- paste(
      time_weight,
      "timewt option not supported for (time1, time2) data"
    )
    expect_error(
      concordancefit(
        Surv(data$start, data$stop, data$status),
        data$x,
        timewt = time_weight
      ),
      expected_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(data$start, data$stop, data$status),
        data$x,
        timewt = time_weight
      ),
      expected_error,
      fixed = TRUE
    )
    expect_error(
      concordance(
        Surv(start, stop, status) ~ x,
        data = data,
        timewt = time_weight
      ),
      expected_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordance(
        survival::Surv(start, stop, status) ~ x,
        data = data,
        timewt = time_weight
      ),
      expected_error,
      fixed = TRUE
    )
  }
})

test_that("one-event concordance ranks match reference dimensions", {
  data <- data.frame(
    time = c(1, 2, 3),
    status = c(1, 0, 0),
    x = c(0.2, 0.6, 0.4),
    z = c(0.8, 0.1, 0.5),
    q = c(0.3, 0.4, 0.9)
  )
  bridged_response <- Surv(data$time, data$status)
  reference_response <- survival::Surv(data$time, data$status)
  score_sets <- list(
    data$x,
    as.matrix(data[c("x", "z")]),
    as.matrix(data[c("x", "z", "q")])
  )

  for (scores in score_sets) {
    bridged_fit <- concordancefit(bridged_response, scores, ranks = TRUE)
    reference_fit <- survival::concordancefit(reference_response, scores, ranks = TRUE)
    expect_identical(names(bridged_fit), names(reference_fit))
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)
  }

  bridged_formula <- concordance(
    Surv(time, status) ~ x + z,
    data = data,
    ranks = TRUE
  )
  reference_formula <- survival::concordance(
    survival::Surv(time, status) ~ x + z,
    data = data,
    ranks = TRUE
  )
  formula_fields <- setdiff(names(reference_formula), "call")
  expect_equal(
    unclass(bridged_formula)[formula_fields],
    unclass(reference_formula)[formula_fields],
    tolerance = 1e-12
  )

  expect_error(
    concordancefit(bridged_response, data$x, ranks = TRUE, reverse = TRUE),
    "undefined columns selected"
  )
  expect_error(
    survival::concordancefit(
      reference_response,
      data$x,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
  expect_error(
    concordance(
      Surv(time, status) ~ x,
      data = data,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
  expect_error(
    survival::concordance(
      survival::Surv(time, status) ~ x,
      data = data,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )

  stratified_response <- Surv(c(1, 2, 1, 2), c(1, 0, 0, 0))
  reference_stratified_response <- survival::Surv(c(1, 2, 1, 2), c(1, 0, 0, 0))
  stratified_scores <- c(0.2, 0.6, 0.4, 0.8)
  stratified_groups <- c("a", "a", "b", "b")
  replacement_error <- "number of items to replace is not a multiple of replacement length"
  expect_error(
    concordancefit(
      stratified_response,
      stratified_scores,
      strata = stratified_groups,
      ranks = TRUE
    ),
    replacement_error
  )
  expect_error(
    survival::concordancefit(
      reference_stratified_response,
      stratified_scores,
      strata = stratified_groups,
      ranks = TRUE
    ),
    replacement_error
  )
})

test_that("unnamed concordance score matrices use reference names", {
  score_matrix <- matrix(
    c(0.2, 0.6, 0.4, 0.9, 0.3, 0.8, 0.1, 0.5, 0.2, 0.7),
    ncol = 2
  )
  fixtures <- list(
    list(time = c(1, 2, 3, 4, 5), status = c(1, 1, 0, 1, 0)),
    list(time = c(1, 2, 3, 4, 5), status = c(1, 0, 0, 0, 0))
  )

  for (fixture in fixtures) {
    bridged_fit <- concordancefit(
      Surv(fixture$time, fixture$status),
      score_matrix,
      influence = 3,
      ranks = TRUE
    )
    reference_fit <- survival::concordancefit(
      survival::Surv(fixture$time, fixture$status),
      score_matrix,
      influence = 3,
      ranks = TRUE
    )

    expect_identical(names(bridged_fit), names(reference_fit))
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)
  }
})

test_that("reversed multi-score concordance ranks match reference errors", {
  data <- data.frame(
    time = c(1, 2, 3, 4, 5),
    status = c(1, 1, 0, 1, 0),
    x = c(0.2, 0.6, 0.4, 0.9, 0.3),
    z = c(0.8, 0.1, 0.5, 0.2, 0.7)
  )
  score_matrix <- as.matrix(data[c("x", "z")])

  expect_error(
    concordancefit(
      Surv(data$time, data$status),
      score_matrix,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(data$time, data$status),
      score_matrix,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
  expect_error(
    concordance(
      Surv(time, status) ~ x + z,
      data = data,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
  expect_error(
    survival::concordance(
      survival::Surv(time, status) ~ x + z,
      data = data,
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected"
  )
})

test_that("disabled standard errors with multiple scores match reference errors", {
  y <- Surv(c(5, 4, 4, 3, 3, 6), c(0, 1, 1, 0, 1, 1))
  reference_y <- survival::Surv(c(5, 4, 4, 3, 3, 6), c(0, 1, 1, 0, 1, 1))
  x <- cbind(
    s1 = c(-1, 0, 1, 0.5, 0.5, 0),
    s2 = c(0.5, -0.5, 1, 0, -0.5, -0.5)
  )

  for (reverse_value in c(FALSE, TRUE)) {
    expect_error(
      concordancefit(
        y,
        x,
        timewt = "I",
        influence = 2,
        ranks = TRUE,
        reverse = reverse_value,
        keepstrata = 1,
        std.err = FALSE
      ),
      "subscript out of bounds",
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        reference_y,
        x,
        timewt = "I",
        influence = 2,
        ranks = TRUE,
        reverse = reverse_value,
        keepstrata = 1,
        std.err = FALSE
      ),
      "subscript out of bounds",
      fixed = TRUE
    )
  }

  strata_values <- rep(c("a", "b"), each = 3)
  dimension_error <- "length of 'dimnames' [1] not equal to array extent"
  length_warning <- paste(
    "data length [20] is not a sub-multiple or multiple",
    "of the number of columns [6]"
  )
  expect_warning(
    expect_error(
      concordancefit(
        y,
        x,
        strata = strata_values,
        timewt = "S",
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    length_warning,
    fixed = TRUE
  )
  expect_warning(
    expect_error(
      survival::concordancefit(
        reference_y,
        x,
        strata = strata_values,
        timewt = "S",
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    length_warning,
    fixed = TRUE
  )

  sparse_y <- Surv(seq_len(6), c(1, 0, 0, 0, 1, 0))
  reference_sparse_y <- survival::Surv(seq_len(6), c(1, 0, 0, 0, 1, 0))
  sparse_x <- cbind(s1 = seq_len(6), s2 = rev(seq_len(6)))
  sparse_strata <- rep(seq_len(3), each = 2)
  sparse_warning <- paste(
    "data length [32] is not a sub-multiple or multiple",
    "of the number of rows [6]"
  )
  expect_warning(
    expect_error(
      concordancefit(
        sparse_y,
        sparse_x,
        strata = sparse_strata,
        timewt = "S",
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    sparse_warning,
    fixed = TRUE
  )
  expect_warning(
    expect_error(
      survival::concordancefit(
        reference_sparse_y,
        sparse_x,
        strata = sparse_strata,
        timewt = "S",
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    sparse_warning,
    fixed = TRUE
  )

  counting_y <- Surv(
    c(4, 3, 0, 1, 3, 2, 3),
    c(6, 6, 1, 4, 4, 5, 4),
    c(1, 0, 0, 1, 0, 0, 0)
  )
  reference_counting_y <- survival::Surv(
    c(4, 3, 0, 1, 3, 2, 3),
    c(6, 6, 1, 4, 4, 5, 4),
    c(1, 0, 0, 1, 0, 0, 0)
  )
  counting_x <- cbind(
    s1 = c(0.5, 0, 1, -1, 1, 1, 1),
    s2 = c(-1, 0, -1, 0.5, 0, 0, 0),
    s3 = c(-0.5, 0.5, 0, 0.5, 0, -1, 1)
  )
  counting_strata <- c(3, 3, 1, 2, 1, 1, 2)
  counting_weights <- c(2, 0.5, 1.5, 0.5, 2, 2, 0.5)
  expect_no_warning(
    expect_error(
      concordancefit(
        counting_y,
        counting_x,
        strata = counting_strata,
        weights = counting_weights,
        ymax = 6,
        influence = 3,
        ranks = TRUE,
        reverse = TRUE,
        keepstrata = 2,
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    )
  )
  expect_no_warning(
    expect_error(
      survival::concordancefit(
        reference_counting_y,
        counting_x,
        strata = counting_strata,
        weights = counting_weights,
        ymax = 6,
        influence = 3,
        ranks = TRUE,
        reverse = TRUE,
        keepstrata = 2,
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    )
  )
})

test_that("disabled standard errors recycle retained strata counts like reference", {
  data <- data.frame(
    time = c(4, 6, 5, 7, 2, 5, 4, 2, 2, 4),
    status = c(0, 0, 1, 0, 1, 0, 1, 0, 0, 0),
    x = c(-0.5, -0.5, -1, 1, 0.5, -0.5, 0, -0.5, -1, 0.5),
    group = c(1, 3, 1, 2, 2, 1, 3, 3, 1, 2)
  )
  warning_message <- paste(
    "data length [15] is not a sub-multiple or multiple",
    "of the number of columns [6]"
  )

  bridged <- NULL
  expect_warning(
    bridged <- concordancefit(
      Surv(data$time, data$status),
      as.matrix(data["x"]),
      strata = data$group,
      ymax = 4,
      keepstrata = TRUE,
      std.err = FALSE
    ),
    warning_message,
    fixed = TRUE
  )
  reference <- NULL
  expect_warning(
    reference <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      as.matrix(data["x"]),
      strata = data$group,
      ymax = 4,
      keepstrata = TRUE,
      std.err = FALSE
    ),
    warning_message,
    fixed = TRUE
  )
  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
})

test_that("single-event strata use unweighted time counts like reference", {
  data <- data.frame(
    time = c(4, 6, 4, 4, 7, 2, 5, 3, 5, 3),
    status = c(0, 1, 1, 0, 0, 1, 0, 0, 0, 0),
    x = c(0.5, -1, -0.5, 1, 1, -0.5, 0, 0.5, 1, 0.5),
    group = c(1, 2, 2, 2, 1, 1, 1, 3, 3, 3)
  )

  bridged <- concordancefit(
    Surv(data$time, data$status),
    data$x,
    strata = data$group,
    timewt = "I",
    reverse = TRUE,
    keepstrata = TRUE,
    influence = 3
  )
  reference <- survival::concordancefit(
    survival::Surv(data$time, data$status),
    data$x,
    strata = data$group,
    timewt = "I",
    reverse = TRUE,
    keepstrata = TRUE,
    influence = 3
  )
  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)

  warning_message <- paste(
    "data length [16] is not a sub-multiple or multiple",
    "of the number of rows [3]"
  )
  bridged_no_error <- NULL
  expect_warning(
    bridged_no_error <- concordancefit(
      Surv(data$time, data$status),
      data$x,
      strata = data$group,
      timewt = "I",
      reverse = TRUE,
      keepstrata = TRUE,
      std.err = FALSE
    ),
    warning_message,
    fixed = TRUE
  )
  reference_no_error <- NULL
  expect_warning(
    reference_no_error <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      data$x,
      strata = data$group,
      timewt = "I",
      reverse = TRUE,
      keepstrata = TRUE,
      std.err = FALSE
    ),
    warning_message,
    fixed = TRUE
  )
  expect_equal(
    unclass(bridged_no_error),
    unclass(reference_no_error),
    tolerance = 1e-12
  )

  bridged_formula <- concordance(
    Surv(time, status) ~ x + strata(group),
    data = data,
    timewt = "I",
    keepstrata = TRUE,
    influence = 3
  )
  reference_formula <- survival::concordance(
    survival::Surv(time, status) ~ x + strata(group),
    data = data,
    timewt = "I",
    keepstrata = TRUE,
    influence = 3
  )
  fields <- setdiff(names(reference_formula), "call")
  expect_equal(
    unclass(bridged_formula)[fields],
    unclass(reference_formula)[fields],
    tolerance = 1e-12
  )
})

test_that("disabled standard errors warn before collapsed strata errors", {
  data <- data.frame(
    time = c(3, 2, 1, 1, 1, 3, 1, 4),
    status = c(1, 0, 0, 0, 1, 0, 0, 0),
    x = c(0.5, 1, -0.5, -1, 0.5, 0, 0, 0),
    group = c(1, 2, 2, 1, 2, 1, 2, 1)
  )
  warning_message <- paste(
    "data length [10] is not a sub-multiple or multiple",
    "of the number of columns [6]"
  )
  dimension_error <- "'x' must be an array of at least two dimensions"

  expect_warning(
    expect_error(
      concordancefit(
        Surv(data$time, data$status),
        data$x,
        strata = data$group,
        ymin = 3,
        timewt = "S",
        reverse = TRUE,
        keepstrata = 1,
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    warning_message,
    fixed = TRUE
  )
  expect_warning(
    expect_error(
      survival::concordancefit(
        survival::Surv(data$time, data$status),
        data$x,
        strata = data$group,
        ymin = 3,
        timewt = "S",
        reverse = TRUE,
        keepstrata = 1,
        std.err = FALSE
      ),
      dimension_error,
      fixed = TRUE
    ),
    warning_message,
    fixed = TRUE
  )
})

test_that("multi-score influence covariance matches reference shape", {
  time <- c(1, 2, 3, 4, 5, 6)
  status <- c(1, 1, 0, 1, 0, 1)
  x <- cbind(
    s1 = c(0, 0.5, 1, -0.5, 0, 0.5),
    s2 = c(1, 0, -0.5, 0.5, 0, -1)
  )
  bridged <- concordancefit(
    Surv(time, status),
    x,
    influence = 2
  )
  reference <- survival::concordancefit(
    survival::Surv(time, status),
    x,
    influence = 2
  )

  expect_true(is.matrix(bridged$var))
  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
})

test_that("stratified multi-score assembly matches reference behavior", {
  data <- data.frame(
    time = rep(c(1, 2, 3), 2),
    status = rep(c(1, 0, 1), 2),
    x = seq_len(6) / 6,
    z = rev(seq_len(6)) / 6,
    group = rep(c("a", "b"), each = 3)
  )
  score_matrix <- as.matrix(data[c("x", "z")])
  dimension_error <- "length of 'dimnames' [1] not equal to array extent"

  for (time_weight in c("n", "S", "S/G", "n/G2", "I")) {
    expect_error(
      concordancefit(
        Surv(data$time, data$status),
        score_matrix,
        strata = data$group,
        timewt = time_weight,
        keepstrata = TRUE
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(data$time, data$status),
        score_matrix,
        strata = data$group,
        timewt = time_weight,
        keepstrata = TRUE
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      concordance(
        Surv(time, status) ~ x + z + strata(group),
        data = data,
        timewt = time_weight,
        keepstrata = TRUE
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordance(
        survival::Surv(time, status) ~ x + z + strata(group),
        data = data,
        timewt = time_weight,
        keepstrata = TRUE
      ),
      dimension_error,
      fixed = TRUE
    )
  }

  large_data <- data.frame(
    group = rep(seq_len(11), each = 2),
    time = rep(c(1, 2), 11),
    status = rep(c(1, 0), 11),
    x = seq_len(22) / 22,
    z = rev(seq_len(22)) / 22
  )
  large_scores <- as.matrix(large_data[c("x", "z")])
  for (time_weight in c("n", "I")) {
    bridged_fit <- concordancefit(
      Surv(large_data$time, large_data$status),
      large_scores,
      strata = large_data$group,
      timewt = time_weight
    )
    reference_fit <- survival::concordancefit(
      survival::Surv(large_data$time, large_data$status),
      large_scores,
      strata = large_data$group,
      timewt = time_weight
    )
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)

    bridged_formula <- concordance(
      Surv(time, status) ~ x + z + strata(group),
      data = large_data,
      timewt = time_weight
    )
    reference_formula <- survival::concordance(
      survival::Surv(time, status) ~ x + z + strata(group),
      data = large_data,
      timewt = time_weight
    )
    fields <- setdiff(names(reference_formula), "call")
    expect_equal(
      unclass(bridged_formula)[fields],
      unclass(reference_formula)[fields],
      tolerance = 1e-12
    )
  }
})

test_that("many concordance strata collapse before rank assembly", {
  n <- 22L
  data <- data.frame(
    start = rep(0, n),
    stop = rep(c(1, 2), 11),
    status = rep(c(1, 0, 0, 0), length.out = n),
    group = rep(seq_len(11), each = 2)
  )
  score_matrix <- cbind(
    s1 = seq_len(n),
    s2 = rev(seq_len(n)),
    s3 = rep(c(0, 1), 11)
  )
  data[c("s1", "s2", "s3")] <- as.data.frame(score_matrix)
  response <- Surv(data$start, data$stop, data$status)
  reference_response <- survival::Surv(data$start, data$stop, data$status)

  bridged <- concordancefit(
    response,
    score_matrix,
    strata = data$group,
    timewt = "I",
    ranks = TRUE
  )
  reference <- survival::concordancefit(
    reference_response,
    score_matrix,
    strata = data$group,
    timewt = "I",
    ranks = TRUE
  )
  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)

  formula <- Surv(start, stop, status) ~ s1 + s2 + s3 + strata(group)
  reference_formula <- survival::Surv(start, stop, status) ~
    s1 + s2 + s3 + strata(group)
  bridged_formula <- concordance(formula, data = data, timewt = "I", ranks = TRUE)
  reference_formula_result <- survival::concordance(
    reference_formula,
    data = data,
    timewt = "I",
    ranks = TRUE
  )
  fields <- setdiff(names(reference_formula_result), "call")
  expect_equal(
    unclass(bridged_formula)[fields],
    unclass(reference_formula_result)[fields],
    tolerance = 1e-12
  )

  expect_error(
    concordancefit(
      response,
      score_matrix,
      strata = data$group,
      timewt = "I",
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected",
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      reference_response,
      score_matrix,
      strata = data$group,
      timewt = "I",
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected",
    fixed = TRUE
  )
})

test_that("uneven stratified concordance ranks match reference errors", {
  data <- data.frame(
    time = c(1, 2, 3, 1, 2, 3),
    status = c(1, 1, 0, 1, 1, 1),
    x = seq_len(6),
    z = rev(seq_len(6)),
    group = rep(c("a", "b"), each = 3)
  )
  replacement_error <- "number of items to replace is not a multiple of replacement length"

  for (scores in list(data$x, as.matrix(data[c("x", "z")]))) {
    expect_error(
      concordancefit(
        Surv(data$time, data$status),
        scores,
        strata = data$group,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(data$time, data$status),
        scores,
        strata = data$group,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
  }

  formulas <- list(
    Surv(time, status) ~ x + strata(group),
    Surv(time, status) ~ x + z + strata(group)
  )
  reference_formulas <- list(
    survival::Surv(time, status) ~ x + strata(group),
    survival::Surv(time, status) ~ x + z + strata(group)
  )
  for (index in seq_along(formulas)) {
    expect_error(
      concordance(formulas[[index]], data = data, ranks = TRUE),
      replacement_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordance(
        reference_formulas[[index]],
        data = data,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
  }
})

test_that("stratified concordance rank recycling uses display times", {
  time <- c(2, 6, 7, 2, 1, 7, 6)
  status <- c(1, 0, 1, 0, 0, 1, 0)
  scores <- c(0.5, 0.5, -1, 0.5, 0, 0.5, 1)
  groups <- c(1, 1, 1, 2, 2, 2, 1)
  weights <- c(1.5, 1.5, 0.5, 1.5, 1, 1.5, 0.5)

  bridged <- concordancefit(
    Surv(time, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 1,
    timewt = "n/G2",
    influence = 3,
    ranks = TRUE,
    timefix = FALSE,
    keepstrata = TRUE
  )
  reference <- survival::concordancefit(
    survival::Surv(time, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 1,
    timewt = "n/G2",
    influence = 3,
    ranks = TRUE,
    timefix = FALSE,
    keepstrata = TRUE
  )
  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
})

test_that("reversed stratified counting ranks flip after recycling", {
  start <- c(0, 4, 3, 1, 3, 1, 2)
  stop <- c(2, 8, 7, 5, 7, 3, 5)
  status <- c(1, 0, 0, 0, 1, 0, 1)
  scores <- c(1, 0.5, 1, 0.5, -1, 0, -0.5)
  groups <- c(1, 1, 1, 2, 1, 2, 2)
  weights <- c(1.5, 1.5, 0.5, 1.5, 1, 1.5, 0.5)

  bridged <- concordancefit(
    Surv(start, stop, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 2,
    timewt = "I",
    influence = 2,
    ranks = TRUE,
    reverse = TRUE,
    timefix = FALSE,
    keepstrata = 10
  )
  reference <- survival::concordancefit(
    survival::Surv(start, stop, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 2,
    timewt = "I",
    influence = 2,
    ranks = TRUE,
    reverse = TRUE,
    timefix = FALSE,
    keepstrata = 10
  )

  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
})

test_that("bounded stratified ranks exclude time-weight padding rows", {
  start <- c(2, 1, 1, 2, 0, 4)
  stop <- c(6, 3, 4, 4, 4, 7)
  status <- c(0, 1, 1, 0, 0, 1)
  scores <- c(-1, -0.5, 1, -1, 0, 0)
  groups <- c(1, 2, 1, 2, 2, 1)
  weights <- c(1, 0.5, 2, 0.5, 0.5, 1)
  clusters <- c(2, 1, 3, 3, 1, 2)

  bridged <- concordancefit(
    Surv(start, stop, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 3,
    ymax = 5,
    timewt = "I",
    cluster = clusters,
    ranks = TRUE,
    reverse = TRUE,
    timefix = FALSE,
    keepstrata = TRUE
  )
  reference <- survival::concordancefit(
    survival::Surv(start, stop, status),
    scores,
    strata = groups,
    weights = weights,
    ymin = 3,
    ymax = 5,
    timewt = "I",
    cluster = clusters,
    ranks = TRUE,
    reverse = TRUE,
    timefix = FALSE,
    keepstrata = TRUE
  )

  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
})

test_that("stratified concordance rank errors follow stratum order", {
  start <- rep(0, 6)
  stop <- c(1, 2, 6, 2, 1, 2)
  status <- c(1, 0, 1, 0, 0, 0)
  scores <- seq_len(6)
  groups <- rep(seq_len(3), each = 2)
  replacement_error <- "replacement has length zero"

  expect_error(
    concordancefit(
      Surv(start, stop, status),
      scores,
      strata = groups,
      ymax = 5,
      timewt = "S",
      ranks = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(start, stop, status),
      scores,
      strata = groups,
      ymax = 5,
      timewt = "S",
      ranks = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )
})

test_that("non-recyclable rank strata fail before later empty strata", {
  time <- c(4, 5, 5, 3, 6, 4, 7, 5, 7, 6)
  status <- c(0, 1, 0, 1, 0, 0, 1, 0, 0, 0)
  scores <- seq_along(time)
  groups <- rep(seq_len(2), each = 5)
  replacement_error <- "number of items to replace is not a multiple of replacement length"

  for (score_input in list(scores, cbind(x = scores, z = rev(scores)))) {
    expect_error(
      concordancefit(
        Surv(time, status),
        score_input,
        strata = groups,
        ymax = 5,
        timewt = "S",
        ranks = TRUE,
        reverse = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(time, status),
        score_input,
        strata = groups,
        ymax = 5,
        timewt = "S",
        ranks = TRUE,
        reverse = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
  }
})

test_that("bounded counting multi-score ranks report the first empty stratum", {
  start <- c(0, 1, 4, 0, 0, 3, 2, 0)
  stop <- c(1, 4, 6, 3, 2, 7, 4, 2)
  status <- c(1, 1, 1, 0, 0, 1, 0, 0)
  scores <- cbind(
    s1 = c(0.5, 0, 0, 0, -0.5, 1, 0, 0),
    s2 = c(0, 0.5, -0.5, 0.5, -0.5, -1, 0.5, 0.5)
  )
  groups <- c(3, 1, 1, 2, 2, 2, 3, 1)
  weights <- c(2, 2, 1.5, 2, 2, 1.5, 1.5, 1.5)
  empty_error <- "replacement has length zero"

  expect_error(
    concordancefit(
      Surv(start, stop, status),
      scores,
      strata = groups,
      weights = weights,
      ymax = 6,
      timewt = "S",
      influence = 3,
      ranks = TRUE,
      keepstrata = FALSE
    ),
    empty_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(start, stop, status),
      scores,
      strata = groups,
      weights = weights,
      ymax = 6,
      timewt = "S",
      influence = 3,
      ranks = TRUE,
      keepstrata = FALSE
    ),
    empty_error,
    fixed = TRUE
  )
})

test_that("counting multi-score rank recycling follows positive S weights", {
  start <- c(2, 4, 3, 4, 2, 1, 3, 2)
  stop <- c(4, 8, 5, 7, 3, 3, 4, 6)
  status <- c(1, 1, 1, 0, 1, 0, 1, 0)
  scores <- cbind(
    s1 = c(-0.5, 1, -0.5, 0.5, 1, -0.5, -0.5, 1),
    s2 = c(1, -0.5, 1, 0, -0.5, -0.5, -0.5, 0)
  )
  groups <- c(1, 3, 1, 3, 2, 2, 2, 1)
  weights <- c(1.5, 2, 2, 1.5, 1.5, 1.5, 0.5, 0.5)
  replacement_error <- "number of items to replace is not a multiple of replacement length"

  expect_error(
    concordancefit(
      Surv(start, stop, status),
      scores,
      strata = groups,
      weights = weights,
      ymax = 4,
      timewt = "S",
      ranks = TRUE,
      keepstrata = FALSE
    ),
    replacement_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(start, stop, status),
      scores,
      strata = groups,
      weights = weights,
      ymax = 4,
      timewt = "S",
      ranks = TRUE,
      keepstrata = FALSE
    ),
    replacement_error,
    fixed = TRUE
  )
})

test_that("multi-score rank errors finish stratum construction first", {
  time <- c(6, 7, 1, 2)
  status <- c(1, 1, 0, 0)
  scores <- cbind(
    x = c(0.2, 0.8, 0.4, 0.6),
    z = c(0.9, 0.1, 0.3, 0.7)
  )
  groups <- c(1, 1, 2, 2)
  null_error <- "'data' must be of a vector type, was 'NULL'"

  expect_error(
    concordancefit(
      Surv(time, status),
      scores,
      strata = groups,
      ymax = 5,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(time, status),
      scores,
      strata = groups,
      ymax = 5,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
})

test_that("retained many-strata rank errors preserve stratum order", {
  time <- c(1, 2, 7, 3, rep(c(1, 2), 9))
  status <- c(1, 0, 1, 0, rep(0, 18))
  scores <- seq_along(time)
  groups <- rep(seq_len(11), each = 2)
  replacement_error <- "replacement has length zero"

  expect_error(
    concordancefit(
      Surv(time, status),
      scores,
      strata = groups,
      ymax = 5,
      timewt = "n",
      ranks = TRUE,
      keepstrata = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(time, status),
      scores,
      strata = groups,
      ymax = 5,
      timewt = "n",
      ranks = TRUE,
      keepstrata = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )
})

test_that("clustered concordance dfbeta matches reference order and names", {
  data <- data.frame(
    time = c(2, 3, 4, 5, 2),
    status = c(0, 1, 1, 0, 1),
    x = c(-1, 0, 1, 0, 1),
    z = c(1, -1, 0, 0.5, 1),
    cluster = c(2, 1, 1, 2, 3)
  )

  for (scores in list(data$x, as.matrix(data[c("x", "z")]))) {
    bridged <- concordancefit(
      Surv(data$time, data$status),
      scores,
      cluster = data$cluster,
      ymin = 3,
      timewt = "S",
      influence = 3,
      timefix = FALSE
    )
    reference <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      scores,
      cluster = data$cluster,
      ymin = 3,
      timewt = "S",
      influence = 3,
      timefix = FALSE
    )
    expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
  }

  formulas <- list(Surv(time, status) ~ x, Surv(time, status) ~ x + z)
  reference_formulas <- list(
    survival::Surv(time, status) ~ x,
    survival::Surv(time, status) ~ x + z
  )
  for (index in seq_along(formulas)) {
    bridged <- concordance(
      formulas[[index]],
      data = data,
      cluster = data$cluster,
      ymin = 3,
      timewt = "S",
      influence = 3,
      timefix = FALSE
    )
    reference <- survival::concordance(
      reference_formulas[[index]],
      data = data,
      cluster = data$cluster,
      ymin = 3,
      timewt = "S",
      influence = 3,
      timefix = FALSE
    )
    fields <- setdiff(names(reference), "call")
    expect_equal(
      unclass(bridged)[fields],
      unclass(reference)[fields],
      tolerance = 1e-12
    )
  }
})

test_that("empty concordance rank strata match reference diagnostics", {
  data <- data.frame(
    time = c(1, 2, 1, 2),
    status = c(1, 0, 0, 0),
    x = c(0.1, 0.4, 0.2, 0.8),
    z = c(0.8, 0.2, 0.6, 0.1),
    group = c("a", "a", "b", "b")
  )
  replacement_error <- "number of items to replace is not a multiple of replacement length"
  null_error <- "'data' must be of a vector type, was 'NULL'"

  expect_error(
    concordancefit(
      Surv(data$time, data$status),
      data$x,
      strata = data$group,
      ranks = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(data$time, data$status),
      data$x,
      strata = data$group,
      ranks = TRUE
    ),
    replacement_error,
    fixed = TRUE
  )

  score_matrix <- as.matrix(data[c("x", "z")])
  expect_error(
    concordancefit(
      Surv(data$time, data$status),
      score_matrix,
      strata = data$group,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(data$time, data$status),
      score_matrix,
      strata = data$group,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    concordance(
      Surv(time, status) ~ x + z + strata(group),
      data = data,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordance(
      survival::Surv(time, status) ~ x + z + strata(group),
      data = data,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )

  censored_data <- transform(data, status = 0)
  expect_error(
    concordancefit(
      Surv(censored_data$time, censored_data$status),
      score_matrix,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(censored_data$time, censored_data$status),
      score_matrix,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    concordance(
      Surv(time, status) ~ x + z,
      data = censored_data,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordance(
      survival::Surv(time, status) ~ x + z,
      data = censored_data,
      ranks = TRUE
    ),
    null_error,
    fixed = TRUE
  )
})

test_that("empty concordance rank tables match reference shapes", {
  data <- data.frame(
    time = c(4, 1, 6, 5, 2, 6, 5),
    status = c(0, 0, 0, 1, 0, 1, 0),
    x = c(1, 0, 0.5, 0, 0, 1, 1),
    z = c(0, -0.5, -0.5, 1, 0.5, 0, -1)
  )
  single_score <- as.matrix(data["x"])
  multi_score <- as.matrix(data[c("x", "z")])

  for (reverse_value in c(FALSE, TRUE)) {
    bridged_single <- concordancefit(
      Surv(data$time, data$status),
      single_score,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = reverse_value
    )
    reference_single <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      single_score,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = reverse_value
    )
    expect_equal(unclass(bridged_single), unclass(reference_single), tolerance = 1e-12)

    bridged_formula <- concordance(
      Surv(time, status) ~ x,
      data = data,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = reverse_value
    )
    reference_formula <- survival::concordance(
      survival::Surv(time, status) ~ x,
      data = data,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = reverse_value
    )
    fields <- setdiff(names(reference_formula), "call")
    expect_equal(
      unclass(bridged_formula)[fields],
      unclass(reference_formula)[fields],
      tolerance = 1e-12
    )
  }

  bridged_multi <- concordancefit(
    Surv(data$time, data$status),
    multi_score,
    ymax = 4,
    timewt = "S/G",
    ranks = TRUE
  )
  reference_multi <- survival::concordancefit(
    survival::Surv(data$time, data$status),
    multi_score,
    ymax = 4,
    timewt = "S/G",
    ranks = TRUE
  )
  expect_equal(unclass(bridged_multi), unclass(reference_multi), tolerance = 1e-12)
  expect_error(
    concordancefit(
      Surv(data$time, data$status),
      multi_score,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected",
    fixed = TRUE
  )
  expect_error(
    survival::concordancefit(
      survival::Surv(data$time, data$status),
      multi_score,
      ymax = 4,
      timewt = "S/G",
      ranks = TRUE,
      reverse = TRUE
    ),
    "undefined columns selected",
    fixed = TRUE
  )

  stratified <- data.frame(
    time = c(1, 5, 2, 6),
    status = c(0, 1, 0, 1),
    x = c(0.1, 0.4, 0.2, 0.8),
    z = c(0.8, 0.2, 0.6, 0.1),
    group = c("a", "a", "b", "b")
  )
  replacement_error <- "replacement has length zero"
  for (scores in list(stratified$x, as.matrix(stratified[c("x", "z")]))) {
    expect_error(
      concordancefit(
        Surv(stratified$time, stratified$status),
        scores,
        strata = stratified$group,
        ymax = 4,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(stratified$time, stratified$status),
        scores,
        strata = stratified$group,
        ymax = 4,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
  }

  for (formula in list(
      Surv(time, status) ~ x + strata(group),
      Surv(time, status) ~ x + z + strata(group))) {
    reference_formula <- stats::update(formula, survival::Surv(time, status) ~ .)
    expect_error(
      concordance(formula, data = stratified, ymax = 4, ranks = TRUE),
      replacement_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordance(
        reference_formula,
        data = stratified,
        ymax = 4,
        ranks = TRUE
      ),
      replacement_error,
      fixed = TRUE
    )
  }
})

test_that("zero-pair concordance diagnostics match reference values", {
  data <- data.frame(
    time = c(4, 1, 6, 5, 2, 6, 5),
    status = c(0, 0, 0, 1, 0, 1, 0),
    x = c(1, 0, 0.5, 0, 0, 1, 1),
    z = c(0, -0.5, -0.5, 1, 0.5, 0, -1)
  )
  score_sets <- list(as.matrix(data["x"]), as.matrix(data[c("x", "z")]))

  for (scores in score_sets) {
    bridged <- concordancefit(
      Surv(data$time, data$status),
      scores,
      ymax = 4,
      timewt = "S/G",
      influence = 3
    )
    reference <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      scores,
      ymax = 4,
      timewt = "S/G",
      influence = 3
    )
    expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
  }

  formulas <- list(Surv(time, status) ~ x, Surv(time, status) ~ x + z)
  reference_formulas <- list(
    survival::Surv(time, status) ~ x,
    survival::Surv(time, status) ~ x + z
  )
  for (index in seq_along(formulas)) {
    bridged <- concordance(
      formulas[[index]],
      data = data,
      ymax = 4,
      timewt = "S/G",
      influence = 3
    )
    reference <- survival::concordance(
      reference_formulas[[index]],
      data = data,
      ymax = 4,
      timewt = "S/G",
      influence = 3
    )
    fields <- setdiff(names(reference), "call")
    expect_equal(
      unclass(bridged)[fields],
      unclass(reference)[fields],
      tolerance = 1e-12
    )
  }
})

test_that("zero-pair conditional variance preserves infinite ratios", {
  start <- c(2, 3, 0, 2, 4)
  stop <- c(5, 6, 1, 4, 5)
  status <- c(1, 0, 0, 0, 1)
  scores <- c(0, 0.5, 1, 1, 1)
  groups <- c(2, 1, 1, 1, 2)

  bridged <- concordancefit(
    Surv(start, stop, status),
    scores,
    strata = groups,
    ymax = 5,
    timewt = "S",
    influence = 2
  )
  reference <- survival::concordancefit(
    survival::Surv(start, stop, status),
    scores,
    strata = groups,
    ymax = 5,
    timewt = "S",
    influence = 2
  )

  expect_equal(unclass(bridged), unclass(reference), tolerance = 1e-12)
  expect_true(is.infinite(bridged$cvar))
})

test_that("collapsed single-score strata match reference behavior", {
  data <- data.frame(
    time = c(1, 2, 1, 2),
    status = c(1, 0, 1, 0),
    x = c(0.2, 0.6, 0.4, 0.8),
    group = c("a", "a", "b", "b")
  )
  dimension_error <- "'x' must be an array of at least two dimensions"

  for (keep in list(FALSE, 1)) {
    expect_error(
      concordancefit(
        Surv(data$time, data$status),
        data$x,
        strata = data$group,
        keepstrata = keep
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordancefit(
        survival::Surv(data$time, data$status),
        data$x,
        strata = data$group,
        keepstrata = keep
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      concordance(
        Surv(time, status) ~ x + strata(group),
        data = data,
        keepstrata = keep
      ),
      dimension_error,
      fixed = TRUE
    )
    expect_error(
      survival::concordance(
        survival::Surv(time, status) ~ x + strata(group),
        data = data,
        keepstrata = keep
      ),
      dimension_error,
      fixed = TRUE
    )
  }

  for (keep in list(TRUE, 2)) {
    bridged_fit <- concordancefit(
      Surv(data$time, data$status),
      data$x,
      strata = data$group,
      keepstrata = keep
    )
    reference_fit <- survival::concordancefit(
      survival::Surv(data$time, data$status),
      data$x,
      strata = data$group,
      keepstrata = keep
    )
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)

    bridged_formula <- concordance(
      Surv(time, status) ~ x + strata(group),
      data = data,
      keepstrata = keep
    )
    reference_formula <- survival::concordance(
      survival::Surv(time, status) ~ x + strata(group),
      data = data,
      keepstrata = keep
    )
    fields <- setdiff(names(reference_formula), "call")
    expect_equal(
      unclass(bridged_formula)[fields],
      unclass(reference_formula)[fields],
      tolerance = 1e-12
    )
  }

  large_data <- data.frame(
    group = rep(seq_len(11), each = 2),
    time = rep(c(1, 2), 11),
    status = rep(c(1, 0), 11),
    x = seq_len(22) / 22
  )
  for (time_weight in c("n", "I")) {
    bridged_fit <- concordancefit(
      Surv(large_data$time, large_data$status),
      large_data$x,
      strata = large_data$group,
      keepstrata = FALSE,
      timewt = time_weight
    )
    reference_fit <- survival::concordancefit(
      survival::Surv(large_data$time, large_data$status),
      large_data$x,
      strata = large_data$group,
      keepstrata = FALSE,
      timewt = time_weight
    )
    expect_equal(unclass(bridged_fit), unclass(reference_fit), tolerance = 1e-12)
  }
})

test_that("stratified concordance results match reference shapes and diagnostics", {
  right_data <- data.frame(
    y = c(1, 3, 2, 4, 4, 2),
    x = c(0.2, 0.9, 0.4, 0.7, 0.7, 0.1),
    w = c(1, 2, 1.5, 0.5, 3, 2.5),
    group = c("a", "a", "b", "b", "b", "a")
  )
  bridged_right <- concordance(
    y ~ x + strata(group),
    data = right_data,
    weights = w,
    influence = 3,
    ranks = TRUE
  )
  reference_right <- survival::concordance(
    y ~ x + strata(group),
    data = right_data,
    weights = w,
    influence = 3,
    ranks = TRUE
  )

  expect_equal(bridged_right$count, reference_right$count, tolerance = 1e-12)
  expect_equal(bridged_right$ranks, reference_right$ranks, tolerance = 1e-12)
  expect_equal(bridged_right$dfbeta, reference_right$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_right$influence, reference_right$influence, tolerance = 1e-12)
  expect_equal(vcov(bridged_right), vcov(reference_right), tolerance = 1e-12)

  multi_data <- transform(
    right_data,
    z = c(0.8, 0.3, 0.5, 0.2, 0.6, 0.4)
  )
  bridged_multi <- concordance(
    y ~ x + z,
    data = multi_data,
    influence = 3,
    ranks = TRUE
  )
  reference_multi <- survival::concordance(
    y ~ x + z,
    data = multi_data,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(bridged_multi$count, reference_multi$count, tolerance = 1e-12)
  expect_equal(bridged_multi$ranks, reference_multi$ranks, tolerance = 1e-12)
  expect_equal(bridged_multi$dfbeta, reference_multi$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_multi$influence, reference_multi$influence, tolerance = 1e-12)

  named_data <- transform(
    multi_data,
    time = y,
    status = c(1L, 0L, 1L, 1L, 0L, 1L),
    start = 0,
    keep = c(TRUE, TRUE, FALSE, TRUE, TRUE, TRUE)
  )
  row.names(named_data) <- paste0("case-", c(11, 22, 33, 44, 55, 66))
  bridged_named <- concordance(
    Surv(time, status) ~ x + z,
    data = named_data,
    ranks = TRUE
  )
  reference_named <- survival::concordance(
    survival::Surv(time, status) ~ x + z,
    data = named_data,
    ranks = TRUE
  )
  expect_equal(bridged_named$ranks, reference_named$ranks, tolerance = 1e-12)

  bridged_named_counting <- concordance(
    Surv(start, time, status) ~ x,
    data = named_data,
    ranks = TRUE
  )
  reference_named_counting <- survival::concordance(
    survival::Surv(start, time, status) ~ x,
    data = named_data,
    ranks = TRUE
  )
  expect_equal(
    bridged_named_counting$ranks,
    reference_named_counting$ranks,
    tolerance = 1e-12
  )

  bridged_named_subset <- concordance(
    Surv(time, status) ~ x,
    data = named_data,
    subset = keep,
    ranks = TRUE
  )
  reference_named_subset <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = named_data,
    subset = keep,
    ranks = TRUE
  )
  expect_equal(
    bridged_named_subset$ranks,
    reference_named_subset$ranks,
    tolerance = 1e-12
  )

  missing_named_data <- named_data
  missing_named_data$x[[6L]] <- NA_real_
  bridged_named_omit <- concordance(
    Surv(time, status) ~ x,
    data = missing_named_data,
    na.action = stats::na.omit,
    ranks = TRUE
  )
  reference_named_omit <- survival::concordance(
    survival::Surv(time, status) ~ x,
    data = missing_named_data,
    na.action = stats::na.omit,
    ranks = TRUE
  )
  expect_equal(
    bridged_named_omit$ranks,
    reference_named_omit$ranks,
    tolerance = 1e-12
  )

  named_response <- survival::Surv(named_data$time, named_data$status)
  rownames(named_response) <- row.names(named_data)
  reference_named_response <- survival::Surv(named_data$time, named_data$status)
  rownames(reference_named_response) <- row.names(named_data)
  bridged_named_fit <- concordancefit(named_response, named_data$x, ranks = TRUE)
  reference_named_fit <- survival::concordancefit(
    reference_named_response,
    named_data$x,
    ranks = TRUE
  )
  expect_equal(bridged_named_fit$ranks, reference_named_fit$ranks, tolerance = 1e-12)

  stratified_named_data <- data.frame(
    time = c(1, 2, 1, 2),
    status = c(1L, 0L, 1L, 0L),
    x = c(0.9, 0.1, 0.2, 0.8),
    group = c("a", "a", "b", "b"),
    row.names = paste0("stratum-case-", seq_len(4L))
  )
  bridged_stratified_named <- concordance(
    Surv(time, status) ~ x + strata(group),
    data = stratified_named_data,
    ranks = TRUE
  )
  reference_stratified_named <- survival::concordance(
    survival::Surv(time, status) ~ x + strata(group),
    data = stratified_named_data,
    ranks = TRUE
  )
  expect_equal(
    bridged_stratified_named$ranks,
    reference_stratified_named$ranks,
    tolerance = 1e-12
  )

  collapsed_error <- "'x' must be an array of at least two dimensions"
  expect_error(
    concordance(
      y ~ x + strata(group),
      data = right_data,
      weights = w,
      keepstrata = FALSE
    ),
    collapsed_error,
    fixed = TRUE
  )
  expect_error(
    survival::concordance(
      y ~ x + strata(group),
      data = right_data,
      weights = w,
      keepstrata = FALSE
    ),
    collapsed_error,
    fixed = TRUE
  )

  right_response <- Surv(right_data$y, rep(1L, nrow(right_data)))
  reference_right_response <- survival::Surv(
    right_data$y,
    rep(1L, nrow(right_data))
  )
  bridged_fit <- concordancefit(
    right_response,
    right_data$x,
    strata = right_data$group,
    weights = right_data$w,
    influence = 3,
    ranks = TRUE
  )
  reference_fit <- survival::concordancefit(
    reference_right_response,
    right_data$x,
    strata = right_data$group,
    weights = right_data$w,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(bridged_fit$count, reference_fit$count, tolerance = 1e-12)
  expect_equal(bridged_fit$ranks, reference_fit$ranks, tolerance = 1e-12)
  expect_equal(bridged_fit$dfbeta, reference_fit$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_fit$influence, reference_fit$influence, tolerance = 1e-12)

  score_matrix <- cbind(x = right_data$x, z = rev(right_data$x))
  bridged_multi_fit <- concordancefit(
    right_response,
    score_matrix,
    weights = right_data$w,
    influence = 3,
    ranks = TRUE
  )
  reference_multi_fit <- survival::concordancefit(
    reference_right_response,
    score_matrix,
    weights = right_data$w,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(bridged_multi_fit$count, reference_multi_fit$count, tolerance = 1e-12)
  expect_equal(bridged_multi_fit$ranks, reference_multi_fit$ranks, tolerance = 1e-12)
  expect_equal(bridged_multi_fit$dfbeta, reference_multi_fit$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_multi_fit$influence, reference_multi_fit$influence, tolerance = 1e-12)

  counting_data <- data.frame(
    start = c(0, 0, 1, 2.5, 0, 0, 0.5, 2),
    stop = c(2, 4, 3, 5, 1, 4, 3, 5),
    status = c(1, 0, 1, 1, 1, 1, 0, 1),
    score = c(0.8, 0.2, 0.5, 0.1, 0.3, 0.9, 0.4, 0.6),
    w = c(1, 2, 1.5, 0.5, 3, 1, 2.5, 2),
    group = c("a", "a", "a", "a", "b", "b", "b", "b")
  )
  bridged_counting <- concordance(
    Surv(start, stop, status) ~ score + strata(group),
    data = counting_data,
    weights = w,
    influence = 3
  )
  reference_counting <- survival::concordance(
    survival::Surv(start, stop, status) ~ score + strata(group),
    data = counting_data,
    weights = w,
    influence = 3
  )
  expect_equal(bridged_counting$count, reference_counting$count, tolerance = 1e-12)
  expect_equal(bridged_counting$dfbeta, reference_counting$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_counting$influence, reference_counting$influence, tolerance = 1e-12)
  expect_equal(vcov(bridged_counting), vcov(reference_counting), tolerance = 1e-12)
  expect_error(
    concordance(
      Surv(start, stop, status) ~ score + strata(group),
      data = counting_data,
      weights = w,
      ranks = TRUE
    ),
    "number of items to replace is not a multiple of replacement length"
  )

  tied_response <- Surv(
    rep(0, 5),
    c(1, 1, 2, 2, 3),
    c(1, 1, 1, 0, 1)
  )
  reference_tied_response <- survival::Surv(
    rep(0, 5),
    c(1, 1, 2, 2, 3),
    c(1, 1, 1, 0, 1)
  )
  tied_score <- c(0.9, 0.2, 0.7, 0.1, 0.8)
  tied_weight <- c(2, 1, 3, 0.5, 4)
  bridged_tied <- concordancefit(
    tied_response,
    tied_score,
    weights = tied_weight,
    timewt = "S",
    reverse = TRUE,
    influence = 3,
    ranks = TRUE
  )
  reference_tied <- survival::concordancefit(
    reference_tied_response,
    tied_score,
    weights = tied_weight,
    timewt = "S",
    reverse = TRUE,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(bridged_tied$count, reference_tied$count, tolerance = 1e-12)
  expect_equal(bridged_tied$concordance, reference_tied$concordance, tolerance = 1e-12)
  expect_equal(bridged_tied$var, reference_tied$var, tolerance = 1e-12)
  expect_equal(bridged_tied$cvar, reference_tied$cvar, tolerance = 1e-12)
  expect_equal(bridged_tied$dfbeta, reference_tied$dfbeta, tolerance = 1e-12)
  expect_equal(bridged_tied$influence, reference_tied$influence, tolerance = 1e-12)
  expect_equal(bridged_tied$ranks, reference_tied$ranks, tolerance = 1e-12)
})

test_that("interval-censored curves match weighted Turnbull reference fits", {
  turnbull_data <- data.frame(
    time = rep(1:4, each = 3L),
    status = rep(c(1L, 0L, 2L), 4L),
    weight = c(12, 3, 2, 6, 2, 4, 2, 0, 2, 3, 3, 5)
  )
  formula <- Surv(time, time, status, type = "interval") ~ 1
  reference_formula <- survival::Surv(time, time, status, type = "interval") ~ 1

  for (robust in c(TRUE, FALSE)) {
    bridged <- as.list(survfit(
      formula,
      data = turnbull_data,
      weights = weight,
      robust = robust
    ))
    reference <- survival::survfit(
      reference_formula,
      data = turnbull_data,
      weights = weight,
      robust = robust
    )

    expect_equal(bridged$time, reference$time, tolerance = 1e-12)
    expect_equal(bridged$n.risk, reference$n.risk, tolerance = 1e-12)
    expect_equal(bridged$n.event, reference$n.event, tolerance = 1e-12)
    expect_equal(bridged$n.censor, reference$n.censor, tolerance = 1e-12)
    expect_equal(bridged$surv, reference$surv, tolerance = 1e-12)
    expect_equal(bridged$std.err, reference$std.err, tolerance = 1e-12)
    expect_equal(bridged$lower, reference$lower, tolerance = 2e-9)
    expect_equal(bridged$upper, reference$upper, tolerance = 2e-9)
  }
})

test_that("concordance formulas accept numeric and orderable outcomes", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    y = c(1, 3, 2, 4, 4, 2),
    x = c(0.2, 0.9, 0.4, 0.7, 0.7, 0.1),
    z = c(1, 0.2, 0.5, 0.8, 0.3, 0.6),
    w = c(1, 2, 1.5, 0.5, 3, 2.5),
    cluster = c(1, 1, 2, 2, 3, 3)
  )
  data$binary <- factor(data$y >= 3)
  data$ordinal <- ordered(
    c("low", "high", "mid", "high", "high", "mid"),
    levels = c("low", "mid", "high")
  )
  data$group <- factor(c("a", "a", "b", "b", "b", "a"))

  expect_concordance_equal <- function(formula, ...) {
    args <- c(list(formula, data = data), list(...))
    bridged <- do.call(concordance, args)
    reference <- do.call(survival::concordance, args)
    bridged_frame <- as.data.frame(bridged)
    strict_concordant <- bridged_frame$concordant - 0.5 * bridged_frame$tied.x
    bridged_count <- cbind(
      concordant = strict_concordant,
      discordant = bridged_frame$comparable - strict_concordant - bridged_frame$tied.x,
      tied.x = bridged_frame$tied.x,
      tied.y = bridged_frame$tied.y,
      tied.xy = bridged_frame$tied.xy
    )
    reference_count <- if (is.matrix(reference$count)) {
      reference$count
    } else {
      matrix(as.numeric(reference$count), nrow = 1L)
    }

    expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
    expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
    expect_equal(
      unname(bridged_count),
      unname(reference_count),
      tolerance = 1e-12
    )
    invisible(bridged)
  }

  numeric_fit <- expect_concordance_equal(
    y ~ x,
    weights = data$w,
    influence = 3,
    ranks = TRUE
  )
  expect_equal(coef(numeric_fit), 0.753246753246753, tolerance = 1e-12)
  expect_equal(
    as.numeric(vcov(numeric_fit)),
    0.0112321292487896,
    tolerance = 1e-12
  )

  expect_diagnostics_equal <- function(bridged, reference) {
    bridged_dfbeta <- survivalr:::.as_numeric_vector(
      survivalr:::.result_field(bridged, "dfbeta")
    )
    bridged_influence <- survivalr:::.as_numeric_matrix(
      survivalr:::.result_field(bridged, "influence")
    )
    expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
    expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
    expect_equal(bridged_dfbeta, as.numeric(reference$dfbeta), tolerance = 1e-12)
    expect_equal(
      unname(bridged_influence),
      unname(reference$influence),
      tolerance = 1e-12
    )
  }

  stratified_formula <- y ~ x + strata(group)
  expect_diagnostics_equal(
    concordance(
      stratified_formula,
      data = data,
      weights = data$w,
      influence = 3
    ),
    survival::concordance(
      stratified_formula,
      data = data,
      weights = data$w,
      influence = 3
    )
  )

  counting_data <- data.frame(
    start = c(0, 0, 1, 2.5, 0, 0, 0.5, 2),
    stop = c(2, 4, 3, 5, 1, 4, 3, 5),
    status = c(1, 0, 1, 1, 1, 1, 0, 1),
    score = c(0.8, 0.2, 0.5, 0.1, 0.3, 0.9, 0.4, 0.6),
    w = c(1, 2, 1.5, 0.5, 3, 1, 2.5, 2),
    group = factor(c("a", "a", "a", "a", "b", "b", "b", "b"))
  )
  counting_formula <- Surv(start, stop, status) ~ score + strata(group)
  expect_diagnostics_equal(
    concordance(
      counting_formula,
      data = counting_data,
      weights = counting_data$w,
      influence = 3
    ),
    survival::concordance(
      counting_formula,
      data = counting_data,
      weights = counting_data$w,
      influence = 3
    )
  )

  expect_concordance_equal(y ~ x + z, weights = data$w)
  expect_concordance_equal(y ~ x + z, weights = data$w, cluster = data$cluster)
  expect_concordance_equal(I(y >= 3) ~ x)
  expect_concordance_equal(binary ~ x)
  expect_concordance_equal(ordinal ~ x)
  expect_concordance_equal(log(y + 1) ~ x)

  forced_rank_weights <- concordance(y ~ x, data = data, timewt = "S")
  ordinary_rank_weights <- concordance(y ~ x, data = data, timewt = "n")
  expect_equal(coef(forced_rank_weights), coef(ordinary_rank_weights))
  expect_equal(vcov(forced_rank_weights), vcov(ordinary_rank_weights))

  bridged_environment_fit <- local({
    y <- data$y
    x <- data$x
    concordance(y ~ x)
  })
  reference_environment_fit <- local({
    y <- data$y
    x <- data$x
    survival::concordance(y ~ x)
  })
  expect_equal(
    coef(bridged_environment_fit),
    coef(reference_environment_fit),
    tolerance = 1e-12
  )
  expect_equal(
    vcov(bridged_environment_fit),
    vcov(reference_environment_fit),
    tolerance = 1e-12
  )

  data$unordered <- factor(c("a", "b", "c", "a", "b", "c"))
  expect_error(
    concordance(unordered ~ x, data = data),
    "orderable factor"
  )
})

test_that("tmerge matches native interval, metadata, and class semantics", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  expect_tmerge_equal <- function(actual, expected) {
    attr(actual, "call") <- NULL
    attr(expected, "call") <- NULL
    expect_equal(actual, expected)
  }

  base <- data.frame(id = 1:2, age = c(40, 50))
  spans <- data.frame(id = 1:2, stop = c(10, 9))
  updates <- data.frame(
    id = c(1, 1, 1, 2, 2),
    time = c(2, 5, 8, 3, 7),
    value = c(10, 20, 30, 5, 9),
    increment = c(1, 2, 3, 4, 5),
    status = c(0, 1, 1, 1, 0)
  )
  initial <- tmerge(base, spans, id = id, tstop = stop)
  reference_initial <- survival::tmerge(base, spans, id = id, tstop = stop)
  mixed <- tmerge(
    initial,
    updates,
    id = id,
    x = tdc(time, value, init = 0),
    count = cumtdc(time, increment, init = 0),
    endpoint = event(time, status),
    cumulative_endpoint = cumevent(time, increment)
  )
  reference_mixed <- survival::tmerge(
    reference_initial,
    updates,
    id = id,
    x = tdc(time, value, init = 0),
    count = cumtdc(time, increment, init = 0),
    endpoint = event(time, status),
    cumulative_endpoint = cumevent(time, increment)
  )
  expect_tmerge_equal(mixed, reference_mixed)
  expect_s3_class(mixed, "tmerge")
  expect_identical(
    colnames(attr(mixed, "tcount")),
    c("early", "late", "gap", "within", "boundary", "leading", "trailing", "tied", "missid")
  )

  gap_base <- data.frame(id = 1, marker = factor("a", levels = c("a", "b")))
  gap_spans <- data.frame(id = c(1, 1), start = c(0, 5), stop = c(3, 10))
  gap_updates <- data.frame(
    id = c(1, 1, 1, 1, 1, 1, 1, 1, 2),
    time = c(-1, 0, 2, 3, 4, 5, 7, 10, 2),
    value = seq_len(9L)
  )
  gap_initial <- tmerge(gap_base, gap_spans, id = id, tstart = start, tstop = stop)
  reference_gap_initial <- survival::tmerge(
    gap_base,
    gap_spans,
    id = id,
    tstart = start,
    tstop = stop
  )
  gap_result <- tmerge(gap_initial, gap_updates, id = id, endpoint = event(time, value))
  reference_gap <- survival::tmerge(
    reference_gap_initial,
    gap_updates,
    id = id,
    endpoint = event(time, value)
  )
  expect_tmerge_equal(gap_result, reference_gap)

  typed_updates <- data.frame(
    id = c(1, 1),
    time = c(2, 4),
    state = factor(c("event", "other"), levels = c("none", "event", "other")),
    date = as.Date(c("2020-01-02", "2020-01-04"))
  )
  typed_initial <- tmerge(gap_base, data.frame(id = 1, stop = 10), id = id, tstop = stop)
  reference_typed_initial <- survival::tmerge(
    gap_base,
    data.frame(id = 1, stop = 10),
    id = id,
    tstop = stop
  )
  typed_result <- tmerge(
    typed_initial,
    typed_updates,
    id = id,
    state = event(time, state),
    state_tdc = tdc(time, state, init = "none"),
    date_tdc = tdc(time, date)
  )
  reference_typed <- survival::tmerge(
    reference_typed_initial,
    typed_updates,
    id = id,
    state = event(time, state),
    state_tdc = tdc(time, state, init = "none"),
    date_tdc = tdc(time, date)
  )
  expect_tmerge_equal(typed_result, reference_typed)
  expect_s3_class(typed_result$state, "factor")
  expect_s3_class(typed_result$date_tdc, "Date")

  missing_updates <- data.frame(id = c(1, 1), time = c(2, 4), value = c(NA, 5))
  missing_result <- tmerge(
    typed_initial,
    missing_updates,
    id = id,
    total = cumtdc(time, value, init = 1),
    endpoint = event(time, value),
    options = list(na.rm = FALSE)
  )
  reference_missing <- survival::tmerge(
    reference_typed_initial,
    missing_updates,
    id = id,
    total = cumtdc(time, value, init = 1),
    endpoint = event(time, value),
    options = list(na.rm = FALSE)
  )
  expect_tmerge_equal(missing_result, reference_missing)
})

test_that("Cox score inference matches native fits at mixed event and censor ties", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = c(1, 1, 2, 2, 3, 4),
    status = c(1, 1, 1, 0, 1, 0),
    x = c(0, 1, 0.5, 1.5, 2, -0.5),
    id = seq_len(6L)
  )

  for (method in c("breslow", "efron")) {
    bridged <- coxph(
      Surv(time, status) ~ x,
      data = data,
      cluster = data$id,
      ties = method,
      max_iter = 50,
      eps = 1e-09,
      toler = 1e-10
    )
    reference <- survival::coxph(
      survival::Surv(time, status) ~ x,
      data = data,
      cluster = data$id,
      ties = method,
      control = survival::coxph.control(
        iter.max = 50,
        eps = 1e-09,
        toler.chol = 1e-10
      )
    )

    expect_equal(
      unname(residuals(bridged, type = "score")),
      unname(stats::residuals(reference, type = "score")),
      tolerance = 1e-10
    )
    expect_equal(unname(vcov(bridged)), unname(vcov(reference)), tolerance = 1e-10)
  }
})

test_that("model summaries match native Cox and survreg coefficient tables", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(91)
  n <- 80L
  x <- seq(-1.2, 1.4, length.out = n) + rep(c(-0.08, 0.04, 0.11), length.out = n)
  z <- rep(c(0, 1), length.out = n)
  site <- factor(
    rep(c("north", "south"), each = n / 2L),
    levels = c("north", "south")
  )
  id <- rep(seq_len(n / 2L), each = 2L)
  linear_predictor <- -0.3 * x + 0.2 * z
  event_time <- stats::rexp(n, rate = exp(linear_predictor) / 8)
  censor_time <- stats::rexp(n, rate = 1 / 15)
  data <- data.frame(
    time = pmax(pmin(event_time, censor_time), 0.01),
    status = as.integer(event_time <= censor_time),
    x = x,
    z = z,
    site = site,
    site_num = as.integer(site),
    id = id
  )

  cox_cases <- list(
    plain = list(
      bridged = coxph(
        Surv(time, status) ~ x + z,
        data = data,
        max_iter = 150,
        eps = 1e-09,
        toler = 1e-10
      ),
      reference = survival::coxph(
        survival::Surv(time, status) ~ x + z,
        data = data,
        control = survival::coxph.control(
          iter.max = 150,
          eps = 1e-09,
          toler.chol = 1e-10
        )
      ),
      columns = c("coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"),
      robust = FALSE
    ),
    clustered = list(
      bridged = coxph(
        Surv(time, status) ~ x + z + cluster(id),
        data = data,
        max_iter = 150,
        eps = 1e-09,
        toler = 1e-10
      ),
      reference = survival::coxph(
        survival::Surv(time, status) ~ x + z + cluster(id),
        data = data,
        control = survival::coxph.control(
          iter.max = 150,
          eps = 1e-09,
          toler.chol = 1e-10
        )
      ),
      columns = c(
        "coef", "exp(coef)", "se(coef)", "robust se", "z", "Pr(>|z|)"
      ),
      robust = TRUE
    )
  )

  for (case in cox_cases) {
    bridged <- summary(case$bridged)
    reference <- summary(case$reference)
    expect_identical(rownames(bridged$coefficients), c("x", "z"))
    expect_identical(rownames(reference$coefficients), c("x", "z"))
    expect_identical(colnames(bridged$coefficients), case$columns)
    expect_identical(colnames(reference$coefficients), case$columns)
    expect_equal(
      unname(bridged$coefficients),
      unname(reference$coefficients),
      tolerance = 2e-04
    )
    expect_identical(bridged$used.robust, case$robust)
    expect_identical(reference$used.robust, case$robust)
    expect_equal(bridged$loglik, reference$loglik, tolerance = 1e-08)
    expect_identical(bridged$nevent, reference$nevent)
    expect_identical(colnames(bridged$conf.int), colnames(reference$conf.int))
    expect_equal(bridged$conf.int, reference$conf.int, tolerance = 1e-06)
    for (field in c("logtest", "sctest", "waldtest")) {
      expect_named(bridged[[field]], c("test", "df", "pvalue"))
      expect_equal(bridged[[field]], reference[[field]], tolerance = 1e-06)
    }
    expect_named(bridged$rsq, c("rsq", "maxrsq"))
    expect_equal(bridged$rsq, reference$rsq, tolerance = 1e-08)

    printed <- paste(capture.output(print(bridged)), collapse = "\n")
    expect_true(grepl("exp(coef)", printed, fixed = TRUE))
    expect_identical(grepl("robust se", printed, fixed = TRUE), case$robust)
    expect_true(grepl("number of events", printed, fixed = TRUE))
    expect_true(grepl("Likelihood ratio test=", printed, fixed = TRUE))
    expect_true(grepl("Wald test            =", printed, fixed = TRUE))
    expect_true(grepl("Score (logrank) test =", printed, fixed = TRUE))
    expect_identical(
      grepl("assume independence", printed, fixed = TRUE),
      case$robust
    )
  }

  for (case in cox_cases) {
    bridged <- summary(case$bridged, conf.int = 0.9, scale = 2)
    reference <- summary(case$reference, conf.int = 0.9, scale = 2)
    expect_identical(
      colnames(bridged$conf.int),
      c("exp(coef)", "exp(-coef)", "lower .90", "upper .90")
    )
    expect_equal(bridged$coefficients, reference$coefficients, tolerance = 1e-06)
    expect_equal(bridged$conf.int, reference$conf.int, tolerance = 1e-06)
    expect_equal(bridged$logtest, reference$logtest, tolerance = 1e-06)
    expect_equal(bridged$sctest, reference$sctest, tolerance = 1e-06)
    expect_equal(bridged$waldtest, reference$waldtest, tolerance = 1e-06)
    expect_equal(bridged$rsq, reference$rsq, tolerance = 1e-08)

    without_confidence <- summary(case$bridged, conf.int = FALSE)
    reference_without_confidence <- summary(case$reference, conf.int = FALSE)
    expect_null(without_confidence$conf.int)
    expect_null(reference_without_confidence$conf.int)
    printed <- paste(capture.output(print(without_confidence)), collapse = "\n")
    expect_false(grepl("lower .", printed, fixed = TRUE))
  }

  for (formula in list(
    survival::Surv(time, status) ~ 1,
    survival::Surv(time, status) ~ cluster(id)
  )) {
    bridged_formula <- stats::as.formula(
      sub("survival::Surv", "Surv", deparse1(formula), fixed = TRUE)
    )
    bridged <- summary(
      coxph(bridged_formula, data = data),
      conf.int = 0.9,
      scale = 2
    )
    reference <- summary(
      survival::coxph(formula, data = data),
      conf.int = 0.9,
      scale = 2
    )
    expect_null(bridged$coefficients)
    expect_null(reference$coefficients)
    expect_null(bridged$used.robust)
    expect_null(reference$used.robust)
  }

  survreg_control <- survival::survreg.control(
    maxiter = 150,
    rel.tolerance = 1e-10
  )
  survreg_cases <- list(
    estimated = list(
      bridged = survreg(
        Surv(time, status) ~ x + z,
        data = data,
        dist = "weibull",
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z,
        data = data,
        dist = "weibull",
        control = survreg_control
      ),
      rows = c("(Intercept)", "x", "z", "Log(scale)"),
      columns = c("Value", "Std. Error", "z", "p"),
      scale_names = NULL,
      robust = FALSE
    ),
    fixed = list(
      bridged = survreg(
        Surv(time, status) ~ x + z,
        data = data,
        dist = "weibull",
        scale = 0.8,
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z,
        data = data,
        dist = "weibull",
        scale = 0.8,
        control = survreg_control
      ),
      rows = c("(Intercept)", "x", "z"),
      columns = c("Value", "Std. Error", "z", "p"),
      scale_names = NULL,
      robust = FALSE
    ),
    stratified = list(
      bridged = survreg(
        Surv(time, status) ~ x + z + strata(site),
        data = data,
        dist = "weibull",
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z + strata(site),
        data = data,
        dist = "weibull",
        control = survreg_control
      ),
      rows = c("(Intercept)", "x", "z", "north", "south"),
      columns = c("Value", "Std. Error", "z", "p"),
      scale_names = c("north", "south"),
      robust = FALSE
    ),
    numeric_strata = list(
      bridged = survreg(
        Surv(time, status) ~ x + z + strata(site_num),
        data = data,
        dist = "weibull",
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z + strata(site_num),
        data = data,
        dist = "weibull",
        control = survreg_control
      ),
      rows = c("(Intercept)", "x", "z", "site_num=1", "site_num=2"),
      columns = c("Value", "Std. Error", "z", "p"),
      scale_names = c("site_num=1", "site_num=2"),
      robust = FALSE
    ),
    multi_strata = list(
      bridged = survreg(
        Surv(time, status) ~ x + z + strata(site_num, site),
        data = data,
        dist = "weibull",
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z + strata(site_num, site),
        data = data,
        dist = "weibull",
        control = survreg_control
      ),
      rows = c(
        "(Intercept)", "x", "z",
        "site_num=1, site=north", "site_num=2, site=south"
      ),
      columns = c("Value", "Std. Error", "z", "p"),
      scale_names = c("site_num=1, site=north", "site_num=2, site=south"),
      robust = FALSE
    ),
    clustered = list(
      bridged = survreg(
        Surv(time, status) ~ x + z + cluster(id),
        data = data,
        dist = "weibull",
        max_iter = 150,
        eps = 1e-10
      ),
      reference = survival::survreg(
        survival::Surv(time, status) ~ x + z + cluster(id),
        data = data,
        dist = "weibull",
        control = survreg_control
      ),
      rows = c("(Intercept)", "x", "z", "Log(scale)"),
      columns = c("Value", "Std. Err", "(Naive SE)", "z", "p"),
      scale_names = NULL,
      robust = TRUE
    )
  )

  location_names <- c("(Intercept)", "x", "z")
  for (case in survreg_cases) {
    bridged <- summary(case$bridged)
    reference <- summary(case$reference)
    expect_true(is.numeric(bridged$coefficients))
    expect_null(dim(bridged$coefficients))
    expect_identical(names(bridged$coefficients), location_names)
    expect_identical(names(reference$coefficients), location_names)
    expect_equal(
      unname(bridged$coefficients),
      unname(reference$coefficients),
      tolerance = 2e-04
    )

    expect_identical(rownames(bridged$table), case$rows)
    expect_identical(rownames(reference$table), case$rows)
    expect_identical(colnames(bridged$table), case$columns)
    expect_identical(colnames(reference$table), case$columns)
    expect_equal(
      unname(bridged$table),
      unname(reference$table),
      tolerance = 5e-04
    )
    expect_identical(bridged$robust, case$robust)
    expect_identical(reference$robust, case$robust)
    expect_identical(names(bridged$scale), case$scale_names)
    expect_identical(names(reference$scale), case$scale_names)
    expect_equal(unname(bridged$scale), unname(reference$scale), tolerance = 5e-04)

    printed <- paste(capture.output(print(bridged)), collapse = "\n")
    expect_true(grepl("Value", printed, fixed = TRUE))
    if ("Log(scale)" %in% case$rows) {
      expect_true(grepl("Log(scale)", printed, fixed = TRUE))
    }
    expect_identical(grepl("(Naive SE)", printed, fixed = TRUE), case$robust)
  }
})

test_that("model term metadata matches native Cox formula outputs", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(1729)
  n <- 60L
  data <- data.frame(
    time = stats::rexp(n) + 0.1,
    status = stats::rbinom(n, 1L, 0.75),
    g = factor(rep(c("a", "b", "c"), length.out = n), levels = c("a", "b", "c")),
    x = stats::rnorm(n)
  )
  cases <- list(
    bare = list(
      rhs = "g + x",
      coefficients = c("gb", "gc", "x"),
      terms = c("g", "x"),
      assign = c(1L, 1L, 2L)
    ),
    factor = list(
      rhs = "factor(g) + x",
      coefficients = c("factor(g)b", "factor(g)c", "x"),
      terms = c("factor(g)", "x"),
      assign = c(1L, 1L, 2L)
    ),
    as_factor = list(
      rhs = "as.factor(g) + x",
      coefficients = c("as.factor(g)b", "as.factor(g)c", "x"),
      terms = c("as.factor(g)", "x"),
      assign = c(1L, 1L, 2L)
    ),
    interaction = list(
      rhs = "g * x",
      coefficients = c("gb", "gc", "x", "gb:x", "gc:x"),
      terms = c("g", "x", "g:x"),
      assign = c(1L, 1L, 2L, 3L, 3L)
    )
  )

  for (case in cases) {
    bridged_formula <- stats::as.formula(paste("Surv(time, status) ~", case$rhs))
    reference_formula <- stats::as.formula(
      paste("survival::Surv(time, status) ~", case$rhs)
    )
    bridged <- coxph(
      bridged_formula,
      data = data,
      max_iter = 50,
      eps = 1e-09,
      toler = 1e-10
    )
    reference <- survival::coxph(
      reference_formula,
      data = data,
      x = TRUE,
      y = TRUE,
      control = survival::coxph.control(iter.max = 50, eps = 1e-09, toler.chol = 1e-10)
    )

    expect_equal(names(coef(bridged)), case$coefficients)
    expect_equal(names(coef(reference)), case$coefficients)
    expect_equal(rownames(summary(bridged)$coefficients), case$coefficients)

    bridged_matrix <- model.matrix(bridged)
    reference_matrix <- stats::model.matrix(reference)
    expect_equal(dim(bridged_matrix), dim(reference_matrix))
    expect_equal(colnames(bridged_matrix), case$coefficients)
    expect_equal(colnames(reference_matrix), case$coefficients)
    expect_equal(attr(bridged_matrix, "assign"), case$assign)
    expect_equal(attr(reference_matrix, "assign"), case$assign)
    expect_equal(as.numeric(bridged_matrix), as.numeric(reference_matrix), tolerance = 1e-12)
    expect_equal(
      attrassign(bridged_matrix, terms(bridged)),
      survival::attrassign(reference_matrix, terms(reference))
    )

    bridged_terms <- predict(bridged, type = "terms")
    reference_terms <- stats::predict(reference, type = "terms")
    expect_equal(dim(bridged_terms), dim(reference_terms))
    expect_equal(colnames(bridged_terms), case$terms)
    expect_equal(colnames(reference_terms), case$terms)
    expect_equal(
      attr(bridged_terms, "constant"),
      attr(reference_terms, "constant"),
      tolerance = 2e-04
    )

    selected_terms <- rev(case$terms)
    bridged_selected <- predict(bridged, type = "terms", terms = selected_terms)
    reference_selected <- stats::predict(
      reference,
      type = "terms",
      terms = selected_terms
    )
    expect_equal(dim(bridged_selected), dim(reference_selected))
    expect_equal(colnames(bridged_selected), selected_terms)
    expect_equal(colnames(reference_selected), selected_terms)

    bridged_with_se <- predict(bridged, type = "terms", se.fit = TRUE)
    reference_with_se <- stats::predict(reference, type = "terms", se.fit = TRUE)
    expect_equal(colnames(bridged_with_se$fit), case$terms)
    expect_equal(colnames(bridged_with_se$se.fit), case$terms)
    expect_equal(colnames(reference_with_se$fit), case$terms)
    expect_equal(colnames(reference_with_se$se.fit), case$terms)
    expect_equal(
      attr(bridged_with_se$fit, "constant"),
      attr(reference_with_se$fit, "constant"),
      tolerance = 2e-04
    )
    expect_null(attr(bridged_with_se$se.fit, "constant"))

    bridged_partial <- residuals(bridged, type = "partial")
    reference_partial <- stats::residuals(reference, type = "partial")
    expect_equal(dim(bridged_partial), dim(reference_partial))
    expect_equal(colnames(bridged_partial), case$terms)
    expect_equal(colnames(reference_partial), case$terms)
    expected_partial <- sweep(
      unclass(bridged_terms),
      1L,
      as.numeric(residuals(bridged, type = "martingale")),
      "+"
    )
    expect_equal(
      as.numeric(bridged_partial),
      as.numeric(expected_partial),
      tolerance = 1e-10
    )

    bridged_score <- residuals(bridged, type = "score")
    reference_score <- stats::residuals(reference, type = "score")
    expect_equal(colnames(bridged_score), colnames(reference_score))

    bridged_zph_terms <- as.data.frame(cox.zph(bridged, transform = "rank", terms = TRUE))
    reference_zph_terms <- survival::cox.zph(reference, transform = "rank", terms = TRUE)
    expect_equal(bridged_zph_terms$name, rownames(reference_zph_terms$table))
    bridged_zph_columns <- as.data.frame(
      cox.zph(bridged, transform = "rank", terms = FALSE)
    )
    reference_zph_columns <- survival::cox.zph(
      reference,
      transform = "rank",
      terms = FALSE
    )
    expect_equal(bridged_zph_columns$name, rownames(reference_zph_columns$table))
  }
})

test_that("interaction contrast expansion matches native Cox and survreg fits", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(2048)
  n <- 90L
  g <- factor(rep(c("a", "b", "c"), length.out = n), levels = c("a", "b", "c"))
  h <- factor(rep(c("u", "v"), length.out = n), levels = c("u", "v"))
  x <- seq(-1.7, 1.9, length.out = n) +
    rep(c(-0.13, 0.07, 0.19, -0.05, 0.11), length.out = n)
  eta <- 0.25 * x + c(a = -0.2, b = 0.15, c = 0.35)[g] +
    c(u = -0.1, v = 0.1)[h]
  event_time <- stats::rexp(n, rate = exp(eta) / 9)
  censor_time <- stats::rexp(n, rate = 1 / 14)
  data <- data.frame(
    time = pmax(pmin(event_time, censor_time), 0.01),
    status = as.integer(event_time <= censor_time),
    x = x,
    g = g,
    h = h
  )

  cox_cases <- list(
    list(
      rhs = "g:x",
      columns = c("ga:x", "gb:x", "gc:x"),
      assign = c(1L, 1L, 1L),
      terms = "g:x"
    ),
    list(
      rhs = "x + g:x",
      columns = c("x", "x:gb", "x:gc"),
      assign = c(1L, 2L, 2L),
      terms = c("x", "x:g")
    ),
    list(
      rhs = "g + g:x",
      columns = c("gb", "gc", "ga:x", "gb:x", "gc:x"),
      assign = c(1L, 1L, 2L, 2L, 2L),
      terms = c("g", "g:x")
    ),
    list(
      rhs = "g:x + g + x",
      columns = c("gb", "gc", "x", "gb:x", "gc:x"),
      assign = c(1L, 1L, 2L, 3L, 3L),
      terms = c("g", "x", "g:x")
    ),
    list(
      rhs = "g*h",
      columns = c("gb", "gc", "hv", "gb:hv", "gc:hv"),
      assign = c(1L, 1L, 2L, 3L, 3L),
      terms = c("g", "h", "g:h")
    )
  )

  expect_design_parity <- function(bridged, reference, case, compare_fit = TRUE,
                                   compare_term_se = FALSE) {
    expect_identical(names(coef(bridged)), case$columns)
    expect_identical(names(coef(reference)), case$columns)
    if (compare_fit) {
      expect_equal(
        unname(coef(bridged)),
        unname(coef(reference)),
        tolerance = 2e-04
      )
    }

    bridged_matrix <- model.matrix(bridged)
    reference_matrix <- stats::model.matrix(reference)
    expected_dim <- c(n, length(case$columns))
    expect_identical(dim(bridged_matrix), expected_dim)
    expect_identical(dim(reference_matrix), expected_dim)
    expect_identical(colnames(bridged_matrix), case$columns)
    expect_identical(colnames(reference_matrix), case$columns)
    expect_identical(attr(bridged_matrix, "assign"), case$assign)
    expect_identical(attr(reference_matrix, "assign"), case$assign)
    expect_identical(as.numeric(bridged_matrix), as.numeric(reference_matrix))

    expect_identical(attr(terms(bridged), "term.labels"), case$terms)
    expect_identical(attr(terms(reference), "term.labels"), case$terms)
    expect_identical(labels(bridged), case$terms)
    if (!is.null(utils::getS3method("labels", "coxph", optional = TRUE))) {
      expect_identical(labels(reference), case$terms)
    }

    bridged_terms <- predict(bridged, type = "terms")
    reference_terms <- stats::predict(reference, type = "terms")
    expected_terms_dim <- c(n, length(case$terms))
    expect_identical(dim(bridged_terms), expected_terms_dim)
    expect_identical(dim(reference_terms), expected_terms_dim)
    expect_identical(colnames(bridged_terms), case$terms)
    expect_identical(colnames(reference_terms), case$terms)
    if (compare_fit) {
      expect_equal(
        as.numeric(bridged_terms),
        as.numeric(reference_terms),
        tolerance = 2e-04
      )
    }
    if (compare_term_se) {
      bridged_with_se <- predict(bridged, type = "terms", se.fit = TRUE)
      reference_with_se <- stats::predict(reference, type = "terms", se.fit = TRUE)
      expect_identical(dim(bridged_with_se$fit), dim(reference_with_se$fit))
      expect_identical(dim(bridged_with_se$se.fit), dim(reference_with_se$se.fit))
      expect_identical(colnames(bridged_with_se$fit), case$terms)
      expect_identical(colnames(bridged_with_se$se.fit), case$terms)
      expect_equal(
        as.numeric(bridged_with_se$fit),
        as.numeric(reference_with_se$fit),
        tolerance = 2e-04
      )
      expect_equal(
        as.numeric(bridged_with_se$se.fit),
        as.numeric(reference_with_se$se.fit),
        tolerance = 2e-04
      )
    }
  }

  for (case in cox_cases) {
    bridged <- coxph(
      stats::as.formula(paste("Surv(time, status) ~", case$rhs)),
      data = data,
      max_iter = 150,
      eps = 1e-09,
      toler = 1e-10
    )
    reference <- survival::coxph(
      stats::as.formula(paste("survival::Surv(time, status) ~", case$rhs)),
      data = data,
      x = TRUE,
      y = TRUE,
      control = survival::coxph.control(
        iter.max = 150,
        eps = 1e-09,
        toler.chol = 1e-10
      )
    )
    expect_design_parity(bridged, reference, case)
  }

  survreg_cases <- lapply(cox_cases, function(case) {
    case$columns <- c("(Intercept)", case$columns)
    case$assign <- c(0L, case$assign)
    case
  })
  survreg_cases <- c(
    survreg_cases,
    list(
      list(
        rhs = "g:x - 1",
        columns = c("ga:x", "gb:x", "gc:x"),
        assign = c(1L, 1L, 1L),
        terms = "g:x"
      ),
      list(
        rhs = "x + g:x - 1",
        columns = c("x", "x:ga", "x:gb", "x:gc"),
        assign = c(1L, 2L, 2L, 2L),
        terms = c("x", "x:g"),
        singular = TRUE
      ),
      list(
        rhs = "g + g:x - 1",
        columns = c("ga", "gb", "gc", "ga:x", "gb:x", "gc:x"),
        assign = c(1L, 1L, 1L, 2L, 2L, 2L),
        terms = c("g", "g:x")
      ),
      list(
        rhs = "g:x + g + x - 1",
        columns = c("ga", "gb", "gc", "x", "gb:x", "gc:x"),
        assign = c(1L, 1L, 1L, 2L, 3L, 3L),
        terms = c("g", "x", "g:x")
      ),
      list(
        rhs = "g*h - 1",
        columns = c("ga", "gb", "gc", "hv", "gb:hv", "gc:hv"),
        assign = c(1L, 1L, 1L, 2L, 3L, 3L),
        terms = c("g", "h", "g:h")
      ),
      list(
        rhs = "g:h - 1",
        columns = c("ga:hu", "gb:hu", "gc:hu", "ga:hv", "gb:hv", "gc:hv"),
        assign = rep(1L, 6L),
        terms = "g:h"
      )
    )
  )

  for (case in survreg_cases) {
    bridged <- survreg(
      stats::as.formula(paste("Surv(time, status) ~", case$rhs)),
      data = data,
      dist = "weibull",
      max_iter = 150,
      eps = 1e-10
    )
    reference <- survival::survreg(
      stats::as.formula(paste("survival::Surv(time, status) ~", case$rhs)),
      data = data,
      dist = "weibull",
      x = TRUE,
      control = survival::survreg.control(
        maxiter = 150,
        rel.tolerance = 1e-10
      )
    )
    expect_design_parity(
      bridged,
      reference,
      case,
      compare_fit = !isTRUE(case$singular),
      compare_term_se = !isTRUE(case$singular)
    )
  }
})

test_that("numeric formula calls match R designs and prediction rebuilding", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5109)
  n <- 120L
  x <- stats::runif(n, -2, 5.5)
  z <- stats::runif(n, 0.2, 5)
  exposure <- stats::runif(n, -0.4, 0.7)
  eta <- 0.18 * pmin(x, 3) -
    0.24 * log(pmax(z, 1), base = 10) +
    0.05 * round(abs(x - z), digits = 2) +
    log(pmax(exposure + 1, 0.5))
  event_time <- stats::rexp(n, rate = exp(eta) / 8)
  censor_time <- stats::rexp(n, rate = 1 / 11)
  data <- data.frame(
    time = pmax(pmin(event_time, censor_time), 0.01),
    status = as.integer(event_time <= censor_time),
    x = x,
    z = z,
    exposure = exposure
  )
  bridged <- coxph(
    Surv(time, status) ~ pmin(x, 3) +
      log(pmax(z, 1), base = 10) +
      round(abs(x - z), digits = 2) +
      offset(log(pmax(exposure + 1, 0.5))),
    data = data,
    max_iter = 150,
    eps = 1e-09,
    toler = 1e-10,
    x = TRUE
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ pmin(x, 3) +
      log(pmax(z, 1), base = 10) +
      round(abs(x - z), digits = 2) +
      offset(log(pmax(exposure + 1, 0.5))),
    data = data,
    x = TRUE,
    control = survival::coxph.control(
      iter.max = 150,
      eps = 1e-09,
      toler.chol = 1e-10
    )
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 1e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-04)

  newdata <- transform(
    data[c(2, 5, 9, 13), ],
    x = x + c(0.4, -0.2, 0.7, -0.5),
    z = z + c(-0.3, 0.5, 0.2, 0.8),
    exposure = exposure + c(0.1, -0.15, 0.2, -0.05)
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp", reference = "zero")),
    unname(stats::predict(reference, newdata = newdata, type = "lp", reference = "zero")),
    tolerance = 3e-04
  )
})

test_that("stratum-specific Cox effects match R designs and predictions", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- survival::lung[
    stats::complete.cases(
      survival::lung[, c("time", "status", "wt.loss", "age", "sex", "ph.ecog")]
    ),
    c("time", "status", "wt.loss", "age", "sex", "ph.ecog")
  ]
  data$status <- as.integer(data$status == 2L)
  bridged <- coxph(
    Surv(time, status) ~ wt.loss + age * strata(sex) + strata(ph.ecog),
    data = data,
    x = TRUE,
    max_iter = 150,
    eps = 1e-09,
    toler = 1e-10
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ wt.loss + age * strata(sex) + strata(ph.ecog),
    data = data,
    x = TRUE,
    control = survival::coxph.control(
      iter.max = 150,
      eps = 1e-09,
      toler.chol = 1e-10
    )
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(as.vector(bridged_matrix), as.vector(reference_matrix), tolerance = 1e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-07)

  newdata <- expand.grid(
    wt.loss = 0,
    age = c(45, 65),
    sex = 1:2,
    ph.ecog = 0:2
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp", reference = "zero")),
    unname(suppressWarnings(
      stats::predict(reference, newdata = newdata, type = "lp", reference = "zero")
    )),
    tolerance = 3e-07
  )

  curve_data <- data.frame(
    wt.loss = c(0, 0),
    age = c(45, 65),
    sex = c(1, 1),
    ph.ecog = c(0, 0)
  )
  bridged_curves <- as.data.frame(survfit(bridged, newdata = curve_data, se.fit = FALSE))
  reference_curves <- survival::survfit(reference, newdata = curve_data, se.fit = FALSE)
  expect_equal(bridged_curves$time, reference_curves$time, tolerance = 1e-12)
  expect_equal(bridged_curves$surv, reference_curves$surv, tolerance = 3e-07)
  expect_equal(as.integer(table(bridged_curves$curve)), unname(reference_curves$strata))
})

test_that("strata interactions match R contrast expansion across formula shapes", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  row <- seq_len(18L)
  data <- data.frame(
    time = as.numeric(row),
    status = rep(c(1L, 0L), 9L),
    x = as.numeric(row),
    z = as.numeric((row - 1L) %% 5L + 1L),
    g = factor(rep(c("a", "b", "c"), 6L), levels = c("a", "b", "c")),
    h = factor(rep(c("u", "v"), 9L), levels = c("u", "v"))
  )
  formulas <- list(
    Surv(time, status) ~ x * strata(g),
    Surv(time, status) ~ x:strata(g),
    Surv(time, status) ~ strata(g) * x,
    Surv(time, status) ~ strata(g):(x + z),
    Surv(time, status) ~ x:strata(g, h)
  )

  for (formula in formulas) {
    bridged <- coxph(formula, data = data, x = TRUE, max_iter = 0)
    reference <- survival::coxph(
      formula,
      data = data,
      x = TRUE,
      control = survival::coxph.control(iter.max = 0)
    )
    bridged_matrix <- model.matrix(bridged)
    reference_matrix <- stats::model.matrix(reference)

    expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
    expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
    expect_equal(as.vector(bridged_matrix), as.vector(reference_matrix), tolerance = 1e-12)
    expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  }
})

test_that("polynomial formula terms match R designs and prediction rebuilding", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5110)
  n <- 100L
  x <- stats::runif(n, -2, 5)
  z <- stats::runif(n, 0.2, 4)
  w <- stats::runif(n, 0.5, 1.5)
  eta <- 0.2 * x - 0.1 * x^2 + 0.3 * log(z) + 0.08 * w * log(z)^2
  event_time <- stats::rexp(n, rate = exp(eta) / 10)
  censor_time <- stats::rexp(n, rate = 1 / 14)
  data <- data.frame(
    time = pmax(pmin(event_time, censor_time), 0.01),
    status = as.integer(event_time <= censor_time),
    x = x,
    z = z,
    w = w
  )
  bridged <- coxph(
    Surv(time, status) ~ poly(x, 3) +
      poly(log(z), degree = 2, raw = TRUE):w,
    data = data,
    max_iter = 150,
    eps = 1e-09,
    toler = 1e-10,
    x = TRUE
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ poly(x, 3) +
      poly(log(z), degree = 2, raw = TRUE):w,
    data = data,
    x = TRUE,
    control = survival::coxph.control(
      iter.max = 150,
      eps = 1e-09,
      toler.chol = 1e-10
    )
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 1e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-08)

  newdata <- transform(
    data[c(2, 7, 15, 22), ],
    x = x + c(0.5, -0.3, 0.8, -0.6),
    z = z + c(0.2, 0.5, -0.1, 0.4),
    w = w + c(-0.1, 0.15, 0.05, -0.08)
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp", reference = "zero")),
    unname(stats::predict(reference, newdata = newdata, type = "lp", reference = "zero")),
    tolerance = 2e-08
  )

  missing_data <- data[seq_len(8), ]
  missing_data$x[3] <- NA_real_
  expect_error(
    coxph(
      Surv(time, status) ~ poly(x, 2),
      data = missing_data,
      na.action = na.omit
    ),
    "missing values are not allowed"
  )
  bridged_raw <- coxph(
    Surv(time, status) ~ poly(x, 2, raw = TRUE),
    data = missing_data,
    na.action = na.omit,
    max_iter = 0,
    x = TRUE
  )
  reference_raw <- survival::coxph(
    survival::Surv(time, status) ~ poly(x, 2, raw = TRUE),
    data = missing_data,
    na.action = na.omit,
    x = TRUE,
    control = survival::coxph.control(iter.max = 0)
  )
  expect_equal(
    unname(model.matrix(bridged_raw)),
    unname(stats::model.matrix(reference_raw)),
    tolerance = 1e-12
  )

  subset_data <- data.frame(
    time = seq_len(10),
    status = rep(c(1L, 0L), 5),
    x = c(-10, -4, -2, -1, 0, 1, 2, 3, 8, 20),
    keep = c(FALSE, rep(TRUE, 7), FALSE, FALSE)
  )
  keep <- subset_data$keep
  bridged_subset <- coxph(
    Surv(time, status) ~ poly(x, 2),
    data = subset_data,
    subset = keep,
    max_iter = 0,
    x = TRUE
  )
  reference_subset <- survival::coxph(
    survival::Surv(time, status) ~ poly(x, 2),
    data = subset_data,
    subset = keep,
    x = TRUE,
    control = survival::coxph.control(iter.max = 0)
  )
  expect_equal(
    unname(model.matrix(bridged_subset)),
    unname(stats::model.matrix(reference_subset)),
    tolerance = 1e-12
  )
})

test_that("survreg polynomial formula terms match R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5112)
  n <- 90L
  x <- stats::runif(n, -2, 5)
  z <- stats::runif(n, 0.2, 4)
  latent <- exp(1.2 + 0.2 * x - 0.03 * x^2 + stats::rnorm(n, sd = 0.45))
  censor <- stats::rexp(n, rate = 1 / 20)
  data <- data.frame(
    time = pmax(pmin(latent, censor), 0.01),
    status = as.integer(latent <= censor),
    x = x,
    z = z
  )
  bridged <- survreg(
    Surv(time, status) ~ poly(x, 3) + poly(log(z), 2, raw = TRUE),
    data = data,
    dist = "weibull",
    max_iter = 150,
    eps = 1e-10,
    x = TRUE
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ poly(x, 3) + poly(log(z), 2, raw = TRUE),
    data = data,
    dist = "weibull",
    x = TRUE,
    control = survival::survreg.control(
      maxiter = 150,
      rel.tolerance = 1e-10
    )
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 1e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-08)

  newdata <- transform(
    data[c(1, 4, 9), ],
    x = x + c(0.4, -0.2, 0.7),
    z = z + c(0.2, 0.3, -0.1)
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp")),
    unname(stats::predict(reference, newdata = newdata, type = "lp")),
    tolerance = 2e-08
  )
})

test_that("coxph scale formula terms match R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5211)
  n <- 100L
  x <- stats::runif(n, -4, 7)
  z <- stats::runif(n, 0.2, 5)
  w <- stats::rnorm(n, sd = 0.4)
  event_time <- stats::rexp(n, exp(0.12 * x - 0.08 * log(z) + 0.15 * w))
  censor <- stats::rexp(n, 0.7)
  data <- data.frame(
    time = pmax(pmin(event_time, censor), 0.001),
    status = as.integer(event_time <= censor),
    x = x,
    z = z,
    w = w
  )
  bridged <- coxph(
    Surv(time, status) ~ scale(x) +
      scale(log(z), center = FALSE) +
      scale(x, scale = FALSE):w +
      offset(scale(w)),
    data = data,
    ties = "breslow",
    max_iter = 150,
    eps = 1e-10,
    x = TRUE
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ scale(x) +
      scale(log(z), center = FALSE) +
      scale(x, scale = FALSE):w +
      offset(scale(w)),
    data = data,
    ties = "breslow",
    x = TRUE,
    control = survival::coxph.control(iter.max = 150, eps = 1e-10)
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 1e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-08)

  newdata <- transform(
    data[c(2, 8, 17, 31), ],
    x = x + c(0.7, -0.4, 0.3, -0.8),
    z = z + c(0.2, 0.4, -0.1, 0.3),
    w = w + c(-0.2, 0.1, 0.25, -0.15)
  )
  expect_equal(
    as.numeric(predict(bridged, newdata = newdata, type = "lp", reference = "zero")),
    as.numeric(stats::predict(reference, newdata = newdata, type = "lp", reference = "zero")),
    tolerance = 2e-08
  )
})

test_that("scale and poly state precedes subset and missing omission", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = seq_len(10),
    status = rep(c(1L, 0L), 5),
    x = c(-10, -4, -2, -1, 0, 1, 2, 3, 8, 20),
    z = c(0.1, 0.2, 0.3, NA, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
  )
  keep <- c(FALSE, rep(TRUE, 7), FALSE, FALSE)
  bridged <- coxph(
    Surv(time, status) ~ poly(scale(x), 2) + z,
    data = data,
    subset = keep,
    na.action = na.omit,
    max_iter = 0,
    x = TRUE
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ poly(scale(x), 2) + z,
    data = data,
    subset = keep,
    na.action = na.omit,
    x = TRUE,
    control = survival::coxph.control(iter.max = 0)
  )
  expect_identical(colnames(model.matrix(bridged)), colnames(stats::model.matrix(reference)))
  expect_identical(
    attr(model.matrix(bridged), "assign"),
    attr(stats::model.matrix(reference), "assign")
  )
  expect_equal(
    unname(model.matrix(bridged)),
    unname(stats::model.matrix(reference)),
    tolerance = 1e-12
  )
})

test_that("survreg scale formula terms match R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5212)
  n <- 90L
  x <- stats::runif(n, -3, 6)
  z <- stats::runif(n, 0.3, 5)
  latent <- exp(1.1 + 0.12 * x - 0.09 * log(z) + stats::rnorm(n, sd = 0.4))
  censor <- stats::rexp(n, rate = 1 / 15)
  data <- data.frame(
    time = pmax(pmin(latent, censor), 0.01),
    status = as.integer(latent <= censor),
    x = x,
    z = z
  )
  bridged <- survreg(
    Surv(time, status) ~ scale(x) + scale(log(z), center = FALSE),
    data = data,
    dist = "weibull",
    max_iter = 150,
    eps = 1e-10,
    x = TRUE
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ scale(x) +
      scale(log(z), center = FALSE),
    data = data,
    dist = "weibull",
    x = TRUE,
    control = survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 1e-12)
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 2e-08)

  newdata <- transform(
    data[c(3, 11, 24), ],
    x = x + c(0.5, -0.3, 0.8),
    z = z + c(0.2, 0.4, -0.1)
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp")),
    unname(stats::predict(reference, newdata = newdata, type = "lp")),
    tolerance = 2e-08
  )
})

test_that("coxph natural spline formula terms match R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not_installed("splines")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5221)
  n <- 120L
  age <- stats::runif(n, 35, 85)
  marker <- stats::runif(n, 0.4, 4.8)
  sex <- rep(c(0, 1), length.out = n)
  event_time <- stats::rexp(n, exp(0.018 * age - 0.12 * log(marker) + 0.15 * sex))
  censor <- stats::rexp(n, 1.1)
  data <- data.frame(
    time = pmax(pmin(event_time, censor), 0.001),
    status = as.integer(event_time <= censor),
    age = age,
    marker = marker,
    sex = sex
  )
  formula <- Surv(time, status) ~ splines::ns(age, df = 3) +
    sex:splines::ns(log(marker), knots = c(0.5, 1), Boundary.knots = c(-1, 2))
  bridged <- coxph(
    formula,
    data = data,
    ties = "breslow",
    max_iter = 150,
    eps = 1e-10,
    x = TRUE
  )
  reference <- survival::coxph(
    formula,
    data = data,
    ties = "breslow",
    x = TRUE,
    control = survival::coxph.control(iter.max = 150, eps = 1e-10)
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 2e-12)
  expect_identical(attr(terms(bridged), "term.labels"), attr(terms(reference), "term.labels"))
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 3e-08)

  newdata <- transform(
    data[c(2, 19, 47, 83), ],
    age = age + c(2, -3, 5, -4),
    marker = marker + c(0.2, -0.1, 0.4, 0.3)
  )
  expect_equal(
    as.numeric(predict(bridged, newdata = newdata, type = "lp", reference = "zero")),
    as.numeric(stats::predict(reference, newdata = newdata, type = "lp", reference = "zero")),
    tolerance = 3e-08
  )
})

test_that("survreg natural splines preserve pre-omission state like R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not_installed("splines")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  set.seed(5222)
  n <- 90L
  x <- stats::runif(n, 0.5, 8)
  z <- stats::rnorm(n)
  latent <- exp(1 + 0.08 * x - 0.12 * z + stats::rnorm(n, sd = 0.35))
  censor <- stats::rexp(n, rate = 1 / 18)
  data <- data.frame(
    time = pmax(pmin(latent, censor), 0.01),
    status = as.integer(latent <= censor),
    x = x,
    z = z
  )
  data$z[c(7, 61)] <- NA_real_
  keep <- seq_len(n) %% 8L != 0L
  formula <- Surv(time, status) ~ splines::ns(scale(x), df = 3) + z
  bridged <- survreg(
    formula,
    data = data,
    subset = keep,
    na.action = na.omit,
    dist = "weibull",
    max_iter = 150,
    eps = 1e-10,
    x = TRUE
  )
  reference <- survival::survreg(
    formula,
    data = data,
    subset = keep,
    na.action = na.omit,
    dist = "weibull",
    x = TRUE,
    control = survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)
  )

  bridged_matrix <- model.matrix(bridged)
  reference_matrix <- stats::model.matrix(reference)
  expect_identical(colnames(bridged_matrix), colnames(reference_matrix))
  expect_identical(attr(bridged_matrix, "assign"), attr(reference_matrix, "assign"))
  expect_equal(unname(bridged_matrix), unname(reference_matrix), tolerance = 2e-12)
  expect_equal(unname(coef(bridged)), unname(coef(reference)), tolerance = 3e-08)

  newdata <- transform(
    data[c(3, 21, 44), ],
    x = x + c(0.3, -0.2, 0.5),
    z = c(-0.4, 0.2, 0.8)
  )
  expect_equal(
    unname(predict(bridged, newdata = newdata, type = "lp")),
    unname(stats::predict(reference, newdata = newdata, type = "lp")),
    tolerance = 3e-08
  )
})

test_that("pmin na.rm controls formula row omission like R", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = seq_len(8),
    status = c(1, 1, 0, 1, 0, 1, 0, 1),
    x = c(NA, NA, 0, 1, 2, 3, 4, 5),
    z = c(5, NA, 3, 2, 1, 0, -1, -2)
  )
  bridged <- coxph(
    Surv(time, status) ~ pmin(x, z, na.rm = TRUE),
    data = data,
    na.action = na.omit,
    max_iter = 0,
    x = TRUE,
    model = TRUE
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ pmin(x, z, na.rm = TRUE),
    data = data,
    na.action = na.omit,
    x = TRUE,
    model = TRUE,
    control = survival::coxph.control(iter.max = 0)
  )

  expect_identical(nrow(model.frame(bridged)), nrow(model.frame(reference)))
  expect_equal(
    unname(model.matrix(bridged)),
    unname(stats::model.matrix(reference)),
    tolerance = 1e-12
  )
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

  special_value <- c(5, NA, Inf, -Inf)
  special_breaks <- c(-Inf, 10, 20, Inf)
  bridged_special <- tcut(special_value, special_breaks)
  reference_special <- survival::tcut(special_value, special_breaks)
  expect_equal(unclass(bridged_special), unclass(reference_special))
  expect_equal(attr(bridged_special, "cutpoints"), attr(reference_special, "cutpoints"))
  expect_equal(attr(bridged_special, "labels"), attr(reference_special, "labels"))

  repeated_breaks <- c(0, 10, 10, 20)
  bridged_repeated <- tcut(c(0, 10, 15, 20), repeated_breaks)
  reference_repeated <- survival::tcut(c(0, 10, 15, 20), repeated_breaks)
  expect_equal(unclass(bridged_repeated), unclass(reference_repeated))
  expect_equal(attr(bridged_repeated, "cutpoints"), attr(reference_repeated, "cutpoints"))
  expect_equal(attr(bridged_repeated, "labels"), attr(reference_repeated, "labels"))

  scalar_value <- c(1, NA, 3)
  bridged_scalar <- tcut(scalar_value, 2, scale = 10)
  reference_scalar <- survival::tcut(scalar_value, 2, scale = 10)
  expect_equal(unclass(bridged_scalar), unclass(reference_scalar))
  expect_equal(attr(bridged_scalar, "cutpoints"), attr(reference_scalar, "cutpoints"))
  expect_equal(attr(bridged_scalar, "labels"), attr(reference_scalar, "labels"))

  bridged_tcut_subset <- get("[.tcut", envir = asNamespace("survivalr"))
  reference_tcut_subset <- get("[.tcut", envir = asNamespace("survival"))
  subset_rows <- c(4L, 2L, NA_integer_, 1L)
  expect_equal(
    bridged_tcut_subset(bridged_special, subset_rows),
    reference_tcut_subset(bridged_special, subset_rows)
  )
  expect_equal(
    bridged_tcut_subset(bridged_scaled, c(3L, 1L), drop = TRUE),
    reference_tcut_subset(bridged_scaled, c(3L, 1L), drop = TRUE)
  )
  expect_equal(
    bridged_tcut_subset(bridged_cut, c(3L, 1L), "ignored"),
    reference_tcut_subset(bridged_cut, c(3L, 1L), "ignored")
  )
  bridged_tcut_levels <- get("levels.tcut", envir = asNamespace("survivalr"))
  reference_tcut_levels <- get("levels.tcut", envir = asNamespace("survival"))
  expect_equal(
    bridged_tcut_levels(bridged_scaled),
    reference_tcut_levels(bridged_scaled)
  )

  capture_call <- function(fun, args) {
    observed_warnings <- character()
    result <- withCallingHandlers(
      tryCatch(
        list(value = do.call(fun, args)),
        error = function(error) list(error = conditionMessage(error))
      ),
      warning = function(warning) {
        observed_warnings <<- c(observed_warnings, conditionMessage(warning))
        invokeRestart("muffleWarning")
      }
    )
    list(result = result, warnings = observed_warnings)
  }
  tcut_boundary_cases <- list(
    list(x = 1, breaks = 2),
    list(x = c(1, 2), breaks = numeric()),
    list(x = c(1, 2), breaks = 2, scale = 0),
    list(x = c(1, 2), breaks = 2, scale = -1),
    list(x = c(1, 2), breaks = 2, scale = Inf),
    list(x = c(1, 2), breaks = 2, scale = NA_real_),
    list(x = c(1, 2), breaks = c(0, 1, 2), scale = c(1, -1)),
    list(x = numeric(), breaks = 2),
    list(x = NA_real_, breaks = 2),
    list(x = c(1, Inf), breaks = 2),
    list(x = c(1, 2), breaks = Inf),
    list(x = c(1, 2), breaks = NaN),
    list(x = c(1, 2), breaks = 0),
    list(x = c(1, 2), breaks = 1.5, labels = c("a", "b")),
    list(x = c(1, 2), breaks = c(0, 1, 2), labels = factor(c("a", "b")))
  )
  for (args in tcut_boundary_cases) {
    expect_identical(
      capture_call(tcut, args),
      capture_call(survival::tcut, args)
    )
  }

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
  neardate_queries <- c(NA, 2, Inf, -Inf)
  neardate_references <- c(1, NA, Inf, -Inf)
  expect_equal(
    neardate(
      rep(1, 4), rep(1, 4), neardate_queries, neardate_references,
      nomatch = 0L
    ),
    survival::neardate(
      rep(1, 4), rep(1, 4), neardate_queries, neardate_references,
      nomatch = 0L
    )
  )
  expect_equal(
    neardate(
      rep(1, 4), rep(1, 4), neardate_queries, neardate_references,
      best = "prior", nomatch = 0L
    ),
    survival::neardate(
      rep(1, 4), rep(1, 4), neardate_queries, neardate_references,
      best = "prior", nomatch = 0L
    )
  )
  expect_equal(
    neardate(c(NA, 1), c(NA, 1), c(1, 2), c(1, 2), nomatch = 0L),
    survival::neardate(c(NA, 1), c(NA, 1), c(1, 2), c(1, 2), nomatch = 0L)
  )
  expect_equal(
    neardate(c(NA, 1), c(NA, 2), c(1, 2), c(1, 2), nomatch = 0L),
    survival::neardate(c(NA, 1), c(NA, 2), c(1, 2), c(1, 2), nomatch = 0L)
  )
  expect_error(neardate(1, 1, 1, NA), "No valid entries")
  expect_error(survival::neardate(1, 1, 1, NA), "No valid entries")
  expect_error(neardate(1, 2, 1, 1), "No valid entries")
  expect_error(survival::neardate(1, 2, 1, 1), "No valid entries")
  expect_error(neardate(1, 1, factor("2020-01-01"), 1), "sortable")
  expect_error(survival::neardate(1, 1, factor("2020-01-01"), 1), "sortable")
  neardate_boundary_cases <- list(
    list(id1 = 1, id2 = c(1, 1), y1 = "b", y2 = c("a", "c")),
    list(id1 = 1, id2 = 1, y1 = 1, y2 = 2, best = "closest"),
    list(
      id1 = c(1, 1, 2), id2 = 1, y1 = c(0, 2, 1), y2 = 1,
      nomatch = c(8L, 9L)
    ),
    list(id1 = numeric(), id2 = numeric(), y1 = numeric(), y2 = numeric()),
    list(
      id1 = c("1", "01"), id2 = c("01", "1"),
      y1 = c(1, 2), y2 = c(1, 2)
    ),
    list(
      id1 = 1, id2 = c(1, 1), y1 = as.Date("2020-06-01"),
      y2 = as.Date(c("2020-01-01", "2021-01-01"))
    ),
    list(
      id1 = 1, id2 = c(1, 1), y1 = as.POSIXct("2020-06-01", tz = "UTC"),
      y2 = as.Date(c("2020-01-01", "2021-01-01"))
    ),
    list(id1 = 1, id2 = 1, y1 = factor("a"), y2 = "b"),
    list(id1 = c(1, 2), id2 = 1, y1 = 1, y2 = 1),
    list(id2 = 1, y1 = 1, y2 = 1),
    list(id1 = 1, y1 = 1, y2 = 1),
    list(id1 = 1, id2 = 1, y2 = 1),
    list(id1 = 1, id2 = 1, y1 = 1),
    list(
      id1 = c(1, 1, 2), id2 = 1, y1 = c(0, 2, 1), y2 = 1,
      nomatch = c("x", "y")
    ),
    list(
      id1 = c(1, 1, 2), id2 = 1, y1 = c(0, 2, 1), y2 = 1,
      nomatch = numeric()
    ),
    list(
      id1 = c(1, 1, 2), id2 = 1, y1 = c(0, 2, 1), y2 = 1,
      nomatch = NULL
    )
  )
  for (args in neardate_boundary_cases) {
    expect_identical(
      capture_call(neardate, args),
      capture_call(survival::neardate, args)
    )
  }
  reference_lvcf <- get0("lvcf", envir = asNamespace("survival"), inherits = FALSE)
  if (!is.null(reference_lvcf)) {
    expect_identical(names(formals(lvcf)), names(formals(reference_lvcf)))
    lvcf_boundary_cases <- list(
      list(id = numeric(), x = numeric()),
      list(id = 1, x = NA_real_, first = NA),
      list(id = c(1, NA), x = c(1, NA)),
      list(id = c(1, 1), x = 1),
      list(id = 1, x = c(1, NA)),
      list(id = c(1, 1), x = c(NA, 1), time = NULL),
      list(id = c(1, 1), x = c(NA, 1), time = numeric()),
      list(id = c(1, 1, 1), x = as.Date(c("2020-01-01", NA, "2020-01-03"))),
      list(id = c(1, 1, 1), x = I(list("a", NULL, "b")))
    )
    for (args in lvcf_boundary_cases) {
      expect_identical(
        capture_call(lvcf, args),
        capture_call(reference_lvcf, args)
      )
    }
    first_ids <- c(1, 1, 2, 2)
    for (first_value in c(TRUE, FALSE)) {
      for (first_values in list(
        c(NA, TRUE, NA, FALSE),
        c(NA_integer_, 1L, NA_integer_, 0L),
        c(NA_real_, 1, NA_real_, 0),
        rep(NA_integer_, 4),
        rep(NA_real_, 4),
        c(NA_real_, 2, NA_real_, 3),
        c(NA_character_, "x", NA_character_, "y"),
        factor(c(NA, "x", NA, "y"), levels = c("x", "y"))
      )) {
        expect_equal(
          lvcf(first_ids, first_values, first = first_value),
          reference_lvcf(first_ids, first_values, first = first_value)
        )
      }
      timed_values <- c(TRUE, NA, FALSE, NA)
      timed_order <- c(2, 1, 2, 1)
      expect_equal(
        lvcf(first_ids, timed_values, timed_order, first = first_value),
        reference_lvcf(first_ids, timed_values, timed_order, first = first_value)
      )
    }
  }
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
  if (!is.null(reference_lvcf)) {
    for (special_time in list(
      c(1, NA, 2),
      c(1, Inf, 2),
      c(1, -Inf, 2),
      c("b", "a", "c"),
      factor(c("b", "a", "c"), levels = c("c", "b", "a"))
    )) {
      expect_equal(
        lvcf(c(1, 1, 1), c(10, NA, 20), special_time),
        reference_lvcf(c(1, 1, 1), c(10, NA, 20), special_time)
      )
    }
    lvcf_factor_id <- factor(c("b", "a", "b", "a"), levels = c("b", "a"))
    expect_equal(
      lvcf(lvcf_factor_id, c(NA, 1, 2, NA)),
      reference_lvcf(lvcf_factor_id, c(NA, 1, 2, NA))
    )
    for (structured in list(
      structure(c(1, NA, 3), names = c("a", "b", "c")),
      structure(c(1L, NA, 3L), names = c("a", "b", "c")),
      structure(c(TRUE, NA, FALSE), names = c("a", "b", "c")),
      structure(c("x", NA, "z"), names = c("a", "b", "c")),
      structure(c(1, NA, 3), names = c("a", "b", "c"), source = "probe")
    )) {
      expect_equal(lvcf(c(1, 1, 1), structured), reference_lvcf(c(1, 1, 1), structured))
    }
    lvcf_matrix <- matrix(
      c(1, NA, 3, 4, NA, 6),
      nrow = 3,
      dimnames = list(c("a", "b", "c"), c("u", "v"))
    )
    expect_equal(lvcf(rep(1, 6), lvcf_matrix), reference_lvcf(rep(1, 6), lvcf_matrix))
    named_lvcf_factor <- structure(
      factor(c("a", NA, "b"), levels = c("a", "b")),
      names = c("a", "b", "c")
    )
    expect_equal(
      lvcf(c(1, 1, 1), named_lvcf_factor),
      reference_lvcf(c(1, 1, 1), named_lvcf_factor)
    )
  }
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
  if (!is.null(reference_nostutter)) {
    expect_identical(names(formals(nostutter)), names(formals(reference_nostutter)))
    nostutter_boundary_cases <- list(
      list(id = numeric(), x = numeric()),
      list(id = 1, x = 1),
      list(id = c(1, 1), x = c(NA, 1)),
      list(id = c(1, NA), x = c(1, 1)),
      list(id = c(1, 1), x = c(TRUE, TRUE)),
      list(id = c(1, 1), x = c(1, 1), censor = NA_real_),
      list(id = c(1, 1), x = c("a", "a"), censor = character()),
      list(id = c(1, 1, 1), x = c(1, 2, 1), single = TRUE),
      list(id = c(1, 1, 1), x = c(NA, 1, 1), single = TRUE),
      list(id = 1, x = c(1, 2)),
      list(id = c(1, 1), x = 1)
    )
    for (args in nostutter_boundary_cases) {
      expect_identical(
        capture_call(nostutter, args),
        capture_call(reference_nostutter, args)
      )
    }
  }
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
  ratetable_date_cases <- list(
    default = 42.5,
    integer = c(10000L, 10001L),
    Date = as.Date(c("1970-01-01", "2000-02-29")),
    POSIXt = as.POSIXct(c("1970-01-01", "2000-02-29"), tz = "UTC"),
    date = structure(c(10000, 10001), class = "date"),
    dates = structure(
      c(0, 1),
      class = "dates",
      origin = c(month = 1, day = 1, year = 1970)
    ),
    chron = structure(
      c(0, 1),
      class = "chron",
      origin = c(month = 1, day = 1, year = 1970)
    )
  )
  for (method in names(ratetable_date_cases)) {
    method_name <- paste0("ratetableDate.", method)
    bridged_method <- get(method_name, envir = asNamespace("survivalr"))
    reference_method <- get(method_name, envir = asNamespace("survival"))
    expect_equal(
      bridged_method(ratetable_date_cases[[method]]),
      reference_method(ratetable_date_cases[[method]])
    )
  }
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
  bridged_subset_method <- get("[.ratetable2", envir = asNamespace("survivalr"))
  reference_subset_method <- get("[.ratetable2", envir = asNamespace("survival"))
  subset_rows <- c(2L, 1L, 2L)
  expect_equal(
    bridged_subset_method(bridged_rtable, subset_rows),
    reference_subset_method(bridged_rtable, subset_rows)
  )
  expect_equal(
    bridged_subset_method(bridged_rtable, 1L, drop = TRUE),
    reference_subset_method(bridged_rtable, 1L, drop = TRUE)
  )
  expect_error(
    bridged_subset_method(bridged_rtable, 1L, 1L),
    "This should never be called!",
    fixed = TRUE
  )

  missing_rtable <- ratetable(
    age = c(50, NA, 70) * 365.25,
    sex = factor(c("male", "female", NA), levels = c("female", "male")),
    year = as.Date(c("2000-01-01", "2001-01-01", NA))
  )
  bridged_missing_method <- get("is.na.ratetable2", envir = asNamespace("survivalr"))
  reference_missing_method <- get("is.na.ratetable2", envir = asNamespace("survival"))
  expect_equal(
    bridged_missing_method(missing_rtable),
    reference_missing_method(missing_rtable)
  )
  expect_equal(is.na(missing_rtable), c(FALSE, TRUE, TRUE))
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

  population_table <- survival::survexp.us[1:4, , 1:3, drop = FALSE]
  bridged_subset_method <- get("[.ratetable", envir = asNamespace("survivalr"))
  reference_subset_method <- get("[.ratetable", envir = asNamespace("survival"))
  expect_equal(
    bridged_subset_method(
      population_table,
      c(4L, 2L), 1L, c(3L, 1L),
      drop = FALSE
    ),
    reference_subset_method(
      population_table,
      c(4L, 2L), 1L, c(3L, 1L),
      drop = FALSE
    )
  )
  expect_equal(
    bridged_subset_method(population_table, c(4L, 2L), 1L, c(3L, 1L)),
    reference_subset_method(population_table, c(4L, 2L), 1L, c(3L, 1L))
  )
  expect_equal(
    bridged_subset_method(population_table, 1L, 1L, 1L),
    reference_subset_method(population_table, 1L, 1L, 1L)
  )

  bridged_matrix_method <- get("as.matrix.ratetable", envir = asNamespace("survivalr"))
  reference_matrix_method <- get("as.matrix.ratetable", envir = asNamespace("survival"))
  expect_equal(
    bridged_matrix_method(population_table),
    reference_matrix_method(population_table)
  )
  missing_population_table <- population_table
  missing_population_table[2L, 1L, 1L] <- NA_real_
  bridged_missing_method <- get("is.na.ratetable", envir = asNamespace("survivalr"))
  reference_missing_method <- get("is.na.ratetable", envir = asNamespace("survival"))
  expect_equal(
    bridged_missing_method(missing_population_table),
    reference_missing_method(missing_population_table)
  )

  bridged_math_method <- get("Math.ratetable", envir = asNamespace("survivalr"))
  reference_math_method <- get("Math.ratetable", envir = asNamespace("survival"))
  bridged_ops_method <- get("Ops.ratetable", envir = asNamespace("survivalr"))
  reference_ops_method <- get("Ops.ratetable", envir = asNamespace("survival"))
  bridged_print_method <- get("print.ratetable", envir = asNamespace("survivalr"))
  reference_print_method <- get("print.ratetable", envir = asNamespace("survival"))
  registerS3method("Math", "survivalr_rate_fixture", bridged_math_method)
  registerS3method("Math", "survival_rate_fixture", reference_math_method)
  registerS3method("Ops", "survivalr_rate_fixture", bridged_ops_method)
  registerS3method("Ops", "survival_rate_fixture", reference_ops_method)
  registerS3method("print", "survivalr_rate_fixture", bridged_print_method)
  registerS3method("print", "survival_rate_fixture", reference_print_method)
  bridged_fixture <- population_table
  reference_fixture <- population_table
  class(bridged_fixture) <- "survivalr_rate_fixture"
  class(reference_fixture) <- "survival_rate_fixture"
  expect_equal(log(bridged_fixture), log(reference_fixture))
  expect_equal(bridged_fixture + 1, reference_fixture + 1)
  expect_equal(1 + bridged_fixture, 1 + reference_fixture)
  expect_equal(
    bridged_fixture == bridged_fixture,
    reference_fixture == reference_fixture
  )
  expect_equal(
    capture.output(print(bridged_fixture)),
    capture.output(print(reference_fixture))
  )
  attr(bridged_fixture, "dimid") <- NULL
  attr(reference_fixture, "dimid") <- NULL
  expect_equal(
    capture.output(print(bridged_fixture)),
    capture.output(print(reference_fixture))
  )

  bridged_summary_method <- get("summary.ratetable", envir = asNamespace("survivalr"))
  reference_summary_method <- get("summary.ratetable", envir = asNamespace("survival"))
  bridged_summary_output <- capture.output(
    bridged_summary <- bridged_summary_method(population_table)
  )
  reference_summary_output <- capture.output(
    reference_summary <- reference_summary_method(population_table)
  )
  expect_equal(bridged_summary_output, reference_summary_output)
  expect_equal(bridged_summary, reference_summary)
  legacy_population_table <- population_table
  attr(legacy_population_table, "factor") <- c(0, 1, 365.25)
  attr(legacy_population_table, "type") <- NULL
  bridged_legacy_output <- capture.output(
    bridged_legacy_summary <- bridged_summary_method(legacy_population_table)
  )
  reference_legacy_output <- capture.output(
    reference_legacy_summary <- reference_summary_method(legacy_population_table)
  )
  expect_equal(bridged_legacy_output, reference_legacy_output)
  expect_equal(bridged_legacy_summary, reference_legacy_summary)
  expect_error(bridged_summary_method(1), "Argument is not a rate table")

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
  cipoisson_edges <- list(
    list(k = c(1.2, 2.8), time = c(2, 4), p = c(0, 1)),
    list(k = c(0, 1.2, Inf), time = Inf, p = 0.95),
    list(k = c(0, 1.2, Inf), time = 1, p = 1),
    list(k = c(0, 1.2), time = 1, p = NA_real_)
  )
  for (args in cipoisson_edges) {
    expect_equal(
      do.call(cipoisson, args),
      do.call(survival::cipoisson, args),
      tolerance = 1e-6
    )
  }
  cipoisson_boundary_cases <- list(
    list(k = numeric()),
    list(k = NA_real_),
    list(k = NaN),
    list(k = -1),
    list(k = -2, method = "exact"),
    list(k = -1, method = "anscombe"),
    list(k = structure(c(1, 2), names = c("a", "b"))),
    list(k = matrix(1:4, 2, dimnames = list(c("r1", "r2"), c("c1", "c2")))),
    list(k = 1:5, time = c(1, 2), p = c(0.9, 0.95, 0.99)),
    list(k = numeric(), time = 1:2),
    list(k = 1:2, time = numeric()),
    list(k = 1:2, p = numeric()),
    list(k = array(numeric(), dim = c(0L, 2L)), time = numeric(), p = numeric()),
    list(
      k = array(numeric(), dim = c(0L, 2L), dimnames = list(character(), c("a", "b"))),
      time = numeric(),
      p = numeric()
    ),
    list(k = array(numeric(), dim = c(0L, 2L, 3L)), time = numeric(), p = numeric()),
    list(k = -1, time = NA_real_),
    list(k = c(-1, 1), time = c(NA_real_, 0)),
    list(k = -1, p = 2, method = "anscombe"),
    list(k = numeric(), method = "invalid")
  )
  for (args in cipoisson_boundary_cases) {
    expect_identical(
      capture_call(cipoisson, args),
      capture_call(survival::cipoisson, args)
    )
  }

  link_x <- c(0, 0.01, 0.05, 0.5, 0.95, 0.99, 1, NA)
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
  capture_linkfun <- function(factory, edge, mu) {
    capture_call(factory(edge)$linkfun, list(mu = mu))
  }
  link_boundary_edges <- list(
    0, 0.5, 0.6, -1, Inf, -Inf, NA_real_, NaN, numeric(), c(0.05, 0.1)
  )
  link_boundary_inputs <- list(
    c(NA, NaN, -Inf, 0, 0.5, 1, Inf),
    numeric(),
    structure(c(0, 0.5, 1), names = c("low", "middle", "high")),
    matrix(c(0, 0.5, 1, NA), nrow = 2L, dimnames = list(c("a", "b"), c("x", "y")))
  )
  for (link_name in c("blogit", "bprobit", "bcloglog", "blog")) {
    bridged_factory <- get(link_name, envir = asNamespace("survivalr"))
    reference_factory <- get(link_name, envir = asNamespace("survival"))
    for (edge in link_boundary_edges) {
      for (mu in link_boundary_inputs) {
        expect_identical(
          capture_linkfun(bridged_factory, edge, mu),
          capture_linkfun(reference_factory, edge, mu)
        )
      }
    }
  }

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
  bridged_formula <- survexp(
    survival::Surv(time, status) ~ 1,
    data = fallback_data,
    times = c(5, 10)
  )
  reference_formula <- survival::survexp(
    survival::Surv(time, status) ~ 1,
    data = fallback_data,
    times = c(5, 10)
  )
  expect_equal(bridged_formula$time, reference_formula$time)
  expect_equal(bridged_formula$surv, reference_formula$surv, tolerance = 1e-12)
  expect_equal(bridged_formula$n.risk, reference_formula$n.risk)

  grouped_data <- data.frame(
    time = c(365, 730, 1095, 1460),
    status = c(1, 0, 1, 0),
    cohort = factor(c("a", "a", "b", "b")),
    age = c(45, 55, 65, 75) * 365.25,
    sex = factor(c("male", "female", "male", "female")),
    year = as.Date(c("1995-03-01", "2000-07-01", "2005-11-01", "2010-01-01"))
  )
  for (survexp_method in c("ederer", "hakulinen", "conditional")) {
    bridged_grouped <- survexp(
      survival::Surv(time, status) ~ cohort,
      data = grouped_data,
      times = c(180, 365, 730, 1000),
      method = survexp_method
    )
    reference_grouped <- survival::survexp(
      survival::Surv(time, status) ~ cohort,
      data = grouped_data,
      times = c(180, 365, 730, 1000),
      method = survexp_method
    )
    expect_equal(bridged_grouped$time, reference_grouped$time)
    expect_equal(bridged_grouped$surv, reference_grouped$surv, tolerance = 1e-12)
    expect_equal(bridged_grouped$n.risk, reference_grouped$n.risk)
  }
  for (survexp_method in c("individual.s", "individual.h")) {
    expect_equal(
      survexp(
        survival::Surv(time, status) ~ 1,
        data = grouped_data,
        method = survexp_method
      ),
      survival::survexp(
        survival::Surv(time, status) ~ 1,
        data = grouped_data,
        method = survexp_method
      ),
      tolerance = 1e-12
    )
  }
  no_response_bridge <- survexp(
    ~ cohort,
    data = grouped_data,
    times = c(180, 365, 730)
  )
  no_response_reference <- survival::survexp(
    ~ cohort,
    data = grouped_data,
    times = c(180, 365, 730)
  )
  expect_equal(no_response_bridge$surv, no_response_reference$surv, tolerance = 1e-12)
  expect_equal(no_response_bridge$n.risk, no_response_reference$n.risk)
  strip_survexp_call <- function(value) {
    value <- unclass(value)
    value$call <- NULL
    value
  }
  expect_warning(
    empty_grouped_bridge <- survexp(
      survival::Surv(time, status) ~ cohort,
      data = grouped_data,
      times = numeric(),
      model = TRUE
    ),
    "no non-missing arguments"
  )
  expect_warning(
    empty_grouped_reference <- survival::survexp(
      survival::Surv(time, status) ~ cohort,
      data = grouped_data,
      times = numeric(),
      model = TRUE
    ),
    "no non-missing arguments"
  )
  expect_equal(
    strip_survexp_call(empty_grouped_bridge),
    strip_survexp_call(empty_grouped_reference),
    tolerance = 1e-12
  )
  expect_warning(
    empty_no_response_bridge <- survexp(
      ~ cohort,
      data = grouped_data,
      times = numeric(),
      x = TRUE,
      y = TRUE
    ),
    "no non-missing arguments"
  )
  expect_warning(
    empty_no_response_reference <- survival::survexp(
      ~ cohort,
      data = grouped_data,
      times = numeric(),
      x = TRUE,
      y = TRUE
    ),
    "no non-missing arguments"
  )
  expect_equal(
    strip_survexp_call(empty_no_response_bridge),
    strip_survexp_call(empty_no_response_reference),
    tolerance = 1e-12
  )

  remapped_data <- transform(grouped_data, attained_age = age)
  expect_equal(
    survexp(
      survival::Surv(time, status) ~ cohort,
      data = remapped_data,
      rmap = list(age = attained_age),
      times = c(365, 730)
    )$surv,
    survival::survexp(
      survival::Surv(time, status) ~ cohort,
      data = remapped_data,
      rmap = list(age = attained_age),
      times = c(365, 730)
    )$surv,
    tolerance = 1e-12
  )

  survexp_cox_data <- stats::na.omit(
    survival::lung[, c("time", "status", "age", "sex", "ph.ecog")]
  )
  survexp_cox_data$case_weight <- seq_len(nrow(survexp_cox_data)) %% 4L + 1L
  bridged_survexp_cox_fit <- coxph(
    Surv(time, status) ~ age + sex,
    data = survexp_cox_data,
    x = TRUE,
    model = TRUE
  )
  reference_survexp_cox_fit <- survival::coxph(
    survival::Surv(time, status) ~ age + sex,
    data = survexp_cox_data,
    x = TRUE,
    model = TRUE
  )
  survexp_cox_formula <- Surv(time, status) ~ factor(ph.ecog)
  for (cox_method in c(
    "ederer", "hakulinen", "conditional", "individual.s", "individual.h"
  )) {
    bridged_survexp_cox <- survexp(
      survexp_cox_formula,
      data = survexp_cox_data,
      weights = case_weight,
      ratetable = bridged_survexp_cox_fit,
      rmap = list(age = age, sex = sex),
      method = cox_method,
      times = c(100, 300, 500)
    )
    reference_survexp_cox <- survival::survexp(
      survexp_cox_formula,
      data = survexp_cox_data,
      weights = case_weight,
      ratetable = reference_survexp_cox_fit,
      rmap = list(age = age, sex = sex),
      method = cox_method,
      times = c(100, 300, 500)
    )
    if (is.list(bridged_survexp_cox)) {
      expect_equal(
        bridged_survexp_cox$surv,
        reference_survexp_cox$surv,
        tolerance = 1e-12
      )
      expect_equal(bridged_survexp_cox$n.risk, reference_survexp_cox$n.risk)
      expect_equal(bridged_survexp_cox$time, reference_survexp_cox$time)
      expect_equal(bridged_survexp_cox$method, reference_survexp_cox$method)
    } else {
      expect_equal(
        bridged_survexp_cox,
        reference_survexp_cox,
        tolerance = 1e-12
      )
    }
  }
  bridged_survexp_cox_no_response <- survexp(
    ~factor(ph.ecog),
    data = survexp_cox_data,
    ratetable = bridged_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "ederer",
    times = c(100, 300, 500),
    x = TRUE
  )
  reference_survexp_cox_no_response <- survival::survexp(
    ~factor(ph.ecog),
    data = survexp_cox_data,
    ratetable = reference_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "ederer",
    times = c(100, 300, 500),
    x = TRUE
  )
  expect_equal(
    bridged_survexp_cox_no_response$surv,
    reference_survexp_cox_no_response$surv,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_survexp_cox_no_response$n.risk,
    reference_survexp_cox_no_response$n.risk
  )
  expect_equal(
    bridged_survexp_cox_no_response$x,
    reference_survexp_cox_no_response$x
  )
  single_survexp_cox_data <- survexp_cox_data[1L, , drop = FALSE]
  bridged_survexp_cox_single <- survexp(
    Surv(time, status) ~ 1,
    data = single_survexp_cox_data,
    weights = 0,
    ratetable = bridged_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "conditional",
    times = c(100, 300, 500)
  )
  reference_survexp_cox_single <- survival::survexp(
    Surv(time, status) ~ 1,
    data = single_survexp_cox_data,
    weights = 0,
    ratetable = reference_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "conditional",
    times = c(100, 300, 500)
  )
  expect_equal(
    bridged_survexp_cox_single$surv,
    reference_survexp_cox_single$surv,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_survexp_cox_single$n.risk,
    reference_survexp_cox_single$n.risk
  )
  signed_survexp_weights <- rep(1, nrow(survexp_cox_data))
  signed_survexp_weights[[1L]] <- -0.5
  bridged_survexp_cox_signed <- survexp(
    survexp_cox_formula,
    data = survexp_cox_data,
    weights = signed_survexp_weights,
    ratetable = bridged_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "ederer",
    times = c(100, 300, 500)
  )
  reference_survexp_cox_signed <- survival::survexp(
    survexp_cox_formula,
    data = survexp_cox_data,
    weights = signed_survexp_weights,
    ratetable = reference_survexp_cox_fit,
    rmap = list(age = age, sex = sex),
    method = "ederer",
    times = c(100, 300, 500)
  )
  expect_equal(
    bridged_survexp_cox_signed$surv,
    reference_survexp_cox_signed$surv,
    tolerance = 1e-12
  )

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
  bridged_pyears_single <- pyears(
    3,
    event = 1,
    group = "only",
    weights = 2,
    scale = 1
  )
  expect_equal(unname(bridged_pyears_single$pyears), 6)
  expect_equal(unname(bridged_pyears_single$n), 1)
  expect_equal(unname(bridged_pyears_single$event), 2)
  pyears_formula_data <- data.frame(
    time = c(10, 20, 30),
    status = c(1, 0, 1),
    group = c("a", "a", "b"),
    wt = c(1, 2, 3)
  )
  expect_error(pyears(), "A formula argument is required")
  expect_error(survival::pyears(), "A formula argument is required")
  expect_error(
    pyears(
      Surv(time, status) ~ 1,
      data = pyears_formula_data,
      rmap = list(age = time)
    ),
    "No rate table specified"
  )
  expect_error(
    survival::pyears(
      survival::Surv(time, status) ~ 1,
      data = pyears_formula_data,
      rmap = list(age = time)
    ),
    "No rate table specified"
  )
  expect_error(
    pyears(
      Surv(time, status) ~ 1,
      data = pyears_formula_data,
      ratetable = matrix(1)
    ),
    "Invalid rate table"
  )
  expect_error(
    survival::pyears(
      survival::Surv(time, status) ~ 1,
      data = pyears_formula_data,
      ratetable = matrix(1)
    ),
    "Invalid rate table"
  )
  expect_error(
    pyears(
      Surv(time, status) ~ 1,
      data = pyears_formula_data,
      ratetable = bridged_survexp_cox_fit,
      rmap = list(age = time, sex = status)
    ),
    "Cox rate models are not supported by pyears"
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
  expect_error(
    pyears(time ~ 1, data = pyears_formula_data, data.frame = TRUE),
    "arguments imply differing number of rows: 1, 0",
    fixed = TRUE
  )
  pyears_single_data <- data.frame(time = 3, status = 1, group = "only", wt = 2)
  bridged_pyears_single_formula <- pyears(
    Surv(time, status) ~ group,
    data = pyears_single_data,
    weights = wt,
    scale = 1
  )
  reference_pyears_single_formula <- survival::pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_single_data,
    weights = wt,
    scale = 1
  )
  expect_equal(
    bridged_pyears_single_formula$pyears,
    reference_pyears_single_formula$pyears
  )
  expect_equal(bridged_pyears_single_formula$n, reference_pyears_single_formula$n)
  expect_equal(
    bridged_pyears_single_formula$event,
    reference_pyears_single_formula$event
  )
  pyears_environment_fixture <- local({
    time <- c(10, 20, 30, 40)
    status <- c(1, 0, 1, 1)
    group <- factor(c("treated", "treated", "control", "control"))
    wt <- c(1, 2, 3, 4)
    list(
      formula = Surv(time, status) ~ group,
      weights = wt
    )
  })
  bridged_pyears_environment <- pyears(
    pyears_environment_fixture$formula,
    weights = pyears_environment_fixture$weights,
    scale = 1,
    data.frame = TRUE
  )
  reference_pyears_environment <- survival::pyears(
    pyears_environment_fixture$formula,
    weights = pyears_environment_fixture$weights,
    scale = 1,
    data.frame = TRUE
  )
  expect_equal(
    bridged_pyears_environment$data,
    reference_pyears_environment$data
  )
  expect_equal(
    bridged_pyears_environment$offtable,
    reference_pyears_environment$offtable
  )
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
  pyears_negative_counting_data <- data.frame(
    start = -2,
    stop = -1,
    event = 1,
    group = "a"
  )
  bridged_pyears_negative_counting <- pyears(
    Surv(start, stop, event) ~ group,
    data = pyears_negative_counting_data,
    scale = 1
  )
  reference_pyears_negative_counting <- survival::pyears(
    survival::Surv(start, stop, event) ~ group,
    data = pyears_negative_counting_data,
    scale = 1
  )
  expect_equal(
    bridged_pyears_negative_counting$pyears,
    reference_pyears_negative_counting$pyears
  )
  expect_equal(
    bridged_pyears_negative_counting$n,
    reference_pyears_negative_counting$n
  )
  expect_equal(
    bridged_pyears_negative_counting$event,
    reference_pyears_negative_counting$event
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

  pyears_tcut_data <- data.frame(
    time = c(25, 8),
    status = c(1, 0),
    base = c(0, 5)
  )
  bridged_pyears_tcut <- pyears(
    Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_data,
    scale = 1
  )
  reference_pyears_tcut <- survival::pyears(
    survival::Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_data,
    scale = 1
  )
  expect_equal(bridged_pyears_tcut$pyears, reference_pyears_tcut$pyears)
  expect_equal(bridged_pyears_tcut$n, reference_pyears_tcut$n)
  expect_equal(bridged_pyears_tcut$event, reference_pyears_tcut$event)
  expect_equal(bridged_pyears_tcut$offtable, reference_pyears_tcut$offtable)
  expect_true(bridged_pyears_tcut$tcut)

  bridged_pyears_tcut_frame <- pyears(
    Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_data,
    scale = 1,
    data.frame = TRUE
  )
  reference_pyears_tcut_frame <- survival::pyears(
    survival::Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_data,
    scale = 1,
    data.frame = TRUE
  )
  expect_equal(bridged_pyears_tcut_frame$data, reference_pyears_tcut_frame$data)

  pyears_tcut_counting_data <- data.frame(
    start = c(5, 12),
    stop = c(25, 18),
    status = c(1, 0),
    base = c(0, 5)
  )
  bridged_pyears_tcut_counting <- pyears(
    Surv(start, stop, status) ~ tcut(base, c(0, 10, 20, 30, 40)),
    data = pyears_tcut_counting_data,
    scale = 1
  )
  reference_pyears_tcut_counting <- survival::pyears(
    survival::Surv(start, stop, status) ~ tcut(base, c(0, 10, 20, 30, 40)),
    data = pyears_tcut_counting_data,
    scale = 1
  )
  expect_equal(bridged_pyears_tcut_counting$pyears, reference_pyears_tcut_counting$pyears)
  expect_equal(bridged_pyears_tcut_counting$n, reference_pyears_tcut_counting$n)
  expect_equal(bridged_pyears_tcut_counting$event, reference_pyears_tcut_counting$event)

  pyears_tcut_mixed_data <- data.frame(
    time = c(25, 8, 12),
    status = c(1, 0, 1),
    base = c(0, 5, 15),
    sex = factor(c("f", "m", "f"), levels = c("f", "m"))
  )
  bridged_pyears_tcut_mixed <- pyears(
    Surv(time, status) ~ tcut(base, c(0, 10, 20, 30, 40)) + sex,
    data = pyears_tcut_mixed_data,
    scale = 1
  )
  reference_pyears_tcut_mixed <- survival::pyears(
    survival::Surv(time, status) ~ tcut(base, c(0, 10, 20, 30, 40)) + sex,
    data = pyears_tcut_mixed_data,
    scale = 1
  )
  expect_equal(bridged_pyears_tcut_mixed$pyears, reference_pyears_tcut_mixed$pyears)
  expect_equal(bridged_pyears_tcut_mixed$n, reference_pyears_tcut_mixed$n)
  expect_equal(bridged_pyears_tcut_mixed$event, reference_pyears_tcut_mixed$event)

  pyears_tcut_outside_data <- data.frame(
    time = c(10, 10, 10),
    status = c(1, 1, 1),
    base = c(-5, 25, 35)
  )
  bridged_pyears_tcut_outside <- pyears(
    Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_outside_data,
    scale = 1
  )
  reference_pyears_tcut_outside <- survival::pyears(
    survival::Surv(time, status) ~ tcut(base, c(0, 10, 20, 30)),
    data = pyears_tcut_outside_data,
    scale = 1
  )
  expect_equal(bridged_pyears_tcut_outside$pyears, reference_pyears_tcut_outside$pyears)
  expect_equal(bridged_pyears_tcut_outside$event, reference_pyears_tcut_outside$event)
  expect_equal(bridged_pyears_tcut_outside$offtable, reference_pyears_tcut_outside$offtable)

  pyears_tcut_weighted_data <- data.frame(
    time = c(20, 20),
    status = c(1, 1),
    base = c(0, 0),
    wt = c(2, 0.5)
  )
  bridged_pyears_tcut_weighted <- pyears(
    Surv(time, status) ~ tcut(base, c(0, 10, 20)),
    data = pyears_tcut_weighted_data,
    weights = wt,
    scale = 1
  )
  reference_pyears_tcut_weighted <- survival::pyears(
    survival::Surv(time, status) ~ tcut(base, c(0, 10, 20)),
    data = pyears_tcut_weighted_data,
    weights = wt,
    scale = 1
  )
  expect_equal(bridged_pyears_tcut_weighted$pyears, reference_pyears_tcut_weighted$pyears)
  expect_equal(bridged_pyears_tcut_weighted$n, reference_pyears_tcut_weighted$n)
  expect_equal(bridged_pyears_tcut_weighted$event, reference_pyears_tcut_weighted$event)

  pyears_ratetable_data <- data.frame(
    time = c(30, 120, 365, 700),
    status = c(1, 0, 1, 1),
    group = factor(c("a", "a", "b", "b")),
    age = c(45, 55, 65, 75) * 365.25,
    sex = factor(c("male", "female", "male", "female")),
    year = as.Date(c("1995-03-01", "2000-07-01", "2005-11-01", "2010-01-01"))
  )
  pyears_ratetable_environment <- local({
    time <- pyears_ratetable_data$time
    status <- pyears_ratetable_data$status
    group <- pyears_ratetable_data$group
    age <- pyears_ratetable_data$age
    sex <- pyears_ratetable_data$sex
    year <- pyears_ratetable_data$year
    survival::Surv(time, status) ~ group
  })
  bridged_pyears_ratetable_environment <- pyears(
    pyears_ratetable_environment,
    ratetable = survival::survexp.us,
    scale = 365.25
  )
  reference_pyears_ratetable_environment <- survival::pyears(
    pyears_ratetable_environment,
    ratetable = survival::survexp.us,
    scale = 365.25
  )
  expect_equal(
    bridged_pyears_ratetable_environment$pyears,
    reference_pyears_ratetable_environment$pyears,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_pyears_ratetable_environment$expected,
    reference_pyears_ratetable_environment$expected,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_pyears_ratetable_environment$event,
    reference_pyears_ratetable_environment$event
  )
  for (expected_scale in c("event", "pyears")) {
    bridged_ratetable <- pyears(
      survival::Surv(time, status) ~ group,
      data = pyears_ratetable_data,
      ratetable = survival::survexp.us,
      scale = 365.25,
      expect = expected_scale
    )
    reference_ratetable <- survival::pyears(
      survival::Surv(time, status) ~ group,
      data = pyears_ratetable_data,
      ratetable = survival::survexp.us,
      scale = 365.25,
      expect = expected_scale
    )
    expect_equal(bridged_ratetable$pyears, reference_ratetable$pyears, tolerance = 1e-12)
    expect_equal(bridged_ratetable$n, reference_ratetable$n)
    expect_equal(bridged_ratetable$event, reference_ratetable$event)
    expect_equal(bridged_ratetable$expected, reference_ratetable$expected, tolerance = 1e-12)
    expect_equal(bridged_ratetable$offtable, reference_ratetable$offtable)
    expect_equal(bridged_ratetable$summary, reference_ratetable$summary)
    expect_identical(names(bridged_ratetable), names(reference_ratetable))
  }

  pyears_ratetable_single_data <- data.frame(
    start = 117,
    stop = 126,
    status = 1,
    age = 41 * 365.25,
    sex = factor("female"),
    year = as.Date("1974-07-09")
  )
  bridged_ratetable_single <- pyears(
    survival::Surv(start, stop, status) ~ 1,
    data = pyears_ratetable_single_data,
    ratetable = survival::survexp.us,
    expect = "pyears",
    scale = 365.25
  )
  reference_ratetable_single <- survival::pyears(
    survival::Surv(start, stop, status) ~ 1,
    data = pyears_ratetable_single_data,
    ratetable = survival::survexp.us,
    expect = "pyears",
    scale = 365.25
  )
  expect_equal(
    bridged_ratetable_single$expected,
    reference_ratetable_single$expected,
    tolerance = 1e-14
  )

  pyears_ratetable_frame <- pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_ratetable_data,
    ratetable = survival::survexp.us,
    scale = 365.25,
    data.frame = TRUE
  )
  reference_ratetable_frame <- survival::pyears(
    survival::Surv(time, status) ~ group,
    data = pyears_ratetable_data,
    ratetable = survival::survexp.us,
    scale = 365.25,
    data.frame = TRUE
  )
  expect_equal(
    pyears_ratetable_frame$data,
    reference_ratetable_frame$data,
    tolerance = 1e-12
  )

  bridged_print <- bridged_pyears_formula
  reference_print <- reference_pyears_formula
  bridged_print$call <- reference_print$call <- quote(pyears(Surv(time, status) ~ group))
  reference_print_method <- getFromNamespace("print.pyears", "survival")
  reference_summary_method <- getFromNamespace("summary.pyears", "survival")
  expect_equal(
    capture.output(print.pyears(bridged_print)),
    capture.output(reference_print_method(reference_print))
  )

  summary_options <- list(
    header = FALSE,
    call = FALSE,
    rate = TRUE,
    ci.r = TRUE,
    totals = TRUE,
    vertical = FALSE,
    vline = TRUE,
    digits = 4
  )
  expect_equal(
    capture.output(do.call(summary.pyears, c(list(object = bridged_pyears_formula), summary_options))),
    capture.output(do.call(reference_summary_method, c(list(object = reference_pyears_formula), summary_options)))
  )
  expect_equal(
    capture.output(summary.pyears(bridged_pyears_multi, header = FALSE, call = FALSE, vline = TRUE)),
    capture.output(reference_summary_method(reference_pyears_multi, header = FALSE, call = FALSE, vline = TRUE))
  )
  expect_equal(
    capture.output(summary.pyears(bridged_pyears_formula_frame, header = FALSE, call = FALSE)),
    capture.output(reference_summary_method(reference_pyears_formula_frame, header = FALSE, call = FALSE))
  )
  ratetable_summary_options <- list(
    header = FALSE,
    call = FALSE,
    rate = TRUE,
    ci.r = TRUE,
    rr = TRUE,
    ci.rr = TRUE,
    digits = 5
  )
  expect_equal(
    capture.output(do.call(summary.pyears, c(list(object = bridged_ratetable), ratetable_summary_options))),
    capture.output(do.call(reference_summary_method, c(list(object = reference_ratetable), ratetable_summary_options)))
  )
  expect_error(summary.pyears(bridged_pyears_formula, header = "yes"), "must be single logical values")

  pyears_ratetable_tcut_data <- transform(
    pyears_ratetable_data,
    attained = c(0, 20, 100, 200)
  )
  bridged_ratetable_tcut <- pyears(
    survival::Surv(time, status) ~ tcut(attained, c(0, 365, 730, 1460)),
    data = pyears_ratetable_tcut_data,
    ratetable = survival::survexp.us,
    scale = 365.25
  )
  reference_ratetable_tcut <- survival::pyears(
    survival::Surv(time, status) ~ tcut(attained, c(0, 365, 730, 1460)),
    data = pyears_ratetable_tcut_data,
    ratetable = survival::survexp.us,
    scale = 365.25
  )
  expect_equal(bridged_ratetable_tcut$pyears, reference_ratetable_tcut$pyears)
  expect_equal(bridged_ratetable_tcut$event, reference_ratetable_tcut$event)
  expect_equal(bridged_ratetable_tcut$expected, reference_ratetable_tcut$expected, tolerance = 1e-12)
  expect_equal(bridged_ratetable_tcut$offtable, reference_ratetable_tcut$offtable)

  remapped_pyears_data <- transform(pyears_ratetable_data, attained_age = age)
  expect_equal(
    pyears(
      survival::Surv(time, status) ~ group,
      data = remapped_pyears_data,
      ratetable = survival::survexp.us,
      rmap = list(age = attained_age),
      scale = 365.25
    )$expected,
    survival::pyears(
      survival::Surv(time, status) ~ group,
      data = remapped_pyears_data,
      ratetable = survival::survexp.us,
      rmap = list(age = attained_age),
      scale = 365.25
    )$expected,
    tolerance = 1e-12
  )

  pyears_ratetable_counting <- transform(
    pyears_ratetable_data,
    start = c(0, 30, 100, 200),
    stop = time
  )
  bridged_ratetable_counting <- pyears(
    survival::Surv(start, stop, status) ~ group,
    data = pyears_ratetable_counting,
    ratetable = survival::survexp.us,
    scale = 365.25,
    x = TRUE,
    y = TRUE
  )
  reference_ratetable_counting <- survival::pyears(
    survival::Surv(start, stop, status) ~ group,
    data = pyears_ratetable_counting,
    ratetable = survival::survexp.us,
    scale = 365.25,
    x = TRUE,
    y = TRUE
  )
  expect_equal(
    bridged_ratetable_counting$pyears,
    reference_ratetable_counting$pyears,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_ratetable_counting$expected,
    reference_ratetable_counting$expected,
    tolerance = 1e-12
  )
  expect_equal(bridged_ratetable_counting$event, reference_ratetable_counting$event)
  expect_equal(bridged_ratetable_counting$x, reference_ratetable_counting$x)
  expect_equal(bridged_ratetable_counting$y, reference_ratetable_counting$y)

  bridged_ratetable_model <- pyears(
    survival::Surv(time, status) ~ 1,
    data = pyears_ratetable_data,
    ratetable = survival::survexp.us,
    scale = 1,
    model = TRUE
  )
  reference_ratetable_model <- survival::pyears(
    survival::Surv(time, status) ~ 1,
    data = pyears_ratetable_data,
    ratetable = survival::survexp.us,
    scale = 1,
    model = TRUE
  )
  expect_equal(bridged_ratetable_model$pyears, reference_ratetable_model$pyears)
  expect_equal(bridged_ratetable_model$expected, reference_ratetable_model$expected, tolerance = 1e-12)
  expect_equal(names(bridged_ratetable_model$model), names(reference_ratetable_model$model))
  for (column in names(bridged_ratetable_model$model)) {
    expect_equal(
      bridged_ratetable_model$model[[column]],
      reference_ratetable_model$model[[column]]
    )
  }

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
  bridged_finegray_formula <- suppressWarnings(finegray(
    survival::Surv(time, status) ~ x,
    data = finegray_data,
    etype = "a"
  ))
  reference_finegray_formula <- suppressWarnings(survival::finegray(
    survival::Surv(time, status) ~ x,
    data = finegray_data,
    etype = "a"
  ))
  expect_equal(bridged_finegray_formula, reference_finegray_formula)
  finegray_environment_fixture <- local({
    time <- seq_len(8L)
    status <- factor(
      c("a", "censored", "b", "a", "censored", "b", "a", "censored"),
      levels = c("censored", "a", "b")
    )
    keeper <- ordered(
      c("z", "a", "z", "a", "z", "a", "z", "a"),
      levels = c("z", "a")
    )
    x <- c(1, 2, 3, 4, NA, 6, 7, 8)
    wt <- c(1, 2, 1, 3, 1, 2, 1, 4)
    list(
      formula = Surv(time, status) ~ keeper + I(x^2),
      weights = wt
    )
  })
  bridged_finegray_environment <- finegray(
    finegray_environment_fixture$formula,
    weights = finegray_environment_fixture$weights,
    subset = seq_len(7L),
    na.action = stats::na.omit,
    etype = "a",
    count = "extra rows"
  )
  reference_finegray_environment <- survival::finegray(
    finegray_environment_fixture$formula,
    weights = finegray_environment_fixture$weights,
    subset = seq_len(7L),
    na.action = stats::na.omit,
    etype = "a",
    count = "extra rows"
  )
  expect_s3_class(bridged_finegray_environment$keeper, "ordered")
  expect_s3_class(bridged_finegray_environment[["I(x^2)"]], "AsIs")
  expect_equal(bridged_finegray_environment, reference_finegray_environment)
  expect_equal(
    finegray(
      finegray_environment_fixture$formula,
      data = NULL,
      na.action = stats::na.omit,
      etype = "b"
    ),
    survival::finegray(
      finegray_environment_fixture$formula,
      data = NULL,
      na.action = stats::na.omit,
      etype = "b"
    )
  )
  expect_error(finegray(), "A formula argument is required")
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
  finegray_weighted_strata_data <- data.frame(
    time = rep(1:3, 2),
    status = factor(
      rep(c("target", "censored", "compete"), 2),
      levels = c("censored", "target", "compete")
    ),
    x = c(10, 11, 12, 20, 21, 22),
    group = factor(rep(c("z", "a"), each = 3)),
    wt = c(101, 102, 103, 201, 202, 203)
  )
  bridged_finegray_weighted_strata <- finegray(
    Surv(time, status) ~ x + strata(group),
    data = finegray_weighted_strata_data,
    weights = wt,
    etype = "target"
  )
  reference_finegray_weighted_strata <- survival::finegray(
    survival::Surv(time, status) ~ x + strata(group),
    data = finegray_weighted_strata_data,
    weights = wt,
    etype = "target"
  )
  expect_equal(bridged_finegray_weighted_strata, reference_finegray_weighted_strata)
  expect_equal(head(bridged_finegray_weighted_strata[["(weights)"]], 3), 201:203)
  expect_equal(head(bridged_finegray_weighted_strata$fgwt, 3), 101:103)
  finegray_class_data <- data.frame(
    time = seq_len(8L),
    status = factor(
      c("a", "censored", "b", "a", "censored", "b", "a", "censored"),
      levels = c("censored", "a", "b")
    ),
    keeper = ordered(
      c("z", "a", "z", "a", "z", "a", "z", "a"),
      levels = c("z", "a")
    ),
    x = c(1, 2, 3, 4, NA, 6, 7, 8),
    wt = c(1, 2, 1, 3, 1, 2, 1, 4)
  )
  finegray_rows <- seq_len(7L)
  finegray_class_formula <- Surv(time, status) ~ keeper + I(x^2)
  bridged_finegray_classes <- finegray(
    finegray_class_formula,
    data = finegray_class_data,
    weights = wt,
    subset = finegray_rows,
    na.action = na.omit,
    etype = "a",
    count = "extra rows"
  )
  reference_finegray_classes <- survival::finegray(
    survival::Surv(time, status) ~ keeper + I(x^2),
    data = finegray_class_data,
    weights = wt,
    subset = finegray_rows,
    na.action = na.omit,
    etype = "a",
    count = "extra rows"
  )
  expect_s3_class(bridged_finegray_classes$keeper, "ordered")
  expect_s3_class(bridged_finegray_classes[["I(x^2)"]], "AsIs")
  expect_equal(bridged_finegray_classes, reference_finegray_classes)
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
  finegray_zero_probability_data <- data.frame(
    id = 1:4,
    start = c(0, 0, 0, 3),
    stop = c(1, 1.5, 2, 5),
    status = factor(
      c("target", "compete", "censored", "target"),
      levels = c("censored", "target", "compete")
    ),
    x = 1:4
  )
  bridged_finegray_zero_probability <- finegray(
    Surv(start, stop, status) ~ x,
    data = finegray_zero_probability_data,
    id = id
  )
  reference_finegray_zero_probability <- survival::finegray(
    survival::Surv(start, stop, status) ~ x,
    data = finegray_zero_probability_data,
    id = id
  )
  expect_equal(bridged_finegray_zero_probability, reference_finegray_zero_probability)
  expect_true(any(is.nan(bridged_finegray_zero_probability$fgwt)))
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
      survival::Surv(time, status) ~ x + strata(group),
      data = obrien_data,
      transform = obrien_transform
    ),
    survival::survobrien(
      survival::Surv(time, status) ~ x + strata(group),
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
  obrien_factor_data <- transform(
    obrien_data,
    keeper = factor(group, levels = c("b", "a"))
  )
  factor_formula <- survival::Surv(time, status) ~ x + keeper
  bridged_obrien_factor <- survobrien(factor_formula, data = obrien_factor_data)
  reference_obrien_factor <- survival::survobrien(
    factor_formula,
    data = obrien_factor_data
  )
  expect_s3_class(bridged_obrien_factor$keeper, "factor")
  expect_equal(levels(bridged_obrien_factor$keeper), c("b", "a"))
  expect_equal(bridged_obrien_factor, reference_obrien_factor)
  factor_strata_formula <- survival::Surv(time, status) ~ x + keeper + strata(group)
  expect_equal(
    survobrien(factor_strata_formula, data = obrien_factor_data),
    survival::survobrien(factor_strata_formula, data = obrien_factor_data)
  )
  for (wrapper in c("factor", "as.factor")) {
    wrapper_formula <- stats::as.formula(paste0(
      "survival::Surv(time, status) ~ x + ", wrapper, "(group) + strata(group)"
    ))
    expect_equal(
      survobrien(wrapper_formula, data = obrien_factor_data),
      survival::survobrien(wrapper_formula, data = obrien_factor_data)
    )
  }
  obrien_factor_row_names <- data.frame(
    time = c(5, 7, 1, 6),
    status = c(0, 1, 1, 1),
    x = c(0.1, 0.4, 0.2, 0.8),
    group = c("a", "b", "b", "b")
  )
  obrien_factor_row_names$keeper <- factor(obrien_factor_row_names$group)
  row_name_formula <- survival::Surv(time, status) ~ x + keeper + strata(group)
  expect_equal(
    survobrien(row_name_formula, data = obrien_factor_row_names),
    survival::survobrien(row_name_formula, data = obrien_factor_row_names)
  )
  obrien_empty_risk <- data.frame(
    time = c(2, 3, 4),
    status = c(1, 1, 0),
    x = c(0.1, 0.4, 0.2),
    group = c("a", "b", "b")
  )
  obrien_empty_risk$keeper <- factor(obrien_empty_risk$group)
  empty_risk_formula <- survival::Surv(time, status) ~ x + keeper + strata(group)
  expect_equal(
    survobrien(empty_risk_formula, data = obrien_empty_risk),
    survival::survobrien(empty_risk_formula, data = obrien_empty_risk)
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
  obrien_counting_data$keeper <- factor(c("a", "a", "b", "b"))
  counting_factor_formula <- survival::Surv(start, stop, status) ~ x + keeper
  expect_equal(
    survobrien(counting_factor_formula, data = obrien_counting_data),
    survival::survobrien(counting_factor_formula, data = obrien_counting_data)
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
  obrien_counting_strata_data$keeper <- factor(
    obrien_counting_strata_data$group,
    levels = c("b", "a")
  )
  counting_factor_strata_formula <-
    survival::Surv(start, stop, status) ~ x + keeper + strata(group)
  expect_equal(
    survobrien(counting_factor_strata_formula, data = obrien_counting_strata_data),
    survival::survobrien(
      counting_factor_strata_formula,
      data = obrien_counting_strata_data
    )
  )
  obrien_expression_formula <-
    survival::Surv(time, status) ~ x + I(off^2)
  expect_equal(
    survobrien(obrien_expression_formula, data = obrien_factor_data),
    survival::survobrien(obrien_expression_formula, data = obrien_factor_data)
  )
  obrien_combined_formula <-
    survival::Surv(time, status) ~ x + keeper + strata(group) + cluster(id)
  expect_equal(
    survobrien(obrien_combined_formula, data = obrien_factor_data),
    survival::survobrien(obrien_combined_formula, data = obrien_factor_data)
  )
  obrien_na_data <- obrien_factor_data
  obrien_na_data$off[[3L]] <- NA_real_
  expect_equal(
    survobrien(
      obrien_expression_formula,
      data = obrien_na_data,
      subset = time != 2,
      na.action = stats::na.omit
    ),
    survival::survobrien(
      obrien_expression_formula,
      data = obrien_na_data,
      subset = time != 2,
      na.action = stats::na.omit
    )
  )
  obrien_environment_result <- local({
    time <- obrien_data$time
    status <- obrien_data$status
    x <- obrien_data$x
    formula <- survival::Surv(time, status) ~ sqrt(x)
    list(
      bridged = survobrien(formula),
      reference = survival::survobrien(formula)
    )
  })
  expect_equal(
    obrien_environment_result$bridged,
    obrien_environment_result$reference
  )

  obrien_error <- function(expression) {
    tryCatch(force(expression), error = function(condition) conditionMessage(condition))
  }
  invalid_obrien_formulas <- list(
    survival::Surv(time, status) ~ x * off,
    survival::Surv(time, status) ~ x + cluster(id) + cluster(group),
    survival::Surv(time, status) ~ keeper,
    time ~ x,
    survival::Surv(time, status, type = "left") ~ x
  )
  for (invalid_formula in invalid_obrien_formulas) {
    expect_identical(
      obrien_error(survobrien(invalid_formula, data = obrien_factor_data)),
      obrien_error(survival::survobrien(
        invalid_formula,
        data = obrien_factor_data
      ))
    )
  }
  expect_identical(
    obrien_error(survobrien(
      survival::Surv(time, status) ~ x,
      data = obrien_data,
      transform = function(value) value[[1L]]
    )),
    obrien_error(survival::survobrien(
      survival::Surv(time, status) ~ x,
      data = obrien_data,
      transform = function(value) value[[1L]]
    ))
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
  expect_identical(typeof(bridged_condense$event), typeof(reference_condense$event))
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
  condense_strata_levels_data <- transform(
    condense_data,
    site = factor(c("west", "north", "north", "west"), levels = c("north", "south", "west"))
  )
  expect_identical(
    survcondense(
      condense_strata_formula,
      data = condense_strata_levels_data,
      id = id
    ),
    survival::survcondense(
      condense_strata_formula,
      data = condense_strata_levels_data,
      id = id
    )
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
  condense_missing_data <- data.frame(
    id = c(4L, 2L, 3L, 1L, 1L, 1L, 1L, 2L),
    tstart = c(0L, 3L, 2L, 6L, 5L, 1L, 3L, 5L),
    tstop = c(4L, 5L, 6L, 7L, 6L, 3L, 5L, 7L),
    event = c(1L, 1L, 0L, 0L, 0L, 0L, 1L, 1L),
    x = factor(
      c("a", "b", NA, "b", "c", "b", "b", "b"),
      levels = c("a", "c", "b")
    ),
    off = c(0.5, 2, 1, 0.5, 0.5, 0.5, 0.5, 1)
  )
  condense_missing_formula <- Surv(tstart, tstop, event) ~ offset(off) + x
  environment(condense_missing_formula) <- environment(reference_formula)
  expect_identical(
    survcondense(condense_missing_formula, data = condense_missing_data, id = id),
    survival::survcondense(
      condense_missing_formula,
      data = condense_missing_data,
      id = id
    )
  )
  condense_empty_data <- data.frame(
    id = c("b", "a", "c"),
    tstart = c(3L, 3L, 2L),
    tstop = c(7L, 4L, 3L),
    event = c(1L, 0L, 1L),
    x = factor(c(NA, "b", "b"), levels = c("c", "b", "a")),
    off = c(0.5, 1, 0.5),
    wt = c(1, 3, 3)
  )
  condense_empty_formula <- Surv(tstart, tstop, event) ~ offset(off) + x
  environment(condense_empty_formula) <- environment(reference_formula)
  expect_identical(
    survcondense(
      condense_empty_formula,
      data = condense_empty_data,
      weights = wt,
      id = id,
      start = "begin",
      end = "finish",
      event = "outcome"
    ),
    survival::survcondense(
      condense_empty_formula,
      data = condense_empty_data,
      weights = wt,
      id = id,
      start = "begin",
      end = "finish",
      event = "outcome"
    )
  )
  condense_multistate_data <- data.frame(
    id = rep(1:3, each = 2),
    tstart = rep(c(0, 1), 3),
    tstop = rep(c(1, 2), 3),
    state = factor(
      c("a", "censor", "b", "a", "a", "b"),
      levels = c("censor", "a", "b")
    ),
    x = rep(1:3, each = 2)
  )
  condense_multistate_formula <- Surv(tstart, tstop, state) ~ x
  environment(condense_multistate_formula) <- environment(reference_formula)
  expect_equal(
    survcondense(
      condense_multistate_formula,
      data = condense_multistate_data,
      id = id
    ),
    survival::survcondense(
      condense_multistate_formula,
      data = condense_multistate_data,
      id = id
    )
  )
  expect_equal(
    survcondense(
      condense_multistate_formula,
      data = condense_multistate_data,
      id = id,
      start = "begin",
      end = "finish",
      event = "transition"
    ),
    survival::survcondense(
      condense_multistate_formula,
      data = condense_multistate_data,
      id = id,
      start = "begin",
      end = "finish",
      event = "transition"
    )
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

test_that("public helper signatures accept R-style named and positional calls", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  expect_identical(names(formals(is.Surv)), names(formals(survival::is.Surv)))
  expect_identical(
    head(names(formals(survdiff)), 6L),
    names(formals(survival::survdiff))
  )
  expect_identical(
    head(names(formals(basehaz)), 3L),
    names(formals(survival::basehaz))
  )
  expect_identical(
    head(names(formals(cox.zph)), 5L),
    names(formals(survival::cox.zph))
  )
  expect_identical(
    head(names(formals(coxph.detail)), 3L),
    names(formals(survival::coxph.detail))
  )

  data <- data.frame(
    time = 1:8,
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    group = rep(c("control", "treated"), 4),
    x = c(0.1, 0.4, 0.2, 0.8, 1.1, 0.7, 1.5, 1.2),
    z = c(1, 0, 1, 0, 1, 1, 0, 0)
  )
  response <- Surv(data$time, data$status)
  expect_true(is.Surv(x = response))

  keep <- c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE)
  bridged_diff <- survdiff(
    Surv(time, status) ~ group,
    data,
    keep,
    stats::na.omit,
    rho = 0.5,
    timefix = FALSE
  )
  reference_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data,
    keep,
    stats::na.omit,
    rho = 0.5
  )
  bridged_diff_frame <- as.data.frame(bridged_diff)
  expect_equal(bridged_diff_frame$observed, unname(reference_diff$obs), tolerance = 1e-06)
  expect_equal(bridged_diff_frame$expected, unname(reference_diff$exp), tolerance = 1e-06)
  expect_equal(
    bridged_diff_frame$variance,
    unname(diag(reference_diff$var)),
    tolerance = 1e-06
  )
  expect_equal(as.numeric(bridged_diff$statistic), reference_diff$chisq, tolerance = 1e-06)
  expect_equal(as.numeric(bridged_diff$p_value), reference_diff$pvalue, tolerance = 1e-06)

  bridged <- coxph(Surv(time, status) ~ x + z, data = data, eps = 1e-10, max_iter = 50)
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + z,
    data = data,
    control = survival::coxph.control(eps = 1e-10, iter.max = 50),
    x = TRUE,
    y = TRUE
  )
  newdata <- data.frame(x = 0.35, z = 1)
  expect_equal(
    unname(predict(bridged, newdata, type = "lp")),
    unname(stats::predict(reference, newdata, type = "lp")),
    tolerance = 1e-05
  )
  bridged_hazard <- as.data.frame(basehaz(bridged, newdata, FALSE))
  reference_hazard <- survival::basehaz(reference, newdata, FALSE)
  expect_equal(bridged_hazard$time, reference_hazard$time)
  expect_equal(bridged_hazard$cumhaz, reference_hazard$hazard, tolerance = 2e-04)

  bridged_zph <- as.data.frame(cox.zph(bridged, "rank", FALSE, FALSE, FALSE))
  reference_zph <- survival::cox.zph(reference, "rank", FALSE, FALSE, FALSE)
  expect_equal(bridged_zph$name, rownames(reference_zph$table))
  expect_equal(bridged_zph$df, as.integer(reference_zph$table[, "df"]))

  bridged_detail <- coxph.detail(object = bridged, TRUE, "time")
  reference_detail <- survival::coxph.detail(object = reference, TRUE, "time")
  bridged_riskmat <- do.call(
    rbind,
    survivalr:::.result_field(bridged_detail, "riskmat")
  )
  expect_equal(unname(bridged_riskmat), unname(reference_detail$riskmat))
})

test_that("Fitted-model concordance supports joint Cox and survreg comparisons", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1.2, 2.1, 2.8, 3.4, 4.2, 5.0, 6.3, 7.1),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    x = c(0.1, 0.3, 0.2, 0.8, 1.0, 0.7, 1.4, 1.1),
    z = c(1, 0, 1, 0, 1, 1, 0, 0)
  )
  cox_x <- coxph(Surv(time, status) ~ x, data = data, eps = 1e-10, max_iter = 50)
  cox_z <- coxph(Surv(time, status) ~ z, data = data, eps = 1e-10, max_iter = 50)
  cox_x_single <- concordance(cox_x, influence = 1)
  cox_z_single <- concordance(cox_z, influence = 1)
  cox_joint <- concordance(cox_x, cox_z, influence = 3, ranks = TRUE)

  expect_s3_class(cox_joint, "concordance")
  expect_equal(names(cox_joint$concordance), c("cox_x", "cox_z"))
  expect_equal(unname(cox_joint$concordance), c(cox_x_single$concordance, cox_z_single$concordance))
  expect_equal(cox_joint$count[1L, ], cox_x_single$count)
  expect_equal(cox_joint$count[2L, ], cox_z_single$count)
  expect_equal(
    cox_joint$var,
    crossprod(cbind(cox_x_single$dfbeta, cox_z_single$dfbeta)),
    tolerance = 1e-12
  )
  expect_equal(dim(cox_joint$dfbeta), c(nrow(data), 2L))
  expect_equal(dim(cox_joint$influence), c(nrow(data), 5L, 2L))
  expect_equal(dimnames(cox_joint$influence)[[3L]], c("cox_x", "cox_z"))
  expect_equal(unique(cox_joint$ranks$fit), c("cox_x", "cox_z"))
  expect_equal(
    concordance(cox_x, cox_z, newdata = data)$concordance,
    concordance(cox_x, cox_z)$concordance
  )

  weighted_data <- transform(data, wt = seq_len(nrow(data)) / nrow(data) + 0.5)
  weighted_x <- coxph(Surv(time, status) ~ x, data = weighted_data, weights = wt)
  weighted_z <- coxph(Surv(time, status) ~ z, data = weighted_data, weights = wt)
  expect_equal(nrow(concordance(weighted_x, weighted_z)$count), 2L)
  expect_error(concordance(weighted_x, cox_z), "same weight vector")

  survreg_x <- survreg(
    Surv(time, status) ~ x,
    data = data,
    dist = "weibull",
    max_iter = 150,
    eps = 1e-10
  )
  survreg_z <- survreg(
    Surv(time, status) ~ z,
    data = data,
    dist = "weibull",
    max_iter = 150,
    eps = 1e-10
  )
  survreg_joint <- concordance(survreg_x, survreg_z, influence = 1)
  reference_survreg_x <- survival::survreg(
    survival::Surv(time, status) ~ x,
    data = data,
    dist = "weibull"
  )
  reference_survreg_z <- survival::survreg(
    survival::Surv(time, status) ~ z,
    data = data,
    dist = "weibull"
  )
  reference_survreg_joint <- survival::concordance(reference_survreg_x, reference_survreg_z)
  expect_equal(
    unname(survreg_joint$concordance),
    unname(reference_survreg_joint$concordance),
    tolerance = 1e-12
  )
  expect_equal(dim(survreg_joint$var), c(2L, 2L))
  expect_equal(dim(survreg_joint$dfbeta), c(nrow(data), 2L))

  short_fit <- coxph(Surv(time, status) ~ x, data = data[-1L, ], max_iter = 0)
  expect_error(concordance(cox_x, short_fit), "same sample size")
  expect_error(concordance(cox_x, bad = survreg_x), "bad argument is not an appropriate fit object")
})

test_that("Cox time transforms agree with R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  right <- data.frame(
    time = c(5, 1, 9, 3, 12, 7, 2, 10, 4, 11, 6, 8),
    status = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1),
    x1 = c(-0.4, 0.2, 1.1, -0.8, 0.5, 1.4, -1.2, 0.7, 0, -0.3, 0.9, -0.6),
    x2 = c(1.2, -0.5, 0.3, 1.1, -0.9, 0.8, -0.2, 1.5, -1.1, 0.4, -0.7, 0.6)
  )
  log_transform <- function(x, time, riskset, weights) x * log(time)
  bridged_right <- coxph(
    Surv(time, status) ~ x1 + tt(x2),
    data = right,
    tt = log_transform,
    eps = 1e-10,
    max_iter = 50,
    x = TRUE
  )
  reference_right <- survival::coxph(
    Surv(time, status) ~ x1 + tt(x2),
    data = right,
    tt = log_transform,
    control = survival::coxph.control(eps = 1e-10, iter.max = 50),
    x = TRUE
  )

  expect_equal(unname(coef(bridged_right)), unname(coef(reference_right)), tolerance = 1e-10)
  expect_equal(unname(vcov(bridged_right)), unname(vcov(reference_right)), tolerance = 1e-10)
  expect_equal(as.numeric(logLik(bridged_right)), reference_right$loglik[[2L]], tolerance = 1e-10)
  expect_equal(unname(model.matrix(bridged_right)), unname(model.matrix(reference_right)))
  expect_equal(summary(bridged_right)$n, summary(reference_right)$n)
  expect_equal(nobs(bridged_right), nobs(reference_right))
  expect_equal(
    anova(bridged_right),
    survival:::anova.coxph(reference_right),
    tolerance = 2e-08
  )

  default_bridged <- coxph(
    Surv(time, status) ~ x1 + tt(x2),
    data = right,
    eps = 1e-10,
    max_iter = 50
  )
  default_reference <- survival::coxph(
    survival::Surv(time, status) ~ x1 + tt(x2),
    data = right,
    control = survival::coxph.control(eps = 1e-10, iter.max = 50)
  )
  expect_equal(
    unname(coef(default_bridged)),
    unname(coef(default_reference)),
    tolerance = 1e-10
  )

  counting <- data.frame(
    start = c(0, 0, 1, 2, 0, 3, 1, 4, 2, 5),
    stop = c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    status = c(1, 0, 1, 1, 0, 1, 0, 1, 1, 1),
    x1 = c(-0.4, 0.2, 1.1, -0.8, 0.5, 1.4, -1.2, 0.7, 0, -0.3),
    x2 = c(1.2, -0.5, 0.3, 1.1, -0.9, 0.8, -0.2, 1.5, -1.1, 0.4)
  )
  root_transform <- function(x, time, riskset, weights) x * sqrt(time)
  bridged_counting <- coxph(
    Surv(start, stop, status) ~ x1 + tt(x2),
    data = counting,
    tt = root_transform,
    eps = 1e-10,
    max_iter = 50
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x1 + tt(x2),
    data = counting,
    tt = root_transform,
    control = survival::coxph.control(eps = 1e-10, iter.max = 50)
  )

  expect_equal(
    unname(coef(bridged_counting)),
    unname(coef(reference_counting)),
    tolerance = 1e-10
  )
  expect_equal(
    unname(vcov(bridged_counting)),
    unname(vcov(reference_counting)),
    tolerance = 1e-10
  )
  expect_equal(
    as.numeric(logLik(bridged_counting)),
    reference_counting$loglik[[2L]],
    tolerance = 1e-10
  )
  expect_equal(summary(bridged_counting)$n, summary(reference_counting)$n)
  expect_equal(nobs(bridged_counting), nobs(reference_counting))
})

test_that("Cox detail weighted tied-event moments agree with R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 1, 2),
    status = c(1, 1, 0),
    x = c(0, 1, 2),
    weight = c(1, 2, 0.5)
  )
  exact <- list(
    breslow = c(
      means = 6 / 7,
      score = -4 / 7,
      imat = 60 / 49,
      hazard = 6 / 7,
      varhaz = 18 / 49
    ),
    efron = c(
      means = 13 / 14,
      score = -11 / 14,
      imat = 267 / 196,
      hazard = 33 / 28,
      varhaz = 585 / 784
    )
  )

  for (method in names(exact)) {
    bridged <- coxph(
      Surv(time, status) ~ x,
      data = data,
      weights = data$weight,
      max_iter = 0,
      method = method
    )
    reference <- survival::coxph(
      survival::Surv(time, status) ~ x,
      data = data,
      weights = data$weight,
      init = c(0),
      ties = method,
      control = survival::coxph.control(iter.max = 0)
    )
    bridged_detail <- coxph.detail(bridged)
    reference_detail <- survival::coxph.detail(reference)
    bridged_risk_detail <- coxph.detail(bridged, riskmat = TRUE)
    reference_risk_detail <- survival::coxph.detail(reference, riskmat = TRUE)

    for (field in c("means", "score", "imat", "hazard", "varhaz", "wtrisk")) {
      actual <- as.numeric(unlist(survivalr:::.result_field(bridged_detail, field)))
      expected <- as.numeric(reference_detail[[field]])
      expect_equal(actual, expected, tolerance = 1e-12)
    }
    expect_equal(
      as.numeric(unlist(survivalr:::.result_field(bridged_detail, "nevent_wt"))),
      as.numeric(reference_detail[["nevent.wt"]]),
      tolerance = 1e-12
    )
    expect_equal(
      as.numeric(unlist(survivalr:::.result_field(bridged_detail, "nrisk_wt"))),
      as.numeric(reference_detail[["nrisk.wt"]]),
      tolerance = 1e-12
    )
    bridged_riskmat <- survivalr:::.result_field(bridged_risk_detail, "riskmat")
    if (is.list(bridged_riskmat)) {
      bridged_riskmat <- do.call(rbind, bridged_riskmat)
    }
    expect_equal(unname(bridged_riskmat), unname(reference_risk_detail$riskmat))
    for (field in names(exact[[method]])) {
      actual <- as.numeric(unlist(survivalr:::.result_field(bridged_detail, field)))
      expect_equal(actual, unname(exact[[method]][[field]]), tolerance = 1e-12)
    }
  }
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
  no_events_summary <- summary(no_events)
  reference_no_events_summary <- summary(reference_no_events)
  expect_equal(no_events_summary$n, nrow(right))
  expect_equal(no_events_summary$n_event, 0L)
  for (field in c("logtest", "sctest", "waldtest")) {
    expect_equal(no_events_summary[[field]], reference_no_events_summary[[field]])
  }

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

test_that("Cox bridge reports converged aliased coefficients like R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 1, 2, 3, 3, 4, 5, 5),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    x1 = c(0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5)
  )
  data$x2 <- 2 * data$x1

  bridged <- coxph(
    Surv(time, status) ~ x1 + x2,
    data = data,
    max_iter = 50,
    eps = 1e-09,
    toler = 1e-10
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x1 + x2,
    data = data,
    singular.ok = TRUE,
    control = survival::coxph.control(iter.max = 50, eps = 1e-09, toler.chol = 1e-10)
  )

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
  expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
  expect_equal(
    vcov(bridged, complete = FALSE),
    vcov(reference, complete = FALSE),
    tolerance = 1e-12
  )
  expect_equal(confint(bridged), confint(reference), tolerance = 1e-12)
  expect_equal(attr(logLik(bridged), "df"), attr(logLik(reference), "df"))
  expect_equal(
    unname(extractAIC(bridged)),
    unname(extractAIC(reference)),
    tolerance = 1e-12
  )

  bridged_summary <- summary(bridged)$coefficients
  reference_summary <- summary(reference)$coefficients
  expect_equal(
    bridged_summary,
    reference_summary,
    tolerance = 1e-12
  )

  term_predictions <- predict(bridged, type = "terms")
  expect_equal(colnames(term_predictions), c("x1", "x2"))
  expect_true(all(is.finite(term_predictions)))
  expect_equal(term_predictions[, "x2"], rep(0, nrow(data)))

  for (group_terms in c(TRUE, FALSE)) {
    bridged_zph <- as.data.frame(
      cox.zph(bridged, transform = "rank", terms = group_terms)
    )
    reference_zph <- survival::cox.zph(
      reference,
      transform = "rank",
      terms = group_terms
    )
    expect_equal(bridged_zph$name, rownames(reference_zph$table))
    expect_equal(bridged_zph$df, unname(reference_zph$table[, "df"]))
    expect_equal(
      bridged_zph$chisq,
      unname(reference_zph$table[, "chisq"]),
      tolerance = 1e-09
    )
  }
})

test_that("Cox zph bridge remaps partially aliased terms like R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 1:8,
    status = c(1, 1, 0, 1, 0, 1, 1, 0),
    group = factor(c("a", "a", "b", "b", "c", "c", "a", "b")),
    is_b = c(0, 0, 1, 1, 0, 0, 0, 1),
    x = c(1, 0.9, 1.1, 0.7, 0.4, 0.3, 0.6, 0.2)
  )
  bridged <- coxph(
    Surv(time, status) ~ is_b + factor(group) + x,
    data = data,
    max_iter = 50,
    eps = 1e-09,
    toler = 1e-10
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ is_b + factor(group) + x,
    data = data,
    singular.ok = TRUE,
    control = survival::coxph.control(iter.max = 50, eps = 1e-09, toler.chol = 1e-10)
  )

  expect_true(is.na(coef(bridged)[[2L]]))
  expect_true(is.na(coef(reference)[[2L]]))
  for (group_terms in c(TRUE, FALSE)) {
    bridged_zph <- as.data.frame(
      cox.zph(bridged, transform = "rank", terms = group_terms)
    )
    reference_zph <- survival::cox.zph(
      reference,
      transform = "rank",
      terms = group_terms
    )
    expect_equal(bridged_zph$name, rownames(reference_zph$table))
    expect_equal(bridged_zph$df, unname(reference_zph$table[, "df"]))
    expect_equal(
      bridged_zph$chisq,
      unname(reference_zph$table[, "chisq"]),
      tolerance = 1e-09
    )
  }
})

test_that("formula factors retain numeric labels and unused levels", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- survival::lung[
    stats::complete.cases(survival::lung[, c("time", "status", "age", "ph.ecog")]),
    c("time", "status", "age", "ph.ecog")
  ]
  data$status <- as.integer(data$status == 2L)
  data$ph.ecog <- factor(data$ph.ecog, levels = 0:4)
  bridged_cox <- coxph(
    Surv(time, status) ~ ph.ecog + pspline(age),
    data = data
  )
  reference_cox <- survival::coxph(
    survival::Surv(time, status) ~ ph.ecog + pspline(age),
    data = data
  )
  active_cox <- !is.na(coef(reference_cox))

  expect_equal(names(coef(bridged_cox)), names(coef(reference_cox)))
  expect_equal(
    unname(coef(bridged_cox)[active_cox]),
    unname(coef(reference_cox)[active_cox]),
    tolerance = 1e-08
  )
  expect_true(is.na(coef(bridged_cox)[["ph.ecog4"]]))
  expect_equal(
    as.vector(model.matrix(bridged_cox)),
    as.vector(model.matrix(reference_cox)),
    tolerance = 1e-12
  )
  bridged_zph <- as.data.frame(cox.zph(bridged_cox, transform = "rank"))
  reference_zph <- survival::cox.zph(reference_cox, transform = "rank")
  expect_equal(bridged_zph$name, rownames(reference_zph$table))
  expect_equal(
    as.vector(as.matrix(bridged_zph[, c("chisq", "df", "p")])),
    as.vector(reference_zph$table),
    tolerance = 1e-08
  )

  bridged_aft <- survreg(
    Surv(time, status) ~ ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  reference_aft <- survival::survreg(
    survival::Surv(time, status) ~ ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  active_aft <- !is.na(coef(reference_aft))
  expect_equal(names(coef(bridged_aft)), names(coef(reference_aft)))
  expect_equal(
    unname(coef(bridged_aft)[active_aft]),
    unname(coef(reference_aft)[active_aft]),
    tolerance = 1e-08
  )
  expect_true(is.na(coef(bridged_aft)[["ph.ecog4"]]))
  expect_equal(vcov(bridged_aft), vcov(reference_aft), tolerance = 1e-08)
  expect_equal(logLik(bridged_aft), logLik(reference_aft), tolerance = 1e-08)
  expect_equal(
    unname(predict(bridged_aft, type = "lp")),
    unname(stats::predict(reference_aft, type = "lp")),
    tolerance = 1e-08
  )
})

test_that("formula factors retain ordered and custom contrasts", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- survival::lung[
    stats::complete.cases(survival::lung[, c("time", "status", "age", "ph.ecog")]),
    c("time", "status", "age", "ph.ecog")
  ]
  data$status <- as.integer(data$status == 2L)
  data$ph.ecog <- ordered(data$ph.ecog, levels = 0:3)
  bridged_cox <- coxph(
    Surv(time, status) ~ ph.ecog * age,
    data = data
  )
  reference_cox <- survival::coxph(
    survival::Surv(time, status) ~ ph.ecog * age,
    data = data
  )

  expect_equal(names(coef(bridged_cox)), names(coef(reference_cox)))
  expect_equal(coef(bridged_cox), coef(reference_cox), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_cox)),
    as.vector(model.matrix(reference_cox)),
    tolerance = 1e-12
  )

  bridged_aft <- survreg(
    Surv(time, status) ~ ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  reference_aft <- survival::survreg(
    survival::Surv(time, status) ~ ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  expect_equal(names(coef(bridged_aft)), names(coef(reference_aft)))
  expect_equal(coef(bridged_aft), coef(reference_aft), tolerance = 1e-08)
  expect_equal(vcov(bridged_aft), vcov(reference_aft), tolerance = 1e-08)
  expect_equal(logLik(bridged_aft), logLik(reference_aft), tolerance = 1e-08)

  bridged_full_factor <- survreg(
    Surv(time, status) ~ 0 + ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  reference_full_factor <- survival::survreg(
    survival::Surv(time, status) ~ 0 + ph.ecog + age,
    data = data,
    dist = "weibull"
  )
  expect_equal(names(coef(bridged_full_factor)), names(coef(reference_full_factor)))
  expect_equal(coef(bridged_full_factor), coef(reference_full_factor), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_full_factor)),
    as.vector(model.matrix(reference_full_factor)),
    tolerance = 1e-12
  )

  data$ph.ecog <- factor(as.character(data$ph.ecog), levels = 0:3)
  contrasts(data$ph.ecog) <- stats::contr.sum(4L)
  bridged_custom <- coxph(Surv(time, status) ~ ph.ecog + age, data = data)
  reference_custom <- survival::coxph(
    survival::Surv(time, status) ~ ph.ecog + age,
    data = data,
    x = TRUE
  )
  expect_equal(names(coef(bridged_custom)), names(coef(reference_custom)))
  expect_equal(coef(bridged_custom), coef(reference_custom), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_custom)),
    as.vector(reference_custom$x),
    tolerance = 1e-12
  )
})

test_that("implicit formula factors use R level ordering after subsetting", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  row <- seq_len(48L)
  x <- ((row * 11L) %% 43L) / 10 - 2.1
  noise <- ((row * 13L) %% 11L - 5L) / 20
  group <- c("zz_unused", rep(c("z", "a", "m"), length.out = 47L))
  code <- c(99L, rep(c(3L, 1L, 2L), length.out = 47L))
  data <- data.frame(
    time = exp(
      2.3 + 0.2 * x + 0.1 * (group == "z") - 0.15 * (code == 3L) + noise
    ),
    status = as.integer(row %% 5L != 0L),
    group = group,
    code = code,
    x = x
  )
  keep <- row > 1L

  bridged_character <- coxph(
    Surv(time, status) ~ group + x,
    data = data,
    subset = keep,
    ties = "breslow",
    max_iter = 50L,
    eps = 1e-09
  )
  reference_character <- survival::coxph(
    survival::Surv(time, status) ~ group + x,
    data = data,
    subset = keep,
    ties = "breslow",
    x = TRUE,
    control = survival::coxph.control(iter.max = 50L, eps = 1e-09)
  )
  expect_equal(names(coef(bridged_character)), names(coef(reference_character)))
  expect_equal(coef(bridged_character), coef(reference_character), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_character)),
    as.vector(reference_character$x),
    tolerance = 1e-12
  )

  prediction_data <- data.frame(group = c("m", "z", "a"), x = c(-0.7, 0.2, 1.1))
  expect_equal(
    unname(predict(bridged_character, newdata = prediction_data, type = "lp")),
    unname(stats::predict(reference_character, newdata = prediction_data, type = "lp")),
    tolerance = 1e-08
  )

  bridged_factor <- coxph(
    Surv(time, status) ~ factor(code) + x,
    data = data,
    subset = keep,
    ties = "breslow",
    max_iter = 50L,
    eps = 1e-09
  )
  reference_factor <- survival::coxph(
    survival::Surv(time, status) ~ factor(code) + x,
    data = data,
    subset = keep,
    ties = "breslow",
    x = TRUE,
    control = survival::coxph.control(iter.max = 50L, eps = 1e-09)
  )
  expect_equal(names(coef(bridged_factor)), names(coef(reference_factor)))
  expect_equal(coef(bridged_factor), coef(reference_factor), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_factor)),
    as.vector(reference_factor$x),
    tolerance = 1e-12
  )

  bridged_aft <- survreg(
    Surv(time, status) ~ as.factor(code) + x,
    data = data,
    subset = keep,
    dist = "weibull"
  )
  reference_aft <- survival::survreg(
    survival::Surv(time, status) ~ as.factor(code) + x,
    data = data,
    subset = keep,
    dist = "weibull"
  )
  expect_equal(names(coef(bridged_aft)), names(coef(reference_aft)))
  expect_equal(coef(bridged_aft), coef(reference_aft), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_aft)),
    as.vector(model.matrix(reference_aft)),
    tolerance = 1e-12
  )
})

test_that("formula factor arguments match survival model construction", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  row <- seq_len(48L)
  code <- c(9L, rep(c(3L, 1L, 2L), length.out = 47L))
  x <- ((row * 11L) %% 43L) / 10 - 2.1
  noise <- ((row * 13L) %% 11L - 5L) / 20
  data <- data.frame(
    time = exp(2.3 + 0.2 * x + 0.16 * (code == 1L) - 0.12 * (code == 2L) + noise),
    status = as.integer(row %% 5L != 0L),
    code = code,
    x = x
  )
  keep <- row > 1L
  prediction_data <- data.frame(code = c(3L, 1L, 2L), x = c(-0.7, 0.2, 1.1))

  bridged_cox <- coxph(
    Surv(time, status) ~ factor(
      code,
      levels = c(3L, 1L, 2L),
      labels = c("third", "first", "second")
    ) + x,
    data = data,
    subset = keep,
    ties = "breslow",
    max_iter = 50L,
    eps = 1e-09
  )
  reference_cox <- survival::coxph(
    survival::Surv(time, status) ~ factor(
      code,
      levels = c(3L, 1L, 2L),
      labels = c("third", "first", "second")
    ) + x,
    data = data,
    subset = keep,
    ties = "breslow",
    x = TRUE,
    control = survival::coxph.control(iter.max = 50L, eps = 1e-09)
  )
  expect_equal(names(coef(bridged_cox)), names(coef(reference_cox)))
  expect_equal(coef(bridged_cox), coef(reference_cox), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_cox)),
    as.vector(reference_cox$x),
    tolerance = 1e-12
  )
  expect_equal(
    unname(predict(bridged_cox, newdata = prediction_data, type = "lp")),
    unname(stats::predict(reference_cox, newdata = prediction_data, type = "lp")),
    tolerance = 1e-08
  )

  bridged_aft <- survreg(
    Surv(time, status) ~ factor(code, levels = c(1L, 2L, 3L), ordered = TRUE) + x,
    data = data,
    subset = keep,
    dist = "weibull"
  )
  reference_aft <- survival::survreg(
    survival::Surv(time, status) ~ factor(
      code,
      levels = c(1L, 2L, 3L),
      ordered = TRUE
    ) + x,
    data = data,
    subset = keep,
    dist = "weibull"
  )
  expect_equal(names(coef(bridged_aft)), names(coef(reference_aft)))
  expect_equal(coef(bridged_aft), coef(reference_aft), tolerance = 1e-08)
  expect_equal(
    as.vector(model.matrix(bridged_aft)),
    as.vector(model.matrix(reference_aft)),
    tolerance = 1e-12
  )
  expect_equal(
    unname(predict(bridged_aft, newdata = prediction_data, type = "lp")),
    unname(stats::predict(reference_aft, newdata = prediction_data, type = "lp")),
    tolerance = 1e-08
  )
})

test_that("formula factor exclusions and reconstruction match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  row <- seq_len(48L)
  x <- ((row * 7L) %% 41L) / 10 - 2
  group <- rep(c("b", "a", "b", "a"), length.out = 48L)
  group[c(7L, 19L)] <- NA_character_
  data <- data.frame(
    time = exp(2.1 + 0.18 * x + 0.14 * (group == "a") + (row %% 9L - 4L) / 25),
    status = as.integer(row %% 5L != 0L),
    code = rep(c(1L, 2L, 3L, 4L), length.out = 48L),
    group = group,
    x = x
  )
  data$group <- factor(data$group, levels = c("c", "b", "a", "unused"))
  contrasts(data$group) <- stats::contr.sum(4L)

  bridged_excluded <- coxph(
    Surv(time, status) ~ factor(code, levels = c(1L, 2L, 3L), exclude = 3L) + x,
    data = data,
    na.action = na.omit,
    max_iter = 0L,
    model = TRUE
  )
  reference_excluded <- survival::coxph(
    survival::Surv(time, status) ~ factor(
      code,
      levels = c(1L, 2L, 3L),
      exclude = 3L
    ) + x,
    data = data,
    na.action = na.omit,
    iter = 0L,
    x = TRUE,
    model = TRUE
  )
  expect_equal(names(coef(bridged_excluded)), names(coef(reference_excluded)))
  expect_equal(nrow(model.frame(bridged_excluded)), nrow(model.frame(reference_excluded)))
  expect_equal(
    as.vector(model.matrix(bridged_excluded)),
    as.vector(reference_excluded$x),
    tolerance = 1e-12
  )

  for (rhs in c("factor(group) + x", "as.factor(group) + x")) {
    bridged_formula <- stats::as.formula(paste("Surv(time, status) ~", rhs))
    reference_formula <- stats::as.formula(paste("survival::Surv(time, status) ~", rhs))
    bridged <- coxph(
      bridged_formula,
      data = data,
      na.action = na.omit,
      max_iter = 0L
    )
    reference <- survival::coxph(
      reference_formula,
      data = data,
      na.action = na.omit,
      iter = 0L,
      x = TRUE
    )
    expect_equal(names(coef(bridged)), names(coef(reference)))
    expect_equal(
      as.vector(model.matrix(bridged)),
      as.vector(reference$x),
      tolerance = 1e-12
    )
  }

  missing_data <- transform(data, code = replace(code, c(8L, 20L), NA_integer_))
  bridged_missing <- coxph(
    Surv(time, status) ~ factor(code, exclude = NULL) + x,
    data = missing_data,
    na.action = na.omit,
    max_iter = 0L,
    model = TRUE
  )
  reference_missing <- survival::coxph(
    survival::Surv(time, status) ~ factor(code, exclude = NULL) + x,
    data = missing_data,
    na.action = na.omit,
    iter = 0L,
    x = TRUE,
    model = TRUE
  )
  expect_equal(names(coef(bridged_missing)), names(coef(reference_missing)))
  expect_equal(nrow(model.frame(bridged_missing)), nrow(model.frame(reference_missing)))
  expect_equal(
    as.vector(model.matrix(bridged_missing)),
    as.vector(reference_missing$x),
    tolerance = 1e-12
  )

  na_level_data <- data.frame(
    time = exp(2.2 + 0.18 * x + (row %% 9L - 4L) / 25),
    status = as.integer(row %% 5L != 0L),
    group = factor(rep(c("a", NA_character_, "b"), length.out = 48L), exclude = NULL),
    x = x
  )
  for (rhs in c("group + x", "factor(group) + x", "as.factor(group) + x")) {
    bridged_formula <- stats::as.formula(paste("Surv(time, status) ~", rhs))
    reference_formula <- stats::as.formula(paste("survival::Surv(time, status) ~", rhs))
    bridged <- coxph(
      bridged_formula,
      data = na_level_data,
      na.action = na.omit,
      max_iter = 0L,
      model = TRUE
    )
    reference <- survival::coxph(
      reference_formula,
      data = na_level_data,
      na.action = na.omit,
      iter = 0L,
      x = TRUE,
      model = TRUE
    )
    expect_equal(names(coef(bridged)), names(coef(reference)))
    expect_equal(nrow(model.frame(bridged)), nrow(model.frame(reference)))
    expect_equal(
      as.vector(model.matrix(bridged)),
      as.vector(reference$x),
      tolerance = 1e-12
    )
  }
})

test_that("Cox zph bridge preserves scaled variance, strata, and subsetting", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 1:12,
    status = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1),
    x1 = c(0.2, 0.4, 0.1, 0.8, 1, 1.2, 0.6, 1.4, 0.3, 0.9, 1.1, 0.5),
    x2 = c(1, 0.9, 1.1, 0.7, 0.4, 0.3, 0.6, 0.2, 0.8, 0.5, 0.25, 0.65),
    group = rep(c("control", "treated"), each = 6)
  )
  bridged_fit <- coxph(
    Surv(time, status) ~ x1 + x2 + strata(group),
    data = data,
    max_iter = 50,
    eps = 1e-09
  )
  reference_fit <- survival::coxph(
    survival::Surv(time, status) ~ x1 + x2 + strata(group),
    data = data,
    x = TRUE,
    control = survival::coxph.control(iter.max = 50, eps = 1e-09)
  )
  for (transform in c("identity", "log", "rank", "km")) {
    transformed_bridged <- cox.zph(bridged_fit, transform = transform)
    transformed_reference <- survival::cox.zph(reference_fit, transform = transform)
    bridged_table <- as.data.frame(transformed_bridged)

    expect_equal(transformed_bridged$x, transformed_reference$x, tolerance = 1e-12)
    expect_equal(
      bridged_table$chisq,
      unname(transformed_reference$table[, "chisq"]),
      tolerance = 1e-09
    )
    expect_equal(
      bridged_table$p,
      unname(transformed_reference$table[, "p"]),
      tolerance = 1e-09
    )
  }

  custom_transform <- function(value) rank(value)^2
  custom_bridged <- cox.zph(bridged_fit, transform = custom_transform)
  custom_reference <- survival::cox.zph(reference_fit, transform = custom_transform)
  expect_equal(custom_bridged$x, custom_reference$x, tolerance = 1e-12)
  expect_equal(
    as.data.frame(custom_bridged)$chisq,
    unname(custom_reference$table[, "chisq"]),
    tolerance = 1e-09
  )

  data$dose <- factor(
    rep(c("low", "medium", "high"), 4L),
    levels = c("low", "medium", "high")
  )
  grouped_bridged <- as.data.frame(cox.zph(
    coxph(Surv(time, status) ~ dose + x1, data = data, max_iter = 50, eps = 1e-09),
    transform = "rank",
    singledf = TRUE
  ))
  grouped_reference <- survival::cox.zph(
    survival::coxph(
      survival::Surv(time, status) ~ dose + x1,
      data = data,
      x = TRUE,
      control = survival::coxph.control(iter.max = 50, eps = 1e-09)
    ),
    transform = "rank",
    singledf = TRUE
  )
  expect_equal(
    grouped_bridged$chisq,
    unname(grouped_reference$table[, "chisq"]),
    tolerance = 1e-09
  )

  bridged <- cox.zph(bridged_fit, transform = "rank")
  reference <- survival::cox.zph(reference_fit, transform = "rank")

  expect_equal(bridged$strata, as.character(reference$strata))
  expect_equal(do.call(rbind, bridged$var), unname(reference$var), tolerance = 1e-08)

  for (selector in list(c("x2", "x1"), 1L, -1L)) {
    bridged_subset <- bridged[selector]
    reference_subset <- reference[selector]
    expect_s3_class(bridged_subset, "survival_py_cox_zph")
    expect_equal(
      bridged_subset$variable_names,
      colnames(reference_subset$y)
    )
    expect_equal(
      do.call(rbind, bridged_subset$y),
      unname(reference_subset$y),
      tolerance = 1e-08
    )
    expect_equal(
      do.call(rbind, bridged_subset$var),
      unname(reference_subset$var),
      tolerance = 1e-08
    )
    expect_equal(bridged_subset$strata, as.character(reference_subset$strata))
    expect_equal(
      as.data.frame(bridged_subset)$name,
      rownames(reference_subset$table)
    )
  }

  for (selector in list(1L, "x2")) {
    bridged_curve <- plot(
      bridged,
      var = selector,
      df = 3,
      nsmo = 17,
      plot = FALSE
    )
    reference_curve <- plot(
      reference,
      var = selector,
      df = 3,
      nsmo = 17,
      plot = FALSE
    )
    expect_equal(bridged_curve$x, reference_curve$x, tolerance = 1e-12)
    expect_equal(bridged_curve$y, reference_curve$y, tolerance = 1e-08)
  }

  no_band <- plot(bridged, var = 1L, se = FALSE, plot = FALSE)
  expect_length(no_band$x, 40L)
  expect_identical(dim(no_band$y), c(40L, 1L))
  expect_error(plot(bridged, var = "missing", plot = FALSE), "Invalid variable requested")
  expect_error(plot(bridged, hr = "yes", plot = FALSE), "hr parameter must be TRUE/FALSE")

  plot_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_file)
  plot_result <- tryCatch(
    plot(bridged[1L], resid = TRUE, se = TRUE, hr = TRUE),
    finally = {
      grDevices::dev.off()
      unlink(plot_file)
    }
  )
  expect_null(plot_result)

  expect_error(bridged["missing"], "invalid variable requested")
  expect_error(bridged[3], "invalid variable requested")

  data$subject <- rep(letters[1:6], each = 2)
  clustered_bridged <- cox.zph(coxph(
    Surv(time, status) ~ x1 + x2 + cluster(subject),
    data = data,
    max_iter = 50,
    eps = 1e-09
  ))
  clustered_reference <- survival::cox.zph(survival::coxph(
    survival::Surv(time, status) ~ x1 + x2 + survival::cluster(subject),
    data = data,
    x = TRUE,
    control = survival::coxph.control(iter.max = 50, eps = 1e-09)
  ))
  expect_equal(
    do.call(rbind, clustered_bridged$var),
    unname(clustered_reference$var),
    tolerance = 1e-08
  )
})

test_that("Cox zph score tests preserve tied, weighted, and counting-process risk sets", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  compare_zph <- function(bridged_fit, reference_fit) {
    for (transform in c("rank", "km")) {
      bridged <- cox.zph(bridged_fit, transform = transform)
      reference <- survival::cox.zph(reference_fit, transform = transform)
      bridged_table <- as.data.frame(bridged)
      expect_equal(bridged$x, reference$x, tolerance = 1e-12)
      expect_equal(
        bridged_table$chisq,
        unname(reference$table[, "chisq"]),
        tolerance = 1e-09
      )
      expect_equal(
        bridged_table$p,
        unname(reference$table[, "p"]),
        tolerance = 1e-09
      )
    }
  }

  tied <- data.frame(
    time = c(1, 1, 2, 3, 3, 4, 5, 6),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    x1 = c(0.2, 0.8, 0.4, 1.1, 0.3, 0.7, 1.4, 0.1),
    x2 = c(1, 0.5, 0.9, 0.2, 0.8, 0.4, 0.1, 0.7),
    weight = c(1, 2, 0.5, 1.5, 0.75, 1, 2, 0.5)
  )
  for (method in c("breslow", "efron", "exact")) {
    case_weights <- if (identical(method, "exact")) NULL else tied$weight
    compare_zph(
      coxph(
        Surv(time, status) ~ x1 + x2,
        data = tied,
        weights = case_weights,
        ties = method,
        max_iter = 100,
        eps = 1e-10
      ),
      survival::coxph(
        survival::Surv(time, status) ~ x1 + x2,
        data = tied,
        weights = case_weights,
        ties = method,
        x = TRUE,
        control = survival::coxph.control(iter.max = 100, eps = 1e-10)
      )
    )
  }

  counting <- data.frame(
    start = c(0, 0, 1, 0, 2, 0, 1, 3),
    stop = c(2, 3, 4, 5, 6, 4, 5, 7),
    status = c(1, 0, 1, 1, 0, 1, 0, 1),
    x1 = c(0.2, 0.8, 0.4, 1.1, 0.3, 0.7, 1.4, 0.1),
    x2 = c(1, 0.5, 0.9, 0.2, 0.8, 0.4, 0.1, 0.7),
    group = rep(c("a", "b"), each = 4)
  )
  compare_zph(
    coxph(
      Surv(start, stop, status) ~ x1 + x2 + strata(group),
      data = counting,
      ties = "efron",
      max_iter = 100,
      eps = 1e-10
    ),
    survival::coxph(
      survival::Surv(start, stop, status) ~ x1 + x2 + survival::strata(group),
      data = counting,
      ties = "efron",
      x = TRUE,
      control = survival::coxph.control(iter.max = 100, eps = 1e-10)
    )
  )
})

test_that("survreg.fit matches built-in low-level fits", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- cbind(
    `(Intercept)` = 1,
    x = c(-1, 0, 1, 2, 3, 4)
  )
  y <- unclass(survival::Surv(
    c(1.2, 2.1, 3.0, 4.5, 6.2, 8.1),
    c(1, 1, 0, 1, 1, 0)
  ))
  weights <- rep(1, nrow(x))
  offset <- rep(0, nrow(x))
  control <- survreg.control(maxiter = 150, rel.tolerance = 1e-10)
  reference_control <- survival::survreg.control(maxiter = 150, rel.tolerance = 1e-10)

  compare_fit <- function(dist, scale = 0, parms = NULL, tolerance = 1e-5) {
    bridged <- survreg.fit(
      x, y, weights, offset, NULL, control, dist,
      scale = scale, nstrat = 1, parms = parms
    )
    reference <- survival::survreg.fit(
      x, y, weights, offset, NULL, reference_control, dist,
      scale = scale, nstrat = 1, parms = parms
    )
    expect_equal(names(bridged), names(reference))
    expect_equal(bridged$coefficients, reference$coefficients, tolerance = tolerance)
    expect_equal(bridged$icoef, reference$icoef, tolerance = tolerance)
    expect_equal(bridged$var, reference$var, tolerance = tolerance)
    expect_equal(bridged$loglik, reference$loglik, tolerance = tolerance)
    expect_equal(bridged$linear.predictors, reference$linear.predictors, tolerance = tolerance)
    expect_equal(bridged$df, reference$df)
    expect_lt(max(abs(bridged$score)), 1e-5)
  }

  for (distribution in c("gaussian", "logistic", "extreme")) {
    compare_fit(distribution)
  }
  compare_fit("gaussian", scale = 1.5)
  compare_fit(survreg.distributions$t, parms = 5, tolerance = 1e-4)

  stratified_x <- cbind(
    `(Intercept)` = 1,
    x = c(-2, -1, 0, 1, -2, -1, 0, 1)
  )
  stratified_y <- unclass(survival::Surv(
    c(1, 2, 3, 5, 2, 4, 6, 9),
    c(1, 1, 0, 1, 1, 0, 1, 1)
  ))
  strata <- rep(1:2, each = 4)
  bridged_stratified <- survreg.fit(
    stratified_x, stratified_y, rep(1, 8), rep(0, 8), NULL,
    control, "gaussian", nstrat = 2, strata = strata
  )
  reference_stratified <- survival::survreg.fit(
    stratified_x, stratified_y, rep(1, 8), rep(0, 8), NULL,
    reference_control, "gaussian", nstrat = 2, strata = strata
  )
  expect_equal(
    bridged_stratified$coefficients,
    reference_stratified$coefficients,
    tolerance = 1e-5
  )
  expect_equal(bridged_stratified$var, reference_stratified$var, tolerance = 1e-5)
  expect_equal(bridged_stratified$loglik, reference_stratified$loglik, tolerance = 1e-5)

  interval_x <- cbind(`(Intercept)` = 1, x = c(0.2, 0.4, 0.1, 0.8, 1.0))
  interval_y <- cbind(
    time = c(1, 2, 3, 4, 5),
    time2 = c(1, 2, 3, 4.5, 5),
    status = c(1, 2, 0, 3, 1)
  )
  bridged_interval <- survreg.fit(
    interval_x, interval_y, rep(1, 5), rep(0, 5), NULL,
    control, "gaussian"
  )
  reference_interval <- survival::survreg.fit(
    interval_x, interval_y, rep(1, 5), rep(0, 5), NULL,
    reference_control, "gaussian"
  )
  expect_equal(bridged_interval$coefficients, reference_interval$coefficients, tolerance = 1e-5)
  expect_equal(bridged_interval$var, reference_interval$var, tolerance = 1e-5)
  expect_equal(bridged_interval$loglik, reference_interval$loglik, tolerance = 1e-5)
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

test_that("nsk survreg formulas retain their fitted basis", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 1:12,
    status = c(1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    z = c(-1, -0.5, 0, 0.5, 1, 1.5, -1.2, 0.2, 0.8, 1.1, -0.8, 0.4),
    x = c(1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1)
  )
  newdata <- data.frame(z = c(0.3, -0.2), x = c(0.8, 1.6))
  bridged <- survreg(
    Surv(time, status) ~ z + nsk(x, df = 3),
    data = data,
    dist = "weibull",
    max_iter = 100L
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ z + nsk(x, df = 3),
    data = data,
    dist = "weibull",
    control = survival::survreg.control(maxiter = 100L)
  )

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-08)
  expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-08)
  expect_equal(
    unname(model.matrix(bridged)),
    unname(model.matrix(reference)),
    tolerance = 1e-12
  )
  expect_equal(
    unname(predict(bridged, newdata, type = "lp")),
    unname(stats::predict(reference, newdata, type = "lp")),
    tolerance = 1e-08
  )
})

test_that("penalized survreg model metadata matches survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  n <- 48L
  index <- 0:(n - 1L)
  x <- ((index * 17L) %% 49L) / 10 - 2.4
  z <- ((index * 7L) %% 47L) / 13 - 1.8
  noise <- ((index * 13L) %% 11L - 5L) / 20
  data <- data.frame(
    time = exp(2.4 + 0.22 * x - 0.08 * x^2 + 0.15 * z + noise),
    status = as.integer(index %% 5L != 0L),
    x = x,
    z = z
  )
  bridged <- survreg(
    Surv(time, status) ~ pspline(x, theta = 0.5, nterm = 6) + z,
    data = data,
    control = survreg.control(maxiter = 100L, outer.max = 25L)
  )
  reference <- survival::survreg(
    survival::Surv(time, status) ~ survival::pspline(x, theta = 0.5, nterm = 6) + z,
    data = data,
    control = survival::survreg.control(maxiter = 100L, outer.max = 25L)
  )

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-08)
  expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-08)
  expect_equal(bridged$df, reference$df, tolerance = 1e-08)
  expect_equal(summary(bridged)$df, summary(reference)$df, tolerance = 1e-08)
  expect_equal(logLik(bridged), logLik(reference), tolerance = 1e-08)
  expect_type(attr(logLik(bridged), "df"), "double")
  expect_equal(df.residual(bridged), df.residual(reference), tolerance = 1e-08)
  expect_type(df.residual(bridged), "double")
  expect_equal(extractAIC(bridged), extractAIC(reference), tolerance = 1e-08)
  expect_equal(AIC(bridged), AIC(reference), tolerance = 1e-08)
})

test_that("coxph analysis of deviance matches survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 18),
    status = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1),
    x = c(-1, .2, .5, 1.2, -.7, .8, 1.7, -1.3, .4, 1.4, -.2, .9),
    z = c(0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1),
    group = factor(rep(c("a", "b"), 6L)),
    offset = seq(.1, 1.2, .1),
    weights = rep(c(1L, 2L), 6L)
  )
  formula <- Surv(time, status) ~ x + z
  bridged <- coxph(formula, data = data)
  reference <- survival::coxph(formula, data = data)

  expect_equal(
    anova(bridged),
    survival:::anova.coxph(reference),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged, test = NULL),
    survival:::anova.coxph(reference, test = NULL),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged, test = "none"),
    survival:::anova.coxph(reference, test = "none"),
    tolerance = 2e-08
  )

  tied_data <- data
  tied_data$time <- rep(seq_len(nrow(data) / 2L), each = 2L)
  bridged_exact <- coxph(formula, data = tied_data, ties = "exact")
  reference_exact <- survival::coxph(formula, data = tied_data, ties = "exact")
  expect_equal(
    anova(bridged_exact),
    survival:::anova.coxph(reference_exact),
    tolerance = 2e-07
  )

  counting_data <- data.frame(
    start = c(0, 2, 0, 3, 0, 4, 0, 2, 0, 5, 0, 3),
    stop = c(2, 6, 3, 7, 4, 9, 2, 8, 5, 10, 3, 11),
    status = rep(c(0, 1), 6L),
    x = data$x,
    group = data$group
  )
  counting_formula <- Surv(start, stop, status) ~ x + strata(group)
  bridged_counting <- coxph(counting_formula, data = counting_data)
  reference_counting <- survival::coxph(counting_formula, data = counting_data)
  expect_equal(
    anova(bridged_counting),
    survival:::anova.coxph(reference_counting),
    tolerance = 2e-08
  )

  x_formula <- Surv(time, status) ~ x
  bridged_x <- coxph(x_formula, data = data)
  reference_x <- survival::coxph(x_formula, data = data)
  expect_equal(
    anova(bridged, bridged_x),
    survival:::anova.coxph(reference, reference_x),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged_x, bridged),
    survival:::anova.coxph(reference_x, reference),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged_x, bridged, test = NULL),
    survival:::anova.coxph(reference_x, reference, test = NULL),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged_x, bridged, test = "none"),
    survival:::anova.coxph(reference_x, reference, test = "none"),
    tolerance = 2e-08
  )

  response_data <- transform(data, other_time = time)
  other_formula <- Surv(other_time, status) ~ x
  bridged_other <- coxph(other_formula, data = response_data)
  reference_other <- survival::coxph(other_formula, data = response_data)
  expect_warning(
    bridged_filtered <- anova(bridged, bridged_other),
    "response.*removed"
  )
  reference_filtered <- suppressWarnings(
    survival:::anova.coxph(reference, reference_other)
  )
  expect_equal(bridged_filtered, reference_filtered, tolerance = 2e-08)

  bridged_weighted <- coxph(formula, data = data, weights = weights)
  reference_weighted <- survival::coxph(formula, data = data, weights = weights)
  expect_equal(
    anova(bridged_weighted),
    survival:::anova.coxph(reference_weighted),
    tolerance = 2e-08
  )

  offset_formula <- Surv(time, status) ~ x + z + offset(offset)
  for (keep_x in c(FALSE, TRUE)) {
    bridged_offset <- coxph(offset_formula, data = data, x = keep_x)
    reference_offset <- survival::coxph(offset_formula, data = data, x = keep_x)
    expect_equal(
      anova(bridged_offset),
      survival:::anova.coxph(reference_offset),
      tolerance = 2e-08
    )
  }

  strata_formula <- Surv(time, status) ~ x + strata(group) + z + offset(offset)
  bridged_strata <- coxph(strata_formula, data = data)
  reference_strata <- survival::coxph(strata_formula, data = data)
  expect_equal(
    anova(bridged_strata),
    survival:::anova.coxph(reference_strata),
    tolerance = 2e-08
  )

  penalty_formula <- Surv(time, status) ~ pspline(x, theta = .5, nterm = 4) + z
  bridged_penalty <- coxph(penalty_formula, data = data)
  reference_penalty <- survival::coxph(penalty_formula, data = data)
  expect_equal(
    suppressWarnings(anova(bridged_penalty)),
    suppressWarnings(survival:::anova.coxph(reference_penalty)),
    tolerance = 2e-08
  )

  frailty_data <- data.frame(
    time = 2:19,
    status = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    x = c(-1.2, -.8, -.4, 0, .4, .8, 1.2, -1, -.6, -.2, .2, .6, 1, 1.4, -1.4, -.9, .1, .9),
    z = rep(c(0, 1), 9L),
    group = rep(letters[1:6], each = 3L)
  )
  frailty_formula <- Surv(time, status) ~ x +
    frailty(group, distribution = "gaussian", theta = .5, sparse = TRUE) + z
  bridged_frailty <- coxph(frailty_formula, data = frailty_data)
  reference_frailty <- survival::coxph(frailty_formula, data = frailty_data)
  expect_equal(
    anova(bridged_frailty),
    survival:::anova.coxph(reference_frailty),
    tolerance = 2e-08
  )

  robust <- coxph(
    Surv(time, status) ~ x + z + cluster(group),
    data = data
  )
  expect_error(anova(robust), "robust variances")
  expect_warning(
    anova(bridged, ignored = bridged_x),
    "invalid and dropped"
  )
})

test_that("survreg analysis of deviance matches survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  n <- 48L
  index <- 0:(n - 1L)
  x <- ((index * 17L) %% 49L) / 10 - 2.4
  z <- ((index * 7L) %% 47L) / 13 - 1.8
  noise <- ((index * 13L) %% 11L - 5L) / 20
  data <- data.frame(
    time = exp(2.4 + 0.22 * x - 0.08 * x^2 + 0.15 * z + noise),
    status = as.integer(index %% 5L != 0L),
    x = x,
    z = z
  )
  null_formula <- Surv(time, status) ~ 1
  spline_formula <- Surv(time, status) ~ pspline(x, theta = 0.5, nterm = 6)
  full_formula <- Surv(time, status) ~ pspline(x, theta = 0.5, nterm = 6) + z
  bridged_null <- survreg(null_formula, data = data)
  bridged_spline <- survreg(
    spline_formula,
    data = data,
    control = survreg.control(maxiter = 100L, outer.max = 25L)
  )
  bridged_full <- survreg(
    full_formula,
    data = data,
    control = survreg.control(maxiter = 100L, outer.max = 25L)
  )
  reference_control <- survival::survreg.control(maxiter = 100L, outer.max = 25L)
  reference_null <- do.call(
    survival::survreg,
    list(formula = null_formula, data = data)
  )
  reference_spline <- do.call(
    survival::survreg,
    list(formula = spline_formula, data = data, control = reference_control)
  )
  reference_full <- do.call(
    survival::survreg,
    list(formula = full_formula, data = data, control = reference_control)
  )

  expect_equal(anova(bridged_full), anova(reference_full), tolerance = 2e-08)
  expect_equal(
    anova(bridged_full, test = "none"),
    anova(reference_full, test = "none"),
    tolerance = 2e-08
  )
  expect_equal(
    anova(bridged_null, bridged_spline, bridged_full),
    anova(reference_null, reference_spline, reference_full),
    tolerance = 2e-08
  )

  adaptive_terms <- c(
    "pspline(x, df = 4, nterm = 6)",
    "pspline(x, df = 0, nterm = 6)",
    "ridge(x, df = .7, eps = 1e-8)"
  )
  for (adaptive_term in adaptive_terms) {
    adaptive_formula <- as.formula(paste(
      "Surv(time, status) ~",
      adaptive_term,
      "+ z"
    ))
    bridged_adaptive <- survreg(
      adaptive_formula,
      data = data,
      control = survreg.control(maxiter = 100L, outer.max = 25L)
    )
    reference_adaptive <- do.call(
      survival::survreg,
      list(
        formula = adaptive_formula,
        data = data,
        control = reference_control
      )
    )
    expect_equal(
      anova(bridged_adaptive),
      anova(reference_adaptive),
      tolerance = 3e-08
    )
  }

  data$group <- factor(rep(c("a", "b"), each = n / 2L))
  strata_formula <- Surv(time, status) ~ x + z + strata(group)
  bridged_strata <- survreg(strata_formula, data = data)
  reference_strata <- do.call(
    survival::survreg,
    list(formula = strata_formula, data = data)
  )
  expect_equal(
    anova(bridged_strata),
    anova(reference_strata),
    tolerance = 2e-08
  )
  expect_error(anova(bridged_full, test = "F"), "arg.*Chisq.*none")
})

test_that("Cox survfit quantiles preserve curve dimensions and conditioning", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(7073)
  n <- 180L
  x <- stats::rnorm(n)
  z <- stats::rnorm(n)
  group <- factor(rep(c("a", "b", "c"), each = n / 3L))
  rate <- exp(0.3 * x - 0.2 * z + c(a = -0.2, b = 0, c = 0.25)[group]) / 8
  event_time <- stats::rexp(n, rate)
  censor_time <- stats::rexp(n, 1 / 14)
  data <- data.frame(
    time = pmin(event_time, censor_time),
    status = as.integer(event_time <= censor_time),
    x = x,
    z = z,
    group = group
  )
  profiles <- data.frame(
    x = c(-0.8, 0.1, 1.0),
    z = c(0.5, 0, -0.5),
    group = factor(c("a", "b", "c"), levels = levels(group)),
    row.names = c("low", "middle", "high")
  )
  probabilities <- c(0, 0.25, 0.5, 0.75)

  bridged <- coxph(Surv(time, status) ~ x + z, data = data)
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + z,
    data = data
  )
  bridged_stratified <- coxph(
    Surv(time, status) ~ x + z + strata(group),
    data = data
  )
  reference_stratified <- survival::coxph(
    survival::Surv(time, status) ~ x + z + strata(group),
    data = data
  )

  compare_quantiles <- function(bridged_curve, reference_curve) {
    for (include_confidence in c(FALSE, TRUE)) {
      expect_equal(
        quantile(
          bridged_curve,
          probs = probabilities,
          conf.int = include_confidence,
          scale = 2
        ),
        quantile(
          reference_curve,
          probs = probabilities,
          conf.int = include_confidence,
          scale = 2
        ),
        tolerance = 2e-06
      )
    }
    expect_equal(
      median(bridged_curve),
      median(reference_curve),
      tolerance = 2e-06
    )
  }

  compare_quantiles(
    survfit(bridged, newdata = profiles),
    survival::survfit(reference, newdata = profiles)
  )
  compare_quantiles(
    survfit(bridged, newdata = profiles, start.time = 2.5),
    survival::survfit(reference, newdata = profiles, start.time = 2.5)
  )
  compare_quantiles(
    survfit(bridged, newdata = profiles[1L, , drop = FALSE], start.time = 2.5),
    survival::survfit(
      reference,
      newdata = profiles[1L, , drop = FALSE],
      start.time = 2.5
    )
  )
  compare_quantiles(
    survfit(bridged_stratified),
    survival::survfit(reference_stratified)
  )
  compare_quantiles(
    survfit(bridged_stratified, newdata = profiles, start.time = 2.5),
    survival::survfit(
      reference_stratified,
      newdata = profiles,
      start.time = 2.5
    )
  )

  expect_equal(
    quantile(
      survfit(bridged, newdata = profiles, se.fit = FALSE),
      probs = probabilities,
      conf.int = TRUE
    ),
    quantile(
      survival::survfit(reference, newdata = profiles, se.fit = FALSE),
      probs = probabilities,
      conf.int = TRUE
    ),
    tolerance = 2e-06
  )
})

test_that("Cox survfit data-margin subsetting matches survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(8081)
  n <- 150L
  x <- stats::rnorm(n)
  z <- stats::rnorm(n)
  event_time <- stats::rexp(n, exp(0.3 * x - 0.2 * z) / 8)
  censor_time <- stats::rexp(n, 1 / 14)
  data <- data.frame(
    time = pmin(event_time, censor_time),
    status = as.integer(event_time <= censor_time),
    x = x,
    z = z
  )
  profiles <- data.frame(
    x = c(-1, -0.2, 0.5, 1.2),
    z = c(0.5, 0.1, -0.2, -0.6),
    row.names = c("low", "middle-low", "middle-high", "high")
  )
  bridged_model <- coxph(Surv(time, status) ~ x + z, data = data)
  reference_model <- survival::coxph(
    survival::Surv(time, status) ~ x + z,
    data = data
  )
  bridged <- survfit(
    bridged_model,
    newdata = profiles,
    start.time = 2.5
  )
  reference <- survival::survfit(
    reference_model,
    newdata = profiles,
    start.time = 2.5
  )

  compare_subset <- function(selector, drop = TRUE) {
    bridged_subset <- bridged[selector, drop = drop]
    reference_subset <- reference[selector, drop = drop]
    expect_identical(dim(bridged_subset), dim(reference_subset))
    expect_equal(
      as.numeric(bridged_subset$surv),
      as.numeric(reference_subset$surv),
      tolerance = 2e-06
    )
    expect_equal(
      as.numeric(bridged_subset$cumhaz),
      as.numeric(reference_subset$cumhaz),
      tolerance = 2e-06
    )
    for (include_confidence in c(FALSE, TRUE)) {
      expect_equal(
        quantile(
          bridged_subset,
          probs = c(0, 0.25, 0.5, 0.75),
          conf.int = include_confidence
        ),
        quantile(
          reference_subset,
          probs = c(0, 0.25, 0.5, 0.75),
          conf.int = include_confidence
        ),
        tolerance = 2e-06
      )
    }
  }

  compare_subset(1L)
  compare_subset(c(4L, 1L, 4L))
  compare_subset(c(TRUE, FALSE, TRUE, FALSE))
  compare_subset(integer())
  compare_subset(2L, drop = FALSE)

  expect_identical(bridged[], bridged)
  expect_error(bridged["low"], "no 'dimnames' attribute")
  expect_error(bridged[5L], "subscript out of bounds")

  bridged_no_se <- survfit(bridged_model, newdata = profiles, se.fit = FALSE)
  reference_no_se <- survival::survfit(
    reference_model,
    newdata = profiles,
    se.fit = FALSE
  )
  expect_equal(
    quantile(bridged_no_se[FALSE], probs = c(0.25, 0.5), conf.int = TRUE),
    quantile(reference_no_se[FALSE], probs = c(0.25, 0.5), conf.int = TRUE)
  )
})

test_that("stratified Cox survfit subsetting matches survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  set.seed(8082)
  n <- 180L
  group <- factor(rep(c("A", "B"), each = n / 2L))
  x <- stats::rnorm(n)
  z <- stats::rnorm(n)
  rate <- exp(0.25 * x - 0.15 * z + ifelse(group == "B", 0.35, 0)) / 9
  event_time <- stats::rexp(n, rate)
  censor_time <- stats::rexp(n, 1 / 16)
  data <- data.frame(
    time = pmin(event_time, censor_time),
    status = as.integer(event_time <= censor_time),
    group = group,
    x = x,
    z = z
  )
  profiles <- data.frame(
    x = c(-0.8, 0.1, 0.6, 1.1),
    z = c(0.4, 0.1, -0.2, -0.5),
    group = factor(c("A", "A", "B", "B"), levels = levels(group)),
    row.names = c("a-low", "a-high", "b-low", "b-high")
  )
  bridged_model <- coxph(
    Surv(time, status) ~ x + z + strata(group),
    data = data
  )
  reference_model <- survival::coxph(
    survival::Surv(time, status) ~ x + z + strata(group),
    data = data
  )
  bridged <- survfit(
    bridged_model,
    newdata = profiles,
    start.time = 2.5
  )
  reference <- survival::survfit(
    reference_model,
    newdata = profiles,
    start.time = 2.5
  )

  compare_subset <- function(selector, drop = TRUE) {
    bridged_subset <- bridged[selector, drop = drop]
    reference_subset <- reference[selector, drop = drop]
    expect_identical(dim(bridged_subset), dim(reference_subset))
    for (field in c("time", "surv", "cumhaz", "std.err", "std.chaz", "lower", "upper")) {
      expect_equal(
        as.numeric(bridged_subset[[field]]),
        as.numeric(reference_subset[[field]]),
        tolerance = 2e-06,
        info = paste("stratified Cox subset field", field)
      )
    }
    for (include_confidence in c(FALSE, TRUE)) {
      expect_equal(
        quantile(
          bridged_subset,
          probs = c(0, 0.25, 0.5, 0.75),
          conf.int = include_confidence
        ),
        quantile(
          reference_subset,
          probs = c(0, 0.25, 0.5, 0.75),
          conf.int = include_confidence
        ),
        tolerance = 2e-06
      )
    }
    expect_null(bridged_subset$start.time)
    expect_null(reference_subset$start.time)
  }

  expect_identical(dim(bridged), dim(reference))
  expect_equal(bridged$time, reference$time, tolerance = 2e-06)
  expect_equal(bridged$surv, reference$surv, tolerance = 2e-06)
  expect_equal(
    quantile(bridged, probs = c(0, 0.5), conf.int = TRUE),
    quantile(reference, probs = c(0, 0.5), conf.int = TRUE),
    tolerance = 2e-06
  )

  bridged_default <- survfit(bridged_model)
  reference_default <- survival::survfit(reference_model)
  expect_identical(dim(bridged_default), dim(reference_default))
  expect_equal(bridged_default$time, reference_default$time, tolerance = 2e-06)
  expect_equal(bridged_default$surv, reference_default$surv, tolerance = 2e-06)
  bridged_default_subset <- bridged_default[c("B", "A", "B")]
  reference_default_subset <- reference_default[c("B", "A", "B")]
  expect_identical(dim(bridged_default_subset), dim(reference_default_subset))
  expect_equal(
    bridged_default_subset$time,
    reference_default_subset$time,
    tolerance = 2e-06
  )
  expect_equal(
    bridged_default_subset$surv,
    reference_default_subset$surv,
    tolerance = 2e-06
  )
  expect_equal(
    quantile(bridged_default_subset, probs = c(0, 0.5), conf.int = TRUE),
    quantile(reference_default_subset, probs = c(0, 0.5), conf.int = TRUE),
    tolerance = 2e-06
  )

  compare_subset(1L)
  compare_subset(c(4L, 1L, 4L))
  compare_subset(c(TRUE, FALSE, TRUE, FALSE))
  compare_subset("a-high")
  compare_subset(2L, drop = FALSE)

  bridged_empty <- bridged[integer()]
  reference_empty <- reference[integer()]
  expect_identical(dim(bridged_empty), dim(reference_empty))
  expect_equal(bridged_empty$time, reference_empty$time)
  expect_error(quantile(bridged_empty), "invalid 'times' argument", fixed = TRUE)
  expect_error(quantile(reference_empty), "invalid 'times' argument", fixed = TRUE)
  expect_error(bridged[5L], "strata 5 not matched")
  expect_error(reference[5L], "strata 5 not matched")
  expect_error(bridged["missing"], "strata missing not matched")
  expect_error(reference["missing"], "strata missing not matched")
})

test_that("Cox survfit counts and structural metadata match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = c(1, 2, 4, 5, 1, 3, 4, 6),
    status = c(1, 1, 0, 1, 0, 1, 1, 0),
    group = factor(rep(c("A", "B"), each = 4L)),
    x = c(0.2, 0.4, 0.1, 0.3, 1, 1.2, 0.8, 1.1),
    weight = c(1, 2, 1, 1, 1, 1.5, 2, 1)
  )
  bridged_unstratified <- coxph(
    Surv(time, status) ~ x,
    data = data,
    weights = weight,
    max_iter = 0
  )
  reference_unstratified <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = data,
    weights = weight,
    init = 0,
    iter.max = 0
  )
  bridged_stratified <- coxph(
    Surv(time, status) ~ x + strata(group),
    data = data,
    weights = weight,
    max_iter = 0
  )
  reference_stratified <- survival::coxph(
    survival::Surv(time, status) ~ x + strata(group),
    data = data,
    weights = weight,
    init = 0,
    iter.max = 0
  )

  compare_structure <- function(bridged, reference) {
    expect_identical(dim(bridged), dim(reference))
    expect_identical(dim(bridged$surv), dim(reference$surv))
    expect_identical(dim(bridged$cumhaz), dim(reference$cumhaz))
    for (field in c("n", "time", "n.risk", "n.event", "n.censor", "strata")) {
      expect_equal(
        bridged[[field]],
        reference[[field]],
        tolerance = 1e-12,
        info = paste("Cox survfit structural field", field)
      )
    }
    expect_equal(bridged$surv, reference$surv, tolerance = 1e-12)
    expect_equal(bridged$cumhaz, reference$cumhaz, tolerance = 1e-12)
    expect_identical(bridged$logse, reference$logse)
    expect_identical(bridged$conf.type, reference$conf.type)
    expect_equal(bridged$conf.int, reference$conf.int)
    for (field in c("std.err", "std.chaz", "lower", "upper")) {
      expect_equal(
        bridged[[field]],
        reference[[field]],
        tolerance = 1e-12,
        info = paste("Cox survfit uncertainty field", field)
      )
    }
  }

  profiles <- data.frame(x = c(0.2, 0.8), row.names = c("low", "high"))
  bridged_profiles <- survfit(bridged_unstratified, newdata = profiles)
  reference_profiles <- survival::survfit(reference_unstratified, newdata = profiles)
  compare_structure(survfit(bridged_unstratified), survival::survfit(reference_unstratified))
  compare_structure(bridged_profiles, reference_profiles)
  compare_structure(
    survfit(bridged_unstratified, censor = FALSE),
    survival::survfit(reference_unstratified, censor = FALSE)
  )
  compare_structure(
    survfit(bridged_unstratified, start.time = 2.5),
    survival::survfit(reference_unstratified, start.time = 2.5)
  )

  stratified_profiles <- data.frame(
    x = c(0.2, 0.8),
    group = factor(c("A", "B"), levels = levels(data$group)),
    row.names = c("a", "b")
  )
  bridged_stratified_profiles <- survfit(
    bridged_stratified,
    newdata = stratified_profiles
  )
  reference_stratified_profiles <- survival::survfit(
    reference_stratified,
    newdata = stratified_profiles
  )
  compare_structure(survfit(bridged_stratified), survival::survfit(reference_stratified))
  compare_structure(bridged_stratified_profiles, reference_stratified_profiles)
  compare_structure(
    survfit(bridged_stratified, censor = FALSE),
    survival::survfit(reference_stratified, censor = FALSE)
  )
  compare_structure(
    survfit(bridged_stratified, start.time = 2.5),
    survival::survfit(reference_stratified, start.time = 2.5)
  )

  counting_data <- data.frame(
    start = c(0, 1, 0, 2, 0, 2, 3, 4),
    stop = c(1, 4, 2, 5, 3, 5, 6, 7),
    status = c(0, 1, 1, 0, 1, 0, 1, 0),
    group = factor(rep(c("A", "B"), each = 4L)),
    x = c(0.2, 0.4, 0.1, 0.3, 1, 1.2, 0.8, 1.1)
  )
  bridged_counting <- coxph(
    Surv(start, stop, status) ~ x + strata(group),
    data = counting_data,
    max_iter = 0
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x + strata(group),
    data = counting_data,
    init = 0,
    iter.max = 0
  )
  compare_structure(
    survfit(bridged_counting),
    survival::survfit(reference_counting)
  )
  compare_structure(
    survfit(bridged_counting, start.time = 2.5),
    survival::survfit(reference_counting, start.time = 2.5)
  )

  compare_structure(bridged_profiles[integer()], reference_profiles[integer()])
  compare_structure(
    bridged_stratified_profiles[integer()],
    reference_stratified_profiles[integer()]
  )
  compare_structure(
    bridged_stratified_profiles[1L, drop = FALSE],
    reference_stratified_profiles[1L, drop = FALSE]
  )

  bridged_no_se <- survfit(bridged_unstratified, se.fit = FALSE)
  reference_no_se <- survival::survfit(reference_unstratified, se.fit = FALSE)
  expect_null(bridged_no_se$logse)
  expect_null(reference_no_se$logse)
  expect_null(bridged_no_se$conf.type)
  expect_null(reference_no_se$conf.type)
  expect_null(bridged_no_se$conf.int)
  expect_null(reference_no_se$conf.int)
})

test_that("Cox cumulative-hazard survfit styles match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = c(1, 1, 2, 2, 3, 4),
    status = c(1, 1, 1, 0, 1, 0),
    x = c(0, 1, 0.2, 0.8, 0.4, 0.6)
  )
  bridged_model <- coxph(
    Surv(time, status) ~ x,
    data = data,
    ties = "efron",
    max_iter = 0
  )
  reference_model <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = data,
    ties = "efron",
    init = 0,
    iter.max = 0
  )
  compare_style <- function(arguments) {
    bridged <- do.call(survfit, c(list(bridged_model), arguments))
    reference <- do.call(survival::survfit, c(list(reference_model), arguments))
    for (field in c(
      "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
      "std.err", "std.chaz", "lower", "upper"
    )) {
      expect_equal(
        bridged[[field]],
        reference[[field]],
        tolerance = 1e-12,
        info = paste("Cox cumulative-hazard style field", field)
      )
    }
  }

  for (arguments in list(
    list(),
    list(stype = 2L),
    list(ctype = 1L),
    list(ctype = 2L),
    list(type = "aalen"),
    list(type = "efron"),
    list(type = "breslow"),
    list(type = "fleming-harrington"),
    list(type = "greenwood"),
    list(type = "tsiatis"),
    list(type = "exact")
  )) {
    compare_style(arguments)
  }
})

test_that("Cox product-limit survfit styles match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  compare_product_limit <- function(bridged_model, reference_model, arguments) {
    bridged <- do.call(survfit, c(list(bridged_model), arguments))
    reference <- do.call(survival::survfit, c(list(reference_model), arguments))
    for (field in c(
      "n", "time", "n.risk", "n.event", "n.censor", "strata",
      "surv", "cumhaz", "std.err", "lower", "upper"
    )) {
      expect_equal(
        bridged[[field]],
        reference[[field]],
        tolerance = 1e-12,
        info = paste("Cox product-limit style field", field)
      )
    }
    expect_identical(bridged$logse, reference$logse)
    expect_null(bridged$std.chaz)
    expect_null(reference$std.chaz)
  }

  right_data <- data.frame(
    time = 1:4,
    status = c(1, 1, 1, 0),
    x = 0:3,
    weight = 1:4,
    curve_offset = c(0.1, 0.2, 0.5, 0.9)
  )
  bridged_right <- coxph(
    Surv(time, status) ~ x + offset(curve_offset),
    data = right_data,
    weights = weight,
    max_iter = 0
  )
  reference_right <- survival::coxph(
    survival::Surv(time, status) ~ x + offset(curve_offset),
    data = right_data,
    weights = weight,
    init = 0,
    iter.max = 0
  )
  profiles <- data.frame(
    x = c(1, 2),
    curve_offset = c(0.3, 0.8),
    row.names = c("low", "high")
  )
  for (arguments in list(
    list(stype = 1L),
    list(type = "kaplan-meier"),
    list(type = "kalbfleisch-prentice"),
    list(stype = 1L, censor = FALSE),
    list(stype = 1L, newdata = profiles),
    list(stype = 1L, newdata = profiles, start.time = 1.5)
  )) {
    compare_product_limit(bridged_right, reference_right, arguments)
  }
  bridged_profiles <- survfit(bridged_right, newdata = profiles, stype = 1L)
  reference_profiles <- survival::survfit(
    reference_right,
    newdata = profiles,
    stype = 1L
  )
  bridged_empty <- bridged_profiles[integer()]
  reference_empty <- reference_profiles[integer()]
  expect_identical(dim(bridged_empty), dim(reference_empty))
  expect_equal(bridged_empty$std.err, reference_empty$std.err)
  expect_null(bridged_empty$std.chaz)
  expect_null(reference_empty$std.chaz)

  counting_data <- data.frame(
    start = c(0, 1, 0, 2, 0, 2, 3, 4),
    stop = c(1, 4, 2, 5, 3, 5, 6, 7),
    status = c(0, 1, 1, 0, 1, 0, 1, 0),
    group = factor(rep(c("A", "B"), each = 4L)),
    x = c(0.2, 0.4, 0.1, 0.3, 1, 1.2, 0.8, 1.1)
  )
  bridged_counting <- coxph(
    Surv(start, stop, status) ~ x + strata(group),
    data = counting_data,
    max_iter = 0
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x + strata(group),
    data = counting_data,
    init = 0,
    iter.max = 0
  )
  counting_profiles <- data.frame(
    x = c(0.2, 0.8),
    group = factor(c("A", "B"), levels = levels(counting_data$group))
  )
  for (arguments in list(
    list(stype = 1L),
    list(stype = 1L, censor = FALSE),
    list(stype = 1L, start.time = 2.5),
    list(stype = 1L, newdata = counting_profiles, start.time = 2.5)
  )) {
    compare_product_limit(bridged_counting, reference_counting, arguments)
  }
})

test_that("Cox individual trajectories match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    start = c(0, 0.5, 0, 1.5, 2, 1, 0.25, 3.5),
    stop = c(1, 2, 2, 3, 4, 4, 2.5, 5),
    status = c(1, 1, 0, 1, 0, 1, 1, 0),
    x1 = c(0.2, 0.8, -0.3, 1.1, 0.4, -0.7, 0.6, -0.1),
    x2 = c(1, 0, 1, 2, -1, 0.5, 1.5, -0.5),
    group = factor(rep(c("a", "b"), each = 4L))
  )
  trajectory <- data.frame(
    start = c(0, 2.5, 0, 3),
    stop = c(2.5, 5, 3, 5),
    status = 0L,
    x1 = c(0.1, 0.3, -0.2, 0.4),
    x2 = c(0.2, -0.1, 0.6, 0.5),
    group = factor(c("a", "b", "a", "b"), levels = levels(data$group)),
    subject = c("one", "one", "two", "two")
  )
  bridged_model <- coxph(
    Surv(start, stop, status) ~ x1 + x2 + strata(group),
    data = data,
    init = c(0.15, -0.1),
    max_iter = 0
  )
  reference_model <- survival::coxph(
    survival::Surv(start, stop, status) ~ x1 + x2 + strata(group),
    data = data,
    init = c(0.15, -0.1),
    iter.max = 0
  )

  compare_trajectory <- function(
    newdata,
    id,
    arguments = list(),
    bridged_fit = bridged_model,
    reference_fit = reference_model
  ) {
    bridged <- do.call(
      survfit,
      c(list(bridged_fit, newdata = newdata, id = id), arguments)
    )
    reference <- do.call(
      survival::survfit,
      c(list(reference_fit, newdata = newdata, id = id), arguments)
    )
    expect_identical(dim(bridged), dim(reference))
    for (field in c(
      "n", "time", "n.risk", "n.event", "n.censor", "strata",
      "surv", "cumhaz", "std.err", "std.chaz", "lower", "upper",
      "start.time"
    )) {
      expect_equal(
        bridged[[field]],
        reference[[field]],
        tolerance = 2e-12,
        info = paste("Cox individual trajectory field", field)
      )
    }
    expect_identical(bridged$logse, reference$logse)
    expect_identical(bridged$conf.type, reference$conf.type)
    expect_equal(bridged$conf.int, reference$conf.int)
  }

  for (arguments in list(
    list(stype = 2L, ctype = 2L),
    list(stype = 2L, ctype = 1L),
    list(stype = 1L, ctype = 2L),
    list(type = "kalbfleisch-prentice"),
    list(stype = 2L, ctype = 2L, censor = FALSE),
    list(stype = 1L, ctype = 2L, censor = FALSE),
    list(stype = 2L, ctype = 2L, start.time = 1.5),
    list(stype = 1L, ctype = 2L, start.time = 1.5),
    list(stype = 2L, ctype = 2L, time0 = TRUE),
    list(stype = 1L, ctype = 2L, time0 = TRUE),
    list(stype = 2L, ctype = 2L, se.fit = FALSE)
  )) {
    compare_trajectory(trajectory, trajectory$subject, arguments)
  }
  compare_trajectory(
    trajectory[1:2, , drop = FALSE],
    trajectory$subject[1:2],
    list(stype = 2L, ctype = 2L)
  )

  weighted_data <- transform(
    data,
    weight = c(1, 2, 1, 3, 2, 1, 2, 1),
    curve_offset = c(0.1, 0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0)
  )
  weighted_trajectory <- transform(
    trajectory,
    curve_offset = c(0.3, -0.1, 0.2, 0.4)
  )
  bridged_weighted <- coxph(
    Surv(start, stop, status) ~ x1 + x2 + strata(group) + offset(curve_offset),
    data = weighted_data,
    weights = weight,
    init = c(0.15, -0.1),
    max_iter = 0
  )
  reference_weighted <- survival::coxph(
    survival::Surv(start, stop, status) ~ x1 + x2 + strata(group) +
      offset(curve_offset),
    data = weighted_data,
    weights = weight,
    init = c(0.15, -0.1),
    iter.max = 0
  )
  for (stype in c(1L, 2L)) {
    compare_trajectory(
      weighted_trajectory,
      weighted_trajectory$subject,
      list(stype = stype, ctype = 2L),
      bridged_weighted,
      reference_weighted
    )
  }

  bridged_legacy <- expect_warning(
    survfit(bridged_model, newdata = trajectory, individual = TRUE),
    "supersedes"
  )
  reference_legacy <- expect_warning(
    survival::survfit(reference_model, newdata = trajectory, individual = TRUE),
    "supersedes"
  )
  expect_equal(bridged_legacy$time, reference_legacy$time, tolerance = 2e-12)
  expect_equal(bridged_legacy$surv, reference_legacy$surv, tolerance = 2e-12)
  expect_error(
    survfit(bridged_model, newdata = trajectory, id = NULL),
    "id=NULL is an invalid argument",
    fixed = TRUE
  )
})

test_that("Cox survfit accepts the reference no-op influence argument", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = 1:6,
    status = c(1, 1, 0, 1, 0, 1),
    x = c(0, 0.5, 1, 1.5, 2, 2.5)
  )
  bridged_model <- coxph(Surv(time, status) ~ x, data = data, max_iter = 0)
  reference_model <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = data,
    init = 0,
    iter.max = 0
  )
  for (value in list(FALSE, TRUE, "unused", c(1, 2))) {
    bridged <- survfit(bridged_model, influence = value)
    reference <- survival::survfit(reference_model, influence = value)
    for (field in c("time", "surv", "cumhaz", "std.err", "std.chaz")) {
      expect_equal(bridged[[field]], reference[[field]], tolerance = 1e-12)
    }
  }
})

test_that("Cox survfit newdata omission matches survival for every na.action", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = 1:6,
    status = c(1, 1, 0, 1, 0, 1),
    x = c(0, 0.5, 1, 1.5, 2, 2.5),
    group = factor(rep(c("a", "b"), 3L))
  )
  newdata <- data.frame(
    x = c(0, NA, 1, 2),
    group = factor(c("a", "a", NA, "b"), levels = levels(data$group)),
    unused = c(NA, 1, 2, NA),
    row.names = c("ok", "missing-x", "missing-group", "ok2")
  )
  bridged_model <- coxph(
    Surv(time, status) ~ x + group,
    data = data,
    init = c(0, 0),
    max_iter = 0
  )
  reference_model <- survival::coxph(
    survival::Surv(time, status) ~ x + group,
    data = data,
    init = c(0, 0),
    iter.max = 0
  )

  for (na_action in list(na.pass, na.omit, na.exclude, na.fail, identity)) {
    bridged <- survfit(bridged_model, newdata = newdata, na.action = na_action)
    reference <- survival::survfit(
      reference_model,
      newdata = newdata,
      na.action = na_action
    )
    expect_identical(dim(bridged), dim(reference))
    for (field in c("time", "surv", "cumhaz", "std.err", "std.chaz", "lower", "upper")) {
      expect_equal(bridged[[field]], reference[[field]], tolerance = 1e-12)
    }
  }
  expect_error(
    survfit(
      bridged_model,
      newdata = transform(newdata[1:2, ], x = NA_real_)
    ),
    "all rows of newdata have missing values",
    fixed = TRUE
  )

  counting_data <- data.frame(
    start = c(0, 0, 1, 1),
    stop = 1:4,
    status = c(1, 0, 1, 0),
    x = c(0, 0.5, 1, 1.5)
  )
  counting_newdata <- data.frame(
    start = c(0, 2, 0, 2),
    stop = c(2, 4, 2, NA),
    status = 0L,
    x = c(0.2, NA, 0.6, 0.8),
    subject = c("one", "one", NA, "two")
  )
  bridged_counting <- coxph(
    Surv(start, stop, status) ~ x,
    data = counting_data,
    max_iter = 0
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x,
    data = counting_data,
    iter.max = 0
  )
  bridged_individual <- survfit(
    bridged_counting,
    newdata = counting_newdata,
    id = counting_newdata$subject
  )
  reference_individual <- survival::survfit(
    reference_counting,
    newdata = counting_newdata,
    id = counting_newdata$subject
  )
  for (field in c(
    "n", "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
    "std.err", "std.chaz", "lower", "upper"
  )) {
    expect_equal(
      bridged_individual[[field]],
      reference_individual[[field]],
      tolerance = 1e-12
    )
  }
})

test_that("Cox survfit formula guards match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = 1:8,
    status = c(1, 1, 0, 1, 0, 1, 0, 1),
    x = 1:8,
    z = rep(c(0, 1), 4L),
    group = factor(rep(c("a", "b"), 4L))
  )
  incomplete <- coxph(
    Surv(time, status) ~ x:z,
    data = data,
    max_iter = 0
  )
  reference_incomplete <- survival::coxph(
    survival::Surv(time, status) ~ x:z,
    data = data,
    iter.max = 0
  )
  expected_error <- "interaction without the lower order effect"
  expect_error(survfit(incomplete), expected_error)
  expect_error(survival::survfit(reference_incomplete), expected_error)

  hierarchical <- coxph(
    Surv(time, status) ~ x * z,
    data = data,
    max_iter = 0
  )
  reference_hierarchical <- survival::coxph(
    survival::Surv(time, status) ~ x * z,
    data = data,
    iter.max = 0
  )
  expect_warning(
    bridged_default <- survfit(hierarchical),
    "model contains interactions"
  )
  expect_warning(
    reference_default <- survival::survfit(reference_hierarchical),
    "model contains interactions"
  )
  expect_equal(bridged_default$surv, reference_default$surv, tolerance = 1e-12)

  newdata <- data.frame(x = c(1, 2), z = c(0, 1))
  bridged_selected <- survfit(hierarchical, newdata = newdata)
  reference_selected <- survival::survfit(reference_hierarchical, newdata = newdata)
  expect_equal(bridged_selected$surv, reference_selected$surv, tolerance = 1e-12)

  stratified <- coxph(
    Surv(time, status) ~ x * strata(group),
    data = data,
    max_iter = 0
  )
  reference_stratified <- survival::coxph(
    survival::Surv(time, status) ~ x * strata(group),
    data = data,
    iter.max = 0
  )
  expect_warning(
    expect_error(survfit(stratified), "strata by covariate interaction"),
    "model contains interactions"
  )
  expect_warning(
    expect_error(
      survival::survfit(reference_stratified),
      "strata by covariate interaction"
    ),
    "model contains interactions"
  )

  transformed <- coxph(
    Surv(time, status) ~ tt(x),
    data = data,
    tt = function(values, ...) values,
    max_iter = 0
  )
  reference_transformed <- survival::coxph(
    survival::Surv(time, status) ~ tt(x),
    data = data,
    tt = function(values, ...) values,
    iter.max = 0
  )
  expected_tt_error <- "can not process coxph models with a tt term"
  expect_error(survfit(transformed), expected_tt_error)
  expect_error(survival::survfit(reference_transformed), expected_tt_error)
})

test_that("Cox survfit newdata shape rules match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = 1:8,
    status = c(1, 1, 0, 1, 0, 1, 0, 1),
    x = 1:8,
    z = rep(c(0, 1), 4L)
  )
  bridged <- coxph(
    Surv(time, status) ~ x + z,
    data = data,
    init = c(0, 0),
    max_iter = 0
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + z,
    data = data,
    init = c(0, 0),
    iter.max = 0
  )

  profile <- c(x = 2, z = 1)
  bridged_profile <- survfit(bridged, newdata = profile)
  reference_profile <- survival::survfit(reference, newdata = profile)
  for (field in c("time", "surv", "cumhaz", "std.err", "std.chaz", "lower", "upper")) {
    expect_equal(
      bridged_profile[[field]],
      reference_profile[[field]],
      tolerance = 1e-12
    )
  }

  expect_error(
    survfit(bridged, newdata = unname(profile)),
    "Newdata argument must be a data frame",
    fixed = TRUE
  )
  expect_error(
    survival::survfit(reference, newdata = unname(profile)),
    "Newdata argument must be a data frame",
    fixed = TRUE
  )
  profile_matrix <- matrix(profile, nrow = 1L, dimnames = list(NULL, names(profile)))
  expected_matrix_error <- "'data' must be a data.frame, not a matrix or an array"
  expect_error(survfit(bridged, newdata = profile_matrix), expected_matrix_error, fixed = TRUE)
  expect_error(
    survival::survfit(reference, newdata = profile_matrix),
    expected_matrix_error,
    fixed = TRUE
  )

  counting_data <- transform(data, start = pmax(time - 1, 0), stop = time)
  bridged_counting <- coxph(
    Surv(start, stop, status) ~ x + z,
    data = counting_data,
    init = c(0, 0),
    max_iter = 0
  )
  reference_counting <- survival::coxph(
    survival::Surv(start, stop, status) ~ x + z,
    data = counting_data,
    init = c(0, 0),
    iter.max = 0
  )
  expect_error(
    survfit(bridged_counting, newdata = profile, id = "one"),
    "newdata must be a data frame",
    fixed = TRUE
  )
  expect_error(
    survival::survfit(reference_counting, newdata = profile, id = "one"),
    "newdata must be a data frame",
    fixed = TRUE
  )
})

test_that("stratified Cox profiles without strata expand across every baseline", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  data <- data.frame(
    time = 1:8,
    status = c(1, 1, 0, 1, 0, 1, 0, 1),
    x = 1:8,
    group = factor(rep(c("a", "b"), 4L))
  )
  bridged <- coxph(
    Surv(time, status) ~ x + strata(group),
    data = data,
    init = list(0.1),
    max_iter = 0
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + strata(group),
    data = data,
    init = 0.1,
    iter.max = 0
  )
  profiles <- data.frame(x = c(2, 3), row.names = c("first", "second"))
  option_sets <- list(
    list(),
    list(stype = 1L, ctype = 2L),
    list(censor = FALSE),
    list(start.time = 2.5),
    list(se.fit = FALSE)
  )

  for (options in option_sets) {
    bridged_profiles <- do.call(survfit, c(list(bridged, newdata = profiles), options))
    reference_profiles <- do.call(
      survival::survfit,
      c(list(reference, newdata = profiles), options)
    )
    expect_identical(dim(bridged_profiles), dim(reference_profiles))
    expect_equal(bridged_profiles$strata, reference_profiles$strata)
    for (field in c(
      "n", "time", "n.risk", "n.event", "n.censor", "surv", "cumhaz",
      "std.err", "std.chaz", "lower", "upper"
    )) {
      expect_equal(
        bridged_profiles[[field]],
        reference_profiles[[field]],
        tolerance = 1e-12
      )
    }
  }

  single_profile <- profiles[1L, , drop = FALSE]
  bridged_single <- survfit(bridged, newdata = single_profile)
  reference_single <- survival::survfit(reference, newdata = single_profile)
  expect_identical(dim(bridged_single), dim(reference_single))
  expect_equal(bridged_single$strata, reference_single$strata)
  for (field in c("time", "surv", "cumhaz", "std.err", "lower", "upper")) {
    expect_equal(bridged_single[[field]], reference_single[[field]], tolerance = 1e-12)
  }

  for (include_confidence in c(FALSE, TRUE)) {
    expect_equal(
      quantile(
        survfit(bridged, newdata = profiles),
        probs = c(0, 0.5),
        conf.int = include_confidence
      ),
      quantile(
        survival::survfit(reference, newdata = profiles),
        probs = c(0, 0.5),
        conf.int = include_confidence
      ),
      tolerance = 1e-12
    )
  }
  expect_equal(
    median(survfit(bridged, newdata = profiles)),
    median(survival::survfit(reference, newdata = profiles)),
    tolerance = 1e-12
  )

  bridged_profiles <- survfit(bridged, newdata = profiles)
  reference_profiles <- survival::survfit(reference, newdata = profiles)
  subsetters <- list(
    function(value) value[c(4L, 1L, 4L)],
    function(value) value[1L, ],
    function(value) value[, 1L],
    function(value) value[1L, 1L],
    function(value) value[c(2L, 1L), c(2L, 1L), drop = FALSE],
    function(value) value[1L, , drop = FALSE],
    function(value) value[, 1L, drop = FALSE],
    function(value) value["b", c(2L, 1L), drop = FALSE],
    function(value) value[, integer(), drop = FALSE],
    function(value) value[integer(), , drop = FALSE]
  )
  for (subsetter in subsetters) {
    bridged_subset <- subsetter(bridged_profiles)
    reference_subset <- subsetter(reference_profiles)
    expect_identical(dim(bridged_subset), dim(reference_subset))
    for (field in c(
      "n", "time", "n.risk", "n.event", "n.censor", "strata",
      "surv", "cumhaz", "std.err", "std.chaz", "lower", "upper"
    )) {
      expect_equal(
        bridged_subset[[field]],
        reference_subset[[field]],
        tolerance = 1e-12,
        info = paste("strata-by-data subset field", field)
      )
    }
  }
  expect_error(
    bridged_profiles["b", c("second", "first"), drop = FALSE],
    "no 'dimnames' attribute for array",
    fixed = TRUE
  )
})

test_that("multi-state survfit tables and summaries agree with R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  reference_survfit <- getFromNamespace("survfit.formula", "survival")

  data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6),
    event = factor(
      c("ill", "death", "censor", "death", "ill", "censor"),
      levels = c("censor", "ill", "death")
    ),
    group = factor(c("a", "a", "a", "b", "b", "b"))
  )
  p0 <- c(0.25, 0.5, 0.25)
  compare_frames <- function(actual, expected) {
    expect_identical(names(actual), names(expected))
    expect_identical(actual$state, expected$state)
    for (name in setdiff(names(expected), c("state", "strata"))) {
      expect_equal(actual[[name]], expected[[name]], tolerance = 1e-06)
    }
    if ("strata" %in% names(expected)) {
      expect_identical(as.character(actual$strata), as.character(expected$strata))
    }
  }

  bridged <- survfit(Surv(time, event) ~ 1, data = data, p0 = p0)
  reference <- reference_survfit(
    survival::Surv(time, event) ~ 1,
    data = data,
    p0 = p0
  )
  compare_frames(
    as.data.frame(bridged),
    summary(reference, data.frame = TRUE, censored = TRUE)
  )
  compare_frames(
    as.data.frame(summary(bridged)),
    summary(reference, data.frame = TRUE)
  )
  compare_frames(
    as.data.frame(summary(bridged, times = c(0, 1.5, 3, 7), extend = TRUE)),
    summary(
      reference,
      times = c(0, 1.5, 3, 7),
      extend = TRUE,
      data.frame = TRUE
    )
  )

  grouped_bridged <- survfit(Surv(time, event) ~ group, data = data, p0 = p0)
  grouped_reference <- reference_survfit(
    survival::Surv(time, event) ~ group,
    data = data,
    p0 = p0
  )
  grouped_expected <- summary(grouped_reference, data.frame = TRUE, censored = TRUE)
  grouped_expected$strata <- sub("^group=", "", as.character(grouped_expected$strata))
  compare_frames(as.data.frame(grouped_bridged), grouped_expected)

  grouped_time_expected <- summary(
    grouped_reference,
    times = c(0, 2.5, 7),
    extend = TRUE,
    data.frame = TRUE
  )
  grouped_time_expected$strata <- sub(
    "^group=",
    "",
    as.character(grouped_time_expected$strata)
  )
  compare_frames(
    as.data.frame(summary(grouped_bridged, times = c(0, 2.5, 7), extend = TRUE)),
    grouped_time_expected
  )

  expect_identical(names(bridged), setdiff(names(reference), "call"))
  expect_identical(length(bridged), length(reference) - 1L)
  expect_identical(dim(bridged), dim(reference))
  for (name in c(
    "n.risk", "n.event", "n.censor", "pstate", "n.transition",
    "cumhaz", "std.err", "std.chaz", "std.auc", "lower", "upper"
  )) {
    expect_equal(bridged[[name]], reference[[name]], tolerance = 1e-06)
  }
  expect_equal(bridged$p0, reference$p0, tolerance = 1e-12)
  expect_equal(bridged$transitions, reference$transitions)
  expect_identical(bridged$states, reference$states)
  expect_identical(bridged$type, reference$type)
  expect_equal(bridged$conf.int, reference$conf.int)
  expect_identical(bridged$conf.type, reference$conf.type)
  expect_equal(bridged$n_risk, bridged$n.risk)
  expect_equal(bridged[["pstate"]], bridged$pstate)

  expect_identical(names(grouped_bridged), setdiff(names(grouped_reference), "call"))
  expect_identical(dim(grouped_bridged), dim(grouped_reference))
  for (name in c(
    "n.risk", "n.event", "n.censor", "pstate", "n.transition",
    "cumhaz", "std.err", "std.chaz", "std.auc", "lower", "upper"
  )) {
    expect_equal(grouped_bridged[[name]], grouped_reference[[name]], tolerance = 1e-06)
  }
  expect_equal(unname(grouped_bridged$n), unname(grouped_reference$n))
  expect_equal(unname(grouped_bridged$n.id), unname(grouped_reference$n.id))
  expect_equal(unname(grouped_bridged$p0), unname(grouped_reference$p0))
  expect_equal(unname(grouped_bridged$strata), unname(grouped_reference$strata))

  direct_ill <- bridged["ill"]
  reference_ill <- reference["ill"]
  expect_identical(names(direct_ill), setdiff(names(reference_ill), "call"))
  expect_identical(dim(direct_ill), dim(reference_ill))
  expect_identical(direct_ill$states, reference_ill$states)
  expect_identical(direct_ill$oldstate, reference_ill$oldstate)
  expect_equal(direct_ill$pstate, reference_ill$pstate, tolerance = 1e-06)
  expect_equal(direct_ill$n.risk, reference_ill$n.risk, tolerance = 1e-06)
  expect_equal(direct_ill$n.event, reference_ill$n.event, tolerance = 1e-06)

  grouped_ill <- grouped_bridged[, "ill"]
  grouped_reference_ill <- grouped_reference[, "ill"]
  expect_identical(names(grouped_ill), setdiff(names(grouped_reference_ill), "call"))
  expect_identical(dim(grouped_ill), dim(grouped_reference_ill))
  expect_identical(grouped_ill$states, grouped_reference_ill$states)
  expect_identical(grouped_ill$oldstate, grouped_reference_ill$oldstate)
  expect_equal(grouped_ill$pstate, grouped_reference_ill$pstate, tolerance = 1e-06)

  group_a_ill <- grouped_bridged["a", "ill"]
  group_a_reference_ill <- grouped_reference["group=a", "ill"]
  expect_identical(dim(group_a_ill), dim(group_a_reference_ill))
  expect_equal(group_a_ill$pstate, group_a_reference_ill$pstate, tolerance = 1e-06)
  expect_error(grouped_bridged[1L], "single index subscripts")
  expect_error(quantile(bridged), "not a well defined quantity")
  expect_error(median(grouped_bridged), "not a well defined quantity")

  diagnostic_bridged <- survfit(
    Surv(time, event) ~ 1,
    data = data,
    model = TRUE
  )
  diagnostic_reference <- reference_survfit(
    survival::Surv(time, event) ~ 1,
    data = data,
    model = TRUE
  )
  # survival 3.8.x does not retain this requested frame, and reconstructing it
  # later loses testthat-local data before the residual comparisons run.
  diagnostic_reference$model <- stats::model.frame.default(
    survival::Surv(time, event) ~ 1,
    data = data
  )
  for (diagnostic_type in c("pstate", "cumhaz", "sojourn")) {
    expect_equal(
      residuals(diagnostic_bridged, times = c(2, 5), type = diagnostic_type),
      stats::residuals(diagnostic_reference, times = c(2, 5), type = diagnostic_type),
      tolerance = 1e-10
    )
    expect_equal(
      pseudo(diagnostic_bridged, times = c(2, 5), type = diagnostic_type),
      survival::pseudo(diagnostic_reference, times = c(2, 5), type = diagnostic_type),
      tolerance = 1e-10
    )
  }
  expect_equal(
    residuals(diagnostic_bridged, times = 2),
    stats::residuals(diagnostic_reference, times = 2),
    tolerance = 1e-10
  )
  expect_equal(
    pseudo(diagnostic_bridged, times = 2),
    survival::pseudo(diagnostic_reference, times = 2),
    tolerance = 1e-10
  )
  expect_equal(
    residuals(diagnostic_bridged, times = c(2, 5), data.frame = TRUE),
    stats::residuals(diagnostic_reference, times = c(2, 5), data.frame = TRUE),
    tolerance = 1e-10
  )
  expect_equal(
    pseudo(diagnostic_bridged, times = c(2, 5), data.frame = TRUE),
    survival::pseudo(diagnostic_reference, times = c(2, 5), data.frame = TRUE),
    tolerance = 1e-10
  )

  grouped_diagnostic_bridged <- survfit(
    Surv(time, event) ~ group,
    data = data,
    model = TRUE
  )
  grouped_diagnostic_reference <- reference_survfit(
    survival::Surv(time, event) ~ group,
    data = data,
    model = TRUE
  )
  grouped_diagnostic_reference$model <- stats::model.frame.default(
    survival::Surv(time, event) ~ group,
    data = data
  )
  for (diagnostic_type in c("pstate", "cumhaz", "sojourn")) {
    expect_equal(
      residuals(
        grouped_diagnostic_bridged,
        times = c(2, 5),
        type = diagnostic_type
      ),
      stats::residuals(
        grouped_diagnostic_reference,
        times = c(2, 5),
        type = diagnostic_type
      ),
      tolerance = 1e-10
    )
    expect_equal(
      suppressWarnings(pseudo(
        grouped_diagnostic_bridged,
        times = c(2, 5),
        type = diagnostic_type
      )),
      suppressWarnings(survival::pseudo(
        grouped_diagnostic_reference,
        times = c(2, 5),
        type = diagnostic_type
      )),
      tolerance = 1e-10
    )
  }

  diagnostic_weights <- c(1, 2, 1.5, 0.5, 3, 1)
  weighted_diagnostic_bridged <- survfit(
    Surv(time, event) ~ 1,
    data = data,
    weights = diagnostic_weights,
    model = TRUE
  )
  weighted_diagnostic_reference <- reference_survfit(
    survival::Surv(time, event) ~ 1,
    data = data,
    weights = diagnostic_weights,
    model = TRUE
  )
  weighted_diagnostic_reference$model <- stats::model.frame.default(
    survival::Surv(time, event) ~ 1,
    data = data,
    weights = diagnostic_weights
  )
  for (weighted_value in c(FALSE, TRUE)) {
    expect_equal(
      residuals(
        weighted_diagnostic_bridged,
        times = c(2, 5),
        weighted = weighted_value
      ),
      stats::residuals(
        weighted_diagnostic_reference,
        times = c(2, 5),
        weighted = weighted_value
      ),
      tolerance = 1e-10
    )
  }
  expect_equal(
    pseudo(weighted_diagnostic_bridged, times = c(2, 5)),
    survival::pseudo(weighted_diagnostic_reference, times = c(2, 5)),
    tolerance = 1e-10
  )

  counting_data <- data.frame(
    id = c(1, 1, 2, 2, 3, 3),
    start = c(0, 1, 0, 2, 0, 3),
    stop = c(1, 4, 2, 5, 3, 6),
    event = factor(
      c("ill", "death", "ill", "censor", "death", "censor"),
      levels = c("censor", "ill", "death")
    )
  )
  counting_diagnostic_bridged <- survfit(
    Surv(start, stop, event) ~ 1,
    data = counting_data,
    id = id,
    model = TRUE
  )
  counting_diagnostic_reference <- reference_survfit(
    survival::Surv(start, stop, event) ~ 1,
    data = counting_data,
    id = counting_data$id,
    model = TRUE
  )
  counting_diagnostic_reference$model <- stats::model.frame.default(
    survival::Surv(start, stop, event) ~ 1,
    data = counting_data,
    id = id
  )
  for (diagnostic_type in c("pstate", "cumhaz", "sojourn")) {
    expect_equal(
      residuals(
        counting_diagnostic_bridged,
        times = c(2, 5),
        type = diagnostic_type
      ),
      stats::residuals(
        counting_diagnostic_reference,
        times = c(2, 5),
        type = diagnostic_type
      ),
      tolerance = 1e-10
    )
    expect_equal(
      residuals(
        counting_diagnostic_bridged,
        times = c(2, 5),
        type = diagnostic_type,
        collapse = TRUE,
        weighted = TRUE
      ),
      stats::residuals(
        counting_diagnostic_reference,
        times = c(2, 5),
        type = diagnostic_type,
        collapse = TRUE,
        weighted = TRUE
      ),
      tolerance = 1e-10
    )
  }
  expect_equal(
    pseudo(counting_diagnostic_bridged, times = c(2, 5)),
    survival::pseudo(counting_diagnostic_reference, times = c(2, 5)),
    tolerance = 1e-10
  )
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

  multigroup_diff_data <- data.frame(
    time = c(1, 2, 3, 2, 4, 6, 3, 5, 7),
    status = c(1, 1, 0, 1, 0, 1, 1, 1, 0),
    group = rep(c("A", "B", "C"), each = 3)
  )
  bridged_multigroup_diff <- survdiff(
    Surv(time, status) ~ group,
    data = multigroup_diff_data
  )
  reference_multigroup_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data = multigroup_diff_data
  )
  expect_equal(
    as.data.frame(bridged_multigroup_diff)$variance,
    unname(diag(reference_multigroup_diff$var)),
    tolerance = 1e-06
  )

  ordered_diff_data <- data.frame(
    time = c(5, 4, 1, 5, 6, 1, 3),
    status = c(0, 0, 0, 1, 0, 0, 0),
    group = factor(
      c("g2", "g3", "g4", "g1", "g3", "g2", "g1"),
      levels = c("g4", "g3", "g2", "g1", "unused")
    )
  )
  bridged_ordered_diff <- survdiff(
    Surv(time, status) ~ group,
    data = ordered_diff_data,
    rho = 0.5
  )
  reference_ordered_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data = ordered_diff_data,
    rho = 0.5
  )
  expect_survdiff_equal(bridged_ordered_diff, reference_ordered_diff)

  near_tie_diff_data <- data.frame(
    time = c(4, 1, 5, 1 + 5e-9, 1),
    status = c(0, 1, 1, 1, 0),
    group = c(20, 10, 20, 10, 10)
  )
  bridged_near_tie_diff <- survdiff(
    Surv(time, status) ~ group,
    data = near_tie_diff_data,
    rho = -0.5
  )
  reference_near_tie_diff <- survival::survdiff(
    survival::Surv(time, status) ~ group,
    data = near_tie_diff_data,
    rho = -0.5
  )
  expect_survdiff_equal(bridged_near_tie_diff, reference_near_tie_diff)

  degenerate_diff_data <- data.frame(
    time = 1:4,
    status = rep(0, 4),
    group = rep(c(1, 2), each = 2)
  )
  expect_warning(
    bridged_degenerate_diff <- survdiff(
      Surv(time, status) ~ group,
      data = degenerate_diff_data
    ),
    "NaNs produced"
  )
  expect_warning(
    reference_degenerate_diff <- survival::survdiff(
      survival::Surv(time, status) ~ group,
      data = degenerate_diff_data
    ),
    "NaNs produced"
  )
  expect_true(is.nan(as.numeric(bridged_degenerate_diff$p_value)))
  expect_true(is.nan(reference_degenerate_diff$pvalue))
  expect_error(
    survdiff(
      Surv(time, status) ~ group,
      data = transform(degenerate_diff_data, time = rep(1, 4), status = rep(1, 4))
    ),
    "Lapack routine dgesv: system is exactly singular: U[1,1] = 0",
    fixed = TRUE
  )

  singular_diff_data <- data.frame(
    time = c(3.0000000000001, 7, 6, 7, 7, 3, 3),
    status = c(1, 0, 1, 1, 1, 0, 0),
    group = factor(
      c("g2", "g1", "g3", "g1", "g4", "g3", "g2"),
      levels = c("g4", "g3", "g2", "g1", "unused")
    ),
    site = c("s1", "s2", "s1", "s1", "s3", "s2", "s3")
  )
  bridged_singular_condition <- tryCatch(
    survdiff(
      Surv(time, status) ~ group + strata(site),
      data = singular_diff_data,
      rho = -0.5,
      timefix = FALSE
    ),
    error = identity
  )
  reference_singular_condition <- tryCatch(
    {
      reference_singular_fit <- survival:::survdiff.fit(
        survival::Surv(singular_diff_data$time, singular_diff_data$status),
        singular_diff_data$group,
        singular_diff_data$site,
        -0.5
      )
      reference_expected <- apply(reference_singular_fit$expected, 1L, sum)
      retained <- reference_expected > 0
      reference_variance <- reference_singular_fit$var[retained, retained]
      reference_variance <- reference_variance[-1L, -1L, drop = FALSE]
      solve(reference_variance, rep(0, nrow(reference_variance)))
    },
    error = identity
  )
  expect_s3_class(bridged_singular_condition, "error")
  expect_s3_class(reference_singular_condition, "error")
  expect_identical(
    conditionMessage(bridged_singular_condition),
    conditionMessage(reference_singular_condition)
  )

  rounded_singular_data <- data.frame(
    time = c(7, 3, 5, 7, 6, 8, 4, 1, 8, 6),
    status = c(0, 0, 1, 1, 0, 1, 1, 1, 0, 1),
    group = factor(
      c("g3", "g4", "g1", "g2", "g1", "g2", "g1", "g4", "g3", "g2"),
      levels = c("g4", "g3", "g2", "g1", "unused")
    ),
    site = c("s1", "s4", "s1", "s3", "s2", "s1", "s2", "s4", "s3", "s2")
  )
  bridged_rounded_condition <- tryCatch(
    survdiff(
      Surv(time, status) ~ group + strata(site),
      data = rounded_singular_data,
      rho = 2
    ),
    error = identity
  )
  reference_rounded_condition <- tryCatch(
    survival::survdiff(
      survival::Surv(time, status) ~ group + strata(site),
      data = rounded_singular_data,
      rho = 2
    ),
    error = identity
  )
  expect_s3_class(bridged_rounded_condition, "error")
  expect_s3_class(reference_rounded_condition, "error")
  expect_identical(
    conditionMessage(bridged_rounded_condition),
    conditionMessage(reference_rounded_condition)
  )

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

test_that("agexact.fit matches tied counting-process reference output", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  n <- 10L
  x <- matrix(
    c(
      -0.8, 0.3, 1.1, -0.2, 0.7, 1.4, -1.0, 0.5, 1.0, -0.4,
      0.2, 1.3, -0.5, 0.8, -1.1, 0.4, 1.2, -0.7, 0.6, 1.0
    ),
    nrow = n,
    ncol = 2L
  )
  colnames(x) <- c("x1", "x2")
  start <- c(0, 0, 0.5, 0.5, 1, 1, 1.5, 2, 2.5, 3)
  stop <- c(2, 3, 3, 3, 4, 4.5, 5, 5, 5, 6)
  event <- c(1, 1, 1, 0, 0, 1, 1, 1, 0, 1)
  offset <- c(0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.25, 0.2, -0.05, 0.1)
  y <- survival::Surv(start, stop, event)
  row_labels <- paste0("row", seq_len(n))

  bridged <- agexact.fit(
    x,
    y,
    strata = NULL,
    offset = offset,
    init = c(0.25, -0.15),
    control = coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  reference <- survival::agexact.fit(
    x,
    y,
    strata = NULL,
    offset = offset,
    init = c(0.25, -0.15),
    control = survival::coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )

  expect_equal(names(bridged), names(reference))
  expect_equal(bridged$coefficients, reference$coefficients, tolerance = 1e-10)
  expect_equal(bridged$var, reference$var, tolerance = 1e-10)
  expect_equal(bridged$loglik, reference$loglik, tolerance = 1e-10)
  expect_equal(bridged$score, reference$score, tolerance = 1e-10)
  expect_equal(bridged$iter, reference$iter)
  expect_equal(bridged$linear.predictors, reference$linear.predictors, tolerance = 1e-10)
  expect_equal(bridged$residuals, reference$residuals, tolerance = 1e-10)
  expect_equal(bridged$means, reference$means, tolerance = 1e-12)

  bridged_wrapped_response <- agexact.fit(
    x,
    Surv(start, stop, event),
    strata = NULL,
    offset = offset,
    init = c(0.25, -0.15),
    control = coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  expect_equal(bridged_wrapped_response, bridged, tolerance = 1e-10)

  negative_y <- survival::Surv(start - 4, stop - 4, event)
  bridged_negative_times <- agexact.fit(
    x,
    negative_y,
    strata = NULL,
    offset = offset,
    init = c(0.25, -0.15),
    control = coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  reference_negative_times <- survival::agexact.fit(
    x,
    negative_y,
    strata = NULL,
    offset = offset,
    init = c(0.25, -0.15),
    control = survival::coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  expect_equal(
    bridged_negative_times$residuals,
    reference_negative_times$residuals,
    tolerance = 1e-10
  )

  strata_labels <- c(rep("north", 5L), rep("south", 5L))
  strata_factor <- factor(strata_labels, levels = c("south", "north"))
  zero_iteration_control <- coxph.control(iter.max = 0L, eps = 1e-09)
  bridged_factor_strata <- agexact.fit(
    x,
    y,
    strata = strata_factor,
    offset = offset,
    init = c(0.25, -0.15),
    control = zero_iteration_control,
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  bridged_character_strata <- agexact.fit(
    x,
    y,
    strata = strata_labels,
    offset = offset,
    init = c(0.25, -0.15),
    control = zero_iteration_control,
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  reference_factor_strata <- survival::agexact.fit(
    x,
    y,
    strata = strata_factor,
    offset = offset,
    init = c(0.25, -0.15),
    control = survival::coxph.control(iter.max = 0L, eps = 1e-09),
    weights = rep(1, n),
    method = "exact",
    rownames = row_labels
  )
  comparable_fields <- c(
    "coefficients", "var", "loglik", "score", "iter",
    "linear.predictors", "residuals", "means", "method"
  )
  for (field in comparable_fields) {
    expect_equal(
      bridged_factor_strata[[field]],
      reference_factor_strata[[field]],
      tolerance = 1e-10
    )
    expect_equal(
      bridged_character_strata[[field]],
      bridged_factor_strata[[field]],
      tolerance = 1e-12
    )
  }

  large_n <- 80L
  large_x <- matrix(
    sin(seq_len(large_n) * 0.73) + cos(seq_len(large_n) * 0.19),
    ncol = 1L
  )
  large_y <- survival::Surv(
    rep(0, large_n),
    seq_len(large_n),
    rep(c(1L, 1L, 0L), length.out = large_n)
  )
  large_rows <- as.character(seq_len(large_n))
  bridged_large <- agexact.fit(
    large_x,
    large_y,
    strata = NULL,
    offset = rep(0, large_n),
    init = 0,
    control = coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, large_n),
    method = "exact",
    rownames = large_rows,
    resid = FALSE
  )
  reference_large <- survival::agexact.fit(
    large_x,
    large_y,
    strata = NULL,
    offset = rep(0, large_n),
    init = 0,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-09),
    weights = rep(1, large_n),
    method = "exact",
    rownames = large_rows,
    resid = FALSE
  )
  expect_equal(bridged_large$coefficients, reference_large$coefficients, tolerance = 1e-10)
  expect_equal(bridged_large$var, reference_large$var, tolerance = 1e-10)
  expect_equal(bridged_large$loglik, reference_large$loglik, tolerance = 1e-10)
  expect_equal(bridged_large$score, reference_large$score, tolerance = 1e-10)
  expect_equal(bridged_large$iter, reference_large$iter)
})

test_that("agexact.fit preserves exact iteration and final-trial semantics", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(
    reticulate::py_module_available("survival"),
    "Python survival package is unavailable"
  )

  halving_start <- c(0, 0, 0, 2, 6, 5, 8, 1, 5, 10)
  halving_stop <- c(1, 2, 2, 6, 7, 7, 10, 10, 11, 11)
  halving_event <- c(1, 1, 0, 0, 0, 0, 1, 0, 1, 1)
  halving_x <- matrix(c(
    -0.7949017577222064, -0.971412993595188, -0.8973640249385726,
    -0.31173291990153856, -0.9513748122177565, 1.204533321642762,
    2.883351093087066, -0.8851941617914357, 0.9821570019435617,
    0.251309986747342
  ), ncol = 1L)
  halving_offset <- c(
    0.3211436207164723, -0.1703476212015294, 0.2757700305715469,
    0.22219156001732224, 0.1257485571910997, -0.18593546677480463,
    0.3816069176371731, -0.694971257421976, 0.496755599155065,
    0.17014843430288
  )
  halving_y <- survival::Surv(halving_start, halving_stop, halving_event)
  halving_rows <- as.character(seq_along(halving_start))
  bridged_halving <- agexact.fit(
    halving_x,
    halving_y,
    strata = NULL,
    offset = halving_offset,
    init = 5.85354416465122,
    control = coxph.control(iter.max = 40L, eps = 1e-09),
    weights = rep(1, length(halving_start)),
    method = "exact",
    rownames = halving_rows,
    resid = FALSE
  )
  reference_halving <- survival::agexact.fit(
    halving_x,
    halving_y,
    strata = NULL,
    offset = halving_offset,
    init = 5.85354416465122,
    control = survival::coxph.control(iter.max = 40L, eps = 1e-09),
    weights = rep(1, length(halving_start)),
    method = "exact",
    rownames = halving_rows,
    resid = FALSE
  )
  expect_equal(bridged_halving$coefficients, reference_halving$coefficients, tolerance = 1e-09)
  expect_equal(bridged_halving$var, reference_halving$var, tolerance = 1e-08)
  expect_equal(bridged_halving$loglik, reference_halving$loglik, tolerance = 1e-09)
  expect_equal(bridged_halving$score, reference_halving$score, tolerance = 1e-09)
  expect_equal(bridged_halving$iter, reference_halving$iter)

  separated_start <- c(0, 0, 0, 1, 2, 2, 3)
  separated_stop <- c(1, 1, 1, 3, 3, 4, 5)
  separated_event <- c(1, 0, 0, 0, 0, 1, 0)
  separated_x <- matrix(c(
    -0.10627590772801489, -1.415108390302514, -0.5982619079224836,
    3.279520010161916, -1.334405338827207, 2.4961790201596363,
    0.1897036691116272
  ), ncol = 1L)
  separated_offset <- c(
    1.4884172952401737, -0.37680328079335645, -0.3108565122880977,
    -1.0850166921555693, 1.234396256848731, 0.42712866434784125,
    -0.15961611577080256
  )
  separated_y <- survival::Surv(separated_start, separated_stop, separated_event)
  separated_rows <- as.character(seq_along(separated_start))
  bridged_separated <- expect_warning(
    agexact.fit(
      separated_x,
      separated_y,
      strata = NULL,
      offset = separated_offset,
      init = 0.7083443262910332,
      control = coxph.control(iter.max = 20L, eps = 1e-09),
      weights = rep(1, length(separated_start)),
      method = "exact",
      rownames = separated_rows,
      resid = FALSE
    ),
    "Ran out of iterations and did not converge"
  )
  reference_separated <- expect_warning(
    survival::agexact.fit(
      separated_x,
      separated_y,
      strata = NULL,
      offset = separated_offset,
      init = 0.7083443262910332,
      control = survival::coxph.control(iter.max = 20L, eps = 1e-09),
      weights = rep(1, length(separated_start)),
      method = "exact",
      rownames = separated_rows,
      resid = FALSE
    ),
    "Ran out of iterations and did not converge"
  )
  expect_equal(bridged_separated$coefficients, reference_separated$coefficients, tolerance = 5e-07)
  expect_equal(bridged_separated$var, reference_separated$var, tolerance = 1e-06)
  expect_equal(bridged_separated$loglik, reference_separated$loglik, tolerance = 1e-09)
  expect_equal(bridged_separated$score, reference_separated$score, tolerance = 1e-09)
  expect_equal(bridged_separated$iter, reference_separated$iter)
})

test_that("cch unstratified fits match survival for right and counting data", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(0, 2, 1, 5, 4, 0, 10, 3, 12, 1, 5, 9, 0, 6, 2, 4, 7, 2, 11, 13),
    stop = c(5, 12, 3, 18, 9, 1, 15, 7, 20, 4, 11, 16, 2, 14, 6, 10, 13, 8, 17, 19),
    status = c(1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1),
    x = c(-1.2, .4, .9, -.3, 1.4, -.8, .2, 1.1, -.5, .7, -1, .1, 1.7, -.6, .5, -1.5, 1, -.1, .8, -.9),
    group = factor(rep(c("a", "b"), 10)),
    id = seq_len(20),
    subcohort = c(rep(1, 14), rep(0, 6))
  )

  compare_fit <- function(formula, method) {
    actual <- cch(
      formula,
      data,
      subcoh = ~subcohort,
      id = ~id,
      cohort.size = 80,
      method = method,
      robust = identical(method, "LinYing")
    )
    reference <- survival::cch(
      formula,
      data,
      subcoh = ~subcohort,
      id = ~id,
      cohort.size = 80,
      method = method,
      robust = identical(method, "LinYing")
    )

    expect_s3_class(actual, "cch")
    expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
    expect_equal(actual$var, reference$var, tolerance = 1e-11)
    expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
    expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
    expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
    expect_equal(actual$iter, reference$iter)
    expect_equal(actual$n, reference$n)
    expect_equal(actual$nevent, reference$nevent)
    expect_equal(actual$method, reference$method)
    expect_equal(actual$cohort.size, reference$cohort.size)
    expect_equal(actual$subcohort.size, reference$subcohort.size)
    expect_false(actual$stratified)
  }

  for (method in c("Prentice", "SelfPrentice", "LinYing")) {
    compare_fit(Surv(stop, status) ~ x + group, method)
    compare_fit(Surv(start, stop, status) ~ x + group, method)
  }
})

test_that("cch preserves small offset risk in counting-process data", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(1, 8, 6, 13, 3, 14, 4, 0, 10, 12, 16, 11, 4, 1, 13, 14, 6, 8, 9, 13, 4),
    stop = c(3, 13, 11, 15, 5, 16, 9, 1, 12, 14, 19, 16, 9, 3, 15, 18, 7, 9, 10, 15, 10),
    status = c(0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1),
    x = c(
      -2.36601980289781, -0.939726558703197, 0.672805414806265,
      -0.476183125795317, -0.636546918038443, -0.687008929997081,
      0.535019844914994, -0.210862903347529, 0.705276653609758,
      -0.678855799561129, -0.832078189498332, -0.956832544488333,
      -0.230958721600656, -0.542591235128462, -1.20622632983076,
      1.48683179341071, 1.28963847269521, 0.271588841450844,
      -1.63591043582559, -0.831208786158255, -0.890202805534755
    ),
    id = seq_len(21),
    subcohort = c(1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ x,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 63,
    method = "Prentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
})

test_that("cch masks aliases from the coefficient-producing fit", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  self_prentice_data <- data.frame(
    start = c(16, 6, 5, 5, 16, 0, 12, 4, 14, 16, 1, 5, 14, 1, 12, 9, 1, 15),
    stop = c(17, 8, 10, 6, 17, 2, 17, 6, 20, 17, 3, 7, 20, 4, 14, 15, 4, 17),
    status = c(1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0),
    group = factor(
      c("d", "d", "d", "d", "d", "b", "b", "b", "a", "a", "d", "a", "b", "c", "c", "d", "a", "a"),
      levels = c("a", "b", "c", "d")
    ),
    id = seq_len(18),
    subcohort = c(1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1)
  )
  self_args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = self_prentice_data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 54,
    method = "SelfPrentice"
  )
  actual_self <- expect_warning(
    do.call(cch, self_args),
    "Loglik converged before variable  1"
  )
  reference_self <- expect_warning(
    do.call(survival::cch, self_args),
    "Loglik converged before variable  1"
  )
  expect_equal(actual_self$coefficients, reference_self$coefficients, tolerance = 1e-11)
  expect_equal(actual_self$var, reference_self$var, tolerance = 1e-11)

  prentice_data <- data.frame(
    stop = c(7, 3, 6, 7, 9, 15, 13, 16, 13, 15, 6, 8, 17, 14, 8, 3, 12, 16, 1, 3),
    status = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1),
    group = factor(
      c("c", "a", "d", "b", "b", "b", "b", "a", "c", "d", "d", "d", "b", "b", "b", "a", "a", "b", "d", "b"),
      levels = c("a", "b", "c", "d")
    ),
    id = seq_len(20),
    subcohort = c(0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1)
  )
  prentice_args <- list(
    formula = Surv(stop, status) ~ group,
    data = prentice_data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 60,
    method = "Prentice"
  )
  actual_prentice <- expect_warning(
    do.call(cch, prentice_args),
    "Loglik converged before variable  3"
  )
  reference_prentice <- expect_warning(
    do.call(survival::cch, prentice_args),
    "Loglik converged before variable  3"
  )
  expect_equal(actual_prentice$coefficients, reference_prentice$coefficients, tolerance = 1e-11)
})

test_that("cch matches LinYing factor separation robust variance", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(18, 0, 14, 13, 9, 14, 13, 2, 0, 1, 5, 9, 8, 15, 4, 9, 17, 2, 16, 7, 16, 5, 14),
    stop = c(20, 3, 16, 18, 13, 17, 17, 4, 3, 4, 8, 12, 9, 19, 8, 12, 18, 7, 18, 11, 17, 8, 20),
    status = c(1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0),
    group = factor(
      c("b", "b", "d", "b", "a", "a", "c", "c", "b", "b", "d", "c", "d", "b", "c", "b", "a", "a", "a", "d", "d", "d", "b"),
      levels = c("a", "b", "c", "d")
    ),
    id = seq_len(23),
    subcohort = c(1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 69,
    method = "LinYing",
    robust = TRUE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Loglik converged before variable  1,2,3"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Loglik converged before variable  1,2,3"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
})

test_that("cch matches factor phase-two roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(1, 11, 4, 0, 2, 0, 13, 0, 0, 12, 15, 0, 14, 0, 3, 0, 1, 0, 2, 12, 17, 9, 8, 15, 4, 11),
    stop = c(7, 16, 7, 1, 3, 1, 19, 1, 4, 13, 16, 2, 15, 4, 6, 2, 2, 2, 8, 15, 19, 12, 11, 18, 6, 17),
    status = c(1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1),
    group = factor(c(
      "a", "b", "b", "b", "a", "a", "a", "c", "a", "a", "a", "c", "a",
      "c", "c", "a", "b", "c", "a", "b", "c", "a", "a", "c", "c", "a"
    )),
    id = seq_len(26),
    subcohort = c(0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ 0 + group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 78,
    method = "Prentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches factor counting offset mean roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(5, 0, 6, 17, 7, 0, 9, 6, 4, 10, 5, 0, 12, 4, 3, 14, 9, 15, 15, 13, 0, 14, 0, 2, 2),
    stop = c(9, 1, 11, 20, 9, 1, 11, 9, 7, 14, 8, 2, 18, 9, 4, 16, 10, 17, 16, 14, 4, 15, 1, 5, 7),
    status = c(0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1),
    group = factor(c(
      "b", "b", "b", "b", "b", "a", "a", "a", "b", "b", "b", "c", "b",
      "a", "b", "b", "b", "a", "a", "c", "b", "a", "b", "a", "a"
    )),
    id = seq_len(25),
    subcohort = c(1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 75,
    method = "Prentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches scalar counting offset mean roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(9, 10, 5, 1, 8, 16, 3, 10, 13, 0, 0, 3, 0, 9, 5, 13, 16, 11, 10, 10, 7, 10, 4, 5, 18, 11, 5),
    stop = c(12, 13, 6, 3, 14, 17, 6, 13, 19, 4, 6, 5, 4, 13, 7, 18, 20, 12, 16, 11, 10, 12, 7, 10, 19, 14, 11),
    status = c(1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1),
    x = c(
      0.008926919521119415, 0.9060546400253241, 1.8322206881267567,
      -0.5487920544142219, -0.38538157279838425, -0.27423854220727134,
      -0.12575800060661754, 0.0282533678905421, -0.14233207563149147,
      -0.19480425083891662, -1.273056905241701, -0.3860241881622384,
      -2.341092420046039, -0.2414972829893251, -0.48662182877357135,
      0.5613828624512708, -0.5624384123439998, -1.0681337423459547,
      0.31883033976783115, -0.8352213781406432, -0.009302242429820969,
      -0.5042953637548548, -0.6629703130264287, -1.0188322597824033,
      -0.41716297872025515, 0.028744936514534088, 0.8739803580994467
    ),
    id = seq_len(27),
    subcohort = c(1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ x,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 81,
    method = "SelfPrentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches Lin-Ying counting separation roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(0, 9, 3, 15, 2, 15, 0, 9, 10, 0, 8, 16, 0, 0, 3, 3, 2, 10, 7),
    stop = c(2, 12, 5, 17, 6, 20, 2, 11, 16, 2, 9, 19, 2, 1, 8, 9, 8, 12, 12),
    status = c(0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
    group = factor(c("a", "d", "c", "b", "d", "d", "c", "a", "c", "b", "d", "c", "c", "d", "a", "c", "b", "c", "d")),
    id = seq_len(19),
    subcohort = c(1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 57,
    method = "LinYing",
    robust = FALSE
  )
  actual <- suppressWarnings(do.call(cch, args))
  reference <- suppressWarnings(do.call(survival::cch, args))

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches Prentice extreme phase-two roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  z <- c(
    -0x1.25b9dca471164p-2, 0x1.fd2dbe115d3c3p-3, -0x1.fd955cb44cb4cp-2,
    0x1.0617df82127a9p-2, 0x1.4135dc186d9d8p-5, 0x1.09c3a94e34199p-2,
    0x1.556a4a939bae1p-3, -0x1.baca2a170cc3ap-2, 0x1.3f2fb73692a7p-5,
    -0x1.c0f0aebdb304ep-3, 0x1.36dfa9a276d1ep-3, -0x1.61b13998c9f4p-3,
    -0x1.304b02c403001p-1, 0x1.b567268e84f76p-1, -0x1.54646fe3fc80ep-3,
    0x1.6d6fc94fc2cdbp+0, 0x1.16ef37594742bp+0, -0x1.c898e8df761cep-1,
    -0x1.6e4aca9103164p-4, 0x1.3d99bfbc4fd8dp-2, -0x1.bdf209e71dadbp-1,
    0x1.8cf8c702ac6e6p-5, -0x1.b68b9f5e4028fp-1, -0x1.e1b0f40cb0b7fp-1,
    -0x1.192737a6cc726p+0, 0x1.01544edab80f7p-1, -0x1.b8e91470d006dp+0,
    -0x1.0ff524ce8fa3fp-1, -0x1.01c3fb2bb56d7p+0
  )
  data <- data.frame(
    stop = c(12, 2, 2, 14, 17, 2, 2, 14, 5, 20, 20, 9, 13, 13, 3, 3, 15, 3, 4, 5, 11, 19, 15, 13, 3, 19, 20, 2, 10),
    status = c(1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1),
    x = seq_len(29) - 1,
    z = z,
    id = seq_len(29),
    subcohort = c(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0)
  )
  args <- list(
    formula = Surv(stop, status) ~ 0 + x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 87,
    method = "Prentice",
    robust = FALSE
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches Prentice nonconverged factor roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(3, 14, 11, 4, 0, 10, 8, 12, 0, 8, 17, 11, 8, 8, 15, 15, 18, 0, 0, 0, 6, 5, 2, 7),
    stop = c(7, 16, 12, 7, 6, 13, 13, 15, 3, 12, 20, 14, 9, 11, 18, 16, 20, 1, 2, 2, 11, 10, 8, 10),
    status = c(0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
    group = factor(c("b", "b", "c", "b", "b", "a", "c", "b", "a", "c", "a", "a", "c", "c", "a", "c", "b", "c", "c", "b", "c", "a", "a", "a")),
    id = seq_len(24),
    subcohort = c(1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 72,
    method = "Prentice",
    robust = TRUE
  )
  actual <- suppressWarnings(do.call(cch, args))
  reference <- suppressWarnings(do.call(survival::cch, args))

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches SelfPrentice scalar rank-change nonconvergence", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  z <- c(
    -0x1.07f5a4b214a4bp+1, -0x1.f38591f4ca8ecp-2, -0x1.c4f10762d8968p-3,
    0x1.ead98e4367054p-1, 0x1.68ca02e83ae8dp-1, 0x1.86d48b04043d2p+0,
    -0x1.5d3edf9d07ec3p+0, 0x1.2e7c7289d4334p-1, -0x1.323db64df3146p-2,
    0x1.491e143954abcp-2, -0x1.1d02ca196b41fp-2, -0x1.869a32eae2edfp-1,
    0x1.d12ebee1e59fcp+0, -0x1.db0f80cbd6f65p-3, -0x1.4f7c463e94cc1p+0,
    0x1.b0ae59554e4edp-1, -0x1.f898caeebe89fp-1, -0x1.31756b6da5469p-1,
    -0x1.b8ed5db64786dp-3, -0x1.9429e7fd46204p+0
  )
  data <- data.frame(
    start = c(4, 5, 8, 16, 4, 7, 12, 10, 4, 0, 4, 18, 7, 14, 4, 5, 0, 15, 0, 15),
    stop = c(7, 11, 10, 20, 10, 12, 13, 13, 6, 1, 10, 19, 13, 15, 9, 6, 3, 16, 4, 18),
    status = c(1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1),
    x = seq_len(20) - 1,
    z = z,
    id = seq_len(20),
    subcohort = c(1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ 0 + x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 60,
    method = "SelfPrentice",
    robust = FALSE
  )
  actual <- suppressWarnings(do.call(cch, args))
  reference <- suppressWarnings(do.call(survival::cch, args))

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch retains a non-finite SelfPrentice final trial", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- c(
    0x1.6ddbe4e80f32cp+1, -0x1.21bd362f7848ep+0, 0x1.0cfd9f38f0233p+0,
    -0x1.3efb77bb09e59p-1, -0x1.3f7842a77d9afp+0, 0x1.e91bf55435d6cp-1,
    0x1.f8db9f93db011p-1, 0x1.c71eda8a6aa8fp-1, 0x1.13b1929f8d551p+1,
    -0x1.fea2c130fe8ap+0, -0x1.7534e6d009c66p-1, -0x1.c2f70106afa1bp-1,
    0x1.815fffb8e9f8bp-2, -0x1.14ed9b9588632p+0, -0x1.587326d9d9f4ep-3,
    -0x1.bd7259f1061b7p+0, -0x1.87b3854ba4eadp-4, 0x1.f371f60d12813p-1,
    0x1.584db496043a7p-4, 0x1.3d0de0a71db89p-1, 0x1.275b93b2d2039p-2,
    -0x1.13c902c7ee50dp-1, 0x1.839745fe36c7fp-1, 0x1.62cf8806c1a28p-2,
    0x1.7a6f1cbbc3845p-1, -0x1.780ab09a7e8b3p+0
  )
  data <- data.frame(
    start = c(3, 2, 7, 10, 7, 5, 0, 12, 0, 12, 2, 6, 1, 0, 6, 9, 0, 5, 15, 9, 1, 16, 14, 12, 13, 14),
    stop = c(8, 8, 10, 13, 11, 6, 1, 18, 4, 18, 3, 7, 5, 5, 7, 13, 1, 7, 20, 11, 3, 20, 16, 15, 16, 18),
    status = c(1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    x = x,
    id = seq_len(26),
    subcohort = c(0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ x,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 78,
    method = "SelfPrentice",
    robust = FALSE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Ran out of iterations and did not converge"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Ran out of iterations and did not converge"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches SelfPrentice right-censored scalar phase-two roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  z <- c(
    -0x1.3b2b5721b82a2p-2, 0x1.b450f3616ca93p-3, -0x1.19f6abf922eaep+0,
    0x1.d738bf00c245fp-3, 0x1.c6ba052fe3987p-3, -0x1.d6186479fcf9dp+0,
    -0x1.da762c079bd03p-1, 0x1.6ed6a701966d1p-1, -0x1.ce7388f116d0ap-3,
    0x1.38bf164899229p-1, 0x1.5105afd19667ap+0, 0x1.2de1bd5a68ec9p+0,
    -0x1.6db7d15d4011p+0, 0x1.c75c688b2ece1p-1, -0x1.4badfa6f5e01ep-2,
    -0x1.3e341301523a1p+0, -0x1.242e66da6d4ep-1, -0x1.9b578604efc38p-1,
    -0x1.8dd68605d0f95p-1, -0x1.04ee71b26bd64p-2, -0x1.64040b3143b2dp-5
  )
  data <- data.frame(
    stop = c(14, 9, 20, 4, 7, 10, 7, 10, 17, 8, 9, 17, 9, 3, 17, 3, 12, 16, 14, 17, 13),
    status = c(0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1),
    x = seq_len(21) - 1,
    z = z,
    id = seq_len(21),
    subcohort = c(1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1)
  )
  args <- list(
    formula = Surv(stop, status) ~ 0 + x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 63,
    method = "SelfPrentice",
    robust = FALSE
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches Prentice two-covariate predictor roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- c(
    0x1.37aa3180b8ec8p+0, -0x1.b5940fb15612bp-1, -0x1.36cc947ee6cd6p+0,
    0x1.c729a5b474629p+0, -0x1.c87ce326f6dc3p+0, 0x1.1510e744c7addp+0,
    -0x1.5c2b127324f7fp-6, -0x1.eac94a40acf16p-2, 0x1.02e1d6600c625p-2,
    0x1.724c0c3e6908dp-2, -0x1.83ed31999bf0fp-2, 0x1.16725bb35aff9p-3,
    0x1.253b0de2973afp+0, -0x1.28772ea8ed503p-2, 0x1.284fe82a35859p-1,
    -0x1.15249b635980ap+1, 0x1.1dc85d9acdc5fp+0, -0x1.34d34b80c4667p-2,
    -0x1.f7c1ff818fc48p+0, 0x1.7dfd1c7acd633p-1, -0x1.4025e3cfa0204p+0,
    0x1.84b02c31ce6f8p+0, -0x1.41a49d0a0d7fap-3, -0x1.9d60e78b8bd95p-4,
    0x1.4b3364109360dp-2, -0x1.7bb192612675fp+0, -0x1.569c4102d25f1p-1,
    0x1.5f5d7ad270f56p-7
  )
  z <- c(
    -0x1.6508ade602132p+0, -0x1.8666e99ceb88cp-2, 0x1.2c1b64e3a753cp-3,
    -0x1.d9ee3f6c3d9bfp-1, 0x1.4c476a694760cp+0, 0x1.0fdb28c11a00fp-1,
    -0x1.2191bdf72fe5ap-1, 0x1.9d35b2fb1a1d5p-1, -0x1.e92c887380936p+0,
    -0x1.21ed4c5b98941p+1, -0x1.076edf2baa016p+0, -0x1.1393b8d4a9313p+0,
    0x1.5c3e9bee218fap+0, 0x1.9a2900dd9ed8dp-2, -0x1.26b29ee964657p-2,
    -0x1.4dcb669a5bd33p-2, 0x1.5afb2f25a5b34p-1, -0x1.110e63b3dde7p-2,
    -0x1.a04991aa0e15dp+0, 0x1.edbcc383514f9p-3, 0x1.632d54ef597b8p-1,
    -0x1.4cdf474450a7dp-1, -0x1.508596ff2b455p+0, 0x1.38ce309304b8cp-1,
    0x1.5240249a2ddb6p-2, 0x1.51fcdb931123ep-2, -0x1.05f395993cd2bp+0,
    0x1.201f87d5a5d5cp+0
  )
  data <- data.frame(
    stop = c(7, 7, 20, 7, 17, 17, 14, 3, 14, 11, 5, 13, 18, 11, 2, 6, 15, 4, 10, 1, 19, 15, 4, 12, 2, 3, 6, 7),
    status = c(1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1),
    x = x,
    z = z,
    id = seq_len(28),
    subcohort = c(1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)
  )
  args <- list(
    formula = Surv(stop, status) ~ x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 84,
    method = "Prentice",
    robust = FALSE
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches nonconverged Prentice tied-duplicate variance", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- c(
    0x1.2641d22ff7141p+1, 0x1.32a9e2f3e92ffp-4, -0x1.62690e707a5dcp-3,
    0x1.9b01e854d4df3p-5, -0x1.f12b42bde91e8p-2, -0x1.a9905d2a4ea91p-2,
    -0x1.4c68927f59f92p+0, 0x1.9b0f3c3a34615p-1, -0x1.a020e5291b6ep-1,
    0x1.18d54c9da6facp-1, -0x1.7b346f44c4f3fp-3, 0x1.4e72dd5ca9e8dp-1,
    0x1.724c1c3470da8p-3, 0x1.9f6b452d1a2fdp-2, -0x1.2e3d196b7dda4p-1,
    -0x1.0ca35e51a19dcp+0, -0x1.ecafad4704c6ep-1, 0x1.3d72d053e528p-1,
    -0x1.6da1f1e9831adp-4, 0x1.6b80257cc989ep-1
  )
  z <- c(
    -0x1.4ea65c96ddefep-1, -0x1.3790d11f31b98p+1, -0x1.692e0367461fep+1,
    -0x1.dc35361c3259ep-1, 0x1.5e8e2b705a38dp-2, 0x1.c14d4a896d9abp+0,
    0x1.2c9d056e93d9fp+0, -0x1.166286ca8d973p+0, -0x1.1f2aa6807cf44p-4,
    0x1.1f09e2306f2f7p+0, -0x1.017c7be522f6ap+1, 0x1.ed5675d2ce60ap+0,
    0x1.64b7a449207ccp-1, -0x1.4049fb30963fep-6, 0x1.187b70b31d4ffp-1,
    -0x1.b5ff4b1929383p-4, -0x1.915d3f9e511b5p-1, -0x1.416a90a02da9bp+0,
    -0x1.71730b6ff4c91p-1, -0x1.b6aaffc6b4385p-1
  )
  data <- data.frame(
    stop = c(20, 18, 8, 4, 16, 15, 10, 11, 11, 3, 16, 12, 17, 17, 20, 5, 5, 15, 18, 10),
    status = c(1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    x = x,
    z = z,
    id = seq_len(20),
    subcohort = c(0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1)
  )
  args <- list(
    formula = Surv(stop, status) ~ x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 60,
    method = "Prentice",
    robust = FALSE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Ran out of iterations and did not converge"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Ran out of iterations and did not converge"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch preserves finite Prentice variance after nonconvergence", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(16, 2, 0, 13, 11, 9, 5, 0, 0, 2, 6, 14, 17, 9, 5, 0, 11, 4, 14),
    stop = c(19, 4, 1, 17, 16, 12, 10, 3, 1, 3, 9, 20, 19, 12, 8, 2, 13, 6, 19),
    status = c(1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    x = c(
      1.31902037016325, -1.55895861820514, 0.09023030919103,
      1.38657406706107, -1.15866615725249, 0.0750902109213203,
      -1.09472225388637, -0.0821319073337703, 0.524933329209907,
      0.210932736086887, 0.450733636825875, -0.249683060507394,
      -1.29703341378867, -0.399826580234449, 0.547798751664738,
      2.07445629664622, -0.65234722868968, -0.0684031851667403,
      0.466825159366182
    ),
    z = c(
      0.268665058925964, -0.130595287012515, 2.72684347576924,
      0.668735481964197, -0.0643360648937243, -0.555460924011281,
      -0.649504574636737, -0.178797409445533, -0.321294398319967,
      0.868148709391882, 0.54346099134323, -0.623271515284646,
      0.700304382098733, -0.653834222693076, 0.00041222982414799,
      2.21504230489358, 0.913000347229081, 0.186292785158738,
      -0.327810686175575
    ),
    id = seq_len(19),
    subcohort = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 57,
    method = "Prentice",
    robust = FALSE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Ran out of iterations and did not converge"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Ran out of iterations and did not converge"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches nonconverged SelfPrentice factor aliases", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(10, 13, 9, 6, 0, 5, 6, 15, 0, 8, 7, 12, 14, 13, 9, 9, 8, 16, 0, 10, 13, 13),
    stop = c(11, 15, 12, 10, 6, 10, 8, 19, 1, 9, 9, 16, 17, 18, 15, 14, 12, 18, 2, 14, 15, 14),
    status = c(0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0),
    group = factor(c(
      "d", "b", "d", "d", "d", "a", "c", "d", "a", "a", "a",
      "d", "a", "a", "d", "b", "b", "a", "b", "d", "d", "a"
    )),
    id = seq_len(22),
    subcohort = c(1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 66,
    method = "SelfPrentice",
    robust = FALSE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Ran out of iterations and did not converge"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Ran out of iterations and did not converge"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches Prentice near-singular factor phase-two variance", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(13, 3, 12, 6, 0, 7, 0, 15, 14, 0, 2, 8, 16, 8, 0, 1, 9, 6, 1, 8, 17, 1, 15, 8, 0, 10, 0, 8, 13, 10, 2),
    stop = c(19, 4, 17, 7, 1, 13, 1, 17, 18, 2, 7, 14, 19, 11, 1, 3, 15, 8, 2, 14, 19, 5, 16, 14, 5, 15, 1, 12, 17, 11, 7),
    status = c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1),
    group = factor(strsplit("aabccacacabbcbcacacbcccbbcbcabc", "")[[1L]]),
    id = seq_len(31),
    subcohort = c(0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 93,
    method = "Prentice",
    robust = FALSE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Loglik converged before variable  2"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Loglik converged before variable  2"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches LinYing nonconverged centered variance", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(18, 12, 1, 11, 10, 9, 8, 4, 0, 5, 10, 5, 8, 1, 4, 0, 10, 12, 1),
    stop = c(20, 15, 3, 13, 12, 14, 11, 9, 1, 8, 11, 6, 13, 5, 10, 3, 15, 15, 5),
    status = c(1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    x = c(
      0x1.25029925b3eb1p-1, 0x1.016b9397d9ed3p+0, 0x1.19221663c0e23p-2,
      -0x1.7ac80406f2ddap+0, 0x1.be8145ab497cep+0, 0x1.2fcc10eab20b3p-3,
      -0x1.4324db8dc025fp-2, -0x1.7c273bafc6ddep-3, 0x1.99d279827da42p-1,
      0x1.ee4b394a992f3p-1, 0x1.95fc3cc20fe97p+0, -0x1.4ee4aa3e51baap-2,
      -0x1.16364996ff5f9p-1, -0x1.6c431223cf01p-1, -0x1.b7333c66af40dp-1,
      -0x1.03450f5ab387dp-2, 0x1.dc73eef9a56fdp-2, -0x1.45d040e5409dap-2,
      -0x1.26c2e258be9abp+0
    ),
    z = c(
      0x1.db6ca27a94789p-3, -0x1.e61490378b997p+0, 0x1.74aeca0c25442p-2,
      0x1.2becfff5064f4p-1, -0x1.76470538b9d62p-1, -0x1.0f35ac0de7bbfp+1,
      -0x1.67d12ff0b7115p-1, 0x1.0651f2dbcf95ap-1, -0x1.f309b92d00d1ep-3,
      0x1.d3ca8c1c8f098p-3, 0x1.1f5f248443812p+0, 0x1.cccf3c8cd5c3fp-2,
      -0x1.9e254a54746afp-2, -0x1.1b6d17cf80072p-2, -0x1.f34fb99242335p-3,
      0x1.fdee04ffe069cp-2, -0x1.f88f53a052bebp-1, 0x1.3ff505fc806fap+0,
      -0x1.eaffc691a5a5p-4
    ),
    id = seq_len(19),
    subcohort = c(1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 57,
    method = "LinYing",
    robust = TRUE
  )
  actual <- expect_warning(
    do.call(cch, args),
    "Ran out of iterations and did not converge"
  )
  reference <- expect_warning(
    do.call(survival::cch, args),
    "Ran out of iterations and did not converge"
  )

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches delayed-entry factor roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(6, 0, 0, 5, 0, 14, 5, 1, 2, 2, 7, 1, 3, 7, 0, 5, 9, 0, 4, 0),
    stop = c(12, 3, 2, 8, 2, 17, 10, 6, 3, 3, 8, 5, 8, 11, 1, 8, 11, 5, 9, 1),
    status = c(0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    group = factor(c(
      "a", "a", "b", "b", "c", "a", "c", "a", "a", "b",
      "c", "b", "c", "b", "c", "a", "b", "b", "b", "b"
    )),
    id = seq_len(20),
    subcohort = c(1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1)
  )
  args <- list(
    formula = Surv(start, stop, status) ~ 0 + group,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 60,
    method = "Prentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches SelfPrentice phase-two roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    stop = c(9, 2, 11, 17, 13, 16, 9, 17, 2, 14, 4, 5, 9, 3, 5, 14, 13, 11, 12, 13, 19, 5, 17, 7, 11, 9, 17, 13, 19, 2, 7, 17, 11, 6, 17, 20),
    status = c(1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1),
    x = c(
      -0.26849665951131463, 0.33551346520048575, 1.543538272789444,
      0.73661127836345663, -1.2931284399001775, 0.20094568317439268,
      1.0082531556307415, 0.36651548728375172, -0.90554602331239875,
      -0.66861007668806816, -1.4508885415021988, 1.1124700969400358,
      0.20691196542930951, 0.76727433862922856, 0.40376254107728937,
      -2.0841535896256036, -1.6417385233227353, 0.752711012030994,
      1.9247483622032153, 0.51576705111040244, -0.28636666029918739,
      0.94570058481698593, -0.59122726943043613, 1.1691592318960138,
      1.0297797444285017, 0.22028710327787904, -0.22325552783457839,
      -0.10624534176536009, 1.0879408908080794, -2.4519626063433879,
      -1.0310589755661379, 0.38879081635885104, -0.60509562057191779,
      0.49475675452226514, 0.19809623388605108, 0.15275431649864765
    ),
    id = seq_len(36),
    subcohort = c(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0)
  )
  args <- list(
    formula = Surv(stop, status) ~ x,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 108,
    method = "SelfPrentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch matches two-covariate SelfPrentice roundoff", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    stop = c(19, 8, 8, 8, 7, 13, 13, 2, 20, 14, 4, 8, 19, 5, 18, 6, 15, 9, 1, 12, 12, 18, 2, 20, 15, 10, 6, 15, 1, 13, 6, 11, 1),
    status = c(0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1),
    x = c(
      -1.0188108160578102, -0.06006343997276465, 0.075588774970652223,
      0.89506083963762928, 0.19110021977866368, -0.20623932062329439,
      -1.6741240206814509, -1.4071972050156059, 0.52300784650598731,
      0.42592948254676222, -0.69774041775912565, 1.0593278488582005,
      -1.9651020544285975, 1.5258169580249781, -0.47542235054327764,
      0.99166635349452226, 0.26224205091380309, 0.47944456907861666,
      -1.4713571923361715, -0.27438818976879015, 0.33983802244882827,
      -1.8238891303746787, 0.38332598501929044, -0.46236488228856687,
      0.76624156431781332, 0.32347679772308208, -0.6493927665595044,
      0.96186734701855436, -1.24769382735025, -1.3601816602575989,
      1.1940233969867884, 1.3314515626782069, -1.8014240641230983
    ),
    z = c(
      1.8158298507915824, -0.81170172799251716, -0.24377867269480011,
      -0.84025554480832887, 0.46699423913797639, -0.97556436610254016,
      -1.0784499937301986, -1.0601047577264089, -0.43192353283011414,
      -0.67913187648187734, -0.32891309960287968, 0.66543471207019378,
      -0.051742381155122856, -1.371400790442056, 0.092725713344919969,
      -2.1454921491553725, -0.34602642876625794, 0.56975127863342678,
      1.4259471174533569, 1.2584774158801246, -1.3840315126637694,
      0.86893351500048044, 0.48295972731585529, 2.0014079335957939,
      0.80484732900251377, 1.1548064445076267, -0.98676465786452616,
      1.3665447338966386, -0.39883367868212188, -1.0561834407739583,
      -0.82503110502992338, 0.042151325226441716, -0.2301161647392603
    ),
    id = seq_len(33),
    subcohort = c(1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1)
  )
  args <- list(
    formula = Surv(stop, status) ~ x + z,
    data = data,
    subcoh = ~subcohort,
    id = ~id,
    cohort.size = 99,
    method = "SelfPrentice"
  )
  actual <- do.call(cch, args)
  reference <- do.call(survival::cch, args)

  expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
  expect_equal(actual$var, reference$var, tolerance = 1e-11)
  expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
  expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
  expect_equal(actual$means, reference$means, tolerance = 1e-11)
  expect_equal(actual$loglik, reference$loglik, tolerance = 1e-11)
  expect_equal(actual$score, reference$score, tolerance = 1e-11)
  expect_equal(actual$iter, reference$iter)
  expect_equal(actual$offset, reference$offset, tolerance = 1e-11)
})

test_that("cch stratified Borgan fits match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    start = c(0, 2, 1, 5, 4, 0, 10, 3, 12, 1, 5, 9, 0, 6, 2, 4, 7, 2, 11, 13),
    stop = c(5, 12, 3, 18, 9, 1, 15, 7, 20, 4, 11, 16, 2, 14, 6, 10, 13, 8, 17, 19),
    status = c(1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1),
    x = c(-1.2, .4, .9, -.3, 1.4, -.8, .2, 1.1, -.5, .7, -1, .1, 1.7, -.6, .5, -1.5, 1, -.1, .8, -.9),
    z = c(0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1),
    sampling = factor(rep(c("a", "b"), 10)),
    id = seq_len(20),
    subcohort = c(rep(1, 14), rep(0, 6))
  )
  cohort_sizes <- c(a = 40, b = 40)

  compare_fit <- function(formula, method) {
    actual <- cch(
      formula,
      data,
      subcoh = ~subcohort,
      id = ~id,
      stratum = ~sampling,
      cohort.size = cohort_sizes,
      method = method
    )
    reference <- survival::cch(
      formula,
      data,
      subcoh = ~subcohort,
      id = ~id,
      stratum = ~sampling,
      cohort.size = cohort_sizes,
      method = method
    )

    expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
    expect_equal(actual$var, reference$var, tolerance = 1e-11)
    expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
    expect_equal(actual$phase2var, reference$phase2var, tolerance = 1e-11)
    expect_equal(actual$opt, reference$opt, tolerance = 1e-11)
    expect_equal(actual$delta, reference$delta, tolerance = 1e-11)
    expect_equal(actual$sc, reference$sc, tolerance = 1e-11)
    expect_equal(actual$stratum, reference$stratum)
    expect_equal(actual$subcohort.size, reference$subcohort.size)
    expect_true(actual$stratified)
  }

  for (method in c("I.Borgan", "II.Borgan")) {
    compare_fit(Surv(stop, status) ~ x + z, method)
    compare_fit(Surv(start, stop, status) ~ x + z, method)
  }
})

test_that("cch rejects invalid unstratified sampling inputs", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 3, 4),
    status = c(1, 0, 1, 1),
    x = c(-1, 0, 1, 2),
    id = seq_len(4),
    subcohort = c(1, 1, 0, 0)
  )
  expect_error(
    cch(
      Surv(time, status) ~ x,
      data,
      subcoh = data$subcohort,
      id = data$id,
      cohort.size = 10,
      method = "I.Borgan"
    ),
    "requires 'stratum'"
  )
  invalid <- data$subcohort
  invalid[[2L]] <- 0
  expect_error(
    cch(
      Surv(time, status) ~ x,
      data,
      subcoh = invalid,
      id = data$id,
      cohort.size = 10
    ),
    "censored observations not in subcohort"
  )
})

test_that("cch drops the reference model-matrix column", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    stop = c(5, 12, 3, 18, 9, 1, 15, 7, 20, 4, 11, 16, 2, 14, 6, 10, 13, 8, 17, 19),
    status = c(1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1),
    x = c(-1.2, .4, .9, -.3, 1.4, -.8, .2, 1.1, -.5, .7, -1, .1, 1.7, -.6, .5, -1.5, 1, -.1, .8, -.9),
    z = c(0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1),
    group = factor(rep(c("a", "b"), 10)),
    id = seq_len(20),
    subcohort = c(rep(1, 14), rep(0, 6))
  )
  common <- list(data = data, subcoh = ~subcohort, id = ~id, cohort.size = 80)
  expect_error(
    do.call(cch, c(list(formula = Surv(stop, status) ~ 1), common)),
    "subscript out of bounds",
    fixed = TRUE
  )
  expect_error(
    do.call(cch, c(list(formula = Surv(stop, status) ~ 0 + x), common)),
    "subscript out of bounds",
    fixed = TRUE
  )

  compare_shape <- function(formula) {
    actual <- do.call(cch, c(list(formula = formula), common))
    reference <- do.call(survival::cch, c(list(formula = formula), common))
    expect_equal(actual$coefficients, reference$coefficients, tolerance = 1e-11)
    expect_equal(actual$var, reference$var, tolerance = 1e-11)
    expect_equal(actual$naive.var, reference$naive.var, tolerance = 1e-11)
    expect_equal(actual$means, reference$means, tolerance = 1e-11)
    expect_equal(actual$x, reference$x, tolerance = 1e-11)
    expect_equal(actual$assign, reference$assign)
  }
  compare_shape(Surv(stop, status) ~ x)
  compare_shape(Surv(stop, status) ~ 0 + x + z)
  compare_shape(Surv(stop, status) ~ 0 + group)
})

test_that("low-level Cox survival curves match right and counting-process fits", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- cbind(
    x1 = c(0.2, 0.8, -0.3, 1.1, 0.4, -0.7, 0.6, -0.1),
    x2 = c(1, 0, 1, 2, -1, 0.5, 1.5, -0.5)
  )
  stop <- c(1, 2, 2, 3, 4, 4, 2.5, 5)
  status <- c(1, 1, 0, 1, 0, 1, 1, 0)
  start <- c(0, 0.5, 0, 1.5, 2, 1, 0.25, 3.5)
  weights <- c(1, 0.5, 2, 1.5, 1, 0.75, 1.25, 0.8)
  risk <- exp(c(-0.2, 0.3, 0.1, -0.4, 0.5, 0.2, -0.1, 0.4))
  strata_value <- factor(c("a", "a", "a", "a", "b", "b", "b", "b"))
  prediction_x <- rbind(first = c(0.1, 0.2), second = c(-0.2, 0.6))
  prediction_risk <- exp(c(0.12, -0.21))
  variance <- matrix(c(0.08, 0.01, 0.01, 0.05), 2L)

  compare_fit <- function(y, ctype, stype, se.fit, unlist) {
    args <- list(
      ctype = ctype,
      stype = stype,
      se.fit = se.fit,
      varmat = variance,
      y = y,
      x = x,
      wt = weights,
      risk = risk,
      strata = strata_value,
      x2 = prediction_x,
      risk2 = prediction_risk,
      unlist = unlist
    )
    expect_equal(
      do.call(coxsurv.fit, args),
      do.call(survival::coxsurv.fit, args),
      tolerance = 1e-10
    )
  }

  for (y in list(cbind(stop, status), cbind(start, stop, status))) {
    for (ctype in 1:2) {
      for (stype in 1:2) {
        compare_fit(y, ctype, stype, FALSE, TRUE)
        compare_fit(y, ctype, stype, TRUE, FALSE)
      }
    }
  }

  one_time_args <- list(
    ctype = 1L,
    stype = 2L,
    se.fit = TRUE,
    varmat = matrix(0.1),
    y = cbind(c(1, 1), c(1, 0)),
    x = cbind(c(0.2, 0.3)),
    wt = c(1, 1),
    risk = c(1, 1),
    x2 = rbind(one = 0.1, two = 0.2),
    risk2 = c(1, 2),
    unlist = TRUE
  )
  expect_equal(
    do.call(coxsurv.fit, one_time_args),
    do.call(survival::coxsurv.fit, one_time_args),
    tolerance = 1e-10
  )
})

test_that("low-level Cox survival curves match individual trajectories", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  x <- cbind(
    x1 = c(0.2, 0.8, -0.3, 1.1, 0.4, -0.7, 0.6, -0.1),
    x2 = c(1, 0, 1, 2, -1, 0.5, 1.5, -0.5)
  )
  start <- c(0, 0.5, 0, 1.5, 2, 1, 0.25, 3.5)
  stop <- c(1, 2, 2, 3, 4, 4, 2.5, 5)
  status <- c(1, 1, 0, 1, 0, 1, 1, 0)
  strata_value <- factor(c("a", "a", "a", "a", "b", "b", "b", "b"))
  y2 <- rbind(c(0, 2.5), c(2.5, 5), c(0, 3), c(3, 5))
  x2 <- rbind(c(0.1, 0.2), c(0.3, -0.1), c(-0.2, 0.6), c(0.4, 0.5))
  args <- list(
    ctype = 2L,
    stype = 2L,
    se.fit = TRUE,
    varmat = matrix(c(0.08, 0.01, 0.01, 0.05), 2L),
    y = cbind(start, stop, status),
    x = x,
    wt = c(1, 0.5, 2, 1.5, 1, 0.75, 1.25, 0.8),
    risk = exp(c(-0.2, 0.3, 0.1, -0.4, 0.5, 0.2, -0.1, 0.4)),
    strata = strata_value,
    y2 = y2,
    x2 = x2,
    risk2 = exp(c(0.12, 0.07, -0.21, 0.18)),
    strata2 = c(1L, 2L, 1L, 2L),
    id2 = c("one", "one", "two", "two"),
    unlist = TRUE
  )
  expect_equal(
    do.call(coxsurv.fit, args),
    do.call(survival::coxsurv.fit, args),
    tolerance = 1e-10
  )

  wrapper_args <- list(
    y = args$y,
    x = args$x,
    wt = args$wt,
    x2 = args$x2,
    risk = args$risk,
    newrisk = args$risk2,
    strata = args$strata,
    se.fit = args$se.fit,
    survtype = 3L,
    vartype = 3L,
    varmat = args$varmat,
    id = args$id2,
    y2 = args$y2,
    strata2 = args$strata2,
    unlist = TRUE
  )
  expect_equal(
    do.call(survfitcoxph.fit, wrapper_args),
    do.call(survival::survfitcoxph.fit, wrapper_args),
    tolerance = 1e-10
  )
})

test_that("ordinary istate inputs match R model-frame semantics", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 3, 4, 5, 6, 7, 8),
    status = c(1, 1, 0, 1, 0, 1, 0, 1),
    x = c(0.2, 0.8, 0.4, 1.1, 0.7, 0.3, 1.3, 0.5),
    state = factor(c("entry", "other", "entry", "other", "entry", "other", "entry", "other"))
  )
  reference_survfit <- getS3method("survfit", "formula", envir = asNamespace("survival"))

  bridged_curve <- survfit(
    Surv(time, status) ~ 1,
    data = data,
    istate = state,
    model = TRUE
  )
  reference_curve <- reference_survfit(
    survival::Surv(time, status) ~ 1,
    data = data,
    istate = state,
    model = TRUE
  )
  expect_equal(bridged_curve$surv, reference_curve$surv, tolerance = 1e-12)
  bridged_curve_frame <- model.frame(bridged_curve)
  curve_istate_name <- grep("istate", names(bridged_curve_frame), value = TRUE)
  expect_length(curve_istate_name, 1L)
  expect_equal(as.character(bridged_curve_frame[[curve_istate_name]]), as.character(data$state))

  bridged_fit <- coxph(
    Surv(time, status) ~ x,
    data = data,
    istate = state,
    statedata = data.frame(state = levels(data$state)),
    model = TRUE,
    control = coxph.control(iter.max = 50L, eps = 1e-09)
  )
  reference_fit <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = data,
    istate = state,
    statedata = data.frame(state = levels(data$state)),
    model = TRUE,
    control = survival::coxph.control(iter.max = 50L, eps = 1e-09)
  )
  expect_equal(coef(bridged_fit), coef(reference_fit), tolerance = 1e-5)
  bridged_fit_frame <- model.frame(bridged_fit)
  fit_istate_name <- grep("istate", names(bridged_fit_frame), value = TRUE)
  expect_length(fit_istate_name, 1L)
  expect_equal(as.character(bridged_fit_frame[[fit_istate_name]]), as.character(data$state))
})

test_that("Student-t frailty formulas match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 2:19,
    status = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    x = c(-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, -1, -0.6, -0.2, 0.2, 0.6, 1, 1.4, -1.4, -0.9, 0.1, 0.9),
    g = factor(rep(letters[1:6], each = 3))
  )
  bridged <- coxph(
    Surv(time, status) ~ x + frailty(g, distribution = "t", theta = 0.5, tdf = 5, method = "fixed"),
    data = data,
    ties = "breslow",
    control = coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13)
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + survival::frailty(g, distribution = "t", theta = 0.5, tdf = 5, method = "fixed"),
    data = data,
    ties = "breslow",
    control = survival::coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13)
  )

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
  expect_equal(bridged$frailty, reference$frail, tolerance = 1e-12)
  expect_equal(bridged$frailty_variance, reference$fvar, tolerance = 1e-12)
  expect_equal(bridged$log_likelihood, reference$loglik, tolerance = 1e-12)
  expect_equal(
    unname(unlist(bridged$term_degrees_of_freedom)),
    unname(reference$df),
    tolerance = 1e-12
  )
})

test_that("dense non-Gaussian frailty formulas match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = 2:19,
    status = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
    x = c(-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, -1, -0.6, -0.2, 0.2, 0.6, 1, 1.4, -1.4, -0.9, 0.1, 0.9),
    g = factor(rep(letters[1:6], each = 3))
  )
  terms <- c(
    'frailty(g, distribution = "gamma", theta = 0.5, sparse = FALSE)',
    'frailty(g, distribution = "t", theta = 0.5, tdf = 5, method = "fixed", sparse = FALSE)'
  )

  for (term in terms) {
    bridged <- coxph(
      stats::as.formula(paste("Surv(time, status) ~ x +", term)),
      data = data,
      ties = "breslow",
      control = coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13)
    )
    reference <- survival::coxph(
      stats::as.formula(
        paste("survival::Surv(time, status) ~ x +", paste0("survival::", term))
      ),
      data = data,
      ties = "breslow",
      control = survival::coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13)
    )

    expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
    expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
    expect_equal(bridged$log_likelihood, reference$loglik, tolerance = 1e-12)
    expect_equal(
      unname(unlist(bridged$term_degrees_of_freedom)),
      unname(reference$df),
      tolerance = 1e-12
    )
  }
})

test_that("gamma frailty target degrees of freedom match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = seq_len(18L),
    status = c(1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1),
    x = c(1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1, 0.6, 1.7, 0.1, 1.3, 0.8, 1.6),
    g = factor(rep(letters[1:6], 3))
  )
  bridged <- coxph(
    Surv(time, status) ~ x + frailty(g, distribution = "gamma", df = 2),
    data = data,
    ties = "breslow",
    control = coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13, outer.max = 30L)
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x + survival::frailty(g, distribution = "gamma", df = 2),
    data = data,
    ties = "breslow",
    control = survival::coxph.control(iter.max = 50L, eps = 1e-10, toler.chol = 1e-13, outer.max = 30L)
  )

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
  expect_equal(bridged$frailty, reference$frail, tolerance = 1e-12)
  expect_equal(bridged$frailty_variance, reference$fvar, tolerance = 1e-12)
  expect_equal(bridged$log_likelihood, reference$loglik, tolerance = 1e-12)
  expect_equal(
    unname(unlist(bridged$term_degrees_of_freedom)),
    unname(reference$df),
    tolerance = 1e-12
  )
  expect_equal(
    unname(bridged$history[[1L]]$theta),
    unname(reference$history[[1L]]$theta),
    tolerance = 1e-12
  )
  expect_equal(
    bridged$history[[1L]]$c.loglik,
    reference$history[[1L]]$c.loglik,
    tolerance = 1e-12
  )
})

test_that("single-formula multi-state Cox models match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  competing <- data.frame(
    id = seq_len(12L),
    time = seq_len(12L),
    status = factor(
      c("a", "b", "0", "a", "0", "b", "a", "b", "0", "a", "b", "0"),
      levels = c("0", "a", "b")
    ),
    x = c(0.2, 0.8, 0.3, 1.1, 0.4, 1.4, 0.6, 1.6, 0.7, 1.8, 1, 2)
  )
  row.names(competing) <- paste0("case", seq_len(nrow(competing)))
  bridged <- coxph(
    Surv(time, status) ~ x,
    data = competing,
    id = id,
    x = TRUE,
    y = TRUE,
    model = TRUE,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = competing,
    id = id,
    x = TRUE,
    y = TRUE,
    model = TRUE,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )

  expect_error(anova(bridged), "not yet available for multistate")

  expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)
  expect_equal(vcov(bridged), vcov(reference), tolerance = 1e-12)
  bridged_naive <- matrix(
    unlist(bridged$naive_var, use.names = FALSE),
    nrow = length(coef(bridged)),
    byrow = TRUE,
    dimnames = dimnames(reference$naive.var)
  )
  expect_equal(bridged_naive, reference$naive.var, tolerance = 1e-12)
  expect_equal(bridged$log_likelihood, reference$loglik, tolerance = 1e-12)
  expect_equal(unname(unlist(bridged$means)), unname(reference$means), tolerance = 1e-12)
  expect_equal(model.matrix(bridged), model.matrix(reference), tolerance = 1e-12)
  expect_equal(predict(bridged, type = "lp"), predict(reference, type = "lp"), tolerance = 1e-12)
  expect_equal(
    predict(bridged, newdata = competing[1:2, ], type = "risk"),
    predict(reference, newdata = competing[1:2, ], type = "risk"),
    tolerance = 1e-12
  )

  default_curve <- survfit(
    bridged,
    newdata = data.frame(x = 0.5),
    time0 = TRUE
  )
  reference_default_curve <- survival::survfit(
    reference,
    newdata = data.frame(x = 0.5),
    se.fit = FALSE,
    time0 = TRUE
  )
  default_curve_list <- as.list(default_curve)
  expect_equal(default_curve_list$time, reference_default_curve$time, tolerance = 1e-12)
  expect_equal(
    default_curve_list$pstate,
    reference_default_curve$pstate,
    tolerance = 1e-12
  )
  expect_equal(
    default_curve_list$cumhaz,
    reference_default_curve$cumhaz,
    tolerance = 1e-12
  )
  expect_equal(dim(default_curve), c(data = 1L, states = 3L))

  explicit_se_curve <- expect_warning(
    survfit(
      bridged,
      newdata = data.frame(x = 0.5),
      se.fit = TRUE,
      time0 = TRUE
    ),
    NA
  )
  reference_explicit_se_curve <- expect_warning(
    survival::survfit(
      reference,
      newdata = data.frame(x = 0.5),
      se.fit = TRUE,
      time0 = TRUE
    ),
    NA
  )
  explicit_se_list <- as.list(explicit_se_curve)
  expect_named(
    explicit_se_list,
    setdiff(names(reference_explicit_se_curve), c("start.time", "newdata", "call"))
  )
  expect_equal(
    explicit_se_list$pstate,
    reference_explicit_se_curve$pstate,
    tolerance = 1e-12
  )
  expect_equal(
    explicit_se_list$cumhaz,
    reference_explicit_se_curve$cumhaz,
    tolerance = 1e-12
  )
  expect_false(any(c(
    "std.err", "std.chaz", "std.auc", "logse", "lower", "upper",
    "conf.type", "conf.int"
  ) %in% names(explicit_se_list)))

  for (curve_style in 1:2) {
    bridged_curve <- as.list(survfit(
      bridged,
      newdata = data.frame(x = 0.5),
      stype = curve_style,
      se.fit = FALSE,
      time0 = TRUE
    ))
    reference_curve <- survival::survfit(
      reference,
      newdata = data.frame(x = 0.5),
      stype = curve_style,
      se.fit = FALSE,
      time0 = TRUE
    )

    for (field in c(
      "time", "pstate", "cumhaz", "n.risk", "n.event", "n.censor",
      "n.transition", "p0", "states", "transitions"
    )) {
      expect_equal(
        bridged_curve[[field]],
        reference_curve[[field]],
        tolerance = 1e-12,
        info = paste("stype", curve_style, "field", field)
      )
    }
  }

  bridged_mixed_start <- as.list(survfit(
    bridged,
    newdata = data.frame(x = 0.5),
    p0 = c(0.5, 0.5, 0),
    se.fit = FALSE,
    time0 = TRUE
  ))
  reference_mixed_start <- survival::survfit(
    reference,
    newdata = data.frame(x = 0.5),
    p0 = c(0.5, 0.5, 0),
    se.fit = FALSE,
    time0 = TRUE
  )
  expect_equal(
    bridged_mixed_start$pstate,
    reference_mixed_start$pstate,
    tolerance = 1e-12
  )

  profile_data <- data.frame(
    x = c(0.5, 1.5),
    row.names = c("low", "high")
  )
  bridged_profiles <- survfit(
    bridged,
    newdata = profile_data,
    time0 = TRUE
  )
  reference_profiles <- survival::survfit(
    reference,
    newdata = profile_data,
    se.fit = FALSE,
    time0 = TRUE
  )
  bridged_profile_list <- as.list(bridged_profiles)
  for (field in c(
    "time", "pstate", "cumhaz", "n.risk", "n.event", "n.censor",
    "n.transition", "p0", "states", "transitions"
  )) {
    expect_equal(
      bridged_profile_list[[field]],
      reference_profiles[[field]],
      tolerance = 1e-12,
      info = paste("batched field", field)
    )
  }
  expect_equal(dim(bridged_profiles), c(data = 2L, states = 3L))
  expect_error(
    bridged_profiles[1L],
    "single index subscripts are not supported"
  )

  bridged_profile_subset <- as.list(
    bridged_profiles[2L, c("(s0)", "b"), drop = FALSE]
  )
  reference_profile_subset <- reference_profiles[
    2L,
    c("(s0)", "b"),
    drop = FALSE
  ]
  expect_equal(
    bridged_profile_subset$pstate,
    reference_profile_subset$pstate,
    tolerance = 1e-12
  )
  expect_equal(bridged_profile_subset$states, reference_profile_subset$states)
  expect_equal(
    dim(bridged_profiles[2L, c("(s0)", "b"), drop = FALSE]),
    c(data = 1L, states = 2L)
  )

  aggregate_profile_data <- data.frame(x = c(0.5, 1.5, 2.0))
  aggregate_profiles <- survfit(
    bridged,
    newdata = aggregate_profile_data,
    time0 = TRUE
  )
  reference_aggregate_profiles <- survival::survfit(
    reference,
    newdata = aggregate_profile_data,
    se.fit = FALSE,
    time0 = TRUE
  )
  bridged_average <- as.list(aggregate(aggregate_profiles))
  reference_average <- aggregate(reference_aggregate_profiles)
  for (field in c(
    "time", "pstate", "n.risk", "n.event", "n.censor",
    "n.transition", "p0", "states", "transitions"
  )) {
    expect_equal(
      bridged_average[[field]],
      reference_average[[field]],
      tolerance = 1e-12,
      info = paste("averaged field", field)
    )
  }
  expect_false("cumhaz" %in% names(bridged_average))
  expect_equal(dim(aggregate(aggregate_profiles)), c(states = 3L))

  aggregate_groups <- c("A", "B", "A")
  bridged_grouped_average <- aggregate(
    aggregate_profiles,
    by = list(aggregate_groups)
  )
  reference_grouped_average <- aggregate(
    reference_aggregate_profiles,
    by = list(aggregate_groups)
  )
  expect_equal(
    as.list(bridged_grouped_average)$pstate,
    reference_grouped_average$pstate,
    tolerance = 1e-12
  )
  expect_false("cumhaz" %in% names(as.list(bridged_grouped_average)))
  expect_equal(dim(bridged_grouped_average), c(data = 2L, states = 3L))

  stratified_competing <- transform(
    competing,
    g = factor(rep(c("g1", "g2"), each = 6L))
  )
  bridged_stratified <- coxph(
    Surv(time, status) ~ x + strata(g),
    data = stratified_competing,
    id = id,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_stratified <- survival::coxph(
    survival::Surv(time, status) ~ x + strata(g),
    data = stratified_competing,
    id = id,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  expect_equal(coef(bridged_stratified), coef(reference_stratified), tolerance = 1e-12)
  for (field in c("states", "transitions", "cmap", "smap", "rmap", "assign", "share")) {
    expect_equal(
      bridged_stratified[[field]],
      reference_stratified[[field]],
      tolerance = 1e-12,
      info = paste("stratified multi-state metadata field", field)
    )
  }

  bridged_stratified_curves <- survfit(
    bridged_stratified,
    newdata = profile_data,
    time0 = TRUE
  )
  reference_stratified_curves <- survival::survfit(
    reference_stratified,
    newdata = profile_data,
    se.fit = FALSE,
    time0 = TRUE
  )
  bridged_stratified_list <- as.list(bridged_stratified_curves)
  for (field in c(
    "n", "time", "strata", "pstate", "cumhaz", "n.risk", "n.event",
    "n.censor", "n.transition", "n.id", "p0", "states", "transitions"
  )) {
    expect_equal(
      bridged_stratified_list[[field]],
      reference_stratified_curves[[field]],
      tolerance = 1e-12,
      info = paste("stratified field", field)
    )
  }
  expect_equal(
    dim(bridged_stratified_curves),
    c(strata = 2L, data = 2L, states = 3L)
  )

  bridged_stratified_subset <- bridged_stratified_curves[
    "g2",
    2L,
    c("(s0)", "b"),
    drop = FALSE
  ]
  reference_stratified_subset <- reference_stratified_curves[
    "g2",
    2L,
    c("(s0)", "b"),
    drop = FALSE
  ]
  expect_equal(
    as.list(bridged_stratified_subset)$pstate,
    reference_stratified_subset$pstate,
    tolerance = 1e-12
  )
  expect_equal(
    dim(bridged_stratified_subset),
    c(strata = 1L, data = 1L, states = 2L)
  )

  bridged_stratified_average <- aggregate(bridged_stratified_curves)
  reference_stratified_average <- aggregate(reference_stratified_curves)
  expect_equal(
    as.list(bridged_stratified_average)$pstate,
    reference_stratified_average$pstate,
    tolerance = 1e-12
  )
  expect_equal(dim(bridged_stratified_average), c(strata = 2L, states = 3L))

  bridged_stratified_unpadded <- survfit(
    bridged_stratified,
    newdata = profile_data
  )
  reference_stratified_unpadded <- survival::survfit(
    reference_stratified,
    newdata = profile_data,
    se.fit = FALSE
  )
  bridged_stratified_padded <- as.list(survfit0(bridged_stratified_unpadded))
  reference_stratified_padded <- survival::survfit0(reference_stratified_unpadded)
  for (field in c("time", "strata", "pstate", "cumhaz")) {
    expect_equal(
      bridged_stratified_padded[[field]],
      reference_stratified_padded[[field]],
      tolerance = 1e-12,
      info = paste("stratified survfit0 field", field)
    )
  }

  stratified_frame <- as.data.frame(bridged_stratified_curves)
  expect_equal(
    names(stratified_frame),
    c(
      "curve", "time", "n.risk", "n.event", "n.censor", "pstate",
      "strata", "state"
    )
  )
  expect_equal(nrow(stratified_frame), 84L)
  stratified_counts <- table(stratified_frame$strata)
  expect_equal(as.integer(stratified_counts), c(42L, 42L))
  expect_equal(names(stratified_counts), c("g1", "g2"))

  formula_data <- transform(
    competing,
    z = c(0.9, -0.2, 0.4, 1.3, -0.5, 0.7, 1.1, -0.8, 0.2, 1.7, -1.2, 0.6),
    q = factor(
      c("u", "v", "w", "u", "u", "v", "w", "u", "u", "v", "w", "u"),
      levels = c("u", "v", "w")
    )
  )
  compare_multistate_zph <- function(bridged_fit, reference_fit) {
    for (group_terms in c(TRUE, FALSE)) {
      bridged_zph <- cox.zph(
        bridged_fit,
        transform = "rank",
        terms = group_terms
      )
      reference_zph <- survival::cox.zph(
        reference_fit,
        transform = "rank",
        terms = group_terms
      )
      bridged_frame <- as.data.frame(bridged_zph)

      expect_equal(bridged_frame$name, rownames(reference_zph$table))
      expect_equal(bridged_frame$df, as.integer(reference_zph$table[, "df"]))
      expect_equal(
        bridged_frame$chisq,
        unname(reference_zph$table[, "chisq"]),
        tolerance = 1e-09
      )
      expect_equal(
        bridged_frame$p,
        unname(reference_zph$table[, "p"]),
        tolerance = 1e-09
      )
      expect_equal(bridged_zph$x, reference_zph$x, tolerance = 1e-12)
      expect_equal(bridged_zph$time, reference_zph$time, tolerance = 1e-12)
      expect_equal(as.character(bridged_zph$transform), reference_zph$transform)
      expect_equal(
        do.call(rbind, bridged_zph$y),
        unname(reference_zph$y),
        tolerance = 1e-08
      )
      expect_equal(
        do.call(rbind, bridged_zph$var),
        unname(reference_zph$var),
        tolerance = 1e-08
      )
      expect_equal(
        as.integer(bridged_zph$strata),
        as.integer(reference_zph$strata)
      )
    }
  }
  compare_multistate_zph(bridged, reference)

  bridged_common <- coxph(
    list(
      Surv(time, status) ~ 1,
      1:2 + 1:3 ~ x / common
    ),
    data = formula_data,
    id = id,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_common <- survival::coxph(
    list(
      survival::Surv(time, status) ~ 1,
      1:2 + 1:3 ~ x / common
    ),
    data = formula_data,
    id = id,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  expect_equal(coef(bridged_common), coef(reference_common), tolerance = 1e-12)
  expect_equal(vcov(bridged_common), vcov(reference_common), tolerance = 1e-12)
  for (field in c("states", "transitions", "cmap", "smap", "rmap", "assign", "share")) {
    expect_equal(
      bridged_common[[field]],
      reference_common[[field]],
      tolerance = 1e-12,
      info = paste("common multi-state metadata field", field)
    )
  }
  expect_s3_class(bridged_common$formula, "formula")
  expect_equal(
    paste(deparse(bridged_common$formula), collapse = " "),
    gsub(
      "survival::",
      "",
      paste(deparse(reference_common$formula), collapse = " "),
      fixed = TRUE
    )
  )
  expect_equal(
    bridged_common$log_likelihood,
    reference_common$loglik,
    tolerance = 1e-12
  )
  expect_equal(
    predict(bridged_common, newdata = formula_data[1:2, ], type = "lp"),
    predict(reference_common, newdata = formula_data[1:2, ], type = "lp"),
    tolerance = 1e-12
  )
  compare_multistate_zph(bridged_common, reference_common)

  bridged_selective <- coxph(
    list(
      Surv(time, status) ~ x + z,
      1:2 ~ -z
    ),
    data = formula_data,
    id = id,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_selective <- survival::coxph(
    list(
      survival::Surv(time, status) ~ x + z,
      1:2 ~ -z
    ),
    data = formula_data,
    id = id,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  expect_equal(coef(bridged_selective), coef(reference_selective), tolerance = 1e-12)
  for (field in c("states", "transitions", "cmap", "smap", "rmap", "assign", "share")) {
    expect_equal(
      bridged_selective[[field]],
      reference_selective[[field]],
      tolerance = 1e-12,
      info = paste("selective multi-state metadata field", field)
    )
  }
  expect_equal(
    predict(bridged_selective, newdata = formula_data[1:2, ], type = "lp"),
    predict(reference_selective, newdata = formula_data[1:2, ], type = "lp"),
    tolerance = 1e-12
  )
  compare_multistate_zph(bridged_selective, reference_selective)

  bridged_factor_zph <- coxph(
    Surv(time, status) ~ x + q,
    data = formula_data,
    id = id,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_factor_zph <- survival::coxph(
    survival::Surv(time, status) ~ x + q,
    data = formula_data,
    id = id,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  compare_multistate_zph(bridged_factor_zph, reference_factor_zph)

  bridged_shared <- coxph(
    list(
      Surv(time, status) ~ 1,
      1:2 + 1:3 ~ x / shared
    ),
    data = formula_data,
    id = id,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_shared <- survival::coxph(
    list(
      survival::Surv(time, status) ~ 1,
      1:2 + 1:3 ~ x / shared
    ),
    data = formula_data,
    id = id,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  expect_equal(coef(bridged_shared), coef(reference_shared), tolerance = 1e-12)
  for (field in c("states", "transitions", "cmap", "smap", "rmap", "assign", "share")) {
    expect_equal(
      bridged_shared[[field]],
      reference_shared[[field]],
      tolerance = 1e-12,
      info = paste("shared multi-state metadata field", field)
    )
  }
  expect_equal(
    bridged_shared$log_likelihood,
    reference_shared$loglik,
    tolerance = 1e-12
  )
  bridged_shared_curve <- as.list(survfit(
    bridged_shared,
    newdata = data.frame(x = 0.5)
  ))
  reference_shared_curve <- survival::survfit(
    reference_shared,
    newdata = data.frame(x = 0.5),
    se.fit = FALSE
  )
  expect_equal(
    bridged_shared_curve$pstate,
    reference_shared_curve$pstate,
    tolerance = 1e-12
  )
  expect_equal(
    bridged_shared_curve$cumhaz,
    reference_shared_curve$cumhaz,
    tolerance = 1e-12
  )

  state_data <- data.frame(
    state = c("(s0)", "a", "b"),
    absorbing = c(0, 1, 1)
  )
  bridged_state_selector <- coxph(
    list(
      Surv(time, status) ~ 1,
      state("(s0)"):absorbing(1) ~ x / common
    ),
    data = formula_data,
    id = id,
    statedata = state_data,
    control = coxph.control(iter.max = 20L, eps = 1e-9)
  )
  reference_state_selector <- survival::coxph(
    list(
      survival::Surv(time, status) ~ 1,
      state("(s0)"):absorbing(1) ~ x / common
    ),
    data = formula_data,
    id = id,
    statedata = state_data,
    control = survival::coxph.control(iter.max = 20L, eps = 1e-9)
  )
  expect_equal(
    coef(bridged_state_selector),
    coef(reference_state_selector),
    tolerance = 1e-12
  )

  histories <- data.frame(
    id = c(1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 9, 10),
    start = c(0, 1, 0, 2, 0, 0, 0, 1.5, 0, 2.5, 0, 3, 0, 0, 2.2, 0),
    stop = c(1, 4, 2, 5, 3, 4, 1.5, 6, 2.5, 7, 3, 8, 5, 2.2, 6.5, 7.5),
    event = factor(
      c(
        "ill", "dead", "ill", "0", "dead", "0", "ill", "dead",
        "ill", "0", "ill", "dead", "dead", "ill", "dead", "0"
      ),
      levels = c("0", "ill", "dead")
    ),
    x = c(0.2, 0.2, 1.4, 1.4, 0.9, 2, 1.1, 1.1, 0.5, 0.5, 1.8, 1.8, 0.4, 1.6, 1.6, 0.8)
  )
  bridged_histories <- coxph(
    Surv(start, stop, event) ~ x,
    data = histories,
    id = id,
    control = coxph.control(iter.max = 30L, eps = 1e-9)
  )
  reference_histories <- survival::coxph(
    survival::Surv(start, stop, event) ~ x,
    data = histories,
    id = id,
    control = survival::coxph.control(iter.max = 30L, eps = 1e-9)
  )

  expect_equal(coef(bridged_histories), coef(reference_histories), tolerance = 1e-12)
  expect_equal(vcov(bridged_histories), vcov(reference_histories), tolerance = 1e-12)
  expect_equal(
    bridged_histories$log_likelihood,
    reference_histories$loglik,
    tolerance = 1e-12
  )
  expect_equal(
    predict(bridged_histories, type = "lp"),
    predict(reference_histories, type = "lp"),
    tolerance = 1e-12
  )

  bridged_history_curve <- as.list(survfit(
    bridged_histories,
    newdata = data.frame(x = 0.75),
    time0 = TRUE
  ))
  reference_history_curve <- survival::survfit(
    reference_histories,
    newdata = data.frame(x = 0.75),
    se.fit = FALSE,
    time0 = TRUE
  )
  for (field in c(
    "time", "pstate", "cumhaz", "n.risk", "n.event", "n.censor",
    "n.transition", "p0", "states", "transitions"
  )) {
    expect_equal(
      bridged_history_curve[[field]],
      reference_history_curve[[field]],
      tolerance = 1e-12,
      info = paste("counting field", field)
    )
  }
})

test_that("multi-state Cox history controls match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  histories <- data.frame(
    id = rep(seq_len(4L), each = 2L),
    start = c(0, 2, 0, 1, 0, 1, 0, 1),
    stop = c(1, 3, 1, 2, 1, 2, 1, 2),
    event = factor(
      c("a", "0", "a", "0", "a", "0", "a", "0"),
      levels = c("0", "a", "b")
    ),
    x = rep(0:3, each = 2L)
  )
  bridged_gap <- coxph(
    Surv(start, stop, event) ~ x,
    data = histories,
    id = id,
    control = coxph.control(iter.max = 0L)
  )
  reference_gap <- survival::coxph(
    survival::Surv(start, stop, event) ~ x,
    data = histories,
    id = id,
    control = survival::coxph.control(iter.max = 0L)
  )
  expect_equal(coef(bridged_gap), coef(reference_gap), tolerance = 1e-12)
  expect_error(
    coxph(
      Surv(start, stop, event) ~ x,
      data = histories,
      id = id,
      control = coxph.control(iter.max = 0L, survcheckallow = "overlap")
    ),
    "data set fails survcheck"
  )

  overlapping <- histories
  overlapping$start[2L] <- 0.5
  expect_error(
    coxph(
      Surv(start, stop, event) ~ x,
      data = overlapping,
      id = id,
      control = coxph.control(iter.max = 0L)
    ),
    "data set fails survcheck"
  )
  expect_error(
    survival::coxph(
      survival::Surv(start, stop, event) ~ x,
      data = overlapping,
      id = id,
      control = survival::coxph.control(iter.max = 0L)
    ),
    "data set fails survcheck"
  )
  bridged_overlap <- coxph(
    Surv(start, stop, event) ~ x,
    data = overlapping,
    id = id,
    control = coxph.control(iter.max = 0L, survcheckallow = "overlap")
  )
  reference_overlap <- survival::coxph(
    survival::Surv(start, stop, event) ~ x,
    data = overlapping,
    id = id,
    control = survival::coxph.control(iter.max = 0L, survcheckallow = "overlap")
  )
  expect_equal(coef(bridged_overlap), coef(reference_overlap), tolerance = 1e-12)
})

test_that("multi-state Cox tied curve types match survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  tied <- data.frame(
    id = seq_len(15L),
    time = rep(seq_len(5L), each = 3L),
    status = factor(
      c("a", "a", "b", "b", "0", "b", "a", "a", "0", "b", "b", "0", "a", "0", "0"),
      levels = c("0", "a", "b")
    ),
    x = seq_len(15L) / 10,
    wt = c(1, 2, 1.5, 0.5, 1.2, 2.2, 1.1, 0.8, 1.4, 2.5, 0.7, 1.3, 1.8, 0.9, 1.6)
  )

  for (method in c("breslow", "efron")) {
    bridged <- coxph(
      Surv(time, status) ~ x,
      data = tied,
      id = id,
      ties = method,
      control = coxph.control(iter.max = 40L, eps = 1e-10)
    )
    reference <- survival::coxph(
      survival::Surv(time, status) ~ x,
      data = tied,
      id = id,
      ties = method,
      control = survival::coxph.control(iter.max = 40L, eps = 1e-10)
    )
    expect_equal(coef(bridged), coef(reference), tolerance = 1e-12)

    for (curve_type in 1:2) {
      bridged_curve <- as.list(survfit(
        bridged,
        newdata = data.frame(x = 0.5),
        ctype = curve_type,
        time0 = TRUE
      ))
      reference_curve <- survival::survfit(
        reference,
        newdata = data.frame(x = 0.5),
        ctype = curve_type,
        se.fit = FALSE,
        time0 = TRUE
      )
      expect_equal(
        bridged_curve$cumhaz,
        reference_curve$cumhaz,
        tolerance = 1e-9,
        info = paste(method, "ctype", curve_type, "cumulative hazard")
      )
      expect_equal(
        bridged_curve$pstate,
        reference_curve$pstate,
        tolerance = 1e-12,
        info = paste(method, "ctype", curve_type, "state probability")
      )
    }

    bridged_default <- as.list(survfit(
      bridged,
      newdata = data.frame(x = 0.5),
      time0 = TRUE
    ))
    reference_default <- survival::survfit(
      reference,
      newdata = data.frame(x = 0.5),
      se.fit = FALSE,
      time0 = TRUE
    )
    expect_equal(bridged_default$cumhaz, reference_default$cumhaz, tolerance = 1e-9)
    expect_equal(bridged_default$pstate, reference_default$pstate, tolerance = 1e-12)
  }

  bridged_weighted <- coxph(
    Surv(time, status) ~ x,
    data = tied,
    id = id,
    weights = wt,
    ties = "efron",
    control = coxph.control(iter.max = 40L, eps = 1e-10)
  )
  reference_weighted <- survival::coxph(
    survival::Surv(time, status) ~ x,
    data = tied,
    id = id,
    weights = wt,
    ties = "efron",
    control = survival::coxph.control(iter.max = 40L, eps = 1e-10)
  )
  expect_equal(coef(bridged_weighted), coef(reference_weighted), tolerance = 1e-12)
  for (curve_type in 1:2) {
    bridged_curve <- as.list(survfit(
      bridged_weighted,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      time0 = TRUE
    ))
    reference_curve <- survival::survfit(
      reference_weighted,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      se.fit = FALSE,
      time0 = TRUE
    )
    expect_equal(bridged_curve$cumhaz, reference_curve$cumhaz, tolerance = 1e-9)
    expect_equal(bridged_curve$pstate, reference_curve$pstate, tolerance = 1e-12)
  }

  bridged_shared <- coxph(
    list(Surv(time, status) ~ 1, 1:2 + 1:3 ~ x / shared),
    data = tied,
    id = id,
    control = coxph.control(iter.max = 40L, eps = 1e-10)
  )
  reference_shared <- survival::coxph(
    list(survival::Surv(time, status) ~ 1, 1:2 + 1:3 ~ x / shared),
    data = tied,
    id = id,
    control = survival::coxph.control(iter.max = 40L, eps = 1e-10)
  )
  expect_equal(coef(bridged_shared), coef(reference_shared), tolerance = 1e-12)
  for (curve_type in 1:2) {
    bridged_curve <- as.list(survfit(
      bridged_shared,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      time0 = TRUE
    ))
    reference_curve <- survival::survfit(
      reference_shared,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      se.fit = FALSE,
      time0 = TRUE
    )
    expect_equal(bridged_curve$cumhaz, reference_curve$cumhaz, tolerance = 1e-9)
    expect_equal(bridged_curve$pstate, reference_curve$pstate, tolerance = 1e-12)
  }

  stratified <- rbind(
    transform(tied, g = "g1"),
    transform(tied, id = id + 15L, x = x + 0.05, g = "g2")
  )
  stratified$g <- factor(stratified$g)
  bridged_stratified <- coxph(
    Surv(time, status) ~ x + strata(g),
    data = stratified,
    id = id,
    ties = "efron",
    control = coxph.control(iter.max = 40L, eps = 1e-10)
  )
  reference_stratified <- survival::coxph(
    survival::Surv(time, status) ~ x + strata(g),
    data = stratified,
    id = id,
    ties = "efron",
    control = survival::coxph.control(iter.max = 40L, eps = 1e-10)
  )
  expect_equal(coef(bridged_stratified), coef(reference_stratified), tolerance = 1e-12)
  for (curve_type in 1:2) {
    bridged_curve <- as.list(survfit(
      bridged_stratified,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      time0 = TRUE
    ))
    reference_curve <- survival::survfit(
      reference_stratified,
      newdata = data.frame(x = 0.5),
      ctype = curve_type,
      se.fit = FALSE,
      time0 = TRUE
    )
    expect_equal(bridged_curve$cumhaz, reference_curve$cumhaz, tolerance = 1e-9)
    expect_equal(bridged_curve$pstate, reference_curve$pstate, tolerance = 1e-12)
  }
})
