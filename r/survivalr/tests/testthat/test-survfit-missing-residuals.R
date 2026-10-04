.missing_survfit_compare <- function(actual, expected, info, labels = TRUE) {
  expect_identical(dim(actual), dim(expected), info = info)
  if (labels) {
    expect_identical(unname(names(actual)), unname(names(expected)), info = info)
    expect_identical(lapply(dimnames(actual), unname), lapply(dimnames(expected), unname), info = info)
  }
  expect_identical(as.vector(is.na(actual)), as.vector(is.na(expected)), info = info)
  expect_identical(as.vector(is.nan(actual)), as.vector(is.nan(expected)), info = info)
  expect_equal(as.numeric(actual), as.numeric(expected), tolerance = 2e-11, info = info)
}

test_that("missing KM and AJ residual and pseudo rows agree with stock R", {
  d <- data.frame(time = c(1, NA, 3, 4, 5, 6, 7, 8),
                  status = c(1, 0, 1, 0, 1, 0, 1, 0),
                  event = factor(c("a", "censor", "b", "censor", "a", "censor", "b", "censor"),
                                 levels = c("censor", "b", "a")),
                  subject = letters[1:8], weight = c(1, 1, .75, 1.25, 1.5, NA, 1, 2),
                  group = rep(c("a", "b"), each = 4))
  for (multi in c(FALSE, TRUE)) for (use_id in c(FALSE, TRUE)) {
    formula <- as.formula(paste0("Surv(time,", if (multi) "event" else "status", ") ~ group"))
    stock_formula <- as.formula(paste0("survival::Surv(time,", if (multi) "event" else "status", ") ~ group"))
    for (action in list(na.omit, na.exclude)) {
      options <- list(data = d, weights = quote(weight), na.action = action,
                      timefix = FALSE, model = TRUE)
      if (use_id) options$id <- quote(subject)
      actual_fit <- do.call(survfit, c(list(formula = formula), options))
      stock_fit <- do.call(get("survfit.formula", asNamespace("survival")),
                           c(list(formula = stock_formula), options))
      for (type in c("pstate", "cumhaz", "auc")) for (collapse in c(FALSE, TRUE)) {
        times <- c(if (type == "auc") 5.5 else 2.5, 6.5)
        for (query in list(times[1L], times)) {
          info <- paste("multi", multi, "id", use_id, "action", class(stock_fit$na.action),
                        "type", type, "collapse", collapse, "times", length(query))
          actual <- residuals(actual_fit, times = query, type = type, collapse = collapse,
                              extra = TRUE)
          expected <- residuals(stock_fit, times = query, type = type, collapse = collapse,
                                extra = TRUE)
          # Grouped AJ's native single-time array has a redundant last dimension.
          if (multi && length(query) == 1L && length(dim(expected$resid)) == 3L) {
            expected$resid <- drop(expected$resid)
          }
          # Stock NA reinsertion loses AJ array labels; retain port metadata.
          labels <- !multi || !is.null(dimnames(expected$resid))
          .missing_survfit_compare(actual$resid, expected$resid, info, labels)
          expect_equal(actual$curve, expected$curve, info = info)
          values <- suppressWarnings(pseudo(actual_fit, times = query, type = type,
                                            collapse = collapse))
          stock_values <- suppressWarnings(survival::pseudo(stock_fit, times = query,
                                                            type = type, collapse = collapse))
          .missing_survfit_compare(values, stock_values, info,
                                   labels = !multi || !is.null(dimnames(stock_values)))
        }
        actual_table <- suppressWarnings(pseudo(actual_fit, times = times, type = type,
                                                collapse = collapse, data.frame = TRUE))
        stock_table <- suppressWarnings(survival::pseudo(stock_fit, times = times, type = type,
                                                        collapse = collapse, data.frame = TRUE))
        expect_equal(actual_table, stock_table, tolerance = 2e-11, info = info)
      }
    }
  }
})

test_that("omission and exclusion preserve string subject labels before and after collapse", {
  d <- data.frame(start = c(0, 1, 0, 2, 0, 1, 0, 3, 0, 4),
                  time = c(1, 5, 2, 6, 1, 4, 3, 7, NA, 8),
                  status = c(0, 1, 1, 0, 0, 1, 0, 1, 0, 1),
                  subject = rep(letters[1:5], each = 2),
                  weight = rep(c(.75, 1.25, 1.5, .5, 2), each = 2),
                  group = rep(c("a", "b"), c(4, 6)))
  d$weight[3L] <- NA
  for (action in list(na.omit, na.exclude)) {
    options <- list(data = d, id = quote(subject), weights = quote(weight),
                    na.action = action, timefix = FALSE, model = TRUE)
    fit <- do.call(survfit, c(list(Surv(start, time, status) ~ group), options))
    stock <- do.call(get("survfit.formula", asNamespace("survival")),
                     c(list(survival::Surv(start, time, status) ~ group), options))
    for (collapse in c(FALSE, TRUE)) {
      values <- suppressWarnings(pseudo(fit, times = c(2.5, 6.5), collapse = collapse))
      expected <- suppressWarnings(survival::pseudo(stock, times = c(2.5, 6.5), collapse = collapse))
      .missing_survfit_compare(values, expected, paste(class(stock$na.action), collapse))
    }
  }
})
