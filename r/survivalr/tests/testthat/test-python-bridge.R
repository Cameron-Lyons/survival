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

  response <- Surv(data$time, data$status)
  expect_true(is.Surv(response))
  response_frame <- as.data.frame(response)
  expect_s3_class(response_frame, "data.frame")
  expect_equal(names(response_frame), c("time", "status", "type"))
  expect_equal(response_frame$time, data$time)
  expect_equal(response_frame$status, data$status)
  counting_response <- Surv(c(0, 1), c(2, 3), c(1, 0))
  counting_frame <- as.data.frame(counting_response)
  expect_equal(names(counting_frame), c("start", "stop", "status", "type"))
  expect_equal(counting_frame$stop, c(2, 3))
  named_response <- Surv(time = data$time, status = data$status)
  named_frame <- as.data.frame(named_response)
  expect_equal(named_frame$time, data$time)
  expect_equal(named_frame$status, data$status)
  named_counting <- Surv(start = c(0, 1), stop = c(2, 3), status = c(1, 0))
  named_counting_frame <- as.data.frame(named_counting)
  expect_equal(named_counting_frame$start, c(0, 1))
  expect_equal(named_counting_frame$stop, c(2, 3))
  named_interval2 <- Surv(time1 = c(-Inf, 2), stop = c(1, Inf), type = "interval2")
  named_interval2_frame <- as.data.frame(named_interval2)
  expect_equal(named_interval2_frame$status, c(2, 0))
  expect_error(
    Surv(time = data$time, start = data$time, status = data$status),
    "multiple time"
  )
  expect_true(any(grepl("status", capture.output(print(response)), fixed = TRUE)))

  cox_control <- coxph.control(iter.max = 0, eps = 1e-05, toler.chol = 1e-08, timefix = FALSE)
  expect_named(cox_control, c("eps", "toler.chol", "iter.max", "toler.inf", "outer.max", "timefix"))
  expect_equal(cox_control[["iter.max"]], 0L)
  expect_false(cox_control[["timefix"]])
  expect_error(coxph.control(iter.max = -1), "iter.max")

  survreg_control <- survreg.control(maxiter = 1, rel.tolerance = 1e-05, toler.chol = 1e-08)
  expect_named(survreg_control, c("iter.max", "rel.tolerance", "toler.chol", "debug", "maxiter", "outer.max"))
  expect_equal(survreg_control[["iter.max"]], 1L)
  expect_equal(survreg_control[["maxiter"]], 1L)
  expect_error(survreg.control(rel.tolerance = 0), "rel.tolerance")

  km <- survfit(Surv(time, status) ~ group, data = data)
  expect_s3_class(km, "survival_py_survfit")
  km_from_string <- survfit("Surv(time, status) ~ group", data = data)
  expect_s3_class(km_from_string, "survival_py_survfit")
  km_from_response <- survfit(response)
  expect_s3_class(km_from_response, "survival_py_survfit")
  grouped_from_response <- survfit(response, group = data$group, se.fit = FALSE)
  expect_s3_class(grouped_from_response, "survival_py_survfit")
  expect_equal(names(grouped_from_response), c("control", "treated"))
  grouped_no_se_frame <- as.data.frame(grouped_from_response)
  expect_s3_class(grouped_no_se_frame, "data.frame")
  expect_false(any(c("std.err", "lower", "upper", "std.chaz") %in% names(grouped_no_se_frame)))
  expect_true(all(vapply(grouped_no_se_frame, length, integer(1)) == nrow(grouped_no_se_frame)))
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
  expect_true(any(grepl("time", capture.output(print(km)), fixed = TRUE)))

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
  expect_false(any(grepl("survival.r_api", aft_print, fixed = TRUE)))
  expect_s3_class(fit, "survival_py_model")
  fit_print <- capture.output(print(fit))
  expect_true(any(grepl("Call:", fit_print, fixed = TRUE)))
  expect_true(any(grepl("Coefficients:", fit_print, fixed = TRUE)))
  expect_true(any(grepl("Surv(time, status) ~ x", fit_print, fixed = TRUE)))
  expect_true(any(grepl("logLik=", fit_print, fixed = TRUE)))
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
  expect_equal(attr(logLik(fit), "df"), 1L)
  expect_equal(nobs(fit), nrow(data))
  fit_aic <- extractAIC(fit)
  expect_named(fit_aic, c("df", "AIC"))
  expect_equal(fit_aic[["df"]], 1)
  expect_equal(fit_aic[["AIC"]], as.numeric(AIC(fit)))
  expect_equal(deparse(formula(fit)), "Surv(time, status) ~ x")
  expect_s3_class(terms(fit), "terms")
  expect_null(weights(fit))
  weighted_fit <- coxph(Surv(time, status) ~ x, data = data, weights = data$wt, max_iter = 0)
  expect_equal(weights(weighted_fit), data$wt)
  fitted_values <- fitted(fit)
  expect_true(is.numeric(unlist(fitted_values, use.names = FALSE)))
  expect_equal(length(unlist(fitted_values, use.names = FALSE)), nrow(data))
  fit_summary <- summary(fit)
  expect_s3_class(fit_summary, "summary.survival_py_model")
  expect_equal(rownames(fit_summary$coefficients), "x")
  expect_true(all(c("coef", "se(coef)", "z", "Pr(>|z|)") %in% colnames(fit_summary$coefficients)))
  expect_equal(fit_summary$n, nrow(data))
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
  stratified_curves <- survfit(
    coxph(Surv(time, status) ~ x + strata(group), data = data, max_iter = 0),
    newdata = data.frame(x = c(0.5, 0.7), group = c("control", "treated")),
    se.fit = FALSE
  )
  stratified_curve_frame <- as.data.frame(stratified_curves)
  expect_equal(unique(stratified_curve_frame$strata), c(1L, 2L))

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
  direct_concordance_frame <- as.data.frame(direct_concordance)
  formula_concordance_frame <- as.data.frame(formula_concordance)
  expect_s3_class(direct_concordance_frame, "data.frame")
  expect_equal(direct_concordance_frame$concordance, formula_concordance_frame$concordance)
  expect_equal(direct_concordance_frame$variance, formula_concordance_frame$variance)
  expect_equal(as.numeric(direct_concordance$var), as.numeric(formula_concordance$var))

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
})

test_that("survreg bridge agrees with R survival distributions", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

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
    expect_equal(as.numeric(summary(bridged)$scale), reference$scale, tolerance = 5e-05)
    expect_equal(as.numeric(logLik(bridged)), reference$loglik[[2L]], tolerance = 1e-05)
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
  }
})

test_that("Kaplan-Meier and log-rank bridge results agree with R survival", {
  skip_if_not_installed("reticulate")
  skip_if_not_installed("survival")
  skip_if_not(reticulate::py_module_available("survival"), "Python survival package is unavailable")

  data <- data.frame(
    time = c(1, 2, 2, 3, 4, 5, 6, 7),
    status = c(1, 1, 0, 1, 1, 0, 1, 0),
    group = c("A", "A", "B", "B", "A", "B", "A", "B")
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

  bridged_diff <- survdiff(Surv(time, status) ~ group, data = data)
  reference_diff <- survival::survdiff(survival::Surv(time, status) ~ group, data = data)
  bridged_diff_frame <- as.data.frame(bridged_diff)
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
})
