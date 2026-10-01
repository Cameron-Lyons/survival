persistence_skip <- function() {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
}

persistence_roundtrip <- function(value) unserialize(serialize(value, NULL))

persistence_cox_values <- function(fit, data) {
  frailty <- grepl("frailty", deparse(formula(fit)))
  list(coefficients = coef(fit), variance = vcov(fit),
       prediction = predict(fit, newdata = data[1:3, ], type = "lp"),
       residuals = residuals(fit, type = "deviance"),
       model = model.frame(fit), matrix = model.matrix(fit),
       curves = as.data.frame(if (any(frailty)) survfit(fit) else survfit(fit, newdata = data[1:3, ])))
}

persistence_aft_values <- function(fit, data) {
  list(coefficients = coef(fit), variance = vcov(fit),
       prediction = predict(fit, newdata = data[1:3, ], type = "quantile", p = c(.1, .5, .9)),
       residuals = residuals(fit, type = "deviance"),
       model = model.frame(fit), matrix = model.matrix(fit))
}

test_that("R serialization preserves ordinary and penalized fitted methods", {
  persistence_skip()
  data <- survival::lung
  for (term in c("age", "ridge(age, theta = 1)", "pspline(age, df = 3)", "age + frailty(inst)")) {
    fit <- coxph(as.formula(paste("Surv(time, status) ~", term, "+ sex")), data,
                 model = TRUE, x = TRUE)
    expected <- persistence_cox_values(fit, data)
    restored <- persistence_roundtrip(fit)
    expect_true(reticulate::py_is_null_xptr(restored))
    expect_equal(persistence_cox_values(restored, data), expected, info = term)
    expect_false(reticulate::py_is_null_xptr(restored))
    expect_equal(persistence_cox_values(persistence_roundtrip(restored), data), expected)
  }
  for (term in c("age", "ridge(age, theta = 1)", "pspline(age, df = 3)")) {
    fit <- survreg(as.formula(paste("Surv(time, status) ~", term, "+ strata(sex)")), data,
                   model = TRUE, x = TRUE)
    expected <- persistence_aft_values(fit, data)
    expect_equal(persistence_aft_values(persistence_roundtrip(fit), data), expected, info = term)
  }
})

test_that("snapshots are lazy and reflect mutations at each save", {
  persistence_skip()
  calls <- 0L
  restores <- 0L
  original <- .pybridge_attr
  local_mocked_bindings(.pybridge_attr = function(name) {
    function_value <- original(name)
    if (name == "_unserialize_r_object") {
      return(function(value) { restores <<- restores + 1L; function_value(value) })
    }
    if (name != "_serialize_r_object") return(function_value)
    function(value) { calls <<- calls + 1L; function_value(value) }
  }, .package = "survivalr")
  fit <- coxph(Surv(time, status) ~ age + sex, survival::lung)
  fit <- .wrap_python(reticulate::import("types")$SimpleNamespace(model = fit, persistence_probe = 17L),
                      "survival_py_object")
  fit$persistence_probe
  expect_identical(calls, 0L)
  saved <- serialize(fit, NULL)
  expect_gt(calls, 0L)
  fit$persistence_probe <- 29L
  restored <- unserialize(saved)
  expect_identical(restored$persistence_probe, 17L)
  expect_identical(restores, 1L)
  expect_identical(restored$persistence_probe, 17L)
  expect_identical(restores, 1L)
  expect_identical(persistence_roundtrip(fit)$persistence_probe, 29L)
  restored$persistence_probe <- 31L
  expect_identical(persistence_roundtrip(restored)$persistence_probe, 31L)
  expect_identical(restored[["persistence_probe"]], 31L)
  nested <- persistence_roundtrip(fit$model$fit)
  expect_equal(nested$coefficients, fit$model$fit$coefficients)
  expect_error(serialize(fit, NULL, version = 2), "serialization version = 3", fixed = TRUE)
})

test_that("saved responses and grouped multistate curves preserve operations", {
  persistence_skip()
  response <- Surv(c(1, 2, NA, 4), c(1, 0, 1, 1))
  restored <- persistence_roundtrip(response)
  expect_equal(as.matrix(restored), as.matrix(response))
  expect_equal(capture.output(print(restored)), capture.output(print(response)))
  expect_equal(as.data.frame(survfit(persistence_roundtrip(response))), as.data.frame(survfit(response)))
  data <- data.frame(time = 1:8, event = factor(rep(c("censor", "a", "b", "a"), 2),
                    levels = c("censor", "a", "b")), group = rep(c("g1", "g2"), each = 4))
  curves <- survfit(Surv(time, event) ~ group, data, id = seq_len(nrow(data)))
  restored <- persistence_roundtrip(curves)
  expect_equal(as.data.frame(restored), as.data.frame(curves))
  expect_equal(as.list(restored), as.list(curves))
  expect_equal(as.data.frame(restored[, "a"]), as.data.frame(curves[, "a"]))
  extracted <- persistence_roundtrip(curves[1, ])
  expect_equal(as.data.frame(extracted), as.data.frame(curves[1, ]))
  ordinary <- survfit(Surv(time, status) ~ sex, survival::lung)
  expect_equal(as.data.frame(persistence_roundtrip(ordinary)), as.data.frame(ordinary))
  expect_equal(summary(persistence_roundtrip(ordinary), times = c(100, 200)),
               summary(ordinary, times = c(100, 200)))
})

test_that("R callbacks retain local environments in saved custom AFT fits", {
  persistence_skip()
  distribution <- local({
    base <- survival::survreg.distributions$gaussian
    shift <- 7
    base$name <- "Shifted local Gaussian"
    base$quantile <- function(p, parms) stats::qnorm(p) + shift
    base
  })
  fit <- survreg(Surv(time, status) ~ age + sex, survival::lung, dist = distribution,
                 model = TRUE, x = TRUE)
  expected <- persistence_aft_values(fit, survival::lung)
  restored <- persistence_roundtrip(fit)
  rm(fit, distribution)
  gc()
  expect_equal(persistence_aft_values(restored, survival::lung), expected)
  expect_equal(persistence_aft_values(persistence_roundtrip(restored), survival::lung), expected)
})

test_that("loaded models support diagnostics, comparisons, and population methods", {
  persistence_skip()
  data <- survival::lung
  fit <- coxph(Surv(time, status) ~ age + sex, data, model = TRUE, x = TRUE)
  smaller <- coxph(Surv(time, status) ~ age, data, model = TRUE, x = TRUE)
  expect_equal(as.data.frame(cox.zph(persistence_roundtrip(fit))), as.data.frame(cox.zph(fit)))
  expect_equal(as.data.frame(coxph.detail(persistence_roundtrip(fit))), as.data.frame(coxph.detail(fit)))
  expect_equal(as.data.frame(basehaz(persistence_roundtrip(fit))), as.data.frame(basehaz(fit)))
  restored <- persistence_roundtrip(list(fit, smaller))
  compared <- concordance(restored[[1L]], restored[[2L]], influence = 1)
  original <- concordance(fit, smaller, influence = 1)
  # Names follow the supplied R expressions, independently of stored fit state.
  for (field in c("concordance", "var", "n", "count", "dfbeta", "cvar")) {
    expect_equal(compared[[field]], original[[field]], ignore_attr = TRUE, info = field)
  }
  expect_equal(as.data.frame(anova(persistence_roundtrip(fit))), as.data.frame(anova(fit)))
  for (result in list(cox.zph(fit), coxph.detail(fit), basehaz(fit), anova(fit),
                     survdiff(Surv(time, status) ~ sex, data))) {
    expect_equal(as.data.frame(persistence_roundtrip(result)), as.data.frame(result))
  }
  expected <- survexp(time ~ 1, data, ratetable = fit, method = "individual.h")
  expect_equal(survexp(time ~ 1, data, ratetable = persistence_roundtrip(fit), method = "individual.h"), expected)
  yates_data <- transform(data, sex = factor(sex))
  yates_fit <- coxph(Surv(time, status) ~ age + sex, yates_data, model = TRUE, x = TRUE)
  expected <- yates(yates_fit, "sex")
  actual <- yates(persistence_roundtrip(yates_fit), "sex")
  actual$call <- expected$call
  expect_equal(actual, expected)
})

test_that("invalid and legacy R proxy states fail with an actionable error", {
  persistence_skip()
  fit <- coxph(Surv(time, status) ~ age, survival::lung)
  attr(fit, "survival_state") <- NULL
  restored <- persistence_roundtrip(fit)
  expect_error(coef(restored), "saved without model state", fixed = TRUE)
  expect_error(.Call(C_snapshot_new, 1), "callback must be a function", fixed = TRUE)
  expect_error(.Call(C_snapshot_get, raw(1)), "Invalid survival model serialization state", fixed = TRUE)
  expect_error(.Call(C_snapshot_get, .Call(C_snapshot_new, function() 1)), "state bundle", fixed = TRUE)
})

test_that("RDS and workspace files restore in a fresh process without attaching survivalr", {
  persistence_skip()
  package_path <- getNamespaceInfo("survivalr", "path")
  skip_if_not(file.exists(file.path(package_path, "Meta", "package.rds")),
              "Fresh-process test requires an installed package")
  data <- survival::lung
  distribution <- local({
    base <- survival::survreg.distributions$gaussian
    shift <- 7
    base$name <- "Local Gaussian"
    base$quantile <- function(p, parms) stats::qnorm(p) + shift
    base
  })
  models <- list(cox = coxph(Surv(time, status) ~ age + sex, data, model = TRUE, x = TRUE),
                 aft = survreg(Surv(time, status) ~ age + sex, data, dist = distribution,
                               model = TRUE, x = TRUE),
                 curves = survfit(Surv(time, status) ~ sex, data),
                 response = Surv(data$time, data$status))
  expected <- list(cox = persistence_cox_values(models$cox, data),
                   aft = persistence_aft_values(models$aft, data),
                   curves = as.data.frame(models$curves), response = as.matrix(models$response))
  directory <- tempfile("persistence-")
  dir.create(directory)
  on.exit(unlink(directory, recursive = TRUE))
  saveRDS(models, file.path(directory, "models.rds"))
  saveRDS(list(values = expected, data = data, cox_values = persistence_cox_values,
               aft_values = persistence_aft_values), file.path(directory, "expected.rds"))
  save(models, file = file.path(directory, "workspace.RData"))
  script <- c(
    paste0(".libPaths(", paste(deparse(.libPaths()), collapse = ""), ")"),
    "stopifnot(!'survivalr' %in% loadedNamespaces())",
    "directory <- commandArgs(trailingOnly = TRUE)[[1L]]",
    "models <- readRDS(file.path(directory, 'models.rds'))",
    "stopifnot('survivalr' %in% loadedNamespaces(), !reticulate::py_available(initialize = FALSE))",
    "saveRDS(models, file.path(directory, 'untouched.rds'))",
    "models <- readRDS(file.path(directory, 'untouched.rds'))",
    "stopifnot(!reticulate::py_available(initialize = FALSE))",
    "library(survivalr)",
    "expected <- readRDS(file.path(directory, 'expected.rds'))",
    "check <- function(models) {",
    " actual <- list(cox = expected$cox_values(models$cox, expected$data),",
    "                aft = expected$aft_values(models$aft, expected$data),",
    "                curves = as.data.frame(models$curves), response = as.matrix(models$response))",
    " stopifnot(isTRUE(all.equal(actual, expected$values)))",
    "}",
    "check(models)",
    "environment <- new.env(); load(file.path(directory, 'workspace.RData'), envir = environment)",
    "check(environment$models)",
    "saveRDS(models, file.path(directory, 'again.rds')); check(readRDS(file.path(directory, 'again.rds')))",
    "cat('fresh process persistence passed\\n')"
  )
  writeLines(script, file.path(directory, "read.R"))
  output <- system2(file.path(R.home("bin"), "Rscript"),
                    c("--vanilla", shQuote(file.path(directory, "read.R")), shQuote(directory)),
                    stdout = TRUE, stderr = TRUE)
  expect_null(attr(output, "status"), info = paste(output, collapse = "\n"))
  expect_true(any(grepl("fresh process persistence passed", output, fixed = TRUE)))
})
