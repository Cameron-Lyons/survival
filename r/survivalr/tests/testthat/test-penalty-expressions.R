test_that("penalty terms evaluate expression arguments as stock R does", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  d <- survival::kidney
  nd <- data.frame(age = c(30, 45, 60), sex = c(1, 2, 1))
  terms <- c("pspline(log(age), df = 3) + sex", "pspline(sqrt(age), theta = 0.4) + sex",
             "ridge(age, I(age^2), theta = 1) + sex", "ridge(age/10, sex, theta = 2)")
  for (term in terms) for (kind in c("coxph", "survreg")) {
    formula <- as.formula(paste("Surv(time, status) ~", term))
    actual <- if (kind == "coxph") coxph(formula, d, x = TRUE) else survreg(formula, d, x = TRUE)
    expected <- if (kind == "coxph") survival::coxph(formula, d, x = TRUE) else
      survival::survreg(formula, d, x = TRUE)
    info <- paste(kind, term)
    expect_identical(names(coef(actual)), names(coef(expected)), info = info)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-7, info = info)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-7, info = info)
    expect_identical(colnames(model.matrix(actual)), colnames(expected$x), info = info)
    expect_equal(unname(model.matrix(actual)), unname(expected$x), ignore_attr = TRUE,
      tolerance = 1e-12, info = info)
    for (type in c("lp", "terms")) {
      expect_equal(unname(predict(actual, newdata = nd, type = type)),
        unname(predict(expected, newdata = nd, type = type)), ignore_attr = TRUE,
        tolerance = 1e-7, info = paste(info, type))
    }
  }
})

test_that("frailty terms evaluate a factor() argument and omit its missing rows", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  d <- survival::lung
  for (rhs in c("age + frailty(factor(inst))",
                "age + frailty(factor(inst), distribution = 'gaussian')",
                "pspline(log(age - 30), theta = 0.8) + frailty(factor(inst))")) {
    formula <- as.formula(paste("Surv(time, status) ~", rhs))
    actual <- coxph(formula, d)
    expected <- survival::coxph(formula, d)
    expect_identical(actual$n, expected$n, info = rhs)
    expect_identical(names(coef(actual)), names(coef(expected)), info = rhs)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-7, info = rhs)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-7, info = rhs)
  }
})

test_that("penalty expressions follow R's cbind, range and model-frame rules", {
  skip_if_not_installed("survival")
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("survival"))
  lung <- survival::lung
  kidney <- survival::kidney
  cases <- list(
    list(lung, "ridge(factor(inst), theta = 1) + sex"),
    list(lung, "ridge(ifelse(is.na(meal.cal), 0, meal.cal), sex, theta = 1)"),
    list(lung, "ridge(sex, is.na(meal.cal), theta = 1)"),
    list(lung, "log(ph.karno) + strata(ph.ecog, na.group = TRUE)"),
    list(lung, "pspline(log(ph.karno), df = 2) + strata(ph.ecog, na.group = TRUE)"),
    list(kidney, "ridge(log(age), I(age^2), theta = 2)"))
  for (case in cases) {
    formula <- as.formula(paste("Surv(time, status) ~", case[[2]]))
    actual <- coxph(formula, case[[1]], x = TRUE)
    expected <- survival::coxph(formula, case[[1]], x = TRUE)
    info <- case[[2]]
    expect_identical(actual$n, expected$n, info = info)
    expect_identical(names(coef(actual)), names(coef(expected)), info = info)
    expect_equal(unname(coef(actual)), unname(coef(expected)), tolerance = 1e-7, info = info)
    expect_equal(actual$loglik, expected$loglik, tolerance = 1e-7, info = info)
    expect_identical(colnames(model.matrix(actual)), colnames(expected$x), info = info)
  }
  expect_error(coxph(Surv(time, status) ~ pspline(factor(sex)), lung), "range")
  expect_error(survival::coxph(Surv(time, status) ~ pspline(factor(sex)), lung), "range")
})
