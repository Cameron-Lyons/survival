.matrix_compare <- function(actual, expected) {
  expect_identical(dim(actual), dim(expected))
  expect_identical(colnames(actual), colnames(expected))
  expect_identical(rownames(actual), rownames(expected))
  expect_identical(attr(actual, "contrasts"), attr(expected, "contrasts"))
  expect_identical(attr(actual, "assign"), as.integer(attr(expected, "assign")))
  expect_equal(unname(actual), unname(expected), ignore_attr = TRUE, tolerance = 1e-12)
  expect_equal(attr(actual, "strata"), attr(expected, "strata"))
}

test_that("penalized matrices use basis labels while coefficients keep penalty labels", {
  d <- survival::kidney
  terms <- c("ridge(age, theta = 2)", "ridge(age, sex, theta = 2)",
             "pspline(age, theta = 0.4)",
             "pspline(age, df = 2, nterm = 3, degree = 1, penalty = FALSE)")
  for (term in terms) for (kind in c("coxph", "survreg")) {
    formula <- as.formula(paste("Surv(time, status) ~", term))
    actual <- if (kind == "coxph") coxph(formula, d, x = TRUE) else survreg(formula, d, x = TRUE)
    expected <- if (kind == "coxph") survival::coxph(formula, d, x = TRUE) else
      survival::survreg(formula, d, x = TRUE)
    .matrix_compare(model.matrix(actual), expected$x)
    expect_identical(names(coef(actual)), names(coef(expected)))
    expect_equal(model.matrix(unserialize(serialize(actual, NULL))), model.matrix(actual))
  }
})

test_that("sparse matrices retain full formula order and recode before omission", {
  d <- survival::kidney
  d$group <- 100 + 3 * d$id
  for (rhs in c("frailty(group, sparse = TRUE, theta = 0.4) + age + strata(sex)",
                "age + frailty(group, sparse = TRUE, theta = 0.4) + strata(sex)",
                "age + strata(sex) + frailty(group, sparse = TRUE, theta = 0.4)",
                "frailty(group, sparse = TRUE, theta = 0.4)")) {
    formula <- as.formula(paste("Surv(time, status) ~", rhs))
    actual <- coxph(formula, d, x = TRUE)
    expected <- survival::coxph(formula, d, x = TRUE)
    .matrix_compare(model.matrix(actual), expected$x)
    for (group in list(c(112, 103, 112), c(999, 888, 999))) {
      nd <- data.frame(age = c(40, NA, 60), sex = c(1, 2, 1), group = group)
      .matrix_compare(model.matrix(actual, data = nd), model.matrix(expected, nd))
    }
    .matrix_compare(model.matrix(actual, data = d[FALSE, ]), model.matrix(expected, d[FALSE, ]))
    expect_equal(model.matrix(actual), expected$x, ignore_attr = TRUE)
    expect_error(model.matrix(actual, data = data.frame(age = 40, sex = 1)), "group")
  }
})

test_that("dense frailty matrices retain full fitted contrasts and factor order", {
  d <- survival::kidney
  d$g <- factor(d$id %% 3, levels = c(2, 0, 1))
  nd <- d[c(8, 1, 8, 4), ]
  for (family in c("gamma", "gaussian", "t")) {
    formula <- as.formula(paste0("Surv(time, status) ~ age + strata(sex) + frailty.",
      family, "(g, sparse = FALSE, theta = 0.4)"))
    actual <- coxph(formula, d, x = TRUE)
    expected <- survival::coxph(formula, d, x = TRUE)
    .matrix_compare(model.matrix(actual), expected$x)
    Terms <- delete.response(terms(expected))
    frame <- suppressWarnings(model.frame(Terms, nd, xlev = expected$xlevels))
    # Stock model.matrix.coxph misspells contrasts.arg in the strata branch.
    # The ordinary stats constructor independently retains fitted contrasts.
    full <- stats::model.matrix(Terms, frame, contrasts.arg = expected$contrasts)
    assign <- attr(full, "assign")
    fixed <- full[, !assign %in% c(0L, 2L), drop = FALSE]
    attr(fixed, "assign") <- assign[!assign %in% c(0L, 2L)]
    attr(fixed, "strata") <- frame[["strata(sex)"]]
    contrasts <- attr(full, "contrasts")
    contrasts[["strata(sex)"]] <- NULL
    attr(fixed, "contrasts") <- contrasts
    .matrix_compare(model.matrix(actual, data = nd), fixed)
    one <- nd; one$g <- factor(rep(2, nrow(one)), levels = levels(d$g))
    expect_error(model.matrix(actual, data = one), "not enough degrees of freedom")
    expect_error(model.matrix(actual, data = d[FALSE, ]), "not enough degrees of freedom")
  }
})

test_that("automatic frailty matrices can switch from sparse codes to dense columns", {
  d <- survival::kidney
  nd <- d[c(8, 1, 8, 4), ]
  for (family in c("gamma", "gaussian", "t")) {
    formula <- as.formula(paste0("Surv(time, status) ~ age + frailty.", family, "(id, theta = 0.4)"))
    actual <- coxph(formula, d, x = TRUE)
    expected <- survival::coxph(formula, d, x = TRUE)
    expect_equal(ncol(model.matrix(actual)), 2L)
    .matrix_compare(model.matrix(actual, data = nd), model.matrix(expected, nd))
    expect_equal(ncol(model.matrix(actual, data = nd)), 4L)
    nd$id <- c(999, 888, 999, 777)
    .matrix_compare(model.matrix(actual, data = nd), model.matrix(expected, nd))
  }
  d$g <- factor(d$id %% 3)
  actual <- coxph(Surv(time, status) ~ frailty(g, theta = 0.4), d)
  expected <- survival::coxph(Surv(time, status) ~ frailty(g, theta = 0.4), d)
  nd$g <- factor(20:23)
  expect_error(model.matrix(actual, data = nd), "new levels")
  larger <- d[seq_len(8), ]; larger$g <- factor(20:27)
  expect_error(model.matrix(actual, data = larger), "contrasts apply only to factors")
  expect_error(suppressWarnings(model.matrix(expected, larger)), "contrasts apply only to factors")
})

test_that("empty spline matrices retain their fitted width", {
  d <- survival::kidney
  for (kind in c("coxph", "survreg")) {
    fit <- if (kind == "coxph") coxph(Surv(time, status) ~ pspline(age, theta = .4), d) else
      survreg(Surv(time, status) ~ pspline(age, theta = .4), d)
    empty <- model.matrix(fit, data = d[FALSE, ])
    expect_identical(dim(empty), c(0L, ncol(model.matrix(fit))))
    expect_identical(colnames(empty), colnames(model.matrix(fit)))
    expect_identical(attr(empty, "assign"), attr(model.matrix(fit), "assign"))
  }
})
