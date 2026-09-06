# Regenerate the static Python reference with:
# Rscript scripts/generate_cox_ridge_reference.R
# Tests consume the JSON only; an R installation is not required to run them.
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.8.11")
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

small <- data.frame(
    time = c(1, 2, 2, 3, 4, 4, 5, 6, 7, 8),
    event = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 1),
    x = c(.2, .5, .7, .1, .4, .8, .3, .9, .6, 1.2),
    z = c(.3, 1.2, .4, .8, 1.1, .6, .2, .7, 1.4, .9),
    w = c(1, 2, 1, .5, 1.5, 1, 2, 1, 1, 1)
)
i <- seq_len(24)
correlated <- data.frame(
    time = c(1, 2, 2, 3, 4, 4, 5, 6, 7, 8, 8, 9,
             1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9),
    event = c(1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0,
              1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1),
    x = sin(i * .73) + i / 19,
    z = .8 * sin(i * .73) + i / 25 + cos(i * 1.17) / 4,
    u = cos(i * .89),
    o = sin(i * .31) / 5,
    g = rep(c("A", "B"), each = 12),
    w = rep(c(1, 2, .5, 1.5, 1, 1), 4)
)
correlated$start <- pmax(0, correlated$time - rep(c(1, 3, 2, 5), 6))

cases <- list()

# Independently evaluate interval uncertainty from R's risk-set accumulator.
# The gradient of H(stop)-H(start) must be differenced before its quadratic
# form, preserving covariance between the two endpoints. Work in uncentered
# coefficient units; offsets remain inside exp(X beta + offset).
interval_expected_reference <- function(fit, newdata = NULL) {
    X <- fit$x
    y <- fit$y
    mf <- fit$model
    weight <- model.weights(mf)
    if (is.null(weight)) weight <- rep(1, nrow(X))
    offset <- model.offset(mf)
    if (is.null(offset)) offset <- rep(0, nrow(X))
    strata_column <- grep("^strata\\(", names(mf), value = TRUE)
    strata <- if (length(strata_column)) as.character(mf[[strata_column]]) else rep("all", nrow(X))
    risk <- exp(drop(X %*% fit$coef) + offset)
    if (is.null(newdata)) {
        newX <- X
        newy <- y
        newoffset <- offset
        newstrata <- strata
    } else {
        newX <- model.matrix(fit, data = newdata)
        newmf <- model.frame(formula(fit), data = newdata)
        newy <- model.response(newmf)
        newoffset <- model.offset(newmf)
        if (is.null(newoffset)) newoffset <- rep(0, nrow(newX))
        newstrata <- if (length(strata_column)) as.character(newmf[[strata_column]]) else rep("all", nrow(newX))
    }
    newrisk <- exp(drop(newX %*% fit$coef) + newoffset)
    predicted <- se <- numeric(nrow(newX))
    method <- if (fit$method == "efron") 3L else 2L
    for (stratum in unique(strata)) {
        keep <- strata == stratum
        curve <- survival:::agsurv(y[keep, , drop = FALSE], X[keep, , drop = FALSE],
                                  weight[keep], risk[keep], method, method)
        chaz <- c(0, curve$cumhaz)
        varhaz <- c(0, cumsum(curve$varhaz))
        xbar <- rbind(0, apply(curve$xbar, 2, cumsum))
        for (row in which(newstrata == stratum)) {
            stop <- newy[row, ncol(newy) - 1L]
            start <- if (ncol(newy) == 3L) newy[row, 1] else -Inf
            j1 <- findInterval(start, curve$time) + 1L
            j2 <- findInterval(stop, curve$time) + 1L
            interval_hazard <- chaz[j2] - chaz[j1]
            gradient <- interval_hazard * newX[row, ] - (xbar[j2, ] - xbar[j1, ])
            predicted[row] <- interval_hazard * newrisk[row]
            variance <- varhaz[j2] - varhaz[j1] + drop(gradient %*% fit$var %*% gradient)
            se[row] <- sqrt(max(variance, 0)) * newrisk[row]
        }
    }
    list(fit = unname(predicted), se_fit = unname(se))
}

add_case <- function(name, formula, data = small, method = "efron",
                     weighted = FALSE, subset = NULL, init = NULL,
                     max_iter = 50L) {
    args <- list(formula = as.formula(formula), data = data, ties = method,
                 robust = FALSE, x = TRUE, y = TRUE, model = TRUE, na.action = na.omit,
                 control = coxph.control(iter.max = max_iter, eps = 1e-11))
    if (weighted) args$weights <- data$w
    if (!is.null(subset)) args$subset <- subset
    if (!is.null(init)) args$init <- init
    fit <- do.call(coxph, args)
    # Call reconstructors below while their original formula environment is live.
    newdata <- data[c(2, 5, 7), , drop = FALSE]
    newdata$x <- c(-.4, .6, 1.7)
    if (grepl("log(x)", formula, fixed = TRUE)) newdata$x <- c(.1, .6, 1.7)
    newdata$z <- c(-.1, .8, 1.1)
    if ("u" %in% names(data)) newdata$u <- c(.1, -.8, .6)
    if ("o" %in% names(data)) newdata$o <- c(.2, -.1, .3)
    if ("g" %in% names(data)) newdata$g <- c("A", "B", "A")
    newdata$time <- c(2, 4, 6)
    if ("start" %in% names(data)) newdata$start <- c(0, 1, 3)
    newdata$event <- c(1, 0, 1)

    predictions <- function(newdata = NULL) {
        out <- list()
        for (type in c("lp", "risk", "terms", "expected")) {
            pargs <- list(object = fit, type = type, se.fit = TRUE)
            if (!is.null(newdata)) pargs$newdata <- newdata
            value <- do.call(predict, pargs)
            out[[type]] <- list(fit = unname(value$fit), se_fit = unname(value$se.fit))
        }
        out
    }
    resid <- list()
    # R labels exact penalty fits "exact" but numerically uses Breslow. Its
    # residual method still rejects the score/Schoenfeld/influence variants.
    residual_types <- if (method == "exact") c("martingale", "deviance", "partial") else
        c("martingale", "deviance", "score", "schoenfeld", "dfbeta", "dfbetas",
          "scaledsch", "partial")
    for (type in residual_types) {
        resid[[type]] <- unname(residuals(fit, type = type))
    }
    if (method != "exact") resid$weighted_score <- unname(residuals(fit, type = "score", weighted = TRUE))
    penalty_diagonal <- rep(0, ncol(fit$x))
    for (label in names(fit$history)) {
        term <- fit$model[[label]]
        scale <- attr(term, "pparm")
        theta <- fit$history[[label]]$theta
        columns <- fit$assign[[label]]
        penalty_diagonal[columns] <- attr(term, "pfun")(
            rep(0, length(columns)), theta, fit$nevent, scale
        )$second
    }
    summary_fit <- summary(fit)
    bh <- basehaz(fit, centered = TRUE)
    bh_zero <- basehaz(fit, centered = FALSE)
    curve <- survfit(fit, newdata = newdata[1, , drop = FALSE])
    training_predictions <- predictions()
    new_predictions <- predictions(newdata)
    corrected_predictions <- list()
    corrected_residuals <- list()
    known_differences <- list()
    if (name == "weighted_offset_strata") {
        # survival 3.8.11 coxpenal.fit/coxfit5c fails to reset the expected
        # event accumulator across right-censored strata. R's ordinary Cox
        # residual routine at the same fitted beta supplies a valid reference.
        evaluated <- coxph(Surv(time, event) ~ u + x + z + offset(o) + strata(g),
                           data, weights = w, robust = FALSE, init = fit$coef,
                           control = coxph.control(iter.max = 0L))
        repaired <- fit
        repaired$residuals <- evaluated$residuals
        for (type in c("martingale", "deviance", "partial")) {
            corrected_residuals[[type]] <- unname(residuals(repaired, type = type))
        }
        corrected_predictions$training <- list(expected = list(
            fit = unname(data$event - evaluated$residuals),
            se_fit = training_predictions$expected$se_fit
        ))
        known_differences$stratum_residual <- paste(
            "R 3.8.11 coxpenal.fit returns incorrect martingales for earlier",
            "right-censored strata. Corrected residuals are ordinary R Cox",
            "residuals evaluated at the identical penalized coefficients."
        )
    }
    if (name %in% c("weighted_offset_strata", "counting_weighted_strata")) {
        corrected_predictions$newdata <- list(expected = interval_expected_reference(fit, newdata))
        known_differences$expected_offset <- paste(
            "R 3.8.11 predict.coxph newdata expected uses exp(X beta) + offset.",
            "The corrected independent agsurv reference uses exp(X beta + offset)."
        )
    }
    if (name == "counting_weighted_strata") {
        training_expected <- interval_expected_reference(fit)
        training_expected$fit <- training_predictions$expected$fit
        corrected_predictions$training <- list(expected = training_expected)
        known_differences$interval_uncertainty <- paste(
            "R 3.8.11 predict.coxph subtracts endpoint coefficient variances.",
            "The corrected agsurv reference differences the cumulative-hazard",
            "gradient before multiplying by V, preserving endpoint covariance."
        )
    }
    cases[[name]] <<- list(
        formula = formula, data = as.list(data), method = method,
        weighted = weighted,
        subset = if (is.null(subset)) NULL else I(which(subset) - 1L),
        initial_beta = init, max_iter = max_iter,
        coefficients = I(unname(coef(fit))),
        coefficient_names = I(names(coef(fit))),
        variance = unname(vcov(fit)), variance2 = unname(fit$var2),
        term_df = I(unname(fit$df)), df = unname(sum(fit$df)),
        term_names = I(names(fit$assign)),
        penalty_diagonal = I(penalty_diagonal), penalty = I(unname(fit$penalty)),
        means = I(unname(fit$means)), log_likelihood = I(unname(fit$loglik)),
        aic = unname(AIC(fit)), bic = unname(BIC(fit)),
        extract_aic = unname(extractAIC(fit)),
        summary_coefficients = unname(summary_fit$coefficients),
        summary_coefficient_names = I(rownames(summary_fit$coefficients)),
        summary_columns = I(colnames(summary_fit$coefficients)),
        model_matrix = unname(fit$x), model_matrix_names = I(colnames(fit$x)),
        predictions = training_predictions, newdata = as.list(newdata),
        new_predictions = new_predictions, residuals = resid,
        known_differences = known_differences,
        corrected_predictions = corrected_predictions, corrected_residuals = corrected_residuals,
        curve = list(time = unname(curve$time), surv = as.numeric(curve$surv),
                     cumhaz = as.numeric(curve$cumhaz),
                     std_chaz = as.numeric(curve$std.chaz),
                     lower = as.numeric(curve$lower), upper = as.numeric(curve$upper)),
        basehaz = list(time = bh$time, cumhaz = bh$hazard,
                       strata = if ("strata" %in% names(bh)) as.character(bh$strata) else NULL),
        basehaz_zero = list(time = bh_zero$time, cumhaz = bh_zero$hazard)
    )
}

add_case("mixed_scaled_efron", "Surv(time, event) ~ z + ridge(x, theta = 2)")
add_case("mixed_scaled_breslow", "Surv(time, event) ~ z + ridge(x, theta = 2)", method = "breslow")
add_case("mixed_scaled_exact", "Surv(time, event) ~ z + ridge(x, theta = 2)", method = "exact")
add_case("mixed_unscaled", "Surv(time, event) ~ z + ridge(x, theta = 2, scale = FALSE)")
add_case("zero_theta", "Surv(time, event) ~ z + ridge(x, theta = 0)")
add_case("weighted_scaled", "Surv(time, event) ~ z + ridge(x, theta = 2)", weighted = TRUE)
add_case("grouped_scaled", "Surv(time, event) ~ ridge(x, z, theta = 2)", correlated)
add_case("separate_equal_scaled", "Surv(time, event) ~ ridge(x, theta = 2) + ridge(z, theta = 2)", correlated)
add_case("separate_scaled", "Surv(time, event) ~ ridge(x, theta = 2) + ridge(z, theta = 5)", correlated)
add_case("duplicate_separate", "Surv(time, event) ~ ridge(x, theta = 2) + ridge(x, theta = 5)")
add_case("duplicate_grouped", "Surv(time, event) ~ ridge(x, x, theta = 2)")
add_case("transformed_argument", "Surv(time, event) ~ z + ridge(log(x), theta = 2)")
add_case("mixed_grouped", "Surv(time, event) ~ u + ridge(x, z, theta = 2)", correlated)
add_case("weighted_offset_strata", "Surv(time, event) ~ u + ridge(x, z, theta = 2) + offset(o) + strata(g)",
         correlated, weighted = TRUE)
add_case("counting_weighted_strata", "Surv(start, time, event) ~ u + ridge(x, z, theta = 2) + offset(o) + strata(g)",
         correlated, method = "breslow", weighted = TRUE)
subset_data <- small
subset_data$x[9:10] <- c(20, 40)
add_case("subset_frozen_scale", "Surv(time, event) ~ z + ridge(x, theta = 2)",
         subset_data, subset = seq_len(nrow(small)) <= 8)
omitted_other <- small
omitted_other$z[10] <- NA_real_
omitted_other$x[10] <- 40
add_case("omit_other_frozen_scale", "Surv(time, event) ~ z + ridge(x, theta = 2)", omitted_other)
omitted_ridge <- small
omitted_ridge$x[10] <- NA_real_
add_case("omit_ridge_frozen_scale", "Surv(time, event) ~ z + ridge(x, theta = 2)", omitted_ridge)
binary <- small
binary$z <- c(0, 1, 1, 0, 1, 0, 0, 1, 0, 1)
add_case("binary_nocenter", "Surv(time, event) ~ z + ridge(x, theta = 2)", binary)
constant <- small
constant$x <- 2
add_case("constant_unscaled", "Surv(time, event) ~ z + ridge(x, theta = 2, scale = FALSE)", constant)
duplicate <- small
duplicate$z <- duplicate$x
add_case("duplicate_unpenalized", "Surv(time, event) ~ z + ridge(x, theta = 2)", duplicate)
add_case("zero_iterations", "Surv(time, event) ~ z + ridge(x, theta = 2)",
         init = c(.3, -.2), max_iter = 0L)

jsonlite::write_json(
    list(reference = list(R = as.character(getRversion()),
                          survival = as.character(packageVersion("survival")),
                          generator = "scripts/generate_cox_ridge_reference.R"),
         cases = cases),
    "python/tests/fixtures/cox_ridge_r_reference.json",
    auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null"
)
