# Regenerate with Rscript scripts/generate_cox_ridge_df_reference.R.
# Python tests read the JSON only; they do not require an R installation.
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.8.11")
stopifnot(requireNamespace("jsonlite", quietly = TRUE))

fixed <- jsonlite::fromJSON("python/tests/fixtures/cox_ridge_r_reference.json")
small <- as.data.frame(fixed$cases$mixed_scaled_efron$data)
correlated <- as.data.frame(fixed$cases$grouped_scaled$data)
cases <- list()

add_case <- function(name, formula, data = small, method = "efron", weighted = FALSE,
                     subset = NULL, init = NULL, max_iter = 50L, outer_max = 10L,
                     standalone_df = NULL, standalone_theta = NULL, standalone_scale = TRUE) {
    args <- list(formula = as.formula(formula), data = data, ties = method,
                 robust = FALSE, x = TRUE, model = TRUE, na.action = na.omit,
                 control = coxph.control(iter.max = max_iter, outer.max = outer_max, eps = 1e-11))
    if (weighted) args$weights <- data$w
    if (!is.null(subset)) args$subset <- subset
    if (!is.null(init)) args$init <- init
    fit <- tryCatch(do.call(coxph, args), error = function(error) {
        stop(sprintf("reference case %s: %s", name, conditionMessage(error)), call. = FALSE)
    })
    histories <- list()
    applied_theta <- list()
    penalty_diagonal <- rep(0, ncol(fit$x))
    for (label in names(fit$history)) {
        value <- fit$history[[label]]
        history <- value$history
        # The R controller returns a next proposal even when it is done.
        # Coefficients/covariance use the final evaluated row, not that proposal.
        theta <- if (is.null(history)) value$theta else tail(history[, 1], 1)
        histories[[label]] <- list(theta = unname(value$theta), done = value$done,
                                  history = if (is.null(history)) NULL else unname(history),
                                  half = value$half)
        applied_theta[[label]] <- unname(theta)
        term <- fit$model[[label]]
        columns <- fit$assign[[label]]
        penalty_diagonal[columns] <- attr(term, "pfun")(
            rep(0, length(columns)), theta, fit$nevent, attr(term, "pparm")
        )$second
    }
    scale_factors <- sqrt(apply(fit$x, 2, var))
    summary_fit <- summary(fit)
    cases[[name]] <<- list(
        formula = formula, data = as.list(data), method = method, weighted = weighted,
        subset = if (is.null(subset)) NULL else I(which(subset) - 1L),
        initial_beta = init, max_iter = max_iter, outer_max = outer_max,
        coefficients = I(unname(coef(fit))), coefficient_names = I(names(coef(fit))),
        variance = unname(fit$var), variance2 = unname(fit$var2),
        term_df = I(unname(fit$df)), df = sum(fit$df),
        means = I(unname(fit$means)), penalty = I(unname(fit$penalty)),
        penalty_diagonal = I(penalty_diagonal), log_likelihood = I(unname(fit$loglik)),
        iterations = I(unname(fit$iter)), history = histories, applied_theta = applied_theta,
        aic = unname(AIC(fit)), bic = unname(BIC(fit)),
        term_names = I(names(fit$assign)), model_matrix = unname(fit$x),
        std_err = I(sqrt(diag(fit$var))), scale_factors = I(unname(scale_factors)),
        standalone_df = standalone_df, standalone_theta = standalone_theta,
        standalone_scale = standalone_scale,
        summary_logtest = I(unname(summary_fit$logtest))
    )
}

add_case("default_single", "Surv(time, event) ~ z + ridge(x)")
add_case("default_grouped", "Surv(time, event) ~ ridge(x, z)", standalone_df = 1)
add_case("default_correlated", "Surv(time, event) ~ ridge(x, z)", correlated, standalone_df = 1)
add_case("explicit_default_epsilon", "Surv(time, event) ~ ridge(x, z, df = 0.6)", standalone_df = .6)
add_case("tight_epsilon", "Surv(time, event) ~ ridge(x, z, df = 0.6, eps = 0.001)")
add_case("unscaled", "Surv(time, event) ~ ridge(x, z, df = 0.8, scale = FALSE)", standalone_df = .8,
         standalone_scale = FALSE)
add_case("unscaled_tight", "Surv(time, event) ~ z + ridge(x, df = 0.4, eps = 0.001, scale = FALSE)")
add_case("separate_selected", "Surv(time, event) ~ ridge(x, df = 0.35, eps = 0.001) + ridge(z, df = 0.7, eps = 0.001)")
add_case("mixed_fixed_selected", "Surv(time, event) ~ ridge(x, df = 0.4) + ridge(z, theta = 2)")
add_case("mixed_unpenalized_grouped", "Surv(time, event) ~ u + ridge(x, z, df = 0.7, eps = 0.01)", correlated)
add_case("weighted", "Surv(time, event) ~ ridge(x, z, df = 0.7)", weighted = TRUE, standalone_df = .7)
add_case("weighted_offset_strata", "Surv(time, event) ~ u + ridge(x, z, df = 1.2, eps = 0.01) + offset(o) + strata(g)",
         correlated, weighted = TRUE)
add_case("counting_breslow", "Surv(start, time, event) ~ u + ridge(x, z, df = 1.2, eps = 0.01) + offset(o) + strata(g)",
         correlated, weighted = TRUE, method = "breslow")
add_case("breslow", "Surv(time, event) ~ z + ridge(x, df = 0.4, eps = 0.01)", method = "breslow")
add_case("exact_uses_breslow", "Surv(time, event) ~ z + ridge(x, df = 0.4, eps = 0.01)", method = "exact")
add_case("near_full_df", "Surv(time, event) ~ ridge(x, z, df = 1.95, eps = 0.001)", correlated)
add_case("full_df", "Surv(time, event) ~ ridge(x, z, df = 2, eps = 0.001)", correlated)
add_case("near_zero_df", "Surv(time, event) ~ ridge(x, z, df = 0.05, eps = 0.001)", correlated)
add_case("zero_df", "Surv(time, event) ~ ridge(x, z, df = 0)", correlated, standalone_df = 0)
add_case("outer_limit", "Surv(time, event) ~ ridge(x, z, df = 0, eps = 0.001)", correlated, outer_max = 4L)
add_case("coupled_outer_limit", "Surv(time, event) ~ ridge(x, df = 0.3, eps = 0.001) + ridge(z, df = 0.7, eps = 0.001)",
         correlated)
subset_data <- small
subset_data$x[9:10] <- c(2, 4)
add_case("subset_frozen_scale", "Surv(time, event) ~ z + ridge(x, df = 0.6)", subset_data,
         subset = seq_len(nrow(small)) <= 8)
subset_data$x[9:10] <- c(20, 40)
add_case("subset_roundoff_sensitive_controller", "Surv(time, event) ~ z + ridge(x, df = 0.6)", subset_data,
         subset = seq_len(nrow(small)) <= 8)
# At theta about 2e-41 R obtains df=1-2^-53; the Rust factorization obtains
# df=1 exactly. Log-power interpolation amplifies this one-bit distinction.
# Keep the raw R output for inspection, but verify the attained Rust fit against
# a fixed-theta fit instead of asserting unstable controller-path equality.
cases$subset_roundoff_sensitive_controller$controller_roundoff_sensitive <- TRUE
missing_data <- small
missing_data$z[10] <- NA_real_
missing_data$x[10] <- 4
add_case("omission_frozen_scale", "Surv(time, event) ~ z + ridge(x, df = 0.6)", missing_data)
add_case("nonzero_initial", "Surv(time, event) ~ z + ridge(x, df = 0.4, eps = 0.01)", init = c(.3, -.2))
add_case("one_outer_fit", "Surv(time, event) ~ z + ridge(x, df = 0.4, eps = 0.01)",
         init = c(.3, -.2), outer_max = 1L)
add_case("zero_inner_limit", "Surv(time, event) ~ z + ridge(x, df = 0.5, eps = 0.001)",
         weighted = TRUE, init = c(.4, -.2), max_iter = 0L)
add_case("standalone_fixed_scaled", "Surv(time, event) ~ ridge(x, z, theta = 2)", correlated,
         standalone_theta = 2)
add_case("standalone_fixed_unscaled", "Surv(time, event) ~ ridge(x, z, theta = 2, scale = FALSE)", correlated,
         standalone_theta = 2, standalone_scale = FALSE)
add_case("small_fixed_scaled", "Surv(time, event) ~ ridge(x, z, theta = 2)", standalone_theta = 2)
add_case("small_fixed_unscaled", "Surv(time, event) ~ ridge(x, z, theta = 2, scale = FALSE)",
         standalone_theta = 2, standalone_scale = FALSE)
add_case("weighted_fixed_scaled", "Surv(time, event) ~ ridge(x, z, theta = 2)", weighted = TRUE,
         standalone_theta = 2)
add_case("weighted_fixed_unscaled", "Surv(time, event) ~ ridge(x, z, theta = 2, scale = FALSE)", weighted = TRUE,
         standalone_theta = 2, standalone_scale = FALSE)

jsonlite::write_json(
    list(reference = list(R = as.character(getRversion()),
                          survival = as.character(packageVersion("survival")),
                          generator = "scripts/generate_cox_ridge_df_reference.R"),
         cases = cases),
    "python/tests/fixtures/cox_ridge_df_r_reference.json",
    auto_unbox = TRUE, pretty = TRUE, digits = 17, na = "null", null = "null"
)
