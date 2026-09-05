# Regenerate from the repository root:
# Rscript python/tests/fixtures/survreg_prediction_r3811.R
# Prediction fixtures deliberately use maxiter=0: optimizer changes are separate.
suppressPackageStartupMessages(library(survival))
stopifnot(as.character(packageVersion("survival")) == "3.8.11")

probabilities <- c(.1, .5, .9)
endpoints <- c(0, 1)
families <- c("gaussian", "logistic", "extreme", "t", "lognormal",
              "loglogistic", "weibull", "exponential", "rayleigh")
log_families <- c("lognormal", "loglogistic", "weibull", "exponential", "rayleigh")

prediction_set <- function(fit, newdata = NULL) {
  get_prediction <- function(type, p = probabilities) {
    args <- list(object = fit, type = type, p = p, se.fit = TRUE)
    if (!is.null(newdata)) args$newdata <- newdata
    suppressWarnings(do.call(predict, args))
  }
  list(lp = get_prediction("lp"), response = get_prediction("response"),
       quantile = get_prediction("quantile"), uquantile = get_prediction("uquantile"),
       quantile_endpoints = get_prediction("quantile", endpoints),
       uquantile_endpoints = get_prediction("uquantile", endpoints))
}

fixtures <- list()
for (distribution in families) {
  modes <- if (distribution %in% c("exponential", "rayleigh")) {
    c("fixed", "offset")
  } else c("estimated", "stratified", "fixed", "offset")
  for (mode in modes) {
    log_response <- distribution %in% log_families
    beta <- c(if (log_response) 2 else 10, .2)
    d <- data.frame(x = rep(c(-1, 0, 1), 4), status = 1,
                    group = rep(c("B", "A"), each = 6),
                    offset_value = if (mode == "offset") rep(c(.05, -.05, .1, -.1, 0, 0), 2) else 0)
    fixed_scale <- if (distribution == "exponential") 1 else if (distribution == "rayleigh") .5 else if (mode == "fixed") .8 else 0
    scales <- if (mode == "stratified") c(B = 1.15, A = .55) else if (fixed_scale > 0) fixed_scale else .8
    row_scales <- if (mode == "stratified") scales[d$group] else rep(scales, nrow(d))
    # Balanced residuals are orthogonal to x, keeping the observed information
    # positive definite at these intentionally unoptimized initial parameters.
    z <- rep(rep(c(-1.2, 1.2), each = 3), 2)
    y <- beta[1] + beta[2] * d$x + d$offset_value + row_scales * z
    d$time <- if (log_response) exp(y) else y
    formula <- if (mode == "stratified") Surv(time, status) ~ x + strata(group) else if (mode == "offset") Surv(time, status) ~ x + offset(offset_value) else Surv(time, status) ~ x
    # R orders strata alphabetically; the native fit codes first occurrence B=0, A=1.
    init <- c(beta, if (fixed_scale == 0) log(if (mode == "stratified") scales[c("A", "B")] else scales))
    args <- list(formula = formula, data = d, dist = distribution, scale = fixed_scale,
                 init = init, control = survreg.control(maxiter = 0), x = TRUE, y = TRUE)
    if (distribution == "t") args$parms <- 7
    fit <- do.call(survreg, args)
    newdata <- data.frame(x = c(.25, -.75, .5), group = c("A", "B", "A"), offset_value = 0)
    parameter_order <- if (mode == "stratified") c(1, 2, 4, 3) else seq_len(ncol(fit$var))
    fixtures[[paste(distribution, mode, sep = "_")]] <- list(
      distribution = distribution, mode = mode, log_response = log_response,
      data = as.list(d), newdata = as.list(newdata), beta = unname(beta),
      scales = unname(scales), fixed_scale = fixed_scale,
      parms = if (distribution == "t") 7 else NA_real_,
      initial = c(beta, if (fixed_scale == 0) log(unname(scales))),
      variance = unname(fit$var[parameter_order, parameter_order, drop = FALSE]),
      training = prediction_set(fit), new = prediction_set(fit, newdata))
  }
}

# Store R's unseen-stratum behavior as an audit observation, not a required
# validation policy: the Python facade deliberately raises on unknown levels.
d <- data.frame(time = c(5.5, 6.5, 13.5, 14.5, 8.5, 9.5, 10.5, 11.5),
                x = rep(c(-1, 1), 4), group = rep(c("B", "A"), each = 4))
fit <- survreg(Surv(time) ~ x + strata(group), data = d, dist = "gaussian",
               init = c(10, .5, 0, log(4)), control = survreg.control(maxiter = 0))
unknown <- lapply(c("lp", "response", "quantile"), function(type) {
  predict(fit, data.frame(x = 0, group = "unseen"), type = type, p = .9)
})
names(unknown) <- c("lp", "response", "quantile")
out <- list(source = "R survival 3.8.11 predict.survreg; exact responses; maxiter=0",
            probabilities = probabilities, endpoints = endpoints,
            notes = c("New-data offsets are zero in R references because predict.survreg omits new-data offsets; separate Python tests check their mathematical effect.",
                      "Estimated-scale endpoint standard errors reflect R's nonfinite arithmetic and are not a stable NaN-versus-Inf contract."),
            unknown_stratum_audit = unknown,
            standalone_t_df4 = qsurvreg(c(0, probabilities, 1), 2, .8, "t", parms = 4),
            fixtures = fixtures)
jsonlite::write_json(out, "python/tests/fixtures/survreg_prediction_r3811.json",
                     auto_unbox = TRUE, digits = 16, pretty = TRUE, na = "string")
