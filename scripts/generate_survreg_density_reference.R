#!/usr/bin/env Rscript
# Rscript scripts/generate_survreg_density_reference.R [output.json]
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else "python/tests/fixtures/survreg_density_reference.json"

# A skewed mixture, deliberately outside survival's built-in families.
density <- function(z, parms = NULL) {
    v <- (z - 1.4) / .8
    a <- .65 * dnorm(z)
    b <- .35 * dnorm(v) / .8
    f <- a + b
    cbind(.65 * pnorm(z) + .35 * pnorm(v),
          .65 * pnorm(z, lower.tail = FALSE) + .35 * pnorm(v, lower.tail = FALSE),
          f, (-z * a - v * b / .8) / f,
          ((z*z - 1) * a + (v*v - 1) * b / .64) / f)
}
mixture <- list(name = "Two normal mixture", density = density,
    variance = function(...) .65 + .35 * .8^2 + .65 * .35 * 1.4^2,
    init = function(x, weights, ...) {
        mu <- sum(x * weights) / sum(weights)
        c(mu, sum(weights * (x - mu)^2) / sum(weights))
    },
    deviance = function(y, scale, ...) {
        scale <- rep_len(scale, nrow(y))
        status <- y[, ncol(y)]
        center <- y[, 1]
        loglik <- rep(0, nrow(y))
        mode <- uniroot(function(z) density(z)[1, 4], c(-2, 3), tol=1e-12)$root
        exact <- status == 1
        center[exact] <- y[exact, 1] - scale[exact] * mode
        loglik[exact] <- log(density(mode)[1, 3]) - log(scale[exact])
        for (i in which(status == 3)) {
            score <- function(eta) diff(-density((y[i, 1:2] - eta) / scale[i])[, 3])
            center[i] <- uniroot(score, c(y[i, 1] - 8*scale[i], y[i, 2] + 8*scale[i]),
                                  tol=1e-12)$root
            z <- (y[i, 1:2] - center[i]) / scale[i]
            d <- density(z)
            loglik[i] <- log(if (z[1] > 0) d[1, 2] - d[2, 2] else d[2, 1] - d[1, 1])
        }
        list(center=center, loglik=loglik)
    },
    quantile = function(p, ...) vapply(p, function(q)
        if (q == 0) -Inf else if (q == 1) Inf else
        uniroot(function(z) density(z)[1, 1] - q, c(-20, 20), tol=1e-12)$root, 0.0))

i <- 0:39
x <- (i %% 7 - 3) / 2
offset <- (i %% 3 - 1) / 10
weights <- .5 + (i %% 4) / 3
y1 <- .6 + .35*x + offset + qnorm(((i * 17) %% 41 + .5) / 41)
y2 <- y1 + .6
status <- rep(c(1, 1, 0, 2, 3), 8)
g <- i %% 2
data <- data.frame(y1, y2, status, x, offset, weights, g)
cases <- lapply(c("estimated", "stratified", "fixed"), function(kind) {
    nstrat <- switch(kind, estimated = 1, stratified = 2, fixed = 0)
    init <- if (nstrat == 0) c(.2, -.1) else c(.2, -.1, rep(.1, nstrat))
    scale <- if (nstrat == 0) .9 else 0
    formula <- if (nstrat == 2) Surv(y1, y2, status, type = "interval") ~ x + offset(offset) + strata(g)
               else Surv(y1, y2, status, type = "interval") ~ x + offset(offset)
    fit <- survreg(formula, data, weights = weights,
                   dist = mixture, init = init, scale = scale,
                   control = survreg.control(maxiter = 50, rel.tolerance = 1e-11))
    # survreg7.c allocates z for n endpoints, but survregc2.c writes a second
    # endpoint for each interval. Avoid that out-of-bounds write in the R
    # reference: compare its penalized fitter with intervals made exact,
    # and use a separate R optimizer for the full interval likelihood.
    no_interval <- data
    no_interval$status[no_interval$status == 3] <- 1
    penalized <- survreg(update(formula, . ~ . - x + ridge(x, theta = .7, scale = FALSE)),
                        no_interval, weights = weights, dist = mixture, init = init, scale = scale,
                        control = survreg.control(maxiter = 50, rel.tolerance = 1e-11))
    objective <- function(beta) {
        sigma <- if (nstrat == 0) rep(scale, nrow(data)) else
                 exp(beta[3 + if (nstrat == 2) g else rep(0, nrow(data))])
        eta <- beta[1] + beta[2] * x + offset
        lower <- density((y1 - eta) / sigma)
        upper <- density((y2 - eta) / sigma)
        likelihood <- ifelse(status == 1, lower[, 3] / sigma,
                      ifelse(status == 0, lower[, 2],
                      ifelse(status == 2, lower[, 1], upper[, 1] - lower[, 1])))
        -sum(weights * log(likelihood)) + .7 * beta[2]^2 / 2
    }
    independent <- optim(init, objective, method = "BFGS",
                         control = list(maxit = 1000, reltol = 1e-13,
                                        ndeps = rep(1e-5, length(init))))
    stopifnot(independent$convergence == 0)
    postfit <- function(model) list(
        response = predict(model, type="response", se.fit=TRUE),
        quantile = predict(model, type="quantile", p=c(.1, .5, .9), se.fit=TRUE),
        residuals = lapply(setNames(c("response", "deviance", "working", "matrix"),
                                   c("response", "deviance", "working", "matrix")),
                           function(type) unname(residuals(model, type=type))))
    transformed <- data
    transformed$y1 <- sinh(y1)
    transformed$y2 <- sinh(y2)
    asinh_mixture <- c(mixture, list(trans=asinh, dtrans=function(y) 1/sqrt(1+y*y), itrans=sinh))
    transformed_fit <- survreg(formula, transformed, weights=weights, dist=asinh_mixture,
                               init=init, scale=scale,
                               control=survreg.control(maxiter=50, rel.tolerance=1e-11))
    robust_fit <- survreg(formula, no_interval, weights=weights, dist=mixture,
                         init=init, scale=scale, cluster=rep(1:10, 4), robust=TRUE,
                         control=survreg.control(maxiter=50, rel.tolerance=1e-11))
    list(name = kind, nstrat = nstrat,
         postfit = postfit(fit),
         transformed = list(loglik=transformed_fit$loglik, postfit=postfit(transformed_fit)),
         robust_no_interval = list(parameters=unname(c(coef(robust_fit), log(robust_fit$scale))),
                                   variance=unname(robust_fit$var),
                                   residuals=unname(residuals(robust_fit, type="dfbeta"))),
         init = if (nstrat == 0) c(init, log(scale)) else init,
         expected = list(parameters = unname(c(coef(fit), log(fit$scale))),
                         variance = unname(fit$var), loglik = fit$loglik[2]),
         penalized_no_interval = list(parameters = unname(c(coef(penalized), log(penalized$scale))),
                                      loglik = penalized$loglik[2] - penalized$penalty[2],
                                      penalty = penalized$penalty[2]),
         penalized_interval = list(parameters = if (nstrat == 0) c(independent$par, log(scale))
                                               else independent$par,
                                   loglik = -independent$value))
})
reference <- list(metadata = list(generator = "scripts/generate_survreg_density_reference.R",
    r_version = as.character(getRversion()),
    survival_version = as.character(packageVersion("survival"))), data = data, cases = cases)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(reference, output, auto_unbox = TRUE, digits = 17, pretty = TRUE, dataframe = "columns")
cat(length(cases), "custom survreg density cases written to", output, "\n")
