#!/usr/bin/env Rscript

library(survival)
library(jsonlite)

cat("Generating expected values from R survival package...\n")
cat("survival package version:", as.character(packageVersion("survival")), "\n")
cat("R version:", R.version.string, "\n\n")

clean_na <- function(x) {
  if (is.list(x)) {
    return(lapply(x, clean_na))
  }
  if (is.numeric(x)) {
    x[is.na(x) | is.nan(x)] <- NA
    x[abs(x) < 1e-14] <- 0
  }
  return(x)
}

results <- list(
  metadata = list(
    survival_version = as.character(packageVersion("survival")),
    r_version = R.version.string,
    note = "These values were generated using the R survival package. Regenerate with: Rscript generate_r_expected_values.R"
  )
)

cat("Processing AML dataset...\n")
data(aml, package = "survival")

aml_maintained <- aml[aml$x == "Maintained", ]
aml_nonmaintained <- aml[aml$x == "Nonmaintained", ]

results$aml <- list(
  maintained = list(
    time = aml_maintained$time,
    status = aml_maintained$status
  ),
  nonmaintained = list(
    time = aml_nonmaintained$time,
    status = aml_nonmaintained$status
  ),
  combined = list(
    time = aml$time,
    status = aml$status,
    group = as.integer(aml$x == "Maintained")
  )
)

km_maintained <- survfit(Surv(time, status) ~ 1, data = aml_maintained)
km_summary <- summary(km_maintained)
results$aml$km_maintained <- list(
  time = km_summary$time,
  n_risk = km_summary$n.risk,
  n_event = km_summary$n.event,
  n_censor = km_summary$n.censor,
  survival = km_summary$surv,
  std_err = km_summary$std.err,
  lower = km_summary$lower,
  upper = km_summary$upper
)

km_nonmaintained <- survfit(Surv(time, status) ~ 1, data = aml_nonmaintained)
km_summary_nm <- summary(km_nonmaintained)
results$aml$km_nonmaintained <- list(
  time = km_summary_nm$time,
  n_risk = km_summary_nm$n.risk,
  n_event = km_summary_nm$n.event,
  n_censor = km_summary_nm$n.censor,
  survival = km_summary_nm$surv,
  std_err = km_summary_nm$std.err
)

na_maintained <- survfit(Surv(time, status) ~ 1, data = aml_maintained, type = "fh")
results$aml$nelson_aalen_maintained <- list(
  time = na_maintained$time,
  n_risk = na_maintained$n.risk,
  n_event = na_maintained$n.event,
  cumulative_hazard = na_maintained$cumhaz
)

sd <- survdiff(Surv(time, status) ~ x, data = aml)
results$aml$logrank <- list(
  n = as.vector(sd$n),
  observed = as.vector(sd$obs),
  expected = as.vector(sd$exp),
  chisq = as.numeric(sd$chisq),
  df = length(sd$n) - 1,
  p_value = 1 - pchisq(sd$chisq, df = length(sd$n) - 1)
)

sd_wilcox <- survdiff(Surv(time, status) ~ x, data = aml, rho = 1)
results$aml$wilcoxon <- list(
  chisq = as.numeric(sd_wilcox$chisq),
  p_value = 1 - pchisq(sd_wilcox$chisq, df = 1)
)

cox_breslow <- coxph(Surv(time, status) ~ x, data = aml, method = "breslow")
cox_summary <- summary(cox_breslow)
results$aml$coxph_breslow <- list(
  coefficients = I(as.vector(coef(cox_breslow))),
  se = I(as.vector(cox_summary$coefficients[, "se(coef)"])),
  hazard_ratio = I(as.vector(exp(coef(cox_breslow)))),
  hr_lower = as.numeric(exp(confint(cox_breslow))[1]),
  hr_upper = as.numeric(exp(confint(cox_breslow))[2]),
  loglik = as.vector(cox_breslow$loglik),
  score_test = as.numeric(cox_summary$sctest["test"]),
  wald_test = as.numeric(cox_summary$waldtest["test"]),
  lr_test = as.numeric(cox_summary$logtest["test"]),
  concordance = as.numeric(cox_summary$concordance["C"])
)

cox_efron <- coxph(Surv(time, status) ~ x, data = aml, method = "efron")
results$aml$coxph_efron <- list(
  coefficients = I(as.vector(coef(cox_efron))),
  se = I(as.vector(summary(cox_efron)$coefficients[, "se(coef)"])),
  hazard_ratio = I(as.vector(exp(coef(cox_efron)))),
  loglik = as.vector(cox_efron$loglik)
)

km_combined <- survfit(Surv(time, status) ~ x, data = aml)
median_surv <- summary(km_combined)$table[, "median"]
results$aml$median_survival <- list(
  maintained = as.numeric(median_surv["x=Maintained"]),
  nonmaintained = as.numeric(median_surv["x=Nonmaintained"])
)

results$aml$martingale_residuals <- list(
  sum = sum(residuals(cox_breslow, type = "martingale"))
)

cat("Processing lung dataset...\n")
data(lung, package = "survival")
lung_subset <- lung[1:20, ]
lung_subset$status_01 <- lung_subset$status - 1

results$lung <- list(
  data = list(
    time = lung_subset$time,
    status = lung_subset$status_01,
    sex = lung_subset$sex,
    age = lung_subset$age
  )
)

cox_lung <- coxph(Surv(time, status_01) ~ age + sex, data = lung_subset, method = "breslow")
cox_lung_summary <- summary(cox_lung)
results$lung$coxph <- list(
  coefficients = as.vector(coef(cox_lung)),
  se = as.vector(cox_lung_summary$coefficients[, "se(coef)"]),
  hazard_ratio = as.vector(exp(coef(cox_lung))),
  loglik = as.vector(cox_lung$loglik),
  concordance = as.numeric(cox_lung_summary$concordance["C"])
)

sd_lung <- survdiff(Surv(time, status_01) ~ sex, data = lung_subset)
results$lung$logrank_sex <- list(
  chisq = as.numeric(sd_lung$chisq),
  p_value = 1 - pchisq(sd_lung$chisq, df = 1)
)

cat("Processing ovarian dataset...\n")
data(ovarian, package = "survival")

results$ovarian <- list(
  data = list(
    time = ovarian$futime,
    status = ovarian$fustat,
    rx = ovarian$rx,
    age = ovarian$age
  )
)

sd_ovarian <- survdiff(Surv(futime, fustat) ~ rx, data = ovarian)
results$ovarian$logrank <- list(
  chisq = as.numeric(sd_ovarian$chisq),
  p_value = 1 - pchisq(sd_ovarian$chisq, df = 1),
  observed = as.vector(sd_ovarian$obs),
  expected = as.vector(sd_ovarian$exp)
)

km_ovarian <- survfit(Surv(futime, fustat) ~ 1, data = ovarian)
km_ov_summary <- summary(km_ovarian)
results$ovarian$km <- list(
  time = km_ov_summary$time,
  survival = km_ov_summary$surv,
  n_risk = km_ov_summary$n.risk,
  n_event = km_ov_summary$n.event
)

cox_ovarian <- coxph(Surv(futime, fustat) ~ rx + age, data = ovarian)
results$ovarian$coxph <- list(
  coefficients = as.vector(coef(cox_ovarian)),
  se = as.vector(summary(cox_ovarian)$coefficients[, "se(coef)"]),
  hazard_ratio = as.vector(exp(coef(cox_ovarian))),
  loglik = as.vector(cox_ovarian$loglik)
)

cat("Processing veteran dataset...\n")
data(veteran, package = "survival")
veteran_subset <- veteran[1:20, ]

results$veteran <- list(
  data = list(
    time = veteran_subset$time,
    status = veteran_subset$status,
    trt = veteran_subset$trt,
    age = veteran_subset$age
  )
)

km_veteran <- survfit(Surv(time, status) ~ 1, data = veteran_subset)
km_vet_summary <- summary(km_veteran)
results$veteran$km <- list(
  time = km_vet_summary$time,
  survival = km_vet_summary$surv,
  n_risk = km_vet_summary$n.risk,
  n_event = km_vet_summary$n.event
)

cox_veteran <- coxph(Surv(time, status) ~ trt + age, data = veteran_subset)
results$veteran$coxph <- list(
  coefficients = as.vector(coef(cox_veteran)),
  hazard_ratio = as.vector(exp(coef(cox_veteran))),
  loglik = as.vector(cox_veteran$loglik)
)

cat("Processing edge cases...\n")

tied_data <- data.frame(
  time = c(5, 5, 5, 10, 10, 15),
  status = c(1, 1, 0, 1, 1, 1)
)
km_tied <- survfit(Surv(time, status) ~ 1, data = tied_data)
km_tied_summary <- summary(km_tied)
results$edge_cases$tied_events <- list(
  time = km_tied_summary$time,
  survival = km_tied_summary$surv,
  n_risk = km_tied_summary$n.risk,
  n_event = km_tied_summary$n.event
)

same_time_data <- data.frame(
  time = c(5, 5, 5, 5, 5),
  status = c(1, 1, 1, 1, 1)
)
km_same <- survfit(Surv(time, status) ~ 1, data = same_time_data)
results$edge_cases$all_same_time <- list(
  time = I(km_same$time),
  survival = I(km_same$surv),
  n_risk = I(km_same$n.risk),
  n_event = I(km_same$n.event)
)

simple_data <- data.frame(
  time = c(1, 2, 3, 4, 5),
  status = c(1, 1, 1, 1, 1)
)
na_simple <- survfit(Surv(time, status) ~ 1, data = simple_data, type = "fh")
results$edge_cases$simple_nelson_aalen <- list(
  time = na_simple$time,
  cumulative_hazard = na_simple$cumhaz,
  n_risk = na_simple$n.risk
)

censored_data <- data.frame(
  time = c(1, 2, 3, 4, 5, 6),
  status = c(1, 0, 1, 0, 1, 0)
)
na_censored <- survfit(Surv(time, status) ~ 1, data = censored_data, type = "fh")
na_censored_summary <- summary(na_censored)
results$edge_cases$with_censoring <- list(
  time = na_censored_summary$time,
  cumulative_hazard = na_censored$cumhaz[na_censored$n.event > 0],
  survival = na_censored_summary$surv,
  n_risk = na_censored_summary$n.risk,
  n_event = na_censored_summary$n.event
)

identical_data <- data.frame(
  time = c(1, 2, 3, 1, 2, 3),
  status = c(1, 1, 1, 1, 1, 1),
  group = c(0, 0, 0, 1, 1, 1)
)
sd_identical <- survdiff(Surv(time, status) ~ group, data = identical_data)
results$edge_cases$identical_groups_logrank <- list(
  chisq = as.numeric(sd_identical$chisq),
  p_value = 1 - pchisq(sd_identical$chisq, df = 1)
)

cat("Processing sample size calculations...\n")

calc_sample_size <- function(hr, power, alpha, ratio = 1) {
  z_alpha <- qnorm(1 - alpha/2)
  z_beta <- qnorm(power)
  p1 <- 1 / (1 + ratio)
  p2 <- ratio / (1 + ratio)
  n_events <- (z_alpha + z_beta)^2 / (log(hr)^2 * p1 * p2)
  return(ceiling(n_events))
}

results$sample_size <- list(
  "hr_0.5_power_0.8" = calc_sample_size(0.5, 0.8, 0.05),
  "hr_0.6_power_0.8" = calc_sample_size(0.6, 0.8, 0.05),
  "hr_0.7_power_0.8" = calc_sample_size(0.7, 0.8, 0.05),
  "hr_0.6_power_0.9" = calc_sample_size(0.6, 0.9, 0.05)
)

cat("Processing RMST calculations...\n")

calc_rmst <- function(km_fit, tau) {
  times <- c(0, km_fit$time[km_fit$time <= tau])
  surv <- c(1, km_fit$surv[km_fit$time <= tau])

  if (max(km_fit$time) < tau) {
    times <- c(times, tau)
    surv <- c(surv, surv[length(surv)])
  } else {
    times <- c(times, tau)
    idx <- which(km_fit$time >= tau)[1]
    if (idx > 1) {
      surv <- c(surv, km_fit$surv[idx])
    } else {
      surv <- c(surv, km_fit$surv[1])
    }
  }

  rmst <- 0
  for (i in 2:length(times)) {
    rmst <- rmst + surv[i-1] * (times[i] - times[i-1])
  }
  return(rmst)
}

km_aml_maint <- survfit(Surv(time, status) ~ 1, data = aml_maintained)
km_aml_nonmaint <- survfit(Surv(time, status) ~ 1, data = aml_nonmaintained)

results$rmst <- list(
  aml_maintained_tau30 = calc_rmst(km_aml_maint, 30),
  aml_maintained_tau48 = calc_rmst(km_aml_maint, 48),
  aml_nonmaintained_tau30 = calc_rmst(km_aml_nonmaint, 30),
  aml_nonmaintained_tau48 = calc_rmst(km_aml_nonmaint, 48)
)

cat("Processing concordance calculations...\n")

conc_aml <- concordance(cox_breslow)
conc_counts <- as.vector(conc_aml$count)
names(conc_counts) <- names(conc_aml$count)
get_count <- function(name) {
  val <- conc_counts[name]
  if (is.na(val)) 0L else as.integer(val)
}
results$concordance <- list(
  aml_coxph = list(
    concordance = as.numeric(conc_aml$concordance),
    n_concordant = get_count("concordant"),
    n_discordant = get_count("discordant"),
    n_tied_risk = get_count("tied.risk"),
    n_tied_time = get_count("tied.time")
  )
)

output_file <- "r_expected_values.json"
cat("\nWriting results to", output_file, "...\n")

results <- clean_na(results)
json_output <- toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10, na = "null")
writeLines(json_output, output_file)

cat("Done! Generated", output_file, "\n")
