#!/usr/bin/env Rscript
#
# Generate expected values from R survival package for validation testing
#
# This script generates a JSON file with exact expected values computed by
# the R survival package (https://github.com/therneau/survival).
#
# Usage: Rscript generate_r_expected_values.R
# Output: r_expected_values.json
#
# Requirements:
#   install.packages(c("survival", "jsonlite"))

library(survival)
library(jsonlite)

cat("Generating expected values from R survival package...\n")
cat("survival package version:", as.character(packageVersion("survival")), "\n")
cat("R version:", R.version.string, "\n\n")

# Initialize results list
results <- list(
  metadata = list(
    survival_version = as.character(packageVersion("survival")),
    r_version = R.version.string,
    generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
)

# =============================================================================
# AML Dataset
# =============================================================================
cat("Processing AML dataset...\n")
data(aml)

# Split into maintained and non-maintained groups
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

# Kaplan-Meier for maintained group
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

# Kaplan-Meier for non-maintained group
km_nonmaintained <- survfit(Surv(time, status) ~ 1, data = aml_nonmaintained)
km_summary_nm <- summary(km_nonmaintained)
results$aml$km_nonmaintained <- list(
  time = km_summary_nm$time,
  n_risk = km_summary_nm$n.risk,
  n_event = km_summary_nm$n.event,
  n_censor = km_summary_nm$n.censor,
  survival = km_summary_nm$surv,
  std_err = km_summary_nm$std.err,
  lower = km_summary_nm$lower,
  upper = km_summary_nm$upper
)

# Nelson-Aalen for maintained group
na_maintained <- survfit(Surv(time, status) ~ 1, data = aml_maintained, type = "fh")
results$aml$nelson_aalen_maintained <- list(
  time = na_maintained$time,
  n_risk = na_maintained$n.risk,
  n_event = na_maintained$n.event,
  cumulative_hazard = na_maintained$cumhaz,
  std_err = na_maintained$std.err
)

# Log-rank test (survdiff)
sd <- survdiff(Surv(time, status) ~ x, data = aml)
results$aml$logrank <- list(
  n = sd$n,
  observed = sd$obs,
  expected = sd$exp,
  chisq = sd$chisq,
  df = length(sd$n) - 1,
  p_value = 1 - pchisq(sd$chisq, df = length(sd$n) - 1)
)

# Wilcoxon (Peto-Peto) test
sd_wilcox <- survdiff(Surv(time, status) ~ x, data = aml, rho = 1)
results$aml$wilcoxon <- list(
  chisq = sd_wilcox$chisq,
  p_value = 1 - pchisq(sd_wilcox$chisq, df = 1)
)

# Cox PH with Breslow method
cox_breslow <- coxph(Surv(time, status) ~ x, data = aml, method = "breslow")
cox_summary <- summary(cox_breslow)
results$aml$coxph_breslow <- list(
  coefficients = as.vector(coef(cox_breslow)),
  se = cox_summary$coefficients[, "se(coef)"],
  hazard_ratio = exp(coef(cox_breslow)),
  hr_lower = exp(confint(cox_breslow))[1],
  hr_upper = exp(confint(cox_breslow))[2],
  loglik = cox_breslow$loglik,
  score_test = cox_summary$sctest["test"],
  wald_test = cox_summary$waldtest["test"],
  lr_test = cox_summary$logtest["test"],
  concordance = cox_summary$concordance["C"]
)

# Cox PH with Efron method
cox_efron <- coxph(Surv(time, status) ~ x, data = aml, method = "efron")
results$aml$coxph_efron <- list(
  coefficients = as.vector(coef(cox_efron)),
  se = summary(cox_efron)$coefficients[, "se(coef)"],
  hazard_ratio = exp(coef(cox_efron)),
  loglik = cox_efron$loglik
)

# Median survival
km_combined <- survfit(Surv(time, status) ~ x, data = aml)
median_surv <- summary(km_combined)$table[, "median"]
results$aml$median_survival <- list(
  maintained = as.numeric(median_surv["x=Maintained"]),
  nonmaintained = as.numeric(median_surv["x=Nonmaintained"])
)

# Martingale residuals
results$aml$martingale_residuals <- list(
  residuals = as.vector(residuals(cox_breslow, type = "martingale")),
  sum = sum(residuals(cox_breslow, type = "martingale"))
)

# Schoenfeld residuals
schoen <- residuals(cox_breslow, type = "schoenfeld")
results$aml$schoenfeld_residuals <- list(
  residuals = as.vector(schoen)
)

# =============================================================================
# Lung Dataset (subset)
# =============================================================================
cat("Processing lung dataset...\n")
data(lung)
lung_subset <- lung[1:20, ]
lung_subset$status_01 <- lung_subset$status - 1  # Convert to 0/1

results$lung <- list(
  data = list(
    time = lung_subset$time,
    status = lung_subset$status_01,
    sex = lung_subset$sex,
    age = lung_subset$age,
    ph_ecog = lung_subset$ph.ecog
  )
)

# Cox PH with age and sex
cox_lung <- coxph(Surv(time, status_01) ~ age + sex, data = lung_subset, method = "breslow")
cox_lung_summary <- summary(cox_lung)
results$lung$coxph <- list(
  coefficients = as.vector(coef(cox_lung)),
  se = cox_lung_summary$coefficients[, "se(coef)"],
  hazard_ratio = exp(coef(cox_lung)),
  loglik = cox_lung$loglik,
  concordance = cox_lung_summary$concordance["C"]
)

# Log-rank by sex
sd_lung <- survdiff(Surv(time, status_01) ~ sex, data = lung_subset)
results$lung$logrank_sex <- list(
  chisq = sd_lung$chisq,
  p_value = 1 - pchisq(sd_lung$chisq, df = 1)
)

# =============================================================================
# Ovarian Dataset
# =============================================================================
cat("Processing ovarian dataset...\n")
data(ovarian)

results$ovarian <- list(
  data = list(
    time = ovarian$futime,
    status = ovarian$fustat,
    rx = ovarian$rx,
    age = ovarian$age
  )
)

# Log-rank test
sd_ovarian <- survdiff(Surv(futime, fustat) ~ rx, data = ovarian)
results$ovarian$logrank <- list(
  chisq = sd_ovarian$chisq,
  p_value = 1 - pchisq(sd_ovarian$chisq, df = 1),
  observed = sd_ovarian$obs,
  expected = sd_ovarian$exp
)

# Kaplan-Meier
km_ovarian <- survfit(Surv(futime, fustat) ~ 1, data = ovarian)
km_ov_summary <- summary(km_ovarian)
results$ovarian$km <- list(
  time = km_ov_summary$time,
  survival = km_ov_summary$surv,
  n_risk = km_ov_summary$n.risk,
  n_event = km_ov_summary$n.event
)

# Cox PH
cox_ovarian <- coxph(Surv(futime, fustat) ~ rx + age, data = ovarian)
results$ovarian$coxph <- list(
  coefficients = as.vector(coef(cox_ovarian)),
  se = summary(cox_ovarian)$coefficients[, "se(coef)"],
  hazard_ratio = exp(coef(cox_ovarian)),
  loglik = cox_ovarian$loglik
)

# =============================================================================
# Veteran Dataset (subset)
# =============================================================================
cat("Processing veteran dataset...\n")
data(veteran)
veteran_subset <- veteran[1:20, ]

results$veteran <- list(
  data = list(
    time = veteran_subset$time,
    status = veteran_subset$status,
    trt = veteran_subset$trt,
    age = veteran_subset$age
  )
)

# Kaplan-Meier
km_veteran <- survfit(Surv(time, status) ~ 1, data = veteran_subset)
km_vet_summary <- summary(km_veteran)
results$veteran$km <- list(
  time = km_vet_summary$time,
  survival = km_vet_summary$surv,
  n_risk = km_vet_summary$n.risk,
  n_event = km_vet_summary$n.event,
  std_err = km_vet_summary$std.err
)

# Cox PH
cox_veteran <- coxph(Surv(time, status) ~ trt + age, data = veteran_subset)
results$veteran$coxph <- list(
  coefficients = as.vector(coef(cox_veteran)),
  hazard_ratio = exp(coef(cox_veteran)),
  loglik = cox_veteran$loglik
)

# =============================================================================
# Edge Cases
# =============================================================================
cat("Processing edge cases...\n")

# Tied event times
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

# All events at same time
same_time_data <- data.frame(
  time = c(5, 5, 5, 5, 5),
  status = c(1, 1, 1, 1, 1)
)
km_same <- survfit(Surv(time, status) ~ 1, data = same_time_data)
results$edge_cases$all_same_time <- list(
  time = km_same$time,
  survival = km_same$surv,
  n_risk = km_same$n.risk,
  n_event = km_same$n.event
)

# Simple 5-observation test for Nelson-Aalen
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

# With censoring
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

# Identical groups (for log-rank test)
identical_data <- data.frame(
  time = c(1, 2, 3, 1, 2, 3),
  status = c(1, 1, 1, 1, 1, 1),
  group = c(0, 0, 0, 1, 1, 1)
)
sd_identical <- survdiff(Surv(time, status) ~ group, data = identical_data)
results$edge_cases$identical_groups_logrank <- list(
  chisq = sd_identical$chisq,
  p_value = 1 - pchisq(sd_identical$chisq, df = 1)
)

# =============================================================================
# Sample Size / Power Calculations
# =============================================================================
cat("Processing sample size calculations...\n")

# Using Schoenfeld formula approximation
# n_events = (z_alpha + z_beta)^2 / (log(HR)^2 * p1 * p2)
calc_sample_size <- function(hr, power, alpha, ratio = 1) {
  z_alpha <- qnorm(1 - alpha/2)
  z_beta <- qnorm(power)
  p1 <- 1 / (1 + ratio)
  p2 <- ratio / (1 + ratio)
  n_events <- (z_alpha + z_beta)^2 / (log(hr)^2 * p1 * p2)
  return(ceiling(n_events))
}

results$sample_size <- list(
  hr_0.5_power_0.8 = calc_sample_size(0.5, 0.8, 0.05),
  hr_0.6_power_0.8 = calc_sample_size(0.6, 0.8, 0.05),
  hr_0.7_power_0.8 = calc_sample_size(0.7, 0.8, 0.05),
  hr_0.6_power_0.9 = calc_sample_size(0.6, 0.9, 0.05)
)

# =============================================================================
# RMST Calculations (manual since survRM2 may not be installed)
# =============================================================================
cat("Processing RMST calculations...\n")

# Calculate RMST from Kaplan-Meier curve
calc_rmst <- function(km_fit, tau) {
  times <- c(0, km_fit$time[km_fit$time <= tau])
  surv <- c(1, km_fit$surv[km_fit$time <= tau])

  if (max(km_fit$time) < tau) {
    times <- c(times, tau)
    surv <- c(surv, surv[length(surv)])
  } else {
    times <- c(times, tau)
    # Interpolate survival at tau
    idx <- which(km_fit$time >= tau)[1]
    if (idx > 1) {
      surv <- c(surv, km_fit$surv[idx])
    } else {
      surv <- c(surv, km_fit$surv[1])
    }
  }

  # Calculate area under curve (trapezoidal rule)
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

# =============================================================================
# Concordance
# =============================================================================
cat("Processing concordance calculations...\n")

conc_aml <- concordance(cox_breslow)
results$concordance <- list(
  aml_coxph = list(
    concordance = conc_aml$concordance,
    se = sqrt(conc_aml$var),
    n_concordant = conc_aml$count["concordant"],
    n_discordant = conc_aml$count["discordant"],
    n_tied_risk = conc_aml$count["tied.risk"],
    n_tied_time = conc_aml$count["tied.time"]
  )
)

# =============================================================================
# Write JSON output
# =============================================================================
output_file <- "r_expected_values.json"
cat("\nWriting results to", output_file, "...\n")

json_output <- toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 10)
writeLines(json_output, output_file)

cat("Done! Generated", output_file, "\n")
cat("Total test cases:", length(unlist(results, recursive = FALSE)), "\n")
