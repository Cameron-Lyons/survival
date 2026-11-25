.aft_matrix_input_setup <- function() {
  data <- survival::ovarian
  data$g <- factor(rep(c("b", "a", "c"), length.out = nrow(data)), levels = c("c", "b", "a"))
  data$h <- factor(rep(c("low", "high"), length.out = nrow(data)))
  data$flag <- data$rx == 1
  data$id <- seq_len(nrow(data)) %% 8L
  data$off <- seq_len(nrow(data)) / 10
  list(data = data, formulas = c(
    "age + strata(g)", "strata(g) + age + flag", "strata(g) + age - 1",
    "strata(g) + flag - 1", "strata(g) + 1", "age + strata(g, h)",
    "strata(g) + age + strata(h)", "age + strata(g) + g",
    "age + strata(g) + age:g", "age + strata(g) + flag:g",
    "age + strata(g):flag", "age + strata(g):h",
    "age + strata(g) + cluster(id)", "age + strata(g) + offset(off)",
    "age + strata(g) + offset(log(off))", "log(age) + strata(g)",
    "ridge(age, theta = 2) + strata(g)", "age + strata(g, shortlabel = FALSE)"
  ), inputs = c(
    "stored", "complete", "missing_g", "all_missing_g", "missing_h", "absent_g",
    "absent_h", "absent_groups", "unknown_g", "numeric_g", "missing_age",
    "missing_flag", "missing_offset", "invalid_offset", "missing_cluster",
    "absent_cluster", "missing_response", "absent_response", "empty", "single",
    "zero_columns"
  ))
}

.aft_matrix_input_data <- function(setup, variant, named) {
  data <- setup$data
  if (named) row.names(data) <- paste0("病人 / ", seq_len(nrow(data)))
  if (variant == "missing") {
    data$g[c(2, 7)] <- NA; data$age[5] <- NA_real_; data$off[9] <- NA_real_
  }
  data
}

.aft_matrix_input_fit <- function(rhs, data, variant, cached, namespace) {
  args <- list(as.formula(paste("Surv(futime,fustat) ~", rhs)), data,
               x = cached, model = TRUE, na.action = "na.exclude")
  if (variant == "subset") args$subset <- c(7L,3L,7L,1L,5L,2L,10L,8L,4L,11L,6L,9L)
  do.call(get("survreg", asNamespace(namespace)), args)
}

.aft_matrix_input_newdata <- function(data, input) {
  if (input == "stored") return(NULL)
  nd <- data[c(7L,3L,7L,1L), ]
  if (input == "missing_g") nd$g[2L] <- NA
  if (input == "all_missing_g") nd$g[] <- NA
  if (input == "missing_h") nd$h[2L] <- NA
  if (input %in% c("absent_g", "absent_groups")) nd$g <- NULL
  if (input %in% c("absent_h", "absent_groups")) nd$h <- NULL
  if (input == "unknown_g") nd$g <- factor(rep("unseen", nrow(nd)))
  if (input == "numeric_g") nd$g <- seq_len(nrow(nd))
  if (input == "missing_age") nd$age[2L] <- NA_real_
  if (input == "missing_flag") nd$flag[2L] <- NA
  if (input == "missing_offset") nd$off[2L] <- NA_real_
  if (input == "invalid_offset") nd$off[2L] <- -1
  if (input == "missing_cluster") nd$id[2L] <- NA_integer_
  if (input == "absent_cluster") nd$id <- NULL
  if (input == "missing_response") nd$futime[] <- NA_real_
  if (input == "absent_response") nd[c("futime", "fustat")] <- NULL
  if (input == "empty") nd <- nd[FALSE, ]
  if (input == "single") nd <- nd[1L,,drop=FALSE]
  if (input == "zero_columns") nd <- nd[, FALSE, drop=FALSE]
  nd
}
