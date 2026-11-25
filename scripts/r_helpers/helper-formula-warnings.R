.formula_warning_setup <- function() {
  setup <- .matrix_na_setup()
  setup$formulas <- c("age + log(z)", "age + sqrt(z)", "age + log(z) + sqrt(w)",
    "age + I(log(z) + sqrt(w))", "age + log(sqrt(z))", "age + offset(log(z))",
    "age + log(z) + offset(off)", "age + sqrt(z) - sqrt(z)", "age + log(z):flag",
    "age + I(z^0 * w)", "age + I(z/z)", "age + I(exp(z))", "age + I(flag)",
    "age + strata(g) + log(z)")
  setup$inputs <- c("complete", "domain", "overlap_covariate", "overlap_response", "overlap_offset",
    "overlap_strata", "source_na", "source_nan", "zero", "overflow_omitted", "all_omitted",
    "empty", "single_omitted")
  setup
}

.formula_warning_newdata <- function(data, input) {
  nd <- data[c(7,3,7,1), ]
  if (input %in% c("domain", "overlap_covariate", "overlap_response", "overlap_offset", "overlap_strata")) {
    nd$z[2] <- -1; nd$w[2] <- -4
    if (input == "overlap_covariate") nd$age[2] <- NA_real_
    if (input == "overlap_response") nd$fustat[2] <- NA_real_
    if (input == "overlap_offset") nd$off[2] <- NA_real_
    if (input == "overlap_strata") nd$g[2] <- NA
  }
  if (input == "source_na") nd$z[2] <- NA_real_
  if (input == "source_nan") nd$z[2] <- NaN
  if (input == "zero") nd$z[2] <- 0
  if (input == "overflow_omitted") {nd$z[2] <- 1000; nd$age[2] <- NA_real_}
  if (input == "all_omitted") {nd$age[] <- NA_real_; nd$z[] <- -1; nd$w[] <- -4}
  if (input == "empty") nd <- nd[FALSE, ]
  if (input == "single_omitted") {nd <- nd[2,,drop=FALSE];nd$age[] <- NA_real_;nd$z[] <- -1;nd$w[] <- -4}
  nd
}

.formula_warning_training <- function(data, variant) {
  data$z[2] <- -1
  args <- list(x=TRUE,model=TRUE)
  if (variant == "covariate") data$age[2] <- NA_real_
  if (variant == "weights") {args$weights<-rep(1,nrow(data));args$weights[2]<-NA_real_}
  if (variant == "subset") args$subset <- c(7,3,7,1,setdiff(seq_len(nrow(data)),c(1,2,3,7)))
  list(data=data,args=args)
}

.formula_warning_messages <- function(messages) {
  sub(" in (log|sqrt)\\(.*\\)$", "", messages)
}
