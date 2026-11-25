.matrix_na_setup <- function() {
  setup <- .aft_matrix_input_setup()
  setup$data$z <- setup$data$age / 10
  setup$data$w <- setup$data$rx
  setup$formulas <- c("age", "age + flag", "age:flag", "age + g", "age + age:g",
    "age + strata(g)", "strata(g) + age + flag", "age + strata(g,h)",
    "age + strata(g):flag", "age + offset(off)", "age + offset(log(off))",
    "log(z) + age", "I(z/w) + age", "age + sqrt(z) - sqrt(z)",
    "age + cluster(id)", "ridge(age, theta = 2)", "ridge(age, rx, theta = 2)", "1")
  setup$inputs <- c("stored", "complete", "missing_numeric", "nan_numeric", "missing_factor",
    "missing_logical", "missing_strata", "all_missing", "missing_offset",
    "invalid_offset", "invalid_numeric", "missing_unused", "invalid_unused", "empty", "single")
  setup
}

.matrix_na_newdata <- function(data, input) {
  if (input == "stored") return(NULL)
  nd <- data[c(7L,3L,7L,1L),]
  if (input == "missing_numeric") nd$age[2L] <- NA_real_
  if (input == "nan_numeric") nd$age[2L] <- NaN
  if (input == "missing_factor") nd$g[2L] <- NA
  if (input == "missing_logical") nd$flag[2L] <- NA
  if (input == "missing_strata") {nd$g[2L] <- NA;nd$h[3L] <- NA}
  if (input == "all_missing") {nd$age[] <- NA_real_;nd$flag[] <- NA;nd$g[] <- NA;nd$h[] <- NA}
  if (input == "missing_offset") nd$off[2L] <- NA_real_
  if (input == "invalid_offset") nd$off[2L] <- -1
  if (input == "invalid_numeric") {nd$z[2L] <- -1;nd$w[3L] <- 0;nd$z[3L] <- 0}
  if (input == "missing_unused") nd$z[2L] <- NA_real_
  if (input == "invalid_unused") nd$z[2L] <- -1
  if (input == "empty") nd <- nd[FALSE,]
  if (input == "single") {nd <- nd[2L,,drop=FALSE];nd$age[] <- NA_real_;nd$flag[] <- NA}
  nd
}

.matrix_na_encode <- function(value) {
  result <- .logical_matrix_encode(value)
  if (!is.matrix(value)) return(result)
  result$na <- I(which(is.na(value) & !is.nan(value)))
  result$nan <- I(which(is.nan(value)))
  result$positive_infinity <- I(which(value == Inf))
  result$negative_infinity <- I(which(value == -Inf))
  group <- attr(value,"strata")
  result$strata <- if(is.null(group)) NULL else list(labels=I(as.character(group)),levels=I(levels(group)))
  result
}
