.logical_matrix_setup <- function() {
  data <- survival::ovarian
  data$flag <- data$rx == 1
  data$other <- seq_len(nrow(data)) %% 3L == 0L
  data$g <- factor(rep(c("b", "a", "c"), length.out = nrow(data)), levels = c("c", "b", "a"))
  data$id <- seq_len(nrow(data)) %% 8L
  list(data = data, formulas = c(
    "age + flag", "age:flag", "flag + age:flag", "flag * other", "flag:other",
    "flag:factor(rx)", "I(flag) + age", "identity(flag) + age",
    "as.numeric(flag) + age", "sqrt(flag) + age", "I(age > 60) + age",
    "I(age > 200) + age", "factor(flag) + age", "factor(rx) * flag",
    "strata(rx) + age * flag", "strata(rx):flag + age",
    "age + cluster(id) + flag", "age + offset(rx) + flag",
    "age + ridge(rx, theta = 2) + flag", "age + g:flag", "1", "flag - 1"
  ))
}

.logical_matrix_data <- function(setup, variant, named) {
  data <- setup$data
  if (named) row.names(data) <- paste0("病人 / ", seq_len(nrow(data)))
  if (variant == "false") data$flag[] <- FALSE
  if (variant == "true") data$flag[] <- TRUE
  if (variant == "missing") { data$flag[c(2, 7)] <- NA; data$age[5] <- NA_real_ }
  data
}

.logical_matrix_fit <- function(kind, rhs, data, variant, namespace) {
  args <- list(as.formula(paste("Surv(futime,fustat) ~", rhs)), data,
               x = TRUE, model = TRUE, na.action = "na.exclude")
  if (variant == "subset") args$subset <- c(7L,3L,7L,1L,5L,2L,10L,8L,4L,11L,6L,9L)
  do.call(get(kind, asNamespace(namespace)), args)
}

.logical_matrix_newdata <- function(data, input) {
  if (input == "stored") return(NULL)
  nd <- data[c(7L,3L,7L,1L), ]
  if (input == "partial") { nd$flag[2L] <- NA; nd$age[4L] <- NA_real_ }
  if (input == "all_missing") nd$flag[] <- NA
  if (input == "empty") nd <- nd[FALSE, ]
  if (input == "single") nd <- nd[1L,,drop=FALSE]
  nd
}

.logical_matrix_capture <- function(fun) {
  messages <- character()
  value <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = messages)
}

.logical_matrix_encode <- function(value) {
  if (is.null(value)) return(NULL)
  if (is.list(value)) return(lapply(value, .logical_matrix_encode))
  if (is.character(value)) return(I(value))
  list(values = I(as.numeric(value)), dim = if (is.null(dim(value))) NULL else I(dim(value)),
       rows = if (is.null(rownames(value))) NULL else I(rownames(value)),
       columns = if (is.null(colnames(value))) NULL else I(colnames(value)),
       names = if (is.null(names(value))) NULL else I(names(value)),
       assign = if (is.null(attr(value, "assign"))) NULL else I(attr(value, "assign")),
       contrasts = .logical_matrix_encode(attr(value, "contrasts")))
}

.logical_matrix_compare <- function(actual, expected, info) {
  expect_identical(dim(actual), dim(expected), info = info)
  expect_identical(dimnames(actual), dimnames(expected), info = info)
  # Stock's strata term-number shift sometimes promotes assign to double;
  # the bridge retains the established integer term-index convention.
  expect_identical(attr(actual, "assign"), as.integer(attr(expected, "assign")), info = info)
  expect_identical(attr(actual, "contrasts"), attr(expected, "contrasts"), info = info)
  expect_equal(attr(actual, "strata"), attr(expected, "strata"), info = info)
  expect_equal(as.numeric(actual), as.numeric(expected), tolerance = 3e-7, info = info)
}
