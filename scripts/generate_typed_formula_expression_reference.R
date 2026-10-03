#!/usr/bin/env Rscript
# Independent references for typed formula evaluation and fitted designs.
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/typed_formula_expression_reference.json"

capture <- function(fun) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(fun(), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = function(e) list(error = conditionMessage(e)))
  list(value = value, warnings = I(warnings))
}
values <- function(x) lapply(unname(x), function(value) {
  if (is.na(value) && !is.nan(value)) NULL else if (is.nan(value)) "NaN" else
    if (is.numeric(value) && is.infinite(value)) if (value > 0) "Inf" else "-Inf" else value
})
encode <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) return(lapply(x, encode))
  list(values = values(if (is.factor(x)) as.character(x) else x),
       dim = if (is.null(dim(x))) NULL else I(dim(x)),
       rows = if (is.null(rownames(x))) NULL else I(rownames(x)),
       columns = if (is.null(colnames(x))) NULL else I(colnames(x)),
       names = if (is.null(names(x))) NULL else I(names(x)),
       kind = if (is.factor(x)) "factor" else if (is.logical(x)) "logical" else
         if (is.character(x)) "character" else "numeric",
       type = typeof(x), class = I(class(x)), levels = if (is.factor(x)) I(levels(x)) else NULL,
       na = I(as.vector(is.na(x))), nan = I(as.vector(is.nan(x))),
       assign = if (is.null(attr(x, "assign"))) NULL else I(attr(x, "assign")),
       contrasts = encode(attr(x, "contrasts")))
}
is_error <- function(value) is.list(value) && !is.null(value$error)
encode_capture <- function(result) list(
  value = if (is_error(result$value)) result$value else encode(result$value), warnings = result$warnings)
columns <- function(data) lapply(data, function(x) I(unname(if (is.factor(x)) as.character(x) else x)))
truth <- expand.grid(a = c(TRUE, FALSE, NA), b = c(TRUE, FALSE, NA))
truth$x <- c(-2, -1, 0, 1, 2, 3, -3, .5, NA)
truth$z <- c(2, 1, 0, -1, NA, 1.5, .5, 2, -2)
truth$g <- factor(rep(c("b", "a", "c"), 3), levels = c("c", "b", "a", "unused"))
truth$h <- c("1", "-2.5", " 3 ", "broken", "", NA, "Inf", "NaN", "0")
truth$i <- c(1L, -2L, 0L, 2147483647L, -2147483647L, 3L, 4L, 5L, 6L)
expressions <- c(
  "a & b", "a | b", "!a", "!(a & b)", "!(a | b)", "!a & b | a", "!a | b & a",
  "x > 0 & z <= 1 | a", "x > 0 | z <= 1 & a", "!(x > 0 | z <= 1) & !a",
  "!x > 0", "!x^2 > 1", "!x == 0", "(-x)^2", "-x^2", "x^2^2",
  "I(a & b)", "identity(I(a | b))", "I(identity(I(!a)))",
  "as.numeric(a & b)", "as.numeric(I(a | b))", "I(x > 0) + 0", "x * (a | b)",
  "I(g)", "identity(g)", "I(identity(I(g)))", "as.numeric(I(g))",
  "identity(g) == 'a'", "!g == 'a'", "!g", "g & a",
  "identity(h)", "I(identity(I(h)))", "as.numeric(h)", "log(x)", "sqrt(x)",
  "exp(x)", "x / z", "(x < z) < x", "i", "I(i)", "identity(i)", "I(identity(I(i)))",
  "+i", "-i", "i + 1L", "i - 1L", "i * 2L", "i / 2L", "i ^ 2L",
  "i + 1", "i + z", "a + b", "a - b", "a * b", "+a", "-a",
  "as.numeric(i)", "as.numeric(i) + 1L", "I(i + 1L)", "identity(I(i * 2L))",
  "(i + 1L) + (i - 1L)"
)
evaluation <- list()
for (variant in c("truth", "empty", "all_missing")) {
  input <- if (variant == "empty") truth[FALSE, ] else truth
  if (variant == "all_missing") {
    for (name in c("a", "b", "x", "z", "g", "h", "i")) input[[name]][] <- NA
  }
  for (expression in expressions) {
    evaluation[[length(evaluation) + 1L]] <- list(
      name = paste(variant, expression, sep = "/"), variant = variant, expression = expression,
      expected = encode_capture(capture(function() eval(parse(text = expression), input))))
  }
}

literal_evaluation <- list()
for (n in c(0L, 3L)) for (expression in c(
  "NA", "NA_real_", "NA_integer_", "NA_character_", "TRUE", "FALSE", "NaN",
  "I(NA)", "identity(I(NA_real_))", "I(identity(I(NA_character_)))",
  "1L", "I(1L)", "identity(I(1L))", "I(NA_integer_)", "identity(I(NA_integer_))",
  "+TRUE", "-FALSE", "+NA_integer_", "-NA_integer_", "NA + 1L", "NA_integer_ + 1",
  "NA_integer_ * 0L", "FALSE * NA", "NA_integer_ ^ 0L", "1L ^ NA_integer_",
  "1L / 0L", "0L ^ 0L", "-(-2147483647L)", "2147483647L + 1L",
  "-2147483647L - 1L", "2147483647L * 2L")) {
  literal_evaluation[[length(literal_evaluation) + 1L]] <- list(
    name = paste(n, expression, sep = "/"), n = n, expression = expression,
    expected = encode_capture(capture(function() rep(eval(parse(text = expression)), length.out = n))))
}

escape_expressions <- c(
  r"("\a\b\f\n\r\t\v\\\"\'")", r"('\x41')", r"("\x4")", r"("\x414")",
  r"("\101")", r"("\u0041")", r"("\u41")", r"("\u{41}")", r"("\U00000041")",
  r"("\U{41}")", r"("\u00e9")", r"("\xC3\xA9")", r"("\303\251")",
  r"("\U0001F600")", r"("\uD83D\uDE00")", r"("\u{D83D}\u{DE00}")",
  r"("\U0000D83D\U0000DE00")", r"("\u0041\n")", r"(`\x41`)", r"(`\xC3\xA9`)",
  r"(identity(I("\u0041")))", r"("\q")", r"("\x")", r"("\u")", r"("\400")",
  r"("\0")", r"("\x00")", r"("\u0000")", r"("\U00110000")",
  r"("\u0041\x42")", r"("\101\u0041")", r"(`\u0041`)", r"("\xFF")",
  r"("\uD800")", r"("\uDE00")"
)
escape_sources <- list(A = c("alpha", "beta", "gamma"), "é" = c("un", "deux", "trois"))
escape_evaluation <- list()
for (n in c(0L, 3L)) for (expression in escape_expressions) {
  input <- lapply(escape_sources, function(x) x[seq_len(n)])
  result <- capture(function() eval(parse(text = expression), input))
  raw <- result$value
  unsupported <- !is_error(raw) && !all(validUTF8(raw))
  if (!is_error(raw)) {
    x <- if (length(raw) == 1L) rep(raw, length.out = n) else raw
    value <- list(type = typeof(x), strings = if (all(validUTF8(x))) I(unname(x)) else NULL,
                  bytes = lapply(x, function(s) I(as.integer(charToRaw(s)))),
                  raw_bytes = lapply(raw, function(s) I(as.integer(charToRaw(s)))))
  } else value <- raw
  escape_evaluation[[length(escape_evaluation) + 1L]] <- list(
    name = paste(n, expression, sep = "/"), n = n, expression = expression,
    unsupported = unsupported, expected = list(value = value, warnings = result$warnings))
}

index <- seq_len(72)
d <- data.frame(futime = 10 + (index * 37 %% 97) + index / 7,
                fustat = as.integer((index * 17 + 3) %% 11 > 2),
                x = sin(index * .63) + (index %% 5) / 10,
                z = 2 * cos(index * .41))
d$a <- rep(c(TRUE, FALSE, NA), length.out = nrow(d))
d$b <- rep(rep(c(TRUE, FALSE, NA), each = 3), length.out = nrow(d))
d$g <- factor(rep(c("b", "a", "c", "a", "b"), length.out = nrow(d)),
              levels = c("c", "b", "a"))
d$i <- rep(truth$i, length.out = nrow(d))
d$h <- rep(c("A", "é", "\b", "\a", "\v", "😀", "other", "backslash\\"), length.out = nrow(d))
row.names(d) <- paste0("patient / ", seq_len(nrow(d)))
rhs <- c(
  "x + I(a & b)", "x + I(a | b)", "x + I(!a)", "x + I(!(a & b))",
  "x + I(!a & b | a)", "x + I(!a | b & a)",
  "x + I(x > 0 & z <= 1 | a)", "x + I(x > 0 | z <= 1 & a)",
  "x + I(!(x > 0 | z <= 1) & !a)", "x + I(!x > 0)", "x + I(!x^2 > 1)",
  "x + identity(I(a & b))", "x + I(identity(I(!a)))",
  "x + as.numeric(a & b)", "x + I(as.numeric(a | b))",
  "x + I(g)", "x + identity(g)", "x + I(identity(I(g)))",
  "x + as.numeric(I(g))", "x + I(identity(g) == 'a')", "x + I(!g == 'a')",
  "x + offset(a)", "x + offset(a & b)", "x + offset(a | b)", "x + offset(x > 0)",
  "x + offset(as.numeric(a & b))", "x + offset(as.numeric(x > 0))",
  "x + I((-z)^2)", "x + I(-z^2)", "x + I(z^2^2)",
  "x * I(a | b)", "x + I(a & b):g", "x + I(a | b) + offset(x > 0)"
)
overflow_rhs <- c("x + I(i + 1L)", "x + I(i - 1L)", "x + I(i * 2L)")
extra_rhs <- c(overflow_rhs, "x + I(as.numeric(i) + 1L)",
  "x + I(i + 1e1L)", "x + I(i + 10.0L)",
  r"(x + I(h == '\x41'))", r"(x + I(h == "\u00e9"))", r"(x + I(h == '\b'))",
  r"(x + I(h == '\a'))", r"(x + I(h == '\v'))", r"(x + I(h == "\U0001F600"))",
  r"(x + I(h == '\303\251'))", r"(x + I(h == '\xC3\xA9'))")
rhs <- c(rhs, extra_rhs)
subset_rhs <- c("x + I(a & b)", "x + I(a | b)", "x + I(x > 0 | z <= 1 & a)",
                "x + I(g)", "x + I(identity(I(g)))", "x + offset(a & b)",
                "x + I(a | b) + offset(x > 0)")
subset_rows <- c(7L, 3L, 7L, 1L, 5L, 2L, 10L, 8L, 4L, 11L, 6L, 9L, 13L, 16L,
                 14L, 23L, 32L, 41L)
newdata <- list(complete = d[c(7L, 3L, 7L, 1L, 5L, 2L, 10L, 8L, 4L), ], truth = d[1:9, ])
newdata$complete$a <- rep(c(TRUE, FALSE, TRUE), length.out = 9)
newdata$complete$b <- rep(c(FALSE, TRUE, TRUE), length.out = 9)
newdata$all_missing <- newdata$truth
for (name in c("a", "b", "g", "h", "i")) newdata$all_missing[[name]][] <- NA
newdata$empty <- d[FALSE, ]
cases <- list()
for (kind in c("coxph", "survreg")) for (term in rhs) {
  for (variant in if (term %in% extra_rhs) "complete" else c("complete", "truth", "subset")) {
    if (variant == "subset" && !(term %in% subset_rhs)) next
    input <- d
    if (variant == "complete") {
      input$a <- rep(c(TRUE, FALSE), length.out = nrow(input))
      input$b <- rep(c(TRUE, TRUE, FALSE, FALSE), length.out = nrow(input))
    }
    actions <- if (term %in% overflow_rhs) c("na.omit", "na.exclude", "na.pass", "na.fail") else
      if (variant == "complete") "na.exclude" else
      if (variant == "truth") c("na.omit", "na.exclude", "na.pass", "na.fail") else
      c("na.omit", "na.exclude")
    for (action in actions) {
      formula <- paste("Surv(futime, fustat) ~", term)
      parsed <- capture(function() as.formula(formula))
      options <- list(formula = parsed$value, data = input, x = TRUE, y = TRUE,
                      model = TRUE, na.action = get(action))
      if (variant == "subset") options$subset <- subset_rows
      fitted <- capture(function() do.call(get(kind, asNamespace("survival")), options))
      case <- list(name = paste(kind, term, variant, action, sep = "/"), kind = kind,
                   rhs = term, variant = variant, na_action = action,
                   subset = if (variant == "subset") I(subset_rows - 1L) else NULL,
                   formula_parse_warnings = parsed$warnings,
                   fit_warnings = fitted$warnings)
      if (!is.null(fitted$value$error)) case$error <- fitted$value$error else {
        fit <- fitted$value
        case$coefficients <- encode(coef(fit))
        case$variance <- encode(vcov(fit))
        case$scale <- if (kind == "survreg") encode(fit$scale) else NULL
        case$means <- if (kind == "coxph") encode(fit$means) else NULL
        case$na_rows <- I(as.integer(fit$na.action))
        case$frame <- lapply(fit$model[-1L], encode)
        case$frame_rows <- I(row.names(fit$model))
        case$offset <- encode(model.offset(fit$model))
        case$matrix <- encode(model.matrix(fit))
        predict_types <- if (kind == "coxph") c("lp", "risk", "terms") else c("lp", "response", "terms")
        training <- lapply(predict_types, function(type) encode_capture(capture(function() predict(fit, type = type))))
        names(training) <- predict_types
        case$training <- training
        predictions <- list()
        for (name in names(newdata)) {
          nd <- newdata[[name]]
          outputs <- list(matrix = encode_capture(capture(function() model.matrix(fit, nd))))
          for (type in predict_types) {
            for (predict_action in if (name == "truth") c("na.pass", "na.omit", "na.exclude", "na.fail") else "na.pass") {
              raw <- capture(function() predict(fit, newdata = nd, type = type,
                                                 na.action = get(predict_action)))
              result <- list(raw = encode_capture(raw))
              # Stock predict.survreg rejects a zero-row terms model matrix.
              # Keep the failure and independently preserve the fitted term schema.
              if (kind == "survreg" && name == "empty" && type == "terms" && is_error(raw$value)) {
                example <- capture(function() predict(fit, newdata = newdata$complete, type = "terms"))
                result$intended <- encode(example$value[FALSE, , drop = FALSE])
              }
              # Stock predict.survreg drops new-data offsets for lp/response.
              # Preserve its raw output and derive the intended fitted-design prediction.
              if (kind == "survreg" && type != "terms" &&
                  length(attr(terms(fit), "offset")) && !is_error(raw$value)) {
                mf <- capture(function() model.frame(delete.response(terms(fit)), nd,
                                                     na.action = get(predict_action), xlev = fit$xlevels))
                if (!is_error(mf$value)) {
                  offset <- model.offset(mf$value)
                  if (is.null(offset)) offset <- rep(0, nrow(mf$value))
                  if (predict_action == "na.exclude") offset <- napredict(attr(mf$value, "na.action"), offset)
                  correction <- if (type == "lp") raw$value + offset else raw$value * exp(offset)
                  result$intended <- encode(correction)
                }
              }
              outputs[[paste(type, predict_action, sep = "/")]] <- result
            }
          }
          predictions[[name]] <- outputs
        }
        case$newdata <- predictions
      }
      cases[[length(cases) + 1L]] <- case
    }
  }
}
offset_overrides <- list()
for (kind in c("coxph", "survreg")) {
  input <- d
  input$a <- rep(c(TRUE, FALSE), length.out = nrow(input))
  fit <- get(kind, asNamespace("survival"))(Surv(futime, fustat) ~ x + offset(a), input)
  for (evaluated in c(FALSE, TRUE)) for (numeric in c(FALSE, TRUE)) {
    nd <- newdata$complete
    if (numeric) nd$a <- as.numeric(nd$a)
    if (evaluated) {
      nd <- model.frame(delete.response(terms(fit)), newdata$complete, na.action = na.pass)
      if (numeric) nd[["offset(a)"]] <- as.numeric(nd[["offset(a)"]])
    }
    offset_overrides[[length(offset_overrides) + 1L]] <- list(
      name = paste(kind, if (evaluated) "evaluated" else "raw", if (numeric) "numeric" else "logical", sep = "/"),
      kind = kind, evaluated = evaluated, numeric = numeric,
      expected = encode_capture(capture(function() model.matrix(fit, nd))))
  }
}
training_failures <- list()
for (kind in c("coxph", "survreg")) for (variant in c("empty", "all_missing")) {
  input <- if (variant == "empty") d[FALSE, ] else d
  if (variant == "all_missing") for (name in c("a", "b")) input[[name]][] <- NA
  for (action in c("na.omit", "na.exclude")) {
    training_failures[[length(training_failures) + 1L]] <- list(
      name = paste(kind, variant, action, sep = "/"), kind = kind, variant = variant, na_action = action,
      expected = encode_capture(capture(function() get(kind, asNamespace("survival"))(
        Surv(futime, fustat) ~ x + I(a & b), input, na.action = get(action)))))
  }
}
reference <- list(metadata = list(R = as.character(getRversion()), survival = as.character(packageVersion("survival")),
                  provenance = "Unmodified stock R evaluation, model frames, model matrices, fits, and predictions; raw failures retained."),
                  truth = columns(truth), truth_levels = I(levels(truth$g)), evaluation = evaluation,
                  literal_evaluation = literal_evaluation,
                  escape_sources = lapply(escape_sources, I), escape_evaluation = escape_evaluation,
                  offset_overrides = offset_overrides,
                  training_failures = training_failures,
                  data = columns(d), levels = I(levels(d$g)), row_names = I(row.names(d)),
                  newdata = lapply(newdata, function(nd) list(data = columns(nd), row_names = I(row.names(nd)))),
                  cases = cases)
serialize <- function(value) toJSON(value, auto_unbox = TRUE, digits = 17, na = "null", null = "null")
header <- serialize(reference[names(reference) != "cases"])
writeLines(c(substring(header, 1L, nchar(header) - 1L), ',"cases":[',
             vapply(seq_along(cases), function(i) paste0(serialize(cases[[i]]),
                     if (i < length(cases)) "," else ""), ""), "]}"), output, useBytes = TRUE)
cat(length(evaluation), "evaluation cases and", length(cases), "fitted model cases written to", output, "\n")
