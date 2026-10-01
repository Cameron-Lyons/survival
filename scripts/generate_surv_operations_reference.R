#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/surv_operations_reference.json"
responses <- list(
  right = Surv(c(1, 2, NA), c(1, 0, 1)),
  left = Surv(c(1, 2, 3), c(1, 0, 1), type = "left"),
  counting = Surv(c(0, 1, 0), c(1, 3, 2), c(1, 0, 1)),
  interval = Surv(c(1, 2, 3), c(2, 4, 5), c(1, 3, 0), type = "interval"),
  multistate = Surv(c(1, 2, 3), factor(c("a", "censor", "b"), levels = c("censor", "a", "b", "unused"))),
  timeline = Surv2(c(1, 2, NA), c(1, 0, 1)),
  timeline_multistate = Surv2(c(1, 2, 3), factor(c("a", "censor", "b"), levels = c("censor", "a", "b")), repeated = "first"),
  empty = Surv(1, 1)[FALSE],
  empty_timeline = Surv2(1, 1)[FALSE]
)
operations <- c(
  add = "x + 1", radd = "1 + x", sub = "x - 1", rsub = "1 - x",
  mul = "x * 2", rmul = "2 * x", div = "x / 2", rdiv = "2 / x",
  floor_div = "x %/% 2", mod = "x %% 2", power = "x ^ 2", rpower = "2 ^ x",
  eq = "x == x", ne = "x != x", lt = "x < 2", le = "x <= 2", gt = "x > 2", ge = "x >= 2",
  logical_and = "x & TRUE", logical_or = "x | FALSE", invert = "!x",
  negative = "-x", positive = "+x", abs = "abs(x)", sqrt = "sqrt(x)", log = "log(x)",
  round = "round(x)", floor = "floor(x)", ceil = "ceiling(x)", trunc = "trunc(x)",
  cumsum = "cumsum(x)", cumprod = "cumprod(x)", sum = "sum(x)", prod = "prod(x)",
  min = "min(x)", max = "max(x)", all = "all(x)", any = "any(x)"
)
cases <- lapply(names(responses), function(name) {
  x <- responses[[name]]
  errors <- lapply(operations, function(code) tryCatch({
    eval(parse(text = code)); stop("operation unexpectedly succeeded")
  }, error = conditionMessage))
  stopifnot(all(unlist(errors) == "Invalid operation on a survival time"))
  matrix <- as.matrix(x)
  list(name = name, class = class(x)[[1L]], type = attr(x, "type"),
       states = I(if (is.null(attr(x, "states"))) character() else attr(x, "states")), clabel = attr(x, "clabel"), repeated = attr(x, "repeated"),
       matrix = lapply(seq_len(nrow(matrix)), function(i) I(as.numeric(matrix[i, ]))), errors = errors)
})
write_json(list(r_version = R.version.string, survival_version = as.character(packageVersion("survival")),
                operations = as.list(operations), cases = cases), output,
           auto_unbox = TRUE, pretty = TRUE, digits = 17, null = "null", na = "null")
