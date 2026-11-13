#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
check <- "--check" %in% args
args <- args[args != "--check"]
if (length(args) > 1L || any(startsWith(args, "--"))) {
  stop("usage: generate_interaction_reference.R [--check] [reference.json]", call. = FALSE)
}
output <- if (length(args)) args[[1L]] else "python/tests/fixtures/interaction_reference.json"

inputs <- list(
  ordinary = list(
    a = factor(c("b", "a", "b", NA, "a", "b", "a"), levels = c("b", "a", "unused")),
    b = factor(c("y", "x", "x", "y", NA, "y", "x"), levels = c("y", "x", "unused")),
    c = factor(c("z", "w", "w", "z", "z", "w", "z"), levels = c("w", "z", "unused"))
  ),
  collisions = list(
    a = factor(c("A", "B", "A", "B", "A", "B"), levels = c("B", "A")),
    b = factor(c("p", "p.q", "p.q", "p", "p.q", NA), levels = c("p", "p.q")),
    c = factor(c("r", "r", "q.r", "q.r", "r", "r"), levels = c("q.r", "r"))
  ),
  no_separator = list(
    a = factor(c("a", "ab", "a", "ab"), levels = c("ab", "a")),
    b = factor(c("bc", "c", "c", "bc"), levels = c("bc", "c")),
    c = factor(c("d", "d", "cd", "d"), levels = c("d", "cd"))
  ),
  all_missing = list(
    a = factor(c(NA, NA, NA), levels = c("b", "a")),
    b = factor(c("x", NA, "y"), levels = c("y", "x")),
    c = factor(c("c", "c", "d"), levels = c("c", "d"))
  ),
  empty = list(
    a = factor(character(), levels = c("b", "a")),
    b = factor(character(), levels = c("y", "x")),
    c = factor(character(), levels = c("c", "d"))
  ),
  single = list(a = factor(c("b", "a", NA, "b"), levels = c("b", "a", "unused")))
)

cases <- list()
for (name in names(inputs)) for (drop in c(FALSE, TRUE)) {
  for (lex in c(FALSE, TRUE)) for (sep in c(".", "")) {
    input <- inputs[[name]]
    result <- do.call(interaction, c(input, list(drop = drop, lex.order = lex, sep = sep)))
    cases[[length(cases) + 1L]] <- list(
      name = paste(name, drop, lex, if (sep == "") "empty" else "dot", sep = "/"),
      input = lapply(input, function(x) list(values = I(as.character(x)), levels = I(levels(x)))),
      drop = drop, lex_order = lex, sep = sep,
      values = I(as.character(result)), levels = I(levels(result))
    )
  }
}

reference <- list(r_version = R.version.string, cases = cases)
if (check) {
  if (!file.exists(output)) stop("interaction reference does not exist: ", output, call. = FALSE)
  actual <- read_json(output, simplifyVector = FALSE)
  expected <- fromJSON(toJSON(reference, auto_unbox = TRUE, null = "null", na = "null"),
                       simplifyVector = FALSE)
  # R versions record provenance; only the generated cases determine parity.
  if (length(actual$cases) != length(expected$cases)) {
    stop(sprintf("interaction reference %s has %d cases; stock R generated %d",
                 output, length(actual$cases), length(expected$cases)), call. = FALSE)
  }
  for (i in seq_along(expected$cases)) {
    if (!identical(actual$cases[[i]], expected$cases[[i]])) {
      stop(sprintf("interaction reference %s differs from stock R in case %d (%s)",
                   output, i, expected$cases[[i]]$name), call. = FALSE)
    }
  }
  cat(length(cases), "interaction cases match stock R\n")
} else {
  write_json(reference, output, pretty = TRUE, auto_unbox = TRUE, null = "null", na = "null")
}
