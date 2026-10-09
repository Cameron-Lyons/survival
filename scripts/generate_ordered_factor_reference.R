#!/usr/bin/env Rscript
# Regenerate stock base-R ordered contrasts, including numerical rank loss.
args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1L]] else
    "python/tests/fixtures/ordered_factor_reference.json"
cases <- lapply(2:24, function(n) {
    scores <- seq_len(n) - (n + 1) / 2
    decomposition <- qr(outer(scores, 0:(n - 1), `^`))
    list(n = n, rank = decomposition$rank, pivot = decomposition$pivot,
         basis = unname(contr.poly(n)), columns = as.list(colnames(contr.poly(n))))
})
reference <- list(r_version = as.character(getRversion()), cases = cases)
jsonlite::write_json(reference, output, auto_unbox = TRUE, digits = 17,
                     pretty = TRUE, na = "null")
