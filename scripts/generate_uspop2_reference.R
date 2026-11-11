#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly=TRUE)
check <- "--check" %in% args
args <- args[args != "--check"]
reference_path <- if (length(args)) args[[1]] else "python/tests/fixtures/uspop2_reference.json"
data_path <- if (length(args) > 1L) args[[2]] else "src/population/data/uspop2.tsv"

population <- survival::uspop2
stopifnot(typeof(population) == "integer", identical(dim(population), c(101L, 2L, 21L)))
stopifnot(!anyNA(population), is.null(names(dimnames(population))))

# Reproduce the standardization example from R's uspop2 help page.
us50 <- population[51:101,, "2000"]
age <- as.integer(dimnames(us50)[[1]])
smat <- model.matrix(~ factor(floor(age/5)) - 1)
totals <- t(smat) %*% us50
starts <- seq(50, 95, by=5)
group_labels <- c(paste(starts, starts+4, sep="-"), "100+")

reference <- list(
    metadata=list(
        generator="scripts/generate_uspop2_reference.R",
        r_version=as.character(getRversion()),
        survival_version=as.character(packageVersion("survival")),
        source="US Census Bureau NP2008_D1 projections, distributed as survival::uspop2"
    ),
    dims=dim(population),
    dimnames=unname(dimnames(population)),
    counts=as.integer(population),
    annual_totals=unname(apply(population, 3, sum)),
    age50_groups_2000=list(labels=group_labels, counts=unname(totals)),
    namespace_exports=sort(getNamespaceExports("survival"), method="radix"),
    data_items=sort(sub(" .*", "", data(package="survival")$results[, "Item"]), method="radix")
)
text <- c(
    paste(c("age", dimnames(population)[[1]]), collapse="\t"),
    paste(c("sex", dimnames(population)[[2]]), collapse="\t"),
    paste(c("year", dimnames(population)[[3]]), collapse="\t"),
    as.character(as.integer(population))
)

if (check) {
    actual <- read_json(reference_path, simplifyVector=FALSE)
    expected <- fromJSON(toJSON(reference, auto_unbox=TRUE, digits=17), simplifyVector=FALSE)
    # R versions are provenance, not part of the immutable census projection data.
    actual$metadata <- NULL
    expected$metadata <- NULL
    if (!identical(actual, expected)) stop("uspop2 reference differs from stock R")
    if (!identical(readLines(data_path), text)) stop("embedded uspop2 values/axes differ from stock R")
    cat("Every uspop2 count, axis, aggregate and package inventory matches stock R\n")
} else {
    dir.create(dirname(reference_path), recursive=TRUE, showWarnings=FALSE)
    dir.create(dirname(data_path), recursive=TRUE, showWarnings=FALSE)
    write_json(reference, reference_path, auto_unbox=TRUE, digits=17, pretty=TRUE)
    writeLines(text, data_path)
    cat(length(population), "population counts and all axis labels written\n")
}
