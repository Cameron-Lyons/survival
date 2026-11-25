# Surv responses saved by survivalr 2.1.0 or earlier are Python proxies of class
# "survival_py_surv"; Surv() now returns survival's own matrix, so these methods exist only
# to keep such saved objects working. Each converts the proxy to survival's matrix.

survfit.survival_py_surv <- function(formula, ...) {
  survfit.Surv(.as_native_surv(formula), ...)
}

as.matrix.survival_py_surv <- function(x, ...) {
  as.matrix.Surv(.as_native_surv(x), ...)
}

length.survival_py_surv <- function(x) {
  length.Surv(.as_native_surv(x))
}

names.survival_py_surv <- function(x) {
  names.Surv(.as_native_surv(x))
}

`names<-.survival_py_surv` <- function(x, value) {
  native <- .as_native_surv(x)
  rownames(native) <- value
  native
}

as.logical.survival_py_surv <- function(x, ...) {
  as.logical.Surv(.as_native_surv(x), ...)
}

barplot.survival_py_surv <- function(height, ...) {
  barplot.Surv(.as_native_surv(height), ...)
}

density.survival_py_surv <- function(x, ...) {
  density.Surv(.as_native_surv(x), ...)
}

hist.survival_py_surv <- function(x, ...) {
  hist.Surv(.as_native_surv(x), ...)
}

identify.survival_py_surv <- function(x, ...) {
  identify.Surv(.as_native_surv(x), ...)
}

image.survival_py_surv <- function(x, ...) {
  image.Surv(.as_native_surv(x), ...)
}

lines.survival_py_surv <- function(x, ...) {
  lines.Surv(.as_native_surv(x), ...)
}

pairs.survival_py_surv <- function(x, ...) {
  pairs.Surv(.as_native_surv(x), ...)
}

plot.survival_py_surv <- function(x, ...) {
  plot.Surv(.as_native_surv(x), ...)
}

points.survival_py_surv <- function(x, ...) {
  points.Surv(.as_native_surv(x), ...)
}

text.survival_py_surv <- function(x, ...) {
  text.Surv(.as_native_surv(x), ...)
}

c.survival_py_surv <- function(...) {
  c.Surv(...)
}

rev.survival_py_surv <- function(x) {
  rev.Surv(.as_native_surv(x))
}

rep.survival_py_surv <- function(x, ...) {
  rep.Surv(.as_native_surv(x), ...)
}

rep.int.survival_py_surv <- function(x, ...) {
  rep.int.Surv(.as_native_surv(x), ...)
}

rep_len.survival_py_surv <- function(x, ...) {
  rep_len.Surv(.as_native_surv(x), ...)
}

t.survival_py_surv <- function(x) {
  t.Surv(.as_native_surv(x))
}

head.survival_py_surv <- function(x, ...) {
  head.Surv(.as_native_surv(x), ...)
}

tail.survival_py_surv <- function(x, ...) {
  tail.Surv(.as_native_surv(x), ...)
}

anyDuplicated.survival_py_surv <- function(x, ...) {
  anyDuplicated.Surv(.as_native_surv(x), ...)
}

duplicated.survival_py_surv <- function(x, ...) {
  duplicated.Surv(.as_native_surv(x), ...)
}

unique.survival_py_surv <- function(x, ...) {
  unique.Surv(.as_native_surv(x), ...)
}

levels.survival_py_surv <- function(x) {
  levels.Surv(.as_native_surv(x))
}

Math.survival_py_surv <- function(...) {
  Math.Surv(...)
}

Ops.survival_py_surv <- function(...) {
  Ops.Surv(...)
}

Summary.survival_py_surv <- function(...) {
  Summary.Surv(...)
}

xtfrm.survival_py_surv <- function(x) {
  xtfrm.Surv(.as_native_surv(x))
}

quantile.survival_py_surv <- function(x, probs = c(0.25, 0.5, 0.75),
                                      na.rm = FALSE, ...) {
  quantile.Surv(.as_native_surv(x), probs = probs, na.rm = na.rm, ...)
}

median.survival_py_surv <- function(x, na.rm = FALSE, ...) {
  median.Surv(.as_native_surv(x), na.rm = na.rm, ...)
}

`[.survival_py_surv` <- function(x, i, j, drop = FALSE) {
  native <- .as_native_surv(x)
  if (missing(j)) {
    if (missing(i)) {
      native[]
    } else {
      native[i]
    }
  } else if (missing(i)) {
    if (missing(drop)) native[, j] else native[, j, drop = drop]
  } else {
    if (missing(drop)) native[i, j] else native[i, j, drop = drop]
  }
}

# A name reads the proxy's field, as $ does; a position reads survival's matrix.
`[[.survival_py_surv` <- function(x, i, ...) {
  if (is.character(i)) return(NextMethod())
  .as_native_surv(x)[[i, ...]]
}

format.survival_py_surv <- function(x, ...) {
  as.character(.call_r_api("format_surv", x, ...))
}

as.character.survival_py_surv <- function(x, ...) {
  as.character.Surv(.as_native_surv(x), ...)
}

is.na.survival_py_surv <- function(x) {
  as.logical(.call_r_api("is_na_surv", x))
}

as.data.frame.survival_py_surv <- function(x, row.names = NULL, optional = FALSE, ...) {
  as.data.frame.Surv(.as_native_surv(x), row.names = row.names, optional = optional, ...)
}

summary.survival_py_surv <- function(object, ...) {
  summary(.as_native_surv(object), ...)
}

print.survival_py_surv <- function(x, ...) {
  print(.as_native_surv(x), ...)
  invisible(x)
}
