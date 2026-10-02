# Bulk numeric data-frame conversion

Complete, unclassed double and integer columns in R data frames cross the
Python bridge as one-dimensional NumPy arrays. Previously, the bridge built
an R list and converted each scalar separately. Empty and single-row columns
retain their vector shape; source row labels still travel separately.

The bridge prepares an R array because reticulate converts arrays in bulk and
maps their storage into NumPy, as described in its
[array conversion documentation](https://rstudio.github.io/reticulate/articles/calling_python.html#arrays).
R can duplicate storage while preparing the array. This change removes scalar
transport; subsequent formula designs and public results still materialize.

Columns containing `NA` or `NaN` keep the existing missing-value conversion.
Factors, ordered factors, logical and character columns, dates, other classed
columns, matrix columns and long vectors exceeding R's array extent limit keep
their existing paths. In particular, integer missing values never become
ordinary NumPy integers. Complete numeric columns may contain infinities;
existing downstream validation determines whether a particular formula accepts
them.

## Validation

The R source archive passes 27,579 checks with zero errors, warnings or notes,
adding 1,410 checks over #705. New checks cover integer/double types, empty and
single-row shapes, infinities, row metadata, mixed missing values, declared
factor levels and fallback columns. NumPy views retain their values after the
original R data are mutated, removed and collected, including ALTREP integer
sequences. The views remain read-only. Mixed numeric/factor Cox and AFT fits
match stock coefficients, variance, design values and prediction values,
errors, dimensions and names across interactions, transformations, strata,
repeated subsets and prediction NA actions.

All 18,654 Python tests pass, with 48 skips and 37 expected differences. Pinned
Ruff lint/format, the 47-file type check and generated stubs/manifest checks pass.
Four models saved by #705 restore in a fresh process with identical complete
predictions, errors, names and attributes. Freshly saved models also retain
those outputs across processes. Rust and Python implementation sources, Cargo
inputs and the release extension are unchanged; Rust suites were not rerun.

## Complete-call measurements

The existing `scripts/benchmark_prediction_row_labels.R` accepts an optional
fourth argument, `transport`, to measure data conversion separately. The
complete public calls include conversion, formula design, numerical work,
output materialization and row/column metadata. Fitting, input creation and
explicit GC are excluded. Each model uses 2,000 training rows, 16 covariates
and 20,000 new rows. Calls include standard errors; term calls select three
repeated/reordered columns. Three warmups precede nine samples.

Separate R/Python libraries select #705 (`376c9a3d`) and the current bridge
with the same release extension. Both implementations check entire output
values, errors, dimensions and names against stock R before timing. On an
Intel Core Ultra 5 325 with R 4.5.3, survival 3.8-12, reticulate 1.47.0,
Python 3.14.7 and NumPy 2.4.6, medians and sample ranges are:

| Operation | #705, ms | Bulk conversion, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Data conversion, 17 numeric columns | 27 (27–28) | 1 (1–2) | — |
| Cox linear predictor | 47 (47–49) | 21 (20–22) | 10 (9–14) |
| Cox selected terms | 87 (80–90) | 58 (57–59) | 10 (9–10) |
| AFT scalar quantile | 52 (51–53) | 24 (23–25) | 7 (5–11) |
| AFT selected terms | 86 (84–95) | 56 (56–60) | 897 (874–965) |

Linear predictors and scalar quantiles take less than half the predecessor's
time on this workload; selected terms take about one third less. Cox and AFT
quantile calls remain slower than stock R, while AFT terms remain faster.
These measurements apply to complete numeric data; they do not establish full
package parity or a speedup for every input.

```sh
# Set R_LIBS and PYTHONPATH to each implementation's libraries before its run.
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 predecessor transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 stock transport
```
