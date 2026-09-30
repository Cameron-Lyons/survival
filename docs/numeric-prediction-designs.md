# Numeric prediction designs

Cox new-data predictions and curves, and formula-based AFT predictions, build
an owned, contiguous float64 NumPy design before calling the existing native
matrix interface. Plain numeric NumPy and pandas columns stay arrays until
copied into that matrix. This removes Python scalar conversion and row-list
assembly from numeric prediction designs.

Interactions multiply float64 columns in fitted order, retaining R's column
order and the previous left-to-right scalar multiplication order. Signed zero,
NaNs, overflow and underflow keep their previous behavior without introducing
NumPy arithmetic warnings. Input arrays, including read-only and strided views,
are never mutated or shared with the resulting design.

Factors, custom contrasts, formula transforms and penalty bases retain their
existing evaluators. Cached numeric and matrix variables take precedence over
source columns. Masked, object, string and nullable columns retain scalar
coercion. Empty numeric designs retain both dimensions, including matrix terms
with known widths. Public Python model matrices and prediction results keep
their list interfaces. Matrix-based AFT fits retain their existing input path.
Rust APIs and the R bridge are unchanged.

## Validation

All 123 independent R formula-algebra references run against row and array
designs with list, NumPy and pandas numeric columns. The fixture regenerates
byte-for-byte. Another 57 checks cover numeric types, non-native byte order,
strides, read-only sources, ownership, empty dimensions, cached variables,
nullable/object/string coercion, IEEE arithmetic and public list results.

The full Python suite passes 22,026 tests, with 48 skips and 37 documented
expected differences; 672 additional cases. Pinned Ruff 0.16.9 lint/format,
47-file Mypy and generated stubs/manifest checks pass. The R source archive
matches all 44 R source/test files and passes 28,083 checks with zero errors,
warnings or notes. Four #707 models and four freshly saved models restore in
separate R processes with identical complete predictions, errors, names and
attributes. Rust and the release extension are unchanged from #707; its Rust
validation remains the latest native validation.

## Complete-call measurements

`scripts/benchmark_prediction_row_labels.R` compares #707 (`954df841`) with
this implementation and stock R using separate Python/R libraries and the
same unchanged release extension built with `extension-module,ml`. Each model
fits 2,000 rows with 16 numeric covariates and predicts 20,000 new rows with
standard errors. Term calls select `x9`, `x1`, `x9`. Each workload has three
warmups and nine samples. Every run checks complete values, errors, dimensions
and names against stock R before timing.

Measurements include R/Python transport, formula design, native calculation,
output materialization and row/column metadata. Fitting, input creation and
explicit GC are excluded. All six runs execute sequentially after validation
finishes. On an Intel Core Ultra 5 325 with Rust 1.94.0, Python 3.14.7,
NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0, medians and
ranges are:

| Numeric formula, with errors | #707, ms | Array designs, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 19 (18–21) | 7 (7–7) | 10 (10–12) |
| Cox selected terms | 20 (19–20) | 9 (9–10) | 10 (10–11) |
| AFT scalar quantile | 22 (21–23) | 10 (10–11) | 4 (4–6) |
| AFT selected terms | 22 (22–23) | 10 (10–11) | 823 (816–888) |

The second workload adds `x1:x2`, `x3:x4` and `x5:x6:x7`; selected output
terms remain the same. It has 19 Cox design columns and 20 AFT columns,
including the intercept.

| Formula with interactions, with errors | #707, ms | Array designs, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 37 (36–39) | 7 (7–8) | 11 (11–17) |
| Cox selected terms | 36 (36–37) | 9 (8–10) | 13 (12–13) |
| AFT scalar quantile | 39 (38–39) | 11 (11–11) | 5 (4–7) |
| AFT selected terms | 38 (38–40) | 10 (10–11) | 992 (973–1050) |

Median complete-call time falls by 55–63% for the numeric formulas and
72–81% with interactions. These are workload-specific elapsed timings;
the timer resolves approximately one millisecond. AFT quantiles remain slower
than stock R. The broader compatibility and performance audit remains open;
these results do not establish full package parity.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 predecessor transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 stock transport
# Repeat all three with a final interactions argument for the second workload.
```
