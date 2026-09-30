# Prediction array snapshots

Native Cox vector/term and AFT predictions provide `to_arrays()` snapshots.
The fit and optional standard errors are independent writable float64 NumPy
buffers; absent errors remain `None`. Term snapshots include the existing Cox
constant. Copying and flattening native storage releases the interpreter lock
and avoids Python scalar and row lists.

The R bridge requests these buffers through a private prediction option.
Missing-row padding, grouped-output restoration, sparse-frailty insertion and
AFT missing term columns operate on arrays. Existing Python list properties
and the default formula prediction results retain their list interfaces.
R receives numeric vectors/matrices directly rather than rebuilding each row.
Reticulate makes a column-ordered copy when converting NumPy to R, as described
in its [array documentation](https://rstudio.github.io/reticulate/articles/calling_python.html#arrays).
Formula design preparation and row-label transfer remain part of the call.

`CoxTermsPrediction` and `SurvregPrediction` retain `n_columns` independently
of row count, preserving zero-row and zero-column dimensions. Rust callers
constructing these result structs directly must supply the new field; existing
prediction method signatures and Python list getters remain unchanged.

Entirely omitted AFT quantile populations now retain the stock zero-row matrix
width for multiple probabilities. Their response/link/quantile outputs omit
names as stock R does. Multistratum quantiles preserve an explicitly present
`list(NULL, NULL)` dimension-name attribute, distinct from no attribute.

## Validation

Both list and array paths check the existing 1,680 complete prediction,
656 term-selection and 146 grouped-prediction references. Another 72 independent
stock calls cover entirely omitted ordinary, stratified and fixed-scale AFT
predictions, with/without errors, scalar/multiple probabilities and
omission/exclusion. That fixture regenerates byte-for-byte. Native snapshot
checks cover ownership through mutation and garbage collection, writable
float64 buffers, repeated selections, absent errors and empty dimensions.

The full Python suite passes 21,354 tests, with 48 skips and 37 expected
differences; 2,700 added cases. The R source archive passes 28,083 checks with
zero errors, warnings or notes; 504 added checks. Four #706 models and four
newly saved models restore in fresh processes with identical complete
predictions, errors, names and attributes.

Rust passes 1,678 all-feature tests plus its integration check, 1,654 with ML
and 1,422 without default features. Two new tests check explicit matrix widths
and invalid row lengths. All-target/all-feature Clippy denies warnings;
documentation, formatting and the regression benchmark smoke check pass.
Rust documentation tests contain no runnable examples. Pinned Ruff lint/format,
47-file Mypy and generated stubs/manifest checks pass with the rebuilt release
extension.

## Complete-call measurements

`scripts/benchmark_prediction_row_labels.R` compares #706 (`159f7ee5`) with
the current implementation using separate R/Python libraries and their release
extensions, built with `extension-module,ml`. Each model fits 2,000 rows and
16 covariates, then predicts 20,000 new rows with standard errors. Term calls
select three repeated/reordered columns. Three warmups precede nine samples.
Whole values, errors, dimensions and names match stock R before timing.

Calls include R/Python transport, formula design, numerical calculation,
output materialization and metadata. Fitting, input creation and explicit GC
are excluded. Runs are sequential, after validation workloads finish.
On an Intel Core Ultra 5 325 with Rust 1.94.0, Python 3.14.7, NumPy 2.4.6,
R 4.5.3, survival 3.8-12 and reticulate 1.47.0, medians and ranges are:

| Public call with errors | #706, ms | Array snapshots, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 21 (19–22) | 20 (19–23) | 12 (9–17) |
| Cox selected terms | 58 (57–59) | 20 (19–20) | 10 (10–11) |
| AFT scalar quantile | 25 (24–27) | 22 (21–22) | 6 (5–9) |
| AFT selected terms | 58 (57–61) | 21 (21–22) | 870 (820–923) |

Selected terms take 64–66% less time on this workload. Cox linear
predictor ranges overlap; no speedup is established for that call. The AFT
quantile median falls by 3 ms in this run. Cox and AFT quantile calls remain
slower than stock R, while AFT selected terms remain faster.

The broader compatibility and performance audit remains open. These checks
and timings cover the stated paths and workload; they do not establish full
package parity.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 predecessor transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 stock transport
```
