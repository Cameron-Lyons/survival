# Grouped Cox prediction performance

Ordinary and penalized Cox predictions with `collapse=groups` now sum results
inside the native prediction call. Only grouped values and standard errors
cross into Python. Term predictions allocate one output row per observed
group, rather than one per input row. The public Python and R result types
remain the same.

Previously, the term path built full native prediction and error matrices,
converted both to Python lists, copied those lists into NumPy matrices, and
returned them to Rust for summation. The new path prepares the label codes
once, passes the retained rows' codes to the predictor, and returns its group
rows directly. Sparse frailty term values are added in their selected formula
positions, with their errors grouped in quadrature. The frailty-only path
retains its vector shape, including excluded rows.

Within a group, values retain the original input addition order. Errors are
`sqrt(sum(se_i^2))`; each per-row error is computed before squaring so that
negative variances continue to propagate NaN. Reference centering, offsets,
weights, aliases, term selection and term prediction constants are unchanged.
Repeated selections and terms with no columns retain their output widths.

Group labels are encoded against the padded output before prediction. The
native call receives codes for retained rows. Afterward, the wrapper restores
groups that contain an omitted row, including a group with no retained row.
This avoids constructing padded full prediction matrices. `na.omit` first
subsets the group labels, retaining factor levels; `na.exclude` propagates
missingness to the appropriate group. With all rows omitted, R term results
retain their zero-row matrix shape and column names.

## Native interfaces and storage

Rust callers can use `CoxPHFit::predict_terms_grouped(newdata, se_fit,
reference, assign, group)`. Existing `predict_terms()` calls retain their
signature and use the same per-term calculation. Vector predictions can use
`CoxPrediction::collapse(group)` after `predict_lp`, `predict_risk`,
`predict_expected` or `predict_survival`.

Native Python calls accept a new keyword-only `collapse` argument:

```python
terms = fit.fit.predict_terms(se_fit=True, collapse=integer_group_codes)
lp = fit.fit.predict("lp", se_fit=True, collapse=integer_group_codes)
```

Native codes have one integer per prediction row; arbitrary signed labels
sort in ascending order. The R-style `r.predict()` interface also accepts
factor, text, logical and missing labels, with one value per padded row.
The native methods validate lengths, accept NumPy layouts through the typed
boundary, release the interpreter lock, and return the existing result classes.
Their list properties are owned snapshots.

For N input rows, G groups and T selected terms, term output storage is
O(G × T), replacing O(N × T) native and Python output buffers. The centered
N × p prediction design remains, along with the row-group map and input
conversion storage. Vector predictions still allocate N native values before
grouping; they avoid transferring those values to Python.

## Independent validation

The existing 710 grouped-model references still cover ordinary, weighted,
stratified, ridge and sparse Cox fits, all five prediction types, six label
forms and missing-data policies. The new
`scripts/generate_grouped_cox_prediction_reference.R` records another 146
complete calls from R 4.5.3 with survival 3.8-12. These cover ordinary,
single/multiple-column ridge, sparse and frailty-only models, groups containing
only excluded rows, repeated/empty term selections, errors on/off and new-data
omission. It retains raw stock results/errors and calculates independent
expected values from uncollapsed R predictions plus base `rowsum`, fitted
frailty values and expected-to-survival transformations.
These calculations follow the upstream
[ordinary](https://github.com/therneau/survival/blob/master/R/predict.coxph.R)
and [penalized](https://github.com/therneau/survival/blob/master/R/predict.coxph.penal.R)
prediction methods, with the documented stock failures kept in the records.

Native checks cover all five prediction types, C/Fortran/strided inputs,
signed group codes, ownership, wrong lengths and zero-column terms. Rust
checks include addition order, error quadrature, repeated and empty assignments,
reference choices and per-term NaN propagation. R tests verify complete
outputs, labels, omitted-only groups and zero-row widths, including empty,
single, repeated and reordered terms with one, two or three groups. A separate thread
check exercises the grouped term kernel. Two saved models from the predecessor
restore in a fresh R process with identical predictions, errors, names and
residuals.

Local validation passes 16,311 Python tests (48 skipped and 37 expected
differences), 11,264 R checks with zero errors, warnings or notes, and
1,676/1,652/1,420 Rust unit tests with all, ML and no-default features
respectively. Integration tests, strict Clippy, rustfmt, Rust documentation,
Python lint/format, full facade typing and generated interface checks pass.
The 146-case reference regenerates byte-for-byte.

## Complete-call measurements

Measurements compare the preceding PR #702 (`1fa9a144`) with this implementation,
using each build's own release extension in separate processes. The machine
is an Intel Core Ultra 5 325 with Python 3.14.7, NumPy 2.4.6, R 4.5.3 and
survival 3.8-12. Complete numerical outputs are checked before timing.

`scripts/benchmark_model_collapse.R` measures public R calls on a prefit
20,000-row model with 16 covariates and 257 groups. Three warmups precede nine
samples. Times are milliseconds (median and range) and include bridge conversion, group encoding, prediction or
residual calculation, aggregation, errors and result metadata; fitting and
explicit garbage collection are excluded.

| Public R call | #702 | Native grouping | Stock R |
| --- | --- | --- | --- |
| Grouped linear predictor with errors | 7 (6–7) | 7 (6–7) | 7 (7–9) |
| Grouped terms with errors | 37 (36–40) | 8 (8–9) | 14 (13–14) |
| Grouped dfbeta | 8 (7–8) | 8 (8–12) | 6 (5–6) |

The term call is about 4.6 times faster than the predecessor and 1.75 times as
fast as stock R in this workload. Linear-predictor and dfbeta ranges overlap;
dfbeta uses its existing calculation.

`scripts/benchmark_grouped_cox_predictions.py` measures the complete public
Python new-data term call with errors: 100,000 rows, 32 covariates and 257
groups, on a model fitted to 2,000 rows. Three warmups precede seven samples.
Timing includes formula evaluation, group encoding, conversion, native
calculation and result materialization, excluding fitting, input setup and
explicit garbage collection.

| Python call | #702 | Native grouping |
| --- | --- | --- |
| Median time, ms (range) | 545.1 (541.9–565.6) | 250.1 (246.8–262.3) |
| Peak process RSS, MiB | 570.2 | 290.2 |

Whole grouped predictions and errors are identical across builds in this run.
Peak RSS includes the interpreter, complete process, fitting and inputs; it
does not isolate temporary allocations. These measurements describe these
workloads and machine, rather than package-wide performance parity.

```sh
# Use the predecessor's copied Python module and R library for baseline runs.
PYTHONPATH=/tmp/previous-survival/python .venv/bin/python \
  scripts/benchmark_grouped_cox_predictions.py --mode baseline --output /tmp/grouped-before
PYTHONPATH=python .venv/bin/python scripts/benchmark_grouped_cox_predictions.py \
  --output /tmp/grouped-after --compare /tmp/grouped-before
Rscript scripts/benchmark_model_collapse.R 20000 9 installed
Rscript scripts/benchmark_model_collapse.R 20000 9 stock
```
