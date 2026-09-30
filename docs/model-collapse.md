# Grouped model predictions and residuals

`survival.r.predict()` groups ordinary Cox predictions with `collapse=groups`.
`survival.r.residuals()` groups ordinary Cox and AFT residuals the same way.
Factor groups follow their declared level order, with unused levels removed.
Other groups follow sorted value order: numeric labels sort numerically and
text labels sort lexically, including text that looks numeric. Missing labels
form a final group and emit a warning. Numeric `NaN` and `None`/R `NA` are
distinct groups; factor missing values share one missing group.

```python
from survival import datasets, r

data = datasets.load_ovarian()
fit = r.coxph("Surv(futime, fustat) ~ age + rx", data)
groups = [i % 3 for i in range(len(data["futime"]))]
predictions = r.predict(fit, type="lp", se_fit=True, collapse=groups)
residuals = r.residuals(fit, type="score", collapse=groups)
```

The Python return types remain lists and `PredictResult`. The R methods attach
group labels as vector names or matrix row names, including each component of
a prediction result with standard errors. Labels are plain character vectors.
The R wrapper preserves labels even when a stock method loses them while
dropping a one-column matrix. Existing partial-residual matrix shapes are
retained by the Python and R interfaces.

Prediction values sum within each group. Standard errors are
`sqrt(sum(se^2))`, as in R's collapse method. The label encoding is shared by
both calculations. For new-data predictions, missing collapse labels also
contribute to `na.omit`, `na.exclude` and `na.fail`. Under `na.pass`, a missing
group can hold valid predictions. Under `na.exclude`, omitted rows are padded
before grouping, so a group containing an omitted row propagates missingness.

Cox residuals accept `collapse=True` to use the fitted cluster or id. Their
factor level order survives subsets and saved-model restoration. With neither
cluster nor id, the call raises an explicit error. Schoenfeld and scaled
Schoenfeld residuals ignore collapse, following R's early-return behavior.

## Shared Rust calculation

`survival::core::grouped_sum(values, group, squares)` and
`survival.core.grouped_sum(values, group, squares=False)` sum matrix rows by
ascending integer group. The function preserves input addition order within
each group, validates the group length and retains empty dimensions. With
`squares=True`, it returns the square root of the sum of squares.

The native Python binding accepts lists and C, Fortran or strided NumPy
matrices, returns an owned NumPy matrix, and releases the interpreter lock
during the calculation. Existing native residual sums use this same kernel.
Grouped Cox predictions now sum inside the native prediction call; term
outputs allocate one row per group. Sparse frailty additions use this kernel.
See [native prediction grouping](cox-grouped-prediction-performance.md) for
storage, omission behavior and subsequent performance measurements.

## Independent references

`scripts/generate_model_collapse_reference.R` records 710 complete calls from
R 4.5.3 with survival 3.8-12. It covers ordinary, weighted/stratified, ridge,
sparse frailty, cluster, id, subset and excluded-row Cox fits; ordinary and
ridge AFT fits; numeric, numeric-looking text, factor, logical, missing and
NaN/NA groups; and new-data omission policies. Numerical comparisons check
whole prediction, error and residual arrays. Separate native checks cover
ownership, input layouts, addition order, empty shapes and thread execution.
R tests check complete live calls, output names, R warning conditions and
saved factor groups. Cluster and id models also restore with identical grouped
values and labels in a fresh R process. Local validation passes 16,133 Python
tests (48 skipped and 37 expected differences), 10,762 R checks with zero
errors, warnings or notes, and 1,673/1,649/1,417 Rust unit tests with all, ML and
no-default features respectively. Strict Clippy, rustfmt, Rust documentation,
Python lint/format, full facade typing and generated interface checks pass.
The reference regenerates byte-for-byte.

The reference records retain unmodified stock results and errors. Independent
calculations handle these known failures:

- Stock sparse Cox predictions can collapse in both the delegated prediction
  method and the penalized method. Group uncollapsed R predictions once.
- Sparse term reconstruction and partial residuals can fail for a one-column
  dense design. Retain the fitted coefficients, covariance and residuals in
  an ordinary R dense-term view; add fitted frailty values and `sqrt(fvar)`
  only to term predictions.
- Penalized Cox prediction rejects `type="survival"`. Derive per-row survival
  as `exp(-expected)` and its error as `se_expected * exp(-expected)` before
  grouping.
- Stock excluded-row collapse can check a padded group vector against an
  unpadded response. Pad predictions or residuals before base R `rowsum`.
  Deviance uses collapsed martingale residuals and event counts.
- Stock Cox residuals reject logical group vectors through a scalar logical
  condition. Apply base R `rowsum` to uncollapsed residuals instead.

The stock behavior is visible in the upstream
[penalized prediction method](https://github.com/therneau/survival/blob/master/R/predict.coxph.penal.R),
[Cox residual method](https://github.com/therneau/survival/blob/master/R/residuals.coxph.R)
and [base rowsum documentation](https://stat.ethz.ch/R-manual/R-devel/library/base/html/rowsum.html).
These references do not require copying or changing upstream functions.

## Complete-call timings

These measurements record PR #702. The subsequent
[native grouping change](cox-grouped-prediction-performance.md) removes full
prediction output transfers before summation.

`scripts/benchmark_model_collapse.R` measures public R calls on a prefit model
with 20,000 rows, 16 dense covariates and 257 numeric groups. Each mode runs in
a separate process with three warmups and nine samples. Times include bridge
conversion, group encoding, prediction or residual calculation, aggregation,
standard errors and output metadata; fitting and explicit garbage collection
are excluded. Complete numerical outputs agree with stock R before timing.

On an Intel Core Ultra 5 325 with R 4.5.3, survival 3.8-12, Python 3.14.7,
NumPy 2.4.6 and a release extension, median milliseconds (sample range) were:

| Public call | Previous PR #701 | PR #702 | Stock R |
| --- | --- | --- | --- |
| Grouped linear predictor with errors | 11 (10–12) | 7 (6–8) | 7 (7–9) |
| Grouped terms with errors | 59 (55–60) | 39 (38–41) | 15 (13–15) |
| Grouped dfbeta residuals | 8 (7–8) | 8 (7–8) | 6 (6–7) |

The prediction calls improve over the predecessor; dfbeta ranges overlap.
Stock R remains faster for the term workload. These measurements cover this
data and machine and do not establish package-wide performance parity.

```sh
Rscript scripts/generate_model_collapse_reference.R
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_model_collapse.py -q
Rscript scripts/benchmark_model_collapse.R 20000 9 installed
Rscript scripts/benchmark_model_collapse.R 20000 9 stock
# For the predecessor, use its Python module and installed R library in a
# separate process and pass baseline as the third argument.
```
