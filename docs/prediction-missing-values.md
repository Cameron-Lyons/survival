# Missing values and omitted prediction rows

Cox and AFT predictions preserve R's distinction between `NA` and numerical
`NaN`. Unclassed double data-frame columns cross into NumPy in bulk even when
they contain missing values. Their floating-point payload survives formula
evaluation, native calculation and conversion back to R. Missing integers
retain the scalar path and their source integer type in saved model frames.
Categorical missing values, missing term selections and rows restored by
`na.exclude` produce `NA`, while numerical NaNs retain their classification.
Python still receives floats for both kinds; both satisfy `math.isnan`.

The marker follows the low-word payload checked by
[R's `R_IsNA` implementation](https://github.com/wch/r-source/blob/R-4-5-branch/src/main/arithmetic.c).
Logarithms, square roots and division retain source missing kinds; domain
errors such as `sqrt(-1)` and `0/0` still produce genuine NaNs. Penalty bases
retain their existing missing-row treatment.

AFT quantiles drop single-row and single-column dimensions before restoring
excluded rows, following `predict.survreg`. For four new rows, three excluded
rows and three quantiles, the retained row's three values form a vector before
`naresid` inserts three missing entries. If the retained row is third, the
result is `c(NA, NA, q10, NA, q50, q90)`, length six. The previous implementation
returned a four-by-three matrix. Fixed-scale errors repeat the retained row's
name before restoring omission labels, while unnamed outputs remain unnamed.
Entirely omitted predictions retain stock dimensions and absent row names.
Empty grouped Cox term matrices retain explicit `character(0)` group labels.

New-data linear, risk and term predictions from frailty-only Cox fits use the
new row count without evaluating the frailty variable, as stock R does.
Missing frailty values therefore do not remove rows on these paths.

## Independent references

`scripts/generate_omitted_prediction_reference.R` records 4,704 cases across
12 ordinary, stratified, ridge, sparse-frailty, frailty-only, categorical,
spline and fixed-scale models. Cases retain zero, one, two or four new rows;
use source NA or genuine NaN; apply pass, omission or exclusion; and cover
prediction types, scalar/multiple quantiles, errors and grouped Cox calls.
Complete values, missing kinds, dimensions, row/column names and warnings
are checked. The fixture regenerates byte-for-byte. Its NA payload comes
independently from R's `writeBin(NA_real_)`.

The fixture records 676 errors from the original stock calls. It retains the
previously documented repairs for grouped/sparse Cox predictions and empty
AFT terms. Expected values for those paths come from ordinary stock predictions,
base R `rowsum`/error quadrature and `naresid`, with fitted stock term labels.
When every spline input is missing, stock raises an empty-derivatives error;
an appended valid control row obtains the missing basis from stock R, then
is removed to reconstruct the expected empty or padded output. These
reconstructions are recorded explicitly alongside raw errors; they are not
successful unmodified stock calls.

Both Python list and array outputs run every reference, adding 9,408 tests.
Live R tests compare all 4,704 cases, including `is.na` and `is.nan` separately.
Earlier row, selector, grouped and empty-array suites also check `is.nan`
without normalizing it away. Bridge tests verify payload preservation,
source integer model-frame types and array ownership after mutation and garbage
collection. Separate formula controls check source NA/NaN and computed domain
errors through logarithms, square roots and division.

## Validation and stored fits

The full Python suite passes 31,434 tests, with 48 skips and 37 documented
expected differences. Pinned Ruff 0.16.9 lint/format, 47-file Mypy and generated
stubs/manifest checks pass. The R source archive matches all 46 source/test
files and passes 80,993 checks with zero errors, warnings or notes, adding
52,910 checks over #708. Rust sources, Cargo inputs, native APIs and the release
extension are unchanged from #707; its Rust gates remain the latest native
validation and were not rerun for this change.

Four #708 models restore in separate processes with unchanged finite values,
errors, shapes, names and attributes. Their 36 generic NaN omission placeholders
now become R NA; comparisons to the old output normalize only this intentional
missing-kind correction. Four newly saved models restore with identical whole
outputs and attributes. An additional fixed-scale AFT fit with only one retained
training row matches stock multiple-quantile shapes, values, errors and names,
and restores identically in a fresh process.

## Complete-call measurements

The benchmark compares #708 (`069f4146`) with this implementation and stock R
using separate R/Python libraries and the same unchanged release extension.
Each model fits 2,000 rows with 16 numeric covariates and predicts 20,000 new
rows with errors. Term calls select `x9`, `x1`, `x9`; AFT quantiles use `p=.5`.
There are three warmups and nine samples per call. All six runs execute
sequentially after validation. Every run checks whole values, errors, dimensions
and names against stock before timing; current and stock runs also check
`is.nan`. The predecessor retains its known missing-kind discrepancy.

Timings include bridge transport, formula design, native calculation, output
materialization and metadata; fitting, input setup and explicit GC are excluded.
The machine has an Intel Core Ultra 5 325, Rust 1.94.0, Python 3.14.7,
NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.

Complete numeric data remain within previous timing ranges:

| Complete data, with errors | #708, ms | Current, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 7 (7–7) | 7 (7–7) | 11 (10–12) |
| Cox selected terms | 9 (9–10) | 9 (8–9) | 10 (10–11) |
| AFT scalar quantile | 10 (9–10) | 10 (9–10) | 4 (3–5) |
| AFT selected terms | 10 (9–10) | 9 (9–10) | 822 (815–875) |

The missing-data workload inserts 213 NA and 213 NaN values in `x1`, using
`na.pass` and the same covariates and selected terms.

| Missing data, with errors | #708, ms | Current, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 17 (17–18) | 7 (7–7) | 12 (11–12) |
| Cox selected terms | 19 (18–20) | 9 (8–9) | 10 (10–11) |
| AFT scalar quantile | 21 (20–21) | 10 (9–10) | 6 (6–7) |
| AFT selected terms | 20 (20–21) | 10 (9–10) | 919 (903–962) |

Median complete-call time falls by 50–59% on this missing-data workload.
Isolated conversion of all 17 input columns takes 1 ms (0–1), compared with
8 ms (7–9) previously; this conversion timing is separate from complete calls.

These workload-specific elapsed timings have approximately one-millisecond
resolution. AFT quantiles remain slower than stock R. The broader compatibility
and performance audit remains open; these results do not establish full package
parity.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 predecessor transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current transport
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 stock transport
# Repeat all three with numeric missing for the missing-data workload:
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current transport numeric missing
```
