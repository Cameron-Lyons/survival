# Prediction row labels

The R bridge preserves source row labels on Cox and AFT predictions, including
custom names, Unicode, repeated/reordered subsets and rows restored by
`na.exclude`. It applies the row mask from the existing prediction preparation;
labeling does not evaluate a second model frame or fetch another design matrix.

A private data-frame adapter carries row count and labels separately from
formula columns. Formula row selection uses the existing R `make.unique` helper,
reserving suffixes before missing rows are removed. Models retain the labels
before their NA action. New-data labels use the same omitted rows as numerical
predictions. Automatic row names stay implicit until selected rows or prediction
metadata need them. Valid Python data-frame indices follow the same path; Python
prediction values retain their existing list/result interfaces.
Unnamed Cox outputs and AFT stored response/link calls without errors skip
prediction label materialization. The bridge checks the model's multistate
status once per prediction and converts shared fit/error names once.

The bridge also preserves intentionally absent names. For example, an ordinary
unstratified Cox model's stored linear predictor is unnamed without errors,
while term matrices name their rows. Expected-count predictions can name the
training fit but leave its errors unnamed; new-data expected outputs are
unnamed. AFT stored response/link fits remain unnamed while their errors carry
model-matrix row names. Cox `fitted()` returns its unpadded stored predictors.

Quantile labels follow the R construction: scalar new-data quantiles carry
subject names; multiple quantiles can carry scale-stratum names, and excluded
rows retain their original source labels. Untransformed quantile errors can
remain unnamed. A fixed-scale prediction for one subject repeats the subject's
name on its quantile errors. Fit/error labels shared by a call cross the bridge
once. Grouped prediction names continue to come from the existing group encoder.
These distinctions are checked against the installed stock methods and are
visible in the upstream [Cox prediction method](https://github.com/therneau/survival/blob/master/R/predict.coxph.R)
and [AFT prediction method](https://github.com/therneau/survival/blob/master/R/predict.survreg.R).

## Independent reference and validation

`scripts/generate_prediction_row_reference.R` records 1,680 complete predictions
from R 4.5.3 / survival 3.8-12 across ten ordinary, stratified, ridge, spline,
sparse-only, fixed-scale and intercept-only fits. Cases cover training and new
data, single subjects, omission/exclusion, scalar/multiple quantiles, complete
values, errors, dimensions and names. Python compares every numerical output
and row-label vector. Live R tests additionally cover sparse terms, selected
and empty columns, entirely omitted rows and fitted-model dispatch.

Raw stock calls remain beside explicitly corrected references:

- The penalized Cox wrapper refuses survival predictions. The reference checks
  the ordinary prediction method on the same fitted state, retaining the raw
  wrapper error, as in the preceding prediction references.
- Stock intercept-only AFT terms fail in the empty term loop. The independent
  reference uses the evaluated R model frame's names and a base R matrix with
  zero columns. The numerical contribution is empty, with the retained row count.
- Live sparse-term checks reconstruct the dense stock prediction and append
  fitted frailty contributions/errors (zeros on new data). This preserves the
  existing correction for the stock wrapper's malformed term dimensions.
- Stock AFT terms on entirely omitted new data fail in `attrassign()`. Live
  checks retain that error and compare the supported result to a base R empty
  matrix with the selected column names and width.

The fixture regenerates byte-for-byte. Four newly saved ordinary Cox, sparse
Cox, stratified AFT and fixed-scale AFT models restore in a fresh process with
identical complete predictions, errors, names and attributes. Previously saved
models retain their numerical behavior; older files did not retain custom
source row names, so those lost names cannot be recovered from their state.
Rust sources, Cargo inputs and the native extension are unchanged from #703.

The full Python suite passes 18,654 tests, with 48 skips and 37 expected
differences. The R source archive passes 26,169 checks with zero errors,
warnings or notes. This adds 1,682 Python cases and 11,729 R checks over #704.
Pinned Ruff lint/format, the 47-file type check and generated stubs/manifest
checks pass. Rust suites were not rerun for this increment; #703 validated the
unchanged native implementation.

## Complete-call measurements

`scripts/benchmark_prediction_row_labels.R` compares #704 (`5b19f274`) and this
implementation with separate Python/R libraries and the same release extension.
Each model is fitted to 2,000 rows with 16 covariates. Public calls predict 20,000
new rows with standard errors; term calls select three repeated/reordered named
columns and quantile calls select one probability. Three warmups precede nine
samples in separate processes.

Whole numerical outputs, widths and column names are checked before timing.
The current outputs' complete names are also checked against stock R. Timing
includes bridge conversion, formula design, calculation, output materialization
and row/column metadata; fitting, input creation and explicit GC are excluded.
Stock R uses its own public methods without the Python bridge.

On an Intel Core Ultra 5 325 with Python 3.14.7 and NumPy 2.4.6, medians and
sample ranges were:

| Public call with errors | #704, ms | This change, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox linear predictor | 46 (45–48) | 46 (45–47) | 10 (10–14) |
| Cox selected terms | 85 (80–91) | 85 (79–89) | 15 (14–16) |
| AFT scalar quantile | 55 (54–60) | 53 (52–53) | 6 (6–7) |
| AFT selected terms | 89 (85–94) | 89 (83–92) | 880 (839–939) |

These calls show no observed regression. The linear predictor and term medians
are unchanged with overlapping ranges; the AFT quantile median is 2 ms lower
in this run. The measurements do not establish a general speedup. Cox and AFT
quantile calls remain slower than stock R on this workload; AFT terms are faster.
The broader compatibility and performance audit remains open.

```sh
# Select the copied predecessor's modules and installed R library for its run.
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 baseline
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 current
Rscript scripts/benchmark_prediction_row_labels.R 20000 9 stock
```
