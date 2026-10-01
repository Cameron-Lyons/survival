# Fitted and new-data model matrices

`survival.r.model_matrix()` and the R `model.matrix()` method now distinguish
formula/basis labels from fitted penalty coefficient labels. For example,
`ridge(age, theta = 2)` retains the coefficient name `ridge(age)` but uses the
full formula call for its matrix column. Multi-column ridge terms append the
argument names; spline and dense frailty bases append numbered suffixes.
Named time-transform penalty matrices retain their original basis suffixes.
The same label correction applies to formula-based AFT models.

Stored Cox matrices include sparse frailty columns in their original formula
positions, with the original numeric codes. Reconstruction inserts that vector
directly into one owned dense row snapshot, avoiding the former column-block
copies and transposes. Changing a returned matrix cannot change the fitted
model, its coefficients, or predictions. Native fitting keeps its compact dense
coefficient matrix.

New-data Cox matrices evaluate the complete formula design, including sparse
frailties. Group coding precedes missing-row omission: groups occurring only
on an omitted row still contribute to the local codes. Explicit sparse terms
accept new groups and recode them locally. Dense terms retain fitted group
levels and full indicator contrasts, including declared factor order. The
automatic frailty default can change a sparse term into a dense basis on a
small new dataset. Dense constructors still reject fewer than two observed
groups; a fitted dense term switching to sparse codes fails because its factor
contrasts no longer apply, matching R.

Term assignments preserve original Cox numbering across strata terms. AFT
assignments retain their existing strata removal convention. The R method
accepts its named `data` argument, retains new-data strata factor levels, and
preserves the width of an empty matrix. Expanded matrix time transforms still
require their fitted time/risk-set context for reconstruction on new data.

## Independent validation

`scripts/generate_cox_model_matrix_reference.R` records 168 cases from R 4.5.3
and survival 3.8-12: ordinary factors/interactions, ridge, splines, all three
frailty families, sparse/dense/automatic bases, formula positions, strata,
offsets, incomplete rows, unseen groups, one-group inputs and empty frames.
Every case retains its raw stock matrix, error and warnings. Tests check
values, names and assignments, with live R checks for saved models and factor
metadata. The existing 32 time-transform penalty references now also check
matrix column names, and a named ridge transform verifies basis labels survive
independently of its coefficient labels.

The full Python suite passes 15,412 tests (48 skipped and 37 documented expected
differences), including 173 new cases. The R source archive passes 8,912 checks
with zero errors, warnings or notes after aligning the method documentation.
Pinned Ruff 0.16.9 lint/format, Mypy across 47 files, generated interfaces and
byte-exact regeneration of both references pass. Four sparse/dense/spline/TT
models restore in a fresh R process with identical stored and new-data matrices.
Rust sources and the native extension are unchanged from #697.

Three reference adjustments are marked explicitly:

- The stock Cox strata-removal branch misspells `contrasts.arg`, causing a
  rebuilt dense frailty to lose its first indicator. The reference uses
  `stats::model.matrix()` with the fitted full contrasts, then removes strata.
- `makepredictcall.pspline` drops original `df` and `penalty=FALSE` options,
  making a valid small unpenalized basis fail under the default `df=4`.
  The reference uses R's spline constructor with original options and fitted
  boundary knots.
- R's spline constructor fails on empty input. The zero-row reference retains
  the fitted columns and assignments.

These differences are supported by the installed implementation and the
upstream [Cox matrix method](https://raw.githubusercontent.com/therneau/survival/master/R/model.matrix.coxph.R)
and [spline implementation](https://raw.githubusercontent.com/therneau/survival/master/R/pspline.R).
The port retains full fitted contrasts and spline metadata for these inputs.

## Complete-call timings

`scripts/benchmark_cox_model_matrix.R` measures complete public R calls on a
prefit model, including conversion and metadata. It uses 10,000 rows, 16 dense
columns, a sparse frailty, three warmups and nine samples in separate processes.
New-data calls use 5,000 rows with renamed groups. Fitting and explicit GC are
excluded; matrix values are checked against R before timing. The predecessor
is #700 (`02f58e0c`), with its own R/Python sources and the same native extension.

| Call | #700 median (range), ms | Current median (range), ms | Stock R median (range), ms |
| --- | --- | --- | --- |
| Stored sparse matrix | 19 (18–27) | 14 (13–15) | Below 1 ms resolution |
| New sparse matrix | Incorrect column count; excluded | 11 (10–11) | 3 (3–4) |

Stored extraction's local median is about 26% lower. The new-data path and
stored extraction remain slower than stock R, whose stored method returns
its retained matrix. These measurements cover extraction, not fitting or
whole-package performance.

```sh
Rscript scripts/generate_cox_model_matrix_reference.R /tmp/matrix-reference.json
Rscript scripts/benchmark_cox_model_matrix.R 10000 9 installed
Rscript scripts/benchmark_cox_model_matrix.R 10000 9 stock
```
