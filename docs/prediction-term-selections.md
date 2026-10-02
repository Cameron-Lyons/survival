# Prediction term selections and column labels

AFT term predictions now use R's matrix column selection rules. Positive
indices are one-based, zero selects no column, negative indices exclude terms,
and finite fractional indices truncate toward zero. Logical masks recycle
across columns; a mask longer than the number of terms raises an error.
Repeated indices retain their order. Numeric missing indices create missing
columns; logical missing values create a missing column at each selected mask
position. Missing columns retain missing names and errors.

```python
from survival import r

# For a model with terms age and rx:
age = r.predict(fit, type="terms", terms=[True, False])
rx = r.predict(fit, type="terms", terms=-1)
empty = r.predict(fit, type="terms", terms=0)
with_missing = r.predict(fit, type="terms", terms=[1, float("nan")], se_fit=True)
```

Python scalar and NumPy scalar/vector inputs share the same rules. In Python,
`None` omits the argument; a list containing `None` represents logical missing
values, while a numeric NaN is a missing numeric index. The R bridge retains
the vector's original type, including an explicit `NULL`, so a numeric `NA`
selects one missing column and a logical `NA` recycles across the terms.
Factor selectors retain their level codes, following R's matrix indexing.

Cox predictions retain their stricter upstream validation: only valid term
names and positive integral indices are accepted. Validation also applies
when a supplied `terms` argument accompanies a vector prediction. AFT ignores
`terms` for other prediction types. An explicit R `terms=NULL` selects no Cox
term columns, while AFT keeps all columns. Mixing positive and negative AFT
indices, selecting an unknown name, using an oversized logical mask or
selecting outside the matrix produces the corresponding R error. Integer
coercion overflow signals R warning conditions, including both matrix
subscripts when errors are requested.

The Rust predictor receives only selected valid indices. The wrapper inserts
missing columns into the requested positions afterward, then applies row
omission or exclusion. This preserves zero-column and zero-row widths without
calculating every term to resolve a selector. Rust numerical routines, native
method signatures and public Python result classes are unchanged from #703.

R prediction columns now use the fitted coefficient assignments, excluding
standalone strata terms. For example, `strata(cl) + age + rx` predicts columns
`age` and `rx`, including when selecting or repeating terms. The public
`model_term_names()` helper continues to return the complete formula labels,
including strata; prediction metadata has a separate internal helper.
Ordinary, penalized and fitted AFT predictions use the same column naming.
The R prediction bridge also accepts the standard `na.omit`, `na.exclude`,
`na.pass` and `na.fail` functions as `na.action` arguments.

## Independent checks and stock R differences

`scripts/generate_prediction_term_selection_reference.R` records 656 calls
from R 4.5.3 with survival 3.8-12. They cover ordinary, stratified, ridge and
spline AFT models, ordinary/stratified/ridge Cox models, missing new-data
predictors, selected widths, ordering, names, errors and warnings. Raw R calls
remain in the fixture. Python checks every recorded numerical matrix, output
width, selected column name, error and warning. Live R tests also cover
training-row exclusion, new-data omission/exclusion, entirely omitted data,
fitted-model dispatch and R warning conditions.

The full Python suite passes 16,972 tests, with 48 skips and 37 expected
failures. The R source archive passes `R CMD check` with 14,440 checks and no
errors, warnings or notes. Ruff, the 47-file type check, generated stubs and
the binding manifest pass. This adds 661 Python cases and 3,176 R checks over
#703. Rust source, Cargo inputs and the native extension are unchanged; this
change uses #703's existing Rust validation rather than rerunning those suites.

Cox and AFT models saved by a separate #703 process restore in a fresh process
with identical numerical term predictions and standard errors. Their column
labels now identify the actual covariates; empty and missing AFT selections
also retain their widths and missing column names after restoration.

Two corrections are explicit in the records:

- Stock AFT prediction passes the full formula terms to `attrassign()` after
  its model matrix has removed standalone strata columns. When strata precede
  covariates, the remaining columns acquire displaced labels and named
  selection can choose the wrong numerical contribution. The independent
  reference groups R's fitted model matrix against its strata-removed terms,
  labels the unselected R numerical predictions accordingly, and applies base
  R matrix indexing. Raw displaced labels, selections and errors remain.
- The stock penalized Cox wrapper refuses `type="survival"` before validating
  an invalid selector. The reference retains that error and checks the same
  selector through the ordinary Cox prediction method, consistent with the
  existing supported penalized survival path.

These behaviors are visible in the upstream
[AFT prediction method](https://github.com/therneau/survival/blob/master/R/predict.survreg.R)
and [Cox prediction method](https://github.com/therneau/survival/blob/master/R/predict.coxph.R).
The reference regenerates byte-for-byte. Its value and column checks do not
establish row-name parity; a subsequent [prediction row-label reference](prediction-row-labels.md)
checks that metadata separately.

## Complete-call measurements

`scripts/benchmark_prediction_term_selections.R` compares #703 (`0004beb3`)
and this implementation with separate R/Python libraries and the same release
extension. It measures public new-data term predictions with errors and a
repeated named selection, checking complete values, errors, shapes and column
names against stock R before timing. Each model is fitted to 2,000 rows with
16 covariates; prediction uses 20,000 rows and selects three columns. Three
warmups precede nine samples in separate processes. Timing includes bridge
conversion, formula design, selection, native calculation, materialization and
column metadata; fitting, input creation and explicit GC are excluded.

On an Intel Core Ultra 5 325 with Python 3.14.7 and NumPy 2.4.6, the measured
medians and sample ranges were:

| Public call | #703, ms | This change, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Cox terms with errors | 87 (81–91) | 86 (81–89) | 13 (12–14) |
| AFT terms with errors | 89 (82–90) | 86 (80–88) | 906 (889–960) |

The predecessor and current ranges overlap, so these samples show no observed
regression and do not establish a speedup from the selector changes. Stock R
uses its own public methods without the Python bridge. Cox remains slower
than stock R on this workload; AFT is faster here. The measurements cover
repeated named selections supported by both versions, not every new selector
or overall package performance.

```sh
# Use the copied predecessor's Python module and installed R library for its run.
Rscript scripts/benchmark_prediction_term_selections.R 20000 9 installed
Rscript scripts/benchmark_prediction_term_selections.R 20000 9 stock
```
