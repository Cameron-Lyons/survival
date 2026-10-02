# Logical covariates and model-matrix metadata

Formula fits treat unclassed logical variables as factors with `FALSE` and
`TRUE` levels, following R's model-matrix coding. Both levels remain available
when subset or omission leaves only one logical value. Main effects use
`flagTRUE`; interaction-only terms such as `age:flag` produce
`age:flagFALSE` and `age:flagTRUE`. AFT formulas without an intercept retain
both dummy columns. Explicit numeric transforms remain numeric, and explicit
factor constructors retain their single-level validation.

R `model.matrix` returns source row names, fitted contrast declarations and
existing column/term metadata. Repeated subsets retain R's unique row labels,
and omissions use the same row mask as the matrix. Its new-data input now uses
the shared data-frame adapter: reticulate's automatic data-frame conversion
previously turned missing logical values into FALSE, retaining incomplete rows
and corrupting their values.

Default contrasts retain their function names. Explicit contrast matrices
retain their dimensions and absent or supplied dimnames. Dense frailty contrasts
retain identity matrices, including different labels for freshly constructed
groups and fitted groups re-leveled by the new model frame. Time transforms
retain the declarations separately from their evaluated numerical bases.
Expanded Cox matrices use risk-set row labels. Newly fitted models retain those
labels; older models reconstruct their indices without invoking transform
callbacks. The stock Cox constructor's data and metadata handling is described
in [its source](https://github.com/therneau/survival/blob/master/R/model.matrix.coxph.R).

Public Python matrices retain their existing dictionaries and list data. The
private bridge metadata option adds row labels and contrasts. R null Cox models
return NULL coefficients, following stock R. Term assignments retain the
established integer convention; stock's strata term-number adjustment sometimes
promotes this attribute to double.

## Independent checks and persistence

The new fixture records 440 unmodified stock fits over 22 formulas, ordinary
and AFT models, mixed/constant/missing logical values, repeated subsets, and
automatic/Unicode row names. The 432 successful fits provide 2,592 whole matrix
references across stored, complete, partially missing, entirely missing, empty
and single-row inputs. Eight explicit factor-constructor failures are retained.
Coefficient values and labels, matrix values, dimensions, names, assignments,
contrasts and warnings are checked. Convergence-warning whitespace is normalized;
the messages and variable lists are retained. Regeneration is byte-identical.

All 440 cases run with list, NumPy and nullable pandas columns, adding 1,320
Python tests. Another 46 tests check older stored time-transform fits without
cached expanded labels and forbid callback evaluation during reconstruction.
Live R checks compare the same fits/matrices. Earlier sparse, dense, automatic
frailty and time-transform suites now also compare row names and contrasts;
their independent dense-frailty reconstruction retains the stock contrast
attribute after removing strata columns.

The full Python suite passes 32,800 tests, with 48 skips and 37 documented
expected differences. Pinned Ruff 0.16.9 lint/format, 47-file Mypy and generated
stubs/manifest checks pass. The source archive matches all 48 R source/test
files and passes 100,675 checks with zero errors, warnings or notes, adding
19,682 checks over #709.

Four #709 models restore with identical complete predictions and attributes in
separate R processes. Four fresh logical/expanded models restore with identical
whole matrices and attributes, including ordered-factor contrasts. Older logical
models retain their originally fitted numerical designs; refitting is required
to obtain corrected interaction widths. Historical time-transform contrast
declarations cannot always be inferred from an already evaluated matrix.

Rust sources, Cargo inputs, native APIs and the release extension are unchanged
from #707; its Rust gates remain the latest native validation and were not rerun.

## Complete-call measurements

`scripts/benchmark_cox_model_matrix.R` compares #709 (`978274b2`), this
implementation and stock R with separate R/Python libraries and the same
unchanged release extension. A prefit sparse-frailty model has 10,000 rows,
16 numeric covariates and 100 groups; the new-data call uses 5,000 rows and
new group labels. Each call has three warmups and nine samples. All three runs
execute sequentially after validation. Numerical outputs are checked against
stock before timing; current calls also check complete dimnames and contrasts.

Timings include public R/Python conversion, matrix construction and metadata,
excluding fitting, input setup and explicit GC. On an Intel Core Ultra 5 325
with Rust 1.94.0, Python 3.14.7, NumPy 2.4.6, R 4.5.3, survival 3.8-12 and
reticulate 1.47.0, medians and ranges are:

| Complete model-matrix call | #709, ms | Current, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Stored sparse model, 10,000 rows | 14 (14–15) | 15 (14–15) | 0 (0–1) |
| New sparse groups, 5,000 rows | 11 (10–11) | 10 (10–10) | 3 (3–4) |

The bridge timings overlap the predecessor's ranges, with a one-millisecond
median increase for the stored matrix and decrease for new data. No general
speedup is established; stock R remains faster. The timer resolves approximately
one millisecond.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_cox_model_matrix.R 10000 9 predecessor
Rscript scripts/benchmark_cox_model_matrix.R 10000 9 current
Rscript scripts/benchmark_cox_model_matrix.R 10000 9 stock
```

The broader compatibility and performance audit remains open. The subsequent
[AFT input correction](aft-model-matrix-inputs.md) removes the previously
reproduced requirement for otherwise unused standalone strata variables.
These checks do not establish full package parity.
