# Model-matrix missing-data rules

Fresh Cox and AFT model matrices honor R's global `na.action` option. `na.omit`
and `na.exclude` omit incomplete model-frame rows; `na.pass` retains rows and
missing matrix entries; `na.fail` refuses incomplete frames. The previous
implementation always omitted rows, even under `na.pass` or `na.fail`.

Python `model_matrix` accepts `na_action`, defaulting to `"na.omit"`. Its usual
short and dotted aliases are accepted, and `None` applies no omission. Stored
matrices retain their fitted omission masks and do not validate an unused new
action. R new-data matrix calls resolve `options("na.action")`; an explicit
`na.action` in `...` is ignored, following the stock methods. The
[stock Cox method](https://github.com/therneau/survival/blob/master/R/model.matrix.coxph.R)
builds its fresh model frame without forwarding that argument. Installed Cox
and AFT methods were also inspected directly.

Cox strata attributes retain missing group labels in the same rows as the
matrix. They no longer substitute a fitted group or lose rows when Python
`None` values cross into R. Strata interactions retain missing dummy entries.
AFT continues to ignore standalone scale strata while evaluating interactions
and ordinary uses of the same variables.

Under `na.pass`, unused formula transforms still signal their domain warnings
without propagating missing values into unrelated design columns. Ridge bases
retain their input columns independently: a missing `age` leaves an observed
`rx` intact in `ridge(age, rx)`, and genuine numeric NaN remains NaN instead of
becoming R NA. Nullable pandas logical values no longer trigger ambiguous
boolean comparisons during factor coding.

Fresh Cox frailty matrices also retain used iterator columns before group
processing. See [model-matrix iterator inputs](model-matrix-iterator-inputs.md)
for independent sparse and dense frailty controls and current measurements.

## Independent checks and persistence

Seventy-two unmodified stock fits cover 18 formulas, both model families and
automatic/Unicode row names. Four omission options and 15 input variants yield
288 cases with 3,952 whole matrices and 368 validation failures. Numeric,
logical and factor covariates, interactions, ordinary/combined/interaction
strata, offsets, arithmetic/domain transforms, unused transforms, cluster
terms, single/multiple ridge columns and intercept-only models are represented.
Inputs include source NA and NaN, individual and entirely missing rows, formula
domain errors, empty frames and single incomplete rows.

All 288 cases run with list, NumPy and pandas columns, adding 864 Python tests.
Two validation/alias controls add another two. Pandas numeric object columns
preserve source None versus NaN; logical columns use pandas' nullable boolean
type. Complete values, dimensions, names, assignments, contrasts and missing
kinds are compared. NA payload masks and genuine NaN masks are checked
separately. Live R tests compare the same outputs and strata factor attributes;
explicit new-data arguments and uncached stored matrices also have controls.
The stock fixture regenerates byte-identically. Only convergence-warning
whitespace and domain-warning expression context are normalized. Failure checks
compare the missing-value cause; successful-call warnings are compared.

The full Python suite passes 34,315 tests, with 48 skips and 37 documented
expected differences, adding 866 tests over #711. Pinned Ruff 0.16.9 lint/format,
47-file Mypy and generated stubs/manifest checks pass. The R source archive
matches all 53 R source/test files and passes 166,020 checks with zero errors,
warnings or notes, adding 36,110 checks over #711.

Four #711 models and four fresh models restore in separate R processes with
identical stored matrices and complete predictions, plus stock-compatible
whole `na.pass` matrices on incomplete new frames. Four earlier AFT models also
restore identical stored matrices/predictions and reduced-input matrices.
Reference assignments retain the established integer convention. Rust sources,
Cargo inputs, native APIs and the release extension are unchanged from #707;
its Rust gates remain the latest native validation and were not rerun.

## Complete-call measurements

`scripts/benchmark_matrix_na_action.R` compares #711 (`41820f0b`), this
implementation and stock R in separate R/Python libraries with the same release
extension. Each model has 20,000 fitted rows, 16 numeric covariates and five
strata. New frames have 10,000 rows. Incomplete frames have numeric NA every
59 rows and strata NA every 73 rows. Stored, complete and incomplete frames are
measured under omission and pass rules.

Each call includes input conversion, omission, matrix construction and metadata;
fitting, input setup, option changes and explicit GC are excluded. Each workload
has three warmups and nine samples. All runs execute sequentially after
validation. Complete values and attributes, including NA/NaN masks, are verified
before timing. Predecessor `na.pass` calls that wrongly omit rows are verified
but not timed, because their result differs from stock.

Milliseconds, median (minimum–maximum), measured on an Intel Core Ultra 5 325
with Python 3.14.7, NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.
The unchanged release extension was built with Rust 1.94.0.

| Family | Call | #711 | This change | Stock R |
| --- | --- | --- | --- | --- |
| Cox | Stored | 34 (32–42) | 36 (33–43) | 0 (0–0) |
| Cox | Complete, omit | 21 (20–22) | 22 (21–22) | 3 (2–3) |
| Cox | Complete, pass | 21 (20–22) | 22 (21–22) | 2 (1–2) |
| Cox | Incomplete, omit | 26 (25–27) | 27 (26–27) | 3 (2–3) |
| Cox | Incomplete, pass | Incorrect; not timed | 25 (24–26) | 1 (1–2) |
| AFT | Stored | 41 (38–54) | 41 (39–53) | 0 (0–0) |
| AFT | Complete, omit | 19 (19–20) | 19 (18–20) | 1 (1–2) |
| AFT | Complete, pass | 20 (19–20) | 19 (18–20) | 1 (0–1) |
| AFT | Incomplete, omit | 24 (23–25) | 23 (23–24) | 1 (1–2) |
| AFT | Incomplete, pass | Incorrect; not timed | 22 (21–22) | 1 (0–1) |

Previous incomplete pass calls returned 9,695 Cox rows and 9,830 AFT rows;
the corrected calls return all 10,000 rows, matching stock. Comparable ranges
overlap, and no general speedup is claimed. Cox new-call medians increase by
1 ms in this workload. Stock remains faster; its stored calls are below this
timer's 1 ms resolution. These measurements cover ordinary numeric designs
with strata, not every formula or penalty family.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_matrix_na_action.R 20000 9 predecessor
Rscript scripts/benchmark_matrix_na_action.R 20000 9 current
Rscript scripts/benchmark_matrix_na_action.R 20000 9 stock
Rscript scripts/generate_matrix_na_action_reference.R
Rscript -e 'library(survival); print(getS3method("model.matrix", "coxph")); print(getS3method("model.matrix", "survreg"))'
```

The broader compatibility and performance audit remains open. The next formula
evaluation audit has reproduced suppressed domain warnings when a different
missing covariate already excludes the same row under omission; matrix values
agree, but the warning differs from R. The subsequent
[R list data-column change](r-list-data-columns.md) resolves logical NA loss
during automatic list conversion. Inputs that are already evaluated R model
frames and stored-method argument overrides remain outside these checks.
These results do not establish full package parity.
