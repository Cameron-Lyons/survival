# AFT model-matrix inputs

AFT `model.matrix` evaluates the covariate terms after removing standalone
scale strata. A new frame can omit those strata columns, contain missing values
in them, or supply different group labels without changing the matrix. The
previous implementation used prediction preparation, requiring the columns,
checking their fitted levels and unnecessarily omitting incomplete strata rows.

Strata used in covariate interactions remain required. A variable used both in
`strata(g)` and as an ordinary covariate still contributes to omission, factor
validation and the matrix. Offset variables still belong to the model frame and
affect omission despite having no matrix columns. Formula cluster variables and
survival responses are not needed. Intercept-only matrices preserve the row
count and labels of frames with zero columns.

Prediction continues to evaluate scale strata and offsets. Stored fits retain
their original matrices and omitted rows. Public Python matrices retain their
list data and dictionary keys; the private R metadata path retains row names,
contrasts and the established integer term assignments. The stock AFT fitter's
term removal and contrast construction are visible in
[its source](https://github.com/therneau/survival/blob/master/R/survreg.R).
The installed `model.matrix.survreg` method was also inspected directly.

R model-matrix calls now use the existing warning capture bridge, so formula
domain warnings become R conditions on every call instead of Python stderr.
Python warnings retain their expression context, such as `NaNs produced in
log(off)`; comparison with R's `NaNs produced` removes only that suffix.
Validation failures compare the missing-column or factor-level cause. Stock's
additional warning before rejecting a numeric replacement for an ordinary
factor is retained in the fixture but is not reproduced by the port.

## Independent checks and persistence

The fixture contains 216 unmodified stock-R fits across 18 formulas, complete
and incomplete training frames, repeated subsets, automatic and Unicode row
names, and both cached and reconstructed matrices. Twenty-one input variants
provide 4,080 whole matrices and 456 validation failures. They cover individual
and combined strata, ordinary/logical/categorical covariates and interactions,
cluster terms, numeric and transformed offsets, ridge terms, absent or changed
groups, missing covariates/responses, empty/single-row frames and zero-column
frames. Regeneration is byte-identical.

List, NumPy and nullable pandas columns run all 216 cases, adding 648 Python
tests; a separate prediction control adds one test. Live R checks compare the
same whole outputs. Matrix values, dimensions, row/column names, assignments,
contrasts and successful-call warnings are checked. Existing logical and Cox
matrix suites also pass. Only convergence-warning whitespace and the specified
domain-warning suffix are normalized; numerical comparisons are unchanged.

The full Python suite passes 33,449 tests, with 48 skips and 37 documented
expected differences, adding 649 tests over #710. Pinned Ruff 0.16.9 lint/format,
Mypy across 47 source files and generated stubs/manifest checks pass. The R source
archive matches all 51 R source/test files and passes 129,910 checks with zero
errors, warnings or notes, adding 29,235 checks over #710.

Four #710 AFT models restore with identical stored matrices and complete
predictions, and now construct reduced-input matrices matching stock R. Four
fresh AFT models pass the same checks after restoration in separate R
processes. Four previously saved logical/expanded #710 models also restore
identical complete matrices and attributes. Stock reference assignments are
cast to the bridge's existing integer convention for these comparisons.

Rust sources, Cargo inputs, native APIs and the release extension are unchanged
from #707. Its Rust gates remain the latest native validation and were not rerun.

## Complete-call measurements

`scripts/benchmark_aft_model_matrix.R` compares #710 (`f1f9c8d7`), this
implementation and stock R in separate R/Python libraries using the same release
extension. An AFT model has 20,000 fitted rows, 16 numeric covariates plus an
intercept and ten scale strata from two factor variables. New frames have
10,000 rows, with and without the standalone strata columns. Whole matrix
values, dimensions, names, assignments and contrasts are verified before timing.

Each complete public call includes R/Python conversion, matrix construction
and metadata, excluding fitting, input setup and explicit GC. Each workload has
three warmups and nine samples. Runs execute sequentially after validation.
The environment is an Intel Core Ultra 5 325, Rust 1.94.0, Python 3.14.7,
NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.

| Complete matrix call | #710, ms | Current, ms | Stock R, ms |
| --- | ---: | ---: | ---: |
| Stored, 20,000 rows | 41 (38–51) | 39 (36–48) | 0 (0–0) |
| New frame with groups, 10,000 rows | 25 (24–25) | 20 (19–20) | 2 (1–2) |
| New frame without groups, 10,000 rows | Rejected | 18 (17–19) | 2 (1–2) |

The new-frame median with groups is 20% lower in this workload, with disjoint
observed ranges. Stored-call ranges overlap, so no stored-matrix speedup is
established. Removing unused groups enables previously rejected frames and
also reduces input conversion. Stock R remains faster. These measurements
do not establish a general speedup; the timer resolves about one millisecond.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_aft_model_matrix.R 20000 9 predecessor
Rscript scripts/benchmark_aft_model_matrix.R 20000 9 current
Rscript scripts/benchmark_aft_model_matrix.R 20000 9 stock
Rscript scripts/generate_aft_matrix_input_reference.R
```

The complete #646 review-index history through #710 is preserved verbatim in
[the archived index](pr-review-history-through-710.md), including all historical
validation and performance evidence. Its 64,738 characters have SHA-256
`a79f8d8de139bdaa5ab6f194352f854e8dcf85240d825ea835d8dfd40d13cfe4`.
The live index links this immutable snapshot to remain within GitHub's body limit.

The broader compatibility and performance audit remains open. The subsequent
[missing-data correction](model-matrix-na-action.md) honors global R `na.pass`
and `na.fail` options and retains incomplete matrix rows under `na.pass`.
Inputs that are already evaluated R model frames remain outside these checks.
These results do not establish full package parity.
