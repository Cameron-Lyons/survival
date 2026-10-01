# Formula evaluation and warnings

Formula transforms now run before subsets and missing-data removal. Previously,
`age + log(z)` could silently omit a row with both missing age and negative z,
without issuing stock R's domain warning. Training subsets and missing weights
could hide the same warning. The call now evaluates each formula variable once,
then carries its values through row selection, fitting and fresh matrix or
prediction construction. Removed terms still evaluate their variables and
signal their warnings.

Omission scans evaluated variables rather than raw sources. For example,
`I(z^0 * w)` can retain a row with missing z because `NA^0` is one. An ordinary
term that also reads z still requires it. Numeric transforms reuse their float
results, and row selection carries the cache without repeating domain checks.
The former separate scan for newly created NaNs is removed. Identity wrappers
retain logical values for factor coding; source NA and genuine NaN remain
distinct. Overflowing exp produces Inf, as in R, without raising a Python
overflow error on a row that is later omitted.

The R bridge captures prior Python warnings even when the call raises an error.
It signals those warnings through R's condition system before rethrowing the
original exception. Positional formula arguments and named model-method
arguments survive the shared path. Direct Python calls retain their normal
exception behavior and warning filters.

## Independent validation

Unmodified survival 3.8-12 produces 56 Cox/AFT reference models across 14
formulas and automatic or Unicode row labels. Four missing-data actions and
13 new-input variants yield 2,912 matrix outputs, including 484 expected
failures. Another 56 training calls cover overlapping covariate/weight omissions,
repeated subsets and removed transforms; 44 fits succeed and 12 fail as expected.
Whole coefficients, covariance matrices, matrix values, dimensions, names,
assignments, contrasts, strata and separate NA/NaN masks are compared. Warnings
are checked on successful and failing calls. List, NumPy and pandas containers
add 840 Python reference tests; two bridge tests cover warning/error transport
and positional calls.

The stock fixture regenerates byte-for-byte. Its SHA-256 is
`da075f4411d1e738efdedfebbd7cae7aea433ddea128cc9ccd0a6d8266cbd5f9`.
Only convergence-warning whitespace and the domain-warning expression suffix
are normalized. Expected output labels and missing-value kinds are preserved;
formula inputs use R's canonical spacing for arithmetic labels. Assignment
indices retain the established integer convention.

Live R tests also compare 144 prediction calls with standard errors across
both families, three formulas, three new inputs, four actions and two prediction
types. Six stock AFT term calls fail when omission leaves zero rows; the port
returns correctly shaped empty matrices or missing-value padding under exclude.
These stock failures are checked explicitly, with unchanged warning expectations.
Sixteen curve, log-rank, concordance and risk-set-weight calls compare selected
outputs and warnings before subsets and errors.

The full Python suite passes 35,253 tests, with 48 skips and 37 documented
expected numerical differences. Pinned Ruff 0.16.9 lint/format, the 47-file
type check and generated stubs/manifest checks pass. The R source archive
matches all 56 R source/test files and passes 217,896 checks with zero errors,
warnings or notes, adding 24,909 checks over #713.

Four transformed models written by #713 and four fresh transformed models
restore in separate R processes with identical stored matrices and predictions.
Fresh matrices then match whole stock outputs and warnings under all four
actions, including warnings before fail errors. Eight earlier ordinary/null
models also restore unchanged stored outputs and stock list matrices. Sixty
public list/data-frame calls match stock values, metadata and warnings after
their source columns are repeatedly mutated. Older frames without the new
cache slot remain readable.

Rust sources, Cargo inputs, native APIs and the release extension are unchanged
from #707. Native gates remain that increment's validation and were not rerun.

## Complete-call measurements

`scripts/benchmark_formula_evaluation.R` compares #713 (`ee15307e`), this change
and stock R in separate processes and libraries with the same release extension.
Stock outputs are captured before loading the bridge's registered S3 methods.
Fits have 20,000 rows, 16 transformed numeric covariates (eight log and eight
sqrt), one logical covariate, five strata and a log offset. New inputs have
10,000 rows; incomplete inputs alter every 59th row. Whole values, attributes,
NA/NaN masks and warnings are verified before timing. The predecessor's lost
domain-warning calls are verified but not timed.

Three warmups precede nine samples per workload. Runs execute sequentially
after validation. Timing includes public input conversion, formula evaluation,
omission, matrix construction, metadata and warning capture; fitting, input
setup, option changes and explicit GC are excluded.

Milliseconds, median (minimum–maximum), measured on an Intel Core Ultra 5 325
with Python 3.14.7, NumPy 2.4.6, R 4.5.3, survival 3.8-12 and reticulate 1.47.0.
The unchanged release extension was built with Rust 1.94.0.

| Family | Call | #713 | This change | Stock R |
| --- | --- | --- | --- | --- |
| Cox | Stored | 35 (32–45) | 36 (34–48) | 0 (0–0) |
| Cox | Complete data frame | 34 (34–35) | 35 (34–36) | 3 (3–4) |
| Cox | Complete list | 33 (33–34) | 34 (33–34) | 3 (3–4) |
| Cox | Source NA data frame | 38 (38–39) | 39 (39–40) | 3 (3–4) |
| Cox | Source NA list | 36 (36–37) | 38 (37–38) | 3 (3–4) |
| Cox | Overlapping domain/NA | Incorrect warning; not timed | 41 (39–42) | 3 (3–4) |
| AFT | Stored | 39 (37–51) | 40 (37–52) | 0 (0–0) |
| AFT | Complete data frame | 32 (31–32) | 32 (32–33) | 2 (2–3) |
| AFT | Complete list | 30 (30–31) | 31 (30–32) | 2 (2–2) |
| AFT | Source NA data frame | 34 (33–34) | 36 (35–37) | 2 (2–3) |
| AFT | Source NA list | 33 (33–34) | 34 (33–46) | 2 (2–3) |
| AFT | Overlapping domain/NA | Incorrect warning; not timed | 37 (36–38) | 2 (2–3) |

Comparable new-call medians increase by 0–2 ms in this workload. Several
ranges overlap, while the AFT source-NA frame range is higher. Stored ranges
overlap. No speedup is claimed, and stock remains faster, with stored results
below this timer's 1 ms resolution. Correct overlapping-domain calls omit
170 rows and emit one warning; the predecessor omits those rows but loses the
warning. These measurements cover the stated numeric/logical transformed
designs, not every formula or penalty family.

```sh
# Select each implementation's R_LIBS and PYTHONPATH before its run.
Rscript scripts/benchmark_formula_evaluation.R 20000 9 predecessor
Rscript scripts/benchmark_formula_evaluation.R 20000 9 current
Rscript scripts/benchmark_formula_evaluation.R 20000 9 stock
Rscript scripts/generate_formula_warning_reference.R
```

The broader compatibility and performance audit remains open. A fresh probe
confirms that an already evaluated R model frame containing `age` and `log(z)`
is accepted by stock Cox model.matrix, while the port still asks for raw z.
Stored-method argument overrides, response-transform warnings in prediction,
general nested/dimensioned inputs and other formula/penalty families require
further checks. These results do not establish full package parity.
