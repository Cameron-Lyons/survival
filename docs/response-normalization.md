# Response normalization before missing-row removal

Formula preparation now constructs `Surv` before missing-row omission and
uses the normalized response to determine which rows are missing. With status
codes `[1, 2, 1, 2]`, removing the second and fourth observations must leave two
censored observations. Previously, rebuilding `Surv` after omission interpreted
the remaining `[1, 1]` as two events. Explicit subsets already preserved the
response; ordinary omission now follows the same path.

Cox, AFT, survival-curve, concordance, log-rank and data-preparation calls share
this rule. Invalid status codes become missing with the constructor's warning;
reversed counting intervals and invalid interval endpoints are handled by the
same constructor. An unused interval endpoint does not remove an observation,
but it still counts as missing when a right-hand-side term independently reads
that column. `na.exclude` retains the omitted-row positions for prediction and
residual padding, and `na.pass` retains normalized missing entries. AFT fitting
rejects those unknown statuses instead of silently turning them into censored
observations.

The response is evaluated once and selected with the other model-frame columns.
This removes duplicate counting-interval checks and the separate AFT/curve
endpoint exceptions. `survcheck` now uses the shared formula/response preparation
while preserving its mapping back to the original observation rows. Warnings
point to the first Python caller outside the package and are emitted once per
response evaluation, including when a subset later discards the invalid row.

R's native model-frame adapter also delegates to the shared response constructor.
It previously stored raw status codes without normalization. Numeric inputs and
normalized output columns now cross the boundary as arrays instead of individual
row values. Integer missing values are converted to floating NaN before array
conversion; nullable logical statuses retain their Boolean/missing semantics.
Scalar vectors, factor levels, multistate attributes, named arguments and
`difftime` inputs are preserved. Constructor diagnostics are signalled as R
warnings. Date-valued time inputs are rejected as in stock R.

This work does not establish full package parity. The documented AFT unused-scale
difference, R object persistence and the broader formula/completeness audit remain.
Timeline responses keep their existing, separate conversion path.

## Verification

`scripts/generate_response_normalization_reference.R` records 180 stock-R model
frames across omission, exclusion, passing and failing policies; ordinary,
reversed and repeated selections; status coding; invalid statuses; left/counting
responses; interval endpoints; weights; offset omission and multistate factors.
The same cases run with Python lists, NumPy arrays and pandas frames. Additional
regressions verify all-censored curves, `survcheck` row mapping, and ownership of
the arrays returned to R. Direct R comparisons cover every response type, scalar
inputs, integer and logical missing values, factor warnings, weighted Cox/AFT
fits, covariance and excluded predictions.

All 549 new Python cases and 204 new R checks pass. The full Python suite passes
15,025 tests, with 48 skips and 37 documented expected differences. The R source
archive passes 7,438 checks with zero errors, warnings or notes. The 180-case
reference regenerates byte-for-byte. Pinned Ruff lint/format, generated
interfaces and Mypy across 47 source files pass.

## Performance

`scripts/benchmark_response_normalization.R` validates outputs, warms each path
three times, alternates seven calls against stock R, and includes response
preparation and fitting. Input setup and explicit garbage collection are outside
the timed calls. The previous package (`1f5cb7d9`) ran in a separate process with
the same release extension. Baseline data use 0/1 status codes, so the previous
results are valid; the incorrect 1/2 omission case is not timed.

Local measurements with 50,000 observations, R 4.5.3, survival 3.8-12 and Python
3.14.7 produced these medians and ranges in milliseconds:

| Complete R-facing call | Previous | Current | Stock R, current run |
| --- | ---: | ---: | ---: |
| Construct a right-censored response | 19 (18–20) | 1 (1–2) | 1 (1–1) |
| Native model frame with missing predictors | 4 (3–4) | 6 (6–7) | 4 (4–4) |
| Weighted Cox fit with missing predictors | 124 (121–125) | 118 (117–120) | 108 (106–109) |
| Weighted interval-censored AFT fit | 248 (246–249) | 233 (233–236) | 96 (91–96) |

Bulk conversion improves direct response construction. Correct normalization
adds work to native model frames. Current Cox medians varied from 118 to 126 ms
across runs, so this measurement does not establish a stable improvement. Interval AFT avoids
repeated response construction but remains slower than R. The timer resolves
milliseconds, so the shortest calls have coarse measurements. These are local
results, and peak memory was not measured. No Rust numerical kernels changed.

```sh
RETICULATE_PYTHON="$PWD/.venv/bin/python" PYTHONPATH="$PWD/python" \
  Rscript scripts/benchmark_response_normalization.R 50000 7
```

The optional third argument names another source R package. Set `PYTHONPATH` to
that version's Python sources when comparing revisions.
