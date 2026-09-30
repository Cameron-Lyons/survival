# Formula subsets across R and Python

R-facing model calls now evaluate `subset` in the data and formula environment,
then translate R's selected rows into the shared Python model frame. Previously,
`subset = c(1L, 3L, 5L)` selected the second, fourth and sixth rows, ordinary
numeric selectors were rejected, and expressions such as `subset = age > 60`
could fail to find the data column.

The translation covers positive, negative and fractional numeric selectors,
repeated and reordered rows, recycled logical selectors, and data-frame row
names. Zero entries select nothing. Missing indices and unmatched row names
create missing rows that reach `na.omit`, `na.exclude`, `na.pass` or `na.fail`.
Mixed positive and negative selectors and out-of-bounds numeric positions raise
R's indexing errors. An empty selection raises the fitting interface's no-rows
error. A scalar selector remains a one-element sequence across the R bridge.

This applies to `coxph`, `survreg`, formula and direct-response `survfit`,
`survdiff`, `rttright`, `survcondense`, formula `concordance`, legacy
`survConcordance`, and direct-input `pyears`. The other formula wrappers already
use R's `model.frame` selection and do not receive a second translation.
Formula strings use the calling environment for subset expressions; R formula
objects use their own environment, with data columns taking precedence.

Python-facing `subset` remains zero-based and requires integer positions or a
full-length Boolean mask. Negative positions and floating-point positions are
rejected. Only the private R conversion can request a missing selected row.
For example, R's `subset = c(1L, 3L)` corresponds to Python's `subset=[0, 2]`.

The shared formula path constructs a survival response before selecting rows,
as R does. This preserves status normalization: selecting only censored rows
from a response encoded as 1/2 must not reinterpret its remaining 1s as events.
The normalized response is selected with the data and reused after missing-row
removal. Its type, multistate labels and interval endpoints survive selection.
Matrix covariates keep their column dimension when a missing row is inserted;
factor metadata, aligned arguments and evaluated strata follow the same rows.

This change does not resolve the documented AFT empty-scale difference: unused
strata are still omitted. Response normalization during omission without an explicit subset and in R
native model frames is covered by the subsequent [normalization correction](response-normalization.md).
R formula evaluation without a data argument remains under review.

## Verification

The Python regressions cover lists, NumPy arrays and pandas frames, missing-row
actions, repeated selections, caller ownership, serialization, response coding,
counting and interval responses, multistate labels and matrix covariates.
The R comparisons cover weighted Cox and AFT coefficients and covariance,
excluded predictions, grouped curves, log-rank tests, both concordance APIs,
redistribution weights, condensed intervals, scalar selections and person-years.
Reference formulas use `survival::Surv` wherever the reference API permits it,
so the attached compatibility layer cannot replace the response constructor.

All 20 new Python cases and 163 new R checks pass. The full Python suite passes
14,476 tests, with 48 skips and 37 documented expected differences. The final R
source archive passes 7,234 checks with zero errors, warnings or notes. Pinned
Ruff lint/format, generated interfaces and Mypy across 47 source files pass.

## Performance

`scripts/benchmark_formula_subsets.R` validates results against R, warms each
path three times and alternates seven complete calls. A logical mask selecting
37,468 of 50,000 rows also works correctly in the previous version, so it gives
a valid baseline. Numeric selectors with incorrect previous results are not
timed. Input construction and explicit garbage collection are excluded; row
selection and fitting or scoring are included.

Two local runs with R 4.5.3, survival 3.8-12, Python 3.14.7 and the same release
extension produced these medians and ranges in milliseconds. The previous
source is `c5e193ae`; its run was separate from the current run.

| Complete weighted call | Previous | Current | Stock R, current run |
| --- | ---: | ---: | ---: |
| Cox fit | 90 (89–92) | 103 (99–106) | 93 (89–94) |
| AFT fit | 79 (78–81) | 93 (92–93) | 68 (67–69) |
| Kaplan–Meier curve | 63 (62–72) | 74 (72–75) | 1,507 (1,383–1,550) |
| Formula concordance | 84 (82–93) | 87 (86–95) | 61 (59–64) |

Preserving the full response before selection adds work: these calls are slower
than the previous implementation. The current curve remains faster than R in
this weighted workload, while the other current calls are slower than R.
These are local measurements, and peak memory was not measured. No Rust
numerical kernels changed.

```sh
RETICULATE_PYTHON="$PWD/.venv/bin/python" PYTHONPATH="$PWD/python" \
  Rscript scripts/benchmark_formula_subsets.R 50000 7
```

The script accepts an optional third argument naming another source R package;
set `PYTHONPATH` to that version's Python sources when comparing revisions.
