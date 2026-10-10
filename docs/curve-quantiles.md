# Survival curve quantiles

`survival.r_api.quantile` accepts an ordinary fitted survival curve or a raw
`Surv` response. `median` requests probability 0.5. Raw responses use the KM
or Turnbull fitter before inversion and include confidence bounds by default;
fitted-curve medians return point estimates. Grouped fits and Cox prediction
columns retain one result row per curve. Conditional Cox predictions use
their starting time as the curve origin.

The R bridge retains R's result dimensions: an unstratified single curve gives
a vector; a retained stratum dimension or multiple prediction columns gives a
matrix; retained strata with multiple prediction columns give a
stratum-by-curve-by-probability array.
Conditional Cox probability-zero quantiles retain `start.time`; ordinary KM
quantiles retain R's zero origin even when fitting starts later. The live R
regressions call stock methods directly from the survival namespace and check
values, dimensions, labels and warnings independently of bridge dispatch.
Ordinary grouped KM fits retain bare bridge group labels (`a`, `b` instead
of R's `g=a`, `g=b`); reference comparisons normalize this label difference.

When confidence bands exist, the R bridge evaluates `conf.int` with R's
conditional rules: numeric one requests bounds and zero requests point
estimates. Missing, empty or multiple conditions raise stock R's errors.
When bands are absent, the bridge returns point estimates without evaluating
the confidence argument, as R does. Independent R comparisons also retain
tolerance arithmetic, recycling, warnings and errors.
`test-curve-quantile-coercion.R` checks these rules in 1,859 expectations
across ordinary, grouped, counting-process and all-censored KM fits and
single, multiple and stratified Cox predictions, with and without bands.
The conditions include logical, numeric, character, factor, missing, empty
and list inputs; lazy arguments check which errors are evaluated first.

The R adapter preserves fitted confidence limits at survival zero. Valid zero
bounds from robust curves participate in inversion; undefined native bounds
remain missing. Full-column stock frame references cover 24 finite and infinite
endpoint cases through Python; the R infinity regressions independently compare
frame columns and curve quantiles with stock methods.

Probability requests follow R's numeric-vector contract. Numeric scalars,
lists, NumPy arrays and numeric pandas or Polars containers are accepted.
An empty numeric vector returns an empty probability dimension. Logical,
character, categorical and missing probabilities are refused, including
missing values inside nullable containers. Valid probabilities lie in [0, 1].
Repeated probabilities and the caller's ordering are retained.

The Python and native interfaces accept a numeric scalar `scale`, including
zero, negative, infinite and missing values. Scaling follows inversion using
IEEE arithmetic, as R does: a zero divisor gives an undefined result for a
zero time and signed infinities for finite nonzero times; an infinite divisor
gives signed zeros at finite times; a missing divisor gives undefined results.
A single-valued numeric container is also accepted by the Python facade.
Character scales are refused.
Summary tables retain their separate positive-scale requirement. R's implicit
recycling of multiple scale values and complex scale arithmetic are not
represented by the native scalar interface.

The default tolerance is the square root of machine epsilon. Finite and
infinite tolerances follow R's curve inversion. Positive infinity yields
undefined quantiles; negative infinity preserves the curve origin at
probability zero and yields undefined quantiles at positive probabilities.
NaN tolerance is refused. Multistate curve quantiles and medians are refused,
as in R.

The native `quantile_survfit` and `quantile_survfit_curves` bindings take owned
numeric arrays. Stacked time, survival, confidence-band and probability
vectors cross the Python boundary in bulk before the interpreter lock is
released for curve construction and inversion. Strided, negative-stride,
unaligned and read-only arrays retain their logical order. Returned values
retain their existing list getters.

`scripts/generate_curve_quantile_boundary_reference.R` records 3,240 stock
survival 3.8-12 calls: 3,042 successful results and 198 errors. It covers
weighted ordinary and grouped KM fits, counting-process data, all-censored
curves, interval censoring, confidence transforms, single and multiple Cox
predictions, conditional origins, empty and repeated probabilities, scale,
and nine tolerance choices. Of the successful calls, 676 use infinite
tolerances. The fixture retains R's original error text. Python regressions
compare both list and array requests and separately check nullable inputs,
array layouts, numeric dtypes, dimensions and concurrent Python-thread
progress. Regenerating the fixture reproduces its bytes.

The fixture also records 20 stock dtype oracles at empty and nonempty widths,
bringing that boundary set to 3,260 stock calls. A bounded table tests 35 Python input
types at both widths against those oracles:

| Container | Accepted numeric dtypes | Refused dtypes |
| --- | --- | --- |
| NumPy | `i`, `u`, `f`, and `O` containing numbers | `b`, `c`, `m`, `M`, `S`, `U`, `V` |
| pandas | Integer, float, nullable integer/float, and object columns containing numbers | Boolean, string, categorical, datetime and timedelta |
| Polars | Integer, unsigned integer, float and decimal | Boolean, string, categorical, Enum, date, datetime, duration, binary and null |

Nullable numeric columns remain valid without missing entries; numeric
columns containing missing probabilities are refused. Empty nullable
numeric columns retain the numeric-zero output shape. Object arrays keep
their numeric-value conversion, so this check preserves their existing
support while refusing declared nonnumeric empty vectors.

Another 1,092 stock calls record twelve scalar scales across every ordinary,
grouped, interval and Cox fit, with and without confidence bounds, empty
probabilities, and medians. The scale references preserve infinities and the
sign of zero rather than serializing them as missing values. Tests compare the
facade, prepared-fit binding and stacked-curve binding, including every point
estimate and bound. The complete fixture contains 4,352 stock calls.

## Complete-call measurement

`scripts/benchmark_curve_quantile_boundary.py` prepares inputs outside timing
and measures complete native calls, including argument conversion and every
result-list getter. It records medians, ranges, samples, result hashes and the
extension hash. A saved extension of the same Python ABI can be loaded with
`--extension`; `--prepared-fit` also measures probability conversion for an
already prepared curve. Its default inputs contain 100,000 and 500,000 rows
in list, contiguous, strided and negative-stride layouts, with and without
confidence bounds.

On Linux x86-64 with Python 3.14.7, nine measured calls after three warmups
gave the following medians and ranges in milliseconds. These calls request
the default three quartiles and both confidence bounds. All 64 measured
cases, including prepared fits and 1,001 probabilities, produced identical
before/after result hashes.

| Rows | Input layout | Previous median (range) | Current median (range) |
| --- | --- | --- | --- |
| 100,000 | Lists | 7.82 (7.64–7.91) | 8.32 (8.07–8.45) |
| 100,000 | Contiguous NumPy | 12.17 (11.97–12.40) | 5.89 (5.35–6.18) |
| 100,000 | Strided NumPy | 12.05 (11.89–12.37) | 5.77 (5.66–6.08) |
| 100,000 | Negative-stride NumPy | 13.69 (13.38–14.10) | 5.91 (5.69–6.02) |
| 500,000 | Lists | 40.55 (39.34–41.70) | 44.25 (41.37–52.96) |
| 500,000 | Contiguous NumPy | 68.36 (65.19–74.34) | 33.74 (32.35–34.53) |
| 500,000 | Strided NumPy | 67.90 (67.21–69.33) | 32.81 (32.11–33.93) |
| 500,000 | Negative-stride NumPy | 70.18 (69.36–77.19) | 33.16 (32.99–33.87) |

The NumPy cases improved by 51–57%. Lists were 6–9% slower in this run;
these results support the bulk-array improvement, without establishing a
speedup for every input path. List extraction still uses the same owned
`Vec<f64>` conversion, and curve construction, validation, allocation and
inversion retain their previous algorithms. The wrapper adds constant-size
array-type checks and interpreter-lock release around the kernel, with no
additional pass over list observations. Prepared-fit and many-probability
measurements varied across layouts; the script preserves all samples and
ranges so they can be assessed separately.

A second list-only control alternated the saved and current extensions in
one process, reversing their order on each iteration. After three warmups,
17 samples measured 7.40 ms (6.68–8.34) versus 7.53 ms (6.50–8.33) at
100,000 rows and 33.03 ms (31.96–37.36) versus 32.31 ms (31.34–35.20) at
500,000 rows. Those overlapping ranges did not reproduce a consistent list
regression; allocation state and measurement order remain relevant to the
first run's comparison.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_curve_quantile_boundary.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_curve_quantile_boundary.py \
  --extension /path/to/saved/_survival.so --prepared-fit --nprobs 3 1001
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_curve_quantile_boundaries.py -q
```
