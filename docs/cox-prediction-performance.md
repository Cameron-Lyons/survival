# Cox predictions at requested times

`CoxPHFit.predict_survival_at(times, newdata=None, new_strata=None,
new_offset=None)` returns a NumPy array with shape `(n_times, n_observations)`.
Omitting `newdata` predicts for the original training rows. New observations
must supply their strata when the model has more than one stratum. Requested
times may be unsorted or repeated, and an empty time grid returns an array
with zero rows.

The Rust counterpart is `CoxPHFit::predict_survival_at(&times, newdata)`,
returning `SurvivalResult<Array2<f64>>`. Both use the default Cox survival
estimate: `stype = 2`, with the hazard type selected by the fitted tie method.
Use `survfit` for other estimates, confidence intervals, or individual curves
whose covariates change over time.

`CoxPHEstimator.predict_survival_function` and the R-style `brier` call this
method internally. The estimator keeps its `(n_observations, n_times)` output
orientation; `brier(detail=True)` keeps its nested-list `phat` result.

## Algorithm

Previously these Python calls expanded the entire Cox survival curve and
cumulative hazard for every observation, copied survival to Python lists,
and selected the requested times afterward. With `n` observations and `u`
unique follow-up times, the intermediate matrices required O(nu) memory and
work even for a small prediction grid.

The new method reads the cached per-stratum baseline at each requested time
and expands only those probabilities. It groups observations by stratum once
and writes predictions in the caller's row order. For `m` requested times and
`s` represented strata, curve lookup takes O(sm log u) and expansion takes
O(nm). The output uses O(nm) memory, plus the existing baseline cache, centered
design matrix and row indices. Asking for every follow-up time still requires
the full output matrix.

The calculation preserves `exp(-H).powf(risk)` from `survfit`, including its
floating-point underflow behavior. It also preserves step values at ties,
survival of 1 before the first time, extension after the last time, offsets,
aliased coefficients and stratum ordering. The Python binding releases the
GIL and returns the Rust matrix as a NumPy array without a list conversion.

The Brier kernel caches censoring survival at subject times and looks it up
once per evaluation time. Its IPCW weighting loop therefore takes O(nm)
instead of O(nm log u), with O(n) additional memory. The existing weighting,
tie shifts, Efron null model and behavior when censoring survival reaches zero
are retained. The binding accepts NumPy matrices and strided arrays directly.

## Reproduce the benchmarks

Build the extension in release mode, then run:

```sh
PYTHONPATH=python .venv/bin/python scripts/bench_cox_predictions.py
cargo bench --bench survival_benchmarks -- coxph_survival_at_requested_times
cargo bench --bench survival_benchmarks -- brier_score
```

The Python benchmark reports raw timing samples, medians, versions and an
extension SHA-256. It excludes fitting, input construction and a warmup call,
and measures the complete Python calls, including boundary conversion. Run
the same command against each build to compare them. These timings measure
this implementation before and after the change, not performance against R.

### Local comparison

On an Intel Core Ultra 5 325, Linux x86-64, Python 3.14.7 and NumPy 2.4.6,
the release builds produced these median times over five warmed calls with
3,000 observations and 64 requested times:

| Call | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| `CoxPHEstimator.predict_survival_function` | 471.93 ms | 1.24 ms | 381× |
| R-style `brier` | 341.41 ms | 4.79 ms | 71× |
| Brier kernel with precomputed predictions, list inputs | 2.87 ms | 2.46 ms | 1.17× |

The main gain comes from avoiding full curve expansion and its Python
conversion. Kernel-only timings are much smaller and vary with scheduling;
the four-time, 3,000-observation list-input kernel case measured 0.97 ms
before and 1.25 ms after. A small time grid does not guarantee faster
kernel-only calls, although the complete Brier call for that case fell from
360.80 ms to 2.69 ms.

Regression tests compare requested-time predictions with full `survfit`
curves across Breslow, Efron and exact fits, including counting-process data,
case weights, offsets, interleaved strata and repeated times. Existing R
fixtures check Brier scores and estimator predictions; a separate direct IPCW
calculation checks the cached weights in sequential and parallel execution.
