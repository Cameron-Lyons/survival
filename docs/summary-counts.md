# Survival-summary count selection

`r.summary_survfit` and `r.model_summary` accept R's `dosum` argument for
Kaplan–Meier, Turnbull, Cox and multistate curves:

```python
from survival import r

fit = r.survfit(r.Surv([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]))
intervals = r.summary_survfit(fit, times=[2, 4], dosum=True)
at_times = r.summary_survfit(fit, times=[2, 4], dosum=False)
```

With `dosum=True`, event, censoring and entry counts accumulate between
requested times, including all preceding observations in the first count.
With `False`, they come from the last curve row at or before each requested
time. Survival probabilities, cumulative hazards and uncertainty use that
same step-function row in both modes. Risk counts use the first row at or
after the requested time and become zero past the end of the curve.

Ordinary curves preserve query order and duplicates. Their default accumulates
counts only for strictly increasing requested times; explicitly requesting
accumulation for unordered or repeated times raises R's error. Multistate
curves always sort and deduplicate requested times and default to accumulation.
Their transition counts follow the same selection. Without requested times,
`dosum` has no effect. Existing `censored`, `extend`, `scale` and `rmean`
options retain their behavior.

## Rust and native Python interfaces

Rust callers can choose counts with
`surv_analysis::summary_survfit_times_with_counts(fit, times, extend, dosum)`
where `dosum: Option<bool>` selects accumulation, lookup, or the ordinary
default. Multistate callers use
`summary_survfit_aj_with_counts(fit, times, censored, extend, dosum)` with a
boolean `dosum`. Existing Rust functions keep their signatures and defaults.

The native Python `surv_analysis.summary_survfit(..., dosum=None)` and
`SurvfitAJResult.summary(..., dosum=True)` accept NumPy time vectors, including
strided arrays, through the checked bulk input converter. Both release the
GIL for numerical work.

## Algorithm and verification

Ordinary summaries build their two time indices with a forward sweep for
dense ordered queries. Sparse and unordered queries retain binary search.
The choice compares query count with the observed grid size divided by its
binary-search depth, so a handful of requested times does not scan a long
curve. Count accumulation streams prefix sums through the selected rows,
preserving R's floating-point addition order without allocating a full
additional prefix vector per count field.

`scripts/generate_summary_counts_reference.R` records 528 stock survival
3.8-12 comparisons across 22 fits: ordinary and multistate curves, Cox and
multistate Cox predictions, grouped/weighted fits, counting-process inputs,
extensions, scalar/ordered/repeated queries, and missing times. The Python
suite checks all recorded count, probability, hazard and uncertainty fields.
Stock R drops grouped multistate count-matrix dimensions when times are absent.
Those 24 cases retain the raw output alongside the intended per-column
accumulation computed from the original stock fit; the port preserves the
state and transition columns.
Native Rust tests also check interval indices against binary search with
tied observation times, repeated queries, negative times and empty grids.

Reproduce the references and timings with:

```sh
Rscript scripts/generate_summary_counts_reference.R /tmp/summary-counts.json
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_summary_counts.py -q
PYTHONPATH=python .venv/bin/python scripts/bench_summary_counts.py
cargo bench --bench survival_benchmarks -- curve_summary
```

The Python benchmark excludes fitting and reports native and full formula
summary times separately, with list and NumPy inputs. Rust benchmarks exclude
fit construction and cover sparse, dense and explicit lookup queries.

On the local Python 3.14.7 release extension, median native call times over
nine samples with list inputs were:

| Observed times | Requested times | Before, ms | After, ms |
| ---: | ---: | ---: | ---: |
| 10,000 | 10 | 0.070 | 0.024 |
| 10,000 | 10,000 | 0.613 | 0.548 |
| 100,000 | 10 | 0.883 | 0.514 |
| 100,000 | 100,000 | 8.941 | 8.843 |
| 300,000 | 10 | 3.476 | 2.719 |
| 300,000 | 300,000 | 24.153 | 16.879 |

The 300,000-query NumPy call took 14.603 ms after the change. These measurements
include input conversion and result construction, exclude fitting and Python
formula summary-table assembly, and vary with allocation and system load.
