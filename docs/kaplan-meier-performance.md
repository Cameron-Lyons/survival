# Survival curve performance

The Rust Kaplan–Meier engine uses a linear variance sweep for right-censored
data with one observation per cluster when influence matrices are not requested.
This includes the usual weighted curve: fractional weights select robust
variance automatically. Sorting still costs O(n log n); the variance sweep uses
O(n) time and O(n) auxiliary memory instead of visiting every observation at
every event time.

While an observation remains at risk, its influence is its case weight times a
shared coefficient. After it exits, its hazard influence is constant, and its
survival influence is multiplied by subsequent product-limit factors. The
implementation tracks the squared influences of departed observations and a
suffix sum of squared risk weights. Suffix sums avoid cancellation when the
risk set becomes small, and weight normalization avoids overflowing squares.
The Nelson–Aalen and Fleming–Harrington derivative formulas are shared with the
general influence kernel.

Repeated clusters, counting-process observations and explicit influence
matrices continue through the general calculation. Tests compare the optimized
standard errors and confidence bands with that calculation across ties, strata,
reverse curves, zero weights, and both survival and hazard estimators. The
existing R-generated differential fixtures also exercise the public APIs.

The Python bindings for `survfitkm` and `nelson_aalen` follow the crate's GIL
policy (see "Releasing the GIL" in `docs/repo-layout.md`): array inputs are
copied into owned Rust buffers by the shared checked converters, and the fit
runs with the GIL released.

## Local benchmark

Run against a release extension:

```sh
maturin develop --release --features extension-module,ml
PYTHONPATH=python python scripts/bench_survival_curves.py --repeats 7
cargo bench --bench survival_benchmarks -- kaplan_meier
```

The Python script reports extension hashes and individual timings. Input setup,
one warmup, and disposal of the previous result are excluded; argument
conversion and result construction are included. Times are `1..n`, every third
observation is censored, and weights cycle through `0.5 + (i % 7) / 4`.

A local comparison with the pre-change release extension (Linux x86-64,
CPython 3.14, seven measured calls) gave these weighted Kaplan–Meier medians:

| Observations | Before | After | Speedup |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.64 ms | 0.099 ms | 17× |
| 10,000 | 200 ms | 1.74 ms | 115× |
| 30,000 | 1,276 ms | 4.97 ms | 257× |

These are measurements of this workload, not general speed guarantees. The
optimization does not change the cost of returning a full influence matrix.

## Aalen–Johansen uncertainty

Independent right-censored competing-risk curves use a moment sweep when
all observations in a curve start in the same state, their initial influences
are zero, and explicit influences are not requested. A fixed initial
distribution may put probability in other states. Ties, nonnegative case
weights, source self-transitions, differing source states across strata, and
conditional start times are supported.

Rows still at risk share weight-scaled probability, cumulative-hazard, and
integrated-occupancy influence coefficients. Departed rows contribute small
triangular QR factors: two columns for the source state and three for each
destination. Event and integration updates preserve triangularity. Adding
an exiting row uses Givens rotations and `hypot`, so the calculation does not
subtract nearly equal variance terms or explicitly square extreme weights.
Backwards risk-weight norms avoid cancellation when a heavy row exits.

After sorting, the kernel takes O((n + t)(s + h)) time, with O(n + s + h)
auxiliary storage, where t, s, and h denote reporting times, states, and
transition-hazard columns. The returned matrices still take O(t(s + h))
storage. Counting-process data, repeated clusters, uncertain initial
distributions, and explicit influences retain the general influence kernel.
Numeric overflow, vanished normalized positive weights, zero-weight events
on empty weighted risk sets, and rounding-sensitive absorbed confidence
limits also retain the general calculation and its missing-value behavior.

Grouped initial-state preparation scans the global minimum entry time once
for counting-process data, rather than once per stratum. Right-censored data
skip that scan. The global minimum is retained because it determines whether
subjects entering exactly at the initial time are included.

Run complete native-call measurements against each release extension:

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_aj_variance.py --repeats 7
cargo bench --bench survfitaj_benchmarks -- standard_errors_long_grid
```

The Python benchmark includes argument conversion and result construction,
validates probabilities, hazards and counts, and compares all three error
matrices with explicit influences in independent and repeated-cluster controls.
Its variance workload uses times 1..n and cycles through two event types and
censoring. Weighted cases cycle through `0.5 + (i % 7) / 4` and censor the
final row. Preparation uses two subjects per curve with different initial
states, with both right-censored and counting-process controls.

Local release-extension medians on Linux x86-64, CPython 3.14.7, NumPy 2.4.6,
with seven measured calls and default standard errors:

| Observations | Weights | Before | After | Speedup |
| ---: | :--- | ---: | ---: | ---: |
| 1,000 | Unit | 5.13 ms | 0.974 ms | 5.3× |
| 4,000 | Unit | 77.1 ms | 3.70 ms | 20.8× |
| 8,000 | Unit | 306.6 ms | 7.25 ms | 42.3× |
| 1,000 | Fractional | 5.43 ms | 1.14 ms | 4.8× |
| 4,000 | Fractional | 77.6 ms | 4.30 ms | 18.1× |
| 8,000 | Fractional | 308.2 ms | 8.53 ms | 36.1× |

For 16,000 rows in 8,000 curves, inferred initial-state preparation fell from
94.7 to 14.4 ms on right-censored data and from 96.7 to 16.5 ms on
counting-process data. Fixed-`p0` controls stayed near 14 and 16 ms.
These measurements describe the specified workloads. Returning full
influences and numerically sensitive inputs still uses the general kernel.

## Grouped multistate tables

Grouped `as_data_frame` output is state-major, then stratum-major, matching R.
Each curve provides one contiguous time block per state column; assembly
copies those blocks by position. Repeated or reordered state selections
therefore retain the requested columns even when labels repeat. Conversion
scales with the output size rather than rescanning every state label for
each state. Groups may have different reporting-time grids.

To compare complete conversion calls with an earlier Python implementation:

```sh
git show MAIN_REVISION:python/survival/r/_models.py > /tmp/previous_models.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_grouped_multistate_frames.py \
  --baseline-source /tmp/previous_models.py --repeats 7
```

This benchmark excludes fitting and checks every output column and its order.
Stock-R fixtures cover repeated and reordered state selections with and
without standard errors and initial rows. R's raw selected matrices supply
the reference: stock `[.survfitms` leaves `n.censor` unsliced, which can break
`summary(..., data.frame=TRUE)` after selecting states. The fixture selects
the corresponding original censor columns explicitly.

The same local environment gave these seven-call conversion medians for
16 groups with 200–202 reporting times each:

| State columns | Output rows | Before | After | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 28,935 | 14.6 ms | 5.63 ms | 2.6× |
| 33 | 106,095 | 114.7 ms | 26.0 ms | 4.4× |
| 129 | 414,735 | 1,351.8 ms | 109.7 ms | 12.3× |
