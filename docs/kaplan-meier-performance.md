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

## Grouped residuals and pseudo-values

Ordinary and multistate residual preparation buckets input rows by curve in
one pass. This takes O(n + g) time and O(n + g) auxiliary storage for n rows
and g curves, replacing O(n g) repeated scans. Each bucket preserves input
order, including complete histories before a conditional start time. Subject
and cluster collapse retain their existing first-appearance order.

The Python pseudo-value endpoint check also walks stratum lengths once,
replacing repeated prefix sums. It uses the greatest normalized query time and
retains the warning when a requested time extends beyond any curve's endpoint.
The numerical residual and infinitesimal-jackknife formulas are unchanged.

Run the complete public-call benchmark against a release extension:

```sh
PYTHONPATH=python python scripts/benchmark_grouped_survfit_residuals.py \
  --samples 5 --output /tmp/grouped-residual-after.json
```

The workload has interleaved event/censor pairs, either one stratum per pair or
64 balanced strata. The initial formula fit is excluded; each measured call
includes formula-frame recovery, native refitting, residual or pseudo-value
calculation, and result materialization. Complete arrays are checked against
analytic results before timing. Reports include individual samples and the
native extension hash; `--compare` additionally checks exact output equality
with an earlier report's saved arrays. `--baseline-source` extracts the earlier
Python `pseudo` function so its endpoint work can be measured separately.

A local release comparison (Linux x86-64, CPython 3.14, NumPy 2.4.6,
five measured calls) gave these medians at 32,000 rows and 16,000 strata:

| Complete public call | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Ordinary residuals | 267 ms | 36.2 ms | 7.4× |
| Ordinary pseudo-values | 677 ms | 37.7 ms | 18× |
| Multistate residuals | 252 ms | 63.9 ms | 3.9× |
| Multistate pseudo-values | 720 ms | 128 ms | 5.6× |

All saved result arrays were exactly equal. With the same rows in 64 balanced
strata, ordinary pseudo-values took 11.7 ms before and 10.8 ms after; multistate
pseudo-values took 29.8 ms before and 28.6 ms after. These measurements describe
this workload; fitting and requested output size still contribute to each call.

Sixty cases generated by unmodified R survival 3.8-11 check ordinary and
multistate residuals and pseudo-values for probabilities, hazards, and integrated
occupancy. They include reversed factor levels, unused levels, fractional
weights, clusters, counting histories, conditional starts, and omitted rows;
both array output and long tables are checked. Regenerate them with:

```sh
Rscript scripts/generate_grouped_residual_reference.R /tmp/grouped-reference.json
```
