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

The Python bindings for `survfitkm` and `nelson_aalen` copy array inputs into
owned Rust buffers before releasing the GIL. Lists, NumPy arrays with arbitrary
strides, and pandas/polars columns use the shared checked converters. No Python
objects or borrowed NumPy buffers are accessed while the fit runs detached.

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
