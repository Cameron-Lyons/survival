# Concordance preparation and zero-weight events

The Rust `concordancefit` engine prepares each stratum's outcome and case-weight
buffers, entry-time order and event-time multipliers once for all predictor
columns. Later predictors reuse the exit-time order and sort only groups tied
on both exit time and event status. Each predictor still builds its own rank
tree, accumulates its own pair counts and influences, and contributes to the
joint covariance matrix. When event ranks are requested, their ascending event
times are prepared once too.

The first predictor establishes the common risk-sum accumulation order. This
preserves one-column numerical behavior; other columns can differ from the
previous independently prepared time multipliers by floating-point rounding.
Equal predictor values retain original observation order, including the order
of their case weights in event-rank results. Delayed entry, strata, all supported
time multipliers, horizon limits, reversed predictions and influence requests
use the same preparation.

This removes repeated outcome allocations, full outcome sorts and Kaplan-Meier
time-weight calculations. Predictor ranking and risk-tree sweeps still scale
with the number of predictors; constructing a full joint covariance matrix
also retains its cost. Datasets where every event time is tied still require a
predictor sort for every column.

A zero-weight event with an empty weighted risk set now contributes zero to
the Cox variance numerator. Previously, times `[1, 2, 3]`, all events,
predictor `[1, 2, 3]` and case weights `[1, 1, 0]` returned `cvar = NaN`
through a `0/0` term. It now returns `0.25`, matching deletion of the zero-weight
observation. Results without any comparable weighted pairs continue to have
undefined concordance and variance.

## Verification

Rust tests compare joint results with independent one-column fits across all
time multipliers, delayed entry, fractional and zero weights, differing tied
predictor orders, signed-zero predictors, near-tied times, horizon clipping,
strata without events, clustered influence rows and reversed predictions.
They also reconstruct cross-predictor covariances from independent influence
vectors and compare event-rank case weights in their returned order. Existing
R reference tests cover numerical pair counts, variance, influences and ranks.

## Benchmarks

`benches/concordance_benchmarks.rs` measures Rust counts, weighted variance and
counting-process influence/rank results at 20,000 observations with 1, 8 and 32
predictor columns:

```sh
cargo bench --bench concordance_benchmarks
```

`scripts/benchmark_multi_concordance.py` measures native Python calls with
fractional weights, unsorted tied event times and tied predictor values. Input
construction, three warmups and disposal of the previous result are excluded;
argument handling, numerical fitting and result construction are included. It
records individual samples and the native extension hash, and checks numerical
agreement with a saved baseline before reporting speedups. Before and after
versions run in separate processes; the measurements do not alternate versions.

```sh
maturin develop --release --features extension-module,ml
PYTHONPATH=python .venv/bin/python scripts/benchmark_multi_concordance.py \
  --output /tmp/concordance-before.json
# Rebuild after applying the change, then:
PYTHONPATH=python .venv/bin/python scripts/benchmark_multi_concordance.py \
  --baseline /tmp/concordance-before.json --output /tmp/concordance-after.json
```

A local Linux x86-64 release-build comparison with Python 3.14.7, 20,000
observations and nine measured calls produced these medians and sample ranges
in milliseconds. The extension hashes are recorded below.

| Predictors | Time multiplier | Variance | Previous median (range) | Current median (range) | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `n` | No | 1.967 (1.892–1.994) | 1.981 (1.945–2.021) | 0.99× |
| 1 | `S` | No | 1.979 (1.902–2.015) | 2.048 (2.032–2.099) | 0.97× |
| 1 | `n` | Yes | 3.002 (2.941–3.109) | 3.097 (3.044–3.199) | 0.97× |
| 1 | `S` | Yes | 2.934 (2.806–3.040) | 3.049 (2.959–3.098) | 0.96× |
| 8 | `n` | No | 10.471 (10.407–10.561) | 5.337 (5.324–5.409) | 1.96× |
| 8 | `S` | No | 9.816 (9.795–9.896) | 5.270 (5.253–5.364) | 1.86× |
| 8 | `n` | Yes | 22.719 (22.183–23.140) | 17.193 (16.958–17.634) | 1.32× |
| 8 | `S` | Yes | 22.326 (21.737–22.613) | 17.321 (16.876–17.456) | 1.29× |

The one-column controls are approximately 1–4% slower in this run. The benefit
is sharing preparation across several predictors; these results do not show
a one-column speedup. All reported concordance, influence-based variance and
Cox-model variance values agree with the saved baseline within `rtol=1e-12`
and `atol=1e-14`. Timing and memory behavior depend on the workload; peak memory
was not measured.

The baseline extension SHA-256 was
`8a83b9ace4231f4f8fc7b78a8ab1be564cc5c1abfe4107962945c663014eb4fc`;
the updated extension was
`7f3a8f7110145923614cea91e482a076afbbb4e685f5276b33ccedf3f4a83335`.
