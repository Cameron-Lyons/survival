# Weighted log-rank tests

The Rust `surv_analysis::survdiff` and Python `surv_analysis.survdiff` /
`validation.logrank_test` compute the G-rho family of tests. The R facade
exposes right-censored tests with formulas through `r.survdiff`.
`rho=0` is the ordinary log-rank test; nonzero rho weights events by
the left-continuous Kaplan-Meier survival of their stratum, `S(t-)^rho`.

For nonzero rho, the implementation prepares these weights using the test's
existing stop-time ordering. It walks tied time blocks once, retaining one
left-limit value per row. The reverse test sweep reads that value directly.
This replaces a separate full KM fit, its input copies, sorts and count/curve
arrays, and a binary search for every time block. The ordinary `rho=0` path
does not allocate a weight vector.

The native counting-process extension reuses the entry-time order needed by
the test. At time `t`, its forward KM risk count is the number of starts
strictly below `t`, less the number of intervals ending before `t`.
Events and tied censoring rows share their pre-event risk set and `S(t-)`.
R's formula `survdiff` refuses counting-process responses; the Python formula
interface retains that restriction.

Group-specific observed, expected and covariance calculations keep their
previous reverse accumulation order. KM updates keep the shared KM engine's
multiplication order. Near-tie normalization still runs before ordering and
rejects intervals whose effective length becomes zero. The added weight
preparation is linear after sorting, with storage proportional to the largest
stratum. Group covariance work retains its previous complexity.

The independent reference generator records 80 complete results from stock R
survival 3.8-12 on R 4.5.3. Right-censored references use `survdiff` and its
`survdiff.fit` kernel with explicitly unrounded responses. Delayed-entry
references use stock KM left limits and explicit risk-set selection to compute
the test moments independently. Cases cover three groups, strata, tied deaths
and censors, near ties, normalization on/off, and rho -1, 0, 0.25, 1 and 2.
Rust and Python tests check every count, moment, covariance, test statistic and
probability; Python also verifies the formula interface for right-censored data.

```sh
Rscript scripts/generate_weighted_logrank_reference.R
cargo test --no-default-features --test weighted_logrank
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_weighted_logrank.py -q
RAYON_NUM_THREADS=1 cargo bench --bench survival_benchmarks -- logrank
RAYON_NUM_THREADS=1 PYTHONPATH=python .venv/bin/python scripts/benchmark_weighted_logrank.py
```

The Python benchmark measures complete native calls on NumPy inputs, including
conversion and result construction. Inputs are prepared outside timing and
near-tie normalization is disabled to isolate the weight calculation. It
reports every result field alongside medians and samples; compare those fields
before interpreting timing differences. `--extension /path/to/saved/_survival.so`
loads a previous build for the same Python ABI in a separate process.

A local Rust comparison compiled the original and revised public test functions
into the same release library, using the deterministic inputs from the Python
benchmark. It checked complete results for exact equality in all 32 workloads
before measuring. Each comparison alternated execution order, with four warmups
and nine measured samples per function and one Rayon thread. Input construction,
Python conversion and formula preparation were excluded; public validation,
sorting, test calculation, result construction and disposal were included.

For 100,000 observations and `rho=1`, the medians were:

| Response | Stop-time grid | Strata | Before | After |
| --- | --- | ---: | ---: | ---: |
| Right-censored | Distinct | 1 | 25.990 ms | 8.857 ms |
| Right-censored | Distinct | 5 | 30.841 ms | 11.918 ms |
| Counting-process | Distinct | 1 | 33.779 ms | 12.801 ms |
| Counting-process | Distinct | 5 | 35.428 ms | 14.985 ms |
| Right-censored | 997 tied times | 1 | 17.499 ms | 8.603 ms |
| Right-censored | 997 tied times | 5 | 17.637 ms | 9.920 ms |
| Counting-process | 997 tied times | 1 | 24.587 ms | 10.288 ms |
| Counting-process | 997 tied times | 5 | 23.888 ms | 13.081 ms |

These workloads improved by 1.78–2.93 times. The eight `rho=0` controls ranged
from 0.95–1.06 times their original speed. This describes the listed native
workloads on local Linux x86-64; formula calls and Python conversions have
additional costs. The initially available Python extension predated the source
baseline, so its separate-process timings are not used to attribute a speedup
to this change.
