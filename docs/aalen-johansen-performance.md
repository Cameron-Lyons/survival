# Aalen–Johansen point-estimate memory

`survfitaj` allocates robust-variance workspaces only when `se_fit=true`.
With standard errors disabled, neither the initial-influence matrix nor the
per-cluster probability, area, hazard and risk-weight matrices are allocated.
This applies to right-censored competing risks and counting-process multi-state
histories, including case weights, repeated clusters, strata, estimated or fixed
initial probabilities, conditional starts, and entry-time reporting.

The numerical transition update retains its addition order. One probability
scratch vector is reused at every reporting time, replacing a new allocation
per update. Fits with standard errors and explicit influences use the same
general influence calculation; independent competing-risk uncertainty retains
the moment sweep described in [survival curve performance](kaplan-meier-performance.md).
Multi-state AJ fits support `stype=1` and `ctype=1`, as in R.

For a curve with `c` clusters, `s` states and `h` transition-hazard columns,
point-estimate fits previously allocated five unused `s × c` matrices and one
unused `h × c` matrix. Removing these buffers saves `8c(5s + h)` allocated
bytes, plus a small state-transition matrix and variance vectors. Input
preparation still uses storage proportional to the number of observations;
count, probability and hazard results still use storage proportional to the
requested reporting grid.

## Reproduce the allocation benchmark

```sh
cargo bench --offline --no-default-features --bench survfitaj_benchmarks -- \
  point_estimates_tied_states --sample-count 15 --sample-size 1 --timer os
```

The benchmark enables Divan's allocation profiler. It prepares borrowed inputs
before measurement, then measures the complete public Rust fit and disposal of
its result. Times cycle through 32 tied values, event codes through 16 labels,
and every fourth subject is censored. There are 17 states including the initial
state and 12 observed transition columns. Default time normalization is enabled,
and standard errors are disabled.

A local Linux x86-64 release comparison measured these peak live fitting
allocations. A standalone system-allocator counter provided exact byte counts;
the Divan profiler reproduced the same peaks.

| Subjects | Before | After | Reduction |
| ---: | ---: | ---: | ---: |
| 10,000 | 9,981,047 bytes | 2,534,823 bytes | 74.6% |
| 100,000 | 98,593,559 bytes | 23,773,127 bytes | 75.9% |

For 100,000 subjects, total allocated bytes fell from 114,134,795 to 36,527,899,
and allocation/reallocation operations fell from 461 to 420. Peak allocation
describes memory allocated during fitting, rather than process RSS, and excludes
the previously prepared input. Timings varied during concurrent builds, so this
comparison makes no speed claim.

## Numerical verification

```sh
cargo test --offline --no-default-features --lib surv_analysis::survfitaj
cargo test --offline --no-default-features --test aj_entry_grid
```

The 64 independent stock-R entry-grid cases verify complete counts,
probabilities, hazards, metadata and uncertainty for the general influence fit.
The same cases additionally require every point-estimate result field to equal
that reference-verified fit, with uncertainty fields absent. They include
subject continuations, delayed entry, case weights, strata, shuffled input,
conditional starts, entry counts and initial-time rows. A further 16 unit-test
combinations compare complete outputs with estimated or fixed initial
probabilities and repeated clusters. Existing AJ tests continue to check
standard errors, influence matrices, competing risks, self-transitions and
extreme weights.
