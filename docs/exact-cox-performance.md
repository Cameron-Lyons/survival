# Exact Cox conditional moments

Both right-censored and counting-process Cox fits with `method="exact"` use
the same Rust conditional-moment kernel. For a risk set of `n` rows and `d`
tied deaths, it computes the exact denominator, expected covariate sum and
covariance over the possible death subsets.

The dynamic program keeps `k + 1` states, with `k = min(d, n - d)`. Its
moment work is `O(n * (k + 1) * (p² + 1))` and storage is
`O((k + 1) * (p² + 1))`, where `p` is the number of covariates. When most
rows die together, this is a
substantial reduction from keeping `d + 1` states. For example, 999 deaths
among 1,000 rows require two states.

For a selected subset `A`, let `eta` be the log risk of each row. The
complement identity is

```text
weight(A) = exp(sum(eta)) * exp(-sum(eta[complement(A)]))
sum(x[A]) = sum(x) - sum(x[complement(A)])
```

The shared factor cancels from the conditional probabilities. The
complement uses negated log risks, and its covariance equals the selected
subset's covariance. The dynamic program carries selected covariate means
directly even while counting complementary subsets: the existing branch
includes the current row, while the added complement branch excludes it.
This avoids subtracting a large excluded mean from the total covariate sum.
A common log-risk shift is removed before accumulating the complementary
denominator, then restored in the final log denominator.
The public fitters already skip complete risk-set ties, whose conditional
likelihood is one and information is zero.

Each dynamic-programming merge computes the smaller mixing probability
directly from the log-weight difference. This retains covariance contributions
below machine epsilon when the larger probability rounds to one. The same
merge serves the running singleton accumulator for untied exact fits.
Means are anchored at the more probable branch, preserving a moderate mean
when the minority branch has much larger covariates.

For counting-process data, the fitter prepares a per-stratum walk over rows
by decreasing stop time and entry time. Risk membership remains `start < t <=
stop`, including rows censored at a death time. Rows whose intervals span no
death are omitted from the walk. Each retained row joins once and leaves once.

If every retained entry precedes the first death in a stratum, its risk set
only grows during the backwards walk and uses one running accumulator. Other
strata use a tree of blocks containing 16 input rows. A membership change
marks its block dirty; before an untied death, the fitter rebuilds dirty blocks
from their active rows and merges their ancestors. Censor changes accumulate
until the next untied death. Each merge combines log denominators, weighted
means and centred covariances. It never subtracts a departing risk score from
a rounded total, so moderate risks remain available after a dominant row leaves.
Growing singleton moments are computed lazily, so a sequence containing only
tied deaths does not compute unused singleton covariances.

Preparing the walk takes `O(n log n)` work. An untied likelihood evaluation
takes `O(n * (p² + 1))` for growing strata and
`O(n log n * (p² + 1))` for general delayed entry, with
`O(n + ceil(n / 16) * (p² + 1))` additional storage. Tied deaths still gather
the active rows and use the subset dynamic program described above; complete
risk-set ties still contribute zero likelihood, score and information.

Regression tests independently enumerate every subset of small, noncontiguous
risk sets and compare their denominator, mean and centred covariance across
all subset sizes. They also check row-order invariance, common log-risk shifts
of ±10,000, risk ratios of `exp(80)`, a 10,000-row nearly complete uniform tie,
and a counting-process fit against R survival 3.8-12 at nonzero initial
coefficients with offsets.

Counting-process tests independently enumerate fresh active sets at every
death time for general delayed entry, growing strata and mixtures of the two,
including overlapping intervals, permutations, common offsets of ±1,000,
entries exactly at death times and tied censors. They compare likelihood,
score and directly inverted information. Tree regressions exercise dominant
risk removal across different blocks, empty/reset states and zero-column
models. Large covariate contrasts check the weighted mean and covariance in
the accumulator, tree and both subset paths.

The separate stock R fixture contains ten initial fits and nine converged
fits, with a generator at `scripts/generate_exact_counting_reference.R`.
It records R's frontend rejection of a common offset of +1,000; the equivalent
fit after removing that common shift supplies the reference for that case.

The public counting-process benchmark includes input validation and one
likelihood evaluation at fixed initial coefficients, with data generation
excluded. Reproduce it with:

```sh
cargo bench --no-default-features --bench survival_benchmarks -- \
  exact_counting_process_cox --sample-count 15 --sample-size 1
```

On a local Linux x86-64 build with release LTO enabled and one code-generation
unit, paired medians from 15 samples with one iteration each were:

| Untied workload | Rows | Previous risk-set reconstruction | Prepared sweep |
| --- | ---: | ---: | ---: |
| General delayed entry, 2 covariates | 1,000 | 6.817 ms | 0.8869 ms |
| General delayed entry, 2 covariates | 2,000 | 23.20 ms | 1.978 ms |
| General delayed entry, 2 covariates | 4,000 | 91.99 ms | 4.343 ms |
| Growing risk set, 1 covariate | 1,000 | 13.99 ms | 65.32 µs |
| Growing risk set, 1 covariate | 2,000 | 55.58 ms | 147.0 µs |
| Growing risk set, 1 covariate | 4,000 | 223.9 ms | 340.6 µs |

The previous reconstruction baseline already included the complementary
subset kernel and structural validation. A separate pair measures the
nearly complete tie against the original kernel with `d + 1` states:

| Rows at risk | Before complementation | Final kernel |
| ---: | ---: | ---: |
| 1,000 | 21.54 ms | 79.32 µs |
| 4,000 | 280.8 ms | 359.2 µs |

Walk preparation and the stable mean arithmetic add visible overhead to
small tied calls. Against the intermediate complementary kernel, the final
nearly complete tie went from 50.37 to 69.57 µs at 1,000 rows and 286.0 to
366.2 µs at 4,000 rows. The balanced 12-death tie among 24 rows went from
5.023 to 7.213 µs in that pair. In the original-kernel pair, the balanced
case went from 7.976 to 6.807 µs. These controls bound the measured benefit
to the workloads listed above; balanced ties retain their subset-DP cost.

The current calls include structural validation at the fitting boundary.
Rust callers can mutate the public `CoxphData` fields after construction;
the fitter checks these fields again before constructing exact risk sets.

The same benchmark group includes untied constant-entry and general
delayed-entry scaling cases. Their full public calls include validation,
centering, walk preparation and a single likelihood evaluation.
