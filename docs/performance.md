# Performance

The numerical work runs in Rust. Python and R entry points prepare inputs, call a
native kernel and assemble results. This page summarizes the current performance
characteristics and how each main kernel scales; timings depend on the data,
machine and build, so measure the workload you care about.

## Measuring

`benches/python/bench_vs_r.py` times the same synthetic data against R's
`survival` at two layers and compares each layer only with the matching R layer:
the `survival.r` formula calls against R's formula calls (what users pay,
including model frames and the extras R computes), and the `survival._survival`
kernels on NumPy arrays against the R entry points that do the same computation
(`coxph.fit` plus `concordancefit`, `survfitKM`, `survdiff.fit`, `concordancefit`
and `survreg.fit`). Run it against a release build:

```sh
PYTHONPATH=python python benches/python/bench_vs_r.py --sizes 1000,10000,100000
```

The Rust kernels have divan benchmarks. `benches/survival_benchmarks.rs` has
groups such as `kaplan_meier`, `curve_summary`, `curve_table`, `matrix_curves`,
`aggregate_curves`, `logrank`, `brier_score`, `exact_counting_process_cox`,
`cox_regression`, `survreg_bench`, `expected_survival_grid`,
`expected_survival_summary_bench`, `rate_matching_bench`, `yates_sgtt_bench` and
`tmerge_bench`; `benches/concordance_benchmarks.rs` and
`benches/survfitaj_benchmarks.rs` cover concordance and Aalen–Johansen fits.

```sh
RAYON_NUM_THREADS=1 cargo bench --bench survival_benchmarks -- kaplan_meier
cargo bench --bench concordance_benchmarks
cargo bench --bench survfitaj_benchmarks -- standard_errors_long_grid
```

`benches/python/bench_formula_cox_fit.py`, `bench_cox_prediction_inputs.py`,
`bench_aft_inputs.py` and `bench_coxms_subset_summary.py` measure complete Python
calls and check their outputs before timing. The first three can save results with
`--output` and require identical results from an earlier run with `--compare`;
`--extension` loads a saved native library for the prediction and AFT input
benchmarks.

## Current characteristics

- Native bindings copy NumPy arrays once into owned Rust buffers (any dtype,
  memory layout or stride, and pandas or Polars columns) and release the GIL while
  the kernel runs, so fits on several Python threads run in parallel. Lists of
  float rows fill one allocation. Results that consumers read as arrays are
  returned as NumPy arrays; several result types offer `to_arrays()` snapshots.
- Formula calls add model-frame preparation in Python: expression evaluation,
  factor coding and omission. Numeric columns stay NumPy arrays until they are
  copied into the design. The R bridge adds reticulate conversion, which is done in
  bulk for numeric columns and native results.
- Because that preparation is a roughly fixed cost per call, stock R is faster for
  cheap R-facing operations, such as returning a stored model matrix or small
  formula predictions. The native kernels are faster for heavier work, such as
  weighted Kaplan–Meier curves with robust variance, AFT term predictions, Cox
  survival simulations in `yates` and rate-table expected survival.

## How the kernels scale

### Survival curves

- Kaplan–Meier fits sort once (O(n log n)). For right-censored data with one
  observation per cluster and no influence matrix, including the usual weighted
  curve where fractional weights select robust variance, standard errors come
  from an O(n) sweep: observations at risk share a coefficient, departed
  observations' influences evolve by product-limit factors, and suffix sums of
  squared risk weights avoid cancellation. Repeated clusters, counting-process
  data and explicit influence matrices use the general influence calculation.
- Independent right-censored competing-risk curves use a moment sweep with small
  triangular QR factors updated by Givens rotations: O((n + t)(s + h)) time and
  O(n + s + h) extra storage for t reporting times, s states and h hazard columns.
  Counting-process data, repeated clusters, uncertain initial distributions,
  explicit influences and numerically sensitive inputs use the general kernel.
  Point-estimate fits (`se_fit=False`) allocate no robust-variance workspace.
- Requested-time summaries choose a forward sweep for dense ordered queries and
  binary search for sparse or unordered ones, comparing the query count with the
  grid size divided by its search depth; count accumulation streams prefix sums in
  R's addition order. Compact tables read restricted-mean rectangles directly from
  the curve with scratch storage proportional to the number of curves, and median
  searches stop at the required crossing.
- Grouped residuals and pseudo-values bucket input rows by curve in one pass
  (O(n + g) for g curves). Grouped multi-state data frames copy contiguous time
  blocks per state.
- Curve aggregation prepares group positions once and fills one reusable buffer per
  time and state; medians use linear-time selection instead of sorting.
- Quantiles use adjacent comparisons on the common monotone path and share one
  interpolation buffer across the two tolerance shifts.
- `survfit.matrix` sorts the event-time union in O(L log L) for L input rows and
  updates discrete curves in O(L + U(K + S)) for U times, K transitions and S
  states, plus O(U S³) for dense matrix exponentials; it allocates no
  time-by-transition jump matrix.

### Cox models

- Breslow and Efron evaluations reuse predictor, risk and risk-set buffers for
  every trial coefficient vector. Exact ties use a subset dynamic program with
  `k + 1` states, `k = min(d, n - d)` for d deaths among n at risk, computing the
  complementary subset when most rows die together, with stable mixing of log
  weights and anchored means. Counting-process exact fits walk each stratum once by
  decreasing stop and entry time, using a running accumulator for growing risk sets
  and a tree of 16-row blocks otherwise: O(n log n) preparation and
  O(n log n (p² + 1)) per untied evaluation in general.
- Predictions at requested times read the cached per-stratum baseline at each
  time and expand only those probabilities: O(s m log u) lookup and O(n m)
  expansion for m times, s strata and u unique times, instead of full n-by-u
  curves. The Brier kernel caches censoring survival once per subject.
- Full curves accumulate coefficient uncertainty with one O(p) vector per curve
  instead of a time-by-covariate matrix; time-dependent subjects find their
  baseline rows by binary search (O(I log T + U) for I intervals and U selected
  rows) and are grouped in one pass.
- Expected-event, survival-at-time and cohort predictions compute risks with one
  reusable centered row (O(p) temporary storage). Grouped predictions are summed
  inside the native call, so term output storage is O(G × T) for G groups and T
  terms.
- Cohort expected survival from a Cox model aggregates baselines and risks
  directly, using O(N + B + T × G) storage for N subjects, B baseline points, T
  times and G groups.

### Other routines

- Concordance prepares each stratum's outcome buffers, entry order and time
  multipliers once for all predictors; each predictor keeps its own rank tree, so
  several predictors cost much less than separate calls while one predictor costs
  the same.
- Weighted log-rank tests compute `S(t-)^rho` weights in the test's own time order
  without a separate curve fit; `rho=0` allocates no weights.
- Rate-table matching folds labels once, caches small categorical axes and promotes
  to a hash map for larger ones; expected-survival grids select requested times
  with one forward scan (O(fitted times + requests)).
- Person-years tabulation accumulates directly into cells for fixed categories
  without a rate table; other inputs use the shared interval stepper.
- O'Brien expansion sorts each covariate once per stratum and writes into
  precomputed output slices: O(P N log N + P M) work for N rows, P continuous
  columns and M expanded rows.
- Term grouping (`attrassign`) is one O(P + T) pass over P columns and T terms.
- Response repetition, `lvcf`, `neardate` and `tmerge` avoid per-row Python index
  lists and repeated sorts; `tmerge` never reads unreferenced update columns.
- P-spline penalties fill the three nonzero entries of each second-difference row
  directly instead of a cubic dense product.
