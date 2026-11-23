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
`tmerge_bench`, plus `turnbull_bench` and `neardate_bench`;
`benches/concordance_benchmarks.rs` and
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

`benches/python/bench_concordance_inputs.py` measures complete concordance calls
for numeric formulas, row lists and NumPy matrix layouts. It saves medians and
result fingerprints with `--output` and checks all outputs with `--compare`.
Input construction and disposal of the previous result are outside the timings.

`benches/python/bench_formula_aft_fit.py` measures AFT formula preparation and
complete fits. Run once with `--row-design-reference --output before`, then with
`--compare before.npz --output after` to compare row-list preparation with the
owned-array path on the same native extension. The comparison requires identical
designs, coefficients, covariance, likelihoods, scales, predictions and residuals.

`benches/python/bench_case_cohort.py` uses the Prentice and Borgan I inputs from
the Rust benchmarks. Save a previous release with `--extension`, then use
`--compare` to check coefficients, covariance, phase-two variance, martingales
and Borgan score contributions alongside the complete-call timings.

`benches/python/bench_turnbull.py` measures disjoint and overlapping intervals,
including full-span intervals, varied inspection widths, fractional weights and
mixed censoring. Use `--extension` to select a saved native build and
`--compare` to check exact agreement of the survival estimates, uncertainty,
counts and iterations on the same platform.

`scripts/benchmark_interaction.py` measures formula interactions with sparse
observed combinations, checking their labels and level order before reporting
runtime and peak Python allocations. The multistate `summary` benchmarks in
`benches/survfitaj_benchmarks.rs` cover sparse requests and event summaries, and the Cox
diagnostic benchmarks include tied Breslow events with ordinary and extreme risks.

## Current characteristics

- Native bindings copy NumPy arrays once into owned Rust buffers (any dtype,
  memory layout or stride, and pandas or Polars columns) and release the GIL while
  the kernel runs, so fits on several Python threads run in parallel. Lists of
  float rows fill one allocation. Results that consumers read as arrays are
  returned as NumPy arrays; several result types offer `to_arrays()` snapshots.
  Masked ndarray subclasses preserve missing entries; ordinary ndarrays keep
  their direct extraction path.
- Formula calls add model-frame preparation in Python: expression evaluation,
  factor coding and omission. Row labels of integer- and string-indexed frames
  are read in bulk, increasing row selections keep their labels without
  `make.unique`, and `tt()` terms expand into an array design. Numeric columns stay NumPy arrays until they are
  copied into the design. List columns, as the bundled datasets and the
  `survSplit`, `finegray` and `tmerge` results hold them, are classified as logical,
  character or numeric and as integer or double storage from their distinct value
  types, so inference costs one C-level pass per column. The R bridge adds
  reticulate conversion, which is done in bulk for numeric columns and native results.
- Row expansions build their columns with NumPy: the `tt()` risk-set rows
  (one per subject at risk at each event time), the `finegray` weighted rows and the
  backward-interval check of counting-process `Surv` objects. Their output sizes
  are those of R; for `tt()` and `finegray` they grow with the number of event
  times times the number at risk.
- Formula interactions fit each distinct component once. An interaction can
  reuse a main effect's design columns when both use the same coding.
  Interaction terms that need full factor indicators build those columns
  separately from treatment contrasts.
- Because that preparation is a roughly fixed cost per call, stock R is faster for
  cheap R-facing operations, such as returning a stored model matrix or small
  formula predictions. The native kernels are faster for heavier work, such as
  weighted Kaplan–Meier curves with robust variance, AFT term predictions, Cox
  survival simulations in `yates` and rate-table expected survival.

## How the kernels scale

### Survival curves

- Turnbull fits store contiguous jump ranges for censored observations, using
  O(n + j) membership storage for n observations and j jump points. Initial
  overlap counts use endpoint updates. For dense grids, intervals with the
  same first jump reuse partial sums of jump probabilities. Large fits split
  the jump grid across workers, each accumulating the observations in input
  order. Each EM iteration still takes O(n j) work in the worst case. Preserving
  the order of the floating-point sums keeps the acceleration and stopping
  decisions reproducible across worker counts. The Kaplan–Meier update sorts
  the augmented observation rows once.
  Each iteration then takes O(n + j) time for the product-limit calculation,
  accumulating tied weights in the same order as the full kernel. The final
  fit supplies risk counts, standard errors and confidence limits. Grouped
  fits gather the rows for each stratum once. The
  `turnbull_bench::interleaved_strata` benchmark holds rows per stratum fixed
  as the number of strata grows.
- Kaplan–Meier fits sort once (O(n log n)). For right-censored data with one
  observation per cluster and no influence matrix, including the usual weighted
  curve where fractional weights select robust variance, standard errors come
  from an O(n) sweep: observations at risk share a coefficient, departed
  observations' influences evolve by product-limit factors, and suffix sums of
  squared weight shares use the total weight at risk at each event.
  Preparing and fitting multiple curves stays sequential for at most 1,000
  observations, avoiding thread-dispatch overhead on small stratified fits.
  Larger fits retain parallel curve preparation and fitting.
- Subtracting a large case weight can erase smaller subjects still at risk.
  For widely separated weights, delayed-entry curves and clustered influence
  calculations keep integer sums for each occupied binary exponent. Removing
  a weight updates its exponent's sum exactly. Reporting a risk total visits
  only the occupied exponents.
- Right-censored Kaplan–Meier fits with repeated clusters and no influence request
  keep each cluster's influence as its weight at risk times a shared coefficient
  plus a departed part, held in running sums of squares and products that change
  only when a row leaves: O(n + t + c log n) per curve for c clusters and t times,
  instead of O(t c). The sums are rebuilt whenever the sum of squared weights at
  risk halves.
  Times at which nearly everyone at risk has the event (|1 - hazard| < 1e-3) take
  the dense step, so a standard error where the curve reaches zero is the dense
  loop's bit for bit; elsewhere the two agree to about 1e-11 relative. The
  expanded sums cancel by about the mean size of the cluster a random row belongs
  to (`sum W_g^2 / sum w_i^2`), so fits where that exceeds 1,000, such as one
  cluster holding most of the rows, keep the general influence calculation, as do
  fits with at most 64 clusters or few event times, counting-process data,
  explicit influence matrices, and weights whose range would lose precision
  in the running sums.
- Independent right-censored competing-risk curves use a moment sweep with small
  triangular QR factors updated by Givens rotations: O((n + t)(s + h)) time and
  O(n + s + h) extra storage for t reporting times, s states and h hazard columns.
  Counting-process data, repeated clusters, uncertain initial distributions,
  explicit influences and numerically sensitive inputs use the general kernel.
  Point-estimate fits (`se_fit=False`) allocate no robust-variance workspace.
  Grouped initial-state preparation scans the global minimum entry time once, and
  only for counting-process data. Outside the moment sweep, each event time updates
  up to four state or hazard rows in one pass over the clusters, keeping every
  sum's order, so results are unchanged bit for bit.
- Stratified Kaplan–Meier and Aalen–Johansen fits number every curve's clusters
  through one table cleared after each curve, then fit the curves in parallel when
  there are several curves and more than 1,000 rows. Integer ids and clusters whose
  values span fewer than 2n values are numbered through a table indexed by value,
  others through a hash map. Kaplan–Meier fits reuse input columns when all
  rows are retained. A single curve uses the fitted cluster codes directly.
- `survfit.coxph(newdata)` curve columns become Kaplan–Meier engines in one native
  call that reads the time-by-curve matrices directly, with no per-element Python
  conversion. The result stores its curve matrices as arrays and builds the nested
  lists of a field only when it is read, so `survfit(coxph, newdata=)` with 1,000
  new rows at 10^4 observations takes 50 ms instead of 182 ms, and 77 ms with a
  requested-time summary; `to_arrays()` reads the curves without any lists.
- Requested-time summaries read the fitted columns directly and supply the
  time-zero row during lookup. A sparse query with per-time counts allocates
  space proportional to the returned points. The lookup uses a forward sweep
  for dense ordered queries and binary search for sparse or unordered ones.
  It compares the query count with the grid size divided by its search depth.
  Count accumulation streams prefix sums in R's addition order. Compact tables
  read restricted-mean rectangles directly from
  the curve with scratch storage proportional to the number of curves, and median
  searches stop at the required crossing. `print_survfit` uses these compact
  kernels directly and never builds the per-observation arrays of
  `summary_survfit`.
- Grouped residuals and pseudo-values bucket input rows by curve in one pass
  (O(n + g) for g curves). Grouped multi-state data frames copy contiguous time
  blocks per state. Python flattens pseudo-values in time, state and subject
  order with one conversion of the native result.
- `survcheck` uses the subject ordering prepared for interval checks to count
  visits to each state. It builds the event table from visit-count histograms,
  avoiding a subject-by-state matrix.
- Multistate summaries allocate selected output rows directly and represent
  inserted initial rows in the selection indices. Sparse requested-time summaries
  and multistate Cox row selection avoid copying the fitted time grid, counts,
  probabilities and confidence arrays. Summed counts still visit the covered
  intervals in their original addition order.
- Curve aggregation prepares group positions once and fills one reusable buffer per
  time and state; medians use linear-time selection instead of sorting. When the
  product of declared factor levels overflows, observed level combinations are
  ranked directly.
- Quantiles use adjacent comparisons on the common monotone path and share one
  interpolation buffer across the two tolerance shifts.
- `survfit.matrix` sorts the event-time union in O(L log L) for L input rows and
  updates discrete curves in O(L + U(K + S)) for U times, K transitions and S
  states, plus O(U S³) for dense matrix exponentials; it allocates no
  time-by-transition jump matrix.
- Restricted-mean comparisons gather each group's rows once in sorted label order
  instead of scanning all observations per group.
- Plot data compress constant runs, so path vertices scale with the number of
  changes rather than the number of censored observations; preparing plots copies
  no observation-by-time influence matrix and skips confidence arrays when
  intervals are off.

### Cox models

- The `agsurv` baseline kernel reuses its risk-set covariate sums for the
  returned `xbar` matrix. Efron variance also keeps tied-death covariate sums
  on the failure-time grid.
- Extreme right-censored Breslow/Efron evaluations use normalized risk moments
  in a reverse-time sweep, retaining O(n p²) work and O(p²) moment scratch space
  alongside the O(n) predictors.
  When every row enters before the earliest stop, extreme risk moments use a
  reverse additive sweep. Martingale and score diagnostics use forward hazard
  prefixes, avoiding cancellation and repeated tree updates in case-cohort fits.
  Additive accumulation takes O(n (p + 1)) work and retained storage; sorting
  and score interval lookup retain O(n log n) work.
  Delayed-entry intervals and covariance requests retain trees for active risk
  sets. Covariance trees are built only for detail and proportional-hazards
  diagnostics that require them. Curve interval trees are built on the first
  range query; full curve expansion does not build them. Ordinary fits retain
  their existing arithmetic.
- Covariate and offset origins are retained when their scale would lose small
  effects. Each risk total separates the compensated predictor from the relative
  mass, preserving multiplicities without changing tree complexity.
  Prediction combines relative risk and baseline hazard before
  exponentiating. Kalbfleisch–Prentice curves keep log survival increments;
  single deaths and equal-risk ties have a closed-form increment, while
  heterogeneous ties use a scalar root solve. Ordinary K-P inputs reuse the
  ordinary risk sums, with a log-sweep fallback when entry subtraction loses
  the active risk set.
- Exact ties at extreme predictor scales normalize each subset state by the
  product of its largest risks. For r active rows and k = min(d, r - d), this
  fallback uses O(r log r + r k p²) work and O(r + k p²) storage. Event likelihoods
  cancel each selected predictor against its reference before summing.
- Penalized fits store their rows in risk-set order once per fit, so each
  Newton evaluation streams covariates, weights and offsets instead of
  gathering them through the sort order. Efron's frailty tie correction adds
  each death step to a few running sums and corrects every tied group once per
  tie set: O(d) instead of O(d²) group updates for d tied deaths.
- Penalized fits use the same risk representation for extreme predictors.
  Sparse frailty groups integrate score and information updates only when their
  active risks change, using hazard interval queries. Efron ties accumulate block
  moments and visit each affected group once, avoiding a loop over every group for
  each tied death and a dense matrix of frailty indicators.
- Fit preparation sorts packed (stratum, time) integer keys instead of row
  indices through a comparator, and centres and scales covariates in row-major
  passes. Tied-death second moments are accumulated only at Efron death times
  with several deaths, the only place they are read.
- A tie set of d deaths adds Efron's information in O(d + p²) instead of the
  O(d p²) of R's per-death loop, for both unpenalized fits and penalized fits
  without a sparse frailty (where Breslow ties also loop over every death). The
  steps' log-likelihood and score are still summed one at a time, so they match R's
  sums bit for bit; the information is written with the survivors' sums so that
  five positive scalar sums over the steps carry it (`efron_steps.rs`). Sets with
  non-finite sums keep the per-death loop. With heavy ties (about 40 deaths per
  event time, 10^5 rows) an Efron `coxph` with ten covariates is about 1.2x faster,
  and penalized spline fits 1.3-1.4x.
- Right-censored fits with at least eight covariates skip the running
  second-moment sums altogether (`cox_optimizer/gram.rs`). Each death time adds
  `sum_j wtave / d_j` to the weight of every subject still at risk (its own deaths
  take `sum_j wtave s_j / d_j` instead), so the second-moment part of the
  information is one weighted Gram matrix `X' diag(w) X`. The mean part is a Gram
  matrix of the death-time means, two rows per Efron tie set from the Cholesky
  factor of its step sums. Rows pass through the products in blocks of 4,096, so
  the scratch storage is O(p^2 + 4096 p). Both products run through
  `matrixmultiply` (`cox_gram.rs`); with 50 covariates and continuous times at
  10^5 rows a fit takes 150 ms instead of 325 ms. The penalized right-censored
  sweep without a sparse frailty (`coxpenal/kernel/gram.rs`) shares the
  construction: `coxph(~ pspline(x1) + x2)` at 10^5 rows takes 251 ms instead of
  423 ms. Counting-process fits on risks centred at zero weight each
  `(start, stop]` row by the alphas of the death times in its interval, a
  difference of double-double prefix sums found by binary search, so late short
  intervals keep their precision; R's running sums lose the rounding of every row
  that leaves them, so against explicit risk sets the Gram information is as
  accurate as theirs. The penalized counting-process sweep without a sparse
  frailty does the same (a pspline fit on 1.6 x 10^5 rows takes 349 ms instead
  of 466 ms); sparse-frailty sweeps keep the running sums.
- Robust and cluster variances, counting-process martingale residuals, and
  score, dfbeta and Schoenfeld residuals derive their row orders from the fit's
  sorted rows in O(n) instead of sorting again, and sum clusters by dense ids in
  one pass. `cox.zph` buckets deaths by stratum once, so standardization costs
  O(n + d) instead of O(strata × d) for d deaths.
- Breslow and Efron evaluations reuse predictor, risk and risk-set buffers for
  every trial coefficient vector. Exact ties use a subset dynamic program with
  `k + 1` states, `k = min(d, n - d)` for d deaths among n at risk, computing the
  complementary subset when most rows die together, with anchored means. Weights
  are scaled by the largest risk score and each state keeps a power-of-two
  exponent, so the recursion runs in linear arithmetic; a risk set whose log-risk
  spread exceeds 256 mixes log weights instead. Right-censored fits and
  counting-process strata with nested risk sets can share one running recursion
  across ties. When the ties justify keeping L levels, the recursion updates
  about `n L` cells per evaluation. Counting-process exact fits walk each stratum once by
  decreasing stop and entry time, using a running accumulator for growing risk sets
  and a tree of 16-row blocks otherwise: O(n log n) preparation and
  O(n log n (p² + 1)) per untied evaluation in general.
- Predictions at requested times read the cached per-stratum baseline at each
  time and expand only those probabilities: O(s m log u) lookup and O(n m)
  expansion for m times, s strata and u unique times, instead of full n-by-u
  curves. Each time row is filled independently, in parallel above 8,192 values.
  Cox Brier scores evaluate each time's row inside the scoring loop with one
  n-length buffer per thread, so `brier(detail=FALSE)` needs O(n) memory per
  thread instead of O(n m); the Brier kernel caches censoring survival once per
  subject.
- Full curves accumulate coefficient uncertainty with one O(p) vector per curve
  instead of a time-by-covariate matrix; time-dependent subjects find their
  baseline rows by binary search (O(I log T + U) for I intervals and U selected
  rows) and are grouped in one pass.
- Expected-event predictions with standard errors integrate baseline variances
  and covariate means on first use of each requested stratum. Subsequent calls
  reuse those integrals, and each observation uses one lookup per endpoint for
  both its prediction and its standard error. The ordinary-risk calculation
  retains one variance and one value per covariate at each baseline time in
  the requested strata.
- Expected-event, survival-at-time and cohort predictions compute risks with one
  reusable centered row (O(p) temporary storage). Grouped predictions are summed
  inside the native call, so term output storage is O(G × T) for G groups and T
  terms.
- Cohort expected survival from a Cox model aggregates baselines and risks
  directly, using O(N + B + T × G) storage for N subjects, B baseline points, T
  times and G groups.
- Sparse frailty fits accumulate each group's score, information diagonal and
  cross terms lazily: a group's risk sum changes only when one of its members
  enters or leaves the risk set, so its terms are flushed from running
  (double-double) sums of the death-time weights at those moments, and only the
  groups of tied deaths are corrected at each Efron step. An evaluation costs
  O(n p² + G p) for G groups instead of O(d G p) for d deaths.
- Stored sparse-frailty model matrices insert the group vector into one owned row
  snapshot, so keeping the sparse labels costs one vector rather than a copy of the
  expanded design.
- The proportional-hazards smoother (`cox_zph_smooth`) forms the spline basis
  once, reuses QR factorizations and prediction variances for terms with the same
  missing-value mask, and computes standard errors by triangular solves without
  normal equations, so one call for all terms is much cheaper than one per term.
- Cox detail calculations evaluate Breslow risk-set moments once per unique death
  time, including at extreme risk scales. Weighted hazard variance retains the
  sum of squared per-death increments. Efron keeps its changing risk-set steps
  and reuses the covariate-mean buffer across those steps.

### AFT models

- Formula designs are built as owned contiguous NumPy arrays, avoiding Python
  row lists and per-element conversion at the native boundary. Retained model
  matrices (`x=True`) still expose the existing list representation.
- Built-in families have one compiled likelihood sweep each, with the stratum
  scales and their logarithms computed once per evaluation. They allocate no
  endpoint or density buffers; callback families use workspace linear
  in the number of rows plus interval endpoints, and the Python adapter reacquires
  the GIL once per batch.
- With eight or more covariates, the covariate block of the information
  (`x' diag(w ddg) x`), the squared scores of robust variances
  (`x' diag(w dg^2) x`) and the weighted least squares of the starting values are
  weighted Gram products through `matrixmultiply` (`cox_gram::add_weighted_gram`),
  in 4,096-row blocks. The log-likelihood, score and scale-parameter terms keep
  the per-observation sums bit for bit, and sweeps with a non-finite term keep the
  scalar sweep. A 50-covariate Weibull fit on 10^5 rows takes 298 ms instead of
  393 ms. Penalized and frailty sweeps keep their scalar sums.
- Built-in distribution queries compose recycling indices and allocate the result
  once; callback queries reuse owned arithmetic buffers.
- With an estimated scale, quantile standard errors reuse each row's covariance
  products across the requested probabilities. For m fitted parameters and q
  quantiles, each row takes O(m² + qm) work. One quantile uses a single direct
  quadratic form.
- Per-term predictions center the selected design columns as they read each
  row. Repeated selections reuse the same contribution and standard error.
  The standard-error calculation shares a work vector across rows and terms.
- Collapsed residuals add each weighted observation directly to its group's
  total. Influence residuals and the robust covariance share the same score
  calculation, reusing two work vectors and reading covariance rows contiguously.

### Other routines

- Raw Cox and Andersen–Gill martingale residuals use logarithmic risk-set sums
  for extreme or zero risks and weights. Range sums of log-hazard increments
  preserve small contributions after large ones, including disjoint entry/exit
  intervals. This fallback costs O(n log n); ordinary inputs retain the existing
  arithmetic.
- Concordance prepares each stratum's outcome buffers, entry order and time
  multipliers once for all predictors; each predictor keeps its own rank tree, so
  several predictors cost much less than separate calls while one predictor costs
  the same.
- The concordance risk-set and death-weight trees share one cache-aligned node
  array, so each death, censoring or entry updates both in one walk. Once the
  death tree is complete, its final influence sums are read once per tree node
  instead of once per row. Predictor ranks and the sweep order come from sorts
  of packed integer keys, avoiding comparisons that load random rows. Without
  standard errors only the risk-set tree is kept.
- Numeric NumPy concordance matrices pass through a typed native constructor,
  avoiding a nested Python list and its per-element conversions. Contiguous,
  strided and Fortran-ordered arrays retain the same row and column order.
- Redistribution-to-the-right weights group observed strata, with storage
  proportional to the rows and observed groups instead of the largest label.
  Strata without censoring skip the time sorts. Widely separated weights and
  totals at risk of overflow use exact risk-set updates and log ratios.
  Counting-process histories can also drive censoring survival below the
  normal floating-point range. Those curves repeat the sweep on the log scale.
  The native redistribution and Cox detail computations release the GIL.
- Weighted log-rank tests compute `S(t-)^rho` weights in the test's own time order.
  The ordinary log-rank test (`rho=0`) skips the weight calculation.
  Sample sizes include every row. Event-free strata skip sorting. In strata
  with more censorings than events, sorting and risk-set calculations start
  with rows whose follow-up reaches the first event.
  The covariance update visits only groups present in the stratum.
  The `logrank::two_groups_per_stratum` benchmark measures this case.
- Rate-table matching folds table labels once and indexes exact matches; a small
  cache serves the observed labels of common sex and race dimensions and is
  promoted to a hash map after four distinct observed labels. Prefix matching
  scans an axis only when there is no exact match. Expected-survival grids select
  requested times with one forward scan (O(fitted times + requests)).
- Expected-survival walks process subjects in blocks bounded by 16 MiB of
  per-time terms: each block is walked in parallel, and its
  terms are then added in parallel over output times but in subject order, so the
  sums match R's sequential loop bit for bit. Date rate variables held as pandas
  Timestamps convert to days by integer division of their nanosecond value, and
  NaT flags come from the column's dtype rather than a per-value scan.
- Person-years tabulation accumulates directly into cells for fixed categories
  without a rate table; other inputs use the shared interval stepper. Summary
  margins use direct column-major slab offsets, and `print_pyears` totals walk the
  stored cells without flattening them.
- Aalen influence reductions for robust summaries and plot variances work in
  blocks of about 1 MiB and never convert or square the whole group by coefficient
  by time array; `se=False` plots never read the stored influences. The dfbeta
  sweep skips rows outside the risk set when the moments are finite (their
  contributions are signed zeros) and accumulates into one cluster by column by
  time buffer.
- Yates survival setups fill one N by (T + 2) result row by row with constant
  scratch space, and response simulations reuse one predictor buffer and
  accumulate the covariance online without keeping each draw's responses.
- Evaluated R model-frame columns with numeric dtypes skip per-value boolean and
  string scans during classification.
- `interaction(..., drop=TRUE)` labels only observed combinations after adding
  each factor, with at most one combination per input row. It preserves declared
  level order, missing values and merged labels when separators occur within
  levels. `drop=FALSE` retains the requested Cartesian level set.
- O'Brien expansion sorts each covariate once per stratum and writes into
  precomputed output slices: O(P N log N + P M) work for N rows, P continuous
  columns and M expanded rows.
- Term grouping (`attrassign`) is one O(P + T) pass over P columns and T terms.
- Response repetition, `lvcf`, `neardate` and `tmerge` avoid per-row Python index
  lists and repeated sorts. `tmerge` converts only referenced update columns,
  matches update IDs once per call, and uses
  the interval split points to place events. The `tmerge_bench::split_covariate_updates`
  benchmark measures interval expansion with four updates per subject.
- `neardate` retains and sorts reference dates only for queried subjects when
  query rows are at most one sixteenth of reference rows. Larger query sets keep
  the single-pass reference index, avoiding extra hashing and string copies.
  `strata` materializes and factor-codes each shared input source once per call,
  including one-shot iterators.
- P-spline penalties fill the three nonzero entries of each second-difference row
  directly instead of a cubic dense product.
