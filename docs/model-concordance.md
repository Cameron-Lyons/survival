# Concordance for fitted survival models

The R method `concordance(fit, ...)` now delegates model preparation and
comparison to the same Python implementation used by `survival.r.concordance`.
The Rust kernel computes the counts, variance, influence functions and event
ranks. Fitted responses and predictors remain in Python instead of travelling
through R to construct another response and score matrix.

The previous R wrapper omitted fitted strata and clusters. On a twelve-row
regression case, this changed stratified concordance from 0.7 to 0.6532258,
and clustered variance from 0.006479042 to 0.01047736. It also used the first
response for every model in a joint comparison and restored omitted rows
through `predict`, which could misalign `na.exclude` fits.

Each model now supplies its own retained response, stored linear predictor,
strata, weights and applicable clusters. New-data responses, predictors,
offsets and strata are omitted together when incomplete; training weights
and clusters do not carry into new-data scoring. Explicit clusters must match
the retained rows and replace fitted clusters. AFT formula `cluster()` terms
follow the existing Python/R convention and are not implicit concordance
clusters. Differing model responses emit an R warning on every call, while
mismatched sample sizes, weights or cluster counts are refused.

Single-model results keep per-stratum count tables. Joint results pool each
model's own strata into a count row and form the covariance from the
cross-product of the already weighted, optionally clustered influence vectors.
This retains the documented correction to R's second multiplication by weights.
For joint models, `influence=1` returns `dfbeta`, `influence=2` returns the
three-dimensional influence array, and `influence=3` returns neither, following
R's fitted-model method. A single model with `influence=3` returns both.

The bridge retains R model names, count labels and cluster ordering. Fitted
categorical clusters now retain their declared level order through row selection
and omission, as explicit factor clusters do. Joint influence rows align by
cluster membership, independent of names or level order; incompatible partitions
are refused. See [cluster alignment](concordance-clusters.md) for the covariance
correction and its validation. Rank tables contain event
rows without R's original model-frame row labels. New-data stratum counts keep
the fitted labels instead of R's integer codes.

Existing differences remain intentional: `timefix=FALSE` is honored, AFT
new-data predictions include formula offsets, stratified ranks work with
censoring, and pooled count tables avoid R's `colSums` error. AFT training
concordance requires a fit retaining its response (`y=TRUE`).

## Validation

The focused R tests reproduce the original failures and compare numerical
results with R 4.5.3 / survival 3.8-12 across ordinary and ridge-penalized Cox
and AFT fits, counting-process responses, weights, offsets, strata, clusters,
time weights, bounds, influence modes, multiple models, missing rows and
new data. They check joint weighted covariance against independent single-fit
influences and stratified ranks against independent per-stratum calculations.
Known reference bugs use those explicit calculations rather than the broken
reference dispatcher. A test disables R's numerical routine while scoring
models to verify that computation stays in the shared implementation.

At commit `7d5c8aeb`, the 398 focused checks passed. The package archive passed 7,023 R checks with
zero errors, warnings or notes; the full Python suite passes 14,413 tests,
with 48 skips and 37 documented expected differences. Pinned lint/format,
generated-interface checks and Mypy across 47 files also pass. The Rust
numerical code is unchanged from the previously validated implementation.

## Complete-call benchmark

`scripts/benchmark_model_concordance.R` verifies results before timing,
performs three warmups, alternates call order and records every sample. Model
fitting, input construction and explicit garbage collection are excluded;
preparation, conversion, calculation and result assembly are included.

A local release-build run on 50,000 rows produced these medians and ranges
across seven samples (milliseconds):

| Cox concordance call | Previous wrapper | Shared wrapper | Stock R |
| --- | ---: | ---: | ---: |
| One fitted model | 88 (85–100) | 21 (21–22) | 44 (43–45) |
| Weighted model | 93 (92–103) | 23 (23–24) | 55 (54–59) |
| Weighted model on new data | 95 (93–107) | 84 (84–87) | 79 (78–83) |
| Two fitted models | 149 (144–162) | 61 (60–62) | 91 (91–92) |
| Stratified and clustered model | Incorrect result | 34 (33–34) | 62 (61–62) |

The previous method was taken from commit `f65388a8` and checked only on
cases where it returned correct numerical results. The comparison normalizes
model names and the old joint `cvar` column-matrix shape. The run used
Python 3.14.7, R 4.5.3 and survival 3.8-12. These are workload-specific timings;
new-data scoring remains slower than stock R here. Peak memory was not measured.

```sh
RETICULATE_PYTHON=$PWD/.venv/bin/python PYTHONPATH=$PWD/python \
  Rscript scripts/benchmark_model_concordance.R 50000 7
```

An optional third argument names an R source file containing the previous
`concordance.survival_py_model` method and its response helper; the script then
checks and times that baseline on the four ungrouped cases too.
