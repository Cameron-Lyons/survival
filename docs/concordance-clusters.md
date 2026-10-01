# Cluster alignment for concordance

Joint fitted-model concordance now aligns influence rows by the observations
that belong to each cluster. Renaming a cluster or reordering a factor's levels
does not change its membership, so neither operation changes the covariance.
Models with different partitions are rejected with `models must have identical
clustering`, even when they happen to have the same number of groups. Supply
one explicit `cluster` vector to score all models with a common grouping.

Previously, the implementation and R's `cord.work` paired already aggregated
influence rows by position. In a 24-observation regression case, cyclically
renaming three identical groups changed the off-diagonal covariance from
0.0021763315 to -0.0040000919. The individual variances were unchanged, making
the error easy to miss when checking only each model separately. The R test
retains this incorrect reference covariance and independently verifies the
correct value by aligning the reference's single-fit influence vectors.

The shared preparation now compares partitions on the retained observation
rows, independently of group labels. Compatible fits are scored with the first
model's encoded groups. Thus returned `dfbeta` rows use the first model's
ordering, while event ranks and observation-level influence arrays keep their
existing conventions. An unclustered fit can be compared with singleton
clusters because both represent the same partition; the first model determines
the output order in that case too.

Cox model preparation also retains categorical cluster levels for formula
`cluster(...)` terms and `cluster=` vectors or column names. This metadata
survives subsets, missing-row omission, model retention and Python serialization,
including multistate Cox model objects. Multistate fitted-model concordance
remains explicitly unsupported. Plain character labels sort lexically, so
`"10"` comes before `"2"`; numeric labels sort numerically. Unused factor levels
never add influence rows. The R bridge uses its own ordering and labels for
explicit and fitted categorical clusters.

Training cluster values and levels are detached from caller-owned vectors.
New-data calls continue to use only an explicit clustering vector, matched to
the retained rows. Joint models still require their observations in matching
order; this change does not identify or reorder subjects across model frames.

## Verification

The new tests cover factor order, numeric-looking strings, unused levels,
missing rows, reordered subsets, list/NumPy/pandas input, ownership,
serialization, multistate metadata, explicit overrides, invalid groups,
singleton partitions and either model order. Independent observation-level
influences verify weighted and counting-process covariance. Direct R comparisons
check full fitted-model results and retain the reference row-position failure.

All 43 new Python cases and 48 new R checks pass. The full Python suite passes
14,456 tests, with 48 skips and 37 documented expected differences. The R source
archive passes 7,071 checks with zero errors, warnings or notes. Pinned lint and
formatting, generated interfaces and Mypy across 47 source files also pass.

## Performance

Cluster codes are prepared once for a joint comparison and reused by the score
calls. An explicit vector shared by all fits is validated once. Group-label
missingness is checked on distinct labels, avoiding a repeated scan of the same
values when determining their order.

`scripts/benchmark_concordance_clusters.py` checks output equality, warms each
path three times, alternates call order and records seven complete-call timings.
The previous source comes from `7d5c8aeb` and uses the same shared fitting and
native scoring modules. Its ordinary numeric-group cases return correct results;
the incorrect renamed-group case is excluded from baseline timing. Fitting,
input construction and explicit garbage collection are outside the timed calls.

One local Python 3.14.7 release-build run with 50,000 observations and 100 groups
produced these medians and ranges in milliseconds:

| Complete Python concordance call | Previous | Current |
| --- | ---: | ---: |
| One fitted clustered model | 27.726 (27.408–27.942) | 26.001 (25.827–26.166) |
| Two fitted clustered models | 55.230 (54.796–56.564) | 54.277 (53.889–55.804) |
| Two models, one explicit NumPy grouping | 48.582 (48.291–48.728) | 48.894 (48.683–49.680) |
| Two models with renamed groups | Incorrect covariance | 54.402 (54.090–55.404) |

These timings are specific to this workload. Joint ranges overlap, and the
explicit-group case is slightly slower. Peak memory was not measured. No Rust
numerical changes were needed.

```sh
git show 7d5c8aeb:python/survival/r/_concordance.py > /tmp/concordance-before.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_concordance_clusters.py \
  --rows 50000 --repeats 7 --baseline /tmp/concordance-before.py
```
