# AFT and Aalen model reports

The R-style API provides formatted reports and full-precision data for
accelerated-failure-time (`survreg`) and Aalen additive (`aareg`) models.
Reports use fitted values and existing summaries without refitting.

```python
from survival import datasets, r

fit = r.survreg("Surv(time, status) ~ age + sex", datasets.load_lung())
print(r.print_survreg(fit))
summary = r.model_summary(fit, correlation=True)
report = r.print_summary_survreg(summary)
print(report)
columns = r.as_data_frame(report)
correlation = report.tables["correlation"]

additive = r.aareg(
    "Surv(futime, fustat) ~ age + ecog.ps", datasets.load_ovarian(), dfbeta=True
)
print(r.print_aareg(additive, maxtime=400, scale=365.25))
print(r.print_summary_aareg(r.model_summary(additive, test="nrisk")))
```

Ordinary reports return `ModelPrint`, also used by the Cox printers.
`tables["coefficients"]` retains the numeric table before display rounding;
`statistics` contains the model's numerical footer fields. AFT summaries with
correlations also retain the complete `tables["correlation"]` matrix.
`as_data_frame` returns independent coefficient columns and a `term` column.
Constructing a report does not write to stdout. `lines` holds the text, and
`str(report)` adds a final newline. Report edits do not change the original
fit or summary.

## Accelerated-failure-time fits

`print_survreg(fit, digits=None, width=80)` displays the location coefficient
vector, fitted or fixed scales, both log likelihoods, a likelihood-ratio test
when its degrees of freedom are positive, and the sample size. Aliased
coefficients receive R's singularity notice. Missing-row notices retain the
fit's `na_action`.

The default is seven digits. As in R's `print.survreg`, an explicit `digits`
changes the coefficient vector and multiple named scales; scalar scales and
likelihood text still use seven digits. `width` wraps vectors into blocks.

`print_summary_survreg(summary, digits=None, signif_stars=False, width=80)`
adds standard errors, z statistics, p-values, the distribution description,
iteration counts and optional coefficient correlations. It defaults to three
digits for the coefficient table and scales. Robust fits display robust and
naive standard errors and identify the likelihood's independence assumption.
Likelihood and test-footer formatting follows R's own precision rules.

With `model_summary(fit, correlation=True)`, the report displays the lower
triangle and retains the full correlation matrix. Aliased coefficients are
excluded from that matrix, with the remaining labels kept aligned. Summary
metadata includes `fixed_scale` and `scale_names`, so stratified scales retain
their labels without needing access to the original fit. Older summaries can
infer them from their coefficient table.

Penalized inputs dispatch to the existing `print_survreg_penal`, returning
`SurvregPenalPrint`. Its `rows` and `columns` retain full-precision term tests,
and it now also supports `width` and `as_data_frame`. Use the explicit method
for `terms=True` or `maxlabel` controls. A summary of a penalized fit still
uses `print_summary_survreg`, matching R; it displays the model's coefficient
table and both iteration counts.

## Aalen additive fits

`print_aareg(fit, maxtime=None, test=None, scale=1, width=80)` shows the sample
size, used and total unique event times, coefficient slopes, weighted tests,
standard errors, z statistics and p-values. Stored influence estimates add
robust standard errors. The footer contains the overall chi-square test.

`maxtime` includes every event at or before the cutoff. `test` selects `aalen`
or `nrisk` weights, defaulting to the fit's test. `scale` changes the displayed
weighted coefficients and their standard errors. Slopes and tests retain their
original units, as in R. Zero scale and cutoffs before all events raise clear
errors. A fit using `test="variance"` needs an explicit supported summary test;
R's summary method likewise accepts only `aalen` and `nrisk`.

`print_summary_aareg(summary, width=80)` displays the same table and test
without the sample/event-count introduction. These methods follow R's fixed
three-significant-figure rounding followed by numeric matrix formatting at
seven digits. They do not offer a coefficient precision override. The direct
printer retains the original fit's test label in its footer when an override
was supplied, matching R; `statistics["test"]` always records the test actually
computed. The summary printer labels that computed test.

Changing `maxtime` or `test` now recomputes robust covariance from the stored
influences. Previously those summary options dropped the robust covariance and
silently used model-based uncertainty. Influences already aggregate clusters;
the reduction applies the weight at each distinct event time and accumulates
the cross-product of the resulting group scores.

## R differences and validation

The new `ModelPrint` reports omit original R call expressions and strip
trailing spaces. The established `SurvregPenalPrint` string representation
still includes its formula-only Call block; it cannot reconstruct other
original R call arguments. Options are explicit and do not alter global
settings. Formula labels use their stored Python spelling.

R 3.8-12's `aareg` computes test influences in event order, then groups them
with cluster identifiers in the original input order. Its default clustered
covariance therefore depends on input row order. The Rust fit keeps the rows
aligned. For four clustered reference reports, the fixture records R's raw
output and a reference obtained by refitting event-sorted input. Native tests
verify the covariance against that aligned R result and check invariance to
input row order. Reweighted summaries independently match R's correctly
grouped stored influences.

`scripts/generate_aft_aalen_report_reference.R` records 89 reports from 28
models with R 4.5.3 / survival 3.8-12. Coverage includes the built-in AFT
families, fixed and stratified scales, robust and aliased fits, missing values,
left censoring, penalties, correlation tables, and Aalen weights, ties,
delayed entry, clusters and truncated summaries. The 156 focused tests compare
native fits and independent R summary snapshots, check numerical covariance,
exercise conversion boundaries, and verify report ownership.

## Influence-reduction memory

The covariance reducer converts at most about 1 MiB of influence data per
block, except when a single group's coefficient-by-time matrix exceeds that
size. It reuses NumPy views and avoids copying complete list columns when
all event times are used. Neither a whole converted influence cube nor a
weighted copy of it is required.

```sh
PYTHONPATH=python .venv/bin/python scripts/bench_aareg_summary.py
```

One local Python 3.14.7 run with three coefficients and 400 event times gave:

| Groups | Stored layout | Bounded median | Full-conversion median | Bounded peak | Full-conversion peak |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | list | 1.29 ms | 1.23 ms | 0.99 MB | 0.97 MB |
| 2,000 | list | 28.65 ms | 28.68 ms | 1.09 MB | 19.46 MB |
| 2,000 | NumPy | 1.45 ms | 1.28 ms | 0.03 MB | 0.05 MB |

The benchmark checks numerical agreement before timing. Peaks use `tracemalloc`
and exclude the already-stored influences; they are not total process memory.
Blocking limits conversion memory for large list inputs, with some overhead
for small or already-contiguous arrays.
