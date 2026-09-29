# Direct Kaplan–Meier fitting

`survival.r.survfitKM` fits a prepared factor and right-censored or
counting-process `Surv` response. It shares the formula interface's Rust
engine and returns numerical curves without retaining observations or a
model frame.

```python
from survival import r

x = r.strata(["control", "control", "treated", "treated"], shortlabel=True)
y = r.Surv([1, 3, 2, 4], [1, 0, 1, 1])
fit = r.survfitKM(x, y, conf_type="log-log")
print(fit.strata, fit.time, fit.surv)
```

## Prepared inputs

`x` must be a `StrataFactor` from `r.strata`, a pandas categorical, or a
categorical Series. Ordinary lists are refused, matching R's requirement
for an explicit factor. Declared level order and unused levels are preserved.
`y` must be a `Surv` with type `right` or `counting`.

Inputs must already be aligned and complete. This interface does not build
a formula model frame, omit missing rows, or merge almost equal times.
Use `r.survfit` for formula processing and its default `timefix=True` behavior.
Weights must be finite and nonnegative. Optional `id` and `cluster` labels
must match the response length; empty vectors are treated as absent.

`stype`, `ctype`, weights, robust variance, influence matrices, entry counts,
confidence levels/types/lower limits and `start_time` use the common native
engine. The old `type` argument overrides `stype` and `ctype`, including
otherwise invalid values. `time0` is accepted and unused, as in R's bare
fitter. Dotted aliases such as `se.fit` and `start.time` are accepted through
keyword dictionaries.

The unweighted `counts` table follows R's relative comparison of weights
with one, using the square root of machine epsilon. Exactly equal weights
are excluded from that comparison's mean. This only controls the presence
of the table: fitting retains the original weights, and fractional weights
still trigger the usual default robust variance.

## Result and R bridge

`SurvfitKMResult` exposes `n`, `time`, `n_risk`, `n_event`, `n_censor`, `surv`,
`cumhaz`, standard errors, confidence limits and metadata, plus optional
`n_enter`, `counts`, `n_id` and influence matrices. Numeric property access
returns independent copies. Influence values are read-only NumPy views of
native storage. Results support pickling.

`n` and `n_id` include zeros for unused or filtered-out levels. `strata` is
`None` for one declared level; otherwise it maps every declared level to its
number of output times, including zeros. Influence components are lists in
declared level order, with `None` for empty curves. Their row labels identify
clusters, subjects, or the original observations. These lists are consistent
with other Python curve results even for a single curve.

The R package's `survfitKM` delegates to this interface and converts output
shapes and names. It preserves R's list-field order, count-matrix column names,
influence row names, and single-matrix versus multiple-matrix distinction.
All fitting and confidence calculations run in Rust.

Two deliberate corrections avoid failures in R 3.8-12's bare interface:

- Default weights and absent `id`/`cluster` are resolved before `start_time`
  filtering. R's lazy default weights can instead be evaluated after filtering,
  producing the wrong length, and R can access missing `id`/`cluster` arguments.
- Unused levels receive empty influence components. Some R combinations try
  to assign row names to a missing matrix and fail.

This is a bare numerical result, corresponding to R's unclassed fitter list.
Use `r.survfit` when you need the formula result's summary, plotting, residual
or prediction methods.

## Performance and validation

`scripts/benchmark_km_lowlevel.py` compares calls on the same prepared response
and factor, checks numerical agreement and measures serialized output size.
A local CPython 3.14 release-build run with 100,000 observations, eight curves,
one warmup and seven measured calls per mode produced:

| Result | Median call | Serialized size |
| --- | ---: | ---: |
| Full interface, `timefix=False` | 50.31 ms | 18,504,873 bytes |
| Direct interface | 11.35 ms | 8,000,443 bytes |

The direct call was about 4.4 times faster with 57% smaller serialized output
for this workload. Both paths use the same native fitting algorithm. Savings
come from skipping model-frame processing and full-result assembly. Input
construction, disposal of the previous result, property access for comparison
and serialization are excluded from the timings. Serialized sizes measure
retained output, not peak memory.

`scripts/generate_km_lowlevel_reference.R` generates 80 cases from R 4.5.3 /
survival 3.8-12. They cover both survival and hazard estimators, all confidence
types and lower-limit methods, ties and almost equal times, weights, robust
variance, clusters, repeated subjects, influence modes, entry counts, unused
levels, start-time filtering and validation. The generator supplies explicit
weights and absent identifiers to avoid R's lazy-default errors. Separate R
bridge tests compare complete lists and their field order, and check the
corrected empty-level and omitted-argument cases.
