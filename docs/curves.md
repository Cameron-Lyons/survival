# Survival curves

This page covers fitted survival and multi-state curves: fitting options, time
handling, summaries, quantiles, confidence bands, selection, aggregation,
transition-matrix curves, residuals and pseudo-values, restricted-mean
comparisons and log-rank tests. Reports and plots of curves are described in
[Printing and graphics](printing-and-graphics.md); differences from R are listed
in [R compatibility](r-compatibility.md#survival-curves).

## Fitting curves

`r.survfit` fits Kaplan–Meier, Turnbull (interval-censored), Aalen–Johansen
(multi-state) and Cox curves. Formula fits honor `conf_level=`, R's
`conf_type=` choices and `start_time=` for conditional curves; `survfit0(...)`
adds the starting row. Right-censored `Surv(time, event)` and counting-process
`Surv(start, stop, event)` data are supported, with delayed-entry risk sets.
Factor-valued events produce Aalen–Johansen curves, with subject histories
through `id=`, observed initial states through `istate=`, event-type conversion
through `etype=`, initial distributions through `p0=` and entry counts through
`entry=True`. With `entry=True`, reporting times include subject entries,
transitions and final exits, omitting censored continuation boundaries.
Multi-state fits support `stype=1` and `ctype=1`, as in R; with `se_fit=False`
no robust-variance workspace is allocated. A fitted Cox model with optional
`newdata` gives model-based curves, and a multi-state Cox model gives R's
`survfit.coxphms` state-probability curves.

`surv_analysis.survfitkm` is the native right-censored and counting-process
kernel (`ctype=1` gives the Nelson–Aalen hazard); it accepts NumPy arrays, pandas or
Polars columns and sequences, require binary status (integral floats are
converted with a check) and release the GIL.

### Times

Formula, response-based and direct KM fits, native `survfitkm` and the Rust
`SurvfitKMData` accept infinite time endpoints. Right censoring permits either
sign; counting intervals still need `start < stop`. Missing times, invalid
statuses and nonfinite weights are refused. KM `start.time` also accepts
infinities: negative infinity keeps all rows, positive infinity keeps only
infinite stop times or reports that all rows were removed. Other routines, such
as Cox fitting, keep their finite-time contracts.

`timefix=True` (the default) merges near-tied times with R's `aeqSurv` tolerance;
actual near ties collapse to the earliest time. Nonfinite endpoints stay in their
own rows with their statuses. When summing large finite times would overflow, the
mean magnitude is computed with scaling, so distinct times such as
`[1e308, 1.1e308, 1.2e308]` stay distinct and counting intervals keep their
endpoints. The same normalization serves Kaplan–Meier, Cox and log-rank fits.

Counting-process KM curves keep every event time, including events on internal
intervals of a subject. Initial interval boundaries without events or terminal
censoring are omitted from the reporting grid, following R 3.8-12. Counting
curves with entry reporting can keep entry rows before a supplied `start.time`;
if inserting that origin makes the times unsorted, summaries return an input
error, as R does.

## Summaries at requested times

`r.summary_survfit` and `r.model_summary` take R's `times`, `censored`, `extend`,
`scale`, `rmean` and `dosum` arguments for Kaplan–Meier, Turnbull, Cox and
multi-state curves.

```python
from survival import r

fit = r.survfit(r.Surv([1, 2, 3, 4, 5], [1, 0, 1, 0, 1]))
intervals = r.summary_survfit(fit, times=[2, 4], dosum=True)
at_times = r.summary_survfit(fit, times=[2, 4], dosum=False)
```

With `dosum=True`, event, censoring and entry counts accumulate between requested
times, the first count including all earlier observations. With `False`, they
come from the last curve row at or before each time. Survival, cumulative hazard
and uncertainty use that step-function row in both modes. Risk counts use the
first row at or after the requested time and are zero past the end of the curve.

Ordinary curves keep query order and duplicates, and accept infinite query times;
their default accumulates only for strictly increasing times, and requesting
accumulation for unordered or repeated times raises R's error. Multi-state curves
sort and deduplicate requested times and default to accumulation; transition
counts follow the same selection. Without requested times, `dosum` has no effect.

Multi-state `summary_survfit` returns probabilities, state-specific counts,
confidence limits and a table of restricted mean time in each state, with the
common, individual, numeric and omitted restricted-mean options. A requested time
before the first curve time reports `p0`.

Rust callers use
`surv_analysis::summary_survfit_times_with_counts(fit, times, extend, dosum)`,
where `dosum: Option<bool>` selects accumulation, lookup or the ordinary default,
and `summary_survfit_aj_with_counts(fit, times, censored, extend, dosum)` or
`surv_analysis::{summary_survfit_aj, survmean_aj}` for multi-state curves. The
native Python `kernels.summary_survfit(..., dosum=None)` and
`SurvfitAJResult.summary(..., dosum=True)` accept NumPy time vectors and release
the GIL.

`r.as_data_frame` of a curve keeps fitted terminal uncertainty and confidence
limits: robust fits can have zero standard errors and zero bounds at survival
zero, and these stay zero, also in R curve quantiles, while undefined native
values stay missing and logarithmic standard errors still convert to the survival
scale. Grouped
multi-state frames are state-major, then stratum-major, as in R; repeated or
reordered state selections keep their columns by position.

## Compact tables and restricted means

`surv_analysis.survmean` computes the compact table behind ordinary curve
summaries and reports: records, risk counts, events, restricted means and their
standard errors, and medians with confidence limits. Restricted means accumulate
the observed curve forward and their variance backward, keeping R's addition
order, scaling and truncation conventions. Infinite time-zero rows keep IEEE
arithmetic, including undefined areas from `Inf - Inf`. The binding releases the
GIL.

## Quantiles and medians

`r.quantile` and `r.median` (also `quantile_surv`, `median_surv` and
`median_survfit`) accept fitted KM, Turnbull and Cox curves and raw `Surv`
responses. Raw responses are first fitted with the KM or Turnbull engine; they
reject missing rows unless `na_rm=True` and refuse multiple-endpoint responses, as
R does. `median(Surv(...))` includes confidence bounds by default, while
`median(survfit(...))` returns point estimates. Results have one row per curve
(grouped fits and Cox prediction columns each give a row) and one column per
probability, and are always a `QuantileSurvfitResult`. Conditional Cox
predictions use their start time as the curve origin; ordinary KM quantiles keep
R's zero origin even when fitting starts later.

- Probabilities follow R's numeric-vector contract: numeric scalars, lists, NumPy
  arrays and numeric pandas or Polars containers are accepted, empty vectors give
  an empty probability dimension, and repeated values and order are kept.
  Logical, character, categorical and missing probabilities are refused, as are
  values outside [0, 1]. NumPy `b`, `c`, `m`, `M`, `S`, `U` and `V` dtypes, pandas
  boolean, string, categorical, datetime and timedelta columns, and Polars
  boolean, string, categorical, Enum, temporal, binary and null columns are
  refused; object arrays of numbers are accepted.
- `scale` is a numeric scalar (or a single-valued container), applied after
  inversion with IEEE arithmetic: zero gives an undefined result at time zero and
  signed infinities elsewhere, infinity gives signed zeros, and missing gives
  missing results. Character scales are refused.
- The default tolerance is the square root of machine epsilon. Positive infinite
  tolerance gives undefined quantiles; negative infinity keeps the origin at
  probability zero and gives undefined quantiles at positive probabilities. NaN
  tolerance is refused. With the default tolerance, an entirely censored curve
  has an undefined quantile even at probability zero.
- Non-monotone bands, duplicate levels, missing band entries and
  tolerance-induced ties use R's sorting and averaged indices.
- Multi-state curve quantiles and medians are refused, as in R.

The native `quantile_survfit` and `quantile_survfit_curves` bindings take owned
numeric arrays, copy stacked time, survival, band and probability vectors in bulk
and release the GIL. In the R bridge, an unstratified single curve gives a vector,
a stratum dimension or several prediction columns give a matrix, and strata with
several prediction columns give a stratum-by-curve-by-probability array; each
prediction column is inverted within each stratum. When bands exist, `conf.int` is
evaluated with R's conditional rules; when they are absent, it is not evaluated.

## Confidence bands

`kernels.survfit_confint` (Rust `surv_analysis::survfit_confint`) computes
confidence limits for a vector of survival estimates. The R-style
`survival.r.survfit_confint` also accepts scalar estimates, recycles a single standard
error or lower-band error, and keeps missing values in list, NumPy, nullable
pandas, Polars and masked inputs:

```python
from survival import r

bands = r.survfit_confint(
    [0.8, None, 0.3], [0.1, 0.2, 0.15], conf_type="log-log"
)
```

The transforms are `plain`, `log`, `log-log`, `logit` and `arcsin`.
`logse=True` means standard errors are on the log-survival scale, `selow` widens
the lower bound, and `ulimit=False` removes the upper cap for plain and log
intervals. The standalone function rejects `conf_type="none"`; curve fitters
accept it to omit bands. Arrays cross in bulk and the native binding releases the
GIL.

## Selecting curves

Formula curves expose `curves.subset(strata=..., drop=...)` and, for multi-state
curves, `curves.subset(strata=..., states=..., drop=...)`, R's `fit[...]`, with
labels or zero-based indices in the order given; `curves.dim` is R's `dim(fit)`.
A single selected stratum loses its label unless `drop=False`.
`summary_survfit(curves, times=..., data_frame=True)` (or `as_data_frame` of a
summary) returns R's `summary(fit, data.frame = TRUE)` table: one row per time
and curve, long by state for multi-state curves.

Ordinary Cox curves expose `curves.subset(strata=..., data=..., drop=...)`.
Python positions are zero-based; stratum labels and prediction-row names can also
select their margins. The R bridge accepts `curves[strata, data, drop=FALSE]`.
Reordering, repeated selections and empty margins keep counts, curve arrays,
confidence components and prediction metadata; repeated stratum display names
remain usable in summaries, quantiles and later selections. `drop=True` removes
singleton margins with R's rules, including keeping a matrix when the selected
stratum has one time row. A single R subscript on an object with both margins
selects cells in column-major order and concatenates them. Single curves without
an explicit margin follow R's raw subscript rules; no-op selectors keep
`start.time` and actual selections remove it.

Multi-state Cox curves expose `curves.subset(strata=..., data=..., states=...)`;
the R wrapper uses its bracket syntax. Select by zero-based indices, stratum
labels or state names; omitted dimensions keep all values. Selected curves keep
their count engine, so `summary_survfit`, `survfit0`, `as_data_frame`, reports and
aggregation stay available. Original state names stay in `oldstate`;
probabilities are not renormalized. Arrays and mutable count rows are
independent of the source. A state selection drops transition hazards
and counts unless every state remains in its original order. `n_censor` follows
the selected state columns, and repeated stratum labels gain suffixes such as `b`,
`b.1`.

```python
death = curves.subset(states=["death"])
summary = r.summary_survfit(death, times=[100, 200])
```

## Aggregating curves

`r.aggregate_survfit` summarizes the prediction columns of survival and
multi-state probability curves at each time, following R's `aggregate.survfit`.
It supports `mean`, `median`, `min`, `max`, `sum` and custom scalar summaries,
with the first grouping variable varying fastest; only observed combinations give
output columns. Iterator and generator `by` columns are read once.

```python
import numpy as np

totals = r.aggregate_survfit(curves, by=groups, FUN="sum")
spread = r.aggregate_survfit(curves, by=groups, FUN=lambda values: np.ptp(values))
```

- Omitting `FUN` keeps R's default distinction: ungrouped ordinary curves (or a
  constant grouping vector) use `rowMeans`, while explicit `"mean"` uses R's
  residual-refined mean. Multi-state probabilities and several groups always use
  the mean reducer.
- As in R, callbacks first receive each group's one-based prediction-row indices;
  all those results are collected before checking that each is a numeric scalar.
  Survival values then arrive in time/group order, followed by state
  probabilities in state/time/group order, each group in original row order. Extra
  keyword arguments are accepted and ignored, so a callback that requires another
  argument raises its own missing-argument error.
- Every Python call receives a fresh one-dimensional float64 array. Results must
  be real numeric scalars or one-element numeric arrays (NaN and infinities
  allowed); boolean, string, list and vector results are rejected, and a callback
  that changes its return length gets an error. Callback exceptions stop
  immediately with their type and message.
- Uncertainty components and cumulative hazards are dropped (counts, times,
  strata and states are kept). Aggregation without `by` keeps a data axis of
  length one.
- Mean and sum use compensated arithmetic for ordinary values; exceptional groups
  replay R's sequential extended-precision arithmetic, preserving signed
  infinities, subnormals and cancellation across intermediate overflow. Ordinary
  results may differ from R in the last bits.
- Empty time or state axes give shaped empty results. Zero data columns and
  missing grouping values are rejected before callbacks.

Rust callers use `aggregate_survfit_with(surv, pstate, by, callback)` with a
closure taking `&[f64]` and returning `SurvivalResult<f64>`; Python's native
`aggregate_survfit(..., fun=callback)` accepts the same callbacks. Built-in names
release the GIL; Python callbacks keep it. The multi-state binding accepts an
owned three-dimensional float array or nested lists. The R bridge accepts a
function or function name, passes callback indices as R integers and keeps the
original R group table, including factor order and unused declared levels, in
`newdata`.

## Curves from transition matrices

`survfit` accepts a square matrix whose cell `[i][j]` is a Kaplan–Meier or Cox
curve for the transition from state `i` to state `j`, with `None` for absent
transitions (R's `survfit.matrix`). All cells must have the same curve class and
numbers of strata and prediction columns, and at least two transitions are
required.

```python
from survival import Surv, survfit

healthy_to_ill = survfit(Surv([1, 2, 3, 4], [1, 0, 1, 0]))
ill_to_dead = survfit(Surv([2, 3, 4, 5], [0, 1, 0, 1]))
fit = survfit(
    [[None, healthy_to_ill, None],
     [None, None, ill_to_dead],
     [None, None, None]],
    p0={"healthy": 1.0, "ill": 0.0, "dead": 0.0},
    method="matexp",
)
print(fit.time, fit.pstate)
```

- `p0` defaults to probability one in the first state. A vector is used for all
  curves; a matrix supplies one row per output curve (strata fastest, then Cox
  prediction columns). A mapping names the states; otherwise names default to
  `"1"`, `"2"`, ..., or come from `states=`. The `p0` sum tolerance is `1e-8`.
- Both methods use increments of the input cumulative hazards. `discrete` (the
  default, R's actual dispatch for KM and Cox input) multiplies by `I + dA` and,
  as in R, caps the diagonal departure probability at one without rescaling the
  off-diagonal increments, so probabilities can sum above one; `matexp` multiplies
  by `exp(dA)`.
- The time grid is the union of transition event times. `start_time` excludes
  times at or before that value and applies `p0` there, subtracting the earlier
  hazard increments; the default start is `min(0, event times)`.
  `time0=True` adds the initial row; an empty event grid gives an empty curve with
  its `p0`.
- The result is a `SurvfitMultiStateResult` without standard errors or influence.
  R's risk and event counts are kept, including the stepwise reuse of the previous
  event count; when several transitions leave one state, the last in column-major
  order supplies its risk count. `cumhaz` holds the input transition hazards and
  `n_transition` their counts; censor counts are zero and `n_id` is unset.

Rust callers use `surv_analysis::survfit_matrix` with borrowed
`SurvfitMatrixTransition` values, state names, an optional initial-probability
matrix, `SurvfitMatrixMethod` and an optional start time. Each transition gives
its zero-based source and destination states and a slice of `SurvfitKMResult`
values, one per prediction column; the result is a
`SurvivalResult<SurvfitAJResult>`. The domain function
`surv_analysis.survfit_matrix` takes nested lists of native KM results and
releases the GIL.

## Residuals and pseudo-values

Ordinary and multi-state fitted curves support R-style influence residuals and
pseudo-values through `r.residuals_survfit` and `r.pseudo`: survival, cumulative
hazard and restricted mean time for ordinary curves, and state probabilities,
cumulative transition hazards and integrated occupancy for multi-state curves.
They keep case weights, subject ids, grouping, estimator settings and conditional
start times; grouped, weighted and subject-collapsed counting-process results are
supported. The fitted restricted-mean path follows R's infinitesimal jackknife;
the direct time/status pseudo-value API keeps its delete-one calculation. Tied
`ctype=2` diagnostics follow R's approximation.

KM and Aalen–Johansen fits keep their missing-data action as `fit.na_action`
(one-based positions after any subset), which `survfit0`, copies and pickle keep:

| Output | `na.omit` | `na.exclude` |
| --- | --- | --- |
| Observation residual matrix or array | Retained rows | Original row count, omitted rows NaN |
| Observation pseudo-value vector, matrix or array | Retained rows | Original row count, omitted rows NaN |
| Collapsed subject output | Retained subjects | Retained subjects |
| Long table (`data_frame=True`) | Retained rows | Retained rows |

Without an explicit id, labels keep the positions before omission: omitting
observations 2 and 6 gives ids `[1, 3, 4, 5, 7, 8]`. Excluded rows are labeled by
their omitted position and their `curve` entries are NaN. As in R, `collapse=True`
without repeated clusters or ids leaves the output at observation level. A
pseudo-value request beyond every curve's endpoint warns.

```python
fit = r.survfit(
    "Surv(time, status) ~ 1",
    {"time": [1, None, 3, 4], "status": [1, 0, 1, 0]},
    na_action="na.exclude",
)
result = r.residuals_survfit(fit, times=[2, 3])
```

## Restricted-mean comparisons

`validation.rmst_comparison` (Rust `validation::rmst_comparison`) fits a
Kaplan–Meier curve per group, computes its restricted mean and uncertainty up to
`tau`, and compares the groups with the smallest label as reference. Observation
vectors must have equal lengths. `rmst_comparison` and `survmean_curves` accept
NumPy vectors and release the GIL.

## Log-rank tests

`r.survdiff` and `kernels.survdiff` (Rust `surv_analysis::survdiff`) compute the G-rho family. `rho=0` is the log-rank test;
nonzero `rho` weights events by the left-continuous Kaplan–Meier survival of
their stratum, `S(t-)^rho`. The native functions also accept counting-process
data, with the forward KM risk count at `t` equal to the number of starts below
`t` less the intervals ending before `t`; the formula interface refuses
counting-process responses, as R's `survdiff` does. Near-tie normalization runs
first and rejects intervals whose length becomes zero.
