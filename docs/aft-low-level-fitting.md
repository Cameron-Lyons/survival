# Bare parametric survival fitting

`survival.r.survreg_fit` fits an explicit design matrix and a prepared response
using the same Rust solver as the full AFT model. It returns coefficients,
the intercept-only coefficients (`icoef`), variance, both log likelihoods,
iteration count, linear predictors, parameter count (`df`) and final score.
Estimated `Log(scale)` parameters remain at the end of the coefficient vector.

```python
import numpy as np
from survival import r

y = np.array([[1.2, 1], [2.5, 0], [.9, 1], [3, 1], [1.8, 1], [2.7, 0]])
x = {"Intercept": np.ones(6), "group": [0, 1, 1, 0, 1, 0]}
fit = r.survreg_fit(x, y, dist="gaussian")
assert fit.coefficient_names == ("Intercept", "group", "Log(scale)")
```

The response is a two-column `(time, status)` matrix or a three-column
`(time, time2, status)` matrix. Status codes are 0 for right censoring, 1
for exact observations, 2 for left censoring and 3 for intervals. As in
R's bare function, the numeric columns of a `Surv` are also accepted without
status recoding: a left-censored `Surv` uses 0/1 internally, so prepare a
matrix with code 2 to fit those observations as left censored. Counting and
multistate responses are refused. No rows are omitted automatically.

Responses must already be on the fitting scale. For example, a Weibull
fit uses logged times and `dist="extreme"`; `dist="weibull"` is rejected
because that R distribution entry has no density of its own. The supported
base names are `extreme`, `logistic`, `gaussian` and `t`. Exact name lookup
matches R. High-level `r.survreg` performs response transforms and adds their
Jacobian to the log likelihood; the bare function does neither.
The Student-t name requires explicit `parms` (degrees of freedom); R's bare
fitter also requires it, but reports an unhelpful length-zero error when it
is omitted. Native distribution objects retain their stored parameters.

`scale=0` estimates scales, and a positive value fixes the scale. `nstrat`
specifies the number of scale strata, including unused strata. When it is
greater than one, `strata` must contain one-based integer codes. With
`nstrat=1`, R ignores the supplied codes; this interface does likewise.
Fixed scales cannot be combined with multiple strata. Controls use
`survreg_control`, passed through `controlvals`. `assign` is accepted and
unused, as in R.

Design matrices can be NumPy arrays of any layout, nested numeric rows,
mappings or data frames. Names come from mappings/data frames or explicit
`column_names`; otherwise they use R's `x 1`, `x 2`, etc. Metadata is stored
in `coefficient_names`, `icoef_names` and `variance_names`. R drops variance
dimension names when undoing covariate rescaling, and the last field is
`None` in that case. The score vector retains R's internal covariate scaling.
Aliased coefficients retain the solver's values; full-model alias marking
belongs to `survreg`, not this interface.

Custom mappings require a density callback returning five columns and an
initialization callback when starting values need it. They need no quantile
or deviance callbacks. Response transforms and distribution-level fixed
scales are ignored. Callback validation occurs on the actual fit inputs;
the bare path does not run the full distribution probe. Native distribution
objects may also be supplied; their underlying density is used. Explicit
callbacks are honored regardless of the display name, matching this port's
existing custom-distribution policy.

The immutable `SurvregFitResult` stores numeric output and name tuples;
array access returns copies. It retains no response, design, distribution
or callbacks. Results therefore remain pickleable even when a custom
distribution contains local functions, and do not support model prediction.

## Rust API and retention

Rust exposes `regression::SurvregFitResult::fit`; Python's native counterpart
is `survival.regression.survreg_fit_raw`. Both accept `SurvregData` and a
`SurvregDistribution`, ignore the distribution's response transform and
fixed scale, and return the bare numerical components. An optional explicit
stratum count can preserve unused scale strata. Cluster variance is refused.

The bare and full paths share initialization, matrix rescaling and Newton
iterations. Only the full path copies the design into a retained model,
computes optional robust variance and marks aliased coefficients. Native
fitting releases the GIL; custom Python callbacks reacquire it for each
vectorized call. The common Python matrix converter avoids creating lists
for every NumPy row and is shared with the bare Cox interfaces.

`scripts/benchmark_aft_lowlevel.py` measures both native binding calls using
the same prepared data and checks numerical agreement. It reports serialized
result sizes as a reproducible measure of retained output, not peak memory.

One local CPython 3.14 run with 100,000 rows, six covariates, one warmup and
seven measured calls per mode produced these medians:

| Result | Binding call | Serialized size |
| --- | ---: | ---: |
| Full model | 70.57 ms | 7,400,786 bytes |
| Bare fit | 71.80 ms | 800,664 bytes |

Runtime was similar (the bare median was 1.7% higher); serialized output was
about 89% smaller. Both modes perform the same numerical fitting work.
Data construction, copying result properties and serialization are excluded
from these times. Results depend on workload and hardware.

## R references

`scripts/generate_aft_lowlevel_reference.R` regenerates 75 cases from
R 4.5.3 / survival 3.8-12. These cover all four base distributions, weights,
offsets, fixed/stratified scales, unused strata, initial values, iteration
limits and warnings, intercept-only and singular designs, interval
censoring, names, validation and minimal custom callbacks. Additional
tests cover matrix layouts, ownership, pickling, response conventions,
native/full-model agreement and GIL release.
