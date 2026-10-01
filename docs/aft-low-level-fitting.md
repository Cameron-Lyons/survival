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

Both ordinary and penalized fitters validate the response, design, weights,
offsets and grouping vectors at the fitting boundary. Rust callers can change
the public fields of `SurvregData` after construction, so constructor checks
alone are insufficient. A mismatched offset length or a design with no columns
returns an input error before initialization or matrix rescaling. Interval
upper endpoints need to be finite and at least the lower endpoint only for
status 3; the other rows' upper endpoints remain unused.

Scale-stratum counts are checked before callbacks or parameter allocation.
Unused strata remain supported, but dimensions whose dense covariance matrix
cannot be represented by the platform's array indexing are refused. This check
does not impose a memory budget or guarantee that every representable matrix
will fit in memory. Native strata are zero-based; cluster codes are opaque
labels and do not set array dimensions.

Python's `SurvregData` constructor owns its converted input buffers and releases
the GIL during validation. The full penalized binding borrows this prepared
data during fitting, avoiding an extra copy of all training buffers before
entering the solver. The fitted model still retains the data required for
prediction and residuals.

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

### Input validation and copy benchmark

[`bench_aft_inputs.py`](../benches/python/bench_aft_inputs.py) measures native
data construction, prepared fitting and construction plus fitting for ordinary
and penalized full/bare models. It also compares coefficients, initial values,
covariance, scores, linear predictors, likelihoods, iteration counts and
penalty outputs across saved extension builds.

On Python 3.14.7 / NumPy 2.4.6 with release builds and 20,000 weighted Gaussian
observations with offsets, these are medians and ranges in milliseconds from
seven samples after two warmups. Columns include the intercept. The baseline
is the extension used by #690, before fitting-boundary validation and removal
of the full penalized binding's extra training-data copy.

| Layout | Columns | Complete call | Before | After |
| --- | ---: | --- | --- | --- |
| C | 3 | Construct + ordinary full fit | 9.438 (9.380–9.466) | 9.797 (9.755–9.875) |
| C | 3 | Construct + penalized full fit | 17.451 (17.403–17.632) | 17.443 (17.401–17.580) |
| C | 16 | Construct + ordinary full fit | 21.796 (21.547–23.296) | 22.076 (21.496–23.213) |
| C | 16 | Construct + penalized full fit | 23.859 (23.598–24.153) | 22.661 (22.533–22.763) |
| Fortran | 3 | Construct + ordinary full fit | 9.641 (9.603–9.699) | 10.492 (10.356–10.541) |
| Fortran | 3 | Construct + penalized full fit | 17.738 (17.621–18.121) | 18.521 (18.224–18.757) |
| Fortran | 16 | Construct + ordinary full fit | 23.391 (23.241–23.603) | 23.703 (23.488–25.332) |
| Fortran | 16 | Construct + penalized full fit | 25.246 (24.485–25.658) | 24.218 (24.103–24.648) |

These measurements show modest, workload-dependent changes, including slower
small fits; they do not establish a general fitting speedup. Each build and
layout runs in a separate process, alternating routine order within each run.
Calls include conversion, validation, fitting and native result construction;
random input generation and copying result properties are excluded. Outputs
agree across builds at `rtol=5e-14, atol=1e-15`. This benchmark does not measure
peak memory; removing the binding copy avoids one duplicate of the training
buffers, while the solver's working storage and retained model remain.

```sh
PYTHONPATH=python .venv/bin/python benches/python/bench_aft_inputs.py \
  --extension /tmp/previous-survival.so --output /tmp/aft-before
PYTHONPATH=python .venv/bin/python benches/python/bench_aft_inputs.py \
  --output /tmp/aft-after --compare /tmp/aft-before.npz
# Add --order F to both commands for Fortran-layout inputs.
```

## R references

`scripts/generate_aft_lowlevel_reference.R` regenerates 75 cases from
R 4.5.3 / survival 3.8-12. These cover all four base distributions, weights,
offsets, fixed/stratified scales, unused strata, initial values, iteration
limits and warnings, intercept-only and singular designs, interval
censoring, names, validation and minimal custom callbacks. Additional
tests cover matrix layouts, ownership, pickling, response conventions,
native/full-model agreement and GIL release.

## R matrix bridge

`survivalr::survreg.fit` calls this same compact Python/Rust interface for
every density. It passes the design as a matrix, avoiding the previous
per-row R lists, and assembles the ordinary R result from the native fit's
names and numerical fields. It retains no training data or callbacks.
Convergence warnings are forwarded through R's condition system on each call.

Named built-ins execute entirely in Rust. An explicit distribution list uses
its R density and initialization functions, including when its display name
is `"Gaussian"`, `"Logistic"` or `"Extreme value"`. This follows the Python
interface's callback policy; R's original C fitter selects its built-in
density based on that display name. The density receives vector batches,
with the original R `parms` supplied when nonempty; the initializer receives
`y`, `weights` and `parms`, including `NULL`. Parameter names are preserved.
Distribution transforms, quantiles, deviance functions and stored fixed
scales are unused. Complete intercept-only starting values need no initializer.

There is no call to R's reference fitter for custom distributions. Callback
errors retain their messages. Interval-censored callback fits use the Rust
solver's existing indexing correction; tests compare a custom Gaussian with
R's equivalent built-in Gaussian path to avoid R's unsafe callback C path.
An explicit Student-t parameter is required for `dist="t"`, consistently with
the Python bare interface. Fractional stratum codes are rejected rather than
silently truncated, and unused scale strata retain their positions.

R tests compare 64 built-in fitting cases with stock survival, including
all four base families, weights/offsets, scale choices, full/partial starts,
iteration limits, aliases, interval censoring and matrix names. Further
checks cover callbacks, parameter names, errors, serialization, unused strata
and singleton vectors. A test disables `survival::survreg.fit` and records
the call to `survival.r.survreg_fit`, so numerical equality alone cannot mask
a reference fallback.

`scripts/benchmark_aft_r_bridge.R` measures complete R calls, including
conversion and result assembly, after checking all returned components.
Fitting inputs, warmup and garbage collection are outside the timing.
The optional third argument is a previous `bridge.R` for a direct comparison:

```sh
git show 0d61027e:r/survivalr/R/bridge.R > /tmp/aft-bridge-before.R
RETICULATE_PYTHON=$PWD/.venv/bin/python PYTHONPATH=$PWD/python \
  Rscript scripts/benchmark_aft_r_bridge.R 20000 7 /tmp/aft-bridge-before.R
```

On an Intel Core Ultra 5 325 with R 4.5.3, survival 3.8-12 and Python 3.14.7,
seven samples with 20,000 rows and six design columns gave these medians:

| Density | Stock R | Previous bridge | Compact bridge |
| --- | ---: | ---: | ---: |
| Gaussian | 27 ms | 68 ms | 18 ms |
| Logistic | 17 ms | 66 ms | 16 ms |
| Custom Gaussian callback | 44 ms | 44 ms | 45 ms |

Built-in calls improved by 3.78–4.13× against the previous bridge. Custom
callbacks now use Rust fitting with similar runtime; the previous bridge
delegated them to R. These local measurements have millisecond resolution,
depend on workload and machine load, and do not measure peak memory.
