# Low-level fitting interfaces

Besides the formula interface, the package exposes R's bare matrix fitters and
the native kernels behind them. These interfaces take prepared rows: they do not
build a formula design, omit missing rows or merge nearly equal times unless a
section says otherwise. Differences from R are listed in
[R compatibility](r-compatibility.md).

| Interface | R counterpart | Domain binding | Rust entry point |
| --- | --- | --- | --- |
| `r.coxph_fit`, `r.agreg_fit`, `r.agexact_fit` | `coxph.fit`, `agreg.fit`, `agexact.fit` | `regression.coxph_fit_raw` | `regression::CoxphFitResult::fit` |
| `r.survfitKM` | `survfitKM` | `surv_analysis.survfitkm` | `surv_analysis::survfitkm` |
| `r.survreg_fit` | `survreg.fit` | `regression.survreg_fit_raw` | `regression::SurvregFitResult::fit` |
| `r.survpenal_fit` | `survpenal.fit` | `regression.survpenal_fit_raw` | `regression::SurvpenalFitResult::fit` |
| `r.coxsurv_fit`, `r.survfitcoxph_fit` | `coxsurv.fit`, `survfitcoxph.fit` | `surv_analysis.coxsurv_fit` | `surv_analysis::coxsurv_fit` |

## Matrix and vector inputs

Numeric arguments accept NumPy arrays of any memory layout and dtype, pandas or
Polars columns, and Python sequences. Matrix arguments are copied once into an
owned row-major Rust matrix: NumPy arrays (including Fortran-ordered and strided
views) are read directly, and lists or tuples of float rows fill one allocation.
Integer values, custom numeric objects and array-valued rows use the general
conversion path. The typed boundary types are described in
[Repository layout](repo-layout.md#boundary-input-types). Kernels receive owned
buffers, so the bindings release the GIL while fitting.

## Bare Cox fitting

`r.coxph_fit`, `r.agreg_fit` and `r.agexact_fit` return a `CoxFitResult` with
coefficients, model-based variance, initial and final log likelihood, score test,
iteration count, linear predictors, means and optional martingale residuals.
`agreg_fit` also returns the final score vector (`first`) and iteration
diagnostics (`info`).

```python
from survival import r

y = r.Surv([1, 2, 3, 4], [1, 1, 0, 1])
fit = r.coxph_fit({"group": [0, 1, 0, 1]}, y, resid=False)
assert fit.coefficient_names == ("group",)
assert fit.residuals is None
```

- `coxph_fit` takes a right-censored response; `agreg_fit` takes counting-process
  data; `agexact_fit` takes either and uses entry time zero for right-censored
  data. The design can be a numeric matrix, nested sequence, mapping of columns or
  data frame; `coxph_fit` also accepts a single vector. Empty designs have zero
  columns. Names come from mappings or data frames or from `column_names`;
  `rownames` supplies row labels.
- Offsets keep their original values, including in the linear predictors.
  Robust variance and concordance are not computed. `nocenter=None` centers every
  covariate; pass `[-1, 0, 1]` for the high-level binary-column policy. Controls
  come from `coxph_control`.
- `coxph_fit` and `agreg_fit` use Efron for `method="efron"` and Breslow for every
  other accepted label, including `"exact"`, as R does. `agexact_fit` always uses
  the exact likelihood and requires unit weights. Use `r.coxph(..., ties="exact")`
  for the usual tie selection. Only the three documented labels are accepted.
- Null `coxph_fit` and `agreg_fit` models have a single log likelihood, no
  coefficient, variance, score or iteration components, and
  `classes=("coxph.null", "coxph")`; ordinary fits have `classes=("coxph",)`.
  `agexact_fit` keeps R's single-column matrix of linear predictors, its
  `method="coxph"` label and an empty class list.
- Arrays are independent copies on access, names are immutable tuples, and results
  pickle. They keep no response or design and cannot predict. R classes and names
  are held in `classes`, `coefficient_names` and `row_names`.

`regression.coxph_fit_raw` returns the same numerical result without R's null-model
component removal or shape adjustments; its `method="exact"` selects the exact
likelihood. Cluster and robust-variance options are refused. `resid=False` skips
the residual pass. Both paths share the optimizer used by full `CoxPHFit` models.

## Direct Kaplan–Meier fitting

`r.survfitKM` fits a prepared factor and a right-censored or counting-process
`Surv` with the same Rust engine as `r.survfit`, without a model frame.

```python
x = r.strata(["control", "control", "treated", "treated"], shortlabel=True)
y = r.Surv([1, 3, 2, 4], [1, 0, 1, 1])
fit = r.survfitKM(x, y, conf_type="log-log")
print(fit.strata, fit.time, fit.surv)
```

- `x` must be a `StrataFactor`, a pandas categorical or a categorical Series;
  ordinary lists are refused, as R requires a factor. Declared order and unused
  levels are kept. `y` must be of type `right` or `counting`.
- Inputs must be aligned and complete. Weights must be finite and nonnegative;
  `id` and `cluster` must match the response length, and empty vectors count as
  absent. Use `r.survfit` for formula processing and its default `timefix=True`.
- `stype`, `ctype`, weights, robust variance, influence matrices, entry counts,
  confidence options and `start_time` use the common engine. The old `type`
  argument overrides `stype` and `ctype`, including otherwise invalid values;
  `time0` is accepted and unused, as in R. Dotted aliases such as `se.fit` work
  through keyword dictionaries.
- The unweighted `counts` table appears when R's relative comparison of weights
  with one (tolerance the square root of machine epsilon) says so; fitting always
  uses the original weights.

`SurvfitKMResult` exposes `n`, `time`, `n_risk`, `n_event`, `n_censor`, `surv`,
`cumhaz`, standard errors, confidence limits and metadata, plus optional
`n_enter`, `counts`, `n_id` and influence matrices. `n` and `n_id` include zeros
for unused or filtered-out levels; `strata` is `None` for one declared level and
otherwise maps every level to its number of output times. Influence components are
lists in declared level order, with `None` for empty curves; their row labels
identify clusters, subjects or observations. Numeric properties return copies;
influence values are read-only views. The R package's `survfitKM` converts this
result to R's list, keeping its field order, count-matrix names and the
single-matrix versus list distinction.

## Bare AFT fitting

`r.survreg_fit` fits an explicit design and a prepared response with the same Rust
solver as `survreg`. It returns coefficients (estimated `Log(scale)` parameters
last), the intercept-only coefficients `icoef`, variance, both log likelihoods,
iterations, linear predictors, parameter count `df` and the final score.

```python
import numpy as np

y = np.array([[1.2, 1], [2.5, 0], [.9, 1], [3, 1], [1.8, 1], [2.7, 0]])
x = {"Intercept": np.ones(6), "group": [0, 1, 1, 0, 1, 0]}
fit = r.survreg_fit(x, y, dist="gaussian")
assert fit.coefficient_names == ("Intercept", "group", "Log(scale)")
```

- The response is a `(time, status)` or `(time, time2, status)` matrix with status
  0 (right censored), 1 (exact), 2 (left censored) or 3 (interval). The numeric
  columns of a `Surv` are accepted without recoding, as in R: a left-censored
  `Surv` uses 0/1, so prepare status 2 yourself. Counting and multi-state
  responses are refused.
- Responses must already be on the fitting scale: a Weibull fit uses logged times
  with `dist="extreme"`, and `dist="weibull"` is rejected because R's entry has no
  density of its own. The base names are `extreme`, `logistic`, `gaussian` and
  `t`, matched exactly; `t` requires explicit `parms`. No response transform or
  Jacobian is applied.
- `scale=0` estimates scales and a positive value fixes the scale. `nstrat` gives
  the number of scale strata, including unused ones; with more than one, `strata`
  holds one-based integer codes (fractional codes are rejected), and with
  `nstrat=1` the codes are ignored. Fixed scales cannot be combined with several
  strata. Controls use `survreg_control` through `controlvals`; `assign` is unused.
- Names come from mappings, data frames or `column_names`, otherwise R's `x 1`,
  `x 2`, .... `variance_names` is `None` when R drops them while undoing covariate
  rescaling. The score keeps R's internal covariate scaling. Aliased coefficients
  keep the solver's values; alias marking belongs to `survreg`.
- Custom distribution mappings need a five-column density callback and, when
  starting values need it, an initialization callback. Explicit callbacks are used
  regardless of the distribution's display name. Response transforms and fixed
  scales stored in a distribution are ignored.

`SurvregFitResult` is immutable, stores numeric output and name tuples, and keeps
no response, design or callbacks, so it pickles even with local callbacks and does
not predict. `regression.survreg_fit_raw` takes `SurvregData` and a
`SurvregDistribution`, accepts an optional explicit stratum count and refuses
cluster variance. Both ordinary and penalized fitters validate the response,
design, weights, offsets and grouping vectors at the fitting boundary, because
Rust callers can change `SurvregData` fields after construction. Interval upper
endpoints must be finite and at least the lower endpoint only for status 3.
Stratum counts whose dense covariance cannot be indexed are refused.

The R bridge's `survivalr::survreg.fit` uses this interface for every density and
passes the design as a matrix. Named built-ins run entirely in Rust. An explicit
distribution list uses its R density and initialization functions, which receive
vector batches with the original `parms` (the initializer receives `y`, `weights`
and `parms`, possibly `NULL`); transforms, quantiles, deviance functions and fixed
scales are unused. Convergence warnings are R conditions, and there is no call to
R's reference fitter.

## Bare penalized AFT fitting

`r.survpenal_fit` fits prepared AFT matrices with ridge, spline, dense or sparse
frailty, and custom penalties, using the full model's solver.

```python
from survival import regression

fit = r.survpenal_fit(
    x, y, dist="gaussian",
    pcols=[[1]], pattr=[regression.CoxPenalty.ridge(theta=1)],
    assign={"Intercept": [0], "group": [1]},
)
print(r.print_survreg_penal(fit))
```

Response codes, scales, densities, controls, weights, offsets, initial values and
stratum codes follow `survreg_fit`. Custom distributions also need a variance
component for the effective sample size used by penalty searches.

- `pattr` is a sequence of native `regression.CoxPenalty` objects (`ridge`,
  `pspline`, `frailty`, `callback`, `controlled`) or `r.pspline` results; `pcols`
  gives each penalty's design columns; `assign` maps term names to column groups
  or is a sequence of groups. Indices in `pcols`, `assign` and the returned
  `assign2` are **zero-based**. Every column belongs to exactly one term, each
  penalty's columns must match a term, term names are unique and `sigma` is
  reserved when scales are estimated.
- A sparse penalty takes one numeric group column, whose sorted unique values
  identify the frailties; only one sparse term is allowed. The sparse column is
  removed from the dense coefficients and covariance; `frail` and `fvar` hold its
  estimates and variances. Dense frailties take indicator columns.
- For a spline, put `spline.basis` in the design and pass the same
  `PsplineResult` in `pattr`; the result keeps the small spline description needed
  for printing. A native `CoxPenalty.pspline` fits but prints ordinary coefficient
  rows. Native penalties do not carry R's `varname` replacement attribute.

`SurvpenalFitResult` exposes coefficients (with log-scales), `icoef`, `var`,
`var2`, `loglik`, `iter`, linear predictors, per-term `df`, `penalty`, `score`
(sparse frailties first, then dense coefficients and log-scales), named `pterms`,
`assign2` and `history`, and `n`, `nvar`, `n_eff`, scales, convergence and
`inner_failures`. `df2` is `None`, as R leaves it unpopulated. The native result
always has a two-element `penalty`; the R bridge returns R's shapes (a scalar for
a sparse fit). History objects keep their column schema even when R omits names
for an empty fixed-gamma history. As in R, exhausted inner iterations do not warn:
inspect `inner_failures` and `converged`. The result is immutable and pickles.
`regression.survpenal_fit_raw` borrows prepared `SurvregData`, releases the GIL
around the solver and refuses clustered inputs.

The R bridge's `survivalr::survpenal.fit` uses the same fitter. It adapts R
`pfun`, `cfun`, `cargs`, `cparm` and `pparm` to the shared controller protocol and
keeps R's controller histories, `printfun` closures, column names, `varname`
replacements and one-based assignments in the returned list.

## Custom penalty searches

`regression.CoxPenalty.controlled(pfun, cfun, diag=True, sparse=False,
needs_df=False)` supplies an R-style penalty and outer tuning search for Cox and
AFT fits. `pfun(coef, theta, neff)` returns a mapping with the positive `penalty`,
its `first` and `second` derivatives and a boolean `flag`; use `diag=False` for a
full second-derivative matrix flattened in column-major order. An optional
`recenter` value is subtracted from the coefficients, and a flagged term may omit
derivatives. This differs from the `callback` constructor's C-level `coxlist`
convention.

`cfun(old, info)` is first called with `old=None` and `info` holding `iter=0` and
`eps2`. After each inner fit, `info` holds `iter`, the term's `coef`, the
unpenalized `plik`, penalized `loglik`, `neff`, `df` and `trH` (`df` and `trH` may
be NaN unless `needs_df=True`). Cox uses the event count for `neff`; AFT uses its
null-fit effective sample size. Every result must contain a finite scalar `theta`,
and updates a boolean `done`. The returned mapping becomes `old` on the next call;
optional numeric `history` rows with `columns`, `c_loglik` and `half` populate
`PenaltyHistory`. Each fit owns its state.

Minimal custom AFT distributions provide `variance` for the standardized density.
The optional `fitting_variance(scale_squared)` hook instead receives the null fit's
squared mean scale, matching R's penalized fitting convention, with any parameters
as a second argument; it is also available on
`SurvregDistribution.from_callbacks`. Its result must be finite; R's Student-t
convention can make it negative, which is kept.

`regression.PenaltyController` exposes the Rust searches behind the penalized
fitters: `fixed`, `gamma`, `df`, `aic` and `gaussian` (REML).

```python
from survival.regression import PenaltyController

search = PenaltyController(
    "df", target_df=1.7, eps=0.1, thetas=[0.0], dfs=[4.0], guess=0.7
)
state = search.initial()
state = search.step(state, 1, df=2.3)
print(state.theta, state.done, search.columns, state.history)
```

Controllers are immutable; `step` takes the previous `PenaltyControlState` and a
consecutive one-based iteration and returns a new state proposing the next theta
(including the final proposal when `done` becomes true). `plik` is the unpenalized
likelihood used by AIC, `loglik` the penalized likelihood used by the gamma
correction, `neff` the effective sample size, `trh` the Hessian trace, `coef` the
frailties and `events_by_group` the integer event counts. `df` and `aic`
controllers accept `gamma_correction=True`. Gamma and Gaussian `init` need at
least two values; the gamma correction accepts at most ten million events in one
group. Nonfinite or negative theta proposals are errors, including a df search
with no upper point at an exact boundary. Rust callers use `PenaltyControl`,
`PenaltyController::new`, `PenaltyControlInput` and `PenaltyControlState` from
`regression`. The R bridge's `ridge`, `pspline` and `frailty` constructors use
these searches through callbacks with R's signatures, history transitions and
trace output.

## AFT distributions and callbacks

Rust and Python accept user-defined location-scale families in ordinary and
penalized `survreg` fits; fits keep the callbacks for prediction, residuals and
robust variance. Built-in families keep their scalar kernels.

Pass a distribution dictionary to `r.survreg(dist=...)`, or register it in
`r.survreg_distributions` and pass its name. A family supplies `name`, `init`,
`density`, `deviance` and `quantile`; penalized fits also need `variance`.

```python
import numpy as np
from survival import datasets, r


def init(y, weights):
    mean = np.average(y, weights=weights)
    return [mean, np.average((y - mean)**2, weights=weights)]


def density(z):
    e = np.exp(-np.abs(z))
    lower = np.where(z > 0, 1 / (1 + e), e / (1 + e))
    upper = np.where(z > 0, e / (1 + e), 1 / (1 + e))
    pdf = lower * upper
    return np.column_stack([lower, upper, pdf, 1 - 2*lower, 1 - 6*pdf])


def quantile(p):
    return np.log(p) - np.log1p(-p)


def deviance(y, scale):
    status = y[:, -1]
    center = y[:, 0].copy()
    loglik = np.zeros(len(y))
    exact = status == 1
    loglik[exact] = -np.log(4 * scale[exact])
    interval = status == 3
    if np.any(interval):
        center[interval] = (y[interval, 0] + y[interval, 1]) / 2
        width = (y[interval, 1] - y[interval, 0]) / scale[interval]
        loglik[interval] = np.log(np.tanh(width / 4))
    return {"center": center, "loglik": loglik}


def variance():
    return np.pi**2 / 3


custom = dict(name="Custom log logistic", init=init, density=density,
              quantile=quantile, deviance=deviance, variance=variance,
              trans="log")
assert r.survregDtest(custom)
fit = r.survreg("Surv(time, status) ~ age + sex", datasets.load_lung(), dist=custom)
median = r.predict(fit, type="quantile", p=0.5)
```

| Callback | Arguments | Result |
| --- | --- | --- |
| `init` | Transformed response, case weights | Location and variance (two values) |
| `density` | Standardized endpoints | Matrix with columns `F`, `1-F`, `f`, `f'/f`, `f''/f` |
| `quantile` | Probabilities | Standardized quantiles |
| `deviance` | Response matrix, one scale per row | `center` and saturated `loglik`, as a dictionary or pair of vectors |
| `variance` | None | Finite positive standardized variance |

The deviance response matrix has `[y, status]` columns, or `[lower, upper,
status]` when any row is interval censored; centers and likelihoods are on the
transformed scale. A nonempty `parms` entry adds a final argument to every
callback; a dictionary supports partial named overrides through
`survreg(parms=...)`, while an unnamed vector must be replaced in full. `scale` in
the definition fixes the model scale. `trans="identity"` (the default) and
`trans="log"` select built-in transforms; an arbitrary increasing transform
supplies callable `trans`, `dtrans` and `itrans`, and `dist="gaussian"` (or
another family) reuses a base family with the transform:

```python
asinh_normal = dict(name="Asinh normal", dist="gaussian", trans=np.arcsinh,
                    dtrans=lambda y: 1 / np.sqrt(1 + y*y), itrans=np.sinh)
```

`survregDtest` checks density output at 0.1 through 1 and transform inversion and
positive finite derivatives at 1 through 10; fitting checks results again on the
actual observations. Callback exceptions name the callback; Python exceptions
during native fitting surface as `RuntimeError`, and invalid shapes or values
detected by Rust as `ValueError`. Pickle, `copy` and `deepcopy` keep custom
distributions in fitted models when their callables are picklable. Generic serde
serialization rejects runtime callbacks. The R facade's serialization saves the
original R functions with their environments; see [R bridge](r-bridge.md).

For direct bindings, use `SurvregDistribution.from_callbacks(...)` with
`with_transform(...)`, `with_parms(...)` or `derived(...)`; `family` and
`transform` report `Custom`. Rust callers implement `regression::SurvregCallbacks`
and optionally `regression::SurvregTransformCallbacks` (owned state,
`SurvivalResult`, `Send + Sync`) and construct
`SurvregDistribution::from_callbacks` with an `Arc`. Scalar distribution
operations (`density`, `quantile`, `deviance`, `variance`, `pdf`, `cdf`,
`quantile_at`) return `SurvivalResult`. `validate` checks metadata without calling
callbacks; `dtest` adds the functional probes. See
[`survreg_callbacks/tests.rs`](../src/regression/survreg_callbacks/tests.rs) and
the [external-crate example](../tests/survreg_callbacks.rs).

Each likelihood evaluation calls the density once with all standardized lower
endpoints, followed by the upper endpoints of interval-censored rows in input
order; linear predictors include offsets and sparse frailties. Shapes, finite
probabilities and nonnegative finite densities are checked; derivative ratios must
be finite where the density is positive and may be undefined where it underflows
to zero. Errors propagate through Newton steps, Fisher fallbacks, penalized
evaluations and line searches. The Python adapter reacquires the GIL once per
batch.

### Distribution queries

`dsurvreg`, `psurvreg`, `qsurvreg` and `rsurvreg` accept distribution names,
dictionaries and resolved objects. Query names use case-folded exact lookup;
fitting also accepts unique prefixes. Location and scale vectors recycle
separately at each arithmetic stage, and empty, missing, nonfinite and
nonpositive values follow R's query arithmetic: `r.qsurvreg([0.25, 0.75],
mean=[0, 1, 2], scale=1, distribution="gaussian")` returns three values with a
fractional-recycling warning. Callbacks receive the complete batch of each stage,
including empty and missing entries; density queries evaluate the transform
derivative before the transform, and quantile queries evaluate the quantile before
scaling, location and inverse transform. Fitting keeps its stricter checks.

Named Student-t queries require explicit numeric `parms`, which recycle inside
`pt`, `dt` and `qt`; other named built-ins ignore `parms`.
`SurvregDistribution("t")` keeps its default four degrees of freedom for fitting
and object queries; `for_query("t", parms)` builds a permissive query object.

`rsurvreg(n, ...)` follows R's `runif` count rules: a singleton supplies a finite
nonnegative count truncated toward zero, any other vector supplies its length,
NumPy arrays their element count, and one-shot iterators are read once. Exactly
that many uniforms are drawn before the quantile is evaluated and location and
scale recycled, so the result can be longer than the draw count. `seed=s`
reproduces R's `set.seed(s)` stream (Mersenne-Twister, inversion), also for custom
families; seeds are R's 32-bit integers.

Python arithmetic warnings use `RuntimeWarning`; callback warnings keep their
category and exceptions keep their identity. The R bridge forwards
`RuntimeWarning` and `UserWarning` as they occur, so warning-as-error settings
stop evaluation. Rust's `pdf_values`, `cdf_values`, `quantile_values` and `sample`
ignore arithmetic warnings; their `_with_warnings` variants pass each
`regression::DpqrWarning` to a fallible callback.

## Direct Cox curves

`r.coxsurv_fit` exposes R's `coxsurv.fit` for prepared response and design
matrices, relative risks and coefficient covariance, computing all strata in one
Rust call.

```python
curves = r.coxsurv_fit(
    y=r.Surv([1, 2, 3], [1, 1, 0]),
    x=np.array([[-1.0], [0.0], [1.0]]),
    risk=np.ones(3),
    x2=np.array([[0.0], [0.5]]),
    risk2=np.ones(2),
    varmat=np.array([[0.2]]),
)
assert curves.surv.shape == (3, 2)
```

- Covariates and risks must already be centered consistently; no fitting,
  centering, omission or time fixing occurs. `y` has `(stop, status)` or
  `(start, stop, status)` columns with binary status, or is a right or counting
  `Surv`. `x` may have zero columns; `x2` is a matrix, or a vector for one row.
- `wt=None` gives unit weights. Training risks must be positive and finite,
  prediction risks may be zero, weights finite and nonnegative, and an observed
  risk set with zero weighted risk is refused. Responses and covariates must be
  finite, with `start < stop`.
- `stype=1` selects Kalbfleisch–Prentice; `stype=2` the exponential of the
  cumulative hazard with Breslow (`ctype=1`) or Efron (`ctype=2`) ties. Standard
  errors need a finite square `varmat`; `se_fit=False` (or `**{"se.fit": False}`)
  omits them. `cluster`, `position` and `oldid` are accepted and unused, as in R.
- `strata` uses sorted distinct values or declared categorical levels, including
  unused ones; omitting it gives one curve labeled `"0"`. Missing stratum values
  are rejected.
- `id2` builds one trajectory per id in first-appearance order from `(start,
  stop]` intervals in `y2`; `strata2` holds **one-based** positions in the training
  levels, or factor codes. Intervals with no baseline times contribute no rows.

With `unlist=True`, `CoxSurvFitResult` stacks time rows across curves; `n` gives
each curve's observation count, `strata` maps labels to row counts, and `surv`,
`cumhaz` and `std_err` (on the cumulative-hazard scale) have one column per
prediction row, or are vectors for one row or a trajectory. `unlist=False`
returns a `CoxSurvFitList` with `names`; ordinary curves also expose `hazard`,
`varhaz`, `ndeath` and `xbar`. Python names individual trajectories by their ids.
Arrays are read-only views of a frozen native result (use `.copy()` for writable
arrays); results and lists pickle. Unused levels, zero-column designs and empty
trajectories have explicit zero-size outputs.

`r.survfitcoxph_fit` is the older wrapper: `survtype` 1, 2 or 3 selects
Kalbfleisch–Prentice, Breslow or Efron, `newrisk` is required and `vartype` is
ignored, as in R. The Rust `surv_analysis::coxsurv_fit` takes `CoxSurvData` and
`CoxSurvNewData` with zero-based stratum indices; the domain binding
`surv_analysis.coxsurv_fit` exposes those codes and releases the GIL.
