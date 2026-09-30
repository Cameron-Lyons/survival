# survival

[![Crates.io](https://img.shields.io/crates/v/survival.svg)](https://crates.io/crates/survival)
[![PyPI version](https://img.shields.io/pypi/v/survival.svg)](https://pypi.org/project/survival/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance survival analysis library written in Rust, with a Python API powered by [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

## Features

- Core survival analysis routines
- Cox proportional hazards models (Breslow, Efron and exact ties, counting-process data)
- Kaplan-Meier and Aalen-Johansen (multi-state) survival curves
- Nelson-Aalen estimator
- Parametric accelerated failure time models
- Fine-Gray competing risks model
- P-spline and natural spline bases (`pspline`, `nsk`)
- Concordance index calculations
- Person-years calculations
- Score calculations for survival models
- Residual analysis (martingale, Schoenfeld, score residuals)
- Bootstrap confidence intervals
- Cross-validation for model assessment
- Statistical tests (log-rank, likelihood ratio, Wald, score, proportional hazards)
- Sample size and power calculations
- RMST (Restricted Mean Survival Time) analysis
- Landmark analysis
- Calibration and risk stratification
- Time-dependent AUC
- Conditional logistic regression
- Time-splitting utilities
- Survival-curve graphics with an optional Matplotlib renderer

The R-style interface supports penalized Cox and AFT formulas, interval-censored
AFT models, multi-state Cox models and their curves, multistate summaries and
expected survival from Cox models. See [R compatibility](docs/r-compatibility.md)
for examples, seeded Yates risk predictions, validation coverage, every known
difference from R and the R entry points that are
[not yet implemented](docs/r-compatibility.md#not-yet-implemented).
Release notes are in the [changelog](CHANGELOG.md).

## Installation

### From PyPI (Recommended)

```sh
pip install survival
```

### From Source

#### Prerequisites

- Python 3.11+
- Rust 1.94+ (see [rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin)

Install maturin:
```sh
pip install maturin
```

#### Build and Install

Build the Python wheel:
```sh
maturin build --release
```

The default source build keeps optional ML bindings out of the extension. To
build the full Python surface locally, include the ML feature explicitly:

```sh
maturin build --release --features extension-module,ml
```

Install the wheel:
```sh
pip install target/wheels/survival-*.whl
```

For development:
```sh
maturin develop --release
```

For development against ML bindings:
```sh
maturin develop --release --features extension-module,ml
```

## Python Package Layout

Prefer domain modules in new code:

```python
from survival import datasets, regression, surv_analysis, validation

lung = datasets.load_lung()
time = [float(t) for t in lung["time"]]
status = [int(s) - 1 for s in lung["status"]]  # R codes status as 1/2
x = [[float(a), float(s)] for a, s in zip(lung["age"], lung["sex"], strict=True)]

cox = regression.coxph_fit(time, status, x)  # R: coxph(Surv(time, status) ~ age + sex)
km = surv_analysis.survfitkm(time, status)  # R: survfit(Surv(time, status) ~ 1)
table = surv_analysis.survmean(km)  # R: summary(fit)$table
test = validation.logrank_test(time, status, [int(s) for s in lung["sex"]])
print(cox.coefficients, table.median, test.p_value)
```

`surv_analysis.survfitkm` and `surv_analysis.nelson_aalen` accept NumPy arrays,
including strided arrays, pandas/polars columns, and Python sequences. Status
values must be binary; integral floating-point arrays are accepted with checked
conversion. The numerical fit runs in Rust with the Python GIL released.

Cox and AFT matrix inputs also accept these array layouts. Lists and tuples
of float rows fill a single Rust matrix allocation. See
[matrix input conversion and benchmarks](docs/python-matrix-inputs.md).

For right-censored curves with independent observations, robust standard errors
use a linear sweep after sorting, including when fractional case weights select
robust variance automatically. Repeated clusters, counting-process data, and
explicit influence matrices use the general influence calculation. Run
`PYTHONPATH=python python scripts/bench_survival_curves.py` against a release build
to measure the Python calls, or
`cargo bench --bench survival_benchmarks -- kaplan_meier` for the Rust kernels.
See [the algorithm and benchmark notes](docs/kaplan-meier-performance.md) for
the scope of the optimization and a local before/after comparison.

Cox survival predictions and R-style Brier scores evaluate only the requested
times, avoiding full survival-curve matrices for every subject. The Rust and
Python `CoxPHFit.predict_survival_at` method exposes the same calculation;
NumPy inputs and results avoid list conversions. See
[Cox prediction performance](docs/cox-prediction-performance.md) for the API,
algorithm and reproducible benchmarks.

Full Cox curves accumulate coefficient uncertainty with one vector per curve.
Individual predictions find each interval's baseline times by binary search
and group subject rows once. See [full-curve performance](docs/cox-curve-performance.md)
for the scope, memory bounds and reproducible measurements.
Prepared matrices and risks can use `r.coxsurv_fit` or `r.survfitcoxph_fit`
directly; see [direct Cox curves](docs/cox-direct-curves.md) for inputs,
read-only results and individual trajectories.
Prepared formula metadata is supported by `r.attrassign` and
`r.untangle_specials`; see [term helpers](docs/term-helpers.md) for column
grouping and special-term indices.
`r.yates_setup` prepares reusable Cox risk/survival predictions and GLM inverse
links; see [Yates setup](docs/yates-setup.md) for layouts and summary corrections.

R-style entry points are intentionally available from the package root for users
porting code from R's `survival` package:

```python
from survival import (
    Surv,
    aic,
    as_data_frame,
    basehaz,
    clogit,
    coxph,
    fitted,
    predict,
    survdiff,
    survfit,
    survreg,
)

data = {
    "time": [1.0, 2.0, 3.0, 4.0],
    "status": [1, 1, 0, 1],
    "group": ["control", "control", "treated", "treated"],
    "age": [52.0, 61.0, 58.0, 63.0],
}

km = survfit("Surv(time, status) ~ group", data=data)
km_table = as_data_frame(km)
cox_model = coxph("Surv(time, status) ~ group + age", data=data)
risk_scores = predict(cox_model, [[1.0, 60.0]], type="risk")
training_lp = fitted(cox_model)
model_aic = aic(cox_model)
hazard_times, cumulative_hazard = basehaz(cox_model)
aft_model = survreg("Surv(time, status) ~ group + age", data=data)
```

Formula support is intentionally conservative: `+` terms, `.` expansion,
`-` exclusions, backtick-quoted column names, categorical treatment coding,
`factor(...)` / `as.factor(...)`, `strata(...)`, interaction terms with `:`,
`*` or `%in%`, and numeric `offset(...)` terms are supported, along with
one-column numeric transforms `log(...)`, `sqrt(...)`, and `exp(...)`, plus
`I(...)`/`identity(...)` arithmetic and comparisons, `cut(...)` and
`tcut(...)`; formula options are read as literals and never evaluated as code.
Formula calls also accept `subset=` as a boolean mask or zero-based row indices
and R's `na_action`: `"na.omit"` (the default, as in R), `"na.exclude"`
(residuals and predictions padded with NaN at the removed rows), `"na.fail"`
and `"na.pass"`, applied across formula columns and external row-aligned
arrays such as `weights`, `offset`, and `strata`.
R `survobrien` formula expansion preserves factor keeper columns while applying
the risk-set transform only to continuous terms.
R `finegray` formulas use the same Python formula engine and Rust interval
expansion, with sorted censoring-risk sweeps and R-compatible factor classes.
Kaplan-Meier `survfit` calls honor `conf_level=`, R-style `conf_type=`
choices for confidence intervals, and `start_time=` for conditional curves;
`survfit0(...)` adds the starting row, as in R.
They support right-censored `Surv(time, event)` data and counting-process
`Surv(start, stop, event)` data with delayed-entry risk sets. Direct and
formula `Surv(...)` calls also accept R-style named aliases including `time=`,
`time1=`, `start=`, `time2=`, `stop=`, `event=`, and `status=`.
Factor-valued event responses produce multi-state Aalen--Johansen curves.
These curves support subject histories through `id=`, observed initial states
through `istate=`, event-type conversion through `etype=`, user-supplied
initial distributions through `p0=`, and entry counts through `entry=True`.
`survfit0(...)` inserts the initial state-probability row into existing
multi-state curves while preserving their typed count, hazard, and uncertainty
outputs.
Multi-state fits with retained model frames also support influence residuals
and pseudo-values for state probabilities, cumulative transition hazards, and
integrated state occupancy, including grouped, weighted, and subject-collapsed
counting-process results.
Ordinary fitted curves support R-style influence residuals and pseudo-values for
survival, cumulative hazard, and RMST, preserving case weights, subject IDs,
grouping, estimator settings, and conditional start times. A native query kernel
uses the fitted risk tables and event prefixes without refitting or building a
full observation-by-event influence matrix. The fitted RMST path follows R's
infinitesimal jackknife; the direct time/status pseudo-value API retains its
existing delete-one RMST calculation. Tied `ctype=2` diagnostics follow R's
approximation and report the same limitation.
Fitted Cox models can also be passed to `survfit(...)` with optional `newdata=`
to produce model-based survival curves; a multi-state Cox fit gives R's
`survfit.coxphms` state-probability curves for each newdata row.
Passing a square matrix of transition-specific KM or Cox curves to `survfit`
also produces multistate curves, using `None` for absent transitions and
`method="discrete"` or `"matexp"`. See
[transition-curve matrices](docs/survfit-matrix.md) for examples, R compatibility
details and benchmarks.
The R facade's low-level `coxsurv.fit` and `survfitcoxph.fit` entry points use
an `O(n log n)` Rust risk-set sweep for weighted, stratified, tied-event, and
counting-process baselines, while retaining R-compatible curve and uncertainty
shapes for ordinary predictions and individual time-dependent trajectories.
`survdiff` uses the same right-censored and delayed-entry response forms.
`coxph` uses Efron's tie handling by default, matching R, and also accepts
`ties="breslow"` or the compatibility alias `method="breslow"`.
Penalty terms (`ridge`, `pspline`, `frailty`) are fitted by a port of R's
`coxpenal.fit`, jointly with the Cox partial likelihood:

```python
fit = coxph("Surv(time, status) ~ age + ridge(x, z, theta=2)", data=data)
fit.df           # effective degrees of freedom of each term
fit.var2         # sampling covariance, distinct from vcov(fit)

selected = coxph("Surv(time, status) ~ age + ridge(x, z, df=1)", data=data)
selected.history # the theta/df search of the outer loop
```

`ridge(..., theta=..., scale=FALSE)` uses the supplied penalty in the original
coefficient units. The default scaling uses each column's unweighted sample
variance before subset or missing-row removal, as in R. Separate ridge calls
may specify different penalties. Weights, offsets, strata, delayed entry,
prediction, residuals, `summary`, `anova` (fractional df), `cox_zph` and
fractional-df information criteria are supported. A robust variance request
warns and is ignored, as in R, and penalized fits take the `"breslow"` or
`"efron"` ties. Omitting `theta` selects the penalty by effective degrees of
freedom, defaulting to half the number of columns in each ridge term;
`ridge(..., df=..., eps=.1)` sets the target and its tolerance, and
`control={"outer.max": 10}` limits the outer search. A penalty term inside an
interaction is refused with R's "Penalty terms cannot be in an interaction".
`survreg` fits `ridge()` and `pspline()` terms through a port of R's
`survpenal.fit`. The typed kernel is `survival.regression.coxpenal_fit` (R's
`coxpenal.fit`), which takes the design with its `CoxPenalty` terms.
Formula fits support `tt(...)` time-varying coefficient terms for right-censored
and counting-process responses, including R's default O'Brien rank transform
and custom `tt(x, time, riskset, weights)` callables.
`clogit("case ~ exposure + strata(set)", data=...)` fits matched case-control
models through the exact stratified Cox likelihood; `method="approximate"`
maps to Breslow handling as it does in R.
`cch("Surv(time, status) ~ exposure + group", data=..., subcoh="sampled",
id="subject", cohort_size=...)` fits case-cohort models with the native
Prentice, Self-Prentice, or Lin--Ying estimators. Sampling-stratified designs
also support I.Borgan and II.Borgan with per-stratum population sizes.
Right-censored and counting-process responses share the Cox optimizer, formula
expansion supports numeric, factor, and interaction terms, and `robust=True`
selects Lin--Ying's robust variance. The risk-set, residual, and phase-two
covariance sweeps stay in Rust; Python performs only formula preparation and
result labeling.
R-style `coxph.control(...)` and `survreg.control(...)` helpers are available
in the bridge and pass named control lists through to the Python API.
Native R Cox control objects are accepted, including `survcheckallow`, which
selects the survcheck flags a multi-state fit tolerates. Starting coefficients supplied through `init=` can
be a single number for a one-parameter model or a vector matching all fitted
parameters, including estimated log scales for AFT models. Explicit R `NULL`
values retain the default initialization regardless of argument order.
Cox and AFT fits warn when iterations are exhausted. Cox coefficient warnings
use `control={"toler.inf": ...}` and the fitted score and covariance, with R's
distinct criteria for right-censored, counting-process, and exact fits.
Zero- and one-iteration requests suppress these diagnostics. Python exposes
them as `RuntimeWarning`; the bridge raises R warning conditions.
AFT's default relative convergence tolerance is `1e-9` in both Rust and
Python, matching `survival::survreg.control()`; callers can set `eps` explicitly.
Time-dependent start/stop data can be built with the R-compatible `tmerge`
workflow. Its update builders preserve R's `(tstart, tstop]` boundary rules,
event placement, cumulative updates, missing-value handling, and classification
metadata while using the native linear-time sweeps underneath:

```python
from survival import cumevent, cumtdc, event, tdc, tmerge

baseline = {"id": [1, 2], "group": ["control", "treated"]}
spans = {"id": [1, 2], "stop": [10.0, 8.0]}
updates = {
    "id": [1, 1, 2],
    "time": [2.0, 6.0, 4.0],
    "dose": [5.0, 3.0, 4.0],
    "status": [0, 1, 1],
}

timeline = tmerge(baseline, spans, "id", tstop="stop")
timeline = tmerge(
    timeline,
    updates,
    "id",
    dose=tdc("time", "dose", init=0.0),
    total_dose=cumtdc("time", "dose", init=0.0),
    endpoint=event("time", "status"),
    endpoint_count=cumevent("time", "status"),
)
```

The raw per-call sweep is `survival.data_prep.tmerge_step`, for callers that
already manage sorted numeric arrays.
The R-style `predict(...)` and `fitted(...)` generics support Cox linear
predictors, relative risk scores, term contributions, survival curves, and
expected event counts.

Custom AFT families can supply vectorized `density`, `init`, `quantile`,
`deviance`, and `variance` callbacks, together with an optional response
transform. Both fitters retain callbacks for predictions, residuals, robust
variance, and Python pickle. Rust callers implement `regression::SurvregCallbacks`;
Python callers pass a distribution dictionary or register it in
`r.survreg_distributions`. See [AFT distribution callbacks](docs/survreg-density.md)
for a complete example and the batch contract.

For `survreg` fits, `predict(fit, newdata, type=...)` gives R's `"lp"`
(`"linear"`), `"response"`, `"terms"`, `"quantile"` and `"uquantile"`
predictions: `type="quantile"` returns response-scale quantiles at the
probabilities `p=` (R's default `[0.1, 0.9]`), and `type="uquantile"` the same
quantiles on the linear (log-time) scale. Probabilities include 0 and 1, which
return the distribution's limits. Quantiles use each row's stratum scale and
keep the Student-t degrees of freedom; `se_fit=True` adds standard errors on the
requested scale. Formula offsets enter the linear predictor of `newdata` rows
as well as of the training rows (R drops them for single-scale models; see
[R compatibility](docs/r-compatibility.md#reference-differences-retained-deliberately)).
The typed `SurvregFit.predict(...)` takes `offset=` and `strata=` for the rows
of its `newdata`.
`AFTEstimator.predict()` returns the fitted response-scale location, matching
R's default prediction; `predict_median()` and `predict_quantile()` return
actual distribution quantiles. Gaussian, logistic, extreme-value, and Student-t
responses use the identity transform.
The distribution functions are ports of R's nmath routines (`pnorm`, `qnorm`,
`pt`, `qt`, ...), used by the fit, the residuals and `dsurvreg`/`psurvreg`/
`qsurvreg`/`rsurvreg` alike; the t family takes `distribution="t", parms=df`.
For example, `qsurvreg(1e-20, 0, distribution="t", parms=4)` returns
`-131607.4013`, as R does.
Gaussian, logistic, extreme-value, and Student-t AFT models accept finite real-valued
responses, including negative values and zero, for all censoring types. Log-time
families retain their positive-response requirement. Right-censored concordance
also accepts real-valued responses, so these models can be scored directly.
AFT coefficient accessors report aliased location coefficients as `NaN`, while
stored training predictions and residuals retain the fitted numeric values.
As in R, an aliased coefficient makes ordinary `newdata` predictions missing;
term predictions retain contributions from other terms. As in R,
`vcov(complete=False)` drops the aliased coefficients and keeps the estimated
scale parameters.
The optimizer is a port of R's `survreg6.c`, with the likelihood and its
derivatives from `survregc1.c`: Newton-Raphson steps solved with `cholesky2` on
the information matrix, the score outer-product matrix `JJ` when the
information is not positive definite, and a step that backs off two thirds of
the way to the last good point when the likelihood does not improve. Case
weights must be positive, as in R's `survreg.fit` ("Invalid weights, must be
>0"). The R bridge also routes built-in `survreg.fit` matrix calls through this
kernel, including fixed or stratified scales and interval-censored responses.
Model helpers include `model_formula`, `model_weights`, `df_residual`,
`loglik`, `aic`, `bic`, `extract_aic`, coefficient, variance-covariance,
confidence-interval, model-matrix/model-frame, and summary accessors for fitted
Cox and `survreg` models.
Grouped Cox predictions and Cox/AFT residuals preserve factor order and missing
groups. Prediction errors combine in quadrature through a shared Rust kernel;
the R interface retains group names. See [grouped model outputs](docs/model-collapse.md)
for omission rules, reference differences and complete-call timings.
`predict(pspline(x), newx)` evaluates a spline basis using the original
boundaries, degree, intercept and column combinations, with linear
extrapolation outside the boundaries.
`quantile(response, probs=[0.25, 0.5, 0.75])` and `median(response)` accept
raw `Surv` data as well as fitted survival curves. Raw responses use the
appropriate Rust KM or Turnbull fitter and require `na_rm=True` to discard
missing observations. Results keep one row per curve and one column per
probability. As in R, raw-response medians include confidence bounds by
default, while fitted-curve medians return point estimates.

Common result objects can be converted to column-oriented tables with
`as_data_frame(...)`; the experimental R bridge exposes the same path through
`as.data.frame(...)`, `summary(...)`, and `print(...)` methods.
`Surv` responses also support table conversion for quick data inspection.
`model_summary` also accepts expected-survival curves, rate tables, and
`tmerge` results. Expected-curve summaries use Rust to select requested
times; rate-table summaries describe dimensions and date ranges, while
merged-data summaries report where updates fell relative to each interval.
The `survival.residuals` name remains the residual diagnostics module; the
R-style residual generic is available as `survival.r_api.residuals(...)` for
fitted Cox and `survreg` models (`survival.r_api` re-exports the `survival.r`
package, which holds the implementation split by concern).
For AFT models, `type="matrix"` returns six analytic diagnostic columns in R's
order (`g`, `dg`, `ddg`, `ds`, `dds`, `dsg`), including its interval-censoring
conventions. Working residuals use the location score divided by negative
curvature; tail probabilities are evaluated directly to avoid cancellation.

Algorithm bindings are not re-exported from the package root: import them from
their domain module (`survival.regression.coxph_fit`, not `survival.coxph_fit`).
The R-style names and the domain modules are resolved lazily instead of being
copied into the package namespace at import time.

`survival.__all__` and `dir(survival)` expose the curated package surface:
domain modules, R-style entry points, and scikit-learn helpers. In lean source
builds, symbols that require the Rust `ml` feature are omitted from their domain
module until the extension is built with `--features extension-module,ml`.

Common modules:
- `survival.datasets`: built-in example and benchmark datasets
- `survival.data_prep`: time splitting and data transformation helpers
- `survival.core`: typed inputs (`SurvivalData`, `CovariateMatrix`, ...), `concordancefit`,
  spline bases and the Cox residual kernels
- `survival.regression`: Cox, AFT, competing-risks, cure, and recurrent-event models
- `survival.surv_analysis`: Kaplan-Meier, Nelson-Aalen, multistate, and log-rank helpers
- `survival.validation`: metrics, calibration, conformal, RMST, and statistical tests
- `survival.residuals`: the `coxmart`/`agmart` kernels; model residuals live on `CoxPHFit`
  and `SurvregFit`
- `survival.population`: rate tables, `match_ratetable`, `pyears` and `survexp`
- `survival.monitoring`: drift and monitoring utilities
- `survival.ml`: neural, tree, and modern ML-oriented survival models
- `survival.reliability_tools`: reliability utilities; the top-level
  `survival.reliability` name remains the callable function

See [`docs/repo-layout.md`](docs/repo-layout.md) for the full Rust and Python
layout and [`examples/python_package_layout.py`](examples/python_package_layout.py)
for a runnable module-oriented example.

`DeepSurvConfig`, `GradientBoostSurvivalConfig`, and `SurvivalForestConfig`
validate settings both when constructed and before fitting. Fields can be changed
between fits; invalid settings raise `ValueError` before training starts.
DeepSurv learning rates must remain positive and finite after conversion to
float32, and its L2 penalty must be nonnegative and finite in that precision.
Boosting requires `min_samples_leaf >= 1`. Empty DeepSurv hidden layers still
create a linear network, tree `max_depth=0` creates stumps, and large leaf or node
minimums disable splitting. Width and count checks use native array size limits.

Native DeepSurv, gradient boosting, and survival forest fitting require finite
features and times, with event statuses exactly `0` (censored) or `1` (event).
Negative and zero times and all-censored data remain supported. DeepSurv training
also requires features to stay finite when converted to float32; ordinary
rounding and underflow are allowed. Predictions use finite float64 features,
including values beyond the float32 range. Invalid values raise `ValueError`
with the field and flattened input index. Forest input construction and typed
fitting both validate the data; empty prediction batches remain supported.

## Usage

### Aalen's Additive Regression Model

```python
import survival

data = {
    "time": [1.0, 2.0, 2.0, 3.0, 4.0, 4.0],
    "status": [1, 1, 1, 1, 0, 1],
    "age": [42.0, 55.0, 61.0, 49.0, 67.0, 38.0],
    "treatment": ["control", "treated", "control", "treated", "control", "treated"],
}

fit = survival.aareg(
    "Surv(time, status) ~ age + treatment",
    data=data,
    nmin=1,
)
print(fit.coefficient_names)
print(fit.coefficient)
```

The formula interface supports right-censored and counting-process responses,
case weights, factors and interactions, clustered influence estimates, tapering,
and retained model, design, and response data. The risk-set sweep and linear
algebra are implemented in Rust.

### P-spline and natural spline bases

```python
from survival import core

x = [0.1 * i for i in range(100)]
# The B-spline basis of R's pspline(x, nterm = 10, degree = 3); the
# difference penalty is applied by the penalised Cox fit.
basis = core.pspline_basis(x, 10, 3, (0.0, 10.0))
print(len(basis.basis[0]), basis.knots[:4])

# R's nsk(): a natural spline whose coefficients are the values at the knots.
spline = core.nsk(x, df=4)
print(spline.n_cols, spline.knots, spline.boundary_knots)
```

### Concordance

```python
from survival import Surv, concordance

time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]
status = [1, 1, 0, 1, 1, 1, 0, 1]
risk = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9]

# R's concordance(Surv(time, status) ~ risk, reverse = TRUE, influence = 1)
fit = core.concordancefit(
    core.SurvivalData(time, status),
    core.CovariateMatrix(risk, len(risk), 1),
    reverse=True,
    influence=1,
)
print(fit.concordance[0], fit.count[0].concordant, fit.var[0][0], fit.dfbeta[0])
```

An observation censored at an event time remains a risk comparator; simultaneous
events contribute outcome ties. Right-censored and counting-process summaries
and influence calculations use O(n log n) risk-set sweeps. Raw influence rows are
derivatives with respect to case weights, holding time-weight multipliers fixed;
dfbeta applies case weights and uses pooled counts across strata. Variance is
available with every result, while
`influence` controls which diagnostic rows are returned. For multiple scores,
`result.covariance` and `vcov(result)` include the covariance between scores.

By default, `timefix=True` groups near-tied times using R's `aeqSurv` tolerance;
`timefix=False` preserves exact observed times. `ymin` clips exit times and
`ymax` limits contributing event times. Strata with fewer than two original
events use unit time multipliers, including when a horizon is supplied.
With no comparable pairs, concordance and its variance are `NaN`.

Direct `Surv` inputs accept `strata=labels`. For a single score, `keepstrata`
controls optional `stratum_labels` and `stratum_counts` fields; counts contain
the five exclusive pair categories: concordant, discordant, tied predictors,
tied outcomes, and ties in both. Concordance and covariance pool across strata.

### Person-Years Calculation

The high-level API accepts a `tcut` result directly for time-changing groups:

```python
import survival

response = survival.Surv([25.0, 8.0], [1, 0])
attained = survival.tcut([0.0, 5.0], [0.0, 10.0, 20.0, 30.0])
result = survival.pyears(response, group=attained, scale=1)
```

```python
from survival import population

# Low-level API: R's pyears() data side with the survexp.us rate table.
us = population.survexp_us()
# match.ratetable: age in days, sex as a 1-based factor code, year as days since 1970-01-01
positions = population.match_ratetable(
    us, ["age", "sex", "year"], [[18262.5, 21915.0], [1.0, 2.0], [10957.0, 12949.0]]
).r
result = population.pyears(
    [365.25, 1826.25],  # follow-up per subject
    event=[1.0, 0.0],
    factors=[1],  # one factor term (sex) ...
    dims=[2],  # ... with two levels
    cuts=[[]],
    categories_data=[[1.0], [2.0]],
    ratetable=us,
    ratetable_positions=positions,
    scale=365.25,
)
print(result.pyears, result.event, result.expected)

# R's survexp(): expected survival of the same two subjects (Ederer method)
expected = population.survexp(us, positions, times=[365.25, 1826.25])
print(expected.method, expected.surv)
```

The formula API (`survival.r.pyears` and `survival.r.survexp`) supports
`model=True` to retain evaluated input columns, or `x=True` and `y=True`
to retain grouping and response components. `model=True` takes precedence.
Use `model_frame`, `model_formula`, and `model_term_names` to inspect them;
see [population model components](docs/r-compatibility.md#population-model-components-and-summaries)
for the retained representations and missing-row behavior.

### Kaplan-Meier Survival Curves

```python
from survival import surv_analysis

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 1 = event, 0 = censored

# R: survfit(Surv(time, status) ~ 1, conf.type = "log-log")
result = surv_analysis.survfitkm(
    time,
    status,
    weights=None,  # Optional: case weights
    start=None,  # Optional: entry times for (start, stop] data
    conf_type="log-log",  # log, log-log, plain, logit, arcsin or none
)

print(f"Time points: {result.time}")
print(f"Survival estimates: {result.surv}")
print(f"Standard errors: {result.std_err}")
print(f"Number at risk: {result.n_risk}")

# R: summary(fit, times = ...), summary(fit)$table and quantile(fit)
at_times = surv_analysis.summary_survfit(result, times=[2.5, 5.0])
table = surv_analysis.survmean(result)
quantiles = surv_analysis.quantile_survfit(result, probs=[0.25, 0.5, 0.75])
print(at_times.surv, table.median, quantiles.quantile)
```

### Fine-Gray Competing Risks Model

```python
from survival import finegray

data = {
    "time": [1.0, 2.0, 3.0, 4.0],
    "event": ["target", "competing", "censor", "target"],
    "x": [0.2, 0.4, 0.1, 0.8],
}

# String labels use a recognized censor label as the censoring state. For
# pandas categoricals, the declared category order is preserved exactly.
expanded = finegray(
    "Surv(time, event) ~ x",
    data=data,
    etype="target",
    count="replication",
)

print(expanded.event)
print(expanded["fgstart"], expanded["fgstop"], expanded["fgwt"])
```

The checked six-vector interval splitter remains available for lower-level
workflows:

```python
from survival import regression

# Example competing risks data
tstart = [0.0, 0.0, 0.0, 0.0]
tstop = [1.0, 2.0, 3.0, 4.0]
ctime = [0.5, 1.5, 2.5, 3.5]  # Cut points
cprob = [0.1, 0.2, 0.3, 0.4]  # Cumulative probabilities
extend = [True, True, False, False]  # Whether to extend intervals
keep = [True, True, True, True]      # Which cut points to keep

result = regression.finegray(
    tstart=tstart,
    tstop=tstop,
    ctime=ctime,
    cprob=cprob,
    extend=extend,
    keep=keep
)

print(f"Row indices: {result.row}")
print(f"Start times: {result.start}")
print(f"End times: {result.end}")
print(f"Weights: {result.wt}")
```

### Parametric Survival Regression (Accelerated Failure Time Models)

```python
from survival import regression

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 0 right, 1 exact, 2 left, 3 interval censored
covariates = [[1.0, x] for x in [0.5, 0.2, 0.9, 0.1, 0.7, 0.3, 0.8, 0.4]]  # intercept + x

# R: survreg(Surv(time, status) ~ x, dist = "weibull")
data = regression.SurvregData(time, status, covariates)  # also time2=, weights=, strata=, cluster=
fit = regression.survreg_fit(
    data,
    regression.SurvregDistribution("weibull"),  # R names: weibull, lognormal, loglogistic, ...
    control=regression.SurvregControl(iter_max=30, rel_tolerance=1e-9),
)

print(f"Coefficients: {fit.coefficients}")  # location coefficients then Log(scale)
print(f"Scale: {fit.scale}")
print(f"Log-likelihood: {fit.log_likelihood}")
print(f"Variance matrix: {fit.variance_matrix}")
print(f"Converged: {fit.converged}")

# R: predict(fit, newdata, type = "quantile", p = c(.1, .5)) and residuals(fit, type = "deviance")
prediction = fit.predict(newdata=[[1.0, 0.25]], predict_type="quantile", p=[0.1, 0.5], se_fit=True)
print(prediction.fit, prediction.se_fit)
print(fit.residuals(residual_type="deviance").values)
```

Omitted AFT starts use R's distribution-specific weighted variance estimates,
censoring-aware working regression, and a preliminary intercept fit when scales
must be estimated for a model with covariates. This applies to every built-in
distribution, fixed scales, and stratified scales. `max_iter=0` returns the
initialized model without taking a main-model optimization step.

With omitted starts and a leading intercept, continuous covariates are centered
and scaled during fitting; binary columns keep their coding. Coefficients and
covariance are returned in the original units, while stored predictions are
computed before converting back to preserve accuracy. As in R, `score_vector`
uses the working design coordinates. Explicit complete starting vectors bypass
initialization and rescaling.
For an estimated-scale model other than an intercept-only model, a numeric
start can contain just the location coefficients. The initializer retains those
coefficients in the original design units and appends log-scales from a
20-iteration intercept-only fit, including weights, offsets, censoring, and
scale strata. It does not solve for the supplied locations. Intercept-only
models require log-scales in a supplied starting vector; fixed-scale models
require only location coefficients. The R `survreg.fit` matrix interface uses
the same native initialization and returns R's null-fit metadata.

Automatic initialization reports an error for an unusable response scale,
constant nonbinary covariate that cannot be rescaled, or interval probability
that rounds to zero. Complete explicit starts remain available for these cases.

### Cox Proportional Hazards Model

```python
from survival import regression

event_times = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1]  # 1 = event, 0 = censored
x1 = [0.0, 0.4, 0.8, 0.2, 1.0, 1.4, 0.6, 1.2, 1.6, 1.8]
x2 = [0.2, 0.16, 0.62, -0.07, 0.95, 0.61, 0.49, 0.68, 1.24, 0.97]
covariates = [[a, b] for a, b in zip(x1, x2, strict=True)]

# R: coxph(Surv(time, status) ~ x1 + x2, ties = "efron")
fit = regression.coxph_fit(event_times, status, covariates, method="efron", iter_max=20)

print(f"Coefficients: {fit.coefficients}")  # NaN marks an aliased column, as R's NA
print(f"Variance: {fit.var}")
print(f"Log-likelihood: {fit.loglik}")  # at the initial and the final coefficients
print(f"Hazard ratios: {fit.hazard_ratios()}")

# R: predict(fit, newdata, type = "lp" | "risk" | "expected" | "survival")
new_covariates = [[0.5, 0.3], [1.5, 0.9]]
risk = fit.predict("risk", newdata=new_covariates, se_fit=True)
print(f"Risk: {risk.fit}, se: {risk.se_fit}")

# R: basehaz(fit) and survfit(fit, newdata)
baseline = fit.basehaz()
(curve,) = fit.survfit(newdata=new_covariates)
print(f"Baseline hazard: {baseline.hazard}")
print(f"Survival curves: {curve.surv}")  # one column per newdata row

# R: residuals(fit, type = ...) and cox.zph(fit)
print(fit.martingale_residuals())
print(fit.dfbeta())
zph = regression.cox_zph(fit)
print([(test.chisq, test.p) for test in zph.table], zph.global_test.p)
```

### Cox Martingale Residuals

```python
from survival import core, residuals

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 1 = event, 0 = censored
score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # exp(linear predictor)

# R's coxmart.c kernel: martingale residuals for given risk scores
martingale_residuals = residuals.coxmart(
    core.CoxMartInput(
        core.SurvivalData(time, status),
        score,
        core.Weights.unit(len(time)),  # case weights
        [0] * len(time),  # strata
    ),
    ties="efron",  # or "breslow"
)

print(f"Martingale residuals: {martingale_residuals}")
```

### Survival Difference Tests (Log-Rank Test)

```python
from survival import surv_analysis

# Example: Compare survival between two groups
time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # Group 1 and Group 2

# R: survdiff(Surv(time, status) ~ group, rho = 0)
result = surv_analysis.survdiff(
    time,
    status,
    group,
    strata=None,  # Optional: stratification variable
    rho=0.0,  # 0.0 = log-rank; nonzero values use G-rho weights
)

print(f"Observed events: {result.obs}")  # groups x strata
print(f"Expected events: {result.exp}")
print(f"Chi-squared statistic: {result.chisq}")
print(f"Degrees of freedom: {result.df}")
print(f"p-value: {result.pvalue}")
print(f"Variance matrix: {result.var}")
```

### Publication Bias Tests

`survival.validation.publication_bias_tests(effects, std_errors)` returns Egger
intercept, standard error, statistic, and two-sided Student-t probability, along
with Begg and trim-and-fill results. The Egger calculation regresses standardized
effects on precision with an intercept and uses `n - 2` degrees of freedom.
Changing units for both effects and standard errors preserves its inference.

If precision has insufficient variation to identify the two-parameter
regression, or standardized effects cannot be represented as finite values,
the four Egger fields are `NaN`; other results remain available. With exactly
zero residual variance, a nonzero intercept has an infinite statistic and zero
probability. A zero intercept and zero standard error produce `NaN` for the
statistic and probability.

### Built-in Datasets

The library includes 33 classic survival analysis datasets:

```python
from survival import datasets

# Load the lung cancer dataset
lung = datasets.load_lung()
print(f"Columns: {list(lung)}")
print(f"Number of rows: {len(lung['time'])}")

# Load the acute myelogenous leukemia dataset
aml = datasets.load_aml()

# Load the veteran's lung cancer dataset
veteran = datasets.load_veteran()
```

Datasets are returned as column-oriented dictionaries that map each column name,
in R's column order, to a list of values, so they can be passed straight to the
formula functions or to `pandas.DataFrame`/`polars.DataFrame`.

**Available datasets** (every data frame shipped by R's `survival` 3.8, with R's
exact values, column names and storage modes: R `double` -> `float`, `integer`
-> `int`, factor/character/Date -> `str` (ISO dates), logical -> `bool`, `NA`
-> `None`):

`aml` (alias `leukemia`), `bladder`, `bladder1`, `bladder2`, `braking`,
`capacitor`, `cgd`, `cgd0`, `colon`, `cracks`, `diabetic`, `flchain`, `gbsg`,
`genfan`, `heart`, `hoel`, `ifluid`, `imotor`, `jasa`, `jasa1`, `kidney`,
`logan`, `lung` (alias `cancer`), `mgus`, `mgus1`, `mgus2`, `myeloid`,
`myeloma`, `nafld1`, `nafld2`, `nafld3`, `nwtco`, `ovarian`, `pbc`, `pbcseq`,
`rats`, `rats2`, `retinopathy`, `rhDNase` (`load_rhdnase`), `rotterdam`,
`solder`, `stanford2`, `tobin`, `transplant`, `turbine`, `udca`, `udca1`,
`udca2`, `valveSeat` (`load_valveseat`), `veteran` — each as
`datasets.load_<name>()`.

The US, US-by-race and Minnesota population rate tables (`survexp.us`,
`survexp.usr`, `survexp.mn`) are shipped as R's exact tables; see
`survival.population`.

## Cox model reports

`r.print_coxph(fit)` returns coefficient tables and formatted text for ordinary,
robust, penalized and multistate Cox fits. Use
`r.print_summary_coxph(r.model_summary(fit))` for confidence intervals and
model tests. Reports retain full-precision tables and statistics and support
`as_data_frame`. See [Cox model reports](docs/cox-model-reports.md) for options
and R compatibility.

## AFT and Aalen model reports

`r.print_survreg(fit)` and `r.print_aareg(fit)` provide model reports, with
`r.print_summary_survreg` and `r.print_summary_aareg` for numerical summaries.
AFT reports include scales and optional correlations; Aalen reports preserve
robust uncertainty when changing event cutoffs or test weights. See
[AFT and Aalen reports](docs/aft-aalen-reports.md) for options, R differences
and influence-reduction memory measurements.

## Diagnostic and test reports

`r.print_cch`, `r.print_summary_cch`, `r.print_clogit`, `r.print_cox_zph`,
`r.print_concordance`, `r.print_survConcordance` and `r.print_survdiff`
return formatted text and full-precision tables. Formula reports retain
missing-row notices; stratified case-cohort and concordance tables retain
their labels. See [diagnostic and test reports](docs/diagnostic-test-reports.md)
for defaults, data-frame conversion and R comparisons.

`r.print_pyears`, `r.print_survcheck` and `r.print_yates` report population
totals, transition consistency checks and marginal means with their tests.
Person-years results retain match summaries for built-in population rate
tables. See [population and validation reports](docs/population-validation-reports.md)
for examples, metadata and reference comparisons.

`r.match_ratetable(data, table)` exposes population starting positions,
cutpoints and built-in summaries. It validates categorical levels before
population calculations and accepts date and duration columns. See
[rate-table matching](docs/rate-table-matching.md) for units, validation and
the shared Rust lookup implementation.

## Survival curve reports

`r.print_survfit(fit)` returns a compact report of sample sizes, events and
median survival. Add `rmean="common"` for restricted means. Multistate fits
report time in each state through the same function or `r.print_survfitms`.
Use `print(report)` for formatted text and `r.as_data_frame(report)` for
full-precision columns. See [survival reports](docs/survival-reports.md) for
cutoffs, units, formatting and allocation measurements.

For detailed time rows, use `r.print_summary_survfit(r.summary_survfit(fit, times=[100, 300]))`.
Expected curves and their summaries have `r.print_survexp` and
`r.print_summary_survexp`. These reports also expose full-precision tables and
support `as_data_frame`.

Raw responses have `r.print_surv` and `r.print_surv2`; population rate arrays
have `r.print_ratetable`. Their reports preserve full-precision data while
supporting R-style wrapping and explicit display limits. See
[response and rate-table reports](docs/response-ratetable-reports.md).

## Survival curve plots

Install `survival[plot]` to render Kaplan–Meier, Cox and multistate curves:

```python
from survival import datasets, plotting, r

fit = r.survfit("Surv(time, status) ~ sex", datasets.load_lung())
plot = plotting.plot_survfit(fit, conf_int=True, conf_style="band", mark_time=True)
plot.axes.figure.savefig("survival.svg")
```

The [plotting guide](docs/survival-plotting.md) covers transformations, confidence
bars, overlays, scaling, R compatibility and benchmarks. Numerical plot data
remain available without Matplotlib through `plotting.survfit_plot_data`.

The generic `plotting.plot` also fits and plots raw `r.Surv` responses.
`plotting.lines(expected_fit, ax=plot.axes)` overlays `r.survexp` results with
R's straight-line default. The same generic functions select the Cox diagnostic
and Aalen methods below.

For proportional-hazards diagnostics, use
`plotting.plot_cox_zph(r.cox_zph(cox_fit))`. This draws natural-spline curves
and two-standard-error bands over the scaled Schoenfeld residuals. Set
`hr=True` for hazard ratios; see the [diagnostic plotting guide](docs/cox-diagnostic-plotting.md).

For Aalen additive models, `plotting.plot_aareg(aalen_fit)` draws cumulative
coefficient curves with ordinary or influence-based bands. Use
`plotting.lines_aareg` for overlays; see the [Aalen plotting guide](docs/aalen-plotting.md).

## Scikit-learn estimators

`survival.sklearn_compat` provides estimators and streaming wrappers that accept
two-dimensional feature arrays with finite real values, including when
scikit-learn is not installed. Numeric strings are converted to float64; complex
values, masked entries, and missing or infinite values are rejected before
fitting or prediction. Streaming methods check each batch without materializing
the full input. Direct calls require at least one sample; batched predictions can
return empty outputs. Cox, AFT, and tree models support arrays with zero feature
columns; DeepSurv requires at least one feature.

## API Reference

The public Python surface is broad and evolves quickly. For the most accurate,
version-matched signatures, use the checked-in type stubs:

`import survival` exposes the domain modules, the R-style entry points and the
scikit-learn estimators; every Rust binding is reachable through exactly one
domain module (`survival.regression.coxph_fit`, `survival.surv_analysis.survfitkm`,
...) and nothing else is re-exported from the package root.

- [`python/survival/__init__.pyi`](python/survival/__init__.pyi): package-level
  typed surface, including the new domain modules.
- [`python/survival/_survival.pyi`](python/survival/_survival.pyi): core
  PyO3 bindings exposed by `survival._survival`, generated by
  `python3 scripts/generate_stubs.py` from the built extension (checked in CI).
- [`python/survival/*.py`](python/survival): curated domain modules layered on
  top of the generated bindings.
- [`python/survival/sklearn_compat.py`](python/survival/sklearn_compat.py):
  scikit-learn-compatible estimators and streaming wrappers.

The sklearn estimators accept targets with shape `(n_samples, 2)` and columns
`[time, status]`. Times must be finite real numbers; each model applies its own
response-domain restrictions. Status must be exactly `0` (censored) or `1`
(event), with boolean values also accepted. Missing, masked, complex, or
fractional event indicators raise `ValueError` before fitting or scoring.

To inspect available symbols at runtime:

```python
import survival

public_names = [name for name in dir(survival) if not name.startswith("_")]
print(public_names)
```

Or inspect a specific domain module:

```python
from survival import regression, validation

print(regression.__all__[:10])
print(validation.__all__[:10])
```

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full local development
workflow, feature-test matrix, and binding/stub update process.

Install development dependencies:
```sh
uv sync --extra dev --extra test --no-install-project
```

Build the extension in your current environment:
```sh
maturin develop --release
```

Build with optional ML bindings:
```sh
maturin develop --release --features extension-module,ml
```

`Cargo.toml` is the source of truth for the published package version.

GitHub Actions publishes from an explicit tag or full commit SHA. PyPI/TestPyPI publishing is configured for trusted publishing rather than a long-lived API token.

Build the Rust library:
```sh
cargo build
```

Run Rust tests:
```sh
cargo test
```

Run Python tests:
```sh
uv run --no-sync pytest python/tests -v
```

Smoke-test benchmarks:
```sh
cargo bench -- --test
```

The Rust benchmarks use divan; a single group runs with, for example:

```sh
RAYON_NUM_THREADS=1 cargo bench --bench survival_benchmarks -- survreg_bench
```

`benches/python/bench_vs_r.py` times the same synthetic data against R's
`survival` at two separate layers: the `survival.r` formula calls against R's
formula calls (what users pay, including model frames and the extras R
computes), and the `survival._survival` kernels on NumPy arrays against the
matching R entry points (`coxph.fit` plus `concordancefit`, `survfitKM`,
`survdiff.fit`, `concordancefit`, `survreg.fit`). Each ratio compares one layer
with the same layer in R; run it against a release build:

```sh
PYTHONPATH=python python benches/python/bench_vs_r.py --sizes 1000,10000,100000
```

Format and lint:
```sh
cargo fmt
uv run --no-sync ruff format python/ test/ --check
uv run --no-sync ruff check python/ test/
uv run --no-sync mypy python/survival/__init__.pyi python/survival/_survival.pyi python/survival/r python/tests/typing_smoke.py --ignore-missing-imports --follow-imports=silent --check-untyped-defs
```

The codebase is organized with:
- Domain-oriented Rust modules in `src/`
- Matching Python domain modules in `python/survival/`
- Experimental R bridge package in `r/survivalr/`
- Package/type stubs in `python/survival/__init__.pyi`,
  `python/survival/_survival.pyi`, and `survival.pyi`
- Runnable examples in `examples/`
- Developer-facing layout notes in `docs/`
- Rust unit/integration tests in `src/tests/`
- Python binding tests in `python/tests/`
- R reference fixtures in `test/r`

## Dependencies

Primary dependencies are defined in [`Cargo.toml`](Cargo.toml) and
[`pyproject.toml`](pyproject.toml), including:

- [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin) for Python bindings
- [reticulate](https://rstudio.github.io/reticulate/) for the experimental R bridge package
- [numpy](https://numpy.org/) and [ndarray](https://github.com/rust-ndarray/ndarray) for array interop
- [faer](https://github.com/sarah-ek/faer-rs), [rayon](https://github.com/rayon-rs/rayon), and [burn](https://github.com/tracel-ai/burn) for numerical compute

## Compatibility

- Native extendr bindings are currently disabled. The experimental
  `r/survivalr` package provides an R facade through reticulate and the Python
  `survival.r_api` module.
- Python 3.11+ and Rust 1.94+ are required.
- macOS users: Ensure you are using the correct Python version and have Homebrew-installed Python if using Apple Silicon.

Where R's own results are wrong (for example the martingale residuals of
stratified penalized Cox fits, or the offsets of `predict.coxph`'s expected
counts), the port returns the intended values; every such case is listed in
[R compatibility](docs/r-compatibility.md#deliberate-fixes-of-r-defects-no-fixture).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
