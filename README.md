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
`factor(...)` / `as.factor(...)`, `strata(...)`, interaction terms with `:`
or `*`, and numeric `offset(...)` terms are supported, along with one-column
numeric transforms `log(...)`, `sqrt(...)`, and `exp(...)`, plus
`I(...)`/`identity(...)` arithmetic with `+`, `-`, `*`, `/`, and `^`;
time transforms should use the lower-level matrix APIs until they have
dedicated Rust-backed support.
Formula calls also accept `subset=` as a boolean mask or zero-based row indices
and `na_action="omit"` for row-wise missing-data omission across formula
columns and external row-aligned arrays such as `weights`, `offset`, and
`strata`.
R `survobrien` formula expansion preserves factor keeper columns while applying
the risk-set transform only to continuous terms.
R `finegray` formulas use the same Python formula engine and Rust interval
expansion, with sorted censoring-risk sweeps and R-compatible factor classes.
Kaplan-Meier `survfit` calls honor `conf_level=`, R-style `conf_type=`
choices for confidence intervals, `start_time=` for conditional curves, and
`time0=True` to include the starting row.
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
Fitted Cox models can also be passed to `survfit(...)` with optional `newdata=`
to produce model-based survival curves.
The R facade's low-level `coxsurv.fit` and `survfitcoxph.fit` entry points use
an `O(n log n)` Rust risk-set sweep for weighted, stratified, tied-event, and
counting-process baselines, while retaining R-compatible curve and uncertainty
shapes for ordinary predictions and individual time-dependent trajectories.
`survdiff` uses the same right-censored and delayed-entry response forms.
`coxph` uses Efron's tie handling by default, matching R, and also accepts
`ties="breslow"` or the compatibility alias `method="breslow"`.
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

The raw `tmerge`, `tmerge2`, and `tmerge3` sweeps remain available from
`survival.data_prep` for callers that already manage sorted numeric arrays.
The R-style `predict(...)` and `fitted(...)` generics support Cox linear
predictors, relative risk scores, term contributions, survival curves, and
expected event counts.
For `survreg` fits it supports response-scale predictions, linear predictors,
term contributions, and quantile predictions via `type="quantile"`.
The AFT optimizer uses positive-definite observed-information Newton steps when
available and falls back to the stable outer-product system otherwise. The R
bridge also routes built-in `survreg.fit` matrix calls through this kernel,
including fixed or stratified scales and interval-censored responses.
Model helpers include `model_formula`, `model_weights`, `df_residual`,
`loglik`, `aic`, `bic`, `extract_aic`, coefficient, variance-covariance,
confidence-interval, model-matrix/model-frame, and summary accessors for fitted
Cox and `survreg` models.
Common result objects can be converted to column-oriented tables with
`as_data_frame(...)`; the experimental R bridge exposes the same path through
`as.data.frame(...)`, `summary(...)`, and `print(...)` methods.
`Surv` responses also support table conversion for quick data inspection.
The `survival.residuals` name remains the residual diagnostics module; the
R-style residual generic is available as `survival.r_api.residuals(...)` for
fitted Cox and `survreg` models (`survival.r_api` re-exports the `survival.r`
package, which holds the implementation split by concern).

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
from survival import core

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

### Built-in Datasets

The library includes 33 classic survival analysis datasets:

```python
from survival import datasets

# Load the lung cancer dataset
lung = datasets.load_lung()
columns = [name for name in lung if not name.startswith("_")]
print(f"Columns: {columns}")
print(f"Number of rows: {lung['_nrow']}")

# Load the acute myelogenous leukemia dataset
aml = datasets.load_aml()

# Load the veteran's lung cancer dataset
veteran = datasets.load_veteran()
```

Datasets are returned as column-oriented dictionaries with `_nrow` and `_ncol`
metadata.

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

Format and lint:
```sh
cargo fmt
uv run --no-sync ruff format python/ test/ --check
uv run --no-sync ruff check python/ test/
uv run --no-sync mypy python/survival/__init__.pyi python/survival/_survival.pyi --ignore-missing-imports
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
