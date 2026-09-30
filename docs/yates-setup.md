# Prepared Yates predictions

`survival.r.yates_setup` exposes R's prediction dispatcher for population
marginal means. Cox `lp`/`linear` return `None`, `risk` returns a callable,
and `survival` returns a `YatesSurvivalSetup` with callable `predict` and
`summary` entries. Unique prediction abbreviations are accepted.

```python
import numpy as np
from survival import datasets, r

fit = r.coxph("Surv(time, status) ~ age", datasets.load_lung())
setup = r.yates_setup(fit, "survival", options={"rmean": 365})
predictions = setup.predict([-0.5, 0.0, 0.5])
assert predictions.shape[0] == 3
# In a simulation, supply the corresponding population means and variances:
curves = setup["summary"](predictions, np.zeros_like(predictions))
```

The Cox baseline is computed once with `survfit(fit, censor=False)`. Each
prediction row contains restricted mean survival, survival at time zero,
then survival at every original baseline time. `options['rmean']` defaults
to the last baseline time. Prediction accepts a scalar, vector, or single-row
or single-column matrix; `X` is accepted and unused. Risk prediction preserves
the input's numeric shape. Full matrices are rejected for survival prediction
because R's `outer`/`cbind` recycling does not produce one row per predictor.

For times `t`, hazards `H`, and `eta`, survival is `exp(-exp(eta) * H)`.
The restricted mean is the left-step sum using widths
`diff(c(0, pmin(rmean, t)))`. There is no extrapolation beyond the last time.
With no events, predictions contain zero restricted mean and unit time-zero
survival for finite `eta`; summary curves have zero time rows and retain
their population count in `ncurve`. Nonfinite `eta` follows floating-point
exponential arithmetic, including NaN from zero times infinite risk.

Stratified and multi-state survival setups remain unsupported, matching R.
Other setup options and extra keywords are unused; `options['seed']` controls
simulation only when passed to `r.yates`. The higher-level `yates` keeps its
own option validation and simulation behavior.

## Inverse links and default dispatch

For an external GLM, provide a callable `fit.family.linkinv`, a family mapping
with `"linkinv"`, or `fit.model.family.link.inverse` as in statsmodels.
`link`/`linear` return `None`; `response` returns a callable that passes a
numeric NumPy input to that inverse link. No GLM fitting dependency is needed.
GLM `terms` and Cox `expected`/`terms` retain R's explicit errors.

Other fitted objects use the default method: return `None`, ignore `predict`,
and warn only for an explicit `type` other than `linear` or `link`. R's
default method uses `type` while its Cox and GLM methods use `predict`.

## Full simulations for external GLMs

`r.YatesModel` adapts an external GLM for full `r.yates` analysis. Supply its
formula, model data, coefficients and coefficient covariance in R design-column
order, including the intercept. The optional `family` accepts a mapping with
`linkinv`, an object with `linkinv`, or a statsmodels-style family with
`link.inverse`. The adapter never refits the model.

```python
adapter = r.YatesModel(
    "y ~ group + age", data, coefficients, covariance,
    family={"linkinv": lambda eta: 1 / (1 + np.exp(-eta))},
)
result = r.yates(adapter, "group", predict="response", nsim=200,
                 options={"seed": 123})
```

Here `data`, `coefficients` and `covariance` come from the external fit.
`link`/`linear` use the exact linear contrast calculation; `response` averages
the inverse link at the supplied coefficients, then estimates uncertainty from
R-compatible normal draws. The shared Rust loop computes predictors, population
reductions, online covariance and contrast tests. Python receives one NumPy
vector containing all populations for the point estimate and for each draw:
`nsim + 1` inverse-link calls. The callback must return a finite numeric vector
with one response per input row. Its errors retain the callback's message.

Optional model `weights` are finite nonnegative case weights with positive
total. As in R, they weight linear estimates for the data population; nonlinear
averages are unweighted. Model offsets are omitted from both calculations,
following R's Yates convention. Data, factorial, SAS and explicit populations,
joint variables, selected levels, interactions, aliased coefficients and
pairwise tests use the existing formula and contrast machinery. SAS type III
tests require linear predictions and the SAS population. GLMs normally leave
`sigma2=None`; it supplies sum-of-squares columns for external linear models.

Simulation state and the random stream belong to each call. Python facade
results retain neither the model nor the inverse link, and support pickle. An adapter can be
pickled when its family function can be pickled. Native Python callers can use
`survival.validation.yates_response` with prepared population matrices; Rust
callers use `YatesPredictor::Response` with a vectorized, fallible inverse link.
The prepared `YatesPrediction` object continues to represent Cox risk or
survival; external links are called directly by the shared simulator.

Thirty stock-R GLM cases in `scripts/generate_yates_glm_reference.R` cover
binomial, Poisson, Gaussian, Gamma and quasi families; eight inverse links;
case weights, offsets, aliases, interactions, populations, level selection,
pairwise and type III tests. Each case runs through all three family protocols.
Independent checks reconstruct the simulated covariance from recorded callback
batches, compare a log link with native Cox risk, and check concurrent calls,
ownership, serialization and invalid callbacks.

## R formula interface and caller-owned simulations

`survivalr::yates` now builds formulas and population designs in R and computes
marginal means, estimability, covariance, contrasts and SAS type III tests with
the shared Rust kernels. It accepts R linear models, GLMs and Cox models, and
Python-backed Cox and AFT fits. It retains fitted factor levels, contrasts,
transformation metadata, case weights, subject ids, result names and the
original call. No model is refitted. AFT calculations select the coefficient
covariance, excluding fitted scale parameters.

```r
set.seed(123)
result <- survivalr::yates(fit, "group", predict = "response", nsim = 200)
```

Nonlinear simulations consume R's global random stream through `rnorm`.
`set.seed`, `RNGkind`, the subsequent random stream, and random draws inside
custom prediction methods retain their R behavior. Linear analyses consume
no random values. R's `options$seed` remains unused. Python `r.yates` keeps
its existing per-call seed convention.

Custom S3 `yates_setup` methods may return a prediction function or a list
with `predict` and `summary` functions. The first prediction receives the
linear predictor and stacked population design; later calls receive the
linear predictor alone. Vector predictions supply marginal means and tests.
Matrix predictions also supply simulation means and sample variances of all
columns to the summary method, or return them as `pmm` and `mvar2`. Column
counts must remain constant. Built-in Cox risk and survival run directly in
Rust, with one baseline prepared per call.

The native Python functions `validation.yates_risk`, `yates_response`,
`yates_survival` and `yates_predict` accept `normal_draws`: an `nsim` by
coefficient-count matrix of standard-normal values, or a callable receiving
those two dimensions. The callable runs once after the point prediction.
When supplied, these draws replace the built-in seeded generator. Rust
callers use `yates_simulate_with_draws`; ordinary `yates_simulate` is unchanged.
`YatesPredictor::Custom` and Python `yates_predict` accept predictions with
multiple columns. The native result's `prediction_mean` and
`prediction_variance` are populated for these multi-column predictions when
any requested population is estimable. Callback errors retain their context.

The R wrapper corrects several stock-R edge cases: the last numeric term is
selectable; reordered variables use their own fitted levels; a levels matrix
specifies joint settings; empty adjustment sets and matrix-valued SAS adjusters
work; and explicit populations preserve custom contrasts. Cox designs align
by coefficient name, excluding strata-only columns. Existing corrections for
non-estimable populations, zero-variance tests and aligned survival summaries
also apply. `trend`, stratified survival prediction, multi-state Cox models,
and functions passed directly as `predict` remain unsupported. Type III tests
require linear predictions, the SAS population and treatment/SAS contrasts.

`scripts/benchmark_yates_r_bridge.R` compares complete R-facing calls against
stock R for linear, GLM response, Cox risk and Cox survival predictions. It
checks numerical results and the final RNG state before timing. Formula and
population preparation, conversion, simulation and result assembly are included;
fitting, warmup and garbage collection are excluded.

On the machine described below, with 5,000 population rows, three target
levels, 200 draws and seven alternating measurements after three warmup calls:

| Complete R-facing analysis | Stock R median (range) | R/Rust median (range) |
| --- | ---: | ---: |
| Linear | 8 ms (7–9) | 5 ms (5–6) |
| GLM response | 83 ms (66–84) | 67 ms (52–70) |
| Cox risk | 74 ms (72–153) | 21 ms (20–21) |
| Cox survival | 4,903 ms (4,609–4,968) | 327 ms (319–334) |

These medians improved by 1.60×, 1.24×, 3.52× and 14.99× respectively.
Survival uses 30 baseline times. A preliminary run with one warmup call
measured 8/6, 77/61, 69/21 and 4,774/320 ms; its first measured linear bridge
call included R JIT compilation. The script now warms each path three times.
Peak memory is not measured, and these local timings do not establish results
for other model or population sizes.

`scripts/benchmark_yates_glm.R` compares complete R and Python marginal-mean
calls on the same already-fitted binomial GLM and explicit population. It
checks means, covariance and the contrast test before timing. A local run on
the machine described below, with 5,000 population rows, three target levels,
seven coefficients, 200 draws and seven measured calls, gave:

| Full response simulation | Median | Sample range |
| --- | ---: | ---: |
| R survival | 68 ms | 65–70 ms |
| Python/Rust | 47 ms | 46–49 ms |

This run improved by 1.45×; an earlier three-sample run measured 63 ms and
42 ms. Timings include population matrix construction, callback execution,
simulation and result assembly. Model fitting, adapter construction, conversion
of the supplied population to a Python mapping, warmup and garbage collection
are excluded. Peak memory is not measured. The numerical kernel reuses its
predictor buffer and accumulates covariance online; it does not retain each
draw's population responses.

## Summary corrections and ownership

`summary(surv, var)` takes matching population-by-prediction matrices and
returns a `CoxSurvfitResult` with time-by-population curves. It shares the
native summary calculation with `yates`, including its documented corrections
to R survival 3.8-12: remove the restricted-mean and time-zero columns, preserve
single-population dimensions, and replace cumulative hazard with `-log(surv)`.
R retains an extra time-zero value without adding a time and leaves the
baseline cumulative hazard unchanged. Both `std_err` and `std_chaz` describe
the returned curves, using `sqrt(var) / surv`; the baseline uncertainty is
not retained. Confidence limits retain R's formula.

The Python setup retains a baseline and precomputed widths and hazards, and
releases the fitted model. Rust fills the output row by row with constant scratch
space: evaluating N predictors at T times uses one N-by-(T+2) result, without
additional predictor-by-time work matrices. Python receives the owned NumPy
result and releases the GIL during evaluation. Cox setups and risk callables
support pickle round trips; an external inverse link must itself be picklable.

Rust callers use `validation::YatesPrediction::new(YatesPredictor)` and
`evaluate(&[f64])`, plus `yates_survival_summary` for matrix views. Python's
`survival.validation` exposes the same prepared predictor and summary helper.
The R bridge preserves R names and matrix attributes. A stock R fit supplies
its stock R baseline; a `survivalr` fit supplies the Rust-computed baseline.

## Validation and benchmark

`scripts/generate_yates_setup_reference.R` records 80 Cox cases from stock R
survival 3.8-12: right/counting responses, case weights, both tie methods,
null fits, one/no event time, four horizons and one/three populations. It keeps
raw R summaries alongside independently corrected results. Seven GLM inverse
links exercise the external-family protocols. Rust tests check the prediction
layout and summary arithmetic by hand, including strided matrix inputs.

An empty-event simulation also has a defined result when no coefficients are
estimable: zero restricted means and standard errors, with no curve times.
R 3.8-12's full `yates` instead fails when it tries to eigendecompose a
zero-by-zero covariance matrix; the direct R setup still defines the empty
baseline predictions. A separate Python regression checks this extension.

With the release extension installed, run
`Rscript scripts/benchmark_yates_setup.R` and set `RETICULATE_PYTHON` to its
Python environment. It requires `pkgload` and `jsonlite`, checks predictions
against stock R before timing, and measures complete R calls including the
R/Python boundary. Fitting, setup, warmup and garbage collection are excluded.
The benchmark measures direct prediction, not a full Yates simulation.

On an Intel Core Ultra 5 325 with R 4.5.3, survival 3.8-12 and Python 3.14.7,
seven repetitions with 5,000 observations and 3,333 baseline times gave:

| Predictors | R survival | Rust bridge |
| ---: | ---: | ---: |
| 1 | 0.07 ms | 0.10 ms |
| 1,000 | 197 ms | 22 ms |

The single-predictor workload batches 100 calls per sample because the timer
has millisecond resolution. Its boundary cost outweighs the smaller native
calculation. The larger workload improved by 8.95× in this run; a previous
seven-sample run measured 146 ms versus 19 ms. These local timings vary with
allocation and machine load and do not measure memory usage.
