# Population marginal means

`survival.r.yates` computes population marginal means (R's `yates`) for Cox, AFT,
linear and generalized linear models, with global, pairwise, trend and SAS-style
type III tests. Means, estimability, covariance, simulation and tests run in
shared Rust kernels. Differences from R are listed in
[R compatibility](r-compatibility.md#marginal-means).

## Models, populations and simulation

```python
from survival import datasets, r

fit = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
means = r.yates(fit, "age", levels=[40, 60, 80], test="pairwise")
risk = r.yates(fit, "sex", predict="risk", nsim=200, options={"seed": 123})
```

- `predict` selects linear predictions or, for Cox models, `"risk"` and
  `"survival"`; external GLMs use `"response"`. Nonlinear predictions estimate
  uncertainty by simulating coefficients from R's default Mersenne-Twister and
  inversion normal stream. `options={"seed": ...}` gives reproducible covariance
  without touching a global RNG; the default seed is zero.
- Populations are `"data"`, `"factorial"`, `"sas"` or an explicit design. Joint
  variables, selected levels, interactions, aliased coefficients and pairwise
  tests use the formula and contrast machinery. Factorial and SAS populations keep
  compound strata as evaluated factors; their labels are not parsed back into the
  source columns.
- Joint requests such as `term="a + b"` or `term="a:b"` compare the Cartesian
  product of both variables' levels, with the first varying fastest. A scalar or
  vector of one-based fitted term numbers also selects variables, including those
  of an interaction term, and the last term is selectable. A
  `levels={"a": [...], "b": [...]}` mapping supplies per-variable values; omitted
  categorical variables use their fitted levels. Tests compare all combinations
  jointly, or pairwise on request, and keep the requested variable order; a joint
  global test is named `global`.
- For a Cox model, the design is aligned by coefficient name: strata-only columns
  and the baseline intercept are removed before testing.
- Linear and link predictions can use case weights for the data population;
  nonlinear means are unweighted and model offsets are omitted, following R.
- With no estimable level, or an `rmean` at or before the first time, the tests
  are `NA`, or a test of 0 on 0 degrees of freedom. An empty-event survival
  simulation with no estimable coefficients returns zero restricted means and
  standard errors with no curve times.

All Python Yates kernels release the GIL and accept NumPy arrays; nested matrices
use `FloatRows`.

## SAS-style type III tests

`yates(method="sgtt")` computes SAS-style type III tests for selected main-effect
variables in a treatment-coded model. It defaults to `population="sas"` and needs
linear predictions. Marginal estimates and covariance match the direct
SAS-population calculation, and the test uses the estimable hypotheses of the full
indicator design. `YatesResult.sas`, `sas_names` and `sas_row_names` expose that
matrix and its labels. SGTT returns one test per selected main-effect variable in
the requested order. The Rust `validation::yates_sgtt` kernel takes a
`YatesSgttInput` with the indicator design, term assignments and adjustments,
coefficients and covariance.

## External models

`r.YatesModel(formula, data, coefficients, variance, sigma2=None, *, family=None,
weights=None, linear_predictors=None)` adapts an externally fitted linear model or
GLM without refitting it. Coefficients and covariance follow the formula's design
columns, including the intercept. The adapter uses treatment contrasts. `sigma2`
supplies sum-of-squares columns for linear models and is normally `None` for GLMs.

```python
import numpy as np

adapter = r.YatesModel(
    "y ~ group + age", data, coefficients, covariance,
    family={"linkinv": lambda eta: 1 / (1 + np.exp(-eta))},
)
result = r.yates(adapter, "group", predict="response", nsim=200,
                 options={"seed": 123})
```

`family` accepts a mapping with `linkinv`, an object with `linkinv`, or a
statsmodels-style family with `link.inverse`. `link` and `linear` use the exact
linear contrast calculation; `response` averages the inverse link at the
coefficients and estimates uncertainty from simulated draws. The shared Rust loop
passes one NumPy vector holding all populations to the inverse link for the point
estimate and each draw (`nsim + 1` calls); the callback must return a finite
numeric vector with one value per row, and its errors keep their message. Weights
are finite nonnegative case weights with positive total. Simulation state belongs
to each call; results keep neither the model nor the inverse link and pickle.
Native callers can use `validation.yates_response` with prepared population
matrices; Rust callers use `YatesPredictor::Response` with a vectorized fallible
inverse link.

## Prepared predictions

`r.yates_setup` exposes R's prediction dispatcher. Cox `lp`/`linear` return
`None`, `risk` returns a callable, and `survival` returns a `YatesSurvivalSetup`
with callable `predict` and `summary` entries; unique abbreviations are accepted.

```python
fit = r.coxph("Surv(time, status) ~ age", datasets.load_lung())
setup = r.yates_setup(fit, "survival", options={"rmean": 365})
predictions = setup.predict([-0.5, 0.0, 0.5])
curves = setup["summary"](predictions, np.zeros_like(predictions))
```

- The baseline is computed once with `survfit(fit, censor=False)`. Each prediction
  row holds the restricted mean survival, survival at time zero, then survival at
  every baseline time; `options['rmean']` defaults to the last baseline time.
  Survival is `exp(-exp(eta) * H)` and the restricted mean is the left-step sum
  with widths `diff(c(0, pmin(rmean, t)))`, without extrapolation.
- Prediction accepts a scalar, vector, or single-row or single-column matrix; `X`
  is accepted and unused. Risk prediction keeps the input's numeric shape. Full
  matrices are rejected for survival prediction.
- With no events, predictions have zero restricted mean and unit time-zero
  survival for finite `eta`, and summaries have no time rows but keep their
  population count in `ncurve`. Nonfinite `eta` follows floating-point arithmetic.
- Stratified and multi-state survival setups are unsupported, as in R; other setup
  options and keywords are unused.
- `summary(surv, var)` takes matching population-by-prediction matrices and
  returns a `CoxSurvfitResult` of time-by-population curves: the restricted-mean
  and time-zero columns are removed, single-population dimensions are kept,
  `cumhaz` is `-log(surv)`, and `std_err` and `std_chaz` are `sqrt(var) / surv`.
  Confidence limits keep R's formula.
- For external GLMs, `link`/`linear` return `None` and `response` a callable that
  applies the inverse link; GLM `terms` and Cox `expected`/`terms` keep R's
  errors. Other fitted objects use the default method: return `None`, ignore
  `predict` and warn only for an explicit `type` other than `linear` or `link`.

The setup keeps a baseline and precomputed widths and hazards and releases the
fitted model; evaluation releases the GIL. Cox setups and risk callables pickle.
Rust callers use `validation::YatesPrediction::new(YatesPredictor)` and
`evaluate(&[f64])`, plus `yates_survival_summary`. Sparse frailty is excluded from
the default curve, following R. When all observations share one censored time, the
baseline keeps its training sample count.

## R interface and caller-owned draws

`survivalr::yates` builds formulas and population designs in R and computes the
results with the shared kernels for R linear models, GLMs and Cox models and for
Python-backed Cox and AFT fits (AFT covariance excludes the scale parameters). It
keeps fitted factor levels, contrasts, transformation metadata, case weights,
subject ids, result names and the original call; no model is refitted. Nonlinear
simulations consume R's global random stream through `rnorm`, so `set.seed` and
`RNGkind` behave as in R; linear analyses consume no random values and
`options$seed` is unused. The wrapper accepts a levels matrix for joint settings,
reordered variables, empty adjustment sets, matrix-valued SAS adjusters and
explicit populations with custom contrasts. `trend`, stratified survival
prediction, multi-state Cox models and functions passed as `predict` are not
supported; type III tests need linear predictions, the SAS population and
treatment or SAS contrasts.

Custom S3 `yates_setup` methods may return a prediction function or a list with
`predict` and `summary`. The first prediction receives the linear predictor and
stacked population design, later calls the linear predictor alone. Vector
predictions supply means and tests; matrix predictions also supply simulation
means and sample variances of all columns to the summary, or return them as `pmm`
and `mvar2`. Column counts must stay constant.

The native `validation.yates_risk`, `yates_response`, `yates_survival` and
`yates_predict` accept `normal_draws`: an `nsim` by coefficient-count matrix of
standard-normal values, or a callable taking those dimensions, called once after
the point prediction. Supplied draws replace the seeded generator; Rust callers use
`yates_simulate_with_draws`. `YatesPredictor::Custom` and `yates_predict` accept
multi-column predictions, filling `prediction_mean` and `prediction_variance` when
any population is estimable.
