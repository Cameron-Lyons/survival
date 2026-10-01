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
This direct callback interface does not add external GLM support to `r.yates`.
GLM `terms` and Cox `expected`/`terms` retain R's explicit errors.

Other fitted objects use the default method: return `None`, ignore `predict`,
and warn only for an explicit `type` other than `linear` or `link`. R's
default method uses `type` while its Cox and GLM methods use `predict`.

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
