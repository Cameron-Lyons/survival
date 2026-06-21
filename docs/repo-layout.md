# Repo Layout

This repo is organized around stable domain boundaries instead of catch-all
folders.

## Rust Layout

Top-level Rust modules in `src/` map to major feature areas:

- `regression/`: Cox, AFT, competing-risks, cure, and recurrent-event models
- `surv_analysis/`: Kaplan-Meier, Nelson-Aalen, multistate, baseline, and log-rank code
- `validation/`: metrics, calibration, conformal, RMST, cross-validation, and tests
- `ml/`: neural, tree, and modern ML-oriented survival methods
- `datasets/`: built-in datasets and dataset loading helpers
- `data_prep/`: public preprocessing and time-splitting utilities
- `population/`: expected-survival and rate-table routines
- `monitoring/`: drift and monitoring helpers
- `residuals/`: residual and diagnostic calculations
- `internal/`: shared non-public helpers

Guidelines used in the current layout:

- Keep one primary public concept per file.
- Split files before they become multi-thousand-line mixed-concern modules.
- Put reusable internal helpers in `src/internal/`; otherwise keep helpers next
  to the feature they support.
- Avoid new bucket modules like `utilities` or `specialized`.

## Python Layout

The Python package in `python/survival/` mirrors the Rust domains:

- `datasets`, `data_prep`, `core`
- `regression`, `surv_analysis`, `validation`, `residuals`
- `population`, `monitoring`, `ml`
- `bayesian`, `causal`, `joint`, `interpretability`, `spatial`, and related domains
- `r_api`: R-style formula façade for `Surv`, `survfit`, `survdiff`, `coxph`,
  `survreg`, Cox `predict`, and residual dispatch

Preferred usage is module-oriented:

```python
from survival import datasets, regression, validation

lung = datasets.load_lung()
fit = regression.survreg(...)
rmst = validation.rmst(...)
```

Top-level imports remain available lazily for compatibility, but new code should
prefer the domain modules unless it is using the intentional R-style façade
(`from survival import Surv, survfit, coxph, survreg`).

Python source builds use the lean `extension-module` feature set by default.
Build with `--features extension-module,ml` when the optional ML bindings should
be present.

## R Layout

The experimental R bridge package lives in `r/survivalr/`. It is named
`survivalr` so it can coexist with CRAN's `survival` package while exposing
familiar R-style entry points:

- `Surv`, `survfit` formula/string/fitted-Cox methods, `survdiff`, `coxph`,
  `coxph.control`, `survreg`, and `survreg.control`
- `basehaz`, `concordance`, `cox.zph`, and `coxph.detail`
- S3 methods for `coef`, `vcov`, `confint`, `logLik`, `nobs`, `df.residual`,
  `extractAIC`, `formula`, `terms`, `model.matrix`, `model.frame`, `summary`,
  `fitted`, `predict`, `residuals`, `weights`, and `anova` on bridged model objects
- `as.data.frame` methods for bridged `Surv` responses and common result objects backed by
  `survival.r_api.as_data_frame`
- `summary` and `print` methods for common tabular result objects

This bridge uses `reticulate` to call `survival.r_api` and should remain a thin
facade until native R/extendr bindings are introduced.

## Naming Notes

- `survival.reliability` is the callable reliability function.
- `survival.reliability_tools` is the module that groups reliability-related APIs.

## Typing Surface

The main typed entry points are:

- `python/survival/__init__.pyi`: package-level typed surface
- `python/survival/_survival.pyi`: generated binding surface
- `survival.pyi`: compatibility typing surface for downstream tooling
