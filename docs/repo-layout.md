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

Preferred usage is module-oriented:

```python
from survival import datasets, regression, validation

lung = datasets.load_lung()
fit = regression.survreg(...)
rmst = validation.rmst(...)
```

Top-level imports remain available for compatibility, but new code should prefer
the domain modules.

## Naming Notes

- `survival.reliability` is the callable reliability function.
- `survival.reliability_tools` is the module that groups reliability-related APIs.

## Typing Surface

The main typed entry points are:

- `python/survival/__init__.pyi`: package-level typed surface
- `python/survival/_survival.pyi`: generated binding surface
- `survival.pyi`: compatibility typing surface for downstream tooling
