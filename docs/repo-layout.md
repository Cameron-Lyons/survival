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
- `internal/`: shared non-public helpers (`validation`, `matrix`, `dist`,
  `statistical`, `sorting`, the typed inputs, and the boundary types below)
- `api/python/`: registration of every `#[pyclass]`/`#[pyfunction]` with the
  `_survival` extension module, split by domain; `api/registration_audit.rs`
  scans `src/` at test time and fails when a declared binding is missing from
  the registration files, or registered twice
- `pybridge/`: `#[pyfunction]`s that exist only as Python entry points, such as
  `cox_callback` (the R `cox_Rcallback.c` hook `coxpenal.fit` uses)

The crate root exposes the domain modules, `error` (`SurvivalError`,
`SurvivalResult`, also re-exported at the root), `data_types` (the typed inputs)
and `prelude` (domain modules, error types, typed inputs). There are no other
root-level re-exports: name items through their domain module
(`survival::regression::coxph_fit`).

Guidelines used in the current layout:

- Keep one primary public concept per file.
- Split files before they become multi-thousand-line mixed-concern modules.
- Put reusable internal helpers in `src/internal/`; otherwise keep helpers next
  to the feature they support.
- Avoid new bucket modules like `utilities` or `specialized`.
- Core algorithms return `SurvivalResult`; `PyResult` appears only in
  `#[pyfunction]`/`#[pymethods]` wrappers (`From<SurvivalError> for PyErr`
  makes `?` work there). Results crossing into Python are typed `#[pyclass]`
  structs with named fields.

### Feature flags and the PyO3 shim

The crate compiles with and without the `python` feature. Without it,
`src/lib.rs` declares `extern crate self as pyo3;` and `src/pyo3_shim.rs`
supplies the handful of PyO3 items domain code names (`PyErr` with its
exception class, `PyResult`, `Python`, `Py`, `Bound`, `PyRefMut`, an inert
`PyDict`), while `survival-pyo3-macros-shim` turns the attribute macros into
no-ops. The shim's module docs spell out its contract; keep it small and add an
item only when a Rust-only build needs it, copying PyO3's signature. Code that
must inspect or build Python objects belongs behind `#[cfg(feature =
"python")]`, or better, behind a typed `#[pyclass]`.

`build.rs` links `libpython` only for a plain `cargo` invocation with the
`extension-module` feature (`cargo test --all-features`); maturin sets
`PYO3_BUILD_EXTENSION_MODULE=1`, so wheels never depend on `libpython`.

### Boundary input types

`src/internal/numpy_utils.rs` (re-exported from `survival::data_types`)
defines the argument types for `#[pyfunction]` signatures:

| Type          | Accepts                                                                 | Converts to   |
| ------------- | ----------------------------------------------------------------------- | ------------- |
| `FloatVec`    | 1-D NumPy array of any dtype/strides, pandas/polars column, sequence    | `Vec<f64>`    |
| `IntVec`      | integer/bool arrays, integral floats, sequences (checked `i32` narrowing) | `Vec<i32>`    |
| `BoolVec`     | bool arrays, `0`/`1` numerics, sequences                                 | `Vec<bool>`   |
| `FloatMatrix` | 2-D array of any layout, list of rows, 1-D input (as one column)        | `Array2<f64>` (row-major) |

Each also implements `IntoPyObject`, so returning one (or exposing it through a
`#[pyo3(get)]` field) hands Python a NumPy array without a `.tolist()` round
trip; `FloatMatrix::from_flat(values, ncol)` covers flat buffers with an
explicit column count.

Migration for binding owners, one signature at a time:

1. Replace `Vec<f64>`/`Vec<i32>`/`Vec<bool>` parameters with `FloatVec`/
   `IntVec`/`BoolVec`, and `Vec<Vec<f64>>` or flat-plus-`ncol` pairs with
   `FloatMatrix`; call `.into_inner()` (or deref to a slice / `Array2`) where
   the core routine is invoked. Python callers keep passing lists; NumPy and
   DataFrame columns now work too.
2. Replace `&Bound<PyAny>` parameters that went through `extract_vec_f64`/
   `extract_vec_i32`/`extract_matrix_f64` with the same types; those helpers
   now delegate to them and disappear once the last caller moves.
3. Return `FloatVec`/`FloatMatrix` (or use them as `#[pyo3(get)]` field types)
   for large numeric results so Python receives NumPy arrays.
4. Do not call `Python::attach` inside a `#[pyfunction]`: it already runs
   attached, and typed `#[pyclass]` results need no `PyDict`.

## Python Layout

The Python package in `python/survival/` mirrors the Rust domains:

- `datasets`, `data_prep`, `core`
- `regression`, `surv_analysis`, `validation`, `residuals`
- `population`, `monitoring`, `ml`
- `bayesian`, `causal`, `joint`, `interpretability`, `spatial`, and related domains
- `r_api`: R-style formula façade for `Surv`, `survfit`, `survdiff`, `coxph`,
  `clogit`, `survreg`, Cox `predict`, and residual dispatch. The implementation
  lives in the `survival.r` package and `r_api` is a thin re-export of its
  public surface (plus the private helpers the R bridge reaches through
  `python_attr`), so `survival.r_api` stays the stable import path.

The `python/survival/r/` package exposes only the R-style API from
`survival.r` itself; the implementation modules are private (underscore
names) and layered so imports form a DAG (each module only imports from the
ones above it):

- `_types`: result containers and formula/design dataclasses. `reticulate`
  names R classes after each Python class's `__module__.__name__`, so any
  `inherits(x, "survival.r._types.<Class>")` guard in `r/survivalr/R/bridge.R`
  must be updated if a result class moves to another module
- `_coerce`: input coercion, option normalisation, shared numeric helpers
- `_surv`: `Surv`, `Surv2`, timeline conversion, `format_surv`, `strata`
- `_formula`: tokenizer/parser, terms, design matrices, model-frame builders
- `_fit`: accessors on fitted models and prediction-input helpers shared by
  `_coxph`, `_survreg`, and `_models`
- `_survdiff`, `_concordance`, `_data_prep` (tmerge, survSplit, survcondense,
  neardate, tcut, aeqSurv, lvcf, nostutter, rttright), `_pyears`, `_finegray`
- `_coxph`: `coxph`/`clogit`, Cox tests, `basehaz`, `cox_zph`, `anova`, Cox
  curves and expected events
- `_survfit`: KM/AJ/Turnbull/Cox curves, `survfit0`, aggregation, confint,
  influence; `_survfit_residuals`: `survfit_residuals` and `pseudo`
- `_survreg`: `survreg` fitting, prediction/residual helpers, d/p/q/rsurvreg
- `_models`: generics (`predict`, `residuals`, `coef`, `vcov`, `confint`,
  `model_summary`, `as_data_frame`, ...)
- `_misc`: statefig, brier, royston, yates, cipoisson, bounded links,
  survobrien, survcheck, nsk, pspline; `_aareg`; `_cch`

The typed surface is declared once in `python/survival/r_api.pyi`;
`python/survival/r/__init__.pyi` re-exports it. Tests live in
`python/tests/test_r_<module>.py` with shared builders in
`python/tests/r_api_support.py`.

Preferred usage is module-oriented:

```python
from survival import datasets, regression, validation

lung = datasets.load_lung()
fit = regression.survreg(...)
rmst = validation.rmst(...)
```

Top-level imports remain available lazily for compatibility, but new code should
prefer the domain modules unless it is using the intentional R-style façade
(`from survival import Surv, survfit, coxph, clogit, survreg`).

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

## Test Layout

- Rust unit tests live next to the code they cover; cross-cutting R-parity
  suites live in `src/tests/`.
- Python tests live in `python/tests/`.
- `test/r/` holds the R differential fixtures (`generate_fixtures.R` ->
  `fixtures/*.json`) read by `python/tests/test_r_fixtures.py` and
  `src/tests/r_fixtures.rs`, kept stable by the `r-fixture-stability` CI job.
  There is no archived Rust reference code in the repo.

## Naming Notes

- `survival.reliability` is the callable reliability function.
- `survival.reliability_tools` is the module that groups reliability-related APIs.

## Typing Surface

The main typed entry points are:

- `python/survival/__init__.pyi`: package-level typed surface
- `python/survival/_survival.pyi`: generated binding surface
- `survival.pyi`: compatibility typing surface for downstream tooling
