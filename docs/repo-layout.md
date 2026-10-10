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

The Cox model's `regression/coxph.rs` is a facade over `coxph/types.rs`,
`fitting.rs`, `prediction.rs`, `curves.rs`, and `bindings.rs`. Numerical methods
live separately from Python argument conversion and result packaging; existing
Rust exports and Python class names remain the compatibility boundary. Result
fields retain their serialization order. The optimizer builder accepts borrowed
slices and strided matrix views, then gathers owned sorted data once. Breslow and
Efron evaluations reuse predictor, risk and risk-set buffers, resetting them for
every trial coefficient vector, including rejected steps. Exact evaluators keep
their existing numerical paths.

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
| `FloatRows`   | Same matrix inputs, for kernels that consume nested rows                | `Vec<Vec<f64>>` |
| `FloatArray3` | 3-D arrays of any layout/dtype, rectangular nested lists or tuples      | `Array3<f64>` (row-major) |

The vector types, `FloatMatrix` and `FloatArray3` also implement `IntoPyObject`, so returning
one (or exposing it through a `#[pyo3(get)]` field) hands Python a NumPy array without a `.tolist()` round
trip; `FloatMatrix::from_flat(values, ncol)` covers flat buffers with an
explicit column count.

Unaligned NumPy storage is normalized before constructing Rust views. Empty
inputs return owned empty buffers without constructing a borrowed view. Boolean
arrays first convert to numeric bytes, so noncanonical NumPy true values never
become invalid Rust `bool` references. Kernels receive owned buffers before
the bindings release the GIL.

Rust input structs with public fields must also be validated at the numerical
entry point: callers can construct them directly or modify them after `try_new`.
`SurvfitKMData`, `SurvfitAJData`, and `SurvdiffData` expose `validate()` and
their fitters check all row lengths, times, event codes, and weights before
accessing rows. The public API integration tests exercise modified inputs with
time correction both enabled and disabled.

Cox risk-set counting, score, Schoenfeld and martingale boundaries also check
mutable right/counting responses, covariate shapes, scores, weights and strata
before sorting or indexing. Weighted numerical controls and malformed-input
regressions cover tied events and delayed entry. This includes refusing a NaN
time before a risk-set sweep that would otherwise fail to advance.

Standalone Cox baseline construction checks its raw observation values.
Fitted-Cox prediction boundaries also check stored row, design and covariance
dimensions before using cached baselines or prediction shortcuts. Unknown
training stratum codes return input errors within the existing grouping pass.
Public curve expansion and subject trajectories check mutable baseline vector
lengths, covariate widths and finite ordered time grids before indexing them.
Trajectory intervals may be empty (`start == stop`) but cannot run backwards.
Fitted-model drivers reuse internally constructed baselines through the same
numerical loops, avoiding a new scan for every subject or prediction row.

Population boundaries revalidate mutable `RateTable` attributes and rates before
matching or integration. Internal subject loops reuse the validated table.
Expected-survival group dimensions and public rate offsets use checked arithmetic;
damaged-table formatting also avoids out-of-range indexing. See
[rate-table input validation](ratetable-inputs.md).

The core bindings take these types for every numeric vector and matrix input
(`coxph_fit`, `coxpenal_fit`, `agexact`, `SurvregData`, `cch_fit`,
`aareg_fit`, `pyears`, `survexp`, `survdiff`, `survfitkm`, `survfitaj`, the
pseudo-value and residual kernels, `cox_survfit_baseline`, ...), so NumPy
arrays are read in one copy instead of element by element, and nested lists
keep working. For new bindings:

1. Take `FloatVec`/`IntVec`/`BoolVec` for vectors and `FloatMatrix` for
   matrices, and move `.into_inner()` into the core data type (an `Array2`
   goes straight into, for example, `CoxphData`). `extract_vec_f64`/
   `extract_vec_i32` remain only for `&Bound<PyAny>` arguments of beyond-R
   code. For a kernel that takes nested rows, use the input-only `FloatRows`
   to avoid flattening and rebuilding list inputs; its kernel checks row widths.
   Use `FloatArray3` for three-dimensional probability arrays.
2. Keep getters that Python consumers iterate by row (`CoxPHFit.x`,
   `SurvregData.covariates`) returning lists; return `FloatVec`/`FloatMatrix`
   where the consumer wants NumPy.
3. Do not call `Python::attach` inside a `#[pyfunction]`: it already runs
   attached, and typed `#[pyclass]` results need no `PyDict`.

### Releasing the GIL

A binding whose kernel does more than O(n) work takes `py: Python<'_>`, does
the Python-facing work attached (argument extraction, `*Data::try_new`
validation, option parsing such as `TieMethod::parse`) and runs only the
kernel inside `py.detach(|| ...)`, converting its `SurvivalResult` after it
returns:

```rust
let data = CoxphData::try_new(time.into_inner(), ...)?;
let options = CoxphOptions { method: TieMethod::parse(Some(method))?, ... };
Ok(py.detach(move || CoxPHFit::fit(data, options))?)
```

The closure may capture owned buffers and `&` references to `#[pyclass]`
values (none is `unsendable`, so they are `Sync`), never a `Bound`, a `PyRef`
or a borrowed NumPy view. Code that must call back into Python from a detached
kernel re-attaches with `Python::attach` (the `coxpenal` callback penalty).
The core fit, prediction, log-rank test and residual bindings follow this rule, so fits on
several Python threads run in parallel; `python/tests/test_gil_release.py`
checks it. Survival-summary tables, restricted-mean comparisons, stacked curve
quantiles and confidence-band transforms also run detached. Building the
nested-list results of methods such as `CoxPHFit.dfbeta`
still runs attached, which bounds how far those calls overlap.

## Python Layout

The Python package in `python/survival/` mirrors the Rust domains; each module
binds a disjoint subset of the extension through `bind_names` (the generated
`_binding_manifest.py` records which module owns which symbol, and
`test_binding_contract.py` checks that every registered binding is bound by
exactly one module):

- `datasets`, `data_prep` (0-based row indices everywhere), `core` (the typed
  inputs `SurvivalData`/`CountingProcessData`/`CovariateMatrix`/`Weights`,
  `concordancefit`, the spline bases and the Cox residual kernels)
- `regression` (`coxph_fit` -> `CoxPHFit`, `survreg_fit` -> `SurvregFit`,
  `cox_zph`, `coxph_detail`, `cch_fit`, ...), `surv_analysis` (`survfitkm`,
  `survfitaj`, `survdiff`, summaries and pseudo values), `validation`,
  `residuals` (`coxmart`/`agmart`; model residuals are methods of the fits)
- `population` (`RateTable`, `match_ratetable`, `pyears`, `survexp`),
  `pybridge` (`cox_callback`), `interval` (`turnbull`), `monitoring`, `ml`
- `bayesian`, `causal`, `joint`, `interpretability`, `spatial`, and related domains
- `r_api`: R-style formula façade for `Surv`, `survfit`, `survdiff`, `coxph`,
  `clogit`, `survreg`, Cox `predict`, and residual dispatch. The implementation
  lives in the `survival.r` package and `r_api` is a thin re-export of its
  public surface (plus the private helpers the R bridge reaches through
  `python_attr`), so `survival.r_api` stays the stable import path.
- `plotting`: optional Matplotlib rendering of fitted survival curves;
  `_plot_data` prepares typed NumPy curve coordinates without importing the renderer.
  `_cox_plot_data` prepares Cox diagnostic curves using the Rust smoother.
  `_aalen_plot` accumulates and renders Aalen coefficient curves, with bounded
  memory for influence reductions. `_plot_helpers` shares axes and styling controls.

The `python/survival/r/` package exposes only the R-style API from
`survival.r` itself: R's exported functions (under Python spellings such as
`survreg_control` and `cox_zph`) and the classes they return (`CoxphModel`,
`SurvregModelResult`, `BrierResult`, ...). The implementation modules are
private (underscore names) and layered so imports form a DAG (each module only
imports from the ones above it):

- `_types`: result containers and formula/design dataclasses. `reticulate`
  names R classes after each Python class's `__module__.__name__`, so any
  `inherits(x, "survival.r._types.<Class>")` guard in `r/survivalr/R/bridge.R`
  must be updated if a result class moves to another module
- `_coerce`: input coercion, option normalisation, shared numeric helpers
- `_names`: R's `make.names` and `make.unique` for data-frame column names
- `_surv`: `Surv`, `Surv2`, timeline conversion, `format_surv`, `strata`,
  `cluster`
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
- `_misc`: statefig, brier, royston, cipoisson, bounded links,
  survobrien, survcheck, nsk, pspline; `_aareg`; `_cch`
- `_yates`: population marginal means, joint-variable contrasts, and SAS type
  III tests; `_yates_model`: adapter for externally fitted linear models

Ordinary right/counting Cox formula fits use owned float64 array designs for
numeric terms and interactions, reusing the prediction design builder. Factors,
matrix terms, penalties, time transforms and multistate responses retain their
row-list fitting paths. Public model matrices and fitted design getters remain
lists. Explicit initial-value checks retain scalar accumulation and error
behavior. Other model-frame consumers retain their existing list designs.

`benches/python/bench_formula_cox_fit.py` measures complete formula and native
Cox fits. Run a saved package and the current package in separate processes by
selecting `PYTHONPATH`; use `--compare` with the earlier run's NPZ file to require
identical fitted values, prediction errors, residuals and baseline hazards.
Build both packages with the same toolchain and release settings and pin them
to the same CPU when comparing timings. The
[Cox architecture measurements](benchmarks/cox-architecture-2026-10-10.json)
record raw samples and compatibility checks: complete numeric formula fits took
31–60% less time at 10,000 rows and 43–55% less at 100,000 rows on this machine.
These measurements cover unstratified right-censored Efron fits. Counting,
exact and special-term behavior is checked separately by regression tests.

`survival.r` has no stub: the inline annotations of its modules are the typed
surface (the package ships `py.typed`), so a signature is changed in one place.
`python/tests/test_public_surface.py` checks that every export resolves and
every public function declares its return type, and CI runs mypy on
`python/tests/typing_smoke.py`. Tests live in
`python/tests/test_r_<module>.py` with shared builders in
`python/tests/r_api_support.py`.

Preferred usage is module-oriented:

```python
from survival import datasets, regression, validation

lung = datasets.load_lung()
fit = regression.survreg_fit(...)
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

- `python/survival/__init__.pyi`: package-level typed surface, mirroring the
  export lists of `__init__.py`
- `python/survival/_survival.pyi`: generated by `scripts/generate_stubs.py` from
  the built extension (structure) and the Rust sources (annotations); checked in
  CI with `--check`
- `python/survival/r/`: the inline annotations of the R-style API, exercised by
  `python/tests/typing_smoke.py` (mypy with `--follow-imports=silent`, so the
  check covers the public types rather than the package internals)
