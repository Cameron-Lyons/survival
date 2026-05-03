# survival

[![Crates.io](https://img.shields.io/crates/v/survival.svg)](https://crates.io/crates/survival)
[![PyPI version](https://img.shields.io/pypi/v/survival.svg)](https://pypi.org/project/survival/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance survival analysis library written in Rust, with a Python API powered by [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

## Features

- Core survival analysis routines
- Cox proportional hazards models with frailty
- Kaplan-Meier and Aalen-Johansen (multi-state) survival curves
- Nelson-Aalen estimator
- Parametric accelerated failure time models
- Fine-Gray competing risks model
- Penalized splines (P-splines) for smooth covariate effects
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

Install the wheel:
```sh
pip install target/wheels/survival-*.whl
```

For development:
```sh
maturin develop --release
```

## Python Package Layout

Prefer domain modules in new code:

```python
from survival import core, datasets, regression, surv_analysis, validation

lung = datasets.load_lung()
fit = regression.survreg(...)
km = surv_analysis.survfitkm(...)
score = validation.rmst(...)
```

Top-level imports such as `from survival import survreg` still work for
compatibility, but module imports are the preferred style because they match the
current repo layout and keep the API easier to navigate.

`survival.__all__` and `dir(survival)` expose the curated package surface:
domain modules and scikit-learn helpers. Legacy root-level algorithm exports are
still available for compatibility and are listed in
`survival.__deprecated_root_exports__`.

Common modules:
- `survival.datasets`: built-in example and benchmark datasets
- `survival.data_prep`: time splitting and data transformation helpers
- `survival.core`: shared concordance, spline, and low-level core routines
- `survival.regression`: Cox, AFT, competing-risks, cure, and recurrent-event models
- `survival.surv_analysis`: Kaplan-Meier, Nelson-Aalen, multistate, and log-rank helpers
- `survival.validation`: metrics, calibration, conformal, RMST, and statistical tests
- `survival.residuals`: martingale, Schoenfeld, and related residual diagnostics
- `survival.population`: expected-survival and rate-table routines
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
from survival import regression

data = [
    [1.0, 0.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
]
variable_names = ["time", "event", "covariate1"]

# Create options with required parameters (formula, data, variable_names)
options = regression.AaregOptions(
    formula="time + event ~ covariate1",
    data=data,
    variable_names=variable_names,
)

# Optional: modify default values via setters
# options.weights = [1.0, 1.0, 1.0]
# options.qrtol = 1e-8
# options.dfbeta = True

result = regression.aareg(options)
print(result)
```

### Penalized Splines (P-splines)

```python
from survival import core

x = [0.1 * i for i in range(100)]
pspline = core.PSpline(
    x=x,
    df=10,
    theta=1.0,
    eps=1e-6,
    method="GCV",
    boundary_knots=(0.0, 10.0),
    intercept=True,
    penalty=True,
)
pspline.fit()
```

### Concordance Index

```python
from survival import core

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
weights = [1.0, 1.0, 1.0, 1.0, 1.0]
indices = [0, 1, 2, 3, 4]
ntree = 5

result = core.perform_concordance1_calculation(time_data, weights, indices, ntree)
print(f"Concordance index: {result['concordance_index']}")
```

### Cox Regression with Frailty

```python
from survival import regression

result = regression.perform_cox_regression_frailty(
    time=[1.0, 2.0, 3.0, 4.0],
    event=[1, 1, 0, 1],
    covariates=[
        [0.2, 1.0],
        [0.1, 0.5],
        [0.4, 1.2],
        [0.3, 0.7],
    ],
    max_iter=20,
    eps=1e-5,
)
print(result["coefficients"])
```

### Person-Years Calculation

```python
from survival import pybridge

# Low-level API: inputs should match ratetable-style dimensions/cuts.
result = pybridge.perform_pyears_calculation(
    time_data=[1.0, 2.0, 3.0, 1.0, 0.0, 1.0],  # [times..., events...], ny=2
    weights=[1.0, 1.0, 1.0],
    expected_dim=1,
    expected_factors=[0],
    expected_dims=[2],
    expected_cuts=[0.0, 2.0],
    expected_rates=[0.01, 0.02],
    expected_data=[0.5, 1.5, 0.5],
    observed_dim=1,
    observed_factors=[0],
    observed_dims=[2],
    observed_cuts=[0.0, 1.5, 3.0],
    method=0,
    observed_data=[0.5, 1.0, 2.0],
    do_event=1,
    ny=2,
)
print(result.keys())
```

### Kaplan-Meier Survival Curves

```python
from survival import surv_analysis

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # 1 = event, 0 = censored
weights = [1.0] * len(time)  # Optional: equal weights

result = surv_analysis.survfitkm(
    time=time,
    status=status,
    weights=weights,
    entry_times=None,  # Optional: entry times for left-truncation
    position=None,     # Optional: position flags
    reverse=False,     # Optional: reverse time order
    computation_type=0 # Optional: computation type
)

print(f"Time points: {result.time}")
print(f"Survival estimates: {result.estimate}")
print(f"Standard errors: {result.std_err}")
print(f"Number at risk: {result.n_risk}")
```

### Fine-Gray Competing Risks Model

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
status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # 1 = event, 0 = censored
covariates = [
    [1.0, 2.0],
    [1.5, 2.5],
    [2.0, 3.0],
    [2.5, 3.5],
    [3.0, 4.0],
    [3.5, 4.5],
    [4.0, 5.0],
    [4.5, 5.5],
]

# Fit parametric survival model
result = regression.survreg(
    time=time,
    status=status,
    covariates=covariates,
    weights=None,          # Optional: observation weights
    offsets=None,          # Optional: offset values
    initial_beta=None,     # Optional: initial coefficient values
    strata=None,           # Optional: stratification variable
    distribution="weibull",  # "extreme_value", "logistic", "gaussian", "weibull", or "lognormal"
    max_iter=20,          # Optional: maximum iterations
    eps=1e-5,             # Optional: convergence tolerance
    tol_chol=1e-9,        # Optional: Cholesky tolerance
)

print(f"Coefficients: {result.coefficients}")
print(f"Log-likelihood: {result.log_likelihood}")
print(f"Iterations: {result.iterations}")
print(f"Variance matrix: {result.variance_matrix}")
print(f"Convergence flag: {result.convergence_flag}")
```

### Cox Proportional Hazards Model

```python
from survival import regression

# Create a Cox PH model
model = regression.CoxPHModel()

# Or create with data
covariates = [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]
event_times = [1.0, 2.0, 3.0]
censoring = [1, 1, 0]  # 1 = event, 0 = censored

model = regression.CoxPHModel.new_with_data(covariates, event_times, censoring)

# Fit the model
model.fit(n_iters=10)

# Get results
print(f"Baseline hazard: {model.baseline_hazard}")
print(f"Risk scores: {model.risk_scores}")
print(f"Coefficients: {model.get_coefficients()}")

# Predict on new data
new_covariates = [[1.0, 2.0], [2.0, 3.0]]
predictions = model.predict(new_covariates)
print(f"Predictions: {predictions}")

# Calculate Brier score
brier = model.brier_score()
print(f"Brier score: {brier}")

# Compute survival curves for new covariates
new_covariates = [[1.0, 2.0], [2.0, 3.0]]
time_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Optional: specific time points
times, survival_curves = model.survival_curve(new_covariates, time_points)
print(f"Time points: {times}")
print(f"Survival curves: {survival_curves}")  # One curve per covariate set

# Create and add subjects
subject = regression.Subject(
    id=1,
    covariates=[1.0, 2.0],
    is_case=True,
    is_subcohort=True,
    stratum=0
)
model.add_subject(subject)
```

### Cox Martingale Residuals

```python
from survival import residuals

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 1 = event, 0 = censored
score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Risk scores

# Calculate martingale residuals
martingale_residuals = residuals.coxmart(
    time=time,
    status=status,
    score=score,
    weights=None,      # Optional: observation weights
    strata=None,       # Optional: stratification variable
    method=0,          # Optional: method (0 = Breslow, 1 = Efron)
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

# Perform log-rank test (rho=0 for standard log-rank)
result = surv_analysis.compute_logrank_components(
    time=time,
    status=status,
    group=group,
    strata=None,  # Optional: stratification variable
    rho=0.0,      # 0.0 = log-rank, 1.0 = Wilcoxon, other = generalized
)

print(f"Observed events: {result.observed}")
print(f"Expected events: {result.expected}")
print(f"Chi-squared statistic: {result.chi_squared}")
print(f"Degrees of freedom: {result.degrees_of_freedom}")
print(f"Variance matrix: {result.variance}")
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

**Available datasets:**
- `load_lung()` - NCCTG Lung Cancer Data
- `load_aml()` - Acute Myelogenous Leukemia Survival Data
- `load_veteran()` - Veterans' Administration Lung Cancer Study
- `load_ovarian()` - Ovarian Cancer Survival Data
- `load_colon()` - Colon Cancer Data
- `load_pbc()` - Primary Biliary Cholangitis Data
- `load_cgd()` - Chronic Granulomatous Disease Data
- `load_bladder()` - Bladder Cancer Recurrences
- `load_heart()` - Stanford Heart Transplant Data
- `load_kidney()` - Kidney Catheter Data
- `load_rats()` - Rat Treatment Data
- `load_stanford2()` - Stanford Heart Transplant Data (Extended)
- `load_udca()` - UDCA Clinical Trial Data
- `load_myeloid()` - Acute Myeloid Leukemia Clinical Trial
- `load_flchain()` - Free Light Chain Data
- `load_transplant()` - Liver Transplant Data
- `load_mgus()` - Monoclonal Gammopathy Data
- `load_mgus2()` - Monoclonal Gammopathy Data (Updated)
- `load_diabetic()` - Diabetic Retinopathy Data
- `load_retinopathy()` - Retinopathy Data
- `load_gbsg()` - German Breast Cancer Study Group Data
- `load_rotterdam()` - Rotterdam Tumor Bank Data
- `load_logan()` - Logan Unemployment Data
- `load_nwtco()` - National Wilms Tumor Study Data
- `load_solder()` - Solder Joint Data
- `load_tobin()` - Tobin's Tobit Data
- `load_rats2()` - Rat Tumorigenesis Data
- `load_nafld()` - Non-Alcoholic Fatty Liver Disease Data
- `load_cgd0()` - CGD Baseline Data
- `load_pbcseq()` - PBC Sequential Data
- `load_hoel()` - Hoel's Cancer Survival Data
- `load_myeloma()` - Myeloma Survival Data
- `load_rhdnase()` - rhDNase Clinical Trial Data

## API Reference

The public Python surface is broad and evolves quickly. For the most accurate,
version-matched signatures, use the checked-in type stubs:

`import survival` exposes the curated package API via domain modules. Legacy
root-level algorithm symbols remain available for compatibility, but new code
should import from the relevant domain module. For lower-level or experimental
extension symbols, import from `survival._survival` explicitly.

- [`python/survival/__init__.pyi`](python/survival/__init__.pyi): package-level
  typed surface, including the new domain modules.
- [`python/survival/_survival.pyi`](python/survival/_survival.pyi): core
  PyO3 bindings exposed by `survival._survival`.
- [`python/survival/*.py`](python/survival): curated domain modules layered on
  top of the generated bindings.
- [`python/survival/sklearn_compat.py`](python/survival/sklearn_compat.py):
  scikit-learn-compatible estimators and streaming wrappers.

To inspect available symbols at runtime:

```python
import survival

public_names = [name for name in dir(survival) if not name.startswith("_")]
print(public_names)
print(survival.__deprecated_root_export_reason__)
```

Or inspect a specific domain module:

```python
from survival import regression, validation

print(regression.__all__[:10])
print(validation.__all__[:10])
```

## PSpline Options

The `PSpline` class provides penalized spline smoothing:

**Constructor Parameters:**
- `x`: Covariate vector (list of floats)
- `df`: Degrees of freedom (integer)
- `theta`: Roughness penalty (float)
- `eps`: Accuracy for degrees of freedom (float)
- `method`: Penalty method for tuning parameter selection. Supported methods:
  - `"GCV"` - Generalized Cross-Validation
  - `"UBRE"` - Unbiased Risk Estimator
  - `"REML"` - Restricted Maximum Likelihood
  - `"AIC"` - Akaike Information Criterion
  - `"BIC"` - Bayesian Information Criterion
- `boundary_knots`: Tuple of (min, max) for the spline basis
- `intercept`: Whether to include an intercept in the basis
- `penalty`: Whether or not to apply the penalty

**Methods:**
- `fit()`: Fit the spline model, returns coefficients
- `predict(new_x)`: Predict values at new x points

**Properties:**
- `coefficients`: Fitted coefficients (None if not fitted)
- `fitted`: Whether the model has been fitted
- `df`: Degrees of freedom
- `eps`: Convergence tolerance

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full local development
workflow, feature-test matrix, and binding/stub update process.

Install development dependencies:
```sh
uv sync --extra dev --extra test --extra sklearn --no-install-project
```

Build the extension in your current environment:
```sh
maturin develop --release
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
- Package/type stubs in `python/survival/__init__.pyi`,
  `python/survival/_survival.pyi`, and `survival.pyi`
- Runnable examples in `examples/`
- Developer-facing layout notes in `docs/`
- Rust unit/integration tests in `src/tests/`
- Python binding tests in `python/tests/`
- R validation fixtures and archived reference cases in `test/`

## Dependencies

Primary dependencies are defined in [`Cargo.toml`](Cargo.toml) and
[`pyproject.toml`](pyproject.toml), including:

- [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin) for Python bindings
- [numpy](https://numpy.org/) and [ndarray](https://github.com/rust-ndarray/ndarray) for array interop
- [faer](https://github.com/sarah-ek/faer-rs), [rayon](https://github.com/rayon-rs/rayon), and [burn](https://github.com/tracel-ai/burn) for numerical compute

## Compatibility

- This build is for Python only. R/extendr bindings are currently disabled.
- Python 3.11+ and Rust 1.94+ are required.
- macOS users: Ensure you are using the correct Python version and have Homebrew-installed Python if using Apple Silicon.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
