# survival-rs

[![PyPI version](https://badge.fury.io/py/survival-rs.svg)](https://badge.fury.io/py/survival-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance survival analysis library written in Rust, with a Python API powered by [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

## Features

- Core survival analysis routines
- Cox proportional hazards models
- Kaplan-Meier and Aalen-Johansen (multi-state) survival curves
- Parametric accelerated failure time models
- Penalized splines (P-splines) for smooth covariate effects
- Concordance index calculations
- Person-years calculations
- Score calculations for survival models
- Residual analysis

## Installation

### From PyPI (Recommended)

```sh
pip install survival-rs
```

### From Source

#### Prerequisites

- Python 3.12 or 3.13 (recommended: 3.12)
- Rust toolchain (see [rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin)
- BLAS libraries (required at runtime):
  - Arch Linux: `sudo pacman -S openblas`
  - Ubuntu/Debian: `sudo apt-get install libopenblas-dev`
  - Fedora: `sudo dnf install openblas-devel`
  - macOS: `brew install openblas`

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
pip install target/wheels/survival_rs-0.1.0-*.whl
```

For development:
```sh
maturin develop
```

## Usage

### Aalen's Additive Regression Model

```python
from survival import AaregOptions, aareg

data = [
    [1.0, 0.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
]
variable_names = ["time", "event", "covariate1"]
options = AaregOptions(
    formula="time + event ~ covariate1",
    data=data,
    variable_names=variable_names,
    weights=None,
    subset=None,
    na_action=None,
    qrtol=1e-8,
    nmin=None,
    dfbeta=False,
    taper=0.0,
    test=[],
    cluster=None,
    model=False,
    x=False,
    y=False,
)
result = aareg(options)
print(result)
```

### Penalized Splines (P-splines)

```python
from survival import PSpline

x = [0.1 * i for i in range(100)]
pspline = PSpline(
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
from survival import perform_concordance1_calculation

time_data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
weights = [1.0, 1.0, 1.0, 1.0, 1.0]
indices = [0, 1, 2, 3, 4]
ntree = 5

result = perform_concordance1_calculation(time_data, weights, indices, ntree)
print(f"Concordance index: {result['concordance_index']}")
```

### Cox Regression with Frailty

```python
from survival import perform_cox_regression_frailty

result = perform_cox_regression_frailty(
    time_data=[...],
    status_data=[...],
    covariates=[...],
    # ... other parameters
)
```

### Person-Years Calculation

```python
from survival import perform_pyears_calculation

result = perform_pyears_calculation(
    time_data=[...],
    weights=[...],
    # ... other parameters
)
```

### Kaplan-Meier Survival Curves

```python
from survival import survfitkm, SurvFitKMOutput

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]  # 1 = event, 0 = censored
weights = [1.0] * len(time)  # Optional: equal weights

result = survfitkm(
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
from survival import finegray, FineGrayOutput

# Example competing risks data
tstart = [0.0, 0.0, 0.0, 0.0]
tstop = [1.0, 2.0, 3.0, 4.0]
ctime = [0.5, 1.5, 2.5, 3.5]  # Cut points
cprob = [0.1, 0.2, 0.3, 0.4]  # Cumulative probabilities
extend = [True, True, False, False]  # Whether to extend intervals
keep = [True, True, True, True]      # Which cut points to keep

result = finegray(
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
from survival import survreg, SurvivalFit, DistributionType

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
result = survreg(
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
from survival import CoxPHModel, Subject

# Create a Cox PH model
model = CoxPHModel()

# Or create with data
covariates = [[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]
event_times = [1.0, 2.0, 3.0]
censoring = [1, 1, 0]  # 1 = event, 0 = censored

model = CoxPHModel.new_with_data(covariates, event_times, censoring)

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
subject = Subject(
    id=1,
    covariates=[1.0, 2.0],
    is_case=True,
    is_subcohort=True,
    stratum=0
)
model.add_subject(&subject)
```

### Cox Martingale Residuals

```python
from survival import coxmart

# Example survival data
time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
status = [1, 1, 0, 1, 0, 1, 1, 0]  # 1 = event, 0 = censored
score = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # Risk scores

# Calculate martingale residuals
residuals = coxmart(
    time=time,
    status=status,
    score=score,
    weights=None,      # Optional: observation weights
    strata=None,       # Optional: stratification variable
    method=0,          # Optional: method (0 = Breslow, 1 = Efron)
)

print(f"Martingale residuals: {residuals}")
```

### Survival Difference Tests (Log-Rank Test)

```python
from survival import survdiff2, SurvDiffResult

# Example: Compare survival between two groups
time = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5]
status = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
group = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # Group 1 and Group 2

# Perform log-rank test (rho=0 for standard log-rank)
result = survdiff2(
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

## API Reference

### Classes

- `AaregOptions`: Configuration options for Aalen's additive regression model
- `PSpline`: Penalized spline class for smooth covariate effects
- `CoxPHModel`: Cox proportional hazards model class
- `Subject`: Subject data structure for Cox PH models
- `SurvFitKMOutput`: Output from Kaplan-Meier survival curve fitting
- `FineGrayOutput`: Output from Fine-Gray competing risks model
- `SurvivalFit`: Output from parametric survival regression
- `DistributionType`: Distribution types for parametric models (extreme_value, logistic, gaussian, weibull, lognormal)
- `SurvDiffResult`: Output from survival difference tests (log-rank test)

### Functions

- `aareg(options)`: Fit Aalen's additive regression model
- `survfitkm(...)`: Fit Kaplan-Meier survival curves
- `survreg(...)`: Fit parametric accelerated failure time models
- `survdiff2(...)`: Perform survival difference tests (log-rank, Wilcoxon, etc.)
- `coxmart(...)`: Calculate Cox martingale residuals
- `finegray(...)`: Fine-Gray competing risks model data preparation
- `perform_concordance1_calculation(...)`: Calculate concordance index (version 1)
- `perform_concordance3_calculation(...)`: Calculate concordance index (version 3)
- `perform_concordance_calculation(...)`: Calculate concordance index (version 5)
- `perform_cox_regression_frailty(...)`: Fit Cox proportional hazards model with frailty
- `perform_pyears_calculation(...)`: Calculate person-years of observation
- `perform_pystep_calculation(...)`: Perform step calculations
- `perform_pystep_simple_calculation(...)`: Perform simple step calculations
- `perform_score_calculation(...)`: Calculate score statistics
- `perform_agscore3_calculation(...)`: Calculate score statistics (version 3)

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
- `penalty`: Whether to apply the penalty

**Methods:**
- `fit()`: Fit the spline model, returns coefficients
- `predict(new_x)`: Predict values at new x points

**Properties:**
- `coefficients`: Fitted coefficients (None if not fitted)
- `fitted`: Whether the model has been fitted
- `df`: Degrees of freedom
- `eps`: Convergence tolerance

## Development

Build the Rust library:
```sh
cargo build
```

Run tests:
```sh
cargo test
```

Format code:
```sh
cargo fmt
```

The codebase is organized with:
- Core routines in `src/`
- Tests and examples in `test/`
- Python bindings using PyO3

## Dependencies

- [PyO3](https://github.com/PyO3/pyo3) - Python bindings
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays
- [numpy](https://github.com/PyO3/rust-numpy) - NumPy integration
- [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) - Linear algebra
- [itertools](https://github.com/rust-itertools/itertools) - Iterator utilities
- [ndarray-stats](https://github.com/rust-ndarray/ndarray-stats) - Statistical functions
- [statrs](https://github.com/statrs-dev/statrs) - Statistical distributions
- [thiserror](https://github.com/dtolnay/thiserror) - Error handling

## Compatibility

- This build is for Python only. R/extendr bindings are currently disabled.
- macOS users: Ensure you are using the correct Python version and have Homebrew-installed Python if using Apple Silicon.

## License

See [LICENSE](LICENSE).
