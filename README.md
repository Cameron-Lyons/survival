# survival

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

### Prerequisites

- Python 3.12 or 3.13 (recommended: 3.12)
- Rust toolchain (see [rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin)

Install maturin:
```sh
pip install maturin
```

Or via Homebrew:
```sh
brew install maturin
```

### Build and Install

Build the Python wheel:
```sh
maturin build
```

Install the wheel:
```sh
pip3 install target/wheels/survival-0.1.0-*.whl --force-reinstall --break-system-packages
```

For development:
```sh
maturin develop --skip-install
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

## API Reference

### Classes

- `AaregOptions`: Configuration options for Aalen's additive regression model
- `PSpline`: Penalized spline class for smooth covariate effects

### Functions

- `aareg(options)`: Fit Aalen's additive regression model
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

- `x`: Covariate vector (list of floats)
- `df`: Degrees of freedom (integer)
- `theta`: Roughness penalty (float)
- `eps`: Accuracy for degrees of freedom (float)
- `method`: Penalty method for tuning parameter selection. Supported: `"GCV"`, `"UBRE"`. Any other value will result in an error.
- `boundary_knots`: Tuple of (min, max) for the spline basis
- `intercept`: Whether to include an intercept in the basis
- `penalty`: Whether to apply the penalty

**Note:** Only "GCV" and "UBRE" are currently supported for the penalty method.

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
