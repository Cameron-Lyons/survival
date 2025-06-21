# survival

A high-performance survival analysis library written in Rust, with a Python API powered by [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

This package provides:
- Core survival analysis routines
- Definition of Surv objects
- Kaplan-Meier and Aalen-Johansen (multi-state) curves
- Cox models
- Parametric accelerated failure time models
- Penalized splines (P-splines) for smooth covariate effects

## Installation

### Prerequisites
- Python 3.12 or 3.13 (recommended: 3.12)
- Rust toolchain (see [rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin) (install via Homebrew: `brew install maturin`)

### Build and Install (Python)
1. Build the Python wheel:
   ```sh
   maturin build
   ```
2. Install the wheel (replace the version as needed):
   ```sh
   pip3 install target/wheels/survival-0.1.0-*.whl --force-reinstall --break-system-packages
   ```

Alternatively, for development:
```sh
maturin develop --skip-install
```

## Usage Example (Python)
```python
from survival import AaregOptions, aareg

# Example data (replace with your own)
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

# Example: Penalized spline (P-spline) usage
from survival import PSpline

x = [0.1 * i for i in range(100)]
pspline = PSpline(
    x=x,
    df=10,
    theta=1.0,
    eps=1e-6,
    method="GCV",  # or "UBRE"
    boundary_knots=(0.0, 10.0),
    intercept=True,
    penalty=True,
)
pspline.fit()
```

## PSpline: Penalized Splines
The `PSpline` class provides penalized spline (P-spline) smoothing for modeling smooth covariate effects. Key options:
- `x`: Covariate vector (list of floats)
- `df`: Degrees of freedom (integer)
- `theta`: Roughness penalty (float)
- `eps`: Accuracy for degrees of freedom (float)
- `method`: Penalty method for tuning parameter selection. Supported: `"GCV"`, `"UBRE"`. Any other value will result in an error.
- `boundary_knots`: Tuple of (min, max) for the spline basis
- `intercept`: Whether to include an intercept in the basis
- `penalty`: Whether to apply the penalty

**Note:** Only "GCV" and "UBRE" are currently supported for the penalty method. Using any other value will result in a runtime error.

## Development
- To build the Rust library for Python, use `maturin build` or `maturin develop`.
- To run tests (if available), use `cargo test`.
- The codebase is organized with core routines in `src/` and tests/examples in `test/`.

## Dependencies
- [PyO3](https://github.com/PyO3/pyo3)
- [ndarray](https://github.com/rust-ndarray/ndarray)
- [numpy](https://github.com/PyO3/rust-numpy)
- [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg)
- [itertools](https://github.com/rust-itertools/itertools)

## Compatibility Notes
- This build is for Python only. R/extendr bindings are currently disabled.
- macOS users: Ensure you are using the correct Python version and have Homebrew-installed Python if using Apple Silicon.

## License
See [LICENSE](LICENSE).
