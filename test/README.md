# Python Binding Tests

This directory contains tests for the Python bindings of the survival-rust library.

## Test Files

- `test_core.py` - Tests for core functions (coxcount1, coxcount2)
- `test_specialized.py` - Tests for specialized functions (cipoisson, norisk, finegray)
- `test_utilities.py` - Tests for utility functions (collapse)
- `test_classes.py` - Tests for Python classes (LinkFunctionParams, PSpline)
- `test_surv_analysis.py` - Tests for survival analysis functions (agsurv4, agsurv5, agmart, survfitkm, survdiff2)
- `test_concordance1.py` - Tests for concordance1 calculation
- `test_concordance_additional.py` - Additional concordance tests
- `test_regression.py` - Tests for regression models (survreg, coxmart, CoxPHModel, Subject)
- `test_all.py` - Runner script to execute all tests

## Running Tests

### Run individual test file:
```bash
python3 test/test_core.py
```

### Run all tests:
```bash
python3 test/test_all.py
```

## Prerequisites

Before running tests, you must:

1. **Install BLAS libraries** (required for the module to run):
   - On Arch Linux: `sudo pacman -S openblas`
   - On Ubuntu/Debian: `sudo apt-get install libopenblas-dev`
   - On Fedora: `sudo dnf install openblas-devel`

2. **Build and install the Python module** - You have two options:

   ### Option 1: Development Mode (Recommended)
   ```bash
   # Create virtualenv (if not already created)
   python3 -m venv .venv
   
   # Install maturin in virtualenv
   .venv/bin/pip install maturin
   
   # Build and install in development mode
   VIRTUAL_ENV=.venv .venv/bin/maturin develop
   ```
   This installs the module in development mode in the virtualenv.

   ### Option 2: Build Wheel
   ```bash
   maturin build
   ```
   Then install the wheel:
   ```bash
   pip install target/wheels/survival-*.whl --force-reinstall
   ```
   Or the tests will automatically extract and use the wheel if pip is not available.

## Running Tests

### Using the test runner script (recommended):
```bash
./test/run_tests.sh
```

### Using Python directly:
```bash
# With virtualenv
.venv/bin/python test/test_all.py

# Or with system Python
python3 test/test_all.py
```

**Note:** The tests will fail with "No module named 'survival'" if the module hasn't been built yet, or with "undefined symbol: cblas_sgemm" if BLAS libraries are missing. Both are expected - just install the prerequisites first.

## Note

These tests verify that the Python bindings are working correctly and that the functions can be called with appropriate arguments. They do not verify the statistical correctness of the results (that's what the Rust integration tests in this directory are for).

