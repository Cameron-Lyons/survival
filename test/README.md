# Python Binding Tests

This directory contains tests for the Python bindings of the survival-rust library.

## Test Files

- `test_core.py` - Tests for core functions (coxcount1, coxcount2)
- `test_specialized.py` - Tests for specialized functions (cipoisson, norisk)
- `test_utilities.py` - Tests for utility functions (collapse)
- `test_classes.py` - Tests for Python classes (LinkFunctionParams, PSpline)
- `test_surv_analysis.py` - Tests for survival analysis functions (agsurv4, agsurv5, agmart)
- `test_concordance1.py` - Tests for concordance1 calculation
- `test_concordance_additional.py` - Additional concordance tests
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

Before running tests, you must build the Python module:

```bash
maturin build
```

Then install the wheel or use `maturin develop` for development.

## Note

These tests verify that the Python bindings are working correctly and that the functions can be called with appropriate arguments. They do not verify the statistical correctness of the results (that's what the Rust integration tests in this directory are for).

