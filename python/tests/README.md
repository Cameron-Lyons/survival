# Python Binding Tests

This directory contains the supported Python binding test suite.

## Recommended Workflow

Build the extension into your active environment, then run pytest:

```bash
maturin develop --release
pytest python/tests -v
```

If you prefer wheel-based verification, build and install a wheel first, then run the
same pytest command.

## Coverage

- Core PyO3 bindings and result objects
- Regression and survival-analysis entry points
- Array/dataframe interoperability
- Deep survival and scikit-learn compatibility wrappers
