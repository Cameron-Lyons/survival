# Validation Assets

This directory is no longer the supported home for the Python binding test suite.
Python tests now live in `python/tests/`.

## Contents

- `generate_r_expected_values.R` and `r_expected_values.json` back the R validation job in CI.
- `legacy-rust/` contains historical Rust reference cases that are not wired into `cargo test`.
- `concordance1.py` is a one-off manual smoke script kept for ad hoc debugging.

## Supported Test Flows

For normal development, use the maintained test entry points:

```bash
maturin develop --release
pytest python/tests -v
cargo test
```

If you update the R comparison fixtures, regenerate them with:

```bash
Rscript test/generate_r_expected_values.R
```
