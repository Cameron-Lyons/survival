# Validation Assets

This directory holds the R reference fixtures used to check parity with the
CRAN `survival` package. It is not the home of the Python test suite: Python
tests live in `python/tests/`, and Rust tests live next to the code they cover
(plus `src/tests/`).

## Contents

- `r/` holds the R differential fixture harness: `r/generate_fixtures.R`
  writes `r/fixtures/*.json` from CRAN `survival`, and both test suites read
  them (`python/tests/test_r_fixtures.py`, `src/tests/r_fixtures.rs`). See
  `r/README.md` for the schema, the `KNOWN_FAILURES` burndown lists, and how
  to regenerate. The `r-fixture-stability` CI job regenerates the fixtures with
  the pinned `survival` version and fails on any diff.
- `concordance1.py` is a one-off manual smoke script kept for ad hoc debugging.

The historical `legacy-rust/` reference scripts were never compiled by
`cargo test` and have been removed; see git history (`git log -- test/legacy-rust`)
if you need them.

## Supported Test Flows

For normal development, use the maintained test entry points:

```bash
maturin develop --release
pytest python/tests -v
cargo test
```

If you update the R comparison fixtures, regenerate them with the pinned
`survival` version recorded in the fixture `metadata` so CI stays green:

```bash
Rscript test/r/generate_fixtures.R --check
```
