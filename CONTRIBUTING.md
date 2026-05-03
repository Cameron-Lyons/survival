# Contributing

This repository has two public surfaces that must stay in sync:

- the Rust crate in `src/`
- the Python package in `python/survival/`, backed by PyO3 bindings

Prefer small, domain-scoped changes. Many modules mirror the upstream R
`survival` package, so behavioral refactors should include focused tests and
avoid changing established numerical conventions unless the change is explicit.

## Local Setup

Use `uv` for the Python environment:

```sh
uv sync --extra dev --extra test --extra sklearn --no-install-project
```

Build the Python extension into that environment when Python tests need the
native module:

```sh
uv run --with maturin maturin develop --features python,ml
```

For release-like wheel checks, use:

```sh
uv run --with maturin maturin build --release --features extension-module,ml
```

## Feature Golden Path

Run the narrowest command that covers the code you changed, then run the
relevant broader checks before handing off.

Rust-only change:

```sh
cargo fmt --check
cargo test --lib --no-default-features
```

ML or optional-model change:

```sh
cargo test --lib --features ml
```

PyO3 binding or Python-facing Rust change:

```sh
cargo test --lib --features python,ml
uv run --with maturin maturin develop --features python,ml
PYTHONPATH=.:python uv run --no-sync pytest python/tests -q
```

Before a PR or broad refactor:

```sh
cargo fmt --check
cargo clippy --all-features --all-targets -- -D warnings
cargo test --lib --all-features
uv run --no-sync ruff format python/ test/ --check
uv run --no-sync ruff check python/ test/
uv run --no-sync mypy python/survival/__init__.pyi python/survival/_survival.pyi --ignore-missing-imports
PYTHONPATH=.:python uv run --no-sync pytest python/tests -q
python3 scripts/generate_binding_manifest.py --check
```

The important feature combinations are:

- `--no-default-features`: Rust crate without Python enabled.
- `--features ml`: optional ML code without PyO3.
- `--features python,ml`: local Python extension development.
- `--features extension-module,ml`: wheel/extension-module builds.
- `--all-features`: broad safety check. This intentionally enables
  `extension-module`, so `build.rs` must keep libpython linkage working for
  local tests.

There is no CUDA backend feature. `ComputeBackend::CUDA` is retained as an API
enum value, but unavailable GPU backends should report unavailable rather than
pretending to execute.

## Binding And Stub Workflow

When adding, removing, or renaming a PyO3 binding:

1. Register the function or class under `src/api/python/`.
2. Regenerate the binding manifest:

   ```sh
   python3 scripts/generate_binding_manifest.py
   ```

3. Update `python/survival/_survival.pyi` for the low-level binding signature.
4. Add the symbol to the appropriate domain wrapper in `python/survival/*.py`
   if it should be available through a curated module.
5. Update `python/survival/__init__.pyi` only when the package-level typed
   surface changes.
6. Run:

   ```sh
   python3 scripts/generate_binding_manifest.py --check
   PYTHONPATH=.:python uv run --no-sync pytest python/tests/test_binding_contract.py -q
   ```

The manifest is generated. The stubs are checked in and maintained by hand.
Treat stub edits like API changes: keep names, optional arguments, and return
types consistent with the Rust `#[pyfunction]`, `#[pymethods]`, and wrapper
module surface.

## Python Module Layout

New Python API should live in a domain module rather than adding another
flattened root export. `python/survival/__init__.py` keeps compatibility exports
for older callers, but the preferred surface is domain-oriented:

- `core.py`: low-level shared routines and typed input containers
- `regression.py`: Cox, AFT, competing risks, cure, and recurrent models
- `surv_analysis.py`: Kaplan-Meier, Nelson-Aalen, multistate, and tests
- `validation.py`: metrics, calibration, conformal, RMST, and model checks
- `ml.py`: optional ML models behind the Rust `ml` feature
- `sklearn_compat.py`: compatibility shim; estimator implementations live in
  private `_sklearn_*` modules

If a module needs many shared helpers, prefer a small private module with a
leading underscore over duplicating validation or scoring logic.

## Tests To Add

Use the smallest test that proves the contract:

- Rust numerical logic: module-local Rust unit tests near the implementation.
- PyO3 binding shape: `python/tests/test_binding_contract.py`.
- Python wrapper behavior: a focused file under `python/tests/`.
- R parity or survival-package behavior: existing R validation tests under
  `src/tests/` or fixtures under `test/`.

For Python-facing changes, rebuild the extension before running Python tests.
An import failure that reports a missing symbol often means the local `.so` is
stale, not that the Python wrapper is wrong.

## Style Notes

- Keep generated files generated and hand-maintained files hand-maintained.
- Prefer typed input structs in `src/internal/typed_inputs.rs` over repeating
  raw `Vec`/length validation in public internals.
- Document every `unsafe` block with a concrete `SAFETY:` invariant.
- Do not advertise feature flags or devices that are placeholders only.
- Preserve compatibility exports, but mark narrower curated surfaces as the
  preferred API for new code.
