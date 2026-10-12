# Migrating to survival 3.0

Version 3.0 makes the breaking clean-ups collected in the 3.0 plan: one R API at
the package root and in `survival.r`, the raw native kernels in
`survival.kernels`, R's names and meanings for result components, and a Rust
crate whose public API is its typed functions rather than its Python bindings.
Numerical results are unchanged; only names, result types and module locations
move.

Each section below lists what changed and what to write instead.

## Python

### `survival.r_api` is removed

`survival.r_api` was a second name for `survival.r`. Import `survival.r` (or use
the package root for R's exported functions):

| 2.x | 3.0 |
| --- | --- |
| `from survival import r_api` | `from survival import r` |
| `survival.r_api.coxph(...)` | `survival.r.coxph(...)` |

The private helpers the R bridge used through `r_api` are attributes of
`survival.pybridge`. `survivalr` 0.2.0 requires survival 3.0 and checks the
Python package version when it first loads it.
