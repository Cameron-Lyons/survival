# Migrating to survival 3.0

Version 3.0 makes the breaking clean-ups collected in the 3.0 plan: one R API at
the package root and in `survival.r`, the raw native kernels in
`survival.kernels`, R's names and meanings for result components, and a Rust
crate whose public API is its typed functions rather than its Python bindings.
Numerical results are unchanged; only names, result types and module locations
move.

Each section below lists what changed and what to write instead.

## Python
