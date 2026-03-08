# Legacy Rust Reference Cases

These `.rs` files are historical reference cases that are not part of the automated
Cargo test suite.

They are kept for comparison and migration work, but the supported Rust test entry
points are the crate tests exercised by `cargo test`.

## Audit Snapshot

- `41` legacy files remain parked here after migrating `aareg`, `anova`, `bladder`, and
  `finegray` into the maintained test suite.
- `32` use `fn main()` script-style entry points instead of `#[test]`.
- `7` still carry `#[test]`, and `7` still depend on the retired
  `survival::prelude::*`-style DSL.
- `6` pull in `polars`-specific data-frame helpers and `3` depend on `linfa` APIs.
- `concordance3.rs` still contains `unimplemented!()` placeholders, and files such as
  `expected2.rs` and `nested.rs` are partial prototypes rather than runnable tests.

## Migrated Into Current Tests

- `aareg.rs` -> covered by `src/regression/aareg.rs` unit tests.
- `anova.rs` -> covered by `src/validation/anova.rs` unit tests.
- `bladder.rs` -> covered by `src/regression/recurrent_events.rs` unit tests.
- `finegray.rs` -> covered by `src/specialized/finegray.rs` unit tests.

## Explicitly Archived File Groups

- Formula/prelude examples: `book1.rs`, `book2.rs`, `book3.rs`, `book4.rs`, `book5.rs`,
  `book6.rs`, `book7.rs`, `doublecolon.rs`, `ovarian.rs`.
- Survival workflow scripts: `checksurv2.rs`, `concordance.rs`, `concordance2.rs`,
  `coxsurv.rs`, `coxsurv2.rs`, `coxsurv3.rs`, `coxsurv4.rs`, `coxsurv5.rs`, `coxsurv6.rs`,
  `detail.rs`, `difftest.rs`, `doaml.rs`, `doweight.rs`, `dropspecial.rs`, `ekm.rs`,
  `expected.rs`, `factor.rs`, `neardate.rs`, `nsk.rs`.
- External dependency demos: `brier.rs`, `cancer.rs`, `clogit.rs`, `factor2.rs`,
  `mstate.rs`, `mstate2.rs`, `mstrata.rs`, `multi2.rs`, `multi3.rs`, `multistate.rs`.
- Incomplete prototypes: `concordance3.rs`, `expected2.rs`, `nested.rs`.

## Low-Value Direct Ports

- `concordance3.rs`, `expected2.rs`, and `nested.rs` are better treated as design
  notes than migration targets because they rely on missing loaders, placeholders, or
  incomplete abstractions.
