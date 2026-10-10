# Mutable Rust rate-table inputs

Public population APIs now check `RateTable` invariants after callers change its
public fields. Shortened dimension attributes, cutpoints or rates produce an
input error before native indexing. Negative and nonfinite rates are rejected
under the same rules as the constructor.

The checks cover level matching, mapped positions, US calendar alignment,
person-years and expected-survival calculations. Table validation runs once at
each public calculation boundary; internal alignment and subject calculations
reuse the validated table. Ordinary constructor behavior and numerical
integration remain unchanged.

Expected-survival group dimensions use checked arithmetic before allocation,
including `usize::MAX` and overflowing group-by-time products. Public rate lookup
uses checked index arithmetic and returns `None` for an invalid index or overflow.
Formatting a damaged table reports incomplete attributes without indexing past
them. US alignment also checks that mapped positions include its required age
and year columns.

[Public integration tests](../tests/rate_table_inputs.rs) exercise 212 invalid
`Result` calls, 25 damaged-table formatting calls and four optional rate lookups.
Valid controls compare weighted person-years and cohort survival with independent
hazard integrals. Eight stock R `survexp.fit` controls cover US, USR, Minnesota,
cohort and conditional calculations, sparse groups and custom date/year axes.
Empty sparse groups retain stock's zero cohort and unit conditional columns.

All four new integration tests pass with both the no-default and all-feature
builds. The existing 59 population unit tests also pass, along with the complete
Rust configurations and strict Clippy checks.
