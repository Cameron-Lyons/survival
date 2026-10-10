# Near ties at large finite times

The shared Rust `aeq_surv` calculation now keeps the mean absolute time finite
when summing finite times would overflow. This affects every fitter that enables
`timefix`, including Kaplan–Meier, Cox and log-rank calculations. Its ordinary
finite-sum arithmetic is unchanged; only an overflowing sum uses a scaled mean.

For example, the distinct times `[1e308, 1.1e308, 1.2e308]` previously shared an
infinite mean denominator and were all treated as ties. With events `[1, 0, 1]`,
that changed the Kaplan–Meier curve from `[2/3, 2/3, 0]` to `[1/3]`. The corrected
curve matches stock R survival. Counting-process responses also keep their
distinct endpoints instead of reporting a spurious zero-length interval.
Actual near ties still collapse to the earliest time. The existing correction
that retains infinite and missing endpoints in their original rows also applies
when the finite times are very large.

`scripts/generate_aeq_large_time_reference.R` independently calls stock
`aeqSurv` and `survfit` for 26 cases. They cover positive and negative magnitudes,
the largest finite double, actual near ties, repeated observations, subnormals,
ordinary tolerance boundaries, explicit tolerances, delayed entry and an actual
interval that becomes zero after normalization. Their results are identical in
survival 3.8-11 and 3.8-12; the checked-in reference records 3.8-12. JSON uses
17 significant digits to keep the largest finite double finite after loading,
and separately preserves infinite standard errors and missing values.

Rust tests compare normalized endpoints exactly and independently check the ten
recorded curve fields. The Python suite checks normalization through the native
and R-style interfaces, complete native, formula and direct-response curves,
and preservation of nonfinite endpoints. Counts and times use exact comparisons;
other curve fields use `rtol=1e-12`, `atol=1e-14`, with matching missing values.

```sh
Rscript scripts/generate_aeq_large_time_reference.R
cargo test --lib --no-default-features data_prep::aeq_surv
PYTHONPATH=python .venv/bin/python -m pytest python/tests/test_aeq_large_times.py -q
```
