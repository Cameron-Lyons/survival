# R survival compatibility

The Python formula interface is `survival.r`; its numerical routines run in
Rust and are also exposed through the Rust crate's domain modules. The checked-in
reference fixtures were generated with R survival **3.8.11**. They cover model
fits, covariance matrices, predictions, residuals, survival curves and utilities.
They establish compatibility for the tested inputs, rather than a claim that
every R expression, plotting method or extension package is supported.

## Formula and population interfaces

Penalized Cox formulas support `ridge`, `pspline`, `frailty`, `frailty.gamma`,
`frailty.gaussian` and `frailty.t`. Basis construction and penalty optimization
run in Rust. Prediction reuses training knots and factor levels. Formula options
are parsed as literals; Python or R code is never evaluated.

```python
from survival import datasets, r

lung = datasets.load_lung()
fit = r.coxph("Surv(time, status) ~ pspline(age, df=4) + sex", lung)
prediction = r.predict(fit, {"age": [50, 70], "sex": [1, 2]}, type="lp")
effective_df = r.degrees_freedom(fit)

cox = r.coxph("Surv(time, status) ~ age + sex", lung, model=True)
expected = r.survexp("~ sex", lung, ratetable=cox, times=[0, 100, 365])
```

Multistate `summary_survfit` returns probabilities, state-specific counts,
confidence limits and a table of restricted mean time in each state. Requested
times are sorted and deduplicated; events and censors accumulate between them.
`scale`, `extend`, `censored` and the common, individual, numeric and omitted
restricted-mean options are supported. Rust callers can use
`surv_analysis::{summary_survfit_aj, survmean_aj}`.

`blogit`, `bprobit`, `bcloglog` and `blog` accept `inverse=True` for the inverse
link, with scalar or vector input.

`yates(..., predict="risk", nsim=200, options={"seed": 123})` uses Rust simulation
with R's default Mersenne-Twister/inversion normal stream. An explicit seed gives
reproducible covariance estimates without altering a global RNG. The default
seed is zero. External linear models can supply their coefficients and covariance
through `r.YatesModel(formula, data, coefficients, variance, sigma2=None)`;
coefficients must follow the formula's columns, including its intercept. R's
`lm` is part of the separate `stats` package; the adapter does not refit it.

## Reference differences retained deliberately

After the completion pass, the Python fixture suite has **37 expected numerical
differences**, with no remaining missing-feature or API-error exemptions. None
of the reference JSON files were changed. The differences are:

| Checks | Fixture family | Reason |
| ---: | --- | --- |
| 7 | Aalen regression, veteran data | The reference continues through rank-deficient risk sets and contains missing coefficients and values around 10¹⁵. The port truncates at the last nonsingular risk set and returns finite estimates. |
| 6 | Cox concordance | Exact linear-predictor ties depend on floating-point/BLAS rounding. The reference and this platform classify a few tied pairs differently. |
| 1 | Exact counting-process Cox deviance residuals | The reference's unclassed fit dispatch returns martingale residuals for a deviance request. The port computes deviance residuals. |
| 18 | Interval-censored AFT residual derivatives | The reference has opposite signs in some scale derivatives. The port's likelihood derivatives are checked against finite differences for every censoring type and distribution family. |
| 4 | AFT prediction with a new-data offset | The fixture omits the new-data offset. The port includes it in the linear predictor and transformed predictions. |
| 1 | `survcondense` with no rows to merge | The reference produces zero rows through negative indexing of an empty index. The port preserves all input rows. |

These cases remain explicit `xfail(strict=True)` entries, so a future change in
behavior requires review. They are not counted as passing comparisons. The Rust
fixture adapter also records cases it cannot construct; those harness limitations
are separate from the Python tests of the same Rust engines.

Two fixture transport details are handled by the adapters: CSV factor columns
recover the recorded R levels, and the near-tie test recovers the generator's
IEEE floating-point expressions before fitting. Output time comparisons allow
only the JSON format's 15-significant-digit rounding. Counts and the number of
distinct times remain checked separately.

## Reproduce validation

```sh
cargo test --lib --no-default-features --offline
PYO3_PYTHON=$PWD/.venv/bin/python PATH=$PWD/.venv/bin:$PATH \
  cargo test --lib --all-features --offline
PYTHONPATH=python .venv/bin/python -m pytest python/tests -q
python3 scripts/generate_binding_manifest.py --check
PYTHONPATH=python .venv/bin/python scripts/generate_stubs.py --check
```

The normal fixture run includes the documented expected differences. Add
`--runxfail` to display their full discrepancies. R itself is needed to regenerate
the reference data; see [the fixture workflow](../test/r/README.md).

For curve complexity, benchmark inputs and measured timings, see
[Kaplan–Meier performance](kaplan-meier-performance.md).
