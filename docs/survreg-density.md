# AFT density callbacks

The internal `SurvregDensitySource` trait supplies the density part of a
location-scale distribution to both AFT optimizers. This is a foundation for
R's user-defined `survreg` distributions; the public Rust and Python fitting
APIs still accept only the existing built-in families and their identity/log
transforms. Initialization, arbitrary transforms, quantiles, deviance,
variance, callback ownership and serialization need public integration before
that restriction can be removed.

## Evaluation contract

Each likelihood evaluation calls the source once with all standardized lower
endpoints, followed by upper endpoints of interval-censored rows in input
order. The linear predictors include offsets and sparse frailties; scales
can be fixed, estimated or stratified. The source returns one `SurvregDensity`
per endpoint: CDF, survival probability, density, `f'/f`, and `f''/f`.

Shape, finite probabilities and nonnegative finite densities are checked before
accumulation. The derivative ratios must be finite where density is positive;
undefined ratios are allowed where the density underflows to zero. Censored
tail derivatives are then zero. Left-censored rows use their CDF to decide
whether the probability is zero, as the built-in kernel does.

Sources return `SurvivalResult`. Errors propagate through ordinary Newton
steps, Fisher fallbacks, penalized evaluations and line searches. A failed
batch leaves a reused `BlockLikelihood` unchanged. The core interface contains
no Python types; a future Python adapter can attach once per batch.

Built-ins select their scalar kernels once per evaluation and allocate no
endpoint or density buffers. Both paths share contribution and derivative
accumulation formulas. Callback workspace is linear in the number of rows and
interval endpoints.

## Reference checks

`scripts/generate_survreg_density_reference.R` writes a deterministic reference
for an asymmetric two-normal mixture, outside the built-in families. It uses
R survival 3.8-12 and exercises all four censoring types, weights, a formula
offset, and fixed/estimated/stratified scales. Ordinary coefficients,
likelihoods and covariance matrices are checked against `survreg`.

Penalized coefficients and likelihoods are checked against `survreg` with
interval rows made exact, and against `stats::optim` on the full interval
likelihood plus a ridge penalty. This avoids an upstream memory error:
[`survreg7.c`](https://github.com/therneau/survival/blob/master/src/survreg7.c)
allocates `z` for `n` values, whereas
[`survregc2.c`](https://github.com/therneau/survival/blob/master/src/survregc2.c)
also writes every interval's upper endpoint. The Rust buffer includes both.

Finite-difference checks independently verify the score and information.
Additional checks cover sparse endpoint ordering, column-major designs,
underflow tails, malformed outputs, and callback failures at each evaluation
of a penalized fit with line searches.

## Built-in benchmark

`PYTHONPATH=python .venv/bin/python scripts/bench_survreg.py` measures lognormal,
Weibull and loglogistic fits at 1,000, 10,000 and 100,000 rows. It excludes input
conversion and one warmup fit, and records every timing, the extension hash,
coefficients and likelihood so that before/after builds can be compared.
Pass `--extension /path/to/_survival.so` to measure a saved library from the
same Python ABI without reinstalling it. Alternate the build order between
runs to reduce clock and temperature effects.

On 2026-09-28, with Rust 1.94, Python 3.14.7 and an Intel Core Ultra 5 325,
two rounds of 15 fits per build (alternating build order) gave these medians
for 100,000 rows. Both used release builds with `extension-module,ml`.

| Distribution | Before (`532857b0`) | Batched density interface |
| --- | ---: | ---: |
| Lognormal | 46.50 ms | 45.88 ms |
| Weibull | 53.54 ms | 53.73 ms |
| Loglogistic | 42.57 ms | 42.39 ms |

All fitted coefficients and likelihoods were identical at all three input
sizes. Timing differences were small compared with variation between runs;
these measurements do not establish a speedup.
