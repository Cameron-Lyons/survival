# AFT distribution callbacks

Rust and Python accept user-defined location-scale families in ordinary and
penalized `survreg` fits. Fits retain the callbacks for prediction, residuals,
and robust variance. Built-in distributions keep their scalar kernels.

## Python

Pass a distribution dictionary to `survival.r.survreg(dist=...)`, or register
it in `survival.r.survreg_distributions` and pass its name. A family supplies
`name`, `init`, `density`, `deviance`, and `quantile`; penalized fits also need
`variance`. For example, this defines the logistic family with a log response
transform using only NumPy:

```python
import numpy as np
from survival import datasets, r


def init(y, weights):
    mean = np.average(y, weights=weights)
    return [mean, np.average((y - mean)**2, weights=weights)]


def density(z):
    e = np.exp(-np.abs(z))
    lower = np.where(z > 0, 1 / (1 + e), e / (1 + e))
    upper = np.where(z > 0, e / (1 + e), 1 / (1 + e))
    pdf = lower * upper
    return np.column_stack([lower, upper, pdf, 1 - 2*lower, 1 - 6*pdf])


def quantile(p):
    return np.log(p) - np.log1p(-p)


def deviance(y, scale):
    status = y[:, -1]
    center = y[:, 0].copy()
    loglik = np.zeros(len(y))
    exact = status == 1
    loglik[exact] = -np.log(4 * scale[exact])
    interval = status == 3
    if np.any(interval):
        center[interval] = (y[interval, 0] + y[interval, 1]) / 2
        width = (y[interval, 1] - y[interval, 0]) / scale[interval]
        loglik[interval] = np.log(np.tanh(width / 4))
    return {"center": center, "loglik": loglik}


def variance():
    return np.pi**2 / 3


custom = dict(name="Custom log logistic", init=init, density=density,
              quantile=quantile, deviance=deviance, variance=variance,
              trans="log")
assert r.survregDtest(custom)
fit = r.survreg("Surv(time, status) ~ age + sex", datasets.load_lung(), dist=custom)
median = r.predict(fit, type="quantile", p=0.5)
```

Callbacks receive NumPy arrays:

| Callback | Arguments | Result |
| --- | --- | --- |
| `init` | Transformed response, case weights | Location and variance (two values) |
| `density` | Standardized endpoints | Matrix with columns `F`, `1-F`, `f`, `f'/f`, `f''/f` |
| `quantile` | Probabilities | Standardized quantiles |
| `deviance` | Response matrix, one scale per row | `center` and saturated `loglik`, as a dictionary or pair of vectors |
| `variance` | None | Finite positive standardized variance |

The deviance response matrix has `[y, status]` columns, or
`[lower, upper, status]` when any row is interval censored. Status is 0 for
right censoring, 1 for an exact observation, 2 for left censoring, and 3 for
interval censoring. Only intervals use the upper endpoint. Centers and
likelihoods are on the transformed response scale.

A nonempty `parms` entry adds a final argument to all family callbacks. A
parameter dictionary is passed as a dictionary and supports partial named
overrides through `survreg(parms=...)`; an unnamed vector is passed as a
NumPy array and must be replaced in full. `scale` in the definition fixes the
model scale, as it does for the exponential and Rayleigh families.

`trans="identity"` (the default) and `trans="log"` select built-in transforms.
An arbitrary increasing transform supplies callable `trans`, `dtrans`, and
`itrans`. A dictionary with `dist="gaussian"` (or another distribution name,
object, or dictionary) reuses that base family with the supplied transform:

```python
asinh_normal = dict(name="Asinh normal", dist="gaussian", trans=np.arcsinh,
                    dtrans=lambda y: 1 / np.sqrt(1 + y*y), itrans=np.sinh)
```

`dsurvreg`, `psurvreg`, `qsurvreg`, and `rsurvreg` accept these dictionaries,
resolved distribution objects, and registered names. Names use case-folded
exact lookup for these functions; fitting also accepts unique prefixes.
`rsurvreg(seed=...)` retains R's uniform stream for custom families.

`rsurvreg(n, ...)` follows R's `runif` count rules. A singleton supplies a
finite nonnegative count truncated toward zero; every other vector length
supplies the number of draws, without converting its values. NumPy arrays use
their total element count and return a flat list. One-shot iterators are read
once. Random generation draws exactly that many uniforms before evaluating
the quantile and recycling the location and scale vectors. The result can
therefore contain more values than the draw count; an empty vector can also
make the result empty. A custom quantile receives empty probabilities when
the draw count is zero.
The independent `rsurvreg_count_reference.json` records forty stock-R count
cases, callback batches, coercion warnings and subsequent recycling.

For direct bindings, use `SurvregDistribution.from_callbacks(...)`, then
`with_transform(...)`, `with_parms(...)`, or `derived(...)` as needed.
`family` and `transform` report `Custom` for runtime implementations.
`pdf_values`, `cdf_values`, `quantile_values`, and `sample` recycle numeric
means and scales at each arithmetic operation, as the R-style functions do.
The R-style functions also accept scalar arguments. For example,
`r.qsurvreg([0.25, 0.75], mean=[0, 1, 2], scale=1, distribution="gaussian")`
returns three values and warns about fractional recycling. Empty, missing,
nonfinite and nonpositive values follow the query arithmetic; fitting retains
its stricter distribution and scale checks.

Named Student-t queries require explicit `parms`, including for empty queries.
Numeric parameters recycle inside `pt`, `dt` and `qt`; other named built-ins
ignore `parms`. `SurvregDistribution("t")` retains its default four degrees of
freedom for fitting and direct object queries. Use `for_query("t", parms)` to
construct a permissive query object; fitting revalidates it before use.

Query callbacks receive the complete batches produced at each stage, including
missing values and empty batches. Density queries evaluate the transform
derivative before the transform. Quantile queries evaluate the quantile before
scale multiplication, location addition and the inverse transform. Python
arithmetic warnings use `RuntimeWarning`; callback warnings retain their
category. The R bridge forwards `RuntimeWarning` and `UserWarning` when they
occur, so warning-as-error settings stop evaluation before later callbacks.
Exceptions raised by Python query callbacks retain their original type and
object.

`survregDtest` checks density output at 0.1 through 1 and checks transform
inversion and positive finite derivatives at 1 through 10. Fitting checks
results again on actual observations. Callback exceptions include the
callback name and original exception text; Python exceptions raised during
native fitting computation surface as `RuntimeError`. Invalid fitting shapes
and numerical values detected by Rust surface as `ValueError`. Standalone
queries allow different callback result lengths and unused density columns,
matching their R evaluation contract.

Python pickle, shallow copy, and deep copy retain custom distributions in
ordinary and penalized fitted models. Their callables must themselves support
pickle (for example, importable module-level functions). Lambdas and local
functions retain Python's normal pickle restrictions. Cloned native objects
share their callback handles; callbacks should not change distribution
semantics after fitting.

## Rust

Implement `regression::SurvregCallbacks` and optionally
`regression::SurvregTransformCallbacks`, then construct
`SurvregDistribution::from_callbacks` with an `Arc` and attach the transform
with `with_transform`. Both traits use owned callback state and
`SurvivalResult`, and require `Send + Sync`. The numerical core contains no
Python objects. See
[`survreg_callbacks/tests.rs`](../src/regression/survreg_callbacks/tests.rs)
for a complete native mixture implementation used with both public fitters.
The [external-crate example](../tests/survreg_callbacks.rs) imports the
callback traits and `SurvregDensity` through the public `regression` module.

Scalar distribution operations (`density`, `quantile`, `deviance`, `variance`,
`pdf`, `cdf`, and `quantile_at`) now return `SurvivalResult` so custom errors
propagate. Transform enum operations also return `SurvivalResult`; evaluate
runtime transforms through their distribution's batch methods. Built-in
scalar operations do not allocate callback buffers. `validate` checks
metadata and ownership without invoking callbacks; `dtest` additionally
performs the functional probes described above.

Rust's `pdf_values`, `cdf_values`, `quantile_values` and `sample` return the
query values while ignoring arithmetic warnings. Their `_with_warnings`
variants accept a fallible callback receiving `regression::DpqrWarning` at
each warning stage. Returning an error stops evaluation at that stage. The
built-in fast path composes recycling indices and allocates the result once;
callback queries reuse owned arithmetic buffers where their lengths permit it.

Generic serde serialization explicitly rejects runtime callbacks. Built-in
objects still serialize; native applications using custom families must
provide their own callback reconstruction. Python pickle serializes metadata
and Python callables separately. Pickle data is tied to the package's binary
state format, as for the other fitted models.

The R facade's standard serialization saves the original R functions and their
environments alongside that Python state. This includes local custom
distribution callbacks. See [R model persistence](r-model-persistence.md).

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
no Python types. The Python adapter reacquires the interpreter once per
batch; the optimizers run detached between callback invocations.

Built-ins select their scalar kernels once per evaluation and allocate no
endpoint or density buffers. Both paths share contribution and derivative
accumulation formulas. Callback workspace is linear in the number of rows and
interval endpoints.

## Reference checks

`scripts/generate_survreg_dpqr_reference.R` writes four independent stock-R
query fixtures covering 5,712 built-in cases and callback batches, failures,
empty inputs and warning order. `scripts/generate_survreg_dpqr_parms_reference.R`
adds 6,584 parameter cases and 144 complete Student-t density matrices. Query
checks distinguish R's `NA` and `NaN`, compare warning sequences and preserve
the seeded uniform stream. CI regenerates these fixtures with survival 3.8-12
and compares them exactly.

`scripts/generate_survreg_density_reference.R` writes a deterministic reference
for an asymmetric two-normal mixture, outside the built-in families. It uses
R survival 3.8-12 and exercises all four censoring types, weights, a formula
offset, and fixed/estimated/stratified scales. Ordinary coefficients,
likelihoods, covariance matrices, predictions and residuals are checked
against `survreg`, with explicit and automatic initialization. An `asinh`
response transform exercises its Jacobian and inverse; clustered robust
variance is checked on the noninterval responses. Interval residual scale
derivatives use the corrected formulas already used by the built-in
families, rather than R's erroneous postfit formulas.

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
15 fits per build gave these medians for 100,000 rows. Both used release
builds with `extension-module,ml`; the public-callback build ran first.

| Distribution | Batched core (`6702f52a`) | Public callbacks |
| --- | ---: | ---: |
| Lognormal | 45.01 ms | 45.21 ms |
| Weibull | 52.88 ms | 53.42 ms |
| Loglogistic | 41.93 ms | 41.62 ms |

All fitted coefficients and likelihoods were identical at 1,000, 10,000 and
100,000 rows. An earlier comparison in the reverse build order also found
unchanged results and similar timings. These measurements show no material
built-in fitting regression; they do not establish a speedup.

## Query benchmark

`PYTHONPATH=python .venv/bin/python scripts/benchmark_survreg_dpqr.py
--baseline-extension /path/to/predecessor/_survival.so` compares complete native
query calls, including NumPy argument conversion, callbacks and returned Python
lists. Inputs and distributions are prepared before timing. An optional
`--candidate-extension` selects a separate candidate library. Every case checks
all returned values before timing; the script pins one CPU and records eleven
alternating before/after pairs by default.

On 2026-10-09, matching `extension-module,ml` release builds with Rust 1.94,
Python 3.14.7, NumPy 2.4.6 and an Intel Core Ultra 5 325 produced these medians
for 100,000 values with scalar mean and scale. The predecessor hash is
`5bde3f26`; the revised query build is `50295ad6`. Each cell gives before →
after milliseconds.

| Family | Density | CDF | Quantile |
| --- | ---: | ---: | ---: |
| Gaussian | 1.459 → 1.314 | 1.621 → 1.439 | 1.085 → 1.049 |
| Weibull | 2.109 → 2.086 | 1.947 → 1.875 | 1.752 → 1.584 |
| Logistic | 1.329 → 1.215 | 1.213 → 1.044 | 0.920 → 0.884 |
| Logistic callbacks | 5.351 → 4.429 | 4.401 → 3.953 | 0.743 → 0.766 |

These are local measurements. The revised scalar built-ins were 1–14% faster;
the callback scalar quantile was about 3% slower. With per-row mean and scale,
Weibull density was about 5% slower and CDF about 3% slower; the other measured
built-ins were similar or faster. Callback density/CDF calls were 10–19% faster
across both layouts. The
[raw results](benchmarks/survreg-dpqr-2026-10-09.json) retain all 48 cases at
1,000 and 100,000 values, extension hashes and individual samples. These timings
measure ordinary matching lengths, rather than the expanded recycling cases.
