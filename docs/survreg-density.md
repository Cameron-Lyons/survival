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

For direct bindings, use `SurvregDistribution.from_callbacks(...)`, then
`with_transform(...)`, `with_parms(...)`, or `derived(...)` as needed.
`family` and `transform` report `Custom` for runtime implementations.
`pdf_values`, `cdf_values`, `quantile_values`, and `sample` take length-one
or per-row arrays for means and scales. The R-style functions also accept
scalar arguments.

`survregDtest` checks density output at 0.1 through 1 and checks transform
inversion and positive finite derivatives at 1 through 10. Fitting checks
results again on actual observations. Callback exceptions include the
callback name and original exception text; Python exceptions raised during
native computation surface as `RuntimeError`. Invalid shapes and numerical
values detected by Rust surface as `ValueError`.

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
