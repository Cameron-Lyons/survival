# Cox time-transform result shapes

`coxph(..., tt=...)` now rebuilds the fitted design after risk-set expansion.
A callback can return a numeric vector, a numeric matrix, a logical vector,
or a factor. Python matrices can be nested rows, NumPy arrays, data frames,
or mappings of column names to columns. Declared factor levels, including
unused levels, survive the expansion; ordered pandas categoricals use
polynomial contrasts. R factor results retain their ordered or custom
contrasts.

For example:

```python
import numpy as np
from survival import r

def time_basis(x, time, riskset, weights):
    return {"log": np.asarray(x) * np.log(time),
            "root": np.asarray(x) * np.sqrt(time)}

fit = r.coxph("Surv(time, status) ~ x + tt(x)", data, tt=time_basis)
```

The fitted columns are `x`, `tt(x)log`, and `tt(x)root`. An unnamed matrix
with multiple columns uses numbered suffixes; a one-column matrix keeps
the plain transform name, matching R. Interaction columns and assignments
use the transformed widths and factor contrasts. Each distinct transform
runs once, in formula variable order, even when it occurs only in an
interaction. Callback risk-set IDs start at one; unweighted callbacks receive
`None` in Python and `NULL` in R.

The response, IDs, clusters, and risk-set strata retained by the fit now
describe the expanded rows. Original observation count and missing-value
records remain available. Stored model matrices, residuals, detailed Cox
output, and serialization use the expanded fitted data. Prediction, curve
creation, and `model=TRUE` retain their existing restrictions for `tt()`
models; evaluating a matrix transform on new data requires its expanded
time and risk-set context.

R matrix results with `coxph.penalty` attributes use the existing Rust
controlled-penalty fitter. The R controller's final history and formatting
callback remain available for spline summaries and saved models. Fitted
coefficients, variances, effective degrees of freedom, and linear/nonlinear
summary rows agree with R for transformed splines. Penalties in interactions
are rejected, as in R. Factor-valued penalties currently raise an explicit
unsupported error instead of losing their penalty.

## Reference checks

The independent generator records 46 fits using R 4.5.3 and survival 3.8-12.
It covers vector and matrix results, logical and unordered/ordered factors,
declared input levels, multiple callbacks and interaction-only terms,
weights, offsets, strata, robust clusters and IDs, counting-process data,
subsets, and missing-value exclusion. Python tests also check matrix memory
layouts, callback order, malformed outputs, initial-value widths, and pickle
round trips. Live R tests cover custom contrasts, native penalty callbacks,
summary output, and serialization.

The full Python suite passes 15,195 tests (48 skipped and 37 documented expected
differences). The R source archive passes 7,992 checks with zero errors, warnings,
or notes. Pinned lint/format, Mypy across 47 files, and generated interface checks
pass. Rust sources and the native extension are unchanged from #697.

Three stock R discrepancies are recorded explicitly:

- Interaction-only transforms can fail because `coxph` counts main-effect
  terms rather than distinct transform variables. Only after that failure,
  the reference generator corrects the count; R's expansion and numerical
  fitting code remain unchanged.
- Stock [`Ccoxcount2`](https://github.com/therneau/survival/blob/master/src/coxcount1.c)
  indexes sorted stratum boundaries through original row indices while walking
  tied deaths. Counting-process references use distinct stop times to avoid
  this separate defect. Right-censored references retain ties.
- Stock transformed spline martingale residuals can have nonzero sums within
  one-event risk sets. Their reference is R's ordinary Cox kernel at the same
  fitted coefficients, with aliased coefficients set to zero and optimization
  disabled. Some stock detailed-output calls also fail with nonfinite values;
  those failures are recorded rather than used as numerical references.

## Complete-call timings

The benchmark uses 500 rows, weighted stratified formulas, three warmups,
and seven samples in separate processes. It includes callback transport,
risk-set expansion, fitting, and retained X; explicit garbage collection is
excluded. The predecessor and current libraries use the same native extension.
Each timed workload first checks coefficients and covariance against R.

| Callback | Predecessor median (range), ms | Current median (range), ms | R median (range), ms |
| --- | ---: | ---: | ---: |
| Vector | 14 (14–15) | 15 (14–15) | 19 (18–19) |
| Two-column matrix | Unsupported | 18 (17–25) | 19 (18–19) |
| Factor | Unsupported | 18 (18–18) | 13 (12–13) |

These measurements support no speedup claim over the predecessor. The factor
callback still costs more than stock R in this workload. Reproduce the
measurements with `scripts/benchmark_cox_time_transforms.R` using isolated
installed libraries and the `baseline`, `installed`, or `stock` mode.
