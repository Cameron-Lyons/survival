# Cox time-transform penalties

Cox callbacks retain `coxph.penalty` attributes on numeric vectors, matrices,
and factors. Sparse frailties previously lost their penalty during numeric
coercion; dense factor penalties raised an unsupported error. Both now use
the existing Rust penalized fitter. Dense factor penalties use one indicator
column for every group, including the first group.

R callbacks can return `survival::frailty()`, `survivalr::frailty()`, splines,
ridge terms, or custom penalty objects. Sparse formatting callbacks receive
the fitted frailties and their variance diagonal, with the third argument
missing, as in R. Dense formatting callbacks receive coefficient and
covariance blocks. Controller history and formatter closures survive model
serialization. R frailty constructors also retain R's numeric/logical group
labels when values cross the Python boundary.

Python callbacks can pair a numeric vector or matrix with a native penalty:

```python
from survival import r, regression

def group_penalty(x, time, riskset, weights):
    return r.CoxPenaltyBasis(
        x, regression.CoxPenalty.frailty(theta=.4, sparse=True, n=len(x))
    )

fit = r.coxph("Surv(time, status) ~ age + tt(group)", data, tt=group_penalty)
```

`CoxPenaltyBasis.column_names` optionally supplies the complete fitted labels,
one per basis column. A sparse penalty requires one numeric group column.
The callback can also return `r.pspline()` directly:

```python
import numpy as np

def smooth_effect(x, time, riskset, weights):
    return r.pspline(np.asarray(x) * np.log(time), df=3)
```

This preserves basis boundaries, degree, combined columns and the native
penalty needed for linear/nonlinear summaries. Its coefficient labels use
the original variable name, such as `ps(x)3`, because Python callback results
do not retain an R argument expression. A raw native spline penalty requires
this `PsplineResult` basis. Unpenalized spline results remain ordinary matrix
terms. Penalty interactions and multiple sparse penalties raise errors.
Existing prediction, curve and model-retention restrictions for `tt()` remain.

Stored model matrices include the original sparse group column, including
custom numeric labels. The fitting matrix retains only the dense coefficient
columns; keeping the sparse labels costs one vector, without copying the
whole expanded design. This also corrects stored matrices for ordinary
sparse frailty formulas. New-data sparse matrices are outside this change.
Sparse callbacks without a custom formatter receive a group Wald summary.

## Independent checks

The generator records 32 complete fits from R 4.5.3 and survival 3.8-12:
gamma, Gaussian and Student-t frailties, sparse/dense representations,
Efron/Breslow ties, fixed/search/df controllers, sparse-only models,
factor level order, custom group codes, weights/offsets/strata,
counting-process inputs, subsets and missing-value exclusion. Spline cases
include searched/fixed penalties, different degrees, combined columns and
an unpenalized basis. Python checks fit values, expanded matrices/responses,
residuals, degrees of freedom, frailties, summaries and formatted history.
Live R tests exercise both frailty constructors, custom numeric penalty
vectors, sparse formatter signatures and saved models.

The full Python suite passes 15,237 tests (48 skipped and 37 documented
expected differences). The R source archive passes 8,629 checks with zero
errors, warnings or notes. Pinned lint/format checks, Mypy across 47 files,
generated interface checks and byte-exact fixture regeneration pass. Three
sparse/dense models also restore in a fresh R process with identical
coefficients, covariance, stored matrix, residuals and summaries. Rust
sources and the native extension remain unchanged from #697.

The reference explicitly retains these stock R differences:

- [`coxfit5.c`](https://github.com/therneau/survival/blob/master/src/coxfit5.c)
  omits event weights from the sparse frailty information diagonal. Weighted
  covariance, frailty variance and degrees of freedom use an independent
  weighted risk-set covariance calculation with R's sparse diagonal
  approximation and gamma penalty derivatives. Finite differences check
  that calculation. Raw stock covariance, degrees of freedom, frailty
  variance and summaries remain in the fixture. Only the two marked weighted
  cases use a larger absolute comparison tolerance of `5e-7`.
- Stock penalized time-transform martingale cleanup can violate risk-set sum
  identities. References use the ordinary R Cox kernel at the same fitted
  linear predictors, without optimization, and retain the raw residuals.
- Stock `vcov()` fails for a sparse-only fit whose dense covariance is `NULL`;
  that error is recorded. The implementation returns an empty dense matrix.
  Counting-process references use distinct stops to avoid the separately
  recorded stock tied-stratum boundary defect.

## Complete-call timings

`scripts/benchmark_cox_time_transform_penalties.R` uses 500 rows, stratified
formulas, three warmups and seven samples in separate processes. It includes
callback transport, risk-set expansion, fitting and retained X; explicit
garbage collection is excluded. Every workload first checks coefficients,
covariance and degrees of freedom against R. The predecessor (#699) and
current libraries use the same native extension.

| Callback | Predecessor median (range), ms | Current median (range), ms | R median (range), ms |
| --- | ---: | ---: | ---: |
| Two-column ridge | 19 (18–20) | 19 (18–20) | 19 (18–19) |
| Spline | 35 (34–36) | 35 (34–36) | 26 (25–27) |
| Sparse gamma frailty | Penalty discarded | 18 (18–24) | 13 (13–14) |
| Dense gamma frailty | Unsupported | 31 (30–32) | 19 (18–19) |

Supported predecessor workloads show no measured regression or speedup.
Spline and frailty complete calls remain slower than stock R in this workload.
Timing the predecessor's unpenalized sparse result would compare different
models, so it is excluded.
