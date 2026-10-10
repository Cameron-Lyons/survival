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
one per basis column. Boolean penalty bases use numeric zero/one values;
nonnumeric bases fail before penalty callbacks run. A sparse penalty requires
one numeric group column.

The facade also provides standalone `r.ridge()` and `r.frailty()` constructors,
with `frailty_gamma`, `frailty_gaussian` and `frailty_t` aliases. Their results
extend `CoxPenaltyBasis`, so a time-transform can return them directly:

```python
def group_penalty(x, time, riskset, weights):
    return r.frailty(x, theta=.4, sparse=True)

def ridge_penalty(x, time, riskset, weights):
    return r.ridge(x, theta=2, column_names=["ridge(x)"])
```

`RidgeResult.scale_values` contains each column's sample variance, calculated
from the supplied values before later row selection. Numeric vectors and
matrices can be combined as separate arguments, following R's `cbind` rules.
Explicit `column_names` supplies complete coefficient labels; Python cannot
recover R's unevaluated argument expressions.
`FrailtyResult.levels` retains used factor levels in their declared order,
and `codes` retains one-based group positions, including missing values.
Sparse constructors prepare one group column with a single lookup per row;
dense constructors prepare an indicator column for every group. Constructor
results and their native penalties support pickle and fitted-model retention.
Penalty configuration and controller APIs represent R's `pfun`, `cfun` and
formatting closure attributes. Python result labels use the documented explicit
names because argument expressions are unavailable.

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
sparse frailty formulas. The subsequent [model-matrix reconstruction change](cox-model-matrices.md)
adds new-data sparse matrices and separates basis labels from coefficient labels.
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

The standalone-constructor reference adds seventy constructor objects and
thirteen complete Cox fits. It checks basis values, variance scaling, factor
order, sparse/dense layouts, native configuration, coefficient labels and
fitted outputs. Ordinary formula fits separately check AIC initialization,
gamma EM's ignored vector initialization and Gaussian REML's retained vector.
Large finite and subnormal ridge inputs check the shared variance helper;
NumPy arithmetic warnings are suppressed to follow stock `var()` behavior.

The full Python suite passes 15,239 tests (48 skipped and 37 documented
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
| Two-column ridge | 19 (18–20) | 18 (17–18) | 19 (18–19) |
| Spline | 35 (34–36) | 35 (34–35) | 26 (25–27) |
| Sparse gamma frailty | Penalty discarded | 18 (18–25) | 13 (13–14) |
| Dense gamma frailty | Unsupported | 31 (30–31) | 19 (18–19) |

An earlier current run measured ridge at 19 ms (18–20 ms). Supported
predecessor workloads show no measured regression or established speedup.
Spline and frailty complete calls remain slower than stock R in this workload.
Timing the predecessor's unpenalized sparse result would compare different
models, so it is excluded.
