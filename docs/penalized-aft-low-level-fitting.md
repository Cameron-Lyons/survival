# Bare penalized parametric survival fitting

`survival.r.survpenal_fit` fits prepared AFT matrices with ridge, spline,
dense or sparse frailty, and custom penalties. It shares the full model's
Rust solver and returns a compact result with no retained training rows
or callbacks.

```python
import numpy as np
from survival import r, regression

x = {"Intercept": np.ones(6), "group": [0, 1, 1, 0, 1, 0]}
y = [[1.2, 1], [2.5, 0], [.9, 1], [3, 1], [1.8, 1], [2.7, 0]]
fit = r.survpenal_fit(
    x, y, dist="gaussian",
    pcols=[[1]], pattr=[regression.CoxPenalty.ridge(theta=1)],
    assign={"Intercept": [0], "group": [1]},
)
print(r.print_survreg_penal(fit))
```

Response codes, fitting scales, base densities, controls, weights, offsets,
initial values and stratum codes follow [bare AFT fitting](aft-low-level-fitting.md).
The response must already be transformed; for Weibull fitting, supply logged
times with `dist="extreme"`. `scale=0` estimates log-scales; positive values
fix the scale. Custom distributions also need their variance component to
compute the effective sample size used by penalty searches.

## Penalty and term inputs

`pattr` is a sequence of native `regression.CoxPenalty` objects or `r.pspline`
results. Native constructors support `ridge`, `pspline`, `frailty` and
`callback`. `pcols` gives the design columns for each penalty. `assign` maps
term names to column groups, or supplies a sequence of groups; the default
uses the penalized groups and one term per remaining column.

Column indices in `pcols`, `assign` and returned `assign2` are **zero-based**.
R's corresponding indices are one-based. Every design column must belong
to exactly one term, and each penalty's column group must match a term.
Term names must be unique; `sigma` is reserved when scales are estimated.
The input order of penalties need not match the term order.

A sparse penalty takes one numeric group column; its sorted unique values
identify the frailty coefficients. Only one sparse term is allowed.
Dense frailty penalties instead take indicator columns. The sparse group
column is removed from the dense coefficients and covariance matrices;
`frail` and `fvar` hold its estimates and diagonal variances separately.

For a spline, place `spline.basis` into the design and pass that same
`PsplineResult` in `pattr`. Its column count must match `pcols`. The result
retains only the small spline description needed for printing, not the
basis or penalty callbacks. Passing a native `CoxPenalty.pspline` also fits,
but without basis metadata its report uses ordinary coefficient rows.
Coefficient labels come from the design or `column_names`; native penalties
do not carry R's optional `varname` replacement attribute.

## Results and reports

`SurvpenalFitResult` exposes coefficients (including estimated log-scales),
`icoef`, `var`, `var2`, `loglik`, `iter`, linear predictors, per-term `df`,
`penalty`, `score`, named `pterms`, named `assign2` and named search `history`.
It also exposes `n`, `nvar`, `n_eff`, scales, convergence status and the
inner iterations that exhausted their limits. `df2` is `None`, matching
R's unpopulated component. Sparse fits return the single final penalty;
dense fits return the initial zero and final penalty, matching R's list
shapes. The native result uses a consistent two-element penalty array.

The score includes sparse frailties first, followed by dense coefficients
and estimated log-scales. Native `PenaltyHistory` objects retain their
column schema even when R omits names for an empty fixed-gamma history.
The result is immutable; numeric property access returns independent copies.
It can be pickled even when fitting used local callback functions.

`r.print_survreg_penal` accepts compact and full penalized AFT fits. Reports
include spline linear/nonlinear tests, dense and sparse frailty Wald tests,
penalty-search summaries, scales, iteration counts and likelihood-ratio
tests. Reports support `str()` and `r.as_data_frame()`. Callback penalties
use built-in coefficient or sparse Wald rows; arbitrary R `printfun`
closures are not retained or executed. Zero frailty variance gives R's
NaN test rather than a Python division error, including in Cox summaries.

The port's existing numerical corrections remain in force: dense linear
predictors include offsets and treat aliased coefficients as zero; a dense
penalty can follow a sparse term. See [R compatibility](r-compatibility.md)
for the other documented penalized AFT differences. As in R's current
implementation, exhausted inner iterations do not emit a warning; inspect
`inner_failures` and `converged`. Robust covariance belongs to the full model.

## Rust API and retention

Rust exposes `regression::SurvpenalFitResult::fit`. Python's native binding
is `regression.survpenal_fit_raw`, which accepts prepared `SurvregData`,
a distribution, penalties and column groups. It borrows the input data
while fitting, releases the GIL around the solver, and reacquires it for
Python callbacks. It refuses clustered inputs. Full models use the same
engine, then retain model data and calculate any requested robust variance.

`scripts/benchmark_penalized_aft_lowlevel.py` compares native calls on the
same prepared data, verifies numerical agreement, and measures serialized
result size. A local CPython 3.14 run with 100,000 rows, six design columns,
one warmup and seven measured calls per mode produced:

| Result | Median binding call | Serialized size |
| --- | ---: | ---: |
| Full model | 102.72 ms | 7,401,304 bytes |
| Bare fit | 100.65 ms | 801,153 bytes |

Runtime was similar; serialized output was about 89% smaller. These are
retained-output measurements, not peak-memory measurements. Data construction,
property copying and serialization are excluded from the timings.

## R reference checks

`scripts/generate_penalized_aft_lowlevel_reference.R` generates 60 cases
from R 4.5.3 / survival 3.8-12. They cover all four base densities, weights,
offsets, interval censoring, fixed and stratified scales, initialization,
iteration limits, ridge/spline searches, reordered terms, aliases, dense
and sparse gamma/Gaussian/t frailty, and report text. Numerical comparisons
retain the documented offset and alias corrections. Near-zero aliased
coefficients can print different tiny values; their report totals are
checked separately from the coefficient text.

Stock R's known callback-density interval/sparse C bugs are excluded from
this generator. Existing tests cover the corrected solver paths against
references from a patched R build. Additional tests check callbacks,
ownership, pickling, matrix layouts, validation and GIL release.
