# Direct Cox survival curves

`survival.r.coxsurv_fit` exposes R's `coxsurv.fit` for prepared response and
design matrices, relative risks and coefficient covariance. It computes all
strata in one Rust call and retains no model frame or original observations.

```python
import numpy as np
from survival import r

curves = r.coxsurv_fit(
    y=r.Surv([1, 2, 3], [1, 1, 0]),
    x=np.array([[-1.0], [0.0], [1.0]]),
    risk=np.ones(3),
    x2=np.array([[0.0], [0.5]]),
    risk2=np.ones(2),
    varmat=np.array([[0.2]]),
)
assert curves.surv.shape == (3, 2)
```

The caller supplies already-centered covariates and risks using the same
convention. No fitting, centering, row omission or near-time adjustment occurs
in this interface. A raw `y` has `(stop, status)` or `(start, stop, status)`
columns with binary status; a right/counting `Surv` is also accepted. `x` is
a numeric matrix, including zero columns; `x2` is a matrix or a vector for
one prediction row. NumPy arrays, including noncontiguous inputs, cross the
native boundary without constructing nested Python row lists.

`wt=None` supplies unit case weights. Training risks must be positive and
finite; prediction risks may be zero. Weights must be finite and nonnegative,
and an observed risk set with zero weighted risk is refused. Responses and
covariates must be finite, with `start < stop` for counting intervals.

`stype=1` selects the Kalbfleisch–Prentice estimate. `stype=2` uses the
exponential of cumulative hazard, with Breslow (`ctype=1`) or Efron (`ctype=2`)
ties. Standard errors require a finite square `varmat` matching the number
of covariates. `se_fit=False` omits them and ignores `varmat`; the R spelling
`**{"se.fit": False}` is also accepted. As in R, `cluster`, `position` and
`oldid` are accepted but unused. They do not request a separate cluster
adjustment.

## Strata and individual trajectories

`strata` uses sorted distinct values, or declared categorical levels in their
original order, including unused levels. Omitting `strata` or supplying an
empty vector selects one curve labeled `"0"`. Missing stratum values are rejected.

Passing `id2` builds one trajectory per ID, in first-appearance order. `y2`
supplies `(start, stop]` intervals (or a counting `Surv`); an optional third
column is ignored. Rows for an ID may be interleaved with other IDs and
retain their input order. `strata2` contains **one-based positions** in the
training stratum levels, as in R, or factor codes. It can be omitted for
one training stratum. Interval time offsets, changes of stratum and
cross-interval coefficient covariance use the same kernels as fitted-model
prediction. Intervals with no baseline times contribute no rows.

## Results and ownership

With `unlist=True`, `CoxSurvFitResult` stacks time rows across curves. `n`
lists each curve's observation count; `strata` maps labels to time-row counts
when there is more than one curve. `surv`, `cumhaz` and optional `std_err`
have one column per prediction row, or a vector for a single prediction or
individual trajectories. `row_names` records explicit `rownames` or a pandas
input index. Standard errors are on the cumulative-hazard scale; this bare
interface does not add confidence limits or a `survfit` class.

`unlist=False` returns a `CoxSurvFitList`, an immutable sequence with `names`
and per-curve views. Ordinary curves also expose baseline `hazard`, `varhaz`,
`ndeath` and `xbar`; individual trajectories have none of these extra fields.
Python uses the actual IDs as individual list names. R's historical list form
overwrites them with training stratum labels, a quirk retained by the R bridge.

Numerical arrays are read-only NumPy views of a frozen native result. They
keep their owner alive; use `.copy()` to obtain writable arrays. List elements
share their parent storage, so retaining one element also retains its sibling
curves. Results and lists support pickle round trips. The flattened form drops
the extra baseline components after calculation.

`survival.r.survfitcoxph_fit` is the older wrapper: `survtype` is 1, 2 or 3
for Kalbfleisch–Prentice, Breslow or Efron. Its `newrisk` argument supplies the
prediction risks and is required. `vartype` is ignored, matching R's wrapper.

Rust callers can use `survival::surv_analysis::coxsurv_fit` with `CoxSurvData`
and `CoxSurvNewData`. This native interface uses zero-based stratum indices.
Its result exposes optional baseline components as `CoxSurvBaselineDetails`.
The Python domain binding `survival.surv_analysis.coxsurv_fit` exposes those
codes directly and releases the GIL during calculation.

## Reference checks and numerical edges

`scripts/generate_coxsurv_reference.R` records 144 cases from stock R survival
3.8-12: both response types, weighted ties, declared stratum order, one/multiple
prediction rows, individual trajectories, both estimators and tie methods,
optional standard errors and both output layouts. Fixtures retain the raw
R outputs. R bridge tests also check matrix/vector attributes, named risks,
single observations and the legacy wrapper.

In 24 Kalbfleisch–Prentice cases, R returns NaN after a terminal death because
risk-set subtraction produces a negative rounding residue before a fractional
power. The shared Rust kernel bounds that survival increment to [0, 1]. Tests
verify that every affected run begins with the full weighted risk set dying
and that its survival is zero to numerical tolerance; all other recorded
components are compared directly. An independent kernel test covers a
denominator rounded one representable value below its death risk.

Unused factor levels, zero-column designs and empty individual trajectories
have explicit zero-size outputs. These cases are checked independently of R's
dimension-dropping behavior. Shared curve calculations retain the working
memory and interval lookup improvements described in
[Cox curve performance](cox-curve-performance.md).

## Complete R-call benchmark

With the release extension installed and `RETICULATE_PYTHON` pointing to its
environment, run `Rscript scripts/benchmark_coxsurv.R`. The script requires
`pkgload` and `jsonlite`, checks the complete R results against stock survival
before timing, and excludes data preparation, garbage collection and warmup.
Both columns below measure complete R calls; the Rust bridge includes the
R/Python conversions.

On an Intel Core Ultra 5 325 with R 4.5.3, survival 3.8-12 and Python 3.14.7,
seven repetitions with 20,000 observations, four covariates and four strata
gave these medians. Ordinary prediction has eight rows; the individual curve
has 1,000 intervals and changes stratum.

| Workload | Standard errors | R survival | Rust bridge | Ratio |
| --- | --- | ---: | ---: | ---: |
| Ordinary curves | No | 34 ms | 15 ms | 2.27× |
| Ordinary curves | Yes | 48 ms | 19 ms | 2.53× |
| Individual trajectory | No | 65 ms | 13 ms | 5.00× |
| Individual trajectory | Yes | 67 ms | 13 ms | 5.15× |

These local measurements have millisecond timer resolution and vary with the
data and machine. They compare the current bridge with stock R survival;
they do not measure memory or isolate the numerical kernel.
