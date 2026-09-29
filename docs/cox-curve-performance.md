# Full Cox survival curves

The Rust `expand_curve` and `individual_curve` kernels calculate full survival
curves and cumulative-hazard standard errors. `CoxPHFit.survfit` and
`survfit_individual`, including ordinary and penalized R-style Cox models,
share these kernels. The estimates and uncertainty formulas follow R's
`coxsurv.fit` for Kalbfleisch–Prentice, Breslow and Efron curves.

## Working memory and interval lookup

Coefficient uncertainty depends on a cumulative vector `dt` and the quadratic
form `dt' V dt`. Ordinary curves formerly allocated a time-by-covariate matrix
for each new covariate row; individual curves retained both the increments
and their cumulative matrix. Both now keep only the current vector, reducing
this temporary storage from O(Tp) to O(p) for T output times and p covariates.
The covariance matrix, cached baselines and requested output arrays retain
their existing sizes. This is a bound on uncertainty working storage, not
total prediction memory. Without standard errors, these uncertainty
calculations and buffers are omitted.

For time-dependent subjects, binary searches select baseline times in
`(start, stop]`. Across I intervals and U selected time rows, this takes
O(I log T + U), replacing a full scan of the relevant stratum for each
interval. The cumulative hazard, survival and standard errors are written
as times are visited, without retaining separate increment matrices.
Subject rows are grouped in one pass, preserving first-appearance subject
order and each subject's input interval order. The grouping uses O(I) row
indices and expected O(I) time, replacing repeated scans for each subject.

Covariate uncertainty still requires O(p²) work per output time. The change
preserves the quadratic-form summation order, relative-risk scaling, interval
time offsets, and cross-interval covariance when strata change. Matrix
dimensions are checked before covariance calculations.

## Reproduce

Build the Python extension in release mode, then run the same command against
both revisions:

```sh
PYTHONPATH=python .venv/bin/python scripts/bench_cox_curves.py
```

The script measures complete native Python method calls. Inputs, fitting,
baseline warmup and result conversion for checksums are excluded. It reports
raw samples, medians, dependency versions, extension hashes and numerical
checksums for ordinary predictions, one subject with many intervals, and
many subjects with two short intervals each. All workloads run with and
without standard errors. These comparisons measure two versions of this
implementation, not execution speed against R.

## Local comparison

On an Intel Core Ultra 5 325, Linux x86-64, Python 3.14.7 and NumPy 2.4.6,
seven warmed calls per case gave these medians. Each fit had 20,000 distinct
stop times and 12 covariates. Ordinary prediction used eight new rows; the
changing subject used 1,000 intervals spanning the full grid; the short
subjects used 1,000 interleaved IDs with two one-time intervals each.

| Workload | Standard errors | Before | After | Before / after |
| --- | --- | ---: | ---: | ---: |
| Ordinary curves | No | 2.470 ms | 2.626 ms | 0.94× |
| Ordinary curves | Yes | 26.937 ms | 18.125 ms | 1.49× |
| One changing subject | No | 10.378 ms | 1.439 ms | 7.21× |
| One changing subject | Yes | 14.220 ms | 3.785 ms | 3.76× |
| Many short subjects | No | 17.311 ms | 0.693 ms | 24.99× |
| Many short subjects | Yes | 17.573 ms | 0.932 ms | 18.85× |

All six numerical output checksums matched exactly. Ordinary prediction
without standard errors was slightly slower in this run; the large gains
come from avoiding interval scans and uncertainty matrices. These are local
timings and vary with scheduling. They do not measure peak resident memory.

## Numerical checks

Eight reference cases from stock R survival 3.8-12 cover weighted counting
data, tied deaths, two covariates, offsets, interleaved subjects, changes of stratum,
both `stype` and `ctype` settings, and standard errors on and off. Regenerate
them with `Rscript scripts/generate_cox_individual_reference.R`.

Independent hand calculations check correlated coefficient covariance across
intervals, exact open/closed endpoints, time offsets through empty intervals,
zero-covariate models, empty outputs, and separate ordinary prediction rows.
Existing R fixture tests also exercise fitted-model predictions and penalized
curves.
