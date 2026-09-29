# Bare Cox fitting

`survival.r.coxph_fit`, `agreg_fit` and `agexact_fit` expose R's exported
matrix fitters. They return a `CoxFitResult` with coefficients, model-based
variance, initial/final log likelihood, score test, iteration count, linear
predictors, means and optional martingale residuals. `agreg_fit` also returns
the final score vector (`first`) and iteration diagnostics (`info`).

```python
from survival import r

y = r.Surv([1, 2, 3, 4], [1, 1, 0, 1])
fit = r.coxph_fit({"group": [0, 1, 0, 1]}, y, resid=False)
assert fit.coefficient_names == ("group",)
assert fit.residuals is None
```

`coxph_fit` accepts a right-censored response; `agreg_fit` accepts counting
process data; `agexact_fit` accepts either and supplies entry time zero for
right-censored data. The design can be a numeric NumPy matrix of any memory
layout, a nested sequence, a mapping of columns or a data frame. A single
vector is also accepted by `coxph_fit`. Empty designs have zero columns.
Column names are inferred from mappings/data frames or supplied using
`column_names`; `rownames` supplies explicit row labels.

These functions consume prepared rows. They do not build a formula design,
omit missing rows, merge nearly equal times, center offsets, or compute
robust variance and concordance. Invalid or missing numerical data is
rejected. `nocenter=None` centers every covariate; pass `[-1, 0, 1]` for the
usual high-level binary-column policy. Controls come from `coxph_control`.
Offsets retain their original values, including in linear predictors.

The R fitters choose the likelihood in a distinctive way: `coxph_fit` and
`agreg_fit` select Efron for `method="efron"` and Breslow for the other
accepted labels, including `"exact"`. `agexact_fit` always uses exact
likelihood and requires unit weights. Use high-level `r.coxph(...,
ties="exact")` for its usual tie selection and model post-processing.

Output follows the bare R lists. Null `coxph_fit`/`agreg_fit` models have a
single log likelihood, no coefficient/variance/score/iteration components,
and `classes=("coxph.null", "coxph")`. Ordinary fits have `classes=("coxph",)`.
`agexact_fit` retains R's single-column matrix for linear predictors, its
historical `method="coxph"` label and empty class list, including for a
zero-column design. Arrays are independent copies on access; names are
immutable tuples. Results support pickling. They retain no response or
design matrix and cannot be used as fitted high-level prediction models.

## Native interface and cost

Rust callers use `regression::CoxphFitResult::fit(data, options, resid)`.
Python's `survival.regression.coxph_fit_raw` exposes the same numerical
result without R's null-model component removal or shape adjustments.
Its `method="exact"` selects exact likelihood. It releases the GIL during
the numerical fit and optional residual calculation. The result holds no
training rows; cluster and robust-variance options are refused in Rust.

Both interfaces share the existing optimizer with `CoxPHFit`; the bare path
skips full-model diagnostics and retained input data. `resid=False` also
skips the residual pass and its arrays. The benchmark
`scripts/benchmark_cox_lowlevel.py` compares full fits and both bare modes
on identical inputs, checks numerical agreement and reports serialized
result sizes. Those sizes measure retained output, not peak allocation.
The `coxph_bare_without_residuals` Rust benchmark isolates native fitting.

One local CPython 3.14 run with 100,000 rows, six covariates, one warmup and
seven measured calls per mode gave these medians:

| Result | Binding call | Serialized size |
| --- | ---: | ---: |
| Full model | 76.03 ms | 9,369,164 bytes |
| Bare, with residuals | 29.47 ms | 1,600,589 bytes |
| Bare, without residuals | 27.56 ms | 800,584 bytes |

The bare calls were about 2.6–2.8 times faster for this workload. Skipping
residuals reduced serialized output by about 91% relative to the full
model. Timings include Python input conversion and result construction;
input generation, property copying and serialization are excluded.
Performance depends on data shape, tie structure and hardware.

## R reference checks

`scripts/generate_cox_lowlevel_reference.R` regenerates 59 cases using
R 4.5.3 / survival 3.8-12. They cover all three fitters, raw offsets, strata,
weights, starting values, centering, iteration limits and warnings,
singular and zero-column designs, no events, near ties, ignored counting
rows, optional residuals and output metadata. Additional tests cover NumPy
layouts, data frames, ownership, pickling, validation, agreement with full
models and GIL release.

The shared optimizer preserves one documented correction to R: after
`coxph.fit` exhausts more than one iteration, R inverts a recomputed
information matrix without first factoring it. This port factors it before
inversion. The reference test checks the corrected variance against the
inverse numerical likelihood Hessian; all other components in that case
are compared with R. Converged fits retain R's variance calculation.

The Python boundary rejects malformed lengths, nonnumeric designs and
missing strata explicitly. It accepts the three documented method labels;
arbitrary R string labels are not accepted. It does not attach R classes
or names to Python numeric lists: `classes`, `coefficient_names` and
`row_names` hold that metadata separately.
