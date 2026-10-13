# Predictions and residuals

This page covers predictions from fitted Cox and AFT models: prediction types,
missing values, term selections, row labels, grouped (collapsed) predictions and
residuals, and the native array interfaces. Formula evaluation is described in
[Formula interface](formula-interface.md); differences from R are listed in
[R compatibility](r-compatibility.md#predictions-and-residuals).

## Prediction types

`r.predict(fit, newdata=None, type=..., se_fit=...)` and `r.fitted(fit)`
support Cox `"lp"`, `"risk"`, `"expected"`, `"survival"` and `"terms"`
predictions, and AFT `"lp"` (`"linear"`), `"response"`, `"terms"`, `"quantile"`
and `"uquantile"` predictions; R's `se.fit` and `na.action` spellings are
accepted. Without `newdata`, predictions refer to the training rows; Cox
`fitted()` returns the unpadded stored predictors. AFT new data may contain a
stratum the fit did not see: the linear predictor, response and terms are
returned, and quantiles of an estimated-scale fit are NA for those rows, as in
R; Cox predictions refuse an unknown stratum, as R does.

- AFT `"quantile"` returns response-scale quantiles at `p` (R's default
  `[0.1, 0.9]`) and `"uquantile"` the same quantiles on the linear scale.
  Probabilities 0 and 1 return the distribution's limits. Quantiles use each
  row's stratum scale and keep the Student-t degrees of freedom; `se_fit=True`
  adds standard errors on the requested scale. Formula offsets enter the linear
  predictor of new rows as well as training rows.
- AFT quantile predictions pass the complete numeric probability vector to the
  distribution, keeping missing and out-of-range columns beside valid ones;
  custom quantile callbacks decide their own behavior for those values. Other
  prediction types ignore `p`. An explicit empty probability request returns a
  zero-column result, including with `se_fit=True`. Nullable list, NumPy, pandas
  and Polars query values also work in `dsurvreg`, `psurvreg` and `qsurvreg`.
- The typed `SurvregFit.predict(...)` takes `offset=` and `strata=` for its new
  rows. `AFTEstimator.predict()` returns the fitted response-scale location;
  `predict_median()` and `predict_quantile()` return distribution quantiles.
- As in R, an aliased AFT coefficient makes ordinary new-data predictions
  missing; term predictions keep the contributions of other terms.

Native expected-count predictions require a stratum for every new row when the
model has several baselines. Ordinary `survfit(newdata)` may omit strata because
it returns a curve block for every baseline, and individual covariate paths keep
R's default of the first stratum. Public Rust fitted-Cox predictions check stored
row, covariate and covariance dimensions before using cached baselines, and
missing stored strata or an unknown training stratum code are input errors.

## Missing values

Under `na.pass` (the prediction default), a missing covariate affects only the
predictions that use it. Term predictions keep unaffected contributions and
standard errors; an unknown offset leaves the linear-predictor standard error
available. This includes ridge and P-spline terms. AFT location and term
predictions do not need a known scale stratum, and expected-count predictions use
follow-up times even when the event indicator is missing. `na.omit` drops
incomplete rows, `na.exclude` restores them as all-missing rows, and `na.fail`
rejects them. A newdata row with an infinite covariate raises "newdata contains
non-finite value".

Predictions keep R's distinction between `NA` and numerical `NaN`. A source `NA`
payload survives formula evaluation, the native calculation and conversion back to
R; logarithms, square roots and division keep the source kind, while domain errors
such as `sqrt(-1)` and `0/0` produce genuine NaN. Categorical missing values,
missing term selections and rows restored by `na.exclude` produce `NA`. Python
receives floats for both kinds; both satisfy `math.isnan`.

AFT quantiles drop single-row and single-column dimensions before restoring
excluded rows, as `predict.survreg` does. For four new rows, three excluded rows
and three quantiles, the retained third row's values form a vector before three
missing entries are inserted: `c(NA, NA, q10, NA, q50, q90)`. Fixed-scale errors
repeat the retained row's name before omission labels are restored. Entirely
omitted predictions keep R's dimensions and absent row names; entirely omitted
AFT quantiles for several probabilities keep a zero-row matrix of the right width,
and multistratum quantiles keep an explicit `list(NULL, NULL)` dimension-name
attribute. New-data linear, risk and term predictions from frailty-only Cox fits
use the new row count without evaluating the frailty variable, so missing frailty
values do not remove rows.

## Term selections

AFT term predictions use R's matrix column selection. Positive indices are
one-based, zero selects nothing, negative indices exclude terms, and finite
fractional indices truncate toward zero. Logical masks recycle across columns; a
mask longer than the number of terms is an error. Repeated indices keep their
order. Numeric missing indices create a missing column; a logical missing value
creates a missing column at each selected position. Missing columns have missing
names and errors.

```python
age = r.predict(fit, type="terms", terms=[True, False])
rx = r.predict(fit, type="terms", terms=-1)
empty = r.predict(fit, type="terms", terms=0)
with_missing = r.predict(fit, type="terms", terms=[1, float("nan")], se_fit=True)
```

In Python, `None` omits the argument, a list containing `None` represents logical
missing values and a numeric NaN is a missing numeric index; NumPy scalars and
vectors follow the same rules. The R bridge keeps the selector's original type,
including an explicit `NULL`, so a numeric `NA` selects one missing column and a
logical `NA` recycles; factor selectors use their level codes. Mixing positive and
negative indices, unknown names, oversized masks and out-of-range selections raise
R's errors, and integer coercion overflow signals R warnings.

Cox predictions keep R's stricter validation: only valid term names and positive
integral indices are accepted, also when `terms` accompanies a vector prediction.
AFT ignores `terms` for other prediction types. An explicit R `terms=NULL` selects
no Cox term columns, while AFT keeps all columns.

Prediction columns are named from the fitted coefficient assignments, excluding
standalone strata terms: `strata(cl) + age + rx` predicts columns `age` and `rx`.
`model_term_names()` still returns the complete formula labels, including strata.
The R prediction bridge accepts `na.omit`, `na.exclude`, `na.pass` and `na.fail`
functions as `na.action`.

## Row labels in R

The R bridge keeps source row labels on Cox and AFT predictions, including custom
names, Unicode, repeated or reordered subsets and rows restored by `na.exclude`.
Models keep their labels from before the missing-data action, and repeated subset
rows get R's `make.unique` suffixes. Automatic row names stay implicit until
needed. Names that R leaves absent stay absent:

- an unstratified Cox model's stored linear predictor is unnamed without errors,
  while term matrices name their rows;
- expected-count predictions name the training fit but leave its errors unnamed,
  and new-data expected outputs are unnamed;
- AFT stored response and link fits are unnamed while their errors carry
  model-matrix row names;
- scalar new-data quantiles carry subject names, multiple quantiles can carry
  scale-stratum names, untransformed quantile errors can be unnamed, and a
  fixed-scale prediction for one subject repeats its name on the quantile errors.

Models saved before row labels were retained cannot recover custom source names;
their numerical behavior is unchanged.

## Grouped predictions and residuals

`r.predict(..., collapse=groups)` sums ordinary Cox predictions within groups,
and `r.residuals(..., collapse=groups)` does the same for Cox and AFT residuals.

```python
from survival import datasets, r

data = datasets.load_ovarian()
fit = r.coxph("Surv(futime, fustat) ~ age + rx", data)
groups = [i % 3 for i in range(len(data["futime"]))]
predictions = r.predict(fit, type="lp", se_fit=True, collapse=groups)
residuals = r.residuals(fit, type="score", collapse=groups)
```

- Factor groups follow their declared level order, with unused levels removed.
  Other groups follow sorted value order: numbers numerically and text lexically,
  including numeric-looking text. Missing labels form a final group with a
  warning; numeric `NaN` and `None`/R `NA` are distinct groups, while factor
  missing values share one group.
- Values sum within each group in input order; standard errors are
  `sqrt(sum(se^2))`, as in R's collapse method, with each per-row error computed
  before squaring so negative variances still propagate NaN.
- `r.predict` takes one label per row of the data before omission; labels may be
  factors, text, logical or missing values.
- For new data, missing collapse labels also count for `na.omit`, `na.exclude`
  and `na.fail`. Under `na.pass`, a missing group can hold valid predictions.
  Under `na.exclude`, groups containing an omitted row propagate missingness;
  groups consisting only of omitted rows are restored after the native call.
- Cox residuals accept `collapse=True` to use the fitted cluster or id, whose
  level order survives subsets and saved models; with neither, the call raises an
  error. Schoenfeld and scaled Schoenfeld residuals ignore `collapse`, as in R.

Python returns lists and `PredictResult`; the R methods attach group labels as
vector names or matrix row names, as plain character vectors, also when a
one-column matrix is dropped. Empty grouped Cox term matrices keep explicit
`character(0)` labels, and an all-omitted term result keeps its zero-row shape
and column names.

Grouped Cox predictions are summed inside the native call, and term predictions
allocate one output row per group. Rust callers can use
`CoxPHFit::predict_terms_grouped(newdata, se_fit, reference, assign, group)` or
`CoxPrediction::collapse(group)` after `predict_lp`, `predict_risk`,
`predict_expected` or `predict_survival`. Native Python methods take a
keyword-only `collapse` with one integer code per prediction row; arbitrary signed
codes sort in ascending order, and list properties are owned snapshots:

```python
terms = fit.fit.predict_terms(se_fit=True, collapse=integer_group_codes)
lp = fit.fit.predict("lp", se_fit=True, collapse=integer_group_codes)
```

`survival.core.grouped_sum(values, group, squares=False)` (Rust
`survival::core::grouped_sum`) sums matrix rows by ascending integer group in
input order, validating the group length and keeping empty dimensions; with
`squares=True` it returns the square root of the sum of squares. It accepts lists
and C, Fortran or strided NumPy matrices, returns an owned NumPy matrix and
releases the GIL.

## Cox predictions at requested times

`CoxPHFit.predict_survival_at(times, newdata=None, new_strata=None,
new_offset=None)` returns a NumPy array of shape `(n_times, n_observations)`.
Omitting `newdata` predicts for the training rows; new observations must supply
their strata when the model has several. Requested times may be unsorted or
repeated, and an empty grid returns zero rows. The Rust counterpart is
`CoxPHFit::predict_survival_at(&times, newdata)`. Both use the default Cox
estimate (`stype = 2`, with the hazard type of the fitted tie method) and keep
`exp(-H).powf(risk)` rounding, step values at ties, survival 1 before the first
time, extension after the last time, offsets, aliased coefficients and stratum
order. Use `survfit` for other estimates, confidence intervals or time-varying
covariates. `CoxPHEstimator.predict_survival_function` and the R-style `brier`
use this method; the estimator keeps its `(n_observations, n_times)` orientation
and `brier(detail=True)` keeps its nested-list `phat` result.

## Native results

Native Cox vector and term predictions and AFT predictions provide `to_arrays()`
snapshots: the fit and optional standard errors as independent writable float64
NumPy arrays, with absent errors as `None`. Term snapshots include the Cox
constant. `CoxTermsPrediction` and `SurvregPrediction` keep `n_columns`
separately from the row count, so zero-row and zero-column results keep their
shape; Rust callers constructing these structs directly must supply it. The list
properties and default formula prediction results keep their list interfaces.

Cox new-data predictions and curves and formula-based AFT predictions build an
owned contiguous float64 design from numeric NumPy and pandas columns.
Interactions multiply columns in fitted order with R's left-to-right scalar
order. Factors, contrasts, transforms and penalty bases keep their evaluators;
masked, object, string and nullable columns keep scalar coercion. Inputs, including
read-only and strided views, are never mutated or shared with the design.
