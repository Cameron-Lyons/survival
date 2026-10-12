# R bridge

The experimental R package in `r/survivalr` is an R facade for the Python
`survival.r` interface, called through reticulate. Its
[README](../r/survivalr/README.md) covers installation and the supported entry
points. This page describes how data cross the bridge, model persistence and the
conditions the bridge signals. Bridge-specific differences from R are listed in
[R compatibility](r-compatibility.md#r-bridge-rsurvivalr).

## Data columns

R data frames and named lists use one column adapter for formula fits, model
matrices, predictions and the other formula functions.

- Unclassed double columns, including those with `NA` and `NaN`, and complete
  integer columns cross as one-dimensional NumPy arrays in bulk. The R `NA`
  payload is kept separately from genuine `NaN` through formula evaluation and the
  native calculation, and back to R. Empty and single-value vectors keep their
  vector shape, and row labels travel separately.
- Missing integer, logical and character values use explicit missing markers;
  integer missing values never become ordinary NumPy integers, and logical `NA`
  stays missing. Factors keep their declared levels. Classed columns (dates,
  ordered factors and other classes), matrix columns and long vectors beyond R's
  array extent use the scalar adapter; nested, `NULL` and dimensioned list
  entries keep their existing path.
- Numeric views are read-only and keep their storage after the original R object
  is changed, removed or collected.
- A variable-free model frame from a list or plain mapping has zero rows, as in
  R, because unused columns do not supply a row count; data frames keep their
  explicit row count.

A data frame with a non-null `terms` attribute is treated as an evaluated model
frame by Cox and AFT `model.matrix`; see
[Formula interface](formula-interface.md#evaluated-r-model-frames). Native
prediction results cross back as owned arrays (see
[Predictions and residuals](prediction.md#native-results)); reticulate makes a
column-ordered copy when converting NumPy arrays to R.

## Warnings and errors

The bridge captures Python warnings raised during a call, also when the call
raises an error, and signals them through R's condition system before rethrowing
the original exception. Model fitting convergence warnings, formula domain
warnings in `model.matrix` and the AFT distribution query warnings all arrive as R
conditions. Positional formula arguments and named model-method arguments are
passed through. Direct Python calls keep Python's exception behavior and warning
filters.

## Model persistence

Fitted models, curves and responses support `saveRDS`/`readRDS`, `save`/`load`
and `serialize`/`unserialize`. Coefficients, prediction, residuals, diagnostics,
model frames and curve methods work on a loaded object, including in a separate R
process.

```r
library(survivalr)
fit <- coxph(Surv(time, status) ~ age + sex, survival::lung)
saveRDS(fit, "cox-fit.rds")
# In a new R session with compatible R and Python packages installed:
loaded <- readRDS("cox-fit.rds")
coef(loaded)
```

Loading a file loads the `survivalr` namespace but does not initialize Python; the
first field access or bridge method restores the Python object in place, keeping
the R proxy's classes, attributes and references, and later calls reuse it. Saving
again captures the current state. Fits do not keep an eager serialized copy.
Grouped curves save their per-stratum objects and the full backend used by summary,
residual and pseudo-value methods; extracted curves, selected multi-state curves
and nested Python objects get their own state.

The saved state has version `1`, a Python pickle payload and an R list of
callbacks. A small C adapter registers an ALTREP raw class whose serialization
hook runs only when R saves the object; it contains no numerical code. R functions
wrapped by reticulate, such as local custom AFT density, initialization, quantile
and deviance callbacks, are saved as the original R functions with their
environments and rebound on loading. An unsupported wrapper representation fails
serialization rather than dropping a callback.

- The package needs R 3.6 or later. Both `survivalr` and a compatible Python
  package and native extension must be installed when the object is used. The
  payload uses the installed Python's highest pickle protocol and the package's
  native binary state; it is not a portable archival or cross-version format.
- Use R's default serialization version 3. Version 2 is rejected because it would
  discard the state. Files saved by versions without this support contain no model
  state and raise an error asking for the model to be recreated.
- Only load trusted files: restoring the embedded pickle has the same execution
  semantics as loading Python pickle. R callback environments keep the ordinary
  limitations of R serialization for external resources.
- Models saved by earlier versions keep their originally fitted numerical designs:
  logical-covariate interaction widths need a refit to change, historical
  time-transform contrast declarations cannot always be recovered, custom source
  row names were not saved, and generic NaN omission placeholders are returned as
  R `NA`.
