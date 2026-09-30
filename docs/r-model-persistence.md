# R model persistence

The R facade now saves complete model state with `saveRDS`/`readRDS`,
`save`/`load`, and `serialize`/`unserialize`. Previously, these functions saved
reticulate's external pointer, which becomes invalid after loading. Coefficients,
prediction, residuals, diagnostics, model frames, and curve methods can now use
a loaded object, including in a separate R process.

```r
library(survivalr)
fit <- coxph(Surv(time, status) ~ age + sex, survival::lung)
saveRDS(fit, "cox-fit.rds")
# In a new R session with compatible R and Python packages installed:
loaded <- readRDS("cox-fit.rds")
coef(loaded)
```

Loading the file loads the `survivalr` namespace through R's serialization
mechanism. It does not initialize Python. The first field access or bridge
method restores the Python object in place, preserving the R proxy's classes,
attributes, and references. Later calls reuse that live object. Saving it again
captures its current state. Fits allocate a small state handle and R closure;
they do not create or retain an eager serialized copy.

Grouped curves save both their per-stratum objects and the full backend used
by summary, residual, and pseudo-value methods. Extracted curves, selected
multistate curves, and nested Python objects returned through field access
also receive their own state handles.

## Implementation and format

A small C adapter registers an ALTREP raw class and three `.Call` routines
with dynamic lookup disabled. Its serialization hook invokes an R closure
only when R saves the object. The adapter owns ordinary R objects and contains
no fitting or numerical routines; those remain in Rust and Python.

The state bundle has version `1`, a raw Python pickle payload, and an R list of
callbacks. Existing Python/native model reducers preserve the complete fit,
retained formula metadata, penalty state, covariance, and curve engines.
The adapter releases the saved bundle after restoring the live Python object.
R's garbage collector handles the closure and proxy references.

Reticulate's Python wrappers for R functions cannot be pickled directly.
A custom pickler records persistent callback references and returns the
wrappers' `r_object` capsules to R, where reticulate converts them back to
the original R functions. R saves those functions and their environments.
On loading, R-to-Python conversion recreates callable wrappers, and the
unpickler binds their references into the fitted model. This preserves local
custom AFT density, initialization, quantile, and deviance callbacks.
Callback extraction follows reticulate's wrapper representation, tested with
reticulate 1.47.0; an unsupported representation fails serialization instead
of silently dropping a callback.

## Requirements

The package declares R 3.6 or later for the ALTREP raw serialization API.
Runtime verification uses R 4.5.3, survival 3.8-12, reticulate 1.47.0, and
Python 3.14.7. Both the R package and a compatible Python package/native
extension must be installed when the object is used. The pickle payload uses
the installed Python's highest protocol and the package's existing native
binary state format; it is not a portable archival or cross-version format.

Use R's default serialization version `3`. Version `2` is rejected explicitly
because it materializes the handle and would discard the state. Files saved
before this change contain no recoverable model state and produce an error
instructing the caller to recreate the model from its original data.
Only load trusted model files: restoring embedded Python pickle has the same
execution semantics as loading Python pickle directly. R callback environments
retain the ordinary limitations of R serialization for external resources.

## Verification

`test-model-persistence.R` checks ordinary, ridge, P-spline, and frailty Cox
models; ordinary and penalized AFT models with scale strata; coefficients,
covariance, prediction, residuals, retained model frames and matrices; grouped
and extracted multistate curves; diagnostics, comparisons, expected survival,
and Yates methods; local R callbacks; repeated saves and mutations; nested
objects; and explicit legacy/version errors. The installed-package test starts
a separate R process, loads RDS and workspace files before attaching the
package, verifies Python remains uninitialized after reading, exercises model
methods, and saves and loads the restored models again.

`test_r_object_persistence.py` checks the state envelope's complete fitted
methods, current-state capture, callback identity, and rejection of invalid
versions, payloads, and callback references. Existing Python pickle tests
continue to cover native state, model methods, copies, and separate processes.

The full Python suite passes 15,133 tests, with 48 skips and 37 documented
expected differences. The R source archive passes 7,698 checks, including
64 new persistence checks, with zero errors, warnings, or notes. All 16 new
Python cases pass. Pinned Ruff lint/format, Mypy across 47 files, and generated
interfaces pass. The C adapter passes `-Wall -Wextra -Werror` syntax checking
with `-Wno-cast-function-type` for R's required routine-registration casts.
Rust numerical sources and the native extension are unchanged from #697.

## Performance

`scripts/benchmark_r_model_persistence.R` verifies coefficients and covariance
or selected curve summaries against R before measuring complete weighted
formula calls on 50,000 rows. It also measures 1,000 coefficient calls and,
for the new implementation, complete serialization and restoration followed
by coefficient extraction. Each operation has three warmups and seven samples;
explicit garbage collection and input construction are excluded. The previous
revision is `d5ba61a3`, installed into a separate R library and measured with
the same Rust extension and Python numerical code. Both revisions use installed
R packages, including byte compilation.
Stock R is measured in a separate process without loading the facade, so its
native S3 methods are preserved. Native compilation and benchmark sampling
do not overlap.

```sh
R_LIBS=/path/to/previous/library Rscript scripts/benchmark_r_model_persistence.R 50000 7 installed
R_LIBS=/path/to/current/library Rscript scripts/benchmark_r_model_persistence.R 50000 7 installed
Rscript scripts/benchmark_r_model_persistence.R 50000 7 stock
```

Local elapsed times in milliseconds, shown as median (minimum–maximum):

| Operation | Previous installed facade | Current installed facade | Stock R |
| --- | ---: | ---: | ---: |
| Cox fit | 103 (101–106) | 101 (100–103) | 115 (114–116) |
| AFT fit | 91 (90–91) | 87 (87–88) | 87 (87–89) |
| Grouped KM fit | 80 (80–81) | 80 (79–81) | 1,303 (1,249–1,365) |
| 1,000 Cox coefficient calls | 58 (57–61) | 56 (55–57) | 1 (1–1) |
| 1,000 AFT coefficient calls | 55 (53–57) | 59 (58–130) | 1 (1–1) |

Cox serialization takes 35 ms (34–35) for a 6,087,622-byte payload;
unserialization followed by coefficient extraction takes 7 ms (7–7).
AFT serialization takes 10 ms (9–10) for 3,502,409 bytes, and restoration
followed by coefficient extraction takes 3 ms (3–4). These calls have valid
model state, whereas previous R serialization retained an unusable pointer.
Grouped curves retain multiple snapshots and can produce larger saved files.

These separate-process measurements do not establish a fitting speedup.
An earlier current run measured 99/89/77 ms for fitting and 56/63 ms for the
two coefficient batches. AFT coefficient batches included a single high
sample in both current runs, retained in the reported ranges. Field access
continues to pay for reticulate crossings and restoration checks; stock R's
direct list extraction is much faster.
