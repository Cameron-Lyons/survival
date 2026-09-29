# Cox proportional-hazards diagnostic plots

Install `survival[plot]` for Matplotlib rendering:

```python
from survival import datasets, plotting, r

fit = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
diagnostic = r.cox_zph(fit)
plot = plotting.plot_cox_zph(diagnostic)
plot.axes[0].figure.savefig("cox-diagnostics.svg")
```

Each term gets a subplot with its scaled Schoenfeld residuals, a natural-spline
estimate of the time-varying coefficient, and bands two standard errors above
and below the estimate. These use the residual variance from `cox_zph`, rather
than estimating another residual variance from the smoother. A horizontal
coefficient curve is consistent with the proportional-hazards assumption;
the numerical tests remain available in `diagnostic.table`.

`df=4` specifies the spline basis dimension and `nsmo=40` specifies its evenly
spaced prediction times. Both must be integers of at least two. As in R,
knots are computed using the event times **and** the prediction grid, so changing
`nsmo` can change the fitted curve. Terms with a singular spline design warn and
are skipped. Missing residuals are excluded separately for each term; their
original event times remain in the common knot calculation.

Use `var="age"`, `var=["sex", "age"]`, or one-based column indices such as
`var=[2, 1]` to select and order terms. `resid=False` hides the residual points;
`se=False` omits the bands. `hr=True` exponentiates the coefficients and residuals
and uses a logarithmic hazard-ratio axis. Identity time remains linear, log
time is converted back to original units on a logarithmic axis, and other
transforms receive R's eight rounded original-time labels.

Pass one Matplotlib axis per selected nonsingular term using `ax=`. Without
axes, the function creates a vertical stack of subplots. `colors` and
`linestyles` set the estimate and band styles, recycling a single value;
`linewidth`, `marker`, `markersize`, `xlabel`, and `ylabel` control their
appearance. Additional line properties, such as `alpha`, apply to the spline
and bands. Matplotlib controls padding, tick formatting and other layout details.
The function never calls `show()` or changes the graphics backend.

The returned `CoxDiagnosticPlot` contains tuples of `axes`, estimate `lines`,
`confidence` lines, and `residuals` artists, plus the prepared numerical `data`.
If every selected spline is singular, these tuples are empty and no figure is
created. Existing axes must match the number of curves that can be drawn.

## Numerical use and the Rust kernel

`plotting.cox_zph_plot_data(diagnostic, ...)` returns the same curve arrays,
residual coordinates, scale flags, tick positions and skipped-term names
without importing Matplotlib. Unlike R's `plot=FALSE` branch, it returns all
selected nonsingular terms and also works with standard errors disabled.
Output arrays can be edited without changing the fitted model or other curves.

The native Rust function is exported as
`survival::regression::cox_zph_smooth(x, y.view(), variance, df, nsmo, se)`;
its Python binding is `survival.regression.cox_zph_smooth`. Here `x` is transformed
death time, `y` is the death-by-term residual matrix, and `variance` contains
the diagonal of the diagnostic residual variance matrix. The return value holds
transformed prediction times, fitted coefficients, one-standard-error values
when requested, and zero-based indices of singular columns. It does not apply
display transforms or multiply standard errors by two.

The kernel uses the existing natural-spline basis and LINPACK-style QR routines.
It forms the basis once, reuses factorizations and prediction variances for
terms with identical missing-value masks, and uses triangular solves for
standard errors without forming normal equations. The binding accepts NumPy
matrices of different memory layouts and releases the GIL during computation.

## Validation and performance

`scripts/generate_cox_zph_plot_reference.R` captures coordinates and axis labels
from R survival 3.8-12's actual `plot.cox.zph` method. Its 32 cases include
identity, log, rank, KM and custom time transforms; stratified, penalized,
weighted, counting-process and multistate fits; missing and singular terms;
different basis and grid sizes; term selection; and hazard-ratio views.
Tests compare numerical arrays and rendered artists, check a known linear fit
and its analytic variance, and export PNG, SVG and PDF.

```sh
PYTHONPATH=python python scripts/bench_cox_zph_plot.py
PYTHONPATH=python python examples/cox_diagnostics.py /tmp/cox-diagnostics.svg
```

On Linux x86-64 with Python 3.14.7, the median of five calls for 100,000 event
times and 20 terms was 35 ms with one missing-value mask and 33 ms with two
masks. Independent calls for each term took 183 ms and 179 ms respectively.
The benchmark checks that fitted curves and errors agree. These timings include
the Python boundary and exclude model fitting and rendering; they compare batch
and per-term use of the same native kernel, not Rust against R.
