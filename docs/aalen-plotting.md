# Aalen cumulative coefficient plots

Install `survival[plot]` to render an additive regression model:

```python
from survival import datasets, plotting, r

fit = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", datasets.load_ovarian(), dfbeta=True)
plot = plotting.plot_aareg(fit, var=["age", "ecog.ps"])
plot.axes[0].figure.savefig("aalen.svg")
```

`plot_aareg` accumulates the fitted coefficient increments and draws right-continuous
step curves, with pointwise bands at the cumulative coefficient plus or minus
1.96 standard errors. It creates one panel per coefficient when bands are enabled.
With `se=False`, all selected coefficients share one panel and a legend, as in R.
`lines_aareg` adds curves to one existing axis and preserves its scales and limits;
its bands are disabled by default. Both return `AalenPlot` with `axes`, estimate
`lines`, `confidence` artists, and prepared numerical `data`.

If the fit retains `dfbeta`, the variance increments are sums of squared
per-subject or per-cluster influences at each unique event time. Otherwise, they
are the squared coefficient increments. Each variance curve is the cumulative
sum of these increments. At tied times, individual coefficient increments are
accumulated before keeping the final curve value; they are squared individually
for ordinary bands, rather than squaring their sum. These are R's plotting
conventions and are separate from the fitted model's overall test covariance.

`maxtime` retains all events at or before the cutoff. It does not extend a curve
beyond its final included event. `var` selects coefficient names or one-based
indices and preserves their requested order, for example `var=[3, 2]`.
The intercept is included by default.

Pass existing axes through `ax=`: one per coefficient with bands, or one for a
plot without bands or an overlay. `colors` recycles across coefficients.
`linewidth`, `xlabel`, `ylabel`, and `legend` control appearance. Other line
properties pass to Matplotlib, including markers and alpha. The default
`drawstyle="steps-post"` corresponds to R's `type="s"`; use `"default"` for
straight lines or `"steps-pre"` for R's `"S"`. Matplotlib's `linestyle="none"`
and a marker produce point-only estimates; bands retain dashed lines. Graphical
controls follow Matplotlib rather than reproducing every R `type` code.
Functions never call `show()` or change the graphics backend.

## Numerical data and memory

`plotting.aareg_plot_data(fit, se=True, maxtime=None, var=None)` works without
Matplotlib. It returns times, coefficient and standard-error matrices, lower
and upper bounds, names, and whether stored influences were used. Matrices have
one row per output time and one column per selected coefficient. Arrays belong
to the result and can be edited without changing the fit.

The fit and influence values come from the Rust Aalen implementation; preparation
uses NumPy reductions. Stored influences are reduced in bounded blocks, avoiding
conversion and squaring of the entire group-by-coefficient-by-time array. The
float64 conversion blocks target approximately 1 MiB; temporary lists and output
arrays require additional memory. For NumPy inputs, slicing retains views when
the dtype permits. `se=False` never reads stored influence values.

Three upstream edge cases are corrected. In R survival 3.8-12, retaining only
the first event can drop matrix dimensions, a cutoff before that event can
include it through `1:0` indexing, and robust bands can misalign when an event
occurs at zero. The port retains matrix dimensions, returns only the zero
origin before the first event, and removes a duplicate zero origin from every
array consistently. As in R, ordinary output starts at zero even when fitted
times are negative; a fitted event at zero replaces that initial zero value.

## Validation and performance

`scripts/generate_aareg_plot_reference.R` captures calls from R's actual
`plot.aareg` and `lines.aareg` methods. Its 37 coordinate cases cover weighted,
tapered, clustered, tied-event, delayed-entry, single-coefficient and ordinary
fits; cutoff and selection behavior; confidence bands; and overlays. It also
records the three upstream failures separately. Tests check independent
coefficient/variance identities, bounded temporary allocations, optional
rendering, input immutability, and PNG/SVG/PDF exports.

```sh
PYTHONPATH=python python scripts/bench_aareg_plot.py
PYTHONPATH=python python examples/aalen_curves.py /tmp/aalen-curves.svg
```

Local Linux x86-64 measurements with Python 3.14.7 used five runs on 1,000 groups,
three coefficients, and 1,000 event times. Influence reduction from NumPy took
2.54 ms with 0.22 MB peak temporary allocation, compared with 3.24 ms and 24.03 MB
for a fully squared array. From fitted nested lists, it took 44.8 ms and 2.10 MB,
compared with 49.4 ms and 48.03 MB for full conversion and squaring. The benchmark
checks equal variance values; timings exclude fitting and rendering.
