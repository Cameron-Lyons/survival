# Survival curve graphics

Install the optional renderer with `pip install 'survival[plot]'`:

```python
from survival import datasets, plotting, r

lung = datasets.load_lung()
fit = r.survfit("Surv(time, status) ~ sex", lung)
result = plotting.plot_survfit(
    fit, conf_int=True, conf_style="band", mark_time=True,
    xscale=365.25, xlabel="Years", colors=["navy", "darkorange"],
)
result.axes.figure.savefig("survival.svg")
```

`plot_survfit`, `lines_survfit`, and `points_survfit` accept Kaplan–Meier,
Turnbull, Cox, Aalen–Johansen, transition-matrix, and multistate Cox curves.
The return value holds `axes`, estimate `lines`, `confidence` artists,
censor `marks`, and numerical `data`. Functions never call `show()` or change
the graphics backend. Pass `ax=` to use an existing subplot. Adding lines or
points preserves the existing axis scales and limits.

The module imports Matplotlib only when rendering. `survfit_plot_data` needs
only NumPy and exposes arrays for another renderer: times, estimates, limits,
event-time masks, and censor coordinates. `curve.step()` gives a compressed
right-continuous path; `data.xend` and `data.yend` give curve endpoints.
Preparing or editing these arrays does not mutate the fitted model.

## Transformations and selections

| Option | Ordinary survival curves | Multistate curves |
| --- | --- | --- |
| Default / `fun="identity"` | Survival probability | State probability |
| `fun="event"` | One minus survival | State probability |
| `fun="cumhaz"` / `cumhaz=True` | Fitted cumulative hazard | Transition hazards |
| `fun="cloglog"` | `log(-log(S))` | `log(-log(1 - p))` |
| `fun="pct"` | Survival percent | State probability percent |
| `fun="log"` | Survival on a log y axis | Log state probability on a linear axis |
| `fun="logpct"` | Survival percent on a log y axis | Not supported |

`cloglog` also selects a logarithmic time axis. `log=True`, `"x"`, `"y"`,
or `"xy"` selects axis scales explicitly. A callable `fun` receives a NumPy
vector and must return one value per input; it transforms estimates and each
confidence limit separately. Decreasing transforms retain the lower/upper
limits' original identities even when their numeric order reverses.

`noplot="(s0)"` hides the initial multistate state; `noplot=[]` shows every
state. `cumprob=True` accumulates state probabilities. `cumprob=[3, 2]` selects
and accumulates the third and second states in that order; `cumhaz=[2, 1]`
selects transition hazards. These selections are **one-based**, as in R's
graphical arguments. Each selection retains all Cox prediction columns.
Curves have strata varying fastest, then prediction rows, then states/transitions.

## Confidence intervals, marks, and axes

`conf_int` defaults to fitted bands for one curve and false for multiple curves.
Use `True` for grouped intervals, `False` or `"none"` to hide intervals,
`"only"` to hide estimates, or a level strictly between zero and one to
recompute limits. Stored limits retain their fitted method; `conf_type`
applies when calculating new limits through the Rust confidence-limit kernel.
State sums with `cumprob` cannot display intervals.

Intervals use dashed lines by default; `conf_style="band"` shades between them.
`conf_times=[...]` instead draws interval bars. Like R, bars use the next fitted
limit between observation times, whereas explicit `mark_time` coordinates use
the preceding estimate. `conf_offset` and `conf_cap` are fractions of the visible
time range. `mark_time=True` marks censoring, at the midpoint of a jump when
events and censoring coincide. `points_survfit` plots event times;
`censor=True` includes every stored time.

`xscale` divides displayed times; `yscale` multiplies displayed estimates.
Limits, `xmax`, marker times, and confidence times remain in original units.
Repeat the scalings for overlays. Matplotlib controls ticks, padding and fonts.
`colors` and `linestyles` recycle over curves; `linewidth`, `marker`, and
`markersize` control line/mark appearance. Other line properties, such as
`alpha`, `label`, and `rasterized`, pass to
[`Axes.plot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html).

On a logarithmic probability axis, zero values use R's display floor: 80% of
the smallest positive plotted estimate or confidence bound. Fitted values
remain unchanged. Nonfinite transformed coordinates are omitted from paths.

The port fixes two R 3.8-12 graphics defects. R sets an internal log-axis flag
without updating the plotting call for `fun="log"` and `"logpct"`; the port
uses the requested log probability axis and avoids applying a second logarithm
to log state probabilities. R's multistate point method can index a time vector
with a flattened event matrix; the port marks event times across states.
Out-of-range NA censor marks left by R after `xmax` are omitted. Other graphical
methods, including diagnostic plots, remain unfinished.

## Validation and performance

`scripts/generate_survfit_plot_reference.R` records coordinates from R's actual
plot/lines/points methods. Its 46 cases cover grouped, weighted, delayed-entry,
conditional, interval-censored, Cox and multistate curves, confidence methods,
tied censoring, truncation, transformations and terminal zero survival. Tests
compare coordinates, inspect artists, and export SVG, PNG and PDF figures.

Preparation does not refit models or copy observation-by-time influence matrices.
It skips confidence-array conversion when intervals are disabled. Constant runs
are compressed for lines and shaded bands, so rendered path sizes depend on the
number of changes rather than the number of censored observations.

```sh
PYTHONPATH=python python scripts/bench_survfit_plot.py --render
PYTHONPATH=python python examples/survival_curves.py /tmp/survival-curves.svg
```

On Linux x86-64 with Python 3.14.7, five runs at 100,000 observation times took
6.7–7.4 ms to prepare data and paths, and 31–40 ms for figure creation and Agg
rendering. A constant curve used two vertices; an event every 20 observations
used 10,001. These local timings exclude fitting and are not comparisons with R.
