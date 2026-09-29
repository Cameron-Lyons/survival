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
Turnbull, Cox, Aalen–Johansen, transition-matrix, multistate Cox, and expected-survival curves.
The return value holds `axes`, estimate `lines`, `confidence` artists,
censor `marks`, and numerical `data`. Functions never call `show()` or change
the graphics backend. Pass `ax=` to use an existing subplot. Adding lines or
points preserves the existing axis scales and limits.

The module imports Matplotlib only when rendering. `survfit_plot_data` needs
only NumPy and exposes arrays for another renderer: times, estimates, limits,
event-time masks, and censor coordinates. `curve.step()` gives a compressed
right-continuous path; `data.xend` and `data.yend` give curve endpoints.
Preparing or editing these arrays does not mutate the fitted model.

## Responses, expected survival, and method dispatch

`plotting.plot`, `plotting.lines`, and `plotting.points` select the method from
the input object. Explicit method names remain available:

| Input | `plot` | `lines` | `points` |
| --- | --- | --- | --- |
| Raw `r.Surv` response | `plot_surv` | Refused, as in R | Refused, as in R |
| Fitted survival curves | `plot_survfit` | `lines_survfit` | `points_survfit` |
| `r.survexp` curves | `plot_survfit` | `lines_survexp` | `points_survfit` |
| `r.aareg` model | `plot_aareg` | `lines_aareg` | Unsupported |
| `r.cox_zph` diagnostics | `plot_cox_zph` | Unsupported | Unsupported |
| `r.Surv2` response | Refused, as in R | Refused, as in R | Refused, as in R |

`plot_surv` first calls the existing Rust-backed `survfit` with its defaults,
then plots the result. It accepts right-, left-, interval-censored, counting-process
and multistate responses when no additional fitting arguments are needed.
Missing rows are omitted. Keywords control the plot; to change fitting options
or supply an identifier, call `r.survfit` explicitly first.

```python
response = r.Surv(lung["time"], lung["status"])
observed = plotting.plot(response, conf_style="band", label="Observed")
reference = r.coxph("Surv(time, status) ~ age + sex", lung)
expected = r.survexp("~1", lung, ratetable=reference, times=[1, 100, 300, 600, 1000])
plotting.lines(expected, ax=observed.axes, colors="darkorange", label="Cox reference")
observed.axes.legend()
```

Expected-survival objects retain their group names and share one time grid
across columns. They contain no fitted standard errors or observed event/censor
counts, so bands and automatic censor marks are absent. Explicit intervals raise
an error; explicit marker times work. Event-only points are empty; `censor=True`
plots every stored time, including the origin. Individual `survexp` methods return
vectors, which are not curve objects.

`lines_survexp` joins estimates with straight lines, as R does. The generic
`lines` selects this default; explicit `lines_survfit` still draws steps.
Override either with `drawstyle="default"`, `"steps-post"`, `"steps-pre"`, or
`"steps-mid"`. Plotting expected survival uses steps by default. Expected
cumulative hazard is `-log(surv)`, computed in NumPy for `fun="cumhaz"` and
`cumhaz=True`; the latter is an extension to R's expected-survival graphics.

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
Out-of-range NA censor marks left by R after `xmax` are omitted.
See also [Cox diagnostic plots](cox-diagnostic-plotting.md) and
[Aalen coefficient plots](aalen-plotting.md).

## Validation and performance

`scripts/generate_survfit_plot_reference.R` records coordinates from R's actual
plot/lines/points methods. Its 46 cases cover grouped, weighted, delayed-entry,
conditional, interval-censored, Cox and multistate curves, confidence methods,
tied censoring, truncation, transformations and terminal zero survival. Tests
compare coordinates, inspect artists, and export SVG, PNG and PDF figures.
`scripts/generate_response_plot_reference.R` adds 32 cases invoking R's response
and expected-survival methods, including grouped population and Cox references,
Hakulinen and conditional estimates, linear overlays, and the absence of events.

Preparing fitted curves does not refit models or copy observation-by-time influence matrices.
It skips confidence-array conversion when intervals are disabled. Constant runs
are compressed for right-continuous lines and shaded bands, so rendered path
sizes depend on the number of changes rather than the number of censored
observations. Other draw styles retain all coordinates to preserve their geometry.

```sh
PYTHONPATH=python python scripts/bench_survfit_plot.py --render
PYTHONPATH=python python examples/survival_curves.py /tmp/survival-curves.svg
PYTHONPATH=python python examples/expected_survival.py /tmp/expected-survival.svg
```

On Linux x86-64 with Python 3.14.7, five runs at 100,000 observation times took
6.7–7.4 ms to prepare data and paths, and 31–40 ms for figure creation and Agg
rendering. A constant curve used two vertices; an event every 20 observations
used 10,001. These local timings exclude fitting and are not comparisons with R.
