# Printing and graphics

`survival.r` returns R's printed output as data objects, and
`survival.plotting` draws R's survival graphics with an optional Matplotlib
renderer. Differences from R are listed in
[R compatibility](r-compatibility.md#printing-and-graphics).

## Print objects

Every `print_*` function returns an object instead of writing to stdout:

- `lines` holds the formatted text without trailing whitespace, and
  `str(obj)` joins it with a final newline; use `print(obj)` to display it.
- Model and test output uses `ModelPrint` objects. `tables` holds independent
  full-precision `NamedMatrix` values, `statistics` holds scalar results and
  metadata, and `as_data_frame(obj)` returns independent columns from
  `tables[obj.primary_table]`, with row labels under `obj.row_label` (a `term`
  column for coefficient tables). A null Cox model gives an empty mapping.
- Curve output uses `SurvfitPrint` (compact tables) and `SurvivalTablePrint`
  (detailed tables); raw responses and rate tables use `ResponsePrint` and
  `RateTablePrint`.
- Editing a print object or its converted columns never changes the fit or
  summary. Printing reuses existing fits and summaries and never refits a model.
- Options are explicit arguments; no global R print options or locale settings are
  read. R call expressions are not stored by Python fits and are omitted. `width`
  (default 80, 10–10,000) wraps column blocks while repeating row labels, and
  `digits`, where accepted, ranges from 1 to 22.

## Survival curves

`r.print_survfit(fit, rmean=..., digits=..., scale=..., width=...)` returns R's
compact table of sample sizes, events, medians and confidence limits for
Kaplan–Meier, Turnbull and Cox curves, including grouped, weighted,
delayed-entry and conditional fits.

```python
from survival import datasets, r

fit = r.survfit("Surv(time, status) ~ sex", datasets.load_lung())
table = r.print_survfit(fit, rmean="common", digits=3)
print(table)
columns = r.as_data_frame(table)
```

```text
        n events rmean* se(rmean) median 0.95LCL 0.95UCL
sex=1 138    112    326      22.9    270     212     310
sex=2  90     53    461      34.7    426     348     550
    * restricted mean with upper limit =  1022
```

`table.table` is a `NamedMatrix` of the displayed values before rounding.
Redundant count columns are omitted with R's rules; when subject ids are
available, both record and subject counts stay visible.

| `rmean` | Ordinary curves | Multi-state curves |
| --- | --- | --- |
| Omitted | No mean columns | Common cutoff |
| `"none"` | No mean columns | Counts only |
| `"common"` | Integrate through the latest time across strata | Same cutoff for every state and curve |
| `"individual"` | Each stratum's last time | Each stratum's last time |
| Numeric | Specified cutoff in original time units | Specified cutoff in original time units |

For ordinary curves, `print_rmean=True` is the legacy shortcut for
`rmean="common"` (multi-state curves ignore it); an explicit `rmean` takes
precedence, and partial matches such as `"ind"` work. `scale`
divides reported times, mean standard errors and the cutoff and must be finite and
positive. `digits` defaults to 3 for ordinary curves and 7 for multi-state curves;
each column chooses fixed or scientific notation independently; missing values
display as `NA`.

Multi-state curves dispatch to `r.print_survfitms`: one row per state and curve
(strata fastest, then Cox prediction rows, then states) with the cohort size,
events entering the state, restricted mean time and its standard error. R's
asterisk marks the last column even when it is `se(rmean)`. Multi-state
integration includes the origin, so cutoffs before the first observation are
valid unless they precede the origin or conditional start; conditional Cox curves
also accept cutoffs between `start_time` and their first observation, while
ordinary curves without a stored conditional start reject cutoffs before their
first time.

`r.print_summary_survfit(r.summary_survfit(fit, times=[100, 300]))` formats
detailed rows and dispatches multi-state summaries to `print_summary_survfitms`.
`tables` holds one `NamedMatrix` per entry in `groups` (`None` for an ungrouped
table), and `as_data_frame` adds `strata` when groups exist.

- Ordinary summaries show time, risk and event counts, survival and available
  standard errors and limits; several Cox prediction curves get separate survival
  columns and, as in R, no confidence columns. Counting-process summaries with
  entry counts also show censor counts.
- Multi-state summaries show total risk and event counts and one probability
  column per state; a one-state result includes its confidence columns, and
  multi-state Cox predictions form `data 1`, `data 2`, ... groups.
- Strata print in separate blocks. One-row blocks use R's named-vector layout with
  shared precision, while `tables` keeps a matrix; empty ordinary strata keep
  their headers.
- An entirely event-free summary raises an error; request
  `summary_survfit(..., censored=True)` to include censoring rows.
- Summary objects keep their response `type`, complete `strata_levels` and the
  conditional `start_time` in the units of the scaled summary times.

`r.print_survexp` and `r.print_summary_survexp` format expected curves and their
summaries in one table with the time grid, risk counts and one probability column
per group. Both use three digits by default; `print_survexp` also accepts a
positive `scale` and `naprint`. As in R, `naprint=False` drops rows with at least
the number of columns minus two missing cells; expected-summary printing keeps
missing cells. Each curve keeps its own risk counts.

## Models

### Cox models

```python
fit = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
coefficients = r.print_coxph(fit)
summary = r.model_summary(fit, conf_int=0.90)
detailed = r.print_summary_coxph(summary, signif_stars=False)
intervals = detailed.tables["conf_int"]
```

- `print_coxph(fit, digits=None, signif_stars=False, width=80)` shows
  coefficients, hazard ratios, standard errors (model-based and robust for robust
  fits), Wald z statistics and p-values, with the likelihood-ratio test, sample
  size, events and missing-row notice. `print_summary_coxph(summary, digits=None,
  signif_stars=True, expand=False, width=80)` adds confidence intervals,
  concordance and the likelihood-ratio, Wald and score tests; clustered fits keep
  the robust score test and R's explanation. The summary's `conf_level` sets the
  interval labels (95% for older summaries). Both default to four significant
  digits and follow R's separate coefficient, test-statistic and p-value
  precision rules. Missing coefficients are `NaN` in tables and `NA` in text.
- A null fit dispatches to `print_coxph_null`, which, as in R, ignores the
  precision argument and uses seven digits; `model_summary(null_fit)` returns the
  fit itself.
- `print_coxph_penal(fit, terms=False, maxlabel=25, digits=None, width=80)`
  (three digits) shows per-term coefficients, both standard errors, chi-square
  tests, effective degrees of freedom and p-values; spline terms split into linear
  and nonlinear rows and frailty terms show random-effect tests. `terms=True`
  combines multi-column terms into one Wald row; `maxlabel` truncates displayed
  labels. `print_summary_coxph_penal(summary, digits=None, signif_stars=True,
  maxlabel=25, width=80)` uses R's character-table layout; as in R it ignores
  `signif_stars` and leaves confidence-interval labels untruncated.
- Multi-state fits group identical coefficient maps under shared transition
  headings, annotate proportional baseline coefficients and show state names and
  subject counts. As in R, `se(coef)` is omitted from that display. With
  `expand=True`, the summary groups coefficients and intervals by transition.

### AFT and Aalen models

```python
fit = r.survreg("Surv(time, status) ~ age + sex", datasets.load_lung())
print(r.print_survreg(fit))
summary = r.model_summary(fit, correlation=True)
detailed = r.print_summary_survreg(summary)
correlation = detailed.tables["correlation"]

additive = r.aareg(
    "Surv(futime, fustat) ~ age + ecog.ps", datasets.load_ovarian(), dfbeta=True
)
print(r.print_aareg(additive, maxtime=400, scale=365.25))
print(r.print_summary_aareg(r.model_summary(additive, test="nrisk")))
```

- `print_survreg(fit, digits=None, width=80)` shows the coefficient vector, fitted
  or fixed scales, both log likelihoods, a likelihood-ratio test when its degrees of
  freedom are positive, and the sample size; aliased coefficients get R's
  singularity notice. The default is seven digits; as in R, an explicit `digits`
  changes the coefficient vector and multiple named scales, while scalar scales and
  likelihood text keep seven digits.
- `print_summary_survreg(summary, digits=None, signif_stars=False, width=80)` adds
  standard errors, z statistics, p-values, the distribution, iteration counts and
  optional correlations (three digits). Robust fits show robust and naive standard
  errors. With `correlation=True`, the lower triangle is displayed, the full matrix
  kept and aliased coefficients excluded. Summaries carry `fixed_scale` and
  `scale_names`.
- Penalized fits dispatch to `print_survreg_penal`, returning `SurvregPenalPrint`
  with full-precision `rows` and `columns`, `width` and `as_data_frame`; it accepts
  compact and full fits and includes spline linear and nonlinear tests, dense and
  sparse frailty Wald tests, search summaries, scales, iteration counts and
  likelihood-ratio tests. Call it explicitly for its `terms=True` and `maxlabel`
  controls. A summary of a penalized fit uses `print_summary_survreg`, as in R.
- `print_aareg(fit, maxtime=None, test=None, scale=1, width=80)` shows the sample
  size, used and total unique event times, slopes, weighted tests, standard errors
  (robust when the fit keeps influences), z statistics, p-values and the overall
  chi-square test. `maxtime` includes events at or before the cutoff; `test` selects
  `aalen` or `nrisk` weights and defaults to the fit's test; `scale` changes the
  displayed weighted coefficients and errors. Changing `maxtime` or `test`
  recomputes the robust covariance from the stored influences. Zero scale and
  cutoffs before all events raise errors; a `test="variance"` fit needs an explicit
  supported test. As in R, the values are rounded to three significant figures and
  then formatted at seven digits, with no precision override, and the direct
  printer's footer keeps the fit's original test label when an override was
  supplied; `statistics["test"]` records the computed test.
  `print_summary_aareg(summary, width=80)` shows the same table and test without the
  sample and event-count introduction, labeled with the computed test.

### Diagnostics and tests

```python
lung = datasets.load_lung()
fit = r.coxph("Surv(time, status) ~ age + sex", lung)
print(r.print_cox_zph(r.cox_zph(fit)))
print(r.print_concordance(r.concordance(fit)))
logrank = r.print_survdiff(r.survdiff("Surv(time, status) ~ sex", lung))
```

| Function | Primary table | Default precision | Other content |
| --- | --- | --- | --- |
| `print_cch(fit)` | `coefficients` | Seven significant figures | Cohort/subcohort sizes |
| `print_summary_cch(model_summary(fit))` | `coefficients` | Three decimal places, then seven-digit layout | Hazard ratios and 1.96-SE confidence limits |
| `print_clogit(fit)` | `coefficients` | Four significant figures | Existing Cox model footer |
| `print_cox_zph(result)` | `tests` | Three significant figures | Term and optional global tests |
| `print_concordance(result)` | `concordance` | Four significant figures for scalar estimates; four decimal places for multiple estimates | Standard errors, pair counts, sample size |
| `print_survConcordance(result)` | `counts` | Seven significant figures | Legacy estimates, counts and sample size |
| `print_survdiff(result)` | `test` | Three significant figures | Chi-square, degrees of freedom, p-value |

`print_clogit`, `print_cox_zph`, `print_concordance` and `print_survdiff` accept
`digits`; the first two also accept `signif_stars=False`.

- Case-cohort output covers all five estimators and robust Lin–Ying fits;
  `tables["sizes"]` keeps named strata, and summaries keep overflowing hazard
  ratios as infinity. As in R, the summary printer ignores `digits`, and the text
  keeps R's literal `x$method,` heading and single-coefficient row labels (`[1,]`
  and `Value`); numeric tables keep the coefficient name.
- Concordance output keeps `tables["counts"]` with one row or separate
  predictor/stratum rows. Displayed counts are rounded to two decimals before R's
  seven-digit formatting; stored counts are unrounded. A missing variance omits
  the standard error, and no comparable pairs display `NaN`. Formula results keep
  `na_action`. `print_survConcordance` is explicit because R 3.8-12 no longer
  registers `print.survConcordance`; the legacy Python estimator keeps its
  deprecation warning, and printing its result adds none.
- Survival-difference output covers log-rank, G-rho, stratified and one-sample
  tests. Stratified counts are summed for display; zero denominators keep R's
  `NaN` and infinity; one-sample tests show observed and expected events, the
  signed z statistic and the p-value. Formula and direct-response results keep
  their omission records.

### Population and validation

```python
years = r.pyears("Surv(time, status) ~ sex", lung)
print(r.print_pyears(years))
checks = r.survcheck("Surv(time, status) ~ 1", lung, id=list(range(len(lung["time"]))))
print(r.print_survcheck(checks))
means = r.yates(fit, "age", levels=[40, 60, 80], test="pairwise")
marginal = r.print_yates(means)
estimates = r.as_data_frame(marginal)
```

- `print_pyears(result)` returns a `totals` table and statistics for tabulated
  person-years, person-years outside the table, observations and events, for array
  and data-frame results, at seven significant figures. Results fitted against
  `survexp.us`, `survexp.usr` or `survexp.mn` keep `summary`: R's age range,
  male/female counts, entry-date range and, for the race table, white/black counts
  of the matched rows. Tables constructed from attributes have no summary;
  `RateTable.source` identifies a built-in table.
- `print_survcheck(result, width=80)` returns `transitions` and optional `events`
  tables with R's dimension titles and zero-count rows; problem notices count
  overlaps, gaps, teleports and jumps, and their ids and one-based rows are in
  `statistics["problems"]`. The printed table labels the censoring column with the
  response's censoring label; `as_data_frame` returns the transition table with
  originating states under `from`.
- `print_yates(result, digits=None, dig_tst=None, eps=1e-8, width=80)` returns
  `YatesPrint` with typed `estimates` columns and supports global, pairwise, trend
  and SAS type III tests. Estimates use five significant figures; `dig_tst`
  defaults to `max(1, min(5, digits - 1))`; `eps` sets the small-p-value threshold
  (zero disables it). `tables["estimates"]` holds `pmm` and `std`,
  `tables["tests"]` the chi-square, degrees of freedom, optional sums of squares
  and p-values; `as_data_frame` returns estimates without display padding.

## Responses and rate tables

```python
response = r.Surv([1.111111111, 2, 3], [1, 0, 1])
printed = r.print_surv(response, names=["first", "second", "third"], width=40)
rates = r.print_ratetable(r.survexp_us(), digits=5, max_print=6)
assert rates.displayed == 6
assert len(rates.rates) == 17820
```

`max_print` defaults to 99,999 and accepts 0 to 2,147,483,646 (R's `max` spelling
also works).

- `print_surv(response, quote=False, right=False, names=None, width=80,
  max_print=99999)` and `print_surv2` keep all labels from `as_character_surv` in
  `labels`, formatting times at seven significant figures. `quote=True` adds double
  quotes and escapes; layout accounts for wide and combining characters. Unnamed
  vectors use one-based indexes; named vectors right-align names and values, as R
  does even with `right=False`. `names` supplies one label per row. `data` and
  `as_data_frame` keep full-precision time and status columns. Truncation affects
  text only. An empty response displays `character(0)`.
- `print_ratetable(table, digits=None, width=80, max_print=99999)` uses seven
  significant figures by default and displays named vectors, matrices or, for more
  dimensions, matrix slices in R's first-index order, with `dimid` as axis titles.
  The object owns `dims`, `dimid`, `dimnames` and `rates`; `as_data_frame` expands
  each dimension's labels and a `rate` column, with R's `make.unique` suffixes for
  repeated names. Display limits follow R: vectors print every entry when only one
  would be omitted, matrices keep complete rows and arrays complete slices; hidden
  slices are never formatted. R's zero-limit matrix display keeps row labels and
  named Unicode axis titles keep R's byte-counted padding. `displayed` counts the
  numeric or response entries actually shown.

## Graphics

Install the renderer with `pip install 'survival[plot]'`. Plot functions never
call `show()` or change the backend; pass `ax=` to draw on existing axes, and
added lines and points keep the existing scales and limits. Matplotlib controls
layout, ticks, padding and fonts; R's base graphics parameters are not emulated.

`plotting.plot`, `plotting.lines` and `plotting.points` dispatch on the input:

| Input | `plot` | `lines` | `points` |
| --- | --- | --- | --- |
| Raw `r.Surv` response | `plot_surv` | Refused, as in R | Refused, as in R |
| Fitted survival curves | `plot_survfit` | `lines_survfit` | `points_survfit` |
| `r.survexp` curves | `plot_survfit` | `lines_survexp` | `points_survfit` |
| `r.aareg` model | `plot_aareg` | `lines_aareg` | Unsupported |
| `r.cox_zph` diagnostics | `plot_cox_zph` | Unsupported | Unsupported |
| `r.Surv2` response | Refused, as in R | Refused, as in R | Refused, as in R |

### Survival curves

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

`plot_survfit`, `lines_survfit` and `points_survfit` accept Kaplan–Meier,
Turnbull, Cox, Aalen–Johansen, transition-matrix, multi-state Cox and
expected-survival curves. The result holds `axes`, estimate `lines`,
`confidence` artists, censor `marks` and numerical `data`.
`plotting.survfit_plot_data` needs only NumPy and returns times, estimates,
limits, event masks and censor coordinates; `curve.step()` gives a compressed
right-continuous path, and `data.xend` and `data.yend` the endpoints.

`plot_surv` fits a raw response with `survfit`'s defaults before plotting
(right, left, interval, counting and multi-state responses); missing rows are
omitted, and other fitting options require an explicit `r.survfit` call.
Expected-survival curves share one time grid and have no bands or automatic
censor marks; explicit intervals raise an error, while explicit marker times
work. Event-only points are empty, and `censor=True` plots every stored time,
including the origin. `lines_survexp` joins estimates
with straight lines, as R does, while `plot_survfit` and `lines_survfit` draw
steps; override with `drawstyle`. Expected cumulative hazard is `-log(surv)`, for
`fun="cumhaz"` and also for `cumhaz=True`, an extension to R.

| Option | Ordinary survival curves | Multi-state curves |
| --- | --- | --- |
| Default / `fun="identity"` | Survival probability | State probability |
| `fun="event"` | One minus survival | State probability |
| `fun="cumhaz"` / `cumhaz=True` | Fitted cumulative hazard | Transition hazards |
| `fun="cloglog"` | `log(-log(S))` | `log(-log(1 - p))` |
| `fun="pct"` | Survival percent | State probability percent |
| `fun="log"` | Survival on a log y axis | Log state probability on a linear axis |
| `fun="logpct"` | Survival percent on a log y axis | Not supported |

- `cloglog` also selects a log time axis; `log=True`, `"x"`, `"y"` or `"xy"`
  select scales explicitly. A callable `fun` receives a NumPy vector and must
  return one value per input; it transforms estimates and each limit separately,
  and a decreasing transform keeps the lower and upper limits' identities.
- `noplot="(s0)"` hides the initial multi-state state and `noplot=[]` shows every
  state; `cumprob=True` accumulates
  state probabilities; `cumprob=[3, 2]` and `cumhaz=[2, 1]` select **one-based**
  states or transitions in that order. Curves have strata fastest, then
  prediction rows, then states.
- `conf_int` defaults to bands for one curve and none for several; `True` shows
  intervals for grouped curves, `False` or `"none"` hides them, `"only"` hides
  estimates, and a level between zero and one recomputes limits with `conf_type`.
  Stored limits keep their fitted method.
  `cumprob` sums cannot show intervals. Intervals are dashed by default;
  `conf_style="band"` shades them and `conf_times=[...]` draws bars, which, as in
  R, use the next fitted limit, while explicit `mark_time` coordinates use the
  preceding estimate. `conf_offset` and `conf_cap` are fractions of the time
  range. `mark_time=True` marks censoring at the midpoint of a jump when events and
  censoring coincide; `points_survfit` plots event times, and `censor=True` every
  stored time.
- `xscale` divides displayed times and `yscale` multiplies estimates; limits,
  `xmax` and marker times stay in original units. `colors` and `linestyles`
  recycle over curves; other line properties pass to Matplotlib's `Axes.plot`.
- On a log probability axis, zero values are drawn at R's floor of 80% of the
  smallest positive plotted value. Nonfinite coordinates are omitted.

### Aalen cumulative coefficients

```python
fit = r.aareg("Surv(futime, fustat) ~ age + ecog.ps", datasets.load_ovarian(), dfbeta=True)
plot = plotting.plot_aareg(fit, var=["age", "ecog.ps"])
```

`plot_aareg` draws right-continuous cumulative coefficients with pointwise bands at
plus or minus 1.96 standard errors, one panel per coefficient when bands are shown;
with `se=False` all coefficients share one panel and a legend. `lines_aareg`
overlays curves on one axis without bands by default. Both return `AalenPlot` with
`axes`, `lines`, `confidence` and `data`. Pass `ax=` with one axis per coefficient
when bands are shown, or one axis without bands or for an overlay. As in R, variance
increments are sums of squared per-subject or per-cluster influences at each event
time when the fit keeps `dfbeta`, and otherwise the squared coefficient increments,
each squared individually at tied times. `maxtime` keeps events at or before the
cutoff; `var` selects names or one-based indices in order, with the intercept
included by default. `drawstyle="steps-post"` corresponds to R's `type="s"` and
`"steps-pre"` to R's `"S"`. Output starts at zero even when fitted times are
negative. `plotting.aareg_plot_data(fit, se=True, maxtime=None, var=None)` returns
the times, coefficient and error matrices (one row per time, one column per selected
coefficient), bounds, names and whether stored influences were used, without
Matplotlib.

### Cox proportional-hazards diagnostics

```python
diagnostic = r.cox_zph(r.coxph("Surv(time, status) ~ age + sex", lung))
plot = plotting.plot_cox_zph(diagnostic)
```

Each term gets a subplot of its scaled Schoenfeld residuals, a natural-spline
estimate of the time-varying coefficient and bands two standard errors wide,
using the residual variance from `cox_zph`. `df=4` sets the spline basis
dimension and `nsmo=40` the number of prediction times (both integers of at least
two); as in R, knots use the event times and the prediction grid, so changing
`nsmo` can change the curve. Terms with a singular spline design warn and are
skipped; missing residuals are excluded per term. `var` selects terms by name or
one-based index; `resid=False` hides residuals; `se=False` omits bands; `hr=True`
exponentiates onto a log hazard-ratio axis, with R's eight rounded time labels for
transforms other than identity and log. The result is a `CoxDiagnosticPlot`;
pass one axis per selected nonsingular term through `ax=`. If every selected
spline is singular, its tuples are empty and no figure is created.
`plotting.cox_zph_plot_data(diagnostic, ...)` returns the curves, residual
coordinates, scale flags, ticks and skipped terms for every selected nonsingular
term, also with standard errors disabled. The smoother is
`survival::regression::cox_zph_smooth(x, y.view(), variance, df, nsmo, se)` in
Rust and `regression.cox_zph_smooth` in Python: `x` is the transformed death time,
`y` the death-by-term residual matrix and `variance` the diagonal of the residual
variance matrix. It returns the transformed prediction times, fitted
coefficients, one-standard-error values when requested and zero-based indices of
singular columns, without display transforms or doubling of the errors. The
binding accepts NumPy matrices of any layout and releases the GIL.
