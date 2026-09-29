# Compact survival reports

`r.print_survfit` returns R's compact curve report as a `SurvfitPrint` object.
Use Python's `print` to display it, or `r.as_data_frame` to get full-precision
columns for another table renderer:

```python
from survival import datasets, r

fit = r.survfit("Surv(time, status) ~ sex", datasets.load_lung())
report = r.print_survfit(fit, rmean="common", digits=3)
print(report)
columns = r.as_data_frame(report)
```

```text
        n events rmean* se(rmean) median 0.95LCL 0.95UCL
sex=1 138    112    326      22.9    270     212     310
sex=2  90     53    461      34.7    426     348     550
    * restricted mean with upper limit =  1022
```

`report.table` is a `NamedMatrix` containing the displayed rows and columns
before rounding. `report.lines` holds the formatted table and footnote;
`str(report)` joins them with a final newline. Construction does not write to
stdout. Editing the report or its converted columns does not change the fit.

## Curves and restricted means

The method accepts Kaplan–Meier, Turnbull, and Cox curves, including grouped,
weighted, delayed-entry and conditional fits. Ordinary curves show record or
subject counts, event counts, medians and fitted median confidence limits.
Redundant count columns are omitted using R's rules. When subject identifiers
are available, both the record and subject counts remain visible, even if equal.

| `rmean` | Ordinary curves | Multistate curves |
| --- | --- | --- |
| Omitted | No mean columns | Common cutoff |
| `"none"` | No mean columns | Counts only |
| `"common"` | Integrate through the latest time across strata | Same cutoff for every state and curve |
| `"individual"` | Each stratum's last time | Each stratum's last time |
| Numeric | Specified cutoff in original time units | Specified cutoff in original time units |

`print_rmean=True` is the legacy shortcut for ordinary curves' `rmean="common"`.
An explicit `rmean` takes precedence. `scale` divides reported times, mean
standard errors and the footnote's cutoff; it must be finite and positive.
Partial string matches such as `rmean="ind"` work as in R.

Multistate inputs dispatch to `r.print_survfitms`, which also works as an
explicit method. It reports one row per state and curve, ordered with strata
varying fastest, then Cox prediction rows, then states. Columns contain the
cohort size, events entering the state, restricted mean time in that state,
and its standard error when available. R marks the last column with an asterisk;
the port preserves that label even when the last column is `se(rmean)`.

Multistate integration includes the origin, so cutoffs before the first
observation are valid if they do not precede the origin or conditional start.
Conditional Cox curves also accept cutoffs between `start_time` and their first
observation, following R's mean calculation. Ordinary curves without a stored
conditional start reject cutoffs before their first fitted time.

## Text layout and differences

`digits` controls significant-digit formatting, with R's defaults of 3 for
ordinary curves and 7 for multistate curves. Each numeric column chooses fixed
or scientific notation independently. `width` defaults to 80 and wraps columns
into blocks while repeating row labels. Missing values display as `NA`.

These functions return a report object rather than R's invisible original fit.
Original R call expressions and omission notices are not reconstructed from
Python result metadata. Trailing spaces are removed, and Python has no global
R print options: pass options explicitly. The numerical kernels retain the
[documented R corrections](r-compatibility.md), including stratum-specific event
counts for multistate Cox curves with several prediction rows.

## Detailed and expected-survival tables

`r.print_summary_survfit` formats the event-time or requested-time rows from
`r.summary_survfit`. It dispatches multistate summaries to
`r.print_summary_survfitms`; both return a `SurvivalTablePrint` object.

```python
summary = r.summary_survfit(fit, times=[100, 300, 600])
detail = r.print_summary_survfit(summary, digits=4)
print(detail)
rows = r.as_data_frame(detail)
```

The report holds one `NamedMatrix` in `tables` per entry in `groups`; an
ungrouped table uses `None` as its label. `as_data_frame` combines groups into
full-precision columns and adds `strata` when groups exist. Each table's values
and labels are independent of the source summary.

Ordinary summaries show time, risk and event counts, survival, and available
standard errors and confidence limits. Several Cox prediction curves receive
separate survival columns; as in R, their confidence columns are omitted from
this report. Counting-process summaries with entry counts also show censor
counts. Multistate reports show total risk and event counts plus one probability
column per state. A one-state result includes its available confidence columns;
multistate Cox predictions form separate `data 1`, `data 2`, ... groups.

Strata print in separate blocks. R's one-row blocks use named-vector layout
with shared precision across values; the port preserves that behavior while
keeping a matrix in `tables`. Empty ordinary strata retain their headers.
An entirely event-free summary raises a clear error: request
`summary_survfit(..., censored=True)` to include censoring observations.

`r.print_survexp` formats expected curves, and `r.print_summary_survexp` formats
their selected-time summaries. Group names on expected-survival objects name
columns, so one table contains the time grid, risk counts and probabilities.
Both return `SurvivalTablePrint`, use three digits by default, and accept
`width`. Expected-curve printing also accepts a positive `scale` and `naprint`.
Like R, `naprint=False` drops rows having at least the number of columns minus
two missing cells; `True` keeps them. Expected-summary printing retains missing
cells. Original call expressions, omission notices and rate-table prose are
not reconstructed.

```python
reference = r.coxph("Surv(time, status) ~ age + sex", datasets.load_lung())
expected = r.survexp("~sex", datasets.load_lung(), ratetable=reference, times=[100, 300, 600])
print(r.print_survexp(expected))
print(r.print_summary_survexp(r.summary_survexp(expected, times=[0, 200, 500])))
```

Native summary objects retain their response `type`, complete `strata_levels`,
and conditional `start_time` for reporting. `start_time` uses the same units as
the scaled summary times. This fixes R 3.8-12's report filtering, which compares
scaled times with an unscaled cutoff and can drop all valid rows. Requested
times before a multistate conditional start are handled by the existing native
summary selector; R can produce mismatched time/probability lengths and fail
when printing them. Expected-summary reports retain each curve's own risk counts,
including when requested times are supplied.

`scripts/generate_detailed_report_reference.R` records 58 R cases, including
56 numeric/text outputs and two explicit failures. Tests compare all displayed
tables and text, then exercise native summary construction, conditional scaling,
empty groups, data-frame conversion, copies and serialization. Report preparation
groups rows in one pass, sums multistate counts in NumPy, and filters expected
curve missingness before creating Python output rows. It does not refit models.

[Cox coefficient and model-summary reports](cox-model-reports.md) are also
available. Other model and test-statistic reports remain separate work.

## Validation and allocation cost

`scripts/generate_survfit_print_reference.R` records both the exact matrix passed
to R's printer and its rendered lines. Its 58 cases cover the curve types above,
confidence levels, restricted means and early cutoffs, origins, unit scaling,
precision and narrow output widths. Python tests compare full-precision tables
and exact text after removing trailing spaces, and check object independence.

Reports call Rust's compact mean and median kernels directly. They do not create
the per-observation arrays returned by `summary_survfit`. In a local Linux x86-64
release build with Python 3.14.7, median times over nine calls were:

| Observation times | Compact report | Full summary | Report Python peak | Summary Python peak |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.042 ms | 0.096 ms | 6.1 KB | 224 KB |
| 10,000 | 0.105 ms | 1.05 ms | 6.1 KB | 2.20 MB |
| 100,000 | 0.856 ms | 14.7 ms | 6.1 KB | 21.9 MB |

Both paths include restricted means. Fitting is excluded. The full summary also
returns event-time data, so this comparison measures the work avoided when only
the compact table is wanted. Memory figures are Python allocations measured by
`tracemalloc`, excluding native allocations and the already fitted model.

```sh
PYTHONPATH=python python scripts/bench_survfit_print.py
Rscript scripts/generate_survfit_print_reference.R
```
