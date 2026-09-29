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

`print.summary.survfit` and model coefficient reports remain separate work;
`r.summary_survfit` already returns their underlying numerical summaries.

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
