# Population and validation reports

`survival.r` provides structured reports for person-years tables, survival-data
consistency checks and population marginal means. Reports retain numeric
values separately from formatted text; construction does not write to stdout.

```python
from survival import datasets, r

lung = datasets.load_lung()
years = r.pyears("Surv(time, status) ~ sex", lung)
print(r.print_pyears(years))
checks = r.survcheck("Surv(time, status) ~ 1", lung, id=list(range(len(lung["time"]))))
print(r.print_survcheck(checks))
fit = r.coxph("Surv(time, status) ~ age + sex", lung)
means = r.yates(fit, "age", levels=[40, 60, 80], test="pairwise")
report = r.print_yates(means)
print(report)
estimates = r.as_data_frame(report)
tests = report.tables["tests"]
```

`lines` holds the displayed lines; `str(report)` adds a final newline. Original
R calls and trailing whitespace are omitted. Options are explicit and do not
change global print settings. Reports and data-frame columns are independent
of their original results.

## Person-years

`print_pyears(result)` returns `ModelPrint` with a single-row `totals` table
and statistics for tabulated person-years, person-years outside the table,
observations and, when available, events. Both array and data-frame result
layouts are supported. Formatting uses seven significant figures as in R.
The `na_action` record supplies the missing-observation notice.

Results fitted against `survexp.us`, `survexp.usr` or `survexp.mn` now retain
`summary`: R's age range, male/female counts, entry-date range and, for the
race-specific table, white/black counts. These counts describe the matched
rows after subset and omission; they are not weighted population totals.
Matching precedes the US tables' birthday adjustment, so dates describe the
original entries. The Rust `RateTable.source` property identifies the built-in
table and survives native cloning. Tables constructed from attributes have
`source=None` and no built-in match summary; this interface does not execute
arbitrary R summary callbacks.

The summary uses the same matched positions passed to the numerical kernel;
it does not repeat matching or copy the rate table. Totals traverse stored
cells without flattening multidimensional arrays. A reproducible comparison
on existing 512 × 512 nested lists measured 792 bytes of peak temporary memory
versus 2,098,028 bytes for flattening, with median times of 24.35 ms and
22.69 ms respectively. The comparison isolates the total reducer and excludes
input storage; the reduced allocation has a modest runtime cost here.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_population_report_totals.py
```

## Survival-data consistency

`print_survcheck(result, width=80)` returns `ModelPrint` with `transitions`
and optional `events` tables. The introduction reports unique subjects,
observations, transitions and removed rows. Tables preserve R's dimension
titles, column-block wrapping and zero-count rows. Problem notices report
subject and row counts for overlaps, gaps, teleports and jumps. Their original
identifiers and one-based row numbers remain in `statistics["problems"]`.

Raw consistency results retain the established `(censored)` column label.
The report uses the response's censoring label, matching survival 3.8-12;
this change affects both `tables["transitions"]` and its text. `as_data_frame`
returns this transition table, with originating states under `from`.

Report reference tests also exposed a response-parser bug: `factor()` and
`as.factor()` wrappers were removed without creating a categorical status.
They now preserve multistate responses and their labels, including through
`I()`/`identity()`. `as.factor` retains declared unused levels; `factor` drops
them. Supplying already categorical status columns continues to work.

## Marginal means

`print_yates(result, digits=None, dig_tst=None, eps=1e-8, width=80)` returns
`YatesPrint`, a `ModelPrint` subclass with the original typed `estimates`
columns. The display aligns marginal means and standard errors beside tests,
padding the shorter side with blank rows. It supports global, pairwise,
trend and SAS type III tests, numeric or categorical levels, aliased means,
and optional sums of squares from external linear models.

The default estimate precision is five significant figures. `dig_tst`
defaults to `max(1, min(5, digits - 1))`; `eps` controls the small-p-value
threshold. Zero disables threshold replacement. `digits` and `dig_tst` must
be integers from 1 to 22; widths range from 10 to 10,000. P-values are computed
from the stored chi-square and degrees of freedom using the native kernel.

`tables["estimates"]` contains full-precision `pmm`/`std`, while
`tables["tests"]` contains chi-square, degrees of freedom, optional sums of
squares and p-values. `as_data_frame` returns independent estimates with their
level columns and no padded rows. The formatted text does not affect the
underlying numeric values.

## R references

`scripts/generate_population_report_reference.R` regenerates report fixtures
with R 4.5.3 and survival 3.8-12. Cases cover right-censored, counting-process,
multistate and timeline consistency checks; missing rows and each reported
problem type; person-years arrays/data frames, weights, time cuts and all
three built-in rate tables; and marginal means with numeric/joint factors,
pairwise/SAS tests, aliased cells, custom precision and p-value thresholds.
Narrow layouts test repeated dimension headings and column wrapping.
Independent R snapshots check formatting separately from native numerical
results. The generator reuses the existing joint-Yates model specifications.
