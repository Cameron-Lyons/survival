# Raw response and rate-table reports

`survival.r` provides structured text reports for `Surv`, `Surv2` and
`RateTable` objects. Reports keep numeric data separately from display labels
and do not write to stdout during construction.

```python
from survival import r

response = r.Surv([1.111111111, 2, 3], [1, 0, 1])
report = r.print_surv(response, names=["first", "second", "third"], width=40)
print(report)
columns = r.as_data_frame(report)
assert columns["time"][0] == 1.111111111

rates = r.print_ratetable(r.survexp_us(), digits=5, max_print=6)
print(rates)
assert rates.displayed == 6
assert len(rates.rates) == 17820
```

`lines` contains the displayed lines without trailing whitespace;
`str(report)` adds a final newline. Widths range from 10 to 10,000.
`max_print` defaults to 99,999 and accepts integers from zero to 2,147,483,646;
R's `max` spelling is also accepted through keyword arguments.

## Responses

`print_surv(response, quote=False, right=False, names=None, width=80,
max_print=99999)` and `print_surv2` return `ResponsePrint` objects.
`labels` contains all response labels from `as_character_surv`, including
censoring marks, interval endpoints and multistate labels. Time formatting
uses seven significant figures, as in R's conversion before printing.
`quote=True` surrounds labels with double quotes. Controls and backslashes
are escaped; layout accounts for wide and combining Unicode characters.
Unnamed vectors use R's one-based row indexes. Named vectors align both
names and values to the right, following R even when `right=False`.

Python responses have no stored row names; the optional `names` argument
supplies one string per response row. `data` and `as_data_frame(report)`
retain full-precision time and status columns, response type, optional second
endpoints and optional names. Truncation affects text only. Data-frame
columns are independent copies.

An empty response displays `character(0)`. R 4.5.3 / survival 3.8-12 instead
fails while converting an empty `Surv`; this interface deliberately allows
empty reports and data frames.

## Rate tables

`print_ratetable(table, digits=None, width=80, max_print=99999)` returns
`RateTablePrint`. Precision defaults to seven significant figures and accepts
integers from 1 to 22. One-dimensional tables display named vectors;
two-dimensional tables display matrices; higher dimensions display matrix
slices in R's first-index-fastest order. `dimid` supplies named axis titles.
The native representation canonicalizes these titles and does not retain
R's older distinction between unnamed axes and a separate `dimid` attribute.

The report owns `dims`, `dimid`, `dimnames` and all full-precision `rates`.
No dense coordinate grid is allocated while rendering. Explicitly calling
`as_data_frame` expands each dimension's labels and a numeric `rate` column.
Repeated names receive `.1`, `.2`, etc. suffixes using R's `make.unique`
rules, including when an axis is already named `rate`.

Display limits follow R's rules, which are not a strict cell cap for vectors:
vectors print all entries when only one would otherwise be omitted. Matrix
limits preserve complete rows where possible; array limits preserve complete
slices followed by available rows. Omission notices distinguish entries,
rows, columns and slices. Matrix numeric precision uses every row of each
visible column, including hidden rows. Unshown slices are never formatted.
R's zero-limit matrix display retains row labels, and named Unicode axis
titles retain R's byte-counted title padding. `displayed` counts numeric or
response entries actually shown.

## Validation and display cost

`scripts/generate_array_report_reference.R` captures R 4.5.3 / survival
3.8-12 output. The fixtures cover right/left/counting/interval and timeline
responses, missing values, state labels, names, escaping and Unicode; custom
one- through four-dimensional rate tables; all built-in tables; narrow
widths, precision and truncation boundaries. Tests also check full-precision
data frames, independent ownership and bounded slice formatting.

`scripts/benchmark_array_reports.py` compares bounded and complete rendering
of the same native tables. Both paths copy and retain every rate. One local
run measured:

| Table | Rates retained | Bounded display | Complete display | Peak bytes, bounded / complete |
| --- | ---: | ---: | ---: | ---: |
| `survexp.us` | 17,820 | 0.55 ms | 38.00 ms | 601,454 / 1,330,466 |
| 128 × 4 × 128 custom table | 65,536 | 1.34 ms | 112.96 ms | 2,147,373 / 3,778,328 |

The bounded request uses `max_print=6`: six cells for the built-in table and
one complete four-cell row for the custom table. Measurements exclude table
construction and use three repetitions for median time. Memory is measured
separately with `tracemalloc`; results depend on the machine and table shape.
