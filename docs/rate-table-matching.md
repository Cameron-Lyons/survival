# Population rate-table matching

`survival.r.match_ratetable(data, ratetable)` matches named data columns to
the dimensions of a native population table. It uses the same Rust matcher
as `pyears` and `survexp`.

```python
from datetime import date
from survival import r

matched = r.match_ratetable(
    {"age": [50 * 365.25, 60 * 365.25],
     "sex": ["m", "f"],
     "year": [date(1996, 1, 1), date(1997, 1, 1)]},
    r.survexp_us(),
)
print(matched.summary)
positions = r.as_data_frame(matched)
assert positions["sex"] == [1.0, 2.0]
assert positions["year"] == [9496.0, 9862.0]
```

The `RateTableMatch` result contains `r` (one row per observation), `dimid`
(column names in table order), `cutpoints` (numeric table cutpoints), and
`summary` (R's matched-population text for the built-in US, US race-specific
and Minnesota tables; `None` for custom tables). Values keep their full
precision. Data-frame conversion returns independent numeric columns.

Mappings and data frames are accepted. Required columns must have equal
lengths; unrelated columns are ignored. Continuous values retain their
units, dates become days since 1970-01-01, and time differences count days.
Dates in a categorical or ordinary continuous dimension are rejected.
Timezone-aware date-times use the UTC calendar date, matching R's POSIX
conversion; this also corrects `ratetableDate` around local midnight.
Numeric categorical positions must be one-based integers within the axis.
Missing or infinite positions are rejected before reaching numerical kernels.

String columns are treated as categorical labels for convenience. For
categorical arrays, every declared level is validated, including levels with
no observations. Matching ignores case and accepts unique prefixes. An exact
match wins over longer prefixes; duplicate exact matches and ambiguous
prefixes are errors. Pandas categories and other supported factor containers
retain their declared levels through subset and missing-row omission.

The shared model-frame path previously discarded these extra-column levels.
It now preserves them, and single-element `rmap` sequences are expanded
before row validation while retaining categorical metadata. These corrections
also apply to `pyears` and `survexp`.

Matching returns original entry positions. The later US calendar birthday
adjustment belongs to the population calculations; it is not applied here.
Empty input returns empty positions and R's empty built-in summary, without
R's warnings about extrema of empty vectors. Duplicate required input names
are rejected explicitly rather than selecting an arbitrary column. Arbitrary
R summary callbacks and R's legacy matrix attributes are not executed;
named Python columns and native rate tables supply the required information.

## Rust API and lookup cost

Rust callers can use `population::match_ratetable` with `RatetableColumn`
values. `RateTable::match_levels(dimension, labels)` validates an explicit
factor-level list; `dimension` is zero-based and returned positions are
one-based. The same method is available on Python `RateTable` objects.

The matcher folds table labels once and indexes exact matches. A small cache
handles the common sex/race dimensions; after four distinct observed labels,
it promotes to a hash map. Large categorical columns no longer scan all
previously observed labels for each row. Prefix matching scans the folded
axis only when there is no exact match. Cached keys borrow input strings,
and matched rows are written directly into the output matrix. The Python
binding releases the GIL during native matching.

`scripts/benchmark_rate_matching.py` measures 100,000 observations with 2,
64 or 1,024 categorical levels, excluding input construction. It can be run
against the parent build and this build with the same interpreter. The
`rate_matching_bench` group in `benches/survival_benchmarks.rs` measures the
Rust kernel without Python conversion.

With CPython 3.14 and seven repetitions, one local comparison against the
parent implementation (`62216fc6`) measured these median binding-call times:

| Levels | Parent | Current |
| ---: | ---: | ---: |
| 2 | 4.00 ms | 4.69 ms |
| 64 | 9.51 ms | 5.60 ms |
| 1,024 | 88.54 ms | 6.15 ms |

The larger cases improve by about 1.7× and 14.4×. The two-level case has a
0.69 ms (17%) cost per 100,000 rows. Timings include input conversion and
native result construction, but exclude reading the result's copying `r`
property. Results depend on the machine, label lengths and repetition pattern.

## R references

`scripts/generate_rate_matching_reference.R` regenerates 23 cases from
R 4.5.3 / survival 3.8-12. They cover all built-in tables, reordered and extra
columns, numeric and categorical positions, dates, date-times and time zones, time
differences, empty input, prefix/exact ambiguity, duplicate axes, unused
levels and input validation. Separate regression tests cover NumPy dates,
pandas categories, scalar expansion, subset/omission, ownership, data-frame
conversion and the native level API. A concurrency check verifies that
Python threads can run during native matching.
