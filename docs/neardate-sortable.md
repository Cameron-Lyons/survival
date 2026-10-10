# Sortable dates in `neardate`

`survival.r.neardate` matches the closest reference row with the same subject
identifier on or after (`best="after"`) or on or before (`best="prior"`) a query.
Python rows are zero based. R facade rows are one based. Tied reference dates
select the first original row for `after` and the last original row for `prior`.
Missing query dates use `nomatch`; missing identifiers remain missing, and
reference rows with missing dates are discarded.

The Python interface accepts numeric values, character dates, Python
`date`/`datetime`, NumPy `datetime64`, and pandas timestamps. Character dates
retain lexical ordering. They are not parsed as numbers or calendar dates:
`"10" < "2" < "3"`. Numeric and character vectors use the common character
type, matching R's `c(y1, y2)`. Factor dates raise R's `y1 and y2 must be sortable`
error.
Mixed character and NumPy floating-point inputs retain the same numeric label
formatting as equivalent Python floats, including `float32` and `longdouble`.

```python
from survival import r

r.neardate([1], [1, 1], ["3"], ["2", "10"])
# [None]: both reference dates precede the query in lexical order

r.neardate([1], [1, 1], ["10"], ["10", "10"], best="prior")
# [1]: the last original row among tied dates
```

Calendar values are ranked together before entering the Rust matcher. Exact
integer time keys preserve NumPy units down to attoseconds, pandas subsecond
precision, and distant dates combined with fine timestamp units. Converting
calendar keys directly to floating point could merge adjacent timestamps or
overflow NumPy's finer date units; the matcher receives compact ranks instead.
Python dates and naive datetimes use midnight/clock time in UTC; aware datetimes
use their absolute instant. Missing `NaT` and masked datetime entries stay
missing. Calendar and numeric date vectors are separate input families;
convert them to a shared calendar representation before mixing them in Python.

Plain numeric inputs retain their values directly. Complete numeric NumPy and
pandas arrays convert in bulk; ordinary numeric lists use direct float
conversion, including a separate path for `None`. Character/calendar ranking
costs O(n log n) time and O(n) temporary storage for both date vectors together.
Subject grouping and nearest-date searches still run in the shared Rust kernel.

## R facade and stock differences

The R facade ranks character and classed date vectors using R's common type and
locale collation before calling the same matcher. Date and Date/character
inputs agree with stock `survival::neardate`. Python character comparisons use
Unicode code-point ordering, so non-ASCII or case-sensitive ordering can differ
from R's active locale. Python does not parse character date strings implicitly.

Stock survival 3.8-12 attempts `methods::as(value, class(other))` for `POSIXt`
queries. Both `POSIXct` and `POSIXlt` carry two class names, causing
`length(class2) == 1L is not TRUE` before matching. Passing just the first class
name also fails for some POSIXlt/POSIXct conversions because `methods::as` lacks
the required method. The facade uses the supported `as.POSIXct`/`as.POSIXlt`
conversions and returns the intended result. Independent stock calls on epoch
seconds verify these matches; tests record the stock class-conversion failure
separately.

The only `best` choices are `after` and `prior`, including their unambiguous
prefixes. The R facade no longer advertises `closest`, which neither the stock
package nor the Rust matcher implements.

## Validation and complete-call performance

[`test_neardate_sortable.py`](../python/tests/test_neardate_sortable.py) adds 44
regressions for stock character and Date reference rows, mixed character/numeric
types, factors, missing values, calendar containers, exact timestamp resolution,
time zones, and mixed NumPy units. The focused data-preparation and typed-formula
run passed 1,486 tests. The live R tests in
[`test-neardate-sortable.R`](../r/survivalr/tests/testthat/test-neardate-sortable.R)
passed 90 expectations against R 4.5.3 / survival 3.8-12, including the explicit
stock POSIXt errors and corrected numeric references.

[`benchmark_neardate.py`](../scripts/benchmark_neardate.py) measures complete
calls with 100,000 query and reference rows across 100 identifiers. Input
creation is excluded; date conversion, native matching, and result construction
are included. Seven samples follow three warmups, with alternating call order.
All numeric outputs are identical across the saved predecessor and current
implementations. On the local release extension with Python 3.14.7 / NumPy 2.4.6:

| Input | Direction | Before, ms | After, ms |
| --- | --- | ---: | ---: |
| NumPy | after | 73.254 | 52.096 |
| NumPy | prior | 75.866 | 56.034 |
| Numeric lists | after | 74.715 | 58.649 |
| Numeric lists | prior | 74.083 | 57.619 |
| Lists with missing values | after | 72.317 | 58.353 |
| Lists with missing values | prior | 74.022 | 59.805 |

These medians show a 19–29% reduction for the measured numeric calls. They do
not measure calendar ranking or isolate native-kernel time. Reproduce with:

```sh
git show <previous-revision>:python/survival/r/_data_prep.py > /tmp/previous-data-prep.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_neardate.py \
  --baseline-source /tmp/previous-data-prep.py
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_neardate_sortable.py python/tests/test_r_data_prep.py \
  python/tests/test_data_prep_regressions.py python/tests/test_lvcf_ordering.py \
  python/tests/test_frontend_fastpaths.py python/tests/test_typed_formula_expressions.py -q
```
