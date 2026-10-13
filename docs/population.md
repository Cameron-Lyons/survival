# Population rates, expected survival and person-years

This page covers population rate tables, rate-table matching, expected survival
(`survexp`), person-years (`pyears`) and their retained components and summaries.
Differences from R are listed in
[R compatibility](r-compatibility.md#population-calculations).

## Rate tables

`survival.population` ships R's exact `survexp.us`, `survexp.usr` and
`survexp.mn` tables (`kernels.survexp_us()` and so on; `r.survexp_us()` in the
R-style interface). `RateTable` objects can also be built from attributes;
construction accepts NumPy rate vectors and owns its copied rates.
`RateTable.source` identifies a built-in table and survives native cloning;
tables built from attributes have `source=None`.

Public Rust population calculations revalidate a table's public fields at each
calculation boundary: shortened dimension attributes, cutpoints or rates are input
errors, and negative or nonfinite rates are rejected as by the constructor. This
covers level matching, mapped positions, US calendar alignment, person-years and
expected survival. Group dimensions use checked arithmetic, public rate lookup
returns `None` for an invalid index or overflow, formatting a damaged table does
not index past its attributes, and US alignment checks that the mapped positions
include its age and year columns.

Native calendar conversion rejects nonfinite day counts and dates outside the
`CalendarDate` range (years representable by `i32`); cutpoints, input dates and
derived birth dates get the same checks. Rust `days_to_date` and `start_of_year`
return `SurvivalResult`; Python raises `ValueError`.

## Matching data to a rate table

`r.match_ratetable(data, ratetable)` matches named data columns to the table
dimensions with the same Rust matcher as `pyears` and `survexp`.

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

`RatetableMatch` contains `r` (one row per observation), `dimid` (column names in
table order), `cutpoints` and `summary` (R's matched-population text for the
built-in tables, `None` for custom tables).

- Mappings and data frames are accepted; required columns must have equal lengths
  and unrelated columns are ignored. Duplicate required names are rejected.
- Continuous values keep their units, dates become days since 1970-01-01 and time
  differences count days. Dates in a categorical or ordinary continuous dimension
  are rejected. Timezone-aware date-times use the UTC calendar date, as R's POSIX
  conversion does. Numeric categorical positions must be one-based integers within
  the axis; missing or infinite positions are rejected.
- String columns are categorical labels. Every declared level of a categorical
  array is validated, including unused ones. Matching ignores case and accepts
  unique prefixes; an exact match wins over longer prefixes, and duplicate exact
  matches, ambiguous prefixes and repeated table dimensions are errors. Pandas
  categories and other factor containers keep their levels through subset and
  omission, also as extra model-frame columns in `pyears` and `survexp`, and
  single-element `rmap` sequences are expanded before row validation.
- Matching returns the original entry positions; the US calendar birthday
  adjustment belongs to the population calculations. Empty input returns empty
  positions and R's empty built-in summary.

Rust callers use `population::match_ratetable` with `RatetableColumn` values.
`RateTable::match_levels(dimension, labels)` validates an explicit level list
(zero-based dimension, one-based returned positions) and is also available on
Python `RateTable` objects. The binding releases the GIL.

## Expected survival

`survexp` computes cohort (Ederer, Hakulinen, conditional) and individual
expected survival from a rate table or a fitted Cox model. The low-level
`kernels.survexp(table, positions, times=...)` takes matched positions; the
formula interface is `r.survexp`.

With a rate table, requested times are selected from the fitted grid with one
forward scan. Duplicate requests keep separate rows, including signed zeros, and
follow-up times can add fitted rows between requests without merging nearby
distinct times. Scaling divides the reported times after selection.
`SurvexpResult.cumhaz` is `-log(surv)` with R's domain behavior: zero survival
gives positive infinity and missing or negative survival gives NaN, including for
Cox curves whose group has an exhausted risk set. As in R, an empty sparse group
has cohort survival zero and conditional survival one.

### Cox rate models

`r.survexp(..., ratetable=cox_fit)` prepares the prediction rows once and calls
`CoxPHFit.expected_survival` (or its penalized counterpart), which aggregates
baseline hazards and subject risks in Rust without per-subject survival curves:
storage is O(N + B + T × G) for N subjects, B baseline points, T requested times
and G groups, beyond the model and prediction design.

- Weighted Ederer, Hakulinen and conditional calculations keep
  `exp(-H).powf(risk)` rounding and underflow and NaN propagation after a cohort's
  risk set empties. Output uses the union of event times in the strata
  represented, with constant interpolation at requested times; repeated output
  times are kept. A stratum without events has survival one, and an event-free
  model can be evaluated at explicit times. Weights must be finite and
  nonnegative, with positive total weight per group.
- Rust callers can supply `CoxExpectedBaseline` values to `survexp_cox_prepared`.
  The Python numeric entry point takes flattened baseline times and hazards with
  their lengths, subject risks, zero-based strata and groups, weights, optional
  follow-up and requested times; it accepts NumPy vectors and releases the GIL.
  `SurvexpResult.to_arrays()` returns independent writable time-by-group survival
  and risk-count matrices.
- The R bridge's `survexp` evaluates formulas, contrasts and factor levels in R
  and computes cohort and individual predictions with the shared kernels for both
  R and Python-backed Cox fits, keeping formula environments, retained data, row
  names, subsets, `na.exclude` and non-syntactic mapping names. For individual
  predictions, the fitted model's own response supplies the step lookup. Sparse
  frailty models reject new cohort curves, as R does; individual predictions keep
  the fitted frailty in the training risk sets and use zero frailty for new rows.
  The native and Python results also supply risk counts for the Hakulinen and
  conditional methods.

## Person-years

`pyears` tabulates person-years, events and, with a rate table, expected events
or expected person-years over fixed categories, `tcut` time-changing categories
and rate-table dimensions.

```python
import survival

response = survival.Surv([25.0, 8.0], [1, 0])
attained = survival.tcut([0.0, 5.0], [0.0, 10.0, 20.0, 30.0])
result = survival.pyears(response, group=attained, scale=1)
```

- The response may be a `Surv` or a numeric matrix. Without a rate table, a
  two-column matrix is time and event; with one, it is start and stop without
  events, and a one-column matrix supplies follow-up without events. Python
  accepts a row-major matrix column (`Y ~ group`) or a `cbind(...)` response,
  including named columns, arithmetic, nested bindings and scalar constants.
  Numeric event counts stay numeric (fractional values and values above one are
  allowed); matrices with more than two columns are rejected. Zero-time
  right-censored events give R's warning, and missing responses or invalid rate
  tables fail explicitly. Missing cells in either column remove the whole row.
  Matrix responses may be lists or NumPy arrays of any layout, including strided,
  Fortran-ordered, read-only and object or nullable inputs; retained responses and
  matrix columns returned by `model_frame` are independent row-major lists, and
  retained `x` values are ordinary nested lists.
- In fixed categories, zero-time rows keep observed events without adding exposure
  or observations. Category products are checked against addressable storage
  before allocation.
- Direct Python calls can supply the response as a time vector, `Surv`, numeric
  matrix or the `time`/`start`/`stop`/`event` keywords, and honor `data`,
  `ratetable`, `rmap`, `expect`, weights and the retention flags:

  ```python
  result = r.pyears(time=[2, 4], ratetable=table, rmap={"age": [40, 50]},
                    expect="pyears", scale=1)
  ```

  Rate positions use the table's units; direct keywords may name columns in `data`,
  and generated columns never overwrite sources used by rate mappings or weights.
  Factor groups keep declared levels and `TcutResult` groups their boundaries.
  Missing follow-up and mapped values go through the same subset and missing-value
  handling as formula inputs. With `start`/`stop` and a rate table, positions
  advance to the entry time before accumulation; without a rate table and without
  events, the response is the follow-up duration. A complete `Surv` or matrix cannot
  be combined with separate start or event values, follow-up cannot be supplied
  twice, and formula strings reject direct response or group keywords. The R
  bridge's direct vector form still needs formula calls for rate tables.
- `pyears(ratetable=cox_fit)` is not supported (R also fails); use `survexp`.

`PyearsResult.to_arrays()` returns independent writable NumPy snapshots of the
column-major cell tables, dimension lengths, observation count and off-table
exposure; event and expected tables are `None` when absent. The R bridge
evaluates formulas once in R, passes prepared category and rate-position matrices
as NumPy arrays and keeps model-frame metadata, X/Y components, missing-data
actions, unused factor levels, category classes and table labels. A no-term
formula with `data.frame=TRUE` returns person-years and counts.

The low-level person-years summary checks each table extent and their product,
and with `totals=TRUE` the expanded shape, before allocating; overflowing shapes
are input errors. Zero extents keep valid empty tables, including R's
one-dimensional empty tables and the shapes `[0, 2]` and `[2, 0]` with their
margins.

## Population formula inputs

`pyears` and `survexp` prepare one-shot data columns, direct response arguments,
weights, subsets and vector rate mappings together before evaluating `cut`, `tcut`
or rate-map expressions; an iterator used as a column and a direct argument is
read once, and one-shot columns the formula does not use are read too. Categorical sources keep their
labels, order and unused levels, and date rate sources stay dates in retained
models. Person-years matrix responses accept NumPy row iterators as well as
arrays and row lists.

Response-free formulas take their rows from the variables they use: `~1` can use
row-aligned weights, referenced rate sources or direct rate vectors, and scalar
rate entries are expanded to that count (or a data frame's row count) before
integration. Unrelated mapping columns do not supply rows. Direct `rmap` vectors
correspond to external vectors referenced by symbols in R's rate-map call;
length-one vectors broadcast. For a Cox rate model, prediction receives the
prepared source columns, keeping categorical metadata.

## Model components and summaries

`pyears` and cohort `survexp` honor `model`, `x` and `y`. `model=True` keeps the
evaluated formula columns, the source columns referenced by `rmap` expressions
and supplied weights; factors keep their level order and unused levels, and
`tcut` columns their cutpoints and labels. Subsets and omission apply to every
retained component, and `model=True` takes precedence over `x` and `y`, as in R.
With `model=False`, `pyears(x=True)` keeps a row-major matrix of one-based
category codes and raw scaled `tcut` times, and `survexp(x=True)` a `StrataFactor`
with zero-based codes, labels and counts; without grouping terms either keeps a
vector of ones. `pyears(y=True)` keeps the `Surv` response or a numeric matrix;
`survexp(y=True)` keeps numeric follow-up times, the largest requested time
(before scaling) for a response-free rate-table call, and `None` for a Cox
reference. Both results support `model_formula` and `model_term_names`, and
`model_frame` when made with `model=True` (plain columns, with a `Surv` response
expanded); `.model` keeps the richer column objects. Individual `survexp` methods
return a plain vector, ignore the retention flags and restore excluded rows as NaN
under `na.exclude`.

`summary_survexp`, `summary_ratetable` and `summary_tmerge` are available through
`survival.r` and the `model_summary` generic. Expected curves and their
summaries, rate tables and their summaries, and the `tmerge` count table support
`as_data_frame`.

- `summary_survexp` returns a `SummarySurvexpResult`: a vector for one curve or a
  time-by-curve matrix, with `strata` labels naming the columns. Requested times
  are sorted with duplicates kept; missing times and times outside the observed
  range are dropped. Survival uses the preceding observation (one before the
  first) and risk counts the next. Omitted times keep the original rows; a scalar
  `scale` divides the output times with R's zero and nonfinite arithmetic; tied
  source times use R's averaged indices without its tie-collapse warning. Rust
  callers use
  `population::summary_survexp`, whose selector validates shapes and time order and
  makes one sweep after sorting.
- `summary_ratetable` returns a `SummaryRatetableResult` with the canonical attributes, a
  dimension table and the native summary text through `str(result)`. Factor
  dimensions have levels; others have bounds in native units, with calendar bounds
  as ISO dates.
- `summary_tmerge` returns a column-oriented count table with one row per
  operation. The R bridge drops retained `tmerge` metadata on subsetting and
  follows ordinary data-frame dispatch.
