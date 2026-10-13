# Data preparation

This page covers the R-style data preparation functions in `survival.r`:
time-dependent data (`tmerge`), date matching (`neardate`), carrying values
forward (`lvcf`), Fine–Gray expansion (`finegray`) and O'Brien risk-set expansion
(`survobrien`). Differences from R are listed in
[R compatibility](r-compatibility.md#data-preparation).

## Time-dependent data

`survSplit`, `tmerge`, `survcondense`, `finegray` and `survobrien` keep factor
columns as factors (`survival.r.Factor`) with their levels, so a model refitted
on their output uses the same reference levels. Factor `event` and `tdc`
arguments of `tmerge` give factor columns; a `tdc` default outside the levels is
appended as a level, as R does. Multi-state statuses of `survSplit` and
`survcondense` are factors with R's `"censor"` first level. `survSplit(zero=)`
may hold one start time per row.

`r.tmerge` builds start/stop data with R's `(tstart, tstop]` boundary rules,
event placement, cumulative updates, missing-value handling and classification
metadata; the README has a complete example. `tdc`, `cumtdc`, `event` and
`cumevent` accept column names or direct vectors. The raw per-call sweep is
`survival.data_prep.tmerge_step`, for callers that manage sorted numeric arrays.

```python
from survival import r

data = {
    "id": iter([1, 2]),
    "time": iter([10.0, 20.0]),
    "status": iter([1, 1]),
}
frame = r.tmerge(
    data, data, id="id", death=r.event(data["time"], data["status"])
)
assert frame["death"] == [1, 1]
assert frame["tstop"] == [10.0, 20.0]
```

- One-shot iterator columns are prepared once per call across both frames, the
  initial time range, ids and all update operations; shared iterator objects reuse
  the same values, including a first event's time vector, which defines the
  subject's range before its event is placed.
- All `data1` columns are kept and must have equal lengths. Only referenced
  `data2` columns are used; unrelated columns may have another length (one-shot
  iterators among them are still read). Each used time or value column must match the ids. For mappings, the ids
  set the update row count; for a data frame, a direct id vector must also match
  its row count.
- Declared factor levels determine censor values even when the first level does
  not occur in the updates. Input mappings keep their original column objects.
- Complete frames keep row alignment, interval boundaries, event values and the
  `tname`, `tevent`, `tdcvar` and `tcount` metadata; `summary_tmerge` reports
  where updates fell relative to each interval.

## Matching dates

`r.neardate(id1, id2, y1, y2, best="after")` matches each query to the closest
reference row with the same identifier on or after (`best="after"`) or on or
before (`best="prior"`) the query date; unambiguous prefixes are accepted.
Python rows are zero-based and R facade rows one-based. Tied reference dates
select the first original row for `after` and the last for `prior`. Missing query
dates give `nomatch`, missing identifiers stay missing, and reference rows with
missing dates are discarded.

```python
r.neardate([1], [1, 1], ["3"], ["2", "10"])
# [None]: both reference dates precede the query in lexical order

r.neardate([1], [1, 1], ["10"], ["10", "10"], best="prior")
# [1]: the last original row among tied dates
```

- Accepted dates are numbers, character strings, Python `date`/`datetime`, NumPy
  `datetime64` and pandas timestamps. Character dates keep lexical order
  (`"10" < "2" < "3"`) and are not parsed. Numeric and character vectors use the
  common character type, as R's `c(y1, y2)` does; mixed character and NumPy float
  inputs format numbers as equivalent Python floats. Factor dates raise R's
  `y1 and y2 must be sortable` error.
- Calendar values are ranked together with exact integer time keys, keeping NumPy
  units down to attoseconds and pandas subsecond precision. Python dates and naive
  datetimes are taken as UTC; aware datetimes use their absolute instant. `NaT`
  and masked entries are missing. Convert calendar and numeric date vectors to one
  representation before mixing them.
- The R facade ranks character and classed date vectors with R's common type and
  locale collation before calling the same matcher; Python compares characters in
  Unicode code-point order.

## Carrying values forward

`r.lvcf` carries non-missing values forward within each subject and returns a list
in input row order. With `time`, rows are ordered within each subject with missing
times last, and equal times keep input order, as in R.

```python
r.lvcf([1, 1, 1], [None, None, 1], time=[None, 1, 2])
# [1, 0, 1]

r.lvcf([1, 1, 1], [None, None, 1], time=[None, 1, 2], first=False)
# [1, None, 1]
```

The default `first=True` sets an unknown first observation to zero for logical and
binary numeric columns; other numeric, character and factor columns keep unknown
first observations, also for pandas categoricals and R factors with numeric or
logical levels. Inputs are not modified.

## Fine–Gray expansion

`r.finegray` expands a multi-state response for the Fine–Gray model:

```python
from survival import finegray

data = {
    "time": [1.0, 2.0, 3.0, 4.0],
    "event": ["target", "competing", "censor", "target"],
    "x": [0.2, 0.4, 0.1, 0.8],
}
expanded = finegray("Surv(time, event) ~ x", data=data, etype="target",
                    count="replication")
print(expanded["fgstart"], expanded["fgstop"], expanded["fgwt"])
```

String labels use a recognized censor label as the censoring state; pandas
categoricals keep their declared order. Python evaluates the formula and
assembles the result; subject checks, time grids, censoring and entry curves and
the interval expansion run in one native call. The R bridge evaluates its model
frame once in R, keeps arbitrary transformed and matrix-valued columns, and indexes
that frame with the returned source rows.

- R's weight convention is kept: censoring and entry fits are unweighted, and user
  weights multiply each output row using its source position *within the stratum*,
  indexing the full weight vector. A stratified result's `(weights)` column
  therefore need not equal its initial `fgwt`.
- Delayed entry is decided from each subject's first row, and a zero censoring
  probability before a selected event raises "censoring probability is zero
  before a selected event". A selected endpoint with no events is rejected.
  Covariate missing values can be kept with `na.pass`; missing responses, strata
  or ids cannot be used.

`regression.finegray_expand` (Rust `regression::finegray_expand` with
`FineGrayInput`) prepares the response without formulas. It takes times,
non-negative integer status codes (zero is censoring), a positive selected
endpoint and optional start times, strata, subject codes and weights. Counting
histories need subject ids and contiguous intervals, and a transition may appear
only on a subject's last row; near ties are resolved first when `timefix=True`.
Strata and subject codes may be arbitrary integers. Output follows ascending
strata, original row order within each stratum, then added intervals;
`FineGrayOutput.row` holds one-based source rows and `wt` includes supplied
weights. The six-vector `kernels.finegray` kernel still takes a caller-supplied
censoring curve. Both results provide `to_arrays()`, and native preparation runs
without the GIL.

## O'Brien risk-set expansion

`r.survobrien` expands right-censored or counting-process observations into one
block per event time and applies the logit of each continuous variable's mid-rank
percentile within the block. With strata, blocks follow the first appearance of
event-time/stratum pairs; without strata, time order. Rows keep input order within
a block. Counting intervals are open on the left and closed on the right. Tied
covariates get average ranks; missing covariates stay missing and do not enter the
rank denominator; infinite covariates take part in ranking, while interval times
must be finite. Factor keeper columns are kept untransformed; Python treats plain
string columns as factors. Data without events return an empty frame with the
expected columns.

`kernels.survobrien` accepts NumPy vectors and a sequence of continuous columns
and releases the GIL. `block_offsets` holds the half-open slices of consecutive
blocks, and `to_arrays()` returns independent writable arrays. `transform=False`
builds only the risk sets (`continuous=[]` allowed); Rust callers have
`validation::survobrien_expand`. A custom transformation is first checked on ten
values, as in R, then receives one column of one risk set per call and must
return one value per row.

The R bridge evaluates formulas in R and uses the shared kernel for every
expansion, including user-defined transformations, protected `I()` terms,
factor/strata/cluster combinations, formula environments without a data argument
and subsets. Character continuous terms follow R's rank ordering, and a
matrix-valued continuous term uses its first column, as in R. Interaction terms
and unsupported response types are rejected.
