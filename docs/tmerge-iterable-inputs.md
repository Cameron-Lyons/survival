# Time-dependent updates from one-shot inputs

`survival.r.tmerge` prepares iterator columns once per call across both input
frames, the initial time range, ids and all update operations. Shared iterator
objects reuse the same prepared values. This includes a first event's time
vector, which defines the subject's range before placing its event.

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

Named and direct update vectors work for `tdc`, `cumtdc`, `event` and
`cumevent`. Declared factor levels determine censor values even when the
first level is absent from the observed updates. Input mappings retain their
original column objects; preparation owns reusable buffers internally.

All `data1` columns are retained and checked for equal lengths. Only referenced
`data2` columns are read: unrelated iterator columns may remain unavailable or
have a different length, as in the formula interface. Each used time/value
column must match the ids. For mappings, ids establish the update row count;
for a data frame, a direct id vector must also match its explicit row count.
This extends R's data-frame inputs, whose columns already have a common length.

Previously, reading all `data2` columns exhausted one-shot sources before their
operations were evaluated. Repeated direct operation vectors and sharing
`data1` with `data2` could also lose rows. Complete frames now preserve row
alignment, interval boundaries, event values and the retained `tname`,
`tevent`, `tdcvar` and `tcount` metadata.

The independent reference generator runs stock R survival 3.8-12 on R 4.5.3.
Its 18 complete cases cover implicit/explicit ranges, repeated initial ranges,
renamed columns, all four update types, missing-value retention and omission,
delays, shared direct vectors and categorical events/covariates. Tests repeat
these outputs with lists, tuples, iterators, generators, NumPy arrays, NumPy
scalar iterators and pandas columns. Separate checks cover unread inputs,
aliases across both frames, input ownership and invalid update lengths.

```sh
Rscript scripts/generate_tmerge_iterable_reference.R
PYTHONPATH=python .venv/bin/python -m pytest \
  python/tests/test_tmerge_iterable_inputs.py python/tests/test_r_data_prep.py -q
git show HEAD:python/survival/r/_data_prep.py > /tmp/tmerge-before.py
PYTHONPATH=python .venv/bin/python scripts/benchmark_tmerge_inputs.py \
  --baseline-source /tmp/tmerge-before.py --rows 40000 --repeat 7
```

Extract the revision preceding this change for comparisons after committing.
The benchmark alternates old/current adapters in the same process with one
native extension and checks the complete output frames and summary metadata
before timing. It includes input conversion, native updates and construction
of every returned component; input generation and JSON verification are
outside timing. The update workload has 40,000 rows, 10,000 subjects and three
operations. Its wide form adds 32 unrelated NumPy columns.

Local medians on Python 3.14.7 and NumPy 2.4.6:

| Complete call | Before | After |
| --- | ---: | ---: |
| Initial shared frames, 10,000 subjects | 27.42 ms | 27.60 ms |
| Three updates, four source columns | 138.57 ms | 138.86 ms |
| Same updates, 32 additional unused columns | 177.21 ms | 143.81 ms |

Required iterator inputs use `O(n)` storage per distinct source. Unrelated
update columns add no value-conversion or row-copy cost. Native sorting and
interval expansion retain their existing complexity.
