# Factor and cluster inputs

`aggregate_survfit` now accepts iterator or generator grouping columns without
losing their rows. The same shared factor correction preserves iterator strata in
the deprecated `survConcordance_fit` method. `concordancefit` also prepares iterator
clusters once before ordering and coding them.

Previously, inferred factor levels consumed a one-shot source, then factor coding
read that exhausted source again. For example, aggregating three curves with
`by=iter(["b", "a", "b"])` raised `arguments must have the same length` even though
the corresponding list worked. With survival rows `[[0.9, 0.3, 0.5],
[0.8, 0.2, 0.4]]`, both forms now return `[[0.3, 0.7], [0.2, 0.6]]` with groups
`a, b`. Clustered concordance had an additional level-ordering pass that exhausted
iterator clusters before the shared helper; its adapter now reuses one prepared
column.

Declared factor categories retain their order. Clustered concordance removes
unused declared categories when collapsing influence values, as R's `rowsum`
does. Values and one-shot category metadata are each read once. Reusable inputs
keep their original values and categories, and returned aggregate values do not
borrow mutable input storage. Missing or wrong-length clusters still raise their
existing errors. Ordinary numeric NumPy arrays retain the existing early numeric
factor path.

The independent reference generator is
[`generate_factor_iterable_reference.R`](../scripts/generate_factor_iterable_reference.R).
Its fixture contains 51 stock R 4.5.3 / survival 3.8.12 cases: eight factor
encodings, 32 ordinary and multi-state curve aggregates, eight clustered
concordance results, and three deprecated concordance stratum results. Coverage
includes declared factor order, signed zero, infinite numeric labels, actual
missing values, empty factors, and compound named grouping columns. Concordance
references include weighted counts, variances, cluster influence values, all
influence columns, and event ranks.

[`test_factor_iterable_inputs.py`](../python/tests/test_factor_iterable_inputs.py)
runs those references with lists, counted iterators, generators, NumPy inputs, and
nullable pandas inputs, plus declared numeric-array factor and ownership controls.
All 265 tests pass; the eight affected Python regression files pass 1,665 tests.
Regenerating the fixture with the current stock R helper produces identical
bytes. Iterator columns in formula model frames and predictions are covered by
the later [formula input correction](formula-iterator-inputs.md).

[`benchmark_factor_iterable_inputs.py`](../scripts/benchmark_factor_iterable_inputs.py)
compares complete public curve aggregation and clustered concordance calls with
the saved pre-change factor helper and concordance adapter, using one shared
native extension. Aggregation uses the current public function body with the
saved helper substituted into its grouping adapter. Inputs are created outside
timing. Each sample includes factor extraction and coding,
native calculation, result construction, and every returned field or getter.
Complete output payloads must agree and are hashed outside timing. Saved and
current calls alternate within one process; `--reverse-order` reverses the initial
order, and `--disable-gc` is available as an explicitly identified diagnostic.
Both versions receive the same reusable inputs: 100,000 and 500,000 grouping
values across 100 groups. Aggregation uses two time rows and list, Unicode-array,
object-array, and numeric-array groups. Concordance uses a prepared response,
numeric predictor, and list, numeric-array, and declared-factor clusters with
`influence=1` and `ranks=False`.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_factor_iterable_inputs.py \
  --baseline-factor-source /tmp/survival-factor-iterables-before.py \
  --baseline-concordance-source /tmp/survival-concordance-iterables-before.py \
  --rows 100000 500000 --repeat 9 --warmup 3
```

The recorded run used Python 3.14.7 and NumPy 2.4.6 on a quiet Linux x86-64
machine with normal garbage collection. A second run added `--reverse-order`;
both runs alternated saved/current calls. Each version received three warmups
and nine measured samples per case in each run. The table pools the 18 samples
per version and shows median milliseconds with the complete minimum–maximum
range. Input construction and payload hashing were outside timing; all output
fields and getters were inside it.

| Operation | Group input | Values | Saved median [range], ms | Current median [range], ms |
| --- | --- | ---: | ---: | ---: |
| Aggregate | Character list | 100,000 | 14.46 [13.84–15.31] | 11.24 [10.80–11.47] |
| Aggregate | Unicode array | 100,000 | 19.63 [19.04–20.48] | 13.89 [13.41–14.48] |
| Aggregate | Object array | 100,000 | 15.75 [15.23–16.44] | 11.92 [11.43–12.62] |
| Aggregate | Numeric array | 100,000 | 6.24 [6.04–6.67] | 6.23 [5.98–6.60] |
| Aggregate | Character list | 500,000 | 77.33 [76.86–77.88] | 59.39 [58.90–60.08] |
| Aggregate | Unicode array | 500,000 | 112.30 [110.67–115.00] | 77.45 [75.74–78.89] |
| Aggregate | Object array | 500,000 | 85.59 [85.09–92.28] | 63.80 [63.15–64.02] |
| Aggregate | Numeric array | 500,000 | 33.75 [33.44–34.27] | 33.78 [33.61–34.32] |
| Concordance | Character list | 100,000 | 22.60 [21.97–23.80] | 22.72 [21.59–23.67] |
| Concordance | Numeric array | 100,000 | 18.05 [17.17–18.53] | 17.94 [17.27–19.06] |
| Concordance | Declared factor | 100,000 | 23.28 [22.34–24.08] | 23.22 [22.59–24.03] |
| Concordance | Character list | 500,000 | 114.07 [111.01–118.16] | 114.45 [111.77–116.91] |
| Concordance | Numeric array | 500,000 | 89.36 [87.93–91.06] | 89.89 [87.82–92.47] |
| Concordance | Declared factor | 500,000 | 116.86 [115.15–134.77] | 116.66 [115.11–122.62] |

Character aggregation's pooled medians are 22–31% lower, with nonoverlapping
ranges in all six cases. Preparing nonnumeric inferred factors once removes a
second label materialization and validation pass. Numeric aggregation medians
differ by less than 0.2%, and the concordance controls by less than 0.7%; their
ranges overlap. These controls establish no consistent speed change for the
tested reusable numeric and cluster inputs.

Every one of the 14 cases has identical saved/current complete result payloads
in both runs, and each payload hash agrees across the two initial orders. The
raw samples, complete outputs, and source/extension hashes are recorded in
`/tmp/survival-factor-iterables-benchmark-final.json` and
`/tmp/survival-factor-iterables-benchmark-final-reversed.json`; pooled measurements
and machine/script metadata are in
`/tmp/survival-factor-iterables-benchmark-comparison.json`. Both versions used the
same extension, SHA256
`0bc3cc881220378b8838c8093675324369a36346c95e2f4212f9fb42d39b7ed0`.
