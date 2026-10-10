# Iterator inputs to formula models

Formula models now retain each used one-shot column or row-aligned argument
within a call. Missing-value checks, subset selection, expression evaluation,
fitting and prediction read the same prepared values. Aliased columns and direct
arguments and subset selectors share that buffer. Unused mapping columns remain
unread, including when they appear before the variables a formula uses.

This applies to the shared model-frame machinery, Cox and AFT fits and their
new-data predictions, ordinary survival curves, Aalen fits, concordance and
formula-based data preparation. Direct survival-response curve, log-rank and
survival-check arguments receive the same preparation before omission. Population
models also prepare their cut expressions and rate mappings before evaluating
them; see [population inputs](population-iterable-inputs.md).

Previously a missingness or row-count pass consumed an iterator before the
response or design builder read it. Valid inputs then failed with mismatched
time/status lengths or an empty grouping column. An iterator shared by a source
column and explicit `weights` argument could fail for the same reason.

Declared factor levels, ordering and contrasts survive preparation. Declared
numeric types retain R's integer or double expression arithmetic, including after
subsetting. Matrix response rows retain their values and missingness. Ordinary
lists, arrays and data frames keep their existing ownership; the original-data
rebuild behavior of a Cox fit with `model=False` is preserved for reusable inputs.

Used iterator values and their metadata survive pickle and copy operations.
Formal variables removed from the design, such as `z` in `x + z - z`, remain
available when rebuilding a model frame. Serialization does not read unrelated
iterators. Their values are unavailable in the restored object's private source
mapping, and explicitly reading such a column raises a clear input error.
The caller must supply a fresh iterator for another independent call after the
original iterator has been consumed.

Rows come from evaluated formula variables and supplied row-aligned arguments.
For example, `~x` uses the length of `x`, regardless of an unrelated first mapping
column. A variable-free `~1` with no extra vectors has zero rows for a raw mapping
and retains a data frame's explicit row count. Adding a weights vector supplies
its row count in both cases, as stock `stats::model.frame` does.

The independent
[reference generator](../scripts/generate_formula_iterator_reference.R) calls
stock R 4.5.3 / survival 3.8.12 methods directly. Its controls cover complete
fit/frame results, reordered and repeated subsets, missing-row actions, factor
metadata, prediction standard errors and shared collapse vectors.
[The Python tests](../python/tests/test_formula_iterator_inputs.py) repeat those
controls with reusable columns, counted iterators, generators, arrays and nullable
pandas columns. They also check aliases, unused-column laziness, retained model
frames, copy operations and prediction curve ids.

## Reusable-input measurements

Complete-call controls cover model frames, Cox/AFT/KM fits with retained models,
Cox risk predictions with standard errors and AFT three-quantile predictions
with standard errors. Each runs with lists, NumPy columns and pandas frames at
64 and 20,000 rows. All 36 complete public payloads match the saved adapters
exactly, including model matrices, retained metadata, summaries and predictions.

Two trials used normal garbage collection, three warmups and nine alternating
before/after pairs; the second reversed case order and initial version order.
Inputs and training fits were prepared outside timing. At 20,000 rows, pooled
paired median changes ranged from -3.1% to +2.9%. Small fit and prediction
measurements have broad, overlapping ranges, so these controls do not establish
a general speed improvement.

The first measurements found a fixed row-count cost for small model frames.
Skipping redundant RHS parsing when response columns already establish rows
reduced the added list cost from 29–44 microseconds to 6–13 microseconds, and the
NumPy cost from 35–39 to 7–8 microseconds across the two trials. Data-frame model
frame median changes were -1.1% and +0.9%. Subset preparation still validates
the RHS before reading a response iterator. Twenty additional ownership,
row-count and invalid-formula controls preserve results and error order.

The saved and current Python adapters share the same release extension, so
these measurements exclude native rate-table validation changes. Python was
3.14.7, NumPy 2.4.6 and pandas 3.0.5. The extension SHA256 was
`7f7b70c34372ff48baf673375a0eb30b3fdebfded6833cbc10d5b5f91df3872d`.
The source manifests, per-trial medians and full ranges are retained in
`/tmp/survival-formula-rowcount-shortcut/combined.json` and
`/tmp/survival-formula-rowcount-shortcut/combined.md`. Individual samples and
complete payload hashes are in that directory's `trial-1.json`, `trial-2.json`
and `payload-parity.json`, alongside the worker and runner. The original
measurements before the row-count shortcut remain in
`/tmp/survival-formula-performance/`.
