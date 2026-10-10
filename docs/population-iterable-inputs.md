# Population formula inputs

`pyears` and `survexp` prepare one-shot data columns, direct response arguments,
weights, subsets and vector-valued rate mappings together before evaluating
`cut`, `tcut` or rate-map expressions. An iterator reused as a data column and a
direct argument is read once within that call. Its cached values then supply
missing-value checks, row selection, calculations and retained model columns.
Unused columns remain unread. Reusable lists, NumPy arrays and data frames keep
their storage; retained responses and model columns have independent values.

Categorical sources keep their declared labels, order and unused levels through
preparation and selection. Date-valued rate sources remain dates in retained
models. Person-years matrix responses accept NumPy row iterators as well as
matrix arrays and iterators of row lists. Missing elements inside either kind
of row remove the entire row under `na.omit` or `na.exclude`.

Response-free population formulas determine rows from the variables they use.
`~1` can use row-aligned weights, referenced rate sources or direct rate vectors;
scalar rate entries expand to that count or an explicit data frame's rows.
Unrelated columns in a plain mapping do not supply rows. Direct `rmap` vectors
correspond to external vectors referenced by symbols in R's rate-map call.
Length-one rate vectors retain the API's existing broadcasting behavior. Python
direct vectors have no R expression-symbol names to add to retained models.

For a Cox rate model, prediction receives the prepared source columns without
eagerly converting every unrelated column into a list. This preserves categorical
metadata and prevents prediction from consuming unused iterators.

`scripts/generate_population_iterable_reference.R` independently records stock
R 4.5.3 and survival 3.8-12 results in
`python/tests/fixtures/population_iterable_reference.json`: 38 complete population
cases, four scalar-map/weight controls and a scalar-only mapping error. The cases
cover missing and complete data, right and counting-process responses, numeric
event-count and entry matrices, cut and time-cut groups, declared categorical
rates, expression mappings, grouped and individual expected survival, and
numeric and categorical Cox rate models. Tests compare tables, expected values,
curve times and survival, available risk counts, methods, labels, omissions,
retained responses, grouping values and model metadata. They also check direct
argument aliases, unread columns and independent retained storage. Stock Cox
`survexp` results omit risk counts, so those cases compare the fields stock R
actually returns.

The one-row scalar-rate/weight controls use stock R directly. The four-row
scalar-broadcast controls use stock external rate vectors with the constants
explicitly repeated to the weight count. Stock `survexp` passes literal scalar
rate entries to its native kernel without expanding them for multiple rows;
that path produced incorrect values and a crash during the audit. Its output
is therefore unsuitable as a numerical oracle for scalar broadcasting. Python
expands those constants before calling the kernel.

## Measurements

Two trials used 100,000 and 500,000 reusable rows with inputs constructed before
timing, normal garbage collection, three warmups per version and nine alternating
before/after pairs per trial. The second trial reversed the initial version
order. Timings include the complete public call and reading every returned
field; all full result payloads matched in every sample and across both trials.
The table pools 18 samples per version and reports milliseconds as median
`[minimum, maximum]`, including outliers.

| Call and input | Rows | Before (ms) | After (ms) |
| --- | ---: | ---: | ---: |
| `pyears`, lists | 100,000 | 67.489 [65.430, 68.534] | 67.779 [65.375, 69.164] |
| `pyears`, arrays | 100,000 | 30.624 [29.816, 32.236] | 30.742 [29.881, 31.612] |
| Cox `survexp`, arrays | 100,000 | 74.885 [72.211, 78.643] | 65.513 [62.083, 68.353] |
| Cox `survexp`, six unused array columns | 100,000 | 88.210 [86.757, 90.012] | 66.520 [65.052, 69.059] |
| Cox `survexp`, declared categorical groups | 100,000 | 82.354 [80.877, 84.376] | 73.300 [71.086, 74.926] |
| `pyears`, lists | 500,000 | 351.636 [341.468, 442.746] | 351.411 [344.341, 454.944] |
| `pyears`, arrays | 500,000 | 149.303 [147.149, 151.329] | 150.377 [146.934, 152.777] |
| Cox `survexp`, arrays | 500,000 | 392.390 [388.811, 395.041] | 344.425 [340.407, 347.924] |
| Cox `survexp`, six unused array columns | 500,000 | 461.960 [458.718, 469.492] | 345.182 [342.128, 350.850] |
| Cox `survexp`, declared categorical groups | 500,000 | 423.814 [392.660, 744.557] | 363.143 [358.208, 388.850] |

The person-years ranges overlap; these measurements support no performance gain
for those controls. Cox calls used one eight-row fitted rate model, three groups
and three query times. Their pooled medians fell 12.2–12.5% for ordinary arrays,
24.6–25.3% with six unused columns and 11.0–14.3% with declared grouping levels.
Both trial orders improved, with separate before/after ranges. These results
apply to the stated inputs.

The comparison loads the saved population adapter and current adapter in one
process, sharing the same `_formula.py`, `_coerce.py` and native extension from
before the formula row-count shortcut. It isolates the population entrance and
Cox source-mapping changes; it excludes that shortcut and native validation
changes. Python was 3.14.7 and NumPy 2.4.6. The extension SHA256 was
`7f7b70c34372ff48baf673375a0eb30b3fdebfded6833cbc10d5b5f91df3872d`.
The measurement artifacts record every sample, complete outputs, payload hashes
and source hashes:

- `/tmp/benchmark_population_iterable_boundaries.py`
- `/tmp/survival-population-iterable-benchmark-final.json`
- `/tmp/survival-population-iterable-benchmark-final-reversed.json`
- `/tmp/survival-population-iterable-benchmark-comparison.json`
