# Survival response inputs

`Surv` and `Surv2` consume a one-shot event input once and retain all response
rows. Previously, a preliminary length check exhausted the event iterator, and
status coding read the empty iterator again. For example,
`Surv([1, 2, 3], iter([0, 1, 0]))` returned three times and zero event codes;
matrix extraction subsequently failed. Right, left, counting-process and
timeline responses now use the prepared event column for both validation and
coding. Interval status preparation already used one pass.

Declared factor levels travel with that prepared column. This includes
numeric factor arrays, whose numeric fast path previously inferred sorted
observed levels instead of using the declared order. For values
`[10, 2, NA, 30, 2, 10]` and levels `[30, 10, 2, 99]`, the censoring label is
`"30"`, states are `("10", "2", "99")`, and codes are
`(1, 2, None, 0, 2, 1)`, matching stock R. The unused `"99"` state remains
present. Missing declared categories are excluded as before, while a literal
`"NA"` label remains distinct from a missing value. Input values and categories
remain unmodified; response columns and state metadata own their normalized
values.

The shared factor encoder also checks declared categories before its numeric
array fast path. Consequently, `strata()` and model preprocessing preserve
the same category order, including for a numeric ndarray subclass with declared
levels. Explicit levels are snapshotted once, so a one-shot level iterable
provides both correct codes and complete labels. Ordinary numeric arrays still
return through the existing numeric encoder without allocating a level snapshot.
The existing strata stock-R fixture supplies 154 additional bounded controls:
numeric factor arrays and list/iterator equivalents; compound and nested strata
reuse; ordinary numeric-array controls; one-shot category and explicit-level
coding; and a three-group Gaussian AFT model with the declared order. The AFT
reference preserves the original baseline group and permutes coefficient,
design and covariance columns together when the other two groups change order.

Plain numeric and logical arrays retain their dtype during length validation,
which removes a discarded full-column `tolist()` allocation. Logical arrays,
masked logical values and nullable pandas logical columns continue to use
logical status coding. Numeric 1/2 inputs retain R's recoding, and normalization
warnings and length errors remain covered separately.

`scripts/generate_surv_iterable_reference.R` freezes independent values from
R 4.5.3 and survival 3.8-12. Its 33 constructor cases cover binary, logical,
1/2, invalid and missing statuses; declared factors with unused levels; origin
adjustments; empty responses; backwards counting intervals; interval and
interval2 responses; timeline repeated-event options; and invalid lengths.
Result serialization distinguishes missing values from positive and negative
infinity. Tests apply six equivalent Python input forms: lists, iterators,
generators, arrays, masked arrays and nullable pandas columns. Additional checks
cover declared numeric factor arrays in three constructor paths, missing
category metadata, input ownership, and a complete Kaplan–Meier fit using
one-shot time and event inputs. The dedicated file contains 206 tests.

Regenerate and verify with:

```sh
Rscript scripts/generate_surv_iterable_reference.R
PYTHONPATH=python .venv/bin/python -m pytest -q python/tests/test_surv_iterable_inputs.py
```

`scripts/benchmark_surv_inputs.py` measures construction, returned matrix
extraction and response metadata access together. It prepares inputs outside
timing and alternates baseline/current order within each process. The default
sizes are 100,000 and 500,000 rows, with right, counting-process and timeline
responses; contiguous and strided float64 events; logical arrays; and reusable
list controls. Reports include every timing sample, median and range, source
hashes, representative output rows and a result-payload hash. Hashing happens
outside timing; all baseline/current payloads must match.

Two quiet trials with opposite initial orders verified identical output hashes
for all 24 constructor cases, including equivalent hashes across input layouts.
Default cyclic-GC timings have overlapping, sometimes bimodal ranges; several
outliers move between versions when the initial order reverses. A separate
`--disable-gc` diagnostic keeps the same complete call while excluding cyclic
collector scheduling. Across two such trials, the 18-sample pooled medians for
float64 inputs were 9–15% lower at 100,000 rows and 10–21% lower at 500,000 rows.
Contiguous float64 results are shown below as median milliseconds followed by
the minimum–maximum sample range. List controls ranged from 1% slower to 3%
faster in that diagnostic; their ranges overlap.

| Response | Rows | Before, ms | After, ms |
| --- | ---: | ---: | ---: |
| Right | 100,000 | 7.55 (7.22–8.70) | 6.57 (6.27–6.98) |
| Counting | 100,000 | 13.14 (12.79–13.60) | 11.96 (11.67–12.78) |
| Timeline | 100,000 | 7.03 (6.91–7.24) | 6.01 (5.86–6.21) |
| Right | 500,000 | 45.79 (42.50–51.30) | 36.57 (35.22–40.93) |
| Counting | 500,000 | 74.24 (73.05–81.76) | 66.94 (65.15–72.62) |
| Timeline | 500,000 | 36.53 (35.33–43.71) | 28.78 (27.40–32.90) |

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_surv_inputs.py \
  --baseline-source /tmp/previous-surv.py --repeat 9 --warmup 3
```

Repeat with `--reverse-order` for the opposite initial order. Add `--disable-gc`
to reproduce the collector diagnostic; the report records the collector state.

`scripts/benchmark_factor_metadata.py` checks ordinary-array speed at the full
`strata()` boundary using a saved `_coerce.py` factor encoder. Numeric extraction,
coding, native grouping, returned factor construction and access to codes,
levels, labels and counts are timed. The report records full-result hashes,
source hashes and timing samples for integer, float, strided float and list
controls at the same default row counts.

Two quiet trials with opposite initial orders verified all eight grouping
payloads. Ordinary numeric-array pooled medians differed by at most 1.1%, with
overlapping timing ranges. At 500,000 rows, contiguous float64 calls measured
63.64 ms (60.15–67.16) before and 63.57 ms (60.18–66.94) after. The 100,000-row
list control measured 4.3% higher and the 500,000-row list control 0.2% lower,
with overlapping ranges. These controls support retaining the numeric fast
path; they do not establish a grouping speed improvement.

```sh
PYTHONPATH=python .venv/bin/python scripts/benchmark_factor_metadata.py \
  --baseline-source /tmp/previous-coerce.py --repeat 9 --warmup 3
```
